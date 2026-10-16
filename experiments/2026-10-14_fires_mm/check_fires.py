"""E4 checks of decomp/fires.py (RESEARCH_PLAN_v3.md S6.3.5), float64, CPU.

Run: python check_fires.py

On em_solvent.npz (the Atenolol solvent leg, 1,345 TIP3P waters, 35 A box),
n = 20 inner waters:
  [1] U = 0 on the frame the inner set was selected on
  [2] an outer water pushed inside R_inner: U > 0; forces vs central FD
  [3] C1 across the boundary: one outer water moved radially through R_inner
  [4] outermost inner water changing during a violation: force continuity of
      the pairwise restraint vs the paper's eq 5 (max)
  [5] PBC: U and forces unchanged under a box translation and rewrapping
  [6] softmin offset against the true closest-atom distance (nearest 60 waters)
  [7] time per energy + force call
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp.fires import Fires, heavy_selection, water_indices, softmin  # noqa: E402

E3D = "/home/ruigengji/E3D"
d = np.load(f"{E3D}/em_solvent.npz", allow_pickle=True)
pos0, box = np.asarray(d["positions"], dtype=np.float64), np.asarray(d["box"], dtype=np.float64)
lig = np.where(d["resids"] == d["resids"][0])[0]
W = water_indices(d["resnames"], d["resids"])
LH, WO = heavy_selection(lig, d["numbers"], W)
F = Fires(LH, WO, n_inner=20)
inner = F.select_inner(pos0, box)
print(f"{len(lig)} ligand atoms ({len(LH)} heavy, used for ordering), {len(W)} waters, n_inner {len(inner)}, k {F.k:.0f} kJ/mol/A^2, "
      f"beta {F.beta} /A")


def dist(pos):
    return F.distances(F._t(pos), F._t(box)).detach().numpy()


def move_water(pos, w, delta):
    p = pos.copy()
    p[W[w]] += delta
    return p


def radial(pos, w):
    """Unit vector from the ligand atom closest to water w's oxygen (min image)."""
    o = pos[W[w][0]]
    dv = o - pos[LH]
    dv -= box * np.round(dv / box)
    i = np.argmin((dv ** 2).sum(-1))
    return dv[i] / np.linalg.norm(dv[i])


# [1]
U, Fr = F.energy_and_forces(pos0, box)
print(f"[1] ordered frame: U = {U:.3e} kJ/mol, max|F| = {np.abs(Fr).max():.3e}, "
      f"R_inner {F.last['r_inner']:.3f} A, nearest outer {F.last['d_outer_min']:.3f} A")

# [2] nearest outer water pushed 0.3 A inside R_inner
dd = dist(pos0)
outer = np.setdiff1d(np.arange(len(W)), inner)
v = outer[np.argmin(dd[outer])]
r_in = dd[inner].max()
u = radial(pos0, v)
pos1 = move_water(pos0, v, -(dd[v] - r_in + 0.3) * u)
U1, F1 = F.energy_and_forces(pos1, box)
print(f"[2] outer water {v} moved to d = {dist(pos1)[v]:.3f} (R_inner {r_in:.3f}): U = {U1:.3f} kJ/mol, "
      f"{F.last['n_viol']} violating pairs")
h, errs = 1e-4, []
atoms = list(W[v]) + list(W[inner[np.argmax(dd[inner])]]) + [int(lig[0])]
for a in atoms:
    for c in range(3):
        pp, pm = pos1.copy(), pos1.copy()
        pp[a, c] += h; pm[a, c] -= h
        fd = -(F.energy_and_forces(pp, box)[0] - F.energy_and_forces(pm, box)[0]) / (2 * h)
        errs.append(abs(fd - F1[a, c]))
print(f"    forces vs central FD (h {h} A, {len(errs)} components): max |d| {max(errs):.2e} "
      f"kJ/mol/A (max |F| on them {max(abs(F1[a]).max() for a in atoms):.1f})")

# [3] C1 across the boundary: radial scan of the same outer water
def scan(npt):
    ts = np.linspace(-0.2, 0.2, npt)                      # d_v - R_inner, A
    Us, Gs = [], []
    for t in ts:
        p = move_water(pos0, v, -(dd[v] - r_in - t) * u)
        Ut, Ft = F.energy_and_forces(p, box)
        Us.append(Ut); Gs.append(-(Ft[W[v]].sum(0) @ u))  # dU/d(radial shift)
    return np.abs(np.diff(Us)).max(), np.abs(np.diff(Gs)).max()


(u1, g1), (u2, g2) = scan(201), scan(401)
print(f"[3] radial scan through R_inner over 0.4 A, max step between neighbours at 201 / 401 points: "
      f"|dU| {u1:.3e} / {u2:.3e} kJ/mol, |d(dU/ds)| {g1:.3f} / {g2:.3f} kJ/mol/A "
      f"(continuous: ratio ~2; a jump would not shrink) -> ratios {u1 / u2:.2f}, {g1 / g2:.2f}")

# [4] outermost inner water changing while an outer water violates
#     two inner waters a, b; b swept radially past a; outer water v kept 0.2 A inside both
order_in = inner[np.argsort(dd[inner])]
a_, b_ = order_in[-1], order_in[-2]
ub = radial(pos0, b_)
p_base = move_water(pos0, v, -(dd[v] - (dd[b_] - 0.2)) * u)


def eq5(pos):
    t = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    dd_ = F.distances(t, F._t(box))
    R = dd_[F.inner_mask].max()
    do = dd_[~F.inner_mask]
    E = 0.5 * F.k * (torch.relu(R - do) ** 2).sum()
    (g,) = torch.autograd.grad(E, t)
    return float(E.detach()), -g.numpy()


def sweep(npt):
    ss = np.linspace(-0.1, 0.1 + (dd[a_] - dd[b_]), npt)
    fp_b, fe_b = [], []
    for s_ in ss:
        p = move_water(p_base, b_, s_ * ub)
        _, Fp = F.energy_and_forces(p, box)
        _, Fe = eq5(p)
        fp_b.append(Fp[W[b_]].sum(0) @ ub); fe_b.append(Fe[W[b_]].sum(0) @ ub)
    return np.abs(np.diff(fp_b)).max(), np.abs(np.diff(fe_b)).max()


(p1_, e1_), (p2_, e2_) = sweep(151), sweep(301)
print(f"[4] inner water {b_} swept radially past {a_} (outer {v} inside both), radial force on {b_}, "
      f"max step between neighbours at 151 / 301 points: pairwise {p1_:.2f} / {p2_:.2f}, "
      f"eq 5 (max) {e1_:.2f} / {e2_:.2f} kJ/mol/A -> ratios {p1_ / p2_:.2f}, {e1_ / e2_:.2f}")

# [5] PBC
U5, F5 = F.energy_and_forces(pos1 + box * np.array([1, -2, 3]), box)
wrapped = pos1.copy()
for w in range(len(W)):                                   # rewrap each water by its O
    s5 = np.floor(wrapped[W[w][0]] / box) * box
    wrapped[W[w]] -= s5
U6, F6 = F.energy_and_forces(wrapped, box)
print(f"[5] box translation: |dU| {abs(U5 - U1):.2e}, max|dF| {np.abs(F5 - F1).max():.2e}; "
      f"rewrap: |dU| {abs(U6 - U1):.2e}, max|dF| {np.abs(F6 - F1).max():.2e}")

# [6] softmin offset
t0 = F._t(pos0)
dv = t0[F.wat][:, :, None, :] - t0[F.lig][None, None, :, :]
dv = dv - F._t(box) * torch.round(dv / F._t(box))
rmin = torch.sqrt((dv ** 2).sum(-1)).reshape(len(W), -1).min(-1).values.numpy()
near = np.argsort(rmin)[:60]
off = rmin[near] - dd[near]
same = np.array_equal(np.sort(np.argsort(rmin)[:20]), np.sort(inner))
print(f"[6] true min - softmin, nearest 60 waters: {off.min():.3f} .. {off.max():.3f} A "
      f"(bound ln({F.lig.numel()})/beta = {np.log(F.lig.numel()) / F.beta:.3f}); same 20 inner by true min: {same}")

# [7] timing
for _ in range(3):
    F.energy_and_forces(pos1, box)
t = time.perf_counter()
for _ in range(20):
    F.energy_and_forces(pos1, box)
print(f"[7] energy + forces, CPU float64, {torch.get_num_threads()} threads: "
      f"{(time.perf_counter() - t) / 20 * 1e3:.1f} ms/call")
