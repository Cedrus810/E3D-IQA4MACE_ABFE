"""Exact checks of the D19 path (option 3, user 2026-10-18; v3 S6.3.4a).

Run: python check_d19.py     (CPU: model float64, OpenMM Reference)

ML carries only E_bb(L) + truth - sum (1 - lambda) c; the inner waters' own
energy and the I-I pairs are MM; path q in [0, 4] = W1 | W2 | cut | W3.
Acetamide leg, its own start frame (inner waters picked there), operating head.
  [1] Evaluator.all_states vs the full Context energy at q over every segment
  [2] dU/dq vs a central difference in q, one interior point per segment
  [3] per-atom sums vs their totals (W1, W2, W3)
  [4] ML force vs a central FD of the ML group energy (ligand and inner-water atoms),
      at q 1.5 (W2) and 2.5 (cut): the -E_bb(I) gradient acts on the inner waters
  [5] relabelling: swap an inner and an outer water. With FIRES left out, U must not
      change at lambda = 0 (q 3, 3.5, 4) -- the shell is plain water. Controls: D19 at
      q 1 (ligand coupled), and the old path at q 3 (ML shell), must change.
"""

import sys

import numpy as np
import openmm
import torch
from openmm import unit

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm  # noqa: E402
from decomp.mlmm_hremd import Evaluator, segments_of  # noqa: E402
from run_abfe import load_model  # noqa: E402

E3D = "/home/ruigengji/E3D"
LEG = f"{E3D}/experiments/2026-10-15_acetamide_leg/em_acetamide.npz"
KJ = unit.kilojoule_per_mole
torch.set_num_threads(16)
plat = openmm.Platform.getPlatformByName("Reference")
P = np.asarray(np.load(LEG, allow_pickle=True)["positions"], dtype=np.float64)
model = load_model(f"{E3D}/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt",
                   "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", device="cpu").to(torch.float64)
qs = np.array([0.0, 0.5, 1.0, 1.5, 1.95, 2.0, 2.5, 3.0, 3.3, 3.7, 4.0])
sysA, mlfA, info = mlmm.build_mlmm_system(LEG, model, device="cpu", positions=P, shell_mm=True)
ev = Evaluator(sysA, mlfA, info, qs, plat)
sysB, mlfB, _ = mlmm.build_mlmm_system(LEG, model, device="cpu", positions=P, shell_mm=True)
box = info["box"]
vec = [openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)]
ctx = openmm.Context(sysB, openmm.VerletIntegrator(0.001), plat)
ctx.setPeriodicBoxVectors(*vec)
lig = info["lig"]
assert "t_sw" not in ctx.getParameters(), "D19 system should carry no t_sw"


def full(q, X=P, groups=None, c=ctx, f=mlfB, shell_mm=True):
    mlmm.set_path(c, f, q, lig, shell_mm=shell_mm)
    c.setPositions(X * unit.angstrom)
    kw = {} if groups is None else {"groups": groups}
    return c.getState(getEnergy=True, **kw).getPotentialEnergy().value_in_unit(KJ)


U, parts = ev.all_states(P)
ref = np.array([full(q) for q in qs])
print(f"[1] all_states vs Context: max |dU| {np.abs(U - ref).max():.2e} kJ/mol over q = {qs.tolist()} "
      f"(|U| ~ {np.abs(ref).max():.3g}); E_bb(L) {parts['E_L']:.1f}, "
      f"U0 - E_bb(L) {parts['U0'] - parts['E_L']:.2e} kJ/mol", flush=True)

for q, h in ((0.5, 1e-3), (1.5, 1e-2), (2.5, 1e-3), (3.5, 1e-3)):
    der, _ = ev.derivatives(P, q, parts, per_atom=False)
    fd = (full(q + h) - full(q - h)) / (2 * h)
    seg = segments_of(q, shell_mm=True)[0]
    print(f"[2] q {q:g} ({seg}, h {h:g}): dU/dq {der[seg]:+.4f} vs FD {fd:+.4f} kJ/mol "
          f"(|d| {abs(der[seg] - fd):.1e})", flush=True)

for q in (0.5, 1.5, 3.5):
    Uq, pq = ev.all_states(P)
    der, at = ev.derivatives(P, q, pq, per_atom=True)
    seg = segments_of(q, shell_mm=True)[0]
    s = sum(v.sum() for v in at.values())
    print(f"[3] q {q:g} ({seg}): per-atom sum {s:+.6f} vs total {der[seg]:+.6f} kJ/mol "
          f"(|d| {abs(s - der[seg]):.1e}; keys {sorted(at)})", flush=True)

inner_atoms = [int(x) for x in info["inner_atoms"]]
atoms = [int(lig[0]), int(lig[3]), inner_atoms[0], inner_atoms[1], inner_atoms[30], inner_atoms[31]]
for q in (1.5, 2.5):
    mlmm.set_path(ctx, mlfB, q, lig, shell_mm=True)
    ctx.setPositions(P * unit.angstrom)
    F = ctx.getState(getForces=True, groups={mlmm.G_ML}).getForces(asNumpy=True).value_in_unit(KJ / unit.nanometer) / 10
    err = []
    for x in atoms:
        for c in range(3):
            Xp, Xm = P.copy(), P.copy()
            Xp[x, c] += 1e-4; Xm[x, c] -= 1e-4
            fd = -(full(q, Xp, {mlmm.G_ML}) - full(q, Xm, {mlmm.G_ML})) / 2e-4
            err.append(abs(fd - F[x, c]))
    print(f"[4] ML force at q {q:g} vs FD (h 1e-4 A, {len(err)} components incl. inner-water atoms): "
          f"max |d| {max(err):.2e} kJ/mol/A (max |F| {np.abs(F[atoms]).max():.1f})", flush=True)

# [5] relabelling: swap one inner and one outer water (whole-molecule coordinates), FIRES left out
W = info["water"]
wi = W[info["inner"][0]]
d_out = [np.linalg.norm((P[W[k], 0] - P[wi[0]]) - box * np.round((P[W[k], 0] - P[wi[0]]) / box))
         for k in np.setdiff1d(np.arange(len(W)), info["inner"])]
wo = W[np.setdiff1d(np.arange(len(W)), info["inner"])[int(np.argsort(d_out)[0])]]   # the outer water nearest it
X = P.copy()
X[wi], X[wo] = P[wo], P[wi]


def no_fires(q, Y, c=ctx, f=mlfB, shell_mm=True):
    e = full(q, Y, c=c, f=f, shell_mm=shell_mm)
    return e - f.last["U_fires"]


for q in (3.0, 3.5, 4.0, 1.0):
    dU = no_fires(q, X) - no_fires(q, P)
    print(f"[5] D19 q {q:g}: U(swapped) - U, FIRES excluded: {dU:+.3e} kJ/mol"
          + ("   (control: ligand coupled, expected nonzero)" if q == 1.0 else ""), flush=True)
sysO, mlfO, _ = mlmm.build_mlmm_system(LEG, model, device="cpu", positions=P)
ctxO = openmm.Context(sysO, openmm.VerletIntegrator(0.001), plat)
ctxO.setPeriodicBoxVectors(*vec)
dU = no_fires(3.0, X, ctxO, mlfO, False) - no_fires(3.0, P, ctxO, mlfO, False)
print(f"[5] old path q 3 (ML shell, t = 0): U(swapped) - U, FIRES excluded: {dU:+.3e} kJ/mol"
      "   (control: expected nonzero)", flush=True)
