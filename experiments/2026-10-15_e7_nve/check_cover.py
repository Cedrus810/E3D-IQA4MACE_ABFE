"""Checks of the S3 cover switch (lambda_mask.cover_switch; v3 S13.41), float64, CPU.

Run: python check_cover.py

ML region (L + 20 inner waters) from build_mlmm_system on thermal200 frame 0.
  [1] start frame: ligand-environment edges inside the switch (r < r_cov),
      sum c with and without the switch
  [2] the crushed state before the first q = 2.0 NaN (diag_q2.0_r0des_last.npz):
      truth, sum c and r = truth - sum c without / with the switch
  [3] forces vs central FD at p = 0.5 on that state (edges inside the switch)
  [4] dU/dp vs a central FD in p at p = 0.5 (= -sum c, switched)
  [5] endpoints p = 0 and p = 2 unchanged by the switch
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm  # noqa: E402
from decomp.lambda_mask import COVER_R, _fragment  # noqa: E402
from decomp.mace_adapter import MACEDecomposition  # noqa: E402
from decomp.openmm_bridge import AnchoredForce  # noqa: E402

E3D = "/home/ruigengji/E3D"
LEG = f"{E3D}/em_solvent.npz"
BB = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"
CKPT = f"{E3D}/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
START = f"{E3D}/experiments/2026-10-12_thermal_frames/thermal200_solvent.npz"
EV = 96.48533212
torch.set_num_threads(8)

bb = torch.load(BB, map_location="cpu", weights_only=False).to(torch.float64).eval()
model = MACEDecomposition(bb, hidden=256, n_pair_channels=4)
model.head.load_state_dict(torch.load(CKPT, map_location="cpu"))
model = model.to(torch.float64).eval()
for p in model.parameters():
    p.requires_grad_(False)
d = np.load(LEG, allow_pickle=True)
num = np.asarray(d["numbers"])
pos0 = np.asarray(np.load(START, allow_pickle=True)["positions"][0], dtype=np.float64)
_, mlf, info = mlmm.build_mlmm_system(LEG, model, n_inner=20, es=True, device="cpu", positions=pos0)
box, nl = info["box"], len(info["lig"])
Z = num[mlf.ml_idx]
frag = np.r_[np.zeros(nl, int), np.ones(len(Z) - nl, int)]
on = AnchoredForce(model, Z, frag, es=True, device="cpu", cover=COVER_R)
off = AnchoredForce(model, Z, frag, es=True, device="cpu")


def sum_c(f, X, p=0.5):
    f.set_lambda(p)
    f.energy_and_forces(X, need_dudl=True)
    return -f.last_dudl_total


def truth(X):
    data = on._graph(X)
    E = lambda dd: float(bb(dd, training=False, compute_force=False)["energy"].sum())  # noqa: E731
    return E(data) - E(_fragment(data, on.frag_id, 0, 0)) - E(_fragment(data, on.frag_id, 1, 0))


def inside(X):
    D = np.linalg.norm(X[:nl, None] - X[None, nl:], axis=-1)
    rc = np.array([[COVER_R[(int(a), int(b))] for b in Z[nl:]] for a in Z[:nl]])
    return int((D < rc).sum()), float((D / rc).min())


X0 = mlf.ml_positions(pos0, box)
n_in, m = inside(X0)
print(f"[1] start frame: {n_in} L-E pairs below r_cov (min r/r_cov {m:.2f}); sum c without / with switch "
      f"{sum_c(off, X0) * EV:.2f} / {sum_c(on, X0) * EV:.2f} kJ/mol", flush=True)

Pc = np.load(f"{E3D}/experiments/2026-10-15_e7_nve/diag_q2.0_r0des_last.npz")["positions"]
Xc = mlf.ml_positions(np.asarray(Pc, dtype=np.float64), box)
n_in, m = inside(Xc)
tr = truth(Xc)
c0, c1 = sum_c(off, Xc), sum_c(on, Xc)
print(f"[2] crushed state: {n_in} pairs below r_cov (min r/r_cov {m:.2f}); truth {tr * EV:.1f}; "
      f"sum c {c0 * EV:.1f} -> {c1 * EV:.1f}; r = truth - sum c {(tr - c0) * EV:.1f} -> {(tr - c1) * EV:.1f} kJ/mol",
      flush=True)

on.set_lambda(0.5)
U, F = on.energy_and_forces(Xc)
D = np.linalg.norm(Xc[:nl, None] - Xc[None, nl:], axis=-1)
rc = np.array([[COVER_R[(int(a), int(b))] for b in Z[nl:]] for a in Z[:nl]])
zone = np.argwhere((D < rc) & (D > rc - 0.2))
atoms = sorted({int(zone[k, 0]) for k in range(min(3, len(zone)))} |
               {nl + int(zone[k, 1]) for k in range(min(3, len(zone)))})
h, err = 1e-4, []
for a in atoms:
    for c in range(3):
        xp, xm = Xc.copy(), Xc.copy()
        xp[a, c] += h; xm[a, c] -= h
        fd = -(on.energy_and_forces(xp)[0] - on.energy_and_forces(xm)[0]) / (2 * h)
        err.append(abs(fd - F[a, c]))
print(f"[3] forces vs central FD (h {h} A, {len(err)} components on atoms with a pair in the switch zone, "
      f"{len(zone)} such pairs): max |d| {max(err):.2e} eV/A (max |F| {np.abs(F[atoms]).max():.2f})", flush=True)

hp = 1e-4
on.set_lambda(0.5 + hp); Up = on.energy_and_forces(Xc)[0]
on.set_lambda(0.5 - hp); Um = on.energy_and_forces(Xc)[0]
on.set_lambda(0.5); on.energy_and_forces(Xc, need_dudl=True)
print(f"[4] dU/dp at p 0.5: {on.last_dudl_total:.8f} vs FD {(Up - Um) / (2 * hp):.8f} eV", flush=True)

for p_ in (0.0, 2.0):
    on.set_lambda(p_); off.set_lambda(p_)
    a_, b_ = on.energy_and_forces(Xc)[0], off.energy_and_forces(Xc)[0]
    print(f"[5] p = {p_}: with / without switch {a_:.8f} / {b_:.8f} eV, |d| {abs(a_ - b_):.1e}", flush=True)
