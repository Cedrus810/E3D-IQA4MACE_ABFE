"""Exact checks of the SW segment (inner waters ML -> MM, D18) and the all-MM W3 (v3 S6.3.4, 2026-10-17).

Run: python check_sw.py     (CPU only: model float64, OpenMM Reference; the GPUs are busy)

Atenolol leg, thermal200 frame 0 (inner waters picked there).
  [1] Evaluator.all_states vs the full Context energy at q in W2 | cut | SW | W3
  [2] dU/dq (SW, W3) vs a central difference in q (h = 1e-3)
  [3] ML force in SW (q 3.5) vs a central FD of the ML group energy, 6 atoms
  [4] q = 5 is all-MM water: moving inner waters rigidly changes the MM groups' energy
      exactly as in the all-MM control system (mm_only) at q = 5
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
LEG = f"{E3D}/em_solvent.npz"
KJ = unit.kilojoule_per_mole
torch.set_num_threads(8)
plat = openmm.Platform.getPlatformByName("Reference")
P = np.asarray(np.load(f"{E3D}/experiments/2026-10-12_thermal_frames/thermal200_solvent.npz",
                       allow_pickle=True)["positions"][0], dtype=np.float64)
model = load_model(f"{E3D}/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt",
                   "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", device="cpu").to(torch.float64)
qs = np.array([1.5, 2.0, 2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.3, 4.7, 5.0])
sysA, mlfA, info = mlmm.build_mlmm_system(LEG, model, device="cpu", positions=P)
ev = Evaluator(sysA, mlfA, info, qs, plat)
sysB, mlfB, _ = mlmm.build_mlmm_system(LEG, model, device="cpu", positions=P)
box = info["box"]
vec = [openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)]
ctx = openmm.Context(sysB, openmm.VerletIntegrator(0.001), plat)
ctx.setPeriodicBoxVectors(*vec)
lig = info["lig"]


def full(q, X=P, groups=None):
    mlmm.set_path(ctx, mlfB, q, lig)
    ctx.setPositions(X * unit.angstrom)
    kw = {} if groups is None else {"groups": groups}
    return ctx.getState(getEnergy=True, **kw).getPotentialEnergy().value_in_unit(KJ)


U, parts = ev.all_states(P)
ref = np.array([full(q) for q in qs])
print(f"[1] all_states vs Context: max |dU| {np.abs(U - ref).max():.2e} kJ/mol over q = {qs.tolist()}; "
      f"E_bb(I) {parts['E_I']:.1f}, NB(t=1) - NB(t=0) {parts['nb_t1'] - parts['nb'][0]:+.2f} kJ/mol", flush=True)

h = 1e-3
for q in (3.25, 3.75, 4.3, 4.7):
    der, _ = ev.derivatives(P, q, parts, per_atom=False)
    fd = (full(q + h) - full(q - h)) / (2 * h)
    seg = segments_of(q)[0]
    print(f"[2] q {q:g} ({seg}): dU/dq {der[seg]:+.4f} vs FD {fd:+.4f} kJ/mol (|d| {abs(der[seg] - fd):.1e})", flush=True)

mlmm.set_path(ctx, mlfB, 3.5, lig)
ctx.setPositions(P * unit.angstrom)
F = ctx.getState(getForces=True, groups={mlmm.G_ML}).getForces(asNumpy=True).value_in_unit(KJ / unit.nanometer) / 10
atoms = [int(lig[0]), int(lig[5]), *[int(x) for x in info["inner_atoms"][:4]]]
err = []
for x in atoms:
    for c in range(3):
        Xp, Xm = P.copy(), P.copy()
        Xp[x, c] += 1e-4; Xm[x, c] -= 1e-4
        fd = -(full(3.5, Xp, {mlmm.G_ML}) - full(3.5, Xm, {mlmm.G_ML})) / 2e-4
        err.append(abs(fd - F[x, c]))
print(f"[3] ML force at q 3.5 vs FD (h 1e-4 A, {len(err)} components): max |d| {max(err):.2e} kJ/mol/A "
      f"(max |F| {np.abs(F[atoms]).max():.1f})", flush=True)

sysC, mlfC, infoC = mlmm.build_mlmm_system(LEG, None, device="cpu", positions=P, mm_only=True, use_fires=False)
ctxC = openmm.Context(sysC, openmm.VerletIntegrator(0.001), plat)
ctxC.setPeriodicBoxVectors(*vec)
mlmm.set_path(ctxC, mlfC, 5.0, lig)
MM = {0, 1, 2, 3}
rng = np.random.default_rng(0)
W = info["water"][info["inner"]]
worst = 0.0
for trial in range(3):
    X = P.copy()
    for w in W[rng.choice(len(W), 5, replace=False)]:
        X[w] += rng.normal(0, 0.15, 3)                     # rigid shifts of 5 inner waters
    dA = full(5.0, X, MM) - full(5.0, P, MM)
    ctxC.setPositions(X * unit.angstrom); eX = ctxC.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    ctxC.setPositions(P * unit.angstrom); eP = ctxC.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    worst = max(worst, abs(dA - (eX - eP)))
    print(f"[4] trial {trial}: dE(ML/MM q5, MM groups) {dA:+.4f} vs dE(all-MM q5) {eX - eP:+.4f} kJ/mol", flush=True)
print(f"[4] max |difference| {worst:.2e} kJ/mol", flush=True)
