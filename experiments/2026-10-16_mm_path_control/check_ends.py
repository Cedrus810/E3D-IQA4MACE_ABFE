"""All-MM path control (v3 S7.2): do the ends match the reference calculation?

Run: python check_ends.py

Acetamide leg (S13.44). OpenMM CUDA double. At the leg's positions (which
also pick the 20 inner waters, so FIRES is 0 there):
  [1] q = 0 (lambda 1, no wall) of build_mlmm_system(mm_only=True) vs the
      reference System (system_solvent.xml)
  [2] q = 4 (decoupled, no wall) vs abfe_mm.py's decoupled state 30
      (lambda_elec 0, lambda_vdw 0), the end of the +34.04 kJ/mol reference
  [3] Evaluator (mlmm_hremd) all_states vs the Context energy at 14 states
  [4] dU/dq vs a finite difference in q (h = 1e-2 on W2, where the PME noise
      floor makes small h useless, S13.46; 1e-4 elsewhere)
  [5] per-atom sums vs totals
"""

import importlib.util
import sys

import numpy as np
import openmm
from openmm import unit

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm  # noqa: E402
from decomp.mlmm_hremd import Evaluator, segments_of  # noqa: E402

LEGDIR = "/home/ruigengji/E3D/experiments/2026-10-15_acetamide_leg"
LEG = f"{LEGDIR}/em_acetamide.npz"
spec = importlib.util.spec_from_file_location("abfe_mm", f"{LEGDIR}/abfe_mm.py")
abfe_mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abfe_mm)
KJ = unit.kilojoule_per_mole
plat = openmm.Platform.getPlatformByName("CUDA")
dbl = {"Precision": "double"}

d = np.load(LEG, allow_pickle=True)
P = np.asarray(d["positions"], dtype=np.float64)
box = np.asarray(d["box"], dtype=np.float64)
vecs = [openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)]


def energy(system, setup=None, groups=None):
    c = openmm.Context(system, openmm.VerletIntegrator(0.001), plat, dbl)
    c.setPeriodicBoxVectors(*vecs)
    if setup:
        setup(c)
    c.setPositions(P * unit.angstrom)
    kw = {} if groups is None else {"groups": groups}
    return c.getState(getEnergy=True, **kw).getPotentialEnergy().value_in_unit(KJ)


system, mlf, info = mlmm.build_mlmm_system(LEG, None, n_inner=20, device="cuda", positions=P, mm_only=True)
lig = info["lig"]
ref = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
E_ref = energy(ref)
E_A = energy(system, lambda c: mlmm.set_path(c, mlf, 0.0, lig))
F_A = energy(system, lambda c: mlmm.set_path(c, mlf, 0.0, lig), groups={mlmm.G_ML})
print(f"[1] q 0 vs reference System: {E_A:.4f} vs {E_ref:.4f} kJ/mol, diff {E_A - E_ref:+.4f} (FIRES {F_A:.4f})",
      flush=True)
_, _, sysB, _ = abfe_mm.build()
E_B_ref = energy(sysB, lambda c: abfe_mm.set_state(c, len(abfe_mm.STATES) - 1))
E_B = energy(system, lambda c: mlmm.set_path(c, mlf, 4.0, lig))
print(f"[2] q 4 vs abfe_mm decoupled state: {E_B:.4f} vs {E_B_ref:.4f} kJ/mol, diff {E_B - E_B_ref:+.4f}", flush=True)
E_A_ref = energy(sysB, lambda c: abfe_mm.set_state(c, 0))
print(f"    (abfe_mm coupled state vs reference: diff {E_A_ref - E_ref:+.4f}; so q 0 -> q 4 at this frame "
      f"{E_B - E_A:+.4f} vs abfe_mm {E_B_ref - E_A_ref:+.4f} kJ/mol)", flush=True)

qs = np.array([0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 1.9, 2.0, 2.5, 3.0, 3.3, 3.6, 3.9, 4.0])
ev = Evaluator(system, mlf, info, qs, plat, dbl)
U, parts = ev.all_states(P)
ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), plat, dbl)
ctx.setPeriodicBoxVectors(*vecs)


def full(q):
    mlmm.set_path(ctx, mlf, q, lig)
    ctx.setPositions(P * unit.angstrom)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)


refU = np.array([full(q) for q in qs])
print(f"[3] all_states vs Context: max |dU| {np.abs(U - refU).max():.2e} kJ/mol over {len(qs)} states", flush=True)
worst, sums = {}, []
for q in qs:
    der, atoms = ev.derivatives(P, q, parts)
    for seg in segments_of(q):
        a = dict(W1=0.0, W2=1.0, cut=2.0, W3=3.0)[seg]
        h = 1e-2 if seg == "W2" else 1e-4
        if q - h < a - 1e-12:
            fd = (full(q + h) - full(q)) / h
        elif q + h > a + 1 + 1e-12:
            fd = (full(q) - full(q - h)) / h
        else:
            fd = (full(q + h) - full(q - h)) / (2 * h)
        worst[seg] = max(worst.get(seg, (0, 0, 0)), (abs(der[seg] - fd), q, der[seg]))
        if seg == "W2":
            sums.append(der[seg] - (atoms["W2_ml"].sum() + atoms["W2_coul"].sum() + atoms["W2_lj"].sum()))
        elif seg in atoms:
            sums.append(der[seg] - atoms[seg].sum())
for seg, (e, q, v) in worst.items():
    print(f"[4] {seg:3s}: max |dU/dq - FD| {e:.2e} kJ/mol (at q {q:g}, dU/dq {v:+.3f})", flush=True)
print(f"[5] per-atom sums vs totals: max |d| {np.abs(sums).max():.2e} kJ/mol over {len(sums)} pairs", flush=True)
