"""Exact checks of decomp/mlmm_hremd.py's all-state energies and derivatives.

Run: python check_mlmm_hremd.py

Float64: model on the CPU, OpenMM CUDA double. Ladder: path_states(2, 5, 4, 4)
(12 states over W1 | W2 | cut | W3). Two configurations: thermal200 frame 0
and the q 2.0 frame with the largest cap excess from route step 1
(experiments/2026-10-15_lambda0_dt/q2.0_frames.npz, max_excess), so the cap
and the cover switch are active in [2]-[4].

  [1] Evaluator.all_states U(q_k) vs the full Context energy after
      mlmm.set_path(q_k), every state
  [2] dU/dq per segment vs a central difference of the full Context energy
      in q (h = 1e-4; one-sided into the segment at its ends)
  [3] per-atom sums vs totals: W2 -(sum c_i + sum Coulomb_i + sum LJ_i) vs
      dU/dq(W2); W1 / W3 wall per-atom sums vs OpenMM's dU/ds_w
"""

import sys

import numpy as np
import openmm
import torch
from openmm import unit

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm  # noqa: E402
from decomp.mlmm_hremd import Evaluator, path_states, segments_of  # noqa: E402
from run_abfe import load_model  # noqa: E402

E3D = "/home/ruigengji/E3D"
LEG = f"{E3D}/em_solvent.npz"
BB = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"
CKPT = f"{E3D}/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
START = f"{E3D}/experiments/2026-10-12_thermal_frames/thermal200_solvent.npz"
torch.set_num_threads(8)

pos0 = np.asarray(np.load(START, allow_pickle=True)["positions"][0], dtype=np.float64)
frames = {"thermal200 frame 0": pos0,
          "q2.0 max cap excess": np.asarray(np.load(f"{E3D}/experiments/2026-10-15_lambda0_dt/q2.0_frames.npz")
                                            ["max_excess"], dtype=np.float64)}
model = load_model(CKPT, BB, device="cpu").to(torch.float64)
qs = path_states(2, 5, 4, 4)
plat = openmm.Platform.getPlatformByName("CUDA")
dbl = {"Precision": "double"}
system, mlf, info = mlmm.build_mlmm_system(LEG, model, n_inner=20, es=True, device="cpu", positions=pos0)
ev = Evaluator(system, mlf, info, qs, plat, dbl)
system2, mlf2, info2 = mlmm.build_mlmm_system(LEG, model, n_inner=20, es=True, device="cpu", positions=pos0)
ctx = openmm.Context(system2, openmm.VerletIntegrator(0.001), plat, dbl)
box = info["box"]
ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])
lig = info["lig"]
print(f"states q = {qs.tolist()}", flush=True)


def full(P, q):
    mlmm.set_path(ctx, mlf2, q, lig)
    ctx.setPositions(P * unit.angstrom)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


for name, P in frames.items():
    print(f"--- {name}", flush=True)
    U, parts = ev.all_states(P)
    ref = np.array([full(P, q) for q in qs])
    d = np.abs(U - ref)
    print(f"[1] all_states vs Context: max |dU| {d.max():.2e} kJ/mol over {len(qs)} states "
          f"(|U| ~ {np.abs(ref).max():.3e}); S_eff {parts['S']:.2f}, U0 - E_bb {parts['U0'] - parts['E_bb']:.2f} kJ/mol",
          flush=True)
    h = 1e-4
    worst = {}
    sums = []
    for q in qs:
        der, atoms = ev.derivatives(P, q, parts)
        for seg in segments_of(q):
            a = dict(W1=0.0, W2=1.0, cut=2.0, W3=3.0)[seg]
            b = a + 1.0
            if q - h < a - 1e-12:
                fd = (full(P, q + h) - full(P, q)) / h
            elif q + h > b + 1e-12:
                fd = (full(P, q) - full(P, q - h)) / h
            else:
                fd = (full(P, q + h) - full(P, q - h)) / (2 * h)
            e = abs(der[seg] - fd)
            worst[seg] = max(worst.get(seg, (0, 0, 0)), (e, q, der[seg]))
            if seg == "W2":
                sums.append(("W2", q, der[seg], atoms["W2_ml"].sum() + atoms["W2_coul"].sum() + atoms["W2_lj"].sum()))
            elif seg in atoms:
                sums.append((seg, q, der[seg], atoms[seg].sum()))
    for seg, (e, q, v) in worst.items():
        print(f"[2] {seg:3s}: max |dU/dq - FD| {e:.2e} kJ/mol (at q {q:g}, dU/dq {v:+.3f})", flush=True)
    ds = max(abs(t - s) for _, _, t, s in sums)
    print(f"[3] per-atom sums vs totals: max |d| {ds:.2e} kJ/mol over {len(sums)} (state, segment) pairs; e.g. "
          + "; ".join(f"{g} q {q:g}: {t:+.3f} vs {s:+.3f}" for g, q, t, s in sums[::max(1, len(sums) // 4)]), flush=True)
