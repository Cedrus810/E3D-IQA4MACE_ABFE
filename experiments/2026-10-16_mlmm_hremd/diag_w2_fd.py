"""Why does the one-sided FD of W2 at q = 1 miss by 0.8 kJ/mol? Energies near q = 1, by group."""
import sys
import numpy as np, openmm, torch
from openmm import unit
sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm
from decomp.mlmm_hremd import Evaluator
from run_abfe import load_model
E3D = "/home/ruigengji/E3D"
torch.set_num_threads(8)
pos0 = np.asarray(np.load(f"{E3D}/experiments/2026-10-12_thermal_frames/thermal200_solvent.npz", allow_pickle=True)["positions"][0], dtype=np.float64)
model = load_model(f"{E3D}/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt", "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", device="cpu").to(torch.float64)
plat = openmm.Platform.getPlatformByName("CUDA"); dbl = {"Precision": "double"}
qs = np.array([1.0, 1.0001, 1.001, 1.01, 1.1, 1.5])
system, mlf, info = mlmm.build_mlmm_system(f"{E3D}/em_solvent.npz", model, n_inner=20, es=True, device="cpu", positions=pos0)
ev = Evaluator(system, mlf, info, qs, plat, dbl)
U, parts = ev.all_states(pos0)
ctx = ev.ctx
ctx.setPositions(pos0 * unit.angstrom)
lig = info["lig"]
rows = []
for q in qs:
    mlmm.set_path(ctx, mlf, q, lig)
    g = [ctx.getState(getEnergy=True, groups={k}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) for k in range(5)]
    rows.append(g)
rows = np.array(rows)
d, _ = ev.derivatives(pos0, 1.0, parts, per_atom=False)
print("analytic dU/dq(W2, q=1)", d["W2"], " parts: S", parts["S"], "nb (a,b,c)", parts["nb"], "lj1", parts["lj1"])
for k in range(1, len(qs)):
    sl = (rows[k] - rows[0]) / (qs[k] - qs[0])
    print(f"q 1 -> {qs[k]:g}: slope by group {np.round(sl, 4).tolist()}  total {sl.sum():.4f}; analytic U slope {(U[k]-U[0])/(qs[k]-qs[0]):.4f}")
