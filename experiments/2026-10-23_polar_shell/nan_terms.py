"""Smoke: which energy term is NaN on the POLAR-shell path at the leg start frame (no dynamics)."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, openmm, torch
from openmm import unit
from decomp import mlmm
from run_abfe import load_model

leg = ROOT / "experiments/2026-10-15_acetamide_leg/em_acetamide.npz"
ckpt = ROOT / "runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
start = np.asarray(np.load(leg, allow_pickle=True)["positions"], dtype=np.float64)
model = load_model(str(ckpt), "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", device="cuda")
system, force, info = mlmm.build_mlmm_system(str(leg), model, n_inner=20, device="cuda", positions=start,
                                             shell_mm=False, polar_shell=True, fast_forces=True, batch_fragments=True)
for i, f in enumerate(system.getForces()):
    f.setForceGroup(i)
integ = openmm.VerletIntegrator(1e-6)
ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"))
box = np.asarray(info["box"], dtype=np.float64)
ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[a] * box[a] / 10)) for a in range(3)])
for q in (3.0, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0):
    mlmm.set_path(ctx, force, q, info["lig"], shell_mm=False, polar_shell=True)
    ctx.setPositions(start * unit.angstrom)
    U, F = force.evaluate(start, box)
    fires = force.fires
    terms = []
    for i, f in enumerate(system.getForces()):
        st = ctx.getState(getEnergy=True, getForces=True, groups={i})
        e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        fm = np.abs(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.angstrom)).max()
        if not (np.isfinite(e) and np.isfinite(fm)) or i == 0:
            terms.append(f"{i}:{type(f).__name__}(E={e:.4g},maxF={fm:.3g})")
    params = {k: v for k, v in ctx.getParameters().items()}
    print(f"q={q:g}: ML+FIRES evaluate U={U:.4g} finiteF={np.isfinite(F).all()}; NaN groups: {terms[1:] or 'none'}; "
          f"params={params}", flush=True)
