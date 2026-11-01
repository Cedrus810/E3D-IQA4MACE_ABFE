"""Smoke: POLAR-shell q=5 from the leg start frame, every 5 steps: per-group energies, ML U / max|F|,
closest ligand-water and inner-inner (different molecules) distances. Stops at the first NaN and saves
the last finite and first NaN frames to nan_trace.npz. Diagnostic only."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, openmm, torch
from openmm import unit
from decomp import mlmm
from run_abfe import load_model

Q = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
leg = ROOT / "experiments/2026-10-15_acetamide_leg/em_acetamide.npz"
ckpt = ROOT / "runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
d = np.load(leg, allow_pickle=True)
start = np.asarray(d["positions"], dtype=np.float64)
model = load_model(str(ckpt), "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", device="cuda")
system, force, info = mlmm.build_mlmm_system(str(leg), model, n_inner=20, device="cuda", positions=start,
                                             shell_mm=False, polar_shell=True, fast_forces=True, batch_fragments=True)
for i, f in enumerate(system.getForces()):
    f.setForceGroup(i)
names = [type(f).__name__ for f in system.getForces()]
integ = openmm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
integ.setRandomNumberSeed(7)
ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"))
box = np.asarray(info["box"], dtype=np.float64)
ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[a] * box[a] / 10)) for a in range(3)])
mlmm.set_path(ctx, force, Q, info["lig"], shell_mm=False, polar_shell=True)
ctx.setPositions(start * unit.angstrom)
ctx.setVelocitiesToTemperature(298.15 * unit.kelvin, 7)
lig = np.asarray(info["lig"]); inner = np.asarray(info["inner"])
inner_atoms = np.concatenate([np.arange(o, o + 3) for o in np.ravel(inner)]) if np.ndim(inner) == 1 else np.ravel(inner)
mol = np.repeat(np.arange(len(inner_atoms) // 3), 3)
print("groups:", list(enumerate(names)), "inner shape", np.shape(inner), flush=True)
def mind(a, b, same=None):
    r = a[:, None] - b[None]; r -= box * np.round(r / box); dd = np.linalg.norm(r, axis=-1)
    if same is not None: dd[same] = np.inf
    return dd.min()
prev = start
for step in range(0, 400, 5):
    st = ctx.getState(getPositions=True)
    P = np.asarray(st.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    E = {i: ctx.getState(getEnergy=True, groups={i}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
         for i in range(len(names))}
    U, F = force.evaluate(P, box)
    Fn = np.linalg.norm(F, axis=1)
    bad = [i for i, e in E.items() if not np.isfinite(e)]
    print(f"step {step}: ML+FIRES U={U:.6g} max|F|={np.nanmax(Fn):.3g} at atom {int(np.nanargmax(Fn))}"
          f" (lig max {np.nanmax(Fn[lig]):.3g}); min d(lig,inner)={mind(P[lig], P[inner_atoms]):.2f}"
          f" min d(inner,inner other mol)={mind(P[inner_atoms], P[inner_atoms], mol[:, None] == mol[None]):.2f};"
          f" non-finite groups {[(i, names[i]) for i in bad]}", flush=True)
    if bad or not np.isfinite(U) or not np.isfinite(P).all():
        np.savez(pathlib.Path(__file__).with_name("nan_trace.npz"), last_good=prev, first_bad=P, step=step, q=Q)
        break
    prev = P
    integ.step(5)
