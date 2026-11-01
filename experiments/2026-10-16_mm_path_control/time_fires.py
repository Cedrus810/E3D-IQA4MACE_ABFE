"""Step time of the all-MM control with and without the FIRES PythonForce (acetamide leg)."""
import sys, time
import numpy as np, openmm
from openmm import unit
sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm
LEG = "/home/ruigengji/E3D/experiments/2026-10-15_acetamide_leg/em_acetamide.npz"
d = np.load(LEG, allow_pickle=True); P = np.asarray(d["positions"]); box = np.asarray(d["box"])
for dev in ("cuda", "cpu"):
    for keep in (True, False):
        system, mlf, info = mlmm.build_mlmm_system(LEG, None, n_inner=20, device=dev, positions=P, mm_only=True)
        if not keep:
            k = [i for i, f in enumerate(system.getForces()) if f.getForceGroup() == mlmm.G_ML][0]
            system.removeForce(k)
        integ = openmm.LangevinMiddleIntegrator(298.15, 1.0, 0.002)
        c = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"))
        c.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])
        mlmm.set_path(c, mlf, 1.5, info["lig"]); c.setPositions(P * unit.angstrom)
        integ.step(20); c.getState(getEnergy=True)
        t = time.perf_counter(); integ.step(500); c.getState(getEnergy=True)
        print(f"FIRES on {dev if keep else '-':4s} {'with' if keep else 'without'} FIRES: {(time.perf_counter() - t) / 500 * 1e3:.2f} ms/step", flush=True)
        if not keep:
            break
