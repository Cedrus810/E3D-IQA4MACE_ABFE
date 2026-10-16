"""Full ML/MM step time at production settings (v3 S8 route step 2, S8.0.2).

Run: python time_step.py [n_inner list, e.g. 20,50] [q list, e.g. 1.5,2.5] [n_steps]

For each n_inner (ML waters) and path state q: the Atenolol solvent leg
(thermal200 frame 0, which also picks the inner waters), the operating head
(S13.37) in float32 on the GPU, build_mlmm_system with its production
defaults (revised D15 radii, cover switch, cap m = 0.5 eV, eps_w 25).
LangevinMiddle 298.15 K, 2 fs. 20 warm-up steps, then n_steps timed with no
energy records (the production integrator loop). Also timed: the same box in
pure MM, and one ML + FIRES force call on its own (MLMMForce.evaluate).
Prints ms/step and ns/day; writes timing_<host>.json next to this file.
"""

import json
import socket
import sys
import time

import numpy as np
import openmm
import torch
from openmm import unit

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm  # noqa: E402
from run_abfe import load_model  # noqa: E402

E3D = "/home/ruigengji/E3D"
LEG = f"{E3D}/em_solvent.npz"
BB = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"
CKPT = f"{E3D}/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
START = f"{E3D}/experiments/2026-10-12_thermal_frames/thermal200_solvent.npz"
OUT = f"{E3D}/experiments/2026-10-16_5090_timing"

a = sys.argv[1:]
ns = [int(x) for x in a[0].split(",")] if a else [20, 50]
qs = [float(x) for x in a[1].split(",")] if len(a) > 1 else [1.5, 2.5]
n_steps = int(a[2]) if len(a) > 2 else 200

d = np.load(LEG, allow_pickle=True)
box = np.asarray(d["box"], dtype=np.float64)
pos0 = np.asarray(np.load(START, allow_pickle=True)["positions"][0], dtype=np.float64)
gpu = torch.cuda.get_device_name(0)
host = socket.gethostname()
print(f"{host}: {gpu}; openmm {openmm.__version__}, torch {torch.__version__}", flush=True)


def time_ctx(system, setup):
    integ = openmm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
    integ.setRandomNumberSeed(7)
    c = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"))
    c.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])
    setup(c)
    c.setPositions(pos0 * unit.angstrom)
    c.setVelocitiesToTemperature(298.15 * unit.kelvin, 7)
    integ.step(20)
    c.getState(getEnergy=True)
    t = time.perf_counter()
    integ.step(n_steps)
    c.getState(getEnergy=True)                      # sync
    return (time.perf_counter() - t) / n_steps * 1e3, c


rows = []
mm = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
ms, _ = time_ctx(mm, lambda c: None)
rows.append(dict(kind="pure MM", n=0, q=None, ms=ms))
print(f"pure MM: {ms:.2f} ms/step ({2e-6 * 86400e3 / ms:.1f} ns/day)", flush=True)

model = load_model(CKPT, BB, device="cuda")
for n in ns:
    system, mlf, info = mlmm.build_mlmm_system(LEG, model, n_inner=n, es=True, device="cuda", positions=pos0)
    for q in qs:
        ms, c = time_ctx(system, lambda c: mlmm.set_path(c, mlf, q, info["lig"]))
        P = c.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        P = np.asarray(P, dtype=np.float64)
        mlf.evaluate(P, box)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            mlf.evaluate(P, box)
        torch.cuda.synchronize()
        ml = (time.perf_counter() - t) / 20 * 1e3
        rows.append(dict(kind="ML/MM", n=n, q=q, ms=ms, ml_call_ms=ml, n_ml_atoms=int(len(mlf.ml_idx))))
        print(f"n {n:3d} ({len(mlf.ml_idx)} ML atoms), q {q}: {ms:.1f} ms/step ({2e-6 * 86400e3 / ms:.3f} ns/day); "
              f"ML + FIRES call alone {ml:.1f} ms", flush=True)
        del c
json.dump(dict(host=host, gpu=gpu, n_steps=n_steps, rows=rows), open(f"{OUT}/timing_{host}.json", "w"), indent=1)
print(f"-> {OUT}/timing_{host}.json", flush=True)
