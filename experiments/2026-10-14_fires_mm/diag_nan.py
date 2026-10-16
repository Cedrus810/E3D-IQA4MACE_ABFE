"""Reproduce the e0_fires NaN (seed 7, CUDA, torch cuda) with FIRES diagnostics.

Run: python diag_nan.py [ps] [chunk_steps] [dt_fs]

Same start and seed as e0_fires.npz's run. Every chunk: U_F, violating
pairs, deepest violation, largest FIRES force and on which atom (element,
inner/outer/ligand), T. Stops at the first NaN or exception and saves the last
good positions and velocities to diag_last_good.npz.
"""

import sys
import time

import numpy as np
import openmm
import torch
from openmm import unit

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp.fires import Fires, water_indices  # noqa: E402

E3D = "/home/ruigengji/E3D"
ps = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 50
dt = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

d = np.load(f"{E3D}/em_solvent.npz", allow_pickle=True)
system = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
box = np.asarray(d["box"], dtype=np.float64)
num = d["numbers"]
lig = np.where(d["resids"] == d["resids"][0])[0]
W = water_indices(d["resnames"], d["resids"])
fires = Fires(lig, W, n_inner=20, device="cuda")
inner = fires.select_inner(d["positions"], box)
role = np.full(len(num), "outer", dtype=object)
role[lig] = "ligand"
role[W[inner].ravel()] = "inner"
f = openmm.PythonForce(fires)
f.setUsesPeriodicBoundaryConditions(True)
system.addForce(f)
integ = openmm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1.0 / unit.picosecond,
                                        dt * unit.femtosecond)
integ.setRandomNumberSeed(7)
ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"))
ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])
ctx.setPositions(d["positions"] * unit.angstrom)
ctx.setVelocitiesToTemperature(298.15 * unit.kelvin, 7)
dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
print(f"dt {dt} fs, chunk {chunk} steps, up to {ps} ps", flush=True)

n_chunks = int(round(ps * 1000 / dt / chunk))
good, t = None, time.perf_counter()
worst = []
for c in range(n_chunks):
    try:
        integ.step(chunk)
        st = ctx.getState(getPositions=True, getVelocities=True, getEnergy=True)
        p = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        if not np.isfinite(p).all():
            raise ValueError("NaN positions")
    except Exception as e:  # noqa: BLE001
        print(f"FAILED in chunk {c} ({(c + 1) * chunk * dt / 1000:.2f} ps): {e}", flush=True)
        break
    U, Fr = fires.energy_and_forces(np.asarray(p), box)
    a = int(np.argmax(np.abs(Fr).max(1)))
    T = 2 * st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) / (dof * 0.0083144626)
    L = fires.last
    worst.append((L["max_depth"], np.abs(Fr).max()))
    if c % 20 == 0 or L["n_viol"] > 6 or np.abs(Fr).max() > 2000:
        print(f"  {(c + 1) * chunk * dt / 1000:7.2f} ps  T {T:5.0f}  U_F {U:9.2f}  viol {L['n_viol']:3d}  "
              f"depth {L['max_depth']:.3f}  max|F_F| {np.abs(Fr).max():8.1f} kJ/mol/A on "
              f"{role[a]} Z={num[a]}  R_inner {L['r_inner']:.2f}", flush=True)
    good = (p, st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond), c)
else:
    print(f"no failure in {ps} ps", flush=True)
wd = np.array(worst)
print(f"over {len(wd)} chunks: max depth {wd[:, 0].max():.3f} A, max |F_F| {wd[:, 1].max():.1f} kJ/mol/A "
      f"[{time.perf_counter() - t:.0f}s]", flush=True)
if good is not None:
    np.savez("diag_last_good.npz", positions=good[0], velocities=good[1], chunk=good[2], inner=inner)
