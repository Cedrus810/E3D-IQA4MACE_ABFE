"""Reproduce the q = 2.0 (lambda = 0+, wall on) NaN of run_e7.sh with diagnostics.

Run: python diag_q.py Q [n_steps] [dt_fs] [chunk] [seed]

Same setup as e7_nve.py (thermal200 frame 0, operating head, Langevin 298.15 K,
seed 7). Every `chunk` steps: T; closest ligand-water contact r / r0 and the
pair; largest force of the ML + FIRES group and its atom; E_bb, sum c
(attributed coupling) and U_F; largest MM force (groups 0-3). Stops at the
first NaN and saves the last good state to diag_q<Q>_last.npz.
"""

import sys
import time

import numpy as np
import openmm
from openmm import unit

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm  # noqa: E402
from run_abfe import load_model  # noqa: E402

E3D = "/home/ruigengji/E3D"
LEG = f"{E3D}/em_solvent.npz"
BB = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"
CKPT = f"{E3D}/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
START = f"{E3D}/experiments/2026-10-12_thermal_frames/thermal200_solvent.npz"
EV = 96.48533212

a = sys.argv[1:]
q = float(a[0])
n_steps = int(a[1]) if len(a) > 1 else 600
dt = float(a[2]) if len(a) > 2 else 2.0
chunk = int(a[3]) if len(a) > 3 else 10
seed = int(a[4]) if len(a) > 4 else 7

d = np.load(LEG, allow_pickle=True)
box = np.asarray(d["box"], dtype=np.float64)
num = np.asarray(d["numbers"])
pos0 = np.asarray(np.load(START, allow_pickle=True)["positions"][0], dtype=np.float64)
model = load_model(CKPT, BB, device="cuda")
system, mlf, info = mlmm.build_mlmm_system(LEG, model, n_inner=20, es=True, device="cuda",
                                           positions=pos0)
lig, watm = info["lig"], np.where(d["resnames"] == "HOH")[0]
inner_atoms = set(info["inner_atoms"].tolist())
r0 = np.array([[mlmm.WALL_R0[int(z)][0 if int(w) == 8 else 1] for w in num[watm]] for z in num[lig]])
integ = openmm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1.0 / unit.picosecond, dt * unit.femtosecond)
integ.setRandomNumberSeed(seed)
ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"))
ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])
lam, s_w, p = mlmm.set_path(ctx, mlf, q, lig)
ctx.setPositions(pos0 * unit.angstrom)
ctx.setVelocitiesToTemperature(298.15 * unit.kelvin, seed)
dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
print(f"q {q}: lambda {lam}, s_w {s_w}, p {p}; dt {dt} fs; chunk {chunk}", flush=True)


def role(i):
    return "L" if i in set(lig.tolist()) else ("I" if i in inner_atoms else "O")


good, t = None, time.perf_counter()
for c in range(n_steps // chunk):
    try:
        integ.step(chunk)
        st = ctx.getState(getPositions=True, getVelocities=True, getEnergy=True)
        P = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        if not np.isfinite(P).all():
            raise ValueError("NaN positions")
    except Exception as e:  # noqa: BLE001
        print(f"FAILED in chunk {c} (step {(c + 1) * chunk}): {e}", flush=True)
        break
    P = np.asarray(P)
    T = 2 * st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) / (dof * 0.0083144626)
    dv = P[lig][:, None] - P[watm][None]
    dv -= box * np.round(dv / box)
    rr = np.sqrt((dv ** 2).sum(-1)) / r0
    li, wi = np.unravel_index(np.argmin(rr), rr.shape)
    F_ml = ctx.getState(getForces=True, groups={mlmm.G_ML}).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer) / 10
    F_mm = ctx.getState(getForces=True, groups={0, 1, 2, 3}).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer) / 10
    fa = int(np.argmax(np.linalg.norm(F_ml, axis=1)))
    fm = int(np.argmax(np.linalg.norm(F_mm, axis=1)))
    af = mlf.anch
    data = af._graph(mlf.ml_positions(P, box))
    _, _, parts = af._U(data, af.lam)
    print(f"  step {(c + 1) * chunk:5d}  T {T:5.0f}  contact {rr.min():.2f} r0 "
          f"(L{lig[li]} Z{num[lig[li]]} - {role(watm[wi])} Z{num[watm[wi]]})  "
          f"max|F_ML+F| {np.linalg.norm(F_ml[fa]):7.1f} on {role(fa)} Z{num[fa]}  "
          f"max|F_MM| {np.linalg.norm(F_mm[fm]):7.1f} on {role(fm)} Z{num[fm]} kJ/mol/A  "
          f"sum_c {float(parts['sum_c']) * EV:9.1f}  U_F {mlf.last.get('U_fires', 0):6.2f}", flush=True)
    good = (P, st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond), c)
else:
    print(f"no failure in {n_steps} steps", flush=True)
print(f"[{time.perf_counter() - t:.0f}s]", flush=True)
if good is not None:
    np.savez(f"diag_q{q}_last.npz", positions=good[0], velocities=good[1], chunk=good[2])
