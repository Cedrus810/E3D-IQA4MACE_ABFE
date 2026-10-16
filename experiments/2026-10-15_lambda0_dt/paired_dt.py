"""Paired 2 fs / 1 fs NVE at one path state (v3 S8 route step 1, S13.42).

Run: python paired_dt.py Q OUT_PREFIX [n_equil] [ps] [seed] [dts]

E7's q 2.0 drift fell 5.85 -> 1.83 kJ/mol/ps from 2 fs to 1 fs, but the two
runs started from different states. Here one Langevin run (2 fs, 298.15 K,
1/ps, thermal200 frame 0, seed 7, as E7) gives one state, and NVE at each dt
in `dts` (default 2,1) starts from it for the same `ps`. Every 10 fs (the
same times at every dt) it records:
  E_tot, KE, PE                         kJ/mol
  excess = S - S_eff of the D17 cap      kJ/mol (from the force call at that step)
  sum_c, truth                           kJ/mol
  cover: pairs with r < r_cov, pairs in the switch zone (r_cov - 0.2, r_cov),
         min r / r_cov                   (ligand x inner-water atoms, ML region)
  U_wall, closest L-w contact r / r0
It saves the trajectory record to OUT_PREFIX_dt<dt>.npz and, from the
largest dt, the frames with the largest excess, the most switch-zone pairs
and the closest contact to OUT_PREFIX_frames.npz (for fd_check.py).

ML in float32 on the GPU, the operating head (S13.37), production settings
of build_mlmm_system (revised D15 radii, cover switch, cap m = 0.5 eV).
"""

import sys
import time

import numpy as np
import openmm
from openmm import unit

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm  # noqa: E402
from decomp.lambda_mask import COVER_DELTA, COVER_R  # noqa: E402
from run_abfe import load_model  # noqa: E402

E3D = "/home/ruigengji/E3D"
LEG = f"{E3D}/em_solvent.npz"
BB = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"
CKPT = f"{E3D}/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
START = f"{E3D}/experiments/2026-10-12_thermal_frames/thermal200_solvent.npz"
KJ = unit.kilojoule_per_mole
EV = 96.48533212
SAMPLE_FS = 10.0

a = sys.argv[1:]
if len(a) < 2:
    sys.exit(__doc__)
q, prefix = float(a[0]), a[1]
n_equil = int(a[2]) if len(a) > 2 else 500
ps = float(a[3]) if len(a) > 3 else 2.0
seed = int(a[4]) if len(a) > 4 else 7
dts = [float(x) for x in a[5].split(",")] if len(a) > 5 else [2.0, 1.0]

d = np.load(LEG, allow_pickle=True)
box = np.asarray(d["box"], dtype=np.float64)
num = np.asarray(d["numbers"])
pos0 = np.asarray(np.load(START, allow_pickle=True)["positions"][0], dtype=np.float64)
t0 = time.perf_counter()
model = load_model(CKPT, BB, device="cuda")
system, mlf, info = mlmm.build_mlmm_system(LEG, model, n_inner=20, es=True, device="cuda", positions=pos0)
lig = info["lig"]
watm = np.where(d["resnames"] == "HOH")[0]
nl = len(lig)
r0 = np.array([[mlmm.WALL_R0[int(z)][0 if int(w) == 8 else 1] for w in num[watm]] for z in num[lig]])
Zml = num[mlf.ml_idx]
rcov = np.array([[COVER_R[(int(x), int(y))] for y in Zml[nl:]] for x in Zml[:nl]])

af = mlf.anch
_U = af._U
stash = {}


def _U_logged(data, lam):                       # keep the cap's parts of the last force call
    U, c, parts = _U(data, lam)
    for k in ("excess", "truth", "sum_c"):
        if k in parts:
            stash[k] = float(parts[k]) * EV
    return U, c, parts


af._U = _U_logged


def context(integ):
    c = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"))
    c.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])
    lam, s_w, p = mlmm.set_path(c, mlf, q, lig)
    return c, (lam, s_w, p)


lang = openmm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
lang.setRandomNumberSeed(seed)
c, path = context(lang)
c.setPositions(pos0 * unit.angstrom)
c.setVelocitiesToTemperature(298.15 * unit.kelvin, seed)
print(f"q {q}: (lambda, s_w, p) = {path}; setup {time.perf_counter() - t0:.0f}s; "
      f"Langevin {n_equil} x 2 fs, then NVE {ps} ps at dt {dts} fs", flush=True)
t = time.perf_counter()
lang.step(n_equil)
st = c.getState(getPositions=True, getVelocities=True)
print(f"  Langevin: {(time.perf_counter() - t) / max(n_equil, 1) * 1e3:.0f} ms/step", flush=True)
P_start = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
V_start = st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
np.savez(f"{prefix}_start.npz", positions=P_start, velocities=V_start, q=q)
del c, lang

dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
frames = {}
for dt in dts:
    verlet = openmm.VerletIntegrator(dt * unit.femtosecond)
    c2, _ = context(verlet)
    c2.setPositions(P_start * unit.angstrom)
    c2.setVelocities(V_start * unit.nanometer / unit.picosecond)
    every = int(round(SAMPLE_FS / dt))
    n_steps = int(round(ps * 1000 / dt))
    rec = {k: [] for k in ["t_ps", "E_tot", "KE", "PE", "excess", "sum_c", "truth", "n_cov", "n_zone",
                           "min_rcov", "U_wall", "contact"]}
    keep = []
    t = time.perf_counter()
    for k in range(0, n_steps + 1, every):
        if k:
            verlet.step(every)
        s = c2.getState(getEnergy=True, getPositions=True)
        ke, pe = s.getKineticEnergy().value_in_unit(KJ), s.getPotentialEnergy().value_in_unit(KJ)
        P = s.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        X = mlf.ml_positions(np.asarray(P, dtype=np.float64), box)
        D = np.linalg.norm(X[:nl, None] - X[None, nl:], axis=-1)
        dv = P[lig][:, None] - P[watm][None]
        dv -= box * np.round(dv / box)
        rec["t_ps"].append(k * dt / 1000)
        rec["E_tot"].append(ke + pe); rec["KE"].append(ke); rec["PE"].append(pe)
        for key in ("excess", "sum_c", "truth"):
            rec[key].append(stash.get(key, np.nan))
        rec["n_cov"].append(int((D < rcov).sum()))
        rec["n_zone"].append(int(((D < rcov) & (D > rcov - COVER_DELTA)).sum()))
        rec["min_rcov"].append(float((D / rcov).min()))
        rec["U_wall"].append(c2.getState(getEnergy=True, groups={mlmm.G_WALL}).getPotentialEnergy().value_in_unit(KJ))
        rec["contact"].append(float((np.sqrt((dv ** 2).sum(-1)) / r0).min()))
        if dt == max(dts):
            keep.append(np.asarray(P, dtype=np.float64))
        if not np.isfinite(ke + pe):
            print(f"  dt {dt}: non-finite energy at step {k}", flush=True)
            break
    ms = (time.perf_counter() - t) / max(n_steps, 1) * 1e3
    R = {k: np.array(v) for k, v in rec.items()}
    np.savez(f"{prefix}_dt{dt:g}.npz", q=q, dt=dt, **R)
    tt, E = R["t_ps"], R["E_tot"]
    h = len(E) // 2
    drift = np.polyfit(tt, E, 1)[0]
    sd = (E - np.polyval(np.polyfit(tt, E, 1), tt)).std()
    T = 2 * R["KE"].mean() / (dof * 0.0083144626)
    ex = R["excess"]
    print(f"  dt {dt:g} fs: {ms:.0f} ms/step (incl. records); drift {drift:+.2f} kJ/mol/ps "
          f"(halves {np.polyfit(tt[:h], E[:h], 1)[0]:+.2f} / {np.polyfit(tt[h:], E[h:], 1)[0]:+.2f}), "
          f"end - start {E[-1] - E[0]:+.2f}, sd {sd:.2f} kJ/mol, T {T:.1f} K", flush=True)
    print(f"     cap excess kJ/mol p50 / p90 / max {np.nanpercentile(ex, 50):.3f} / {np.nanpercentile(ex, 90):.3f} / "
          f"{np.nanmax(ex):.2f}; samples with excess > 1 kT {100 * np.mean(ex > 2.479):.1f}%; "
          f"sum_c p50 {np.nanmedian(R['sum_c']):.1f}, truth p50 {np.nanmedian(R['truth']):.1f} kJ/mol", flush=True)
    print(f"     cover: samples with a pair below r_cov {100 * np.mean(R['n_cov'] > 0):.1f}% "
          f"(max {R['n_cov'].max()} pairs; in the switch zone max {R['n_zone'].max()}), "
          f"min r/r_cov {R['min_rcov'].min():.3f}; U_wall p50 / max {np.median(R['U_wall']):.2f} / "
          f"{R['U_wall'].max():.2f} kJ/mol; contact r/r0 min {R['contact'].min():.3f}", flush=True)
    if keep:
        pick = {"max_excess": int(np.nanargmax(ex)), "max_zone": int(np.argmax(R["n_zone"])),
                "min_contact": int(np.argmin(R["contact"]))}
        frames = {k: keep[i] for k, i in pick.items()}
        frames.update({f"{k}_t_ps": tt[i] for k, i in pick.items()})
    del c2, verlet
if frames:
    np.savez(f"{prefix}_frames.npz", q=q, **frames)
print(f"done [{time.perf_counter() - t0:.0f}s]", flush=True)
