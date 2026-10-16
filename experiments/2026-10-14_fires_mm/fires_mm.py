"""E0 (RESEARCH_PLAN_v3.md S6.3.5): FIRES on pure MM, against no restraint.

Run: python fires_mm.py MODE SEED OUT.npz [ns] [platform] [device] [t0_ps]
     MODE = free | fires

Rowley & Roux's own check (JCTC 2012, Fig. 2) on our box and metric: the
prepared Atenolol solvent leg (4,076 atoms, TIP3P, PME, 35 A box, NVT), MM
only, Langevin 298.15 K, 1/ps, 2 fs (the paper: 1 fs; D16 takes 2 fs and
checks it here). `fires` adds decomp/fires.py (heavy-atom ordering, S13.38) with n = 20 inner waters
(chosen nearest at t = 0, never changed), k = 500 kcal/mol/A^2.

Every 0.5 ps after 100 ps of equilibration it records the ordering distance
d_w of every water (softmin closest-atom, the restraint's own metric), so the
two runs can be compared by the density of waters against d, and the FIRES
diagnostics (U_F, violating pairs, deepest violation, R_inner).
"""

import sys
import time

import numpy as np
import openmm
import torch
from openmm import unit

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp.fires import Fires, heavy_selection, water_indices  # noqa: E402

E3D = "/home/ruigengji/E3D"


def main(mode, seed, out, ns=1.0, platform="CUDA", device="cpu", t0_ps=100.0,
         dense_ps=0.5, n_inner=20):
    d = np.load(f"{E3D}/em_solvent.npz", allow_pickle=True)
    system = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
    box = np.asarray(d["box"], dtype=np.float64)
    lig = np.where(d["resids"] == d["resids"][0])[0]
    W = water_indices(d["resnames"], d["resids"])
    fires = Fires(*heavy_selection(lig, d["numbers"], W), n_inner=n_inner, device=device)
    inner = fires.select_inner(d["positions"], box)
    if mode == "fires":
        f = openmm.PythonForce(fires)
        f.setUsesPeriodicBoundaryConditions(True)
        f.setForceGroup(31)
        system.addForce(f)
    elif mode != "free":
        sys.exit(f"MODE must be free or fires, not {mode!r}")

    integ = openmm.LangevinMiddleIntegrator(
        298.15 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
    integ.setRandomNumberSeed(seed)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName(platform))
    ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])
    ctx.setPositions(d["positions"] * unit.angstrom)
    ctx.setVelocitiesToTemperature(298.15 * unit.kelvin, seed)
    dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
    print(f"{mode}, seed {seed}, {ctx.getPlatform().getName()}, torch {device}, "
          f"{system.getNumParticles()} atoms, {len(W)} waters, n_inner {n_inner}, "
          f"k {fires.k:.0f} kJ/mol/A^2", flush=True)

    def observe():
        st = ctx.getState(getPositions=True, getEnergy=True)
        p = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        with torch.no_grad():
            dd = fires.distances(fires._t(p), fires._t(box)).cpu().numpy()
        fires.energy(fires._t(p), fires._t(box))                 # fills fires.last
        T = 2 * st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) / (dof * 0.0083144626)
        return dd, dict(fires.last), T

    t = time.perf_counter()
    integ.step(int(round(t0_ps * 500)))
    print(f"  equilibrated {t0_ps:.0f} ps [{time.perf_counter() - t:.0f}s]", flush=True)
    per = int(round(dense_ps * 500))
    n_obs = int(round(ns * 1000 / dense_ps))
    D, diag, temps = [], [], []
    t1 = time.perf_counter()
    for k in range(n_obs):
        integ.step(per)
        dd, last, T = observe()
        D.append(dd.astype(np.float32)); diag.append(last); temps.append(T)
        if k % max(1, n_obs // 10) == 0:
            el = time.perf_counter() - t1
            print(f"  {t0_ps + (k + 1) * dense_ps:8.1f} ps  T {T:5.0f} K  U_F {last['U']:8.3f}  "
                  f"viol {last['n_viol']:3d}  R_inner {last['r_inner']:.2f}  "
                  f"[{el:.0f}s, {el / ((k + 1) * per) * 1e3:.2f} ms/step]", flush=True)
    keys = ["U", "n_viol", "max_depth", "r_inner", "d_outer_min"]
    np.savez(out, d=np.stack(D), inner=inner, mode=mode, seed=seed, dense_ps=dense_ps,
             t0_ps=t0_ps, n_inner=n_inner, k=fires.k, beta=fires.beta, T=np.array(temps),
             **{k_: np.array([x[k_] for x in diag]) for k_ in keys})
    el = time.perf_counter() - t1
    print(f"  -> {out}: {n_obs} observations, {el:.0f}s, "
          f"{el / (n_obs * per) * 1e3:.2f} ms/step, mean T {np.mean(temps):.1f} K", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 3:
        sys.exit(__doc__)
    main(a[0], int(a[1]), a[2], ns=float(a[3]) if len(a) > 3 else 1.0,
         platform=a[4] if len(a) > 4 else "CUDA", device=a[5] if len(a) > 5 else "cpu",
         t0_ps=float(a[6]) if len(a) > 6 else 100.0)
