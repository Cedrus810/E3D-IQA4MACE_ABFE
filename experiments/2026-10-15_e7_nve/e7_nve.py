"""E7 (RESEARCH_PLAN_v3.md S6.3.5): is the ML/MM + FIRES + wall potential continuous?

Run: python e7_nve.py Q OUT.npz [n_equil] [n_nve] [dt_fs] [seed] [start.npz:frame]
     Q = path coordinate in [0, 4] (S6.3.4), or "mm" for the same box in pure MM

Langevin (298.15 K, 1/ps) for n_equil steps at Q, then NVE (velocity Verlet)
for n_nve steps from that state. Every NVE step it records the total energy
and its parts, so the drift and the jumps of E_tot can be compared with pure
MM on the same box at the same step. Discontinuities in U show up as steps in
E_tot that pure MM does not have.

ML in float32 on the GPU (production precision); the operating head (S13.37).
Start: em_solvent.npz, or a frame of an unminimised MD stack (e.g.
thermal200_solvent.npz:0), which also picks the 20 inner waters. Every 10 NVE
steps it also records the closest ligand-water contact as r / r0 (wall radii,
D15): at lambda = 0 that sets eps_w (S6.3.3).
"""

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
KJ = unit.kilojoule_per_mole


def main(q, out, n_equil=1000, n_nve=1000, dt=2.0, seed=7, start=None):
    d = np.load(LEG, allow_pickle=True)
    box = np.asarray(d["box"], dtype=np.float64)
    pos0 = np.asarray(d["positions"], dtype=np.float64)
    if start:
        f_, k_ = start.rsplit(":", 1)
        st0 = np.load(f_, allow_pickle=True)
        assert (st0["numbers"] == d["numbers"]).all() and np.allclose(st0["box"], box)
        pos0 = np.asarray(st0["positions"][int(k_)], dtype=np.float64)
    num = np.asarray(d["numbers"])
    ligm = np.where(d["resids"] == d["resids"][0])[0]
    watm = np.where(d["resnames"] == "HOH")[0]
    r0 = np.array([[mlmm.WALL_R0[int(z)][0 if int(w) == 8 else 1] for w in num[watm]]
                   for z in num[ligm]])                                   # [n_lig, n_wat] A
    t0 = time.perf_counter()
    if q == "mm":
        system = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
        mlf, lig = None, None
    else:
        model = load_model(CKPT, BB, device="cuda")
        system, mlf, info = mlmm.build_mlmm_system(LEG, model, n_inner=20, es=True,
                                                   device="cuda", positions=pos0)
        lig = info["lig"]
    groups = sorted({f.getForceGroup() for f in system.getForces()})

    def context(integ):
        c = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"))
        c.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])
        if mlf is not None:
            mlmm.set_path(c, mlf, float(q), lig)
        return c

    lang = openmm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1.0 / unit.picosecond,
                                           dt * unit.femtosecond)
    lang.setRandomNumberSeed(seed)
    c = context(lang)
    c.setPositions(pos0 * unit.angstrom)
    c.setVelocitiesToTemperature(298.15 * unit.kelvin, seed)
    print(f"Q {q}, dt {dt} fs, {system.getNumParticles()} atoms, groups {groups}, "
          f"setup {time.perf_counter() - t0:.0f}s", flush=True)
    t = time.perf_counter()
    lang.step(n_equil)
    t_eq = (time.perf_counter() - t) / max(n_equil, 1)
    st = c.getState(getPositions=True, getVelocities=True)
    print(f"  Langevin {n_equil} steps: {t_eq * 1e3:.1f} ms/step", flush=True)

    verlet = openmm.VerletIntegrator(dt * unit.femtosecond)
    c2 = context(verlet)
    c2.setPositions(st.getPositions())
    c2.setVelocities(st.getVelocities())
    rec = {k: [] for k in ["E_tot", "KE", "PE"] + [f"g{g}" for g in groups]}
    fires, contact, wall1 = [], [], []
    t = time.perf_counter()
    for k in range(n_nve + 1):
        if k:
            verlet.step(1)
        s = c2.getState(getEnergy=True)
        ke, pe = s.getKineticEnergy().value_in_unit(KJ), s.getPotentialEnergy().value_in_unit(KJ)
        rec["KE"].append(ke); rec["PE"].append(pe); rec["E_tot"].append(ke + pe)
        if k % 10 == 0 or k == n_nve:
            p = c2.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            dv = p[ligm][:, None] - p[watm][None]
            dv -= box * np.round(dv / box)
            contact.append(float((np.sqrt((dv ** 2).sum(-1)) / r0).min()))
            if mlf is not None:                    # W1 FEP: the wall's energy at s_w = 1 here
                s_now = c2.getParameter("s_w")
                c2.setParameter("s_w", 1.0)
                wall1.append(c2.getState(getEnergy=True, groups={mlmm.G_WALL})
                             .getPotentialEnergy().value_in_unit(KJ))
                c2.setParameter("s_w", s_now)
            for g in groups:
                rec[f"g{g}"].append(c2.getState(getEnergy=True, groups={g})
                                    .getPotentialEnergy().value_in_unit(KJ))
            if mlf is not None:
                fires.append((mlf.last.get("U_fires", 0.0), mlf.last.get("n_viol", 0),
                              mlf.last.get("max_depth", 0.0)))
    t_nve = (time.perf_counter() - t) / max(n_nve, 1)
    E = np.array(rec["E_tot"])
    tt = np.arange(len(E)) * dt / 1000                          # ps
    slope = np.polyfit(tt, E, 1)[0] if len(E) > 2 else np.nan
    dE = np.diff(E)
    dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
    T = 2 * np.array(rec["KE"]) / (dof * 0.0083144626)
    np.savez(out, q=str(q), dt=dt, t_ps=tt, fires=np.array(fires), contact=np.array(contact), wall1=np.array(wall1),
             **{k: np.array(v) for k, v in rec.items()})
    print(f"  NVE {n_nve} steps: {t_nve * 1e3:.1f} ms/step; E_tot drift {slope:+.3f} kJ/mol/ps, "
          f"sd {E.std():.3f}, max |step| {np.abs(dE).max() if len(dE) else 0:.3f} kJ/mol; "
          f"T {T.mean():.1f} K; closest contact r/r0 min {min(contact):.2f}"
          + (f"; U_F max {max(f[0] for f in fires):.2f}, viol max {max(f[1] for f in fires)}"
             if fires else "") + f"  -> {out}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 2:
        sys.exit(__doc__)
    main(a[0], a[1], n_equil=int(a[2]) if len(a) > 2 else 1000,
         n_nve=int(a[3]) if len(a) > 3 else 1000, dt=float(a[4]) if len(a) > 4 else 2.0,
         seed=int(a[5]) if len(a) > 5 else 7, start=a[6] if len(a) > 6 else None)
