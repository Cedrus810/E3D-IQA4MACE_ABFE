"""MM hydration free energy of the acetamide leg: the MM control for Stage A.

Run: python abfe_mm.py check                         (lambda = 1 vs the reference System)
     python abfe_mm.py run REP [equil_ps] [prod_ps]  (all states in series; resumable)
     python abfe_mm.py analyze                       (MBAR over every finished rep)

Decoupling (couple-intramol = no), in the sign of E3D's dG_decouple and of
the IBS solvent legs: dG = G(decoupled) - G(coupled) > 0 for a soluble solute.
- Electrostatics: ligand charges = lambda_elec x q_i (NonbondedForce particle
  offsets, so direct + reciprocal space both scale). Every ligand-ligand pair
  without an exception gets one carrying its full q_i q_j and LJ, so the
  ligand's own terms do not depend on lambda (as lambda_frames.scaled_system,
  S13.36).
- Sterics: ligand epsilon = 0 in the NonbondedForce; ligand-water LJ moves to
  a CustomNonbondedForce (interaction group ligand x rest, same 1.0 nm
  cutoff, long-range correction on) with the Beutler softcore, alpha = 0.5:
    U = lambda_vdw 4 eps (1/x^2 - 1/x),  x = alpha (1 - lambda_vdw) + (r/sigma)^6
- States: lambda_elec 1 -> 0 in 11 steps at lambda_vdw = 1, then lambda_vdw
  1 -> 0 in 20 steps of 0.05 at lambda_elec = 0 (31 states).
Per state: Langevin 298.15 K, 1/ps, 2 fs, fixed box (the leg's NVT box);
equil_ps, then prod_ps sampled every 1 ps; at each sample the reduced
energy at every state is evaluated in the same context (u_kn). One npz per
(rep, state) in runs/; a rerun skips finished states. Analysis: pymbar MBAR
per rep after subsampling each state's series by its statistical
inefficiency (on the energy difference to its neighbours); mean and SD over
reps.
"""

import glob
import os
import sys
import time

import numpy as np
import openmm
from openmm import unit

HERE = "/home/ruigengji/E3D/experiments/2026-10-15_acetamide_leg"
LEG = f"{HERE}/em_acetamide.npz"
T = 298.15
KT = 0.0083144626 * T                                  # kJ/mol
ALPHA = 0.5
STATES = [(le, 1.0) for le in np.round(np.linspace(1, 0, 11), 4)] + \
         [(0.0, lv) for lv in np.round(np.linspace(1, 0, 21)[1:], 4)]
KJ = unit.kilojoule_per_mole


def build():
    d = np.load(LEG, allow_pickle=True)
    ref = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
    system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(ref))
    lig = [int(i) for i in np.where(d["resids"] == d["resids"][0])[0]]
    rest = [i for i in range(system.getNumParticles()) if i not in set(lig)]
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    nb.addGlobalParameter("lambda_elec", 1.0)
    prm = {i: nb.getParticleParameters(i) for i in lig}
    have = {tuple(sorted(nb.getExceptionParameters(k)[:2])) for k in range(nb.getNumExceptions())}
    for a, i in enumerate(lig):
        for j in lig[a + 1:]:
            if (i, j) not in have:
                (qi, si, ei), (qj, sj, ej) = prm[i], prm[j]
                nb.addException(i, j, qi * qj, 0.5 * (si + sj), (ei * ej).sqrt())
    sc = openmm.CustomNonbondedForce(
        "lambda_vdw*4*eps*(1/x^2 - 1/x); x = alpha*(1-lambda_vdw) + (r/sig)^6;"
        "sig = 0.5*(s1+s2); eps = sqrt(e1*e2)")
    sc.addGlobalParameter("lambda_vdw", 1.0)
    sc.addGlobalParameter("alpha", ALPHA)
    sc.addPerParticleParameter("s")
    sc.addPerParticleParameter("e")
    for i in range(system.getNumParticles()):
        _, s, e = nb.getParticleParameters(i)
        sc.addParticle([s.value_in_unit(unit.nanometer), e.value_in_unit(KJ)])
    for k in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(k)
        sc.addExclusion(i, j)
    sc.addInteractionGroup(lig, rest)
    sc.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    sc.setCutoffDistance(nb.getCutoffDistance())
    sc.setUseSwitchingFunction(False)
    sc.setUseLongRangeCorrection(nb.getUseDispersionCorrection())
    system.addForce(sc)
    for i in lig:
        q, s, e = prm[i]
        nb.setParticleParameters(i, 0.0, s, 0.0)
        nb.addParticleParameterOffset("lambda_elec", i, q, 0.0, 0.0)
    return d, ref, system, lig


def context(system, d, integ):
    c = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"))
    box = np.asarray(d["box"]) / 10
    c.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i])) for i in range(3)])
    return c


def set_state(c, k):
    c.setParameter("lambda_elec", STATES[k][0])
    c.setParameter("lambda_vdw", STATES[k][1])


def check():
    d, ref, system, lig = build()
    P = np.asarray(d["positions"]) / 10
    out = []
    for sys_ in (ref, system):
        c = context(sys_, d, openmm.VerletIntegrator(0.001))
        c.setPositions(P)
        if sys_ is system:
            set_state(c, 0)
        s = c.getState(getEnergy=True, getForces=True)
        out.append((s.getPotentialEnergy().value_in_unit(KJ), s.getForces(asNumpy=True).value_in_unit(KJ / unit.nanometer)))
    dF = np.abs(out[0][1] - out[1][1])
    print(f"lambda = 1 vs reference: E {out[1][0]:.3f} vs {out[0][0]:.3f} kJ/mol (diff {out[1][0] - out[0][0]:+.4f}); "
          f"max |dF| {dF.max():.4f} kJ/mol/nm (ligand {dF[lig].max():.4f})")
    c = context(system, d, openmm.VerletIntegrator(0.001))
    c.setPositions(P)
    for k in (10, 30):
        set_state(c, k)
        print(f"state {k} {STATES[k]}: E {c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ):.3f} kJ/mol")


def run(rep, equil_ps=100.0, prod_ps=1000.0):
    d, _, system, _ = build()
    os.makedirs(f"{HERE}/runs", exist_ok=True)
    integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
    integ.setRandomNumberSeed(1000 * rep + 1)
    c = context(system, d, integ)
    c.setPositions(np.asarray(d["positions"]) / 10)
    c.setVelocitiesToTemperature(T * unit.kelvin, 1000 * rep + 1)
    for k in range(len(STATES)):
        out = f"{HERE}/runs/rep{rep}_state{k:02d}.npz"
        set_state(c, k)
        if os.path.exists(out):                     # resume: continue from the saved last frame
            z = np.load(out)
            c.setPositions(z["last_positions"]); c.setVelocities(z["last_velocities"])
            continue
        t = time.perf_counter()
        integ.step(int(equil_ps / 0.002))
        n = int(prod_ps)
        u = np.zeros((len(STATES), n))
        for s in range(n):
            integ.step(500)
            for m in range(len(STATES)):
                set_state(c, m)
                u[m, s] = c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ) / KT
            set_state(c, k)
        st = c.getState(getPositions=True, getVelocities=True)
        np.savez(out, u=u, state=k, lam=np.array(STATES[k]),
                 last_positions=st.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                 last_velocities=st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond))
        dt = time.perf_counter() - t
        print(f"rep {rep} state {k:2d} {STATES[k]}: {equil_ps:.0f} + {prod_ps:.0f} ps in {dt:.0f}s "
              f"({(equil_ps + prod_ps) / 1000 / (dt / 86400):.0f} ns/day incl. u_kn)", flush=True)
    print(f"rep {rep} done", flush=True)


def analyze():
    import pymbar
    from pymbar import timeseries
    res = []
    for rep in sorted({int(os.path.basename(f)[3:].split("_")[0]) for f in glob.glob(f"{HERE}/runs/rep*_state*.npz")}):
        fs = [f"{HERE}/runs/rep{rep}_state{k:02d}.npz" for k in range(len(STATES))]
        if not all(os.path.exists(f) for f in fs):
            print(f"rep {rep}: incomplete, skipped")
            continue
        blocks, N = [], []
        for k, f in enumerate(fs):
            u = np.load(f)["u"]
            nb = [m for m in (k - 1, k + 1) if 0 <= m < len(STATES)]
            g = max(timeseries.statistical_inefficiency(u[m] - u[k]) for m in nb)
            idx = timeseries.subsample_correlated_data(u[k], g=g)
            blocks.append(u[:, idx]); N.append(len(idx))
        mbar = pymbar.MBAR(np.concatenate(blocks, axis=1), np.array(N))
        r = mbar.compute_free_energy_differences()
        f, df = r["Delta_f"] * KT, r["dDelta_f"] * KT
        e = 10
        res.append((f[0, -1], f[0, e], f[e, -1]))
        ov = mbar.compute_overlap()["matrix"]
        adj = min(ov[k, k + 1] for k in range(len(STATES) - 1))
        print(f"rep {rep}: dG_decouple {f[0, -1]:+.2f} +- {df[0, -1]:.2f} (MBAR) = elec {f[0, e]:+.2f} + vdW "
              f"{f[e, -1]:+.2f} kJ/mol; samples/state {min(N)}-{max(N)}; min adjacent overlap {adj:.3f}")
    if res:
        a = np.array(res)
        sd = a.std(0, ddof=1) if len(a) > 1 else np.full(3, np.nan)
        print(f"mean over {len(a)} reps: dG {a[:, 0].mean():+.2f} +- {sd[0]:.2f} (SD), elec {a[:, 1].mean():+.2f} "
              f"+- {sd[1]:.2f}, vdW {a[:, 2].mean():+.2f} +- {sd[2]:.2f} kJ/mol")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "check":
        check()
    elif a[0] == "run":
        run(int(a[1]), *(float(x) for x in a[2:4]))
    elif a[0] == "analyze":
        analyze()
