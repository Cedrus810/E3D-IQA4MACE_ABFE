"""One-step decoupling + water relocation on the acetamide leg, MM only (user, 2026-10-23).

    lambda1 --(wall on)--> lambda1' --(jump: all ligand-env coupling off)--> lambda0' --(wall off)--> lambda0

lambda1 / lambda0 are exactly abfe_mm.py's end states ((lambda_elec, lambda_vdw) = (1,1) / (0,0), the
ligand's own terms kept by exceptions), so the total is checked against its MBAR value +34.04 kJ/mol.
The wall W = kw * sum over (ligand heavy atom, water O) of max(0, r_in - r)^2 keeps the region
R = {r < r_in of any ligand heavy atom} free of water oxygens (lambda' = wall on). For any W:

    dG = -kT ln <exp(-bW)>_lambda1  +  dG_jump(lambda1' -> lambda0')  +  kT ln <exp(-bW)>_lambda0

with a stiff wall the two averages are P(N_R = 0). dG_jump by BAR from lambda1' and lambda0' samples.
<exp(-bW)>_lambda0 is sampled by MD + water relocation MC (ABFE_IBS tools/pocket_water, R / shell S
around the ghost ligand's heavy atoms).

Phases (each resumable: a finished phase's npz is reused):
  A  lambda1, no wall, MD          -> W per frame; lambda1' samples = frames with W = 0 (stiff wall)
  B  lambda0', wall on, MD         -> U(lambda1') - U(lambda0') per frame
  C  lambda0, no wall, MD + MC     -> W and N_R per frame (P(N_R = 0) directly, if reachable)
  D  lambda0, wall radius grown 0 -> r_in in --d-states steps, MD per state, MBAR over the wall
     energies at every radius -> G(lambda0') - G(lambda0) (used when C never sees N_R = 0)
Run: python jump_swap.py PHASE --r-in 0.25 [--ps 1000] ; python jump_swap.py analyze --r-in 0.25
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import openmm
from openmm import unit

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "2026-10-15_acetamide_leg"))
sys.path.insert(0, "/home/ruigengji/ABFE_IBS/ABFE_IBS")
import abfe_mm  # noqa: E402

T = abfe_mm.T
KT = abfe_mm.KT
BETA = 1.0 / KT
KJ = unit.kilojoule_per_mole
G_WALL = 30


def build(r_in, kw):
    d, _, system, lig = abfe_mm.build()
    numbers = np.asarray(d["numbers"])
    heavy = [i for i in lig if numbers[i] > 1]
    res = np.asarray(d["resnames"])
    water_O = [int(i) for i in np.where((res == "HOH") & (numbers == 8))[0]]
    wall = openmm.CustomNonbondedForce("kw*step(rw-r)*(rw-r)^2")
    wall.addGlobalParameter("kw", float(kw))
    wall.addGlobalParameter("rw", float(r_in))
    for _ in range(system.getNumParticles()):
        wall.addParticle([])
    wall.addInteractionGroup(heavy, water_O)
    wall.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    wall.setCutoffDistance(r_in + 0.05)
    wall.setForceGroup(G_WALL)
    system.addForce(wall)
    waters = np.array([[o, o + 1, o + 2] for o in water_O])
    assert np.all(res[waters] == "HOH")
    return d, system, lig, heavy, waters


def make_context(system, d, integ):
    c = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"})
    box = np.asarray(d["box"]) / 10
    c.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i])) for i in range(3)])
    return c


def energy(c, groups=-1):
    return c.getState(getEnergy=True, groups=groups).getPotentialEnergy().value_in_unit(KJ)


def set_lambda(c, lam, kw):
    c.setParameter("lambda_elec", lam)
    c.setParameter("lambda_vdw", lam)
    c.setParameter("kw", kw)


def min_dists(P, heavy, waters, box):
    d = P[waters[:, 0]][:, None, :] - P[heavy][None, :, :]
    d -= box * np.round(d / box)
    return np.sqrt((d ** 2).sum(-1)).min(1)          # per water O: distance to the nearest heavy atom


def run_phase(args):
    kw_on = args.kw
    d, system, lig, heavy, waters = build(args.r_in, kw_on)
    box = np.asarray(d["box"]) / 10
    lam, kw_md = {"A": (1.0, 0.0), "B": (0.0, kw_on), "C": (0.0, 0.0)}[args.phase]
    out = HERE / f"phase{args.phase}_r{args.r_in:g}{args.tag}.npz"
    if out.exists():
        print(f"{out.name} exists; skip", flush=True)
        return
    integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    integ.setRandomNumberSeed(args.seed)
    c = make_context(system, d, integ)
    ev = make_context(system, d, openmm.VerletIntegrator(0.001))
    P0 = np.asarray(d["positions"]) / 10
    c.setPositions(P0)
    set_lambda(c, lam, kw_md)
    set_lambda(ev, lam, kw_md)
    if args.phase == "B":                            # push waters out of R before MD
        sys.path.insert(0, str(HERE.parents[1]))
        from decomp.mlmm import steepest_descent     # OpenMM's minimiser falls back to CPU, which rejects
        print("EM %.1f -> %.1f kJ/mol" % steepest_descent(c), flush=True)   # non-identical exclusions
    c.setVelocitiesToTemperature(T * unit.kelvin, args.seed)
    mc = None
    if args.phase == "C" and args.attempts:
        from tools.pocket_water.relocation import RelocationMC
        from tools.pocket_water.u0 import U0Engine
        nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
        eng = U0Engine.from_openmm(nb, c, waters)
        # U0 only steers proposals; the exact Context energy (delayed acceptance) sets the target
        rng = np.random.default_rng(args.seed)

        def exact(p):
            ev.setPositions(p)
            return energy(ev)

        centers_fn = lambda p: p[heavy]               # noqa: E731  (the ghost ligand's heavy atoms)
        mc = RelocationMC(eng, BETA, args.r_in, args.r_out, P0[heavy], centers_fn, exact, rng)
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                           for i in range(system.getNumParticles())])
    t0 = time.perf_counter()
    integ.step(int(args.equil_ps / 0.002))
    n = int(args.ps / args.every_ps)
    rec = {k: np.zeros(n) for k in ("U1", "U0", "W", "N_R")}
    rec["dmin"] = np.zeros((n, 4))
    for k in range(n):
        integ.step(int(args.every_ps / 0.002))
        st = c.getState(getPositions=True, getVelocities=True)
        P = np.asarray(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        if mc is not None:
            P2, moved = mc.run_phase(P, box, args.attempts, batch=args.batch)
            if moved:
                c.setPositions(P2)
                v = np.asarray(st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond))
                idx = waters[moved].reshape(-1)
                v[idx] = mc.rng.normal(size=(len(idx), 3)) * np.sqrt(0.0083144626 * T / masses[idx])[:, None]
                c.setVelocities(v)
                c.applyVelocityConstraints(1e-6)
                P = P2
        ev.setPositions(P)
        for lam_e, key in ((1.0, "U1"), (0.0, "U0")):
            set_lambda(ev, lam_e, 0.0)
            rec[key][k] = energy(ev)
        set_lambda(ev, lam, kw_on)
        rec["W"][k] = energy(ev, {G_WALL})
        dm = min_dists(P, heavy, waters, box)
        rec["N_R"][k] = int((dm < args.r_in).sum())
        rec["dmin"][k] = np.sort(dm)[:4]
        if (k + 1) % 100 == 0:
            extra = (f"; MC acc in/out {mc.stats['acc_in']}/{mc.stats['acc_out']} of {mc.stats['attempts']}"
                     if mc is not None else "")
            print(f"phase {args.phase} {k + 1}/{n} frames, {time.perf_counter() - t0:.0f} s; "
                  f"N_R mean {rec['N_R'][:k + 1].mean():.3f}, frac W=0 {(rec['W'][:k + 1] == 0).mean():.3f}{extra}",
                  flush=True)
    meta = dict(phase=args.phase, r_in=args.r_in, kw=kw_on, ps=args.ps, equil_ps=args.equil_ps,
                every_ps=args.every_ps, seed=args.seed, attempts=args.attempts if mc else 0,
                mc_stats=({k: v for k, v in mc.stats.items() if k != "s2_log_alpha"} if mc else None),
                V_R=(mc.V_R if mc else None), seconds=time.perf_counter() - t0)
    np.savez(out, meta=json.dumps(meta), **rec)
    print(f"-> {out}", flush=True)


def run_wall_growth(args):
    out = HERE / f"phaseD_r{args.r_in:g}{args.tag}.npz"
    if out.exists():
        print(f"{out.name} exists; skip", flush=True)
        return
    d, system, lig, heavy, waters = build(args.r_in, args.kw)
    radii = np.linspace(0.0, args.r_in, args.d_states)
    integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    integ.setRandomNumberSeed(args.seed)
    c = make_context(system, d, integ)
    ev = make_context(system, d, openmm.VerletIntegrator(0.001))
    set_lambda(c, 0.0, args.kw)
    set_lambda(ev, 0.0, args.kw)
    c.setPositions(np.asarray(d["positions"]) / 10)
    c.setVelocitiesToTemperature(T * unit.kelvin, args.seed)
    n = int(args.ps / args.every_ps)
    u = np.zeros((len(radii), len(radii) * n))          # reduced wall energy u[state l, sample]
    t0 = time.perf_counter()
    for k, rk in enumerate(radii):
        c.setParameter("rw", rk)
        if k:
            sys.path.insert(0, str(HERE.parents[1]))
            from decomp.mlmm import steepest_descent
            steepest_descent(c)
        integ.step(int(args.equil_ps / 0.002))
        for s_ in range(n):
            integ.step(int(args.every_ps / 0.002))
            ev.setPositions(c.getState(getPositions=True).getPositions())
            for l, rl in enumerate(radii):
                ev.setParameter("rw", rl)
                u[l, k * n + s_] = BETA * energy(ev, {G_WALL})
        print(f"phase D state {k} r {rk:.3f} nm done, {time.perf_counter() - t0:.0f} s", flush=True)
    meta = dict(phase="D", r_in=args.r_in, kw=args.kw, radii=radii.tolist(), ps=args.ps,
                equil_ps=args.equil_ps, every_ps=args.every_ps, seed=args.seed,
                seconds=time.perf_counter() - t0)
    np.savez(out, meta=json.dumps(meta), u=u, N=np.full(len(radii), n))
    print(f"-> {out}", flush=True)


def analyze(args):
    from pymbar import other_estimators
    have = [p for p in "ABC" if (HERE / f"phase{p}_r{args.r_in:g}{args.tag}.npz").exists()]
    z = {p: np.load(HERE / f"phase{p}_r{args.r_in:g}{args.tag}.npz") for p in have}
    meta = {p: json.loads(str(z[p]["meta"])) for p in have}
    kw = meta["B"]["kw"]
    # lambda1 term: -kT ln <exp(-bW)>_lambda1, W re-evaluated with the run's kw
    WA = z["A"]["W"]
    WC = z["C"]["W"] if "C" in z else None
    f1 = np.mean(np.exp(-BETA * WA))
    f0 = np.mean(np.exp(-BETA * WC)) if WC is not None else np.nan
    # jump: forward work from lambda1' samples (A frames reweighted by exp(-bW)),
    # reverse from B (wall on both sides, so W cancels in U0' - U1')
    w_A = np.exp(-BETA * WA)
    keep = w_A > 1e-12
    wR = BETA * (z["B"]["U1"] - z["B"]["U0"])
    # resample forward works by their weights (stiff wall: the kept frames have weight ~1)
    rng = np.random.default_rng(0)
    pick = rng.choice(np.where(keep)[0], size=keep.sum(), p=w_A[keep] / w_A[keep].sum())
    wF = BETA * (z["A"]["U0"] - z["A"]["U1"])[pick]
    bar = other_estimators.bar(wF, wR)
    dG_jump, dG_jump_err = KT * bar["Delta_f"], KT * bar["dDelta_f"]
    boot = []
    for _ in range(200):                          # bootstrap over frames (frames 1 ps apart; not decorrelated)
        bF = wF[rng.integers(len(wF), size=len(wF))]
        bR = wR[rng.integers(len(wR), size=len(wR))]
        try:
            boot.append(KT * other_estimators.bar(bF, bR)["Delta_f"])
        except Exception:
            pass
    boot = np.asarray(boot)
    fep_f = -KT * np.log(np.mean(np.exp(-wF)))
    fep_r = KT * np.log(np.mean(np.exp(-wR)))
    term1 = -KT * np.log(f1)
    term3 = KT * np.log(f0) if f0 > 0 else np.nan
    growth = None
    fD = HERE / f"phaseD_r{args.r_in:g}{args.tag}.npz"
    if fD.exists():
        from pymbar import MBAR
        zD = np.load(fD)
        mb = MBAR(zD["u"], zD["N"])
        r = mb.compute_free_energy_differences()
        dG_grow = KT * r["Delta_f"][0, -1]            # G(lambda0') - G(lambda0)
        ov = mb.compute_overlap()["matrix"]
        growth = dict(dG_wall_on=dG_grow, err=KT * r["dDelta_f"][0, -1],
                      min_adjacent_overlap=float(min(ov[i, i + 1] for i in range(len(ov) - 1))),
                      radii=json.loads(str(zD["meta"]))["radii"])
        term3_D = -dG_grow
    total = term1 + dG_jump + term3
    total_D = term1 + dG_jump + term3_D if growth else None
    res = dict(r_in=args.r_in, kw=kw,
               lambda1_wall=dict(term=term1, mean_exp=f1, frac_W0=float((WA == 0).mean()), n=len(WA)),
               jump=dict(BAR=dG_jump, BAR_err=dG_jump_err, BAR_bootstrap_sd=float(boot.std()),
                         BAR_bootstrap_n=len(boot), BAR_bootstrap_range=[float(boot.min()), float(boot.max())], FEP_forward=fep_f, FEP_reverse=fep_r,
                         n_forward=int(keep.sum()), n_reverse=len(wR),
                         work_forward_kT=dict(mean=float(wF.mean()), sd=float(wF.std())),
                         work_reverse_kT=dict(mean=float(wR.mean()), sd=float(wR.std()))),
               lambda0_wall=(dict(term=term3, mean_exp=f0, frac_W0=float((WC == 0).mean()), n=len(WC),
                                  N_R_hist=np.bincount(z["C"]["N_R"].astype(int)).tolist(),
                                  mc=meta["C"]["mc_stats"]) if WC is not None else None),
               wall_growth=growth, total_with_C=total, total_with_D=total_D, reference_MM_MBAR=34.04)
    (HERE / f"result_r{args.r_in:g}{args.tag}.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["A", "B", "C", "D", "analyze"])
    ap.add_argument("--r-in", type=float, default=0.25, help="nm, region R around ligand heavy atoms")
    ap.add_argument("--r-out", type=float, default=0.7, help="nm, outer radius of the MC shell S")
    ap.add_argument("--kw", type=float, default=1e5, help="wall stiffness, kJ/mol/nm^2")
    ap.add_argument("--ps", type=float, default=1000.0)
    ap.add_argument("--equil-ps", type=float, default=100.0)
    ap.add_argument("--every-ps", type=float, default=1.0)
    ap.add_argument("--attempts", type=int, default=2000, help="MC attempts per frame (phase C)")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--d-states", type=int, default=11, help="wall radii in phase D, 0 .. r_in")
    ap.add_argument("--tag", default="", help="suffix of every output file, e.g. _s12")
    a = ap.parse_args()
    if a.phase == "analyze":
        analyze(a)
    elif a.phase == "D":
        run_wall_growth(a)
    else:
        run_phase(a)
