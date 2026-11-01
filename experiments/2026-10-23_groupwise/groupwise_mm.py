"""Group-by-group dummy path, MM only, acetamide leg (DESIGN_2026-10-23_groupwise_dummy.md S4).

The ligand is split into groups (acetamide: CH3 | C(=O)NH2). Groups become dummy one after another;
within a group, charges go first (lambda_elec_g 1 -> 0, 11 states), then LJ with the Beutler softcore
(lambda_vdw_g 1 -> 0, 20 steps), exactly as abfe_mm.py does for the whole ligand. The ligand's own
terms are kept by exceptions, so the end states are abfe_mm.py's and the total must return its MBAR
value +34.04 kJ/mol (path independence). Lambda here only serves MM sampling.

Per state: Langevin 298.15 K, 2 fs, equil_ps then prod_ps sampled every every_ps; at each sample the
reduced energies at the neighbouring states k-1, k, k+1 are stored. Analysis: BAR between neighbours,
summed per group and in total. States run in series, each from the previous state's last frame;
resumable (one npz per state).

Run: python groupwise_mm.py run REP [--order 0,1] [--equil-ps 100 --prod-ps 1000]
     python groupwise_mm.py analyze [--order 0,1]
"""

import argparse
import json
import pathlib
import time

import numpy as np
import openmm
from openmm import unit

HERE = pathlib.Path(__file__).resolve().parent
LEG = "/home/ruigengji/E3D/experiments/2026-10-15_acetamide_leg/em_acetamide.npz"
T = 298.15
KT = 0.0083144626 * T
ALPHA = 0.5
KJ = unit.kilojoule_per_mole


def ligand_groups(d, lig):
    """acetamide: [methyl C + its 3 H] and [carbonyl C, O, N + the 2 amide H], H assigned by distance."""
    P, Z = np.asarray(d["positions"]), np.asarray(d["numbers"])
    heavy = [i for i in lig if Z[i] > 1]
    owner = {i: heavy[int(np.argmin([np.linalg.norm(P[i] - P[h]) for h in heavy]))] for i in lig if Z[i] == 1}
    nH = {h: sum(1 for o in owner.values() if o == h) for h in heavy}
    methyl = next(h for h in heavy if Z[h] == 6 and nH[h] == 3)
    g0 = [methyl] + [i for i, o in owner.items() if o == methyl]
    g1 = [i for i in lig if i not in g0]
    return [sorted(g0), sorted(g1)], ["CH3", "C(=O)NH2"]


def states(order):
    """(lambda_elec_g, lambda_vdw_g) for every group, per state; groups vanish in `order`."""
    n = len(order)
    cur = [[1.0, 1.0] for _ in range(n)]
    out = [tuple(tuple(x) for x in cur)]
    seg = [None]
    for g in order:
        for le in np.round(np.linspace(1, 0, 11)[1:], 4):
            cur[g][0] = float(le)
            out.append(tuple(tuple(x) for x in cur)); seg.append((g, "elec"))
        for lv in np.round(np.linspace(1, 0, 21)[1:], 4):
            cur[g][1] = float(lv)
            out.append(tuple(tuple(x) for x in cur)); seg.append((g, "vdw"))
    return out, seg


def build(groups):
    d = np.load(LEG, allow_pickle=True)
    system = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
    lig = [int(i) for i in np.where(d["resids"] == d["resids"][0])[0]]
    rest = [i for i in range(system.getNumParticles()) if i not in set(lig)]
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    prm = {i: nb.getParticleParameters(i) for i in lig}
    have = {tuple(sorted(nb.getExceptionParameters(k)[:2])) for k in range(nb.getNumExceptions())}
    for a, i in enumerate(lig):
        for j in lig[a + 1:]:
            if (i, j) not in have:
                (qi, si, ei), (qj, sj, ej) = prm[i], prm[j]
                nb.addException(i, j, qi * qj, 0.5 * (si + sj), (ei * ej).sqrt())
    for g, atoms in enumerate(groups):
        nb.addGlobalParameter(f"lambda_elec_{g}", 1.0)
        for i in atoms:
            q, s, e = prm[i]
            nb.setParticleParameters(i, 0.0, s, 0.0)
            nb.addParticleParameterOffset(f"lambda_elec_{g}", i, q, 0.0, 0.0)
        sc = openmm.CustomNonbondedForce(
            f"lv*4*eps*(1/x^2 - 1/x); x = alpha*(1-lv) + (r/sig)^6; lv = lambda_vdw_{g};"
            "sig = 0.5*(s1+s2); eps = sqrt(e1*e2)")
        sc.addGlobalParameter(f"lambda_vdw_{g}", 1.0)
        sc.addGlobalParameter("alpha", ALPHA)
        sc.addPerParticleParameter("s")
        sc.addPerParticleParameter("e")
        for i in range(system.getNumParticles()):
            _, s, e = (prm[i] if i in prm else nb.getParticleParameters(i))
            sc.addParticle([s.value_in_unit(unit.nanometer), e.value_in_unit(KJ)])
        for k in range(nb.getNumExceptions()):
            i, j, *_ = nb.getExceptionParameters(k)
            sc.addExclusion(i, j)
        sc.addInteractionGroup(atoms, rest)
        sc.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        sc.setCutoffDistance(nb.getCutoffDistance())
        sc.setUseSwitchingFunction(False)
        sc.setUseLongRangeCorrection(nb.getUseDispersionCorrection())
        system.addForce(sc)
    return d, system, lig


def set_state(c, st):
    for g, (le, lv) in enumerate(st):
        c.setParameter(f"lambda_elec_{g}", le)
        c.setParameter(f"lambda_vdw_{g}", lv)


def context(system, d, integ):
    c = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"})
    box = np.asarray(d["box"]) / 10
    c.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i])) for i in range(3)])
    return c


def tagname(order):
    return "order" + "".join(str(g) for g in order)


def run(args):
    d0 = np.load(LEG, allow_pickle=True)
    lig0 = [int(i) for i in np.where(d0["resids"] == d0["resids"][0])[0]]
    groups, names = ligand_groups(d0, lig0)
    d, system, lig = build(groups)
    order = [int(x) for x in args.order.split(",")]
    sts, seg = states(order)
    outdir = HERE / f"rep{args.rep}_{tagname(order)}"
    outdir.mkdir(exist_ok=True)
    (outdir / "groups.json").write_text(json.dumps(dict(groups=groups, names=names, order=order,
                                                         states=sts, segments=seg), indent=1))
    integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    integ.setRandomNumberSeed(1000 * args.rep + 7)
    c = context(system, d, integ)
    ev = context(system, d, openmm.VerletIntegrator(0.001))
    c.setPositions(np.asarray(d["positions"]) / 10)
    c.setVelocitiesToTemperature(T * unit.kelvin, 1000 * args.rep + 7)
    t0 = time.perf_counter()
    for k, st in enumerate(sts):
        out = outdir / f"state{k:03d}.npz"
        set_state(c, st)
        if out.exists():
            z = np.load(out)
            c.setPositions(z["last_positions"]); c.setVelocities(z["last_velocities"])
            continue
        integ.step(int(args.equil_ps / 0.002))
        n = int(args.prod_ps / args.every_ps)
        nbr = [j for j in (k - 1, k, k + 1) if 0 <= j < len(sts)]
        u = np.zeros((len(nbr), n))
        for s in range(n):
            integ.step(int(args.every_ps / 0.002))
            ev.setPositions(c.getState(getPositions=True).getPositions())
            for a, j in enumerate(nbr):
                set_state(ev, sts[j])
                u[a, s] = ev.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ) / KT
        st_ = c.getState(getPositions=True, getVelocities=True)
        np.savez(out, u=u, nbr=np.array(nbr),
                 last_positions=st_.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                 last_velocities=st_.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond))
        print(f"state {k}/{len(sts) - 1} {seg[k]} done, {time.perf_counter() - t0:.0f} s", flush=True)
    print("run done", flush=True)


def analyze(args):
    import pymbar
    from pymbar import other_estimators
    order = [int(x) for x in args.order.split(",")]
    res = {}
    for outdir in sorted(HERE.glob(f"rep*_{tagname(order)}")):
        meta = json.loads((outdir / "groups.json").read_text())
        nst = len(meta["states"])
        files = [outdir / f"state{k:03d}.npz" for k in range(nst)]
        if not all(f.exists() for f in files):
            print(f"{outdir.name}: {sum(f.exists() for f in files)}/{nst} states, skipped")
            continue
        z = [np.load(f) for f in files]
        steps, per_group = [], {}
        for k in range(nst - 1):
            a = z[k]; b = z[k + 1]
            ia_self, ia_next = list(a["nbr"]).index(k), list(a["nbr"]).index(k + 1)
            ib_self, ib_prev = list(b["nbr"]).index(k + 1), list(b["nbr"]).index(k)
            ua = a["u"][:, pymbar.timeseries.subsample_correlated_data(a["u"][ia_next] - a["u"][ia_self])]
            ub = b["u"][:, pymbar.timeseries.subsample_correlated_data(b["u"][ib_prev] - b["u"][ib_self])]
            r = other_estimators.bar(ua[ia_next] - ua[ia_self], ub[ib_prev] - ub[ib_self])
            dg, err = KT * r["Delta_f"], KT * r["dDelta_f"]
            steps.append((dg, err))
            g, kind = meta["segments"][k + 1]
            key = f"{meta['names'][g]} {kind}"
            per_group.setdefault(key, [0.0, 0.0])
            per_group[key][0] += dg
            per_group[key][1] += err ** 2
        total = sum(s[0] for s in steps)
        err = np.sqrt(sum(s[1] ** 2 for s in steps))
        res[outdir.name] = dict(total=total, err=err,
                                per_group={k: (v[0], np.sqrt(v[1])) for k, v in per_group.items()})
        print(f"{outdir.name}: total {total:+.2f} +- {err:.2f} kJ/mol (MM ABFE +34.04)")
        for k, v in per_group.items():
            print(f"   {k:16s} {v[0]:+8.2f} +- {np.sqrt(v[1]):.2f}")
    (HERE / f"result_{tagname(order)}.json").write_text(json.dumps(res, indent=2, default=float) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("rep", type=int, nargs="?", default=1)
    ap.add_argument("--order", default="0,1")
    ap.add_argument("--equil-ps", type=float, default=100.0)
    ap.add_argument("--prod-ps", type=float, default=1000.0)
    ap.add_argument("--every-ps", type=float, default=1.0)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else analyze(a)
