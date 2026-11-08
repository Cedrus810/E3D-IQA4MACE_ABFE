"""Group-by-group dummy MM sampling on a solvent leg (DESIGN_2026-10-23_groupwise_dummy.md §3-§4).

Run: python groupwise_mm.py run SYSTEM REP [equil_ps] [prod_ps]   (states in series; resumable)
     python groupwise_mm.py analyze SYSTEM                        (adjacent BAR on MM energies, refill)

Groups: groups.json (groups.py, RDKit; order methyl -> polar -> heterocycle -> carbocycle -> chain).
States S0 .. SK: S0 all coupled, S_k = first k groups dummy, SK = all dummy (= gas-phase ligand).
Dummy = the group's charges and LJ with the environment (water + ions) off, switch on_g in {0, 1},
no softcore, no windows (user, 2026-10-23). The ligand's own terms do not change: every ligand-ligand
pair without an exception gets one with its full q_i q_j and LJ (as abfe_mm.py).
- Charges: NonbondedForce particle offsets on_g * q_i (direct + reciprocal space).
- LJ: ligand epsilon 0 in the NonbondedForce; one CustomNonbondedForce per group, Beutler softcore
  form (alpha 0.5, as abfe_mm.py) with on_g in {0, 1}:
      on_g * 4 eps (1/x^2 - 1/x),  x = alpha (1 - on_g) + (r/sig)^6
  on_g = 1 is plain LJ; on_g = 0 gives exactly 0 energy and force at any r. The plain form
  on_g * 4 eps ((sig/r)^12 - (sig/r)^6) blows up when water sits on a dummy atom: at r < 3e-4 nm
  0 x overflow gives |F| 3.7e9 / NaN (debug/zero_times_inf.log; both 2026-10-23 5090 runs died so).
  Interaction group (group, environment), same cutoff, LRC on.
Per state: Langevin 300 K (IBS) or 298.15 K (acetamide), 1/ps, 2 fs, fixed input box;
equil_ps, then prod_ps with a frame every 1 ps: full-box positions (float32, nm), and the reduced
energy of the frame at every state (u_kn). States run in series, each starting from the previous
state's last frame, so water fills the dummy volume by MD. One npz per (rep, state) in runs/SYSTEM/.
"""

import json
import pathlib
import sys
import time

import numpy as np
import openmm
from openmm import app, unit

HERE = pathlib.Path(__file__).resolve().parent
E3D = HERE.parents[1]
T = 300.0
R = 0.0083144626
KT = R * T
KJ = unit.kilojoule_per_mole
DT_PS = 0.002
ALPHA = 0.5
REF_IBS = {"PWWP1_ligand-8": (47.59, 0.27), "CyclophilinD_ligand-8": (60.63, 0.86),   # [unverified] IBS, 300 K
           "HSP90_ligand-17": (48.86, 0.45), "HSP90_ligand-19": (35.94, 2.19),
           "HSP90_ligand-27": (56.78, 0.45), "CyclophilinD_ligand-4": (65.24, 0.46)}


def temperature(name):
    """Preparation temperature of the current solvent inputs, in kelvin."""
    return 298.15 if name == "acetamide" else T


def thermal_energy(temperature_K):
    value = float(temperature_K)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("temperature must be finite and positive")
    return R * value


def input_dir(name):
    return (E3D / "experiments/2026-10-15_acetamide_leg" if name == "acetamide"
            else E3D / "data/solvent_legs_ibs" / name / "rep1")


def leg(name):
    r = input_dir(name)
    if name == "acetamide":
        top = app.PDBxFile(str(r / "topology_solvent.cif")).topology
        lig = [a.index for a in top.atoms() if a.residue.index == 0]
        with np.load(r / "em_acetamide.npz", allow_pickle=False) as z:
            pos, box = z["positions"].copy() / 10.0, np.diag(z["box"] / 10.0)
    else:
        lig = json.load(open(r / "ligand_indices_solvent.json"))["ligand_indices"]
        with np.load(r / "pre_equil_final_state.npz", allow_pickle=False) as z:
            pos, box = z["positions_nm"].copy(), z["box_nm"].copy()
    groups = [g["system_index"] for g in json.loads((HERE / "groups.json").read_text())[name]["groups"]]
    assert sorted(a for g in groups for a in g) == sorted(lig)
    return r / "system_solvent.xml", lig, groups, pos, box


def build(name):
    xml, lig, groups, pos, box = leg(name)
    ref = openmm.XmlSerializer.deserialize(open(xml).read())
    system = openmm.XmlSerializer.deserialize(open(xml).read())
    ligset = set(lig)
    env = [i for i in range(system.getNumParticles()) if i not in ligset]
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    prm = {i: nb.getParticleParameters(i) for i in lig}
    have = {tuple(sorted(nb.getExceptionParameters(k)[:2])) for k in range(nb.getNumExceptions())}
    for a, i in enumerate(lig):
        for j in lig[a + 1:]:
            if (min(i, j), max(i, j)) not in have:
                (qi, si, ei), (qj, sj, ej) = prm[i], prm[j]
                nb.addException(i, j, qi * qj, 0.5 * (si + sj), (ei * ej).sqrt())
    excl = [nb.getExceptionParameters(k)[:2] for k in range(nb.getNumExceptions())]
    for g, atoms in enumerate(groups):
        p = f"on_{g}"
        nb.addGlobalParameter(p, 1.0)
        for i in atoms:
            q, s, _ = prm[i]
            nb.setParticleParameters(i, 0.0, s, 0.0)
            nb.addParticleParameterOffset(p, i, q, 0.0, 0.0)
        lj = openmm.CustomNonbondedForce(
            f"{p}*4*eps*(1/x^2 - 1/x); x = alpha*(1-{p}) + (r/sig)^6; sig = 0.5*(s1+s2); eps = sqrt(e1*e2)")
        lj.addGlobalParameter("alpha", ALPHA)
        lj.addGlobalParameter(p, 1.0)
        lj.addPerParticleParameter("s")
        lj.addPerParticleParameter("e")
        for i in range(system.getNumParticles()):
            _, s, e = (prm[i] if i in ligset else nb.getParticleParameters(i))
            lj.addParticle([s.value_in_unit(unit.nanometer), e.value_in_unit(KJ)])
        for i, j in excl:
            lj.addExclusion(i, j)
        lj.addInteractionGroup(atoms, env)
        lj.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        lj.setCutoffDistance(nb.getCutoffDistance())
        lj.setUseSwitchingFunction(False)
        lj.setUseLongRangeCorrection(nb.getUseDispersionCorrection())
        system.addForce(lj)
    return ref, system, groups, pos, box


def context(system, box, integ):
    c = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"))
    c.setPeriodicBoxVectors(*[openmm.Vec3(*row) for row in np.asarray(box)])
    return c


def set_state(c, k, K):
    for g in range(K):
        c.setParameter(f"on_{g}", 0.0 if g < k else 1.0)


def run(name, rep, equil_ps=100.0, prod_ps=500.0):
    T = temperature(name)
    KT = thermal_energy(T)
    ref, system, groups, pos, box = build(name)
    K = len(groups)
    out_dir = HERE / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    e = []
    for sys_ in (ref, system):                                  # S0 must reproduce the reference System
        c = context(sys_, box, openmm.VerletIntegrator(0.001))
        c.setPositions(pos)
        e.append(c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ))
        del c
    print(f"{name}: {K} groups -> {K + 1} states; S0 vs reference E {e[1]:.3f} vs {e[0]:.3f} kJ/mol "
          f"(diff {e[1] - e[0]:+.4f})", flush=True)
    integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1.0 / unit.picosecond, DT_PS * unit.picosecond)
    integ.setRandomNumberSeed(1000 * rep + 7)
    c = context(system, box, integ)
    c.setPositions(pos)
    c.setVelocitiesToTemperature(T * unit.kelvin, 1000 * rep + 7)
    for k in range(K + 1):
        out = out_dir / f"rep{rep}_S{k}.npz"
        set_state(c, k, K)
        if out.exists():                                        # resume from the saved last frame
            z = np.load(out)
            if ("T_K" in z and float(z["T_K"]) != T) or (name == "acetamide" and "T_K" not in z):
                raise ValueError(f"{out}: missing or incompatible temperature; preserve the source run")
            c.setPositions(z["last_positions"])
            c.setVelocities(z["last_velocities"])
            continue
        t = time.perf_counter()
        integ.step(int(round(equil_ps / DT_PS)))
        n = int(prod_ps)
        u = np.zeros((K + 1, n))
        xyz = np.zeros((n, system.getNumParticles(), 3), np.float32)
        for s in range(n):
            integ.step(int(round(1.0 / DT_PS)))
            xyz[s] = c.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            for m in range(K + 1):
                set_state(c, m, K)
                u[m, s] = c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ) / KT
            set_state(c, k, K)
        st = c.getState(getPositions=True, getVelocities=True)
        np.savez(out, u=u, xyz=xyz, box=np.asarray(box, np.float32), state=k, T_K=T,
                 dummy=np.array([g for g in range(K) if g < k]),
                 last_positions=st.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                 last_velocities=st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond))
        dt = time.perf_counter() - t
        print(f"{name} rep {rep} S{k}: {equil_ps:.0f} + {prod_ps:.0f} ps in {dt:.0f} s "
              f"({(equil_ps + prod_ps) / 1000 / (dt / 86400):.0f} ns/day incl. u_kn and frames)", flush=True)
    print(f"{name} rep {rep} done", flush=True)


def refill(z, groups, lig_heavy, water_O, r_cut=0.25):
    """Mean number of water O within r_cut of any heavy atom of each group, over the frames of z."""
    box = np.diag(z["box"])
    xyz = z["xyz"]
    out = []
    for g in groups:
        h = [i for i in g if i in lig_heavy]
        d = xyz[:, h, None, :] - xyz[:, None, water_O, :]
        d -= box * np.round(d / box)
        out.append(float((np.sqrt((d ** 2).sum(-1)).min(1) < r_cut).sum(-1).mean()))
    return out


def analyze(name):
    from pymbar import other_estimators, timeseries
    KT = thermal_energy(temperature(name))
    xml, lig, groups, _, _ = leg(name)
    K = len(groups)
    cif = app.PDBxFile(str(pathlib.Path(xml).parent / "topology_solvent.cif"))
    el = [a.element.symbol if a.element else "" for a in cif.topology.atoms()]
    res = [a.residue.name for a in cif.topology.atoms()]
    lig_heavy = {i for i in lig if el[i] != "H"}
    water_O = [i for i in range(len(el)) if res[i] in ("HOH", "WAT") and el[i] == "O"]
    out_dir = HERE / "runs" / name
    reps = sorted({int(f.name[3:].split("_")[0]) for f in out_dir.glob("rep*_S*.npz")})
    summary = {}
    for rep in reps:
        fs = [out_dir / f"rep{rep}_S{k}.npz" for k in range(K + 1)]
        if not all(f.exists() for f in fs):
            print(f"{name} rep {rep}: incomplete, skipped")
            continue
        zs = [np.load(f) for f in fs]
        if any(("T_K" in z and float(z["T_K"]) != temperature(name)) or
               (name == "acetamide" and "T_K" not in z) for z in zs):
            raise ValueError("endpoint temperature differs from the input protocol")
        steps, tot = [], 0.0
        for k in range(K):
            wF = zs[k]["u"][k + 1] - zs[k]["u"][k]
            wR = zs[k + 1]["u"][k] - zs[k + 1]["u"][k + 1]
            wF = wF[timeseries.subsample_correlated_data(wF)]
            wR = wR[timeseries.subsample_correlated_data(wR)]
            b = other_estimators.bar(wF, wR)
            fF = other_estimators.exp(wF)["Delta_f"]
            fR = -other_estimators.exp(wR)["Delta_f"]
            steps.append({"group": k + 1, "dG": b["Delta_f"] * KT, "err": b["dDelta_f"] * KT,
                          "EXP_fwd": fF * KT, "EXP_rev": fR * KT, "nF": len(wF), "nR": len(wR)})
            tot += b["Delta_f"] * KT
        rf = [refill(z, groups, lig_heavy, water_O) for z in zs]
        summary[rep] = {"total": tot, "steps": steps, "refill_water_O_within_0.25nm": rf}
        ref = REF_IBS.get(name)
        reference = f"; IBS {ref[0]:+.2f} +- {ref[1]:.2f} [unverified]" if ref else ""
        print(f"\n{name} rep {rep}: total {tot:+.2f} kJ/mol (sum of adjacent BAR on MM energies){reference}")
        for s in steps:
            print(f"  G{s['group']}: BAR {s['dG']:+8.2f} +- {s['err']:.2f}   EXP fwd {s['EXP_fwd']:+8.2f}  "
                  f"rev {s['EXP_rev']:+8.2f}   n {s['nF']}/{s['nR']}")
        print("  water O within 0.25 nm of each group's heavy atoms (rows S0..SK):")
        for k, r in enumerate(rf):
            print(f"   S{k}: " + " ".join(f"{x:5.2f}" for x in r))
    json.dump(summary, open(out_dir / "analysis.json", "w"), indent=1)
    if len(summary) > 1:
        t = np.array([v["total"] for v in summary.values()])
        print(f"\n{name}: mean over {len(t)} reps {t.mean():+.2f} +- {t.std(ddof=1):.2f} (SD)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "run":
        run(a[1], int(a[2]), *(float(x) for x in a[3:5]))
    elif a[0] == "analyze":
        analyze(a[1])
