"""Group-internal MM bridge windows on an IBS solvent leg (DESIGN_2026-10-23_groupwise_dummy.md §3, §4.2, §5, §7.3).

Run: python bridge_mm.py check   SYSTEM [--frames 3]                         endpoint equivalence, charge window
     python bridge_mm.py run     SYSTEM REP --groups 1 [--elec 1,0.5,0] [--vdw 1,0.8,0.6,0.4,0.2,0]
                                 [--equil-ps 100] [--prod-ps 500] [--frame-ps 1] [--tag T]
     python bridge_mm.py analyze SYSTEM REP [--tag T]                        adjacent BAR per window pair

System: groupwise_mm.build, with the charge offsets of group g moved from on_g to a separate el_g:
    el_g  scales the group's charges (NonbondedForce particle offsets: direct + reciprocal space),
    on_g  is the group's Beutler softcore LJ with the environment (alpha 0.5, groupwise_mm), now LJ only.
Ligand-ligand pairs keep their full charges and LJ through exceptions (groupwise_mm.build), so the ligand's
own terms do not depend on el / on. Endpoint S_k: el = on = 0 for the first k groups, 1 for the rest; this is
groupwise_mm.set_state(k). Bridge of group k (S_{k-1} -> S_k, 0-based g = k-1), groups before g dummy,
after g coupled:
    (el_g, on_g) = (1, 1) = S_{k-1} -> (elec[i], 1) ... -> (0, 1) -> (0, vdw[i]) ... -> (0, 0) = S_k
The windows of consecutive groups form one sequence; the shared endpoint is one window.

Per window (§5): Langevin 300 K, 1/ps, 2 fs, fixed box; equil_ps, then prod_ps with a frame every frame_ps:
full-box positions (float32, nm), time, and the reduced energies of the frame at the previous, own and next
window (u, rows prev/self/next; NaN where there is none). Windows run in series, each from the previous
window's last state. One npz per window, written to a temporary file and renamed when complete; it carries the
metadata, last positions / velocities and an OpenMM checkpoint. Resume (same command): a finished window with
the same state, neighbours, temperature, step and lengths is reused (checkpoint, or positions / velocities if
the checkpoint does not load on this platform, recorded as a new continuation segment). If only its neighbours
differ (chain extended, e.g. --groups 1 then 1,2,3,4), its u rows are recomputed from its saved frames and the
samples kept; any other difference stops with the differences listed. Resume granularity is one window. No file hash is used.
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
E3D = HERE.parents[1]
sys.path.insert(0, str(E3D / "experiments/2026-10-23_groupwise_dummy"))
import groupwise_mm as gm  # noqa: E402

KJ = unit.kilojoule_per_mole
T, KT, DT_PS = gm.T, gm.KT, gm.DT_PS


def build(name):
    """groupwise_mm.build with the group charge offsets moved from on_g to el_g."""
    ref, system, groups, pos, box = gm.build(name)
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    for g in range(len(groups)):
        nb.addGlobalParameter(f"el_{g}", 1.0)
    for k in range(nb.getNumParticleParameterOffsets()):
        p, i, q, s, e = nb.getParticleParameterOffset(k)
        assert p.startswith("on_")
        nb.setParticleParameterOffset(k, "el_" + p[3:], i, q, s, e)
    return ref, system, groups, pos, box


def params(K, g, el, on):
    """Context parameters: groups < g dummy, g at (el, on), groups > g coupled."""
    out = {}
    for h in range(K):
        e, o = (0.0, 0.0) if h < g else (1.0, 1.0) if h > g else (el, on)
        out[f"el_{h}"], out[f"on_{h}"] = e, o
    return out


def endpoint(K, k):
    return params(K, min(k, K - 1), *((1.0, 1.0) if k < K else (0.0, 0.0)))


def windows(K, ks, elec, vdw):
    """Window sequence for groups ks (1-based, consecutive). Each: dict(label, group, el, on, endpoint, params)."""
    assert elec[0] == 1 and elec[-1] == 0 and vdw[0] == 1 and vdw[-1] == 0
    seq = []
    for k in ks:
        g = k - 1
        path = [(e, 1.0) for e in elec] + [(0.0, v) for v in vdw[1:]]
        for i, (e, o) in enumerate(path):
            if seq and i == 0:                                   # S_{k-1}: already the last window of group k-1
                continue
            ep = f"S{k - 1}" if i == 0 else f"S{k}" if i == len(path) - 1 else None
            seq.append({"label": ep or f"G{k}_el{e:.3f}_on{o:.3f}", "group": k, "el": e, "on": o, "endpoint": ep,
                        "params": params(K, g, e, o)})
    return seq


def set_params(c, p):
    for k, v in p.items():
        c.setParameter(k, v)


def energy(c):
    return c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)


def context(system, box, integ):
    return gm.context(system, box, integ)


# ---------------------------------------------------------------- check

def check(a):
    """(1) endpoints S0..SK of the split System vs groupwise_mm.set_state(k): energy and force on saved frames;
    (2) a charge-only window el_g = 0.5 vs the same System with the group's charges set to 0.5 q directly
    (particle charges, not offsets) on the same frames."""
    ref, sys0, groups, _, box = gm.build(a.system)
    _, sys1, _, _, _ = build(a.system)
    K = len(groups)
    c0 = context(sys0, box, openmm.VerletIntegrator(0.001))
    c1 = context(sys1, box, openmm.VerletIntegrator(0.001))

    def ef(c):
        s = c.getState(getEnergy=True, getForces=True)
        return s.getPotentialEnergy().value_in_unit(KJ), s.getForces(asNumpy=True).value_in_unit(KJ / unit.nanometer)

    for k in range(K + 1):
        xyz = np.load(gm.HERE / "runs" / a.system / f"rep1_S{k}.npz")["xyz"][:a.frames].astype(np.float64)
        gm.set_state(c0, k, K)
        set_params(c1, endpoint(K, k))
        dE, dF = [], []
        for x in xyz:
            c0.setPositions(x)
            c1.setPositions(x)
            (e0, f0), (e1, f1) = ef(c0), ef(c1)
            dE.append(e1 - e0)
            dF.append(np.abs(f1 - f0).max())
        print(f"S{k} (rep1_S{k} frames x{len(xyz)}): split - groupwise_mm  max|dE| {np.abs(dE).max():.2e} kJ/mol, "
              f"max|dF| {max(dF):.2e} kJ/mol/nm", flush=True)
    # charge-only window, independent construction: plain particle charges 0.5 q for group g
    for g in range(K):
        sys2 = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(sys0))
        nb = [f for f in sys2.getForces() if isinstance(f, openmm.NonbondedForce)][0]
        for k in range(nb.getNumParticleParameterOffsets()):
            p, i, q, s, e = nb.getParticleParameterOffset(k)
            if p == f"on_{g}":
                q0, s0, e0 = nb.getParticleParameters(i)
                nb.setParticleParameters(i, q0 + 0.5 * q * unit.elementary_charge, s0, e0)
                nb.setParticleParameterOffset(k, p, i, 0.0, s, e)
        c2 = context(sys2, box, openmm.VerletIntegrator(0.001))
        xyz = np.load(gm.HERE / "runs" / a.system / f"rep1_S{g}.npz")["xyz"][:a.frames].astype(np.float64)
        p1 = params(K, g, 0.5, 1.0)
        p2 = {k_: v for k_, v in params(K, g, 1.0, 1.0).items() if k_.startswith("on_")}
        set_params(c1, p1)
        set_params(c2, p2)
        dE, dF, span = [], [], []
        for x in xyz:
            c1.setPositions(x)
            c2.setPositions(x)
            (e1, f1), (e2, f2) = ef(c1), ef(c2)
            dE.append(e1 - e2)
            dF.append(np.abs(f1 - f2).max())
            set_params(c1, params(K, g, 1.0, 1.0))
            ea = energy(c1)
            set_params(c1, params(K, g, 0.0, 1.0))
            eb = energy(c1)
            set_params(c1, p1)
            span.append((ea, e1, eb))
        print(f"G{g + 1} el=0.5 (rep1_S{g} frames): offsets vs direct 0.5 q  max|dE| {np.abs(dE).max():.2e} kJ/mol, "
              f"max|dF| {max(dF):.2e} kJ/mol/nm;  E(el 1, 0.5, 0) - E(el 1), frame 0: "
              f"{0.0:+.2f} {span[0][1] - span[0][0]:+.2f} {span[0][2] - span[0][0]:+.2f} kJ/mol", flush=True)
        del c2


# ---------------------------------------------------------------- run

def renew_neighbours(c, out, z, old, want, nb_p):
    """A finished window whose neighbours changed (chain extended / windows added): recompute all three u rows
    from its saved frames (float32 positions, so prev / self / next stay consistent within the window),
    keep its samples. Prints the change of the self row against the energies computed during the run."""
    t = time.perf_counter()
    u = np.full_like(z["u"], np.nan)
    for s, x in enumerate(z["xyz"]):
        c.setPositions(x.astype(np.float64))
        for r in range(3):
            if nb_p[r] is not None:
                set_params(c, nb_p[r])
                u[r, s] = energy(c) / KT
    d_self = np.abs(u[1] - z["u"][1]).max() * KT
    meta = {**old, "neighbour_params": want["neighbour_params"],
            "u_recomputed": {"from": "saved xyz (float32)", "previous_neighbour_params": old["neighbour_params"],
                             "self_row_max_abs_change_kJmol": float(d_self), "date": time.strftime("%F %T")}}
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: z[k] for k in z.files if k not in ("meta", "u")}, meta=json.dumps(meta), u=u)
    tmp.replace(out)
    print(f"{out.name}: neighbours changed; u recomputed from {len(u[0])} saved frames in "
          f"{time.perf_counter() - t:.1f} s (self row max |change| {d_self:.2e} kJ/mol)", flush=True)


def run_dir(a):
    return HERE / "runs" / a.system / f"rep{a.rep}{a.tag}"


def run(a):
    ref, system, groups, pos, box = build(a.system)
    K = len(groups)
    ks = [int(x) for x in a.groups.split(",")]
    assert ks == list(range(ks[0], ks[-1] + 1)) and 1 <= ks[0] and ks[-1] <= K
    elec = [float(x) for x in a.elec.split(",")]
    vdw = [float(x) for x in a.vdw.split(",")]
    seq = windows(K, ks, elec, vdw)
    d = run_dir(a)
    d.mkdir(parents=True, exist_ok=True)
    seed = 1000 * a.rep + 17
    cfg = {"system": a.system, "rep": a.rep, "groups": ks, "elec": elec, "vdw": vdw, "T_K": T, "dt_ps": DT_PS,
           "friction_per_ps": 1.0, "equil_ps": a.equil_ps, "prod_ps": a.prod_ps, "frame_ps": a.frame_ps,
           "seed": seed, "system_xml": str(gm.leg(a.system)[0]), "groups_json": str(gm.HERE / "groups.json"),
           "alpha": gm.ALPHA, "windows": [{k: w[k] for k in ("label", "group", "el", "on", "endpoint")} for w in seq]}
    json.dump(cfg, open(d / "config.json", "w"), indent=1)
    integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1.0 / unit.picosecond, DT_PS * unit.picosecond)
    integ.setRandomNumberSeed(seed)
    c = context(system, box, integ)
    if ks[0] == 1:                                               # S0 from the leg's pre-equilibrated state
        start = {"source": "pre_equil_final_state.npz"}
        c.setPositions(pos)
        c.setVelocitiesToTemperature(T * unit.kelvin, seed)
    else:                                                        # S_{k-1} from the groupwise_mm endpoint run
        z = np.load(gm.HERE / "runs" / a.system / f"rep{a.rep}_S{ks[0] - 1}.npz")
        start = {"source": f"groupwise_mm rep{a.rep}_S{ks[0] - 1}.npz last frame"}
        c.setPositions(z["last_positions"])
        c.setVelocities(z["last_velocities"])
    n_eq, n_fr = int(round(a.equil_ps / DT_PS)), int(round(a.frame_ps / DT_PS))
    n_frames = int(round(a.prod_ps / a.frame_ps))
    prev_file = None
    for m, w in enumerate(seq):
        nb_p = [seq[m - 1]["params"] if m else None, w["params"], seq[m + 1]["params"] if m + 1 < len(seq) else None]
        want = {"system": a.system, "rep": a.rep, "window": m, "label": w["label"], "group": w["group"],
                "el": w["el"], "on": w["on"], "endpoint": w["endpoint"], "params": w["params"],
                "neighbour_params": nb_p, "T_K": T, "dt_ps": DT_PS, "equil_ps": a.equil_ps, "prod_ps": a.prod_ps,
                "frame_ps": a.frame_ps}
        out = d / f"w{m:03d}_{w['label']}.npz"
        if out.exists():
            z = np.load(out)
            old = json.loads(str(z["meta"]))
            diff = [k for k in want if old.get(k) != want[k]]
            if diff == ["neighbour_params"] and old.get("complete"):
                renew_neighbours(c, out, z, old, want, nb_p)       # the chain was extended: same samples
                z = np.load(out)
            elif diff or not old.get("complete"):
                sys.exit(f"{out.name}: exists but differs ({', '.join(diff) or 'not complete'}); "
                         f"use another --tag or move the file")
            try:
                c.loadCheckpoint(z["checkpoint"].tobytes())
                how = "checkpoint"
            except Exception as err:                             # other platform / GPU: new continuation segment
                c.setPositions(z["last_positions"])
                c.setVelocities(z["last_velocities"])
                how = f"positions/velocities (checkpoint not loadable: {type(err).__name__})"
            print(f"{out.name}: finished, reused; continuing from its {how}", flush=True)
            prev_file = out.name
            continue
        t = time.perf_counter()
        set_params(c, w["params"])
        integ.step(n_eq)
        t_eq = time.perf_counter() - t
        u = np.full((3, n_frames), np.nan)
        xyz = np.zeros((n_frames, system.getNumParticles(), 3), np.float32)
        times = np.zeros(n_frames)
        t_e = 0.0
        for s in range(n_frames):
            integ.step(n_fr)
            st = c.getState(getPositions=True, getEnergy=True)
            xyz[s] = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            times[s] = st.getTime().value_in_unit(unit.picosecond)
            u[1, s] = st.getPotentialEnergy().value_in_unit(KJ) / KT
            te = time.perf_counter()
            for r in (0, 2):
                if nb_p[r] is not None:
                    set_params(c, nb_p[r])
                    u[r, s] = energy(c) / KT
            set_params(c, w["params"])
            t_e += time.perf_counter() - te
        st = c.getState(getPositions=True, getVelocities=True)
        dt = time.perf_counter() - t
        meta = {**want, "complete": True, "start": start if m == 0 else {"source": f"{prev_file} last state"},
                "seed": seed, "u_rows": ["prev", "self", "next"], "u_units": "kT", "kT_kJmol": KT,
                "seconds": dt, "seconds_equil": t_eq, "seconds_neighbour_energies": t_e}
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, meta=json.dumps(meta), u=u, xyz=xyz, times_ps=times, box=np.asarray(box, np.float32),
                 last_positions=st.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                 last_velocities=st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond),
                 checkpoint=np.frombuffer(c.createCheckpoint(), np.uint8))
        tmp.replace(out)
        prev_file = out.name
        ns_day = (a.equil_ps + a.prod_ps) / 1000 / ((dt - t_e) / 86400)
        print(f"{out.name}: {a.equil_ps:g} + {a.prod_ps:g} ps, {n_frames} frames in {dt:.1f} s "
              f"(MD {ns_day:.0f} ns/day; neighbour energies {t_e:.1f} s)", flush=True)
    print(f"{a.system} rep {a.rep}{a.tag}: groups {ks}, {len(seq)} windows done", flush=True)


# ---------------------------------------------------------------- analyze

def bar_pair(wF, wR):
    from pymbar import other_estimators, timeseries
    iF, iR = timeseries.subsample_correlated_data(wF), timeseries.subsample_correlated_data(wR)
    b = other_estimators.bar(wF[iF], wR[iR])
    return float(b["Delta_f"]), float(b["dDelta_f"]), len(iF), len(iR)


def analyze(a):
    d = run_dir(a)
    cfg = json.load(open(d / "config.json"))
    files = [d / f"w{m:03d}_{w['label']}.npz" for m, w in enumerate(cfg["windows"])]
    have = [f.exists() for f in files]
    print(f"{a.system} rep {a.rep}{a.tag}: {sum(have)}/{len(files)} windows present (kJ/mol, kT {KT:.4f})")
    zs = [np.load(f) if h else None for f, h in zip(files, have)]
    out, per_group = [], {}
    print("  pair                                   BAR      +- dBAR   halves (1st / 2nd)      sd(wF) sd(wR) kT   nF/nR")
    for m in range(len(files) - 1):
        w0, w1 = cfg["windows"][m], cfg["windows"][m + 1]
        if zs[m] is None or zs[m + 1] is None:
            print(f"  {w0['label']:>18s} -> {w1['label']:<18s} missing")
            out.append({"from": w0["label"], "to": w1["label"], "missing": True})
            continue
        wF = zs[m]["u"][2] - zs[m]["u"][1]
        wR = zs[m + 1]["u"][0] - zs[m + 1]["u"][1]
        f, df, nF, nR = bar_pair(wF, wR)
        h = [bar_pair(wF[: len(wF) // 2], wR[: len(wR) // 2])[0], bar_pair(wF[len(wF) // 2:], wR[len(wR) // 2:])[0]]
        out.append({"from": w0["label"], "to": w1["label"], "group": w1["group"], "dF": f * KT, "ddF": df * KT,
                    "halves": [x * KT for x in h], "sd_wF_kT": float(wF.std()), "sd_wR_kT": float(wR.std()),
                    "nF": nF, "nR": nR})
        per_group.setdefault(w1["group"], []).append(f * KT)
        print(f"  {w0['label']:>18s} -> {w1['label']:<18s} {f * KT:+8.2f} +- {df * KT:5.2f}   "
              f"{h[0] * KT:+7.2f} / {h[1] * KT:+7.2f}   {wF.std():6.2f} {wR.std():6.2f}   {nF}/{nR}")
    groups = {}
    for g in cfg["groups"]:                       # pair m -> m+1 belongs to the group of window m+1
        v = per_group.get(g, [])
        expected = sum(1 for w in cfg["windows"][1:] if w["group"] == g)
        complete = len(v) == expected
        groups[g] = {"dF_V": float(np.sum(v)), "complete": complete, "pairs": f"{len(v)}/{expected}"}
        print(f"  G{g}: dF^V (S{g - 1} -> S{g}) {np.sum(v):+.2f} kJ/mol" + ("" if complete else
              f"  INCOMPLETE: sum over {len(v)}/{expected} pairs, not dF^V"))
    json.dump({"config": cfg, "pairs": out, "groups": groups},
              open(d / "analysis.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run", "analyze"])
    ap.add_argument("system")
    ap.add_argument("rep", type=int, nargs="?", default=1)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--groups", default="1")
    ap.add_argument("--elec", default="1,0.5,0")
    ap.add_argument("--vdw", default="1,0.8,0.6,0.4,0.2,0")
    ap.add_argument("--equil-ps", type=float, default=100.0)
    ap.add_argument("--prod-ps", type=float, default=500.0)
    ap.add_argument("--frame-ps", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    {"check": check, "run": run, "analyze": analyze}[a.cmd](a)
