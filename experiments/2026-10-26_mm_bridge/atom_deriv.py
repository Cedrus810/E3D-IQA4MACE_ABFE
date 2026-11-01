"""MM per-atom path contributions b_(k,i) on saved bridge frames (CONTRACT.md §3; DESIGN §7.5).

Run: python atom_deriv.py eval   SYSTEM REP --group K [--frames N] [--check 5] [--out-tag T] [--tag T]
     python atom_deriv.py report SYSTEM REP --group K [--out-tag T] [--tag T]

Per-atom extension of bridge_mm.build for the bridged group g = K-1 (other groups, the ligand's exceptions,
water / ions unchanged):
    el_g -> el_g_a{i}  the NonbondedForce charge offset of atom i (PME direct + reciprocal space, self term,
                       exception corrections and neutralising background as OpenMM applies them to the charge),
    on_g -> on_g_a{i}  one copy of the group's Beutler softcore CustomNonbondedForce per atom, interaction group
                       ([i], environment), same exclusions, cutoff and LRC.
With every el_g_a{i} = el and on_g_a{i} = on this is bridge_mm.build at (el, on).
Derivatives at the window's own parameters, on every saved frame:
    dV/del_i  NonbondedForce energy is exactly quadratic in each charge scale (NonbondedForce has no parameter
              derivatives in OpenMM), so the central difference (V(el_i+h) - V(el_i-h)) / 2h is exact up to
              rounding; NonbondedForce energy only (force group 1), h = 0.5.
    dV/don_i  OpenMM energy parameter derivative of the atom's softcore copy (force group 2), LRC included.
Electrostatic derivatives on the windows with on = 1, LJ derivatives on the windows with el = 0.
Checks on the first --check frames of each window (reported, no pass / fail line): synchronised per-atom System
energy - bridge_mm.build energy; sum_i dV/del_i - central difference of el_g on bridge_mm.build;
sum_i dV/don_i - analytic dV/don_g of bridge_mm.build, and - its central difference (LJ group only, h 1e-3).
One npz per window under runs/SYSTEM/rep{REP}{tag}/atom_deriv_G{K}{out_tag}/, temporary file then rename;
a window file with the same window, parameters, frames and h is reused (content check, no hash).
report: b_(K,i) = int_1^0 <dV/del_i>_(el=s, on=1) ds + int_1^0 <dV/don_i>_(el=0, on=s) ds, trapezoid rule on
the window grid; sum_i b against the adjacent BAR dF^V of analysis.json, r_MM = BAR - sum_i b reported, not
distributed. Standard errors from per-window statistical inefficiency (pymbar), first / second halves:
diagnostics only, not calibrated intervals.
"""

import argparse
import json
import pathlib
import re
import time

import numpy as np
import openmm

import bridge_mm as bm

H_EL = 0.5
H_LJ = 1e-3


def path_windows(cfg, k):
    """Windows of the group-k path S_{k-1} -> S_k: (index in the chain, label, el, on) for group g = k-1."""
    out = []
    for m, w in enumerate(cfg["windows"]):
        if w["endpoint"] == f"S{k - 1}":
            out.append((m, w["label"], 1.0, 1.0))
        elif w["group"] == k:
            out.append((m, w["label"], w["el"], w["on"]))
    return out


def find_lj(system, g):
    for n, f in enumerate(system.getForces()):
        if isinstance(f, openmm.CustomNonbondedForce) and \
                f"on_{g}" in [f.getGlobalParameterName(p) for p in range(f.getNumGlobalParameters())]:
            return n, f
    raise KeyError(f"no softcore LJ force for on_{g}")


def build_ref(name, g):
    """bridge_mm.build; NonbondedForce in force group 1, group g softcore LJ in group 2 with dV/don_g."""
    _, system, groups, _, _ = bm.build(name)
    for f in system.getForces():
        f.setForceGroup(1 if isinstance(f, openmm.NonbondedForce) else 0)
    find_lj(system, g)[1].setForceGroup(2)
    find_lj(system, g)[1].addEnergyParameterDerivative(f"on_{g}")
    return system, groups


def build_atoms(name, g):
    """Per-atom extension for group g (module docstring)."""
    _, system, groups, _, _ = bm.build(name)
    atoms = groups[g]
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    for k in range(nb.getNumParticleParameterOffsets()):
        p, i, q, s, e = nb.getParticleParameterOffset(k)
        if p == f"el_{g}":
            nb.addGlobalParameter(f"el_{g}_a{i}", 1.0)
            nb.setParticleParameterOffset(k, f"el_{g}_a{i}", i, q, s, e)
    n, lj = find_lj(system, g)
    lj = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(lj))   # removeForce frees the original
    env = sorted(lj.getInteractionGroupParameters(0)[1])
    expr = lj.getEnergyFunction()
    system.removeForce(n)
    for f in system.getForces():
        f.setForceGroup(1 if isinstance(f, openmm.NonbondedForce) else 0)
    for i in atoms:
        p = f"on_{g}_a{i}"
        f = openmm.CustomNonbondedForce(re.sub(rf"\bon_{g}\b", p, expr))
        f.addGlobalParameter("alpha", bm.gm.ALPHA)
        f.addGlobalParameter(p, 1.0)
        f.addEnergyParameterDerivative(p)
        f.addPerParticleParameter("s")
        f.addPerParticleParameter("e")
        for j in range(lj.getNumParticles()):
            f.addParticle(lj.getParticleParameters(j))
        for x in range(lj.getNumExclusions()):
            f.addExclusion(*lj.getExclusionParticles(x))
        f.addInteractionGroup([i], env)
        f.setNonbondedMethod(lj.getNonbondedMethod())
        f.setCutoffDistance(lj.getCutoffDistance())
        f.setUseSwitchingFunction(lj.getUseSwitchingFunction())
        f.setUseLongRangeCorrection(lj.getUseLongRangeCorrection())
        f.setForceGroup(2)
        system.addForce(f)
    return system, groups


def context(system, box, precision):
    c = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("CUDA"),
                       {"Precision": precision})
    c.setPeriodicBoxVectors(*[openmm.Vec3(*row) for row in np.asarray(box, np.float64)])
    return c


def set_params(c, p):
    have = c.getParameters()
    for k, v in p.items():
        if k in have:
            c.setParameter(k, v)


def atom_params(p, g, atoms):
    out = dict(p)
    for i in atoms:
        out[f"el_{g}_a{i}"], out[f"on_{g}_a{i}"] = p[f"el_{g}"], p[f"on_{g}"]
    return out


def energy(c, groups=-1):
    return c.getState(getEnergy=True, groups=groups).getPotentialEnergy().value_in_unit(bm.KJ)


def d_el(c, names, s):
    """Central difference of the NonbondedForce energy in each charge scale; exact for a quadratic energy."""
    out = np.empty(len(names))
    for a, p in enumerate(names):
        c.setParameter(p, s + H_EL)
        ep = energy(c, {1})
        c.setParameter(p, s - H_EL)
        em = energy(c, {1})
        c.setParameter(p, s)
        out[a] = (ep - em) / (2 * H_EL)
    return out


def d_on(c, names):
    d = c.getState(getParameterDerivatives=True, groups={2}).getEnergyParameterDerivatives()
    return np.array([d[p] for p in names])


def out_dir(a):
    return bm.run_dir(a) / f"atom_deriv_G{a.group}{a.out_tag}"


def evaluate(a):
    d = bm.run_dir(a)
    cfg = json.load(open(d / "config.json"))
    k, g = a.group, a.group - 1
    sys_a, groups = build_atoms(a.system, g)
    sys_r, _ = build_ref(a.system, g)
    K, atoms = len(groups), groups[g]
    el_names, on_names = [f"el_{g}_a{i}" for i in atoms], [f"on_{g}_a{i}" for i in atoms]
    od = out_dir(a)
    od.mkdir(parents=True, exist_ok=True)
    ca = cr = None
    print(f"{a.system} rep {a.rep}{a.tag} G{k}: atoms {atoms}, precision {a.precision}", flush=True)
    for m, label, el, on in path_windows(cfg, k):
        out = od / f"w{m:03d}_{label}.npz"
        z = np.load(d / f"w{m:03d}_{label}.npz")
        xyz = z["xyz"] if a.frames is None else z["xyz"][:a.frames]
        want = {"system": a.system, "rep": a.rep, "tag": a.tag, "group": k, "window": m, "label": label,
                "el": el, "on": on, "atoms": atoms, "n_frames": len(xyz), "h_el": H_EL, "h_lj": H_LJ,
                "precision": a.precision, "check_frames": min(a.check, len(xyz))}
        if out.exists():
            old = json.loads(str(np.load(out)["meta"]))
            diff = [x for x in want if old.get(x) != want[x]]
            if not diff:
                print(f"{out.name}: done, reused", flush=True)
                continue
            raise SystemExit(f"{out.name}: exists but differs ({', '.join(diff)}); use another --out-tag")
        if ca is None:
            ca, cr = context(sys_a, z["box"], a.precision), context(sys_r, z["box"], a.precision)
        t = time.perf_counter()
        p = bm.params(K, g, el, on)
        set_params(ca, atom_params(p, g, atoms))
        set_params(cr, p)
        n, do_el, do_on = len(xyz), on == 1.0, el == 0.0
        D_el = np.full((n, len(atoms)), np.nan)
        D_on = np.full((n, len(atoms)), np.nan)
        chk = {"dE_sync": [], "sum_el_minus_group_fd": [], "sum_on_minus_group": [], "group_on_minus_fd": []}
        for s, x in enumerate(xyz):
            ca.setPositions(x.astype(np.float64))
            if do_el:
                D_el[s] = d_el(ca, el_names, el)
            if do_on:
                D_on[s] = d_on(ca, on_names)
            if s < a.check:
                cr.setPositions(x.astype(np.float64))
                chk["dE_sync"].append(energy(ca) - energy(cr))
                if do_el:
                    chk["sum_el_minus_group_fd"].append(D_el[s].sum() - d_el(cr, [f"el_{g}"], el)[0])
                if do_on:
                    dg = cr.getState(getParameterDerivatives=True, groups={2}).getEnergyParameterDerivatives()[f"on_{g}"]
                    cr.setParameter(f"on_{g}", on + H_LJ)
                    ep = energy(cr, {2})
                    cr.setParameter(f"on_{g}", on - H_LJ)
                    em = energy(cr, {2})
                    cr.setParameter(f"on_{g}", on)
                    chk["sum_on_minus_group"].append(D_on[s].sum() - dg)
                    chk["group_on_minus_fd"].append(dg - (ep - em) / (2 * H_LJ))
        dt = time.perf_counter() - t
        meta = {**want, "complete": True, "source": f"w{m:03d}_{label}.npz", "seconds": dt,
                "date": time.strftime("%F %T")}
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, meta=json.dumps(meta), d_el=D_el, d_on=D_on,
                 **{f"check_{x}": np.asarray(v) for x, v in chk.items()})
        tmp.replace(out)
        mx = {x: (f"{np.abs(v).max():.2e}" if v else "-") for x, v in chk.items()}
        print(f"{out.name}: {n} frames in {dt:.1f} s;  <sum dV/del> "
              f"{np.nanmean(D_el.sum(1)) if do_el else float('nan'):+9.3f}  <sum dV/don> "
              f"{np.nanmean(D_on.sum(1)) if do_on else float('nan'):+9.3f} kJ/mol;  max|check| {mx}", flush=True)


def mean_se(y):
    """Mean and its standard error from the statistical inefficiency (pymbar) of one time series."""
    from pymbar import timeseries
    g = timeseries.statistical_inefficiency(y) if y.std() > 0 else 1.0
    return y.mean(), y.std(ddof=1) * np.sqrt(g / len(y))


def trapz_w(x):
    """Trapezoid weights for int over the grid x in path order (1 -> 0 gives int_1^0)."""
    dx = np.diff(np.asarray(x, float))
    w = np.zeros(len(x))
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


def report(a):
    d = bm.run_dir(a)
    cfg = json.load(open(d / "config.json"))
    ana = json.load(open(d / "analysis.json"))
    k = a.group
    wins = path_windows(cfg, k)
    zs = [np.load(out_dir(a) / f"w{m:03d}_{label}.npz") for m, label, _, _ in wins]
    atoms = json.loads(str(zs[0]["meta"]))["atoms"]
    seg = {"el": [(w, z) for w, z in zip(wins, zs) if w[3] == 1.0],
           "on": [(w, z) for w, z in zip(wins, zs) if w[2] == 0.0]}
    res = {"system": a.system, "rep": a.rep, "tag": a.tag, "group": k, "atoms": atoms,
           "n_frames": [json.loads(str(z["meta"]))["n_frames"] for z in zs], "units": "kJ/mol",
           "rule": "trapezoid on the window grid, int_1^0"}
    b = np.zeros(len(atoms))
    var = np.zeros(len(atoms))
    halves = np.zeros(2)
    for key, lst in seg.items():
        x = [w[2] if key == "el" else w[3] for w, _ in lst]
        wt = trapz_w(x)
        Y = np.array([z[f"d_{key}"] for _, z in lst])                 # windows x frames x atoms
        mse = np.array([[mean_se(Y[m, :, i]) for i in range(len(atoms))] for m in range(len(lst))])
        b_seg = wt @ mse[:, :, 0]
        se_seg = np.sqrt((wt ** 2) @ mse[:, :, 1] ** 2)
        tot = np.array([mean_se(Y[m].sum(1)) for m in range(len(lst))])
        n2 = Y.shape[1] // 2
        h = [float(wt @ Y[:, :n2].sum(2).mean(1)), float(wt @ Y[:, n2:].sum(2).mean(1))]
        res[key] = {"grid": x, "b_atom": b_seg.tolist(), "se_atom": se_seg.tolist(),
                    "sum": float(b_seg.sum()), "se_sum": float(np.sqrt((wt ** 2) @ tot[:, 1] ** 2)),
                    "halves_sum": h, "window_mean_sum": tot[:, 0].tolist()}
        b += b_seg
        var += se_seg ** 2
        halves += h
    labels = {w[1]: w for w in wins}
    bar = {"el": 0.0, "on": 0.0}
    for p in ana["pairs"]:
        if p.get("group") == k and not p.get("missing"):
            bar["el" if labels[p["to"]][3] == 1.0 else "on"] += p["dF"]
    res["b_atom"], res["se_atom"] = b.tolist(), np.sqrt(var).tolist()
    res["sum_b"], res["halves_sum_b"] = float(b.sum()), halves.tolist()
    res["bar"] = {"el": bar["el"], "on": bar["on"], "total": ana["groups"][str(k)]["dF_V"],
                  "complete": ana["groups"][str(k)]["complete"]}
    res["r_MM"] = res["bar"]["total"] - res["sum_b"]
    res["r_MM_segment"] = {x: bar[x] - res[x]["sum"] for x in ("el", "on")}
    chk = {}
    for x in ("dE_sync", "sum_el_minus_group_fd", "sum_on_minus_group", "group_on_minus_fd"):
        v = np.concatenate([z[f"check_{x}"] for z in zs if z[f"check_{x}"].size])
        chk[x] = {"n": int(v.size), "max_abs": float(np.abs(v).max())}
    res["checks"] = chk
    out = d / f"atom_deriv_G{k}{a.out_tag}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"{a.system} rep {a.rep}{a.tag} G{k}  (kJ/mol; TI trapezoid; SE from statistical inefficiency)")
    print("  atom    b_el      b_on      b_i     +- se")
    for i, a_ in enumerate(atoms):
        print(f"  {a_:4d} {res['el']['b_atom'][i]:+9.3f} {res['on']['b_atom'][i]:+9.3f} {b[i]:+9.3f}  {np.sqrt(var[i]):.3f}")
    for x in ("el", "on"):
        print(f"  {x}: sum_i b {res[x]['sum']:+8.3f} +- {res[x]['se_sum']:.3f} (halves {res[x]['halves_sum'][0]:+.3f} / "
              f"{res[x]['halves_sum'][1]:+.3f});  BAR {bar[x]:+8.3f};  BAR - TI {res['r_MM_segment'][x]:+.3f}")
    print(f"  total: sum_i b {res['sum_b']:+8.3f};  BAR dF^V {res['bar']['total']:+8.3f};  r_MM {res['r_MM']:+.3f}")
    print(f"  checks max|.|: " + ", ".join(f"{x} {v['max_abs']:.2e} (n {v['n']})" for x, v in chk.items()))
    print(f"  -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "report"])
    ap.add_argument("system")
    ap.add_argument("rep", type=int)
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--check", type=int, default=5)
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args()
    {"eval": evaluate, "report": report}[a.cmd](a)
