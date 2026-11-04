"""T1 / T5 local weaken - rotate - recouple path MC proposals, fixed panel starts, eta = 1
(TORSION_PATH_MC_DECISION_2026-11-02.md; DESIGN §10.2). MM gives every force; the model is queried only at S0 ends.

Run: python torsion_ncmc.py check    [--frames 0-7]                         topology, rotations, S0 equivalence (no model)
     python torsion_ncmc.py diagnose [--frames 0-7] [--per 4] [--tag _v1]   the 8 x 5 x per fixed-start proposals

Path (one proposal): the selected groups go through 25 points (el 1, 0.9 .. 0 at on 1; then on 0.9 .. 0.4, 0.35 .. 0.05, 0
at el 0), other groups stay at el = on = 1 (bridge_mm.build carriers); one Metropolized MM-HMC block (eta_hmc.mm_updates,
L = 20, dt = 2 fs) at every point; the move R at the weakest point; the reverse points back to S0:
    K0, switch, K1, .., K24, R, K24, .., switch, K0          (50 blocks, 48 switches)
    W = sum_switch [V_new(x) - V_old(x)] + [V_weak(Rx) - V_weak(x)]       (block dH are diagnostics, not work)
    log R = -[W + eta (dU(y) - dU(x))] / RT
R: rigid rotation of the moving side about the axis by theta ~ U(-pi, pi); T15 does T1 and T5 in a random order (1/2),
independent angles; reverse = opposite order and angles (same probability), unit Jacobian. path_attempt holds the
acceptance and is the same function the finite-state tests drive (test_torsion_ncmc.py).
Every proposal restarts from the panel frame (fixed-start feasibility diagnostic, not a chain); one model query per
candidate, one per frame for the start (checked against the panel cache).
Records: one npz per (frame, category) under runs/SYSTEM/torsion_ncmc{tag}/, temporary file then rename; a unit with the
same contract is reused (content check, no hash), a different contract stops. An unfinished unit is recomputed from its
seeds: decisions replay, numbers agree to ~1e-3 (CUDA MM integration is not bitwise reproducible across Contexts, also
with DeterministicForces; checked 2026-11-02), so a recomputed unit is a replay to that level, not a bitwise one.
"""

import argparse
import json
import pathlib
import time

import numpy as np

CATEGORIES = {"C2": ([1], []), "C24": ([1, 3], []), "T1": ([1, 3], ["T1"]), "T5": ([1], ["T5"]),
              "T15": ([1, 3], ["T1", "T5"])}                     # 0-based group indices, rotations
TORSIONS = {"T1": (14, 3, 6, 16), "T5": (3, 6, 16, 2)}            # dihedral atoms; axis = middle pair
SIDES = {"T1": [2, 5, 6, 7, 8, 11, 16, 20, 21, 22, 23, 24, 29], "T5": [2, 5, 7, 8, 11, 16, 20, 23, 24, 29]}
ELEC = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
VDW = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
POINTS = [(e, 1.0) for e in ELEC] + [(0.0, v) for v in VDW]      # 25 points, S0 first, weakest last


# ---------------------------------------------------------------- acceptance (pure; shared with the tests)

def path_attempt(x, points, propagate, energy, move, final_dU, dU_x, eta, rt, rng, label=None):
    """One forward / move / reverse path from x. propagate(p, x) -> x keeps exp(-V_p / RT) (its own Metropolis);
    energy(p, x) = V_p(x); move(x) -> (y, spec), a symmetric unit-Jacobian map or (x, None); final_dU(y) = dU at the end.
    Work is protocol work only (switches + move); propagation heat is not work."""
    label = label or (lambda a, b: "switch")
    W = {"rot": 0.0}
    steps = []                                        # per switch / move work in path order (diagnostic)
    n = len(points)
    x = propagate(points[0], x)
    for i in range(1, n):
        k = label(points[i - 1], points[i])
        w = energy(points[i], x) - energy(points[i - 1], x)
        W[k] = W.get(k, 0.0) + w
        steps.append(w)
        x = propagate(points[i], x)
    y, spec = move(x)
    if spec is not None:
        W["rot"] = energy(points[-1], y) - energy(points[-1], x)
    steps.append(W["rot"])
    x = propagate(points[-1], y)
    for i in range(n - 2, -1, -1):
        k = label(points[i + 1], points[i])
        w = energy(points[i], x) - energy(points[i + 1], x)
        W[k] = W.get(k, 0.0) + w
        steps.append(w)
        x = propagate(points[i], x)
    dU_y = final_dU(x)
    total = float(sum(W.values()))
    logR = -(total + eta * (dU_y - dU_x)) / rt
    logu = float(np.log(rng.random()))
    return {"y": x, "spec": spec, "W": W, "W_steps": steps, "W_total": total, "dU_y": float(dU_y), "logR": float(logR),
            "logu": logu, "acc": bool(logu < logR)}


# ---------------------------------------------------------------- geometry (pure)

def rotate(xyz, a, b, side, theta):
    """Rigid rotation of `side` about the a -> b axis by theta (rad); a, b stay put. Returns a new array."""
    out = np.array(xyz, np.float64, copy=True)
    p = out[a]
    k = out[b] - p
    k /= np.linalg.norm(k)
    v = out[side] - p
    c, s = np.cos(theta), np.sin(theta)
    out[side] = p + v * c + np.cross(k, v) * s + k[None] * (v @ k)[:, None] * (1 - c)
    return out


def apply_spec(xyz, spec):
    for name, theta in spec:
        t = TORSIONS[name]
        xyz = rotate(xyz, t[1], t[2], SIDES[name], theta)
    return xyz


def inverse_spec(spec):
    return [(n, -th) for n, th in reversed(spec)]


def draw_spec(names, rng):
    """theta ~ U(-pi, pi) per torsion; with two, the order is T1 first or T5 first with probability 1/2."""
    if not names:
        return None
    th = {n: float(rng.uniform(-np.pi, np.pi)) for n in names}
    order = list(names) if len(names) == 1 or rng.random() < 0.5 else list(reversed(names))
    return [(n, th[n]) for n in order]


def dihedral(x, t):
    p0, p1, p2, p3 = (x[i] for i in t)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - (b0 @ b1) * b1, b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


# ---------------------------------------------------------------- molecular parts

def ligand_topology(name):
    """Ligand bond graph from HarmonicBondForce bonds and constraints, atomic numbers from the model topology;
    checks the contract's moving sides, non-ring axes and no constraint across a cut."""
    import openmm
    import eta_hmc as eh
    from model_eval import topology
    xml, lig, groups, _, _ = eh.gm.leg(name)
    s = openmm.XmlSerializer.deserialize(open(xml).read())
    L = set(lig)
    edges = set()
    for f in s.getForces():
        if isinstance(f, openmm.HarmonicBondForce):
            for k in range(f.getNumBonds()):
                a, b = f.getBondParameters(k)[:2]
                if a in L and b in L:
                    edges.add((min(a, b), max(a, b)))
    cons = []
    for k in range(s.getNumConstraints()):
        a, b = s.getConstraintParameters(k)[:2]
        if a in L or b in L:
            cons.append((min(a, b), max(a, b)))
            edges.add((min(a, b), max(a, b)))
    nb = {i: set() for i in lig}
    for a, b in edges:
        nb[a].add(b), nb[b].add(a)
    for name_, t in TORSIONS.items():
        a, b = t[1], t[2]
        seen, todo = {b}, [b]
        while todo:
            u = todo.pop()
            for v in nb[u]:
                if (u, v) != (b, a) and v not in seen:
                    seen.add(v), todo.append(v)
        if a in seen:
            raise ValueError(f"{name_}: axis {a}-{b} is in a ring")
        if sorted(seen) != SIDES[name_]:
            raise ValueError(f"{name_}: moving side {sorted(seen)} differs from the contract")
        if any((i in seen) != (j in seen) for i, j in cons):
            raise ValueError(f"{name_}: a constraint crosses the cut")
    z = np.asarray(topology(name)[3])[np.asarray(lig)]
    heavy = [i for i in range(len(lig)) if z[i] > 1]
    polar = [i for i in range(len(lig)) if z[i] in (7, 8)]
    if len(heavy) != 18:
        raise ValueError(f"expected 18 heavy atoms, found {len(heavy)}")
    side_groups = {"T1": sorted(groups[1] + groups[3]), "T5": sorted(groups[1])}
    for n, g in side_groups.items():
        if g != SIDES[n]:
            raise ValueError(f"{n}: moving side is not the switched groups")
    return {"lig": list(map(int, lig)), "groups": groups, "edges": sorted(edges), "constraints": cons,
            "heavy": heavy, "polar": polar}


def mm_bridge_system(name):
    """bridge_mm.build (separate el_g / on_g), CM-motion remover removed as in eta_hmc.mm_system."""
    import sys
    import openmm
    import eta_hmc as eh
    sys.path.insert(0, str(eh.E3D / "experiments/2026-10-26_mm_bridge"))
    import bridge_mm
    _, system, groups, _, box = bridge_mm.build(name)
    for i in reversed(range(system.getNumForces())):
        if isinstance(system.getForce(i), openmm.CMMotionRemover):
            system.removeForce(i)
    return system, groups, box


def lam_params(K, sel, el, on):
    out = {}
    for g in range(K):
        out[f"el_{g}"], out[f"on_{g}"] = (el, on) if g in sel else (1.0, 1.0)
    return out


class Molecular:
    """Context-backed propagate / energy for path_attempt; the Context always holds the current x."""

    def __init__(self, c, ig, K, sel, rng):
        self.c, self.ig, self.K, self.sel, self.rng = c, ig, K, sel, rng
        self.cur = None
        self.blocks = []

    def set(self, p):
        if p != self.cur:
            for k, v in lam_params(self.K, self.sel, *p).items():
                self.c.setParameter(k, v)
            self.cur = p

    def energy(self, p, x):
        import eta_hmc as eh
        self.set(p)
        self.c.setPositions(x)
        return self.c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(eh.KJ)

    def propagate(self, p, x):
        import eta_hmc as eh
        self.set(p)
        self.c.setPositions(x)
        y, _, dH, acc = eh.mm_updates(self.c, self.ig, np.array(x, np.float64), 1, self.rng)
        self.blocks.append((float(dH[0]), bool(acc[0])))
        return y


def label(a, b):
    return "elec" if a[1] == b[1] == 1.0 else "lj"


def frames_of(s):
    a, b = (s.split("-") + [None])[:2]
    return list(range(int(a), int(b) + 1)) if b else [int(a)]


def panel_info(i):
    from eta_exchange import load_initial_panel
    return load_initial_panel(PANEL, i)


PANEL = pathlib.Path(__file__).resolve().parent / "logs" / "deep_shallow_panel_20261031.npz"
SYSTEM = "CyclophilinD_ligand-8"


def whole(x, top, tol=0.3):
    """Every ligand bond / constraint shorter than tol nm: the ligand is in one periodic image (no rewrapping)."""
    e = np.asarray(top["edges"])
    d = float(np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1).max())
    return d < tol, d


def panel_model():
    """Head / backbone recorded in the panel contract (the target the start states were checked against)."""
    with np.load(PANEL, allow_pickle=False) as z:
        c = json.loads(str(z["meta"]))["contract"]
    return c["head"], c["backbone"]


def check(a):
    """Topology, rotation inverses and preserved bond / constraint lengths on panel frames, S0 equivalence of the
    split System with eta_hmc.mm_system (energy, force). No model."""
    import openmm
    import eta_hmc as eh
    top = ligand_topology(SYSTEM)
    print(f"topology: 18 heavy, sides as contract, axes not in rings, no constraint across a cut; "
          f"{len(top['edges'])} ligand edges ({len(top['constraints'])} constraints)")
    sys_b, groups, box = mm_bridge_system(SYSTEM)
    sys_r, _, _ = eh.mm_system(SYSTEM)
    cb = eh.make_context(sys_b, box, openmm.VerletIntegrator(0.001))
    cr = eh.make_context(sys_r, box, openmm.VerletIntegrator(0.001))
    for k, v in lam_params(len(groups), [], 1.0, 1.0).items():
        cb.setParameter(k, v)
    eh.gm.set_state(cr, 0, len(groups))
    rng = np.random.default_rng(7)
    E = np.array(top["edges"])
    for i in frames_of(a.frames):
        x, _, _ = panel_info(i)
        ok, dmax = whole(x, top)
        st = []
        for c in (cb, cr):
            c.setPositions(x)
            s = c.getState(getEnergy=True, getForces=True)
            st.append((s.getPotentialEnergy().value_in_unit(eh.KJ), s.getForces(asNumpy=True).value_in_unit(eh.KJ / openmm.unit.nanometer)))
        spec = draw_spec(["T1", "T5"], rng)
        y = apply_spec(x, spec)
        back = apply_spec(y, inverse_spec(spec))
        dl = np.abs(np.linalg.norm(y[E[:, 0]] - y[E[:, 1]], axis=1) - np.linalg.norm(x[E[:, 0]] - x[E[:, 1]], axis=1)).max()
        tors = {n: (dihedral(x, t), dihedral(y, t)) for n, t in TORSIONS.items()}
        print(f"frame {i}: ligand whole {ok} (max bond {dmax:.3f} nm); S0 split - mm_system dE {st[0][0] - st[1][0]:+.2e} kJ/mol, "
              f"max|dF| {np.abs(st[0][1] - st[1][1]).max():.2e}; {spec[0][0]}+{spec[1][0]} rotation: inverse max|dx| "
              f"{np.abs(back - x).max():.1e} nm, max|d bond| {dl:.1e} nm; T1 {tors['T1'][0]:+.1f}->{tors['T1'][1]:+.1f}, "
              f"T5 {tors['T5'][0]:+.1f}->{tors['T5'][1]:+.1f} (expected change {np.degrees(dict(spec)['T1']):+.1f} / "
              f"{np.degrees(dict(spec)['T5']):+.1f})", flush=True)


def diagnose(a):
    import openmm  # noqa: F401
    import eta_hmc as eh
    import local_E
    from eta_exchange import DETAILS, initial_state
    from compare_ladders import kabsch_rmsd
    top = ligand_topology(SYSTEM)
    system, groups, box = mm_bridge_system(SYSTEM)
    K = len(groups)
    leg = local_E.Leg(SYSTEM)
    head, backbone = panel_model()
    head = a.head or head
    me = eh.ModelEnergy(leg, 0, "float64", ckpt=head, backbone=backbone)
    ident = eh.model_identity(ckpt=head, backbone=backbone)
    out_dir = eh.HERE / "runs" / SYSTEM / f"torsion_ncmc{a.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lig, heavy = np.asarray(top["lig"]), top["heavy"]
    starts = {}
    for i in frames_of(a.frames):
        for ci, (cat, (sel, rots)) in enumerate(CATEGORIES.items()):
            if a.cats and cat not in a.cats.split(","):
                continue
            seed = a.seed_base + 1000 * i + 10 * ci
            want = {"system": SYSTEM, "panel": str(PANEL), "frame": i, "category": cat, "groups_switched": sel,
                    "rotations": rots, "torsions": {n: TORSIONS[n] for n in rots}, "sides": {n: SIDES[n] for n in rots},
                    "points": POINTS, "L": 20, "dt_fs": 2.0, "eta": 1.0, "RT": eh.KT, "per": a.per, "seed": seed,
                    "angle": "uniform(-pi, pi) per torsion; T15 order 1/2", "E_construction": "local_E v1",
                    "dtype": "float64", **ident, "mm_system": "bridge_mm.build, CMMotionRemover removed, Precision mixed",
                    "acceptance": "-[W_switch + W_rot + eta (dU_y - dU_x)] / RT; block dH diagnostic only"}
            want = json.loads(json.dumps(want))                        # compare in stored form (tuples -> lists)
            out = out_dir / f"f{i}_{cat}.npz"
            if out.exists():
                old = json.loads(str(np.load(out)["meta"]))
                diff = [k for k in want if old.get(k) != want[k]]
                if diff:
                    raise SystemExit(f"{out.name}: exists with a different contract ({', '.join(diff)}); use another --tag")
                print(f"{out.name}: complete, reused", flush=True)
                continue
            t0 = time.perf_counter()
            calls0 = me.calls
            ig = eh.proposal_integrator(20, 2.0)
            ig.setRandomNumberSeed(seed + 1)
            c = eh.make_context(system, box, ig)
            for k, v in lam_params(K, [], 1.0, 1.0).items():
                c.setParameter(k, v)
            x0, _, info = panel_info(i)
            if i not in starts:                                        # one checked start query per frame
                starts[i] = initial_state(c, x0, me, info)
            st = starts[i]
            ok, dmax = whole(st["x"], top)
            if not ok:
                raise ValueError(f"frame {i}: ligand not whole (max bond {dmax:.3f} nm)")
            rng = np.random.default_rng(seed)
            rows = []
            failures = []
            for n in range(a.per):
                mol = Molecular(c, ig, K, sel, rng)
                drawn = []

                def move(x):
                    sp = draw_spec(rots, rng)
                    drawn.append(sp)
                    return (apply_spec(x, sp), sp) if sp else (x, None)
                try:
                    r = path_attempt(st["x"], POINTS, mol.propagate, mol.energy, move, me, st["e"], 1.0, eh.KT, rng, label)
                    if not (np.isfinite(r["W_total"]) and np.isfinite(r["dU_y"])):
                        raise FloatingPointError("non-finite work / energy")
                except FloatingPointError as err:      # a failure, recorded as such: not a rejection, not a sample
                    failures.append({"attempt": n, "error": str(err), "blocks_done": len(mol.blocks),
                                     "spec": drawn[0] if drawn else None})
                    print(f"  f{i} {cat} #{n}: FAILED ({err}) after {len(mol.blocks)} blocks; spec "
                          f"{[(q, round(np.degrees(t), 1)) for q, t in (drawn[0] or [])] if drawn else 'not drawn'}", flush=True)
                    # the proposal integrator draws v = draw*gaussian + (1-draw)*v: 0*NaN would carry the failed
                    # velocities into every later attempt, so clear them (the next block redraws momenta anyway)
                    c.setVelocities(np.zeros_like(st["x"]))
                    continue
                if mol.cur != POINTS[0]:
                    raise RuntimeError("path did not end at S0")
                det = {k: np.asarray(me.last_components[k]).copy() for k in DETAILS}
                y = r["y"]
                om_x, om_y = set(st["details"]["omega_water_resid"]), set(det["omega_water_resid"])
                rows.append({**r, "details": det, "blocks": mol.blocks,
                             "tors_x": [dihedral(st["x"][lig], t) for t in TORSIONS.values()],
                             "tors_y": [dihedral(y[lig], t) for t in TORSIONS.values()],
                             "rmsd_heavy_A": kabsch_rmsd(y[lig][heavy] * 10, st["x"][lig][heavy] * 10),
                             "omega_kept": len(om_x & om_y) / 50})
                print(f"  f{i} {cat} #{n}: spec {[(s, round(np.degrees(t), 1)) for s, t in (r['spec'] or [])]} "
                      f"W elec {r['W'].get('elec', 0):+.2f} lj {r['W'].get('lj', 0):+.2f} rot {r['W']['rot']:+.2f}; "
                      f"dU {st['e']:+.2f} -> {r['dU_y']:+.2f}; logR {r['logR']:+.2f}; acc {r['acc']}; "
                      f"T1 {rows[-1]['tors_x'][0]:+.0f}->{rows[-1]['tors_y'][0]:+.0f} T5 {rows[-1]['tors_x'][1]:+.0f}->"
                      f"{rows[-1]['tors_y'][1]:+.0f}; rmsd {rows[-1]['rmsd_heavy_A']:.2f} A", flush=True)
            del c, ig
            meta = {**want, "complete": True, "attempts_ok": len(rows), "failures": failures,
                    "note": "arrays hold the finite attempts only; failures (non-finite MM work) are listed, not rejected",
                    "model_calls": me.calls - calls0, "seconds": time.perf_counter() - t0,
                    "start_dU": st["e"], "date": time.strftime("%F %T")}
            data = {"meta": json.dumps(meta), "x_start": st["x"], "dU_start": st["e"],
                    "y": np.array([r["y"] for r in rows]),
                    "W_elec": np.array([r["W"].get("elec", 0.0) for r in rows]),
                    "W_lj": np.array([r["W"].get("lj", 0.0) for r in rows]),
                    "W_rot": np.array([r["W"]["rot"] for r in rows]),
                    "W_steps": np.array([r["W_steps"] for r in rows]), "W_total": np.array([r["W_total"] for r in rows]),
                    "dU_y": np.array([r["dU_y"] for r in rows]), "logR": np.array([r["logR"] for r in rows]),
                    "logu": np.array([r["logu"] for r in rows]), "acc": np.array([r["acc"] for r in rows]),
                    "spec": json.dumps([r["spec"] for r in rows]),
                    "block_dH": np.array([[b[0] for b in r["blocks"]] for r in rows]),
                    "block_acc": np.array([[b[1] for b in r["blocks"]] for r in rows]),
                    "tors_x": np.array([r["tors_x"] for r in rows]), "tors_y": np.array([r["tors_y"] for r in rows]),
                    "rmsd_heavy_A": np.array([r["rmsd_heavy_A"] for r in rows]),
                    "omega_kept": np.array([r["omega_kept"] for r in rows]),
                    **{k: np.array([r["details"][k] for r in rows]) for k in DETAILS}}
            tmp = out.with_suffix(".tmp.npz")
            np.savez(tmp, **data)
            tmp.replace(out)
            print(f"{out.name}: {a.per} attempts, {len(failures)} failed, accepted {int(data['acc'].sum())}, model calls {meta['model_calls']}, "
                  f"{meta['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "diagnose"])
    ap.add_argument("--frames", default="0-7")
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--tag", default="_v1")
    ap.add_argument("--seed-base", type=int, default=700000)
    ap.add_argument("--head", default=None)
    ap.add_argument("--cats", default="", help="comma list of categories to run (default all)")
    a = ap.parse_args()
    {"check": check, "diagnose": diagnose}[a.cmd](a)
