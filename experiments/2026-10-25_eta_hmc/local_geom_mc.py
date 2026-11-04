"""Local geometric MC on the original S0 target, stepwise two-stage acceptance (LOCAL_GEOMETRIC_MC_REVIEW_2026-11-02.md).

Run: python local_geom_mc.py check                               constraint Gram blocks / inverses on the real frames (CPU)
     python local_geom_mc.py diagnose [--frames 0-7] [--steps 50] [--tag _localgeom_v1]

Target pi(x) ~ exp(-(V + eta dU)/RT) at S0, eta = 1; MM gives V, the frozen model gives dU (energy only, no force).
One step from the cached state (x, V(x), dU(x), components):
  stage 1 (MM only, reversible for exp(-V/RT)): k = 1, or Multiple-Try Metropolis with w = exp(-V/RT):
      k candidates y_i = apply(x, m_i), m_i ~ draw (symmetric); y = y_j chosen with probability w(y_j) / sum w(y_i);
      k - 1 reference points x*_i = apply(y, m*_i) plus x itself;
      log a1 = logsumexp(-(V(y_i) - V(x))/RT) - logsumexp(-(V(x*_i) - V(x))/RT)   (common zero V(x); x* includes x)
      k = 1 is exactly the single-candidate step: log a1 = -(V(y) - V(x))/RT.
      admissible(x, y) False (reverse selection impossible, e.g. the chosen water left Omega) rejects before any energy.
  stage 2 (only after stage 1 passes; one model query): log a2 = -eta (dU(y) - dU(x)) / RT.
  Any rejection keeps x with its own cached V, dU and components (never the last queried candidate's) and is a step.
Moves (fixed per branch, chosen independently of the configuration):
  torsion: T1 or T5 with 1/2 each, rigid rotation of the moving side about the current axis, theta ~ U(-20, 20) deg;
           branch 0: k = 8 (MTM), branch 1: k = 1;
  water (branch 2, k = 1): one water chosen uniformly from the current Omega (50), translation (each component
           U(-0.01, 0.01) nm) or rotation about its O (axis uniform on the sphere, phi ~ U(-15, 15) deg) with 1/2 each;
           rejected (no MM, no model) unless the water is still in Omega(y).
Non-finite V or dU is a failure (recorded, the chain stops), not a rejection.
"""

import argparse
import json
import pathlib
import time

import numpy as np

THETA_MAX = np.radians(20.0)
K_MTM = 8
D_TRANS = 0.01                                        # nm, per Cartesian component
PHI_MAX = np.radians(15.0)
BRANCHES = {0: "torsion_mtm", 1: "torsion_single", 2: "water"}
SEED_BASE = 900000


def logsumexp(a):
    a = np.asarray(a, float)
    m = a.max()
    return float(m + np.log(np.exp(a - m).sum()))


# ---------------------------------------------------------------- the step (pure; shared with the tests)

def two_stage_step(state, k, draw, apply, V, dU, eta, rt, rng, admissible=None):
    """state = {"x", "V", "dU", "details"}; V(x) -> float; dU(y) -> (float, details); draw(rng) -> move;
    apply(x, move) -> y (symmetric: the inverse move has the same density). Returns (new_state, record)."""
    x, Vx = state["x"], state["V"]
    rec = {"k": k, "n_V": 0, "n_model": 0, "stage1": False, "stage2": None, "acc": False,
           "log_a1": None, "log_a2": None, "log_u1": None, "log_u2": None, "move": None, "admissible": True}
    moves = [draw(rng) for _ in range(k)]
    ys = [apply(x, m) for m in moves]
    if admissible is not None and k == 1 and not admissible(x, ys[0]):
        rec.update({"admissible": False, "move": moves[0]})
        return state, rec
    vy = np.array([V(y) for y in ys])
    rec["n_V"] += k
    if not np.isfinite(vy).all():
        raise FloatingPointError("non-finite MM energy for a candidate")
    lw_y = -(vy - Vx) / rt
    if k == 1:
        j = 0
    else:
        p = np.exp(lw_y - logsumexp(lw_y))
        j = int(rng.choice(k, p=p / p.sum()))
    y, Vy = ys[j], float(vy[j])
    lw_x = [0.0]                                           # x itself, at the common zero
    if k > 1:
        xs = [apply(y, draw(rng)) for _ in range(k - 1)]
        vx = np.array([V(z) for z in xs])
        rec["n_V"] += k - 1
        if not np.isfinite(vx).all():
            raise FloatingPointError("non-finite MM energy for a reference point")
        lw_x = list(-(vx - Vx) / rt) + lw_x
    log_a1 = logsumexp(lw_y) - logsumexp(lw_x)
    log_u1 = float(np.log(rng.random()))
    rec.update({"move": moves[j], "y": y, "log_a1": float(log_a1), "log_u1": log_u1, "V_y": Vy,
                "lw_y": lw_y.tolist(), "lw_x": [float(v) for v in lw_x]})
    if not log_u1 < log_a1:
        return state, rec
    rec["stage1"] = True
    dUy, det = dU(y)
    rec["n_model"] += 1
    if not np.isfinite(dUy):
        raise FloatingPointError("non-finite model energy for the candidate")
    log_a2 = -eta * (dUy - state["dU"]) / rt
    log_u2 = float(np.log(rng.random()))
    rec.update({"dU_y": float(dUy), "log_a2": float(log_a2), "log_u2": log_u2, "stage2": bool(log_u2 < log_a2)})
    if not log_u2 < log_a2:
        return state, rec
    rec["acc"] = True
    return {"x": y, "V": Vy, "dU": float(dUy), "details": det}, rec


# ---------------------------------------------------------------- geometry (pure)

def rotate_about(xyz, center, axis, angle, atoms):
    """Rigid rotation of `atoms` about the line through `center` along unit `axis`."""
    out = np.array(xyz, np.float64, copy=True)
    v = out[atoms] - center
    c, s = np.cos(angle), np.sin(angle)
    out[atoms] = center + v * c + np.cross(axis, v) * s + axis[None] * (v @ axis)[:, None] * (1 - c)
    return out


def torsion_move(name):
    from torsion_ncmc import SIDES, TORSIONS
    a, b = TORSIONS[name][1], TORSIONS[name][2]
    side = SIDES[name]

    def apply(x, theta):
        axis = x[b] - x[a]
        return rotate_about(x, x[a], axis / np.linalg.norm(axis), theta, side)
    return (lambda rng: float(rng.uniform(-THETA_MAX, THETA_MAX))), apply


def water_move(atoms, kind):
    """atoms = (O, H, H) system indices; kind 'trans' or 'rot' (about O)."""
    atoms = list(atoms)
    if kind == "trans":
        def draw(rng):
            return rng.uniform(-D_TRANS, D_TRANS, 3)

        def apply(x, d):
            out = np.array(x, np.float64, copy=True)
            out[atoms] += d
            return out
    else:
        def draw(rng):
            ax = rng.normal(size=3)
            return (ax / np.linalg.norm(ax), float(rng.uniform(-PHI_MAX, PHI_MAX)))

        def apply(x, m):
            return rotate_about(x, np.array(x[atoms[0]]), m[0], m[1], atoms)
    return draw, apply


def gram(xyz, cons, masses=None):
    """Constraint Gram matrix grad g M^-1 grad g^T for g = |r_i - r_j|^2 (unweighted if masses is None)."""
    n = len(cons)
    G = np.zeros((n, n))
    grads = []
    for i, j in cons:
        d = 2 * (xyz[i] - xyz[j])
        grads.append({i: d, j: -d})
    for p in range(n):
        for q in range(p, n):
            s = sum(grads[p][a] @ grads[q][a] / (1.0 if masses is None else masses[a])
                    for a in grads[p] if a in grads[q])
            G[p, q] = G[q, p] = s
    return G


# ---------------------------------------------------------------- molecular parts

SYSTEM = "CyclophilinD_ligand-8"
PANEL = pathlib.Path(__file__).resolve().parent / "logs" / "deep_shallow_panel_20261031.npz"


def frames_of(s):
    a, b = (s.split("-") + [None])[:2]
    return list(range(int(a), int(b) + 1)) if b else [int(a)]


def constraints_and_masses():
    import openmm
    import eta_hmc as eh
    system, _, _ = eh.mm_system(SYSTEM)
    cons = [system.getConstraintParameters(k) for k in range(system.getNumConstraints())]
    masses = np.array([system.getParticleMass(i).value_in_unit(openmm.unit.dalton) for i in range(system.getNumParticles())])
    pairs = [(int(a), int(b)) for a, b, _ in cons]
    dist = np.array([d.value_in_unit(openmm.unit.nanometer) for _, _, d in cons])
    return system, pairs, dist, masses


def check(a):
    """Real frames, CPU: constraint lengths, mass-weighted and unweighted Gram blocks of the affected constraints and
    exact inverses for random T1 / T5 rotations and water moves."""
    from eta_exchange import load_initial_panel
    from torsion_ncmc import SIDES
    _, pairs, dist, masses = constraints_and_masses()
    rng = np.random.default_rng(1)
    for i in frames_of(a.frames):
        x, _, info = load_initial_panel(PANEL, i)
        dev0 = np.abs(np.linalg.norm(x[[p for p, _ in pairs]] - x[[q for _, q in pairs]], axis=1) - dist).max()
        out = [f"frame {i}: constraint |dev| at start {dev0:.1e} nm"]
        for name in ("T1", "T5"):
            draw, apply = torsion_move(name)
            th = draw(rng)
            y = apply(x, th)
            lig_cons = [c for c in pairs if c[0] < 35]
            dG = max(np.abs(gram(y, lig_cons, masses) - gram(x, lig_cons, masses)).max(),
                     np.abs(gram(y, lig_cons) - gram(x, lig_cons)).max())
            back = apply(y, -th)
            moved = [c for c in pairs if (c[0] in SIDES[name]) != (c[1] in SIDES[name])]
            out.append(f"{name} {np.degrees(th):+.1f} deg: max|dGram| {dG:.1e}, inverse {np.abs(back - x).max():.1e} nm, "
                       f"cross-cut constraints {len(moved)}")
        sysidx = info["expected_components"]["cluster_system_indices"]
        w = np.asarray(sysidx[35:]).reshape(50, 3)[int(rng.integers(50))]
        wc = [c for c in pairs if c[0] in w or c[1] in w]
        for kind in ("trans", "rot"):
            draw, apply = water_move(w, kind)
            m = draw(rng)
            y = apply(x, m)
            inv = (-m) if kind == "trans" else (m[0], -m[1])
            dG = max(np.abs(gram(y, wc, masses) - gram(x, wc, masses)).max(), np.abs(gram(y, wc) - gram(x, wc)).max())
            dl = np.abs(np.linalg.norm(y[[p for p, _ in wc]] - y[[q for _, q in wc]], axis=1)
                        - np.linalg.norm(x[[p for p, _ in wc]] - x[[q for _, q in wc]], axis=1)).max()
            out.append(f"water {kind}: {len(wc)} constraints, max|dGram| {dG:.1e}, max|dlength| {dl:.1e}, "
                       f"inverse {np.abs(apply(y, inv) - x).max():.1e} nm")
        print("; ".join(out), flush=True)


def six_torsions():
    import openmm
    import eta_hmc as eh
    from eta0_m25_verdict import ligand_graph
    system, _, _ = eh.mm_system(SYSTEM)
    meta = {"system": SYSTEM, "ligand_system_indices": list(range(35)),
            "system_definition": openmm.XmlSerializer.serialize(system)}
    heavy, polar, tors = ligand_graph(meta, 35)
    return heavy, polar, tors


def dipole(x, w):
    d = 0.5 * (x[w[1]] + x[w[2]]) - x[w[0]]
    return d / np.linalg.norm(d)


def diagnose(a):
    import openmm
    import eta_hmc as eh
    import local_E
    from compare_ladders import kabsch_rmsd
    from eta_exchange import DETAILS, initial_state, load_initial_panel
    from torsion_ncmc import dihedral, panel_model
    head, backbone = panel_model()
    leg = local_E.Leg(SYSTEM)
    me = eh.ModelEnergy(leg, 0, "float64", ckpt=head, backbone=backbone)
    ident = eh.model_identity(ckpt=head, backbone=backbone)
    system, groups, box = eh.mm_system(SYSTEM)
    ig = openmm.VerletIntegrator(0.001)
    c = eh.make_context(system, box, ig)
    eh.gm.set_state(c, 0, len(groups))
    heavy, polar, tors = six_torsions()
    out_dir = eh.HERE / "runs" / SYSTEM / f"S0{a.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    timers = {"V": 0.0, "model": 0.0, "carve": 0.0}

    def V(x):
        t = time.perf_counter()
        c.setPositions(x)
        v = c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(eh.KJ)
        timers["V"] += time.perf_counter() - t
        return v

    def dU(y):
        t = time.perf_counter()
        v = me(y)
        timers["model"] += time.perf_counter() - t
        return v, {k: np.asarray(me.last_components[k]).copy() for k in DETAILS}

    _, pairs, dist, _ = constraints_and_masses()
    total_model = 0
    for i in frames_of(a.frames):
        todo = []
        for b, name in BRANCHES.items():
            seed = SEED_BASE + 10000 * i + 100 * b
            want = json.loads(json.dumps({
                "system": SYSTEM, "state": 0, "eta": 1.0, "RT": eh.KT, "panel": str(PANEL), "frame": i, "branch": b,
                "branch_name": name, "steps": a.steps, "seed": seed, "E_construction": "local_E v1", "dtype": "float64",
                **ident, "theta_max_deg": 20.0, "k": K_MTM if b == 0 else 1, "torsion_choice": "T1/T5 1/2",
                "water": "uniform in current Omega; translation U(-0.01,0.01) nm per component or rotation about O, "
                         "axis uniform, phi U(-15,15) deg, 1/2 each; reject before energies unless still in Omega(y)",
                "acceptance": "two-stage: MM (MTM Rosenbluth, common zero V(x), x in reference set) then "
                              "exp(-eta dU difference / RT); rejection = self-loop step",
                "mm_system": "eta_hmc.mm_system S0, CUDA mixed"}))
            out = out_dir / f"f{i}_b{b}.npz"
            if out.exists():
                old = json.loads(str(np.load(out)["meta"]))
                diff = [k for k in want if old.get(k) != want[k]]
                if diff:
                    raise SystemExit(f"{out.name}: exists with a different contract ({', '.join(diff)}); use another --tag")
                print(f"{out.name}: complete, reused", flush=True)
                continue
            todo.append((b, name, seed, want, out))
        if not todo:
            continue
        x0, _, info = load_initial_panel(PANEL, i)
        calls0 = me.calls
        st = initial_state(c, x0, me, info)                 # one checked model query per frame
        V0 = V(st["x"])
        start = {"x": st["x"], "V": V0, "dU": st["e"], "details": st["details"]}
        cdev = float(np.abs(np.linalg.norm(st["x"][[p for p, _ in pairs]] - st["x"][[q for _, q in pairs]], axis=1) - dist).max())
        init_calls = me.calls - calls0
        print(f"frame {i}: start dU {st['e']:+.3f}, V {V0:.3f} kJ/mol, constraint |dev| {cdev:.1e} nm, init model calls {init_calls}",
              flush=True)
        lig = np.arange(35)
        for b, name, seed, want, out in todo:
            rng = np.random.default_rng(seed)
            state = {"x": start["x"].copy(), "V": start["V"], "dU": start["dU"],
                     "details": {k: v.copy() for k, v in start["details"].items()}}
            for t in timers:
                timers[t] = 0.0
            calls_b = me.calls
            recs, failure = [], None
            x_ref = start["x"]
            t0 = time.perf_counter()
            for s in range(a.steps):
                try:
                    if b in (0, 1):
                        tname = "T1" if rng.random() < 0.5 else "T5"
                        draw, apply = torsion_move(tname)
                        new, r = two_stage_step(state, K_MTM if b == 0 else 1, draw, apply, V, dU, 1.0, eh.KT, rng)
                        r["what"] = tname
                    else:
                        wat = np.asarray(state["details"]["cluster_system_indices"][35:]).reshape(50, 3)
                        jw = int(rng.integers(50))
                        w = [int(v) for v in wat[jw]]
                        kind = "trans" if rng.random() < 0.5 else "rot"
                        draw, apply = water_move(w, kind)

                        def admissible(x, y, o=w[0]):
                            t = time.perf_counter()
                            ok = o in set(leg.carve(y, 50)[3][35:].tolist())
                            timers["carve"] += time.perf_counter() - t
                            return ok
                        new, r = two_stage_step(state, 1, draw, apply, V, dU, 1.0, eh.KT, rng, admissible)
                        r["what"] = f"{kind} water O{w[0]}"
                        r["water"] = w
                except FloatingPointError as err:
                    failure = {"step": s, "error": str(err)}
                    print(f"  f{i} b{b} step {s}: FAILED ({err})", flush=True)
                    break
                r.pop("y", None)                                # the candidate itself is not part of the record
                state = new
                xs = state["x"]
                r.update({"V": state["V"], "dU": state["dU"],
                          "tors": [dihedral(xs[lig], tt) for tt in tors],
                          "rmsd_heavy_A": kabsch_rmsd(xs[heavy] * 10, x_ref[heavy] * 10)})
                if b == 2:
                    w = r["water"]
                    r["water_O_disp_A"] = float(np.linalg.norm(xs[w[0]] - x_ref[w[0]]) * 10)
                    r["water_dipole_turn_deg"] = float(np.degrees(np.arccos(np.clip(dipole(xs, w) @ dipole(x_ref, w), -1, 1))))
                    r["omega_same_set"] = bool(set(np.asarray(state["details"]["cluster_system_indices"]).tolist())
                                               == set(np.asarray(start["details"]["cluster_system_indices"]).tolist()))
                if isinstance(r["move"], np.ndarray):
                    r["move"] = r["move"].tolist()
                elif isinstance(r["move"], tuple):
                    r["move"] = [r["move"][0].tolist(), r["move"][1]]
                recs.append(r)
            wall = time.perf_counter() - t0
            meta = {**want, "complete": failure is None, "failure": failure, "steps_done": len(recs),
                    "model_calls_chain": me.calls - calls_b, "init_model_calls_frame": init_calls,
                    "init_shared_by_branches": [x[0] for x in todo], "start_dU": start["dU"], "start_V": start["V"],
                    "start_constraint_dev_nm": cdev, "seconds": wall, "timers_s": dict(timers),
                    "n_V": int(sum(r["n_V"] for r in recs)), "date": time.strftime("%F %T")}
            data = {"meta": json.dumps(meta), "records": json.dumps(recs), "x_start": start["x"], "x_final": state["x"],
                    **{f"final_{k}": v for k, v in state["details"].items()}}
            tmp = out.with_suffix(".tmp.npz")
            np.savez(tmp, **data)
            tmp.replace(out)
            total_model += meta["model_calls_chain"]
            acc = sum(r["acc"] for r in recs)
            s1 = sum(r["stage1"] for r in recs)
            print(f"{out.name}: {name}, {len(recs)} steps, stage1 {s1}, accepted {acc}, model calls {meta['model_calls_chain']}, "
                  f"MM evals {meta['n_V']}, {wall:.1f} s (V {timers['V']:.1f}, model {timers['model']:.1f}, carve {timers['carve']:.1f})",
                  flush=True)
        total_model += init_calls
    print(f"model calls this process {me.calls}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "diagnose"])
    ap.add_argument("--frames", default="0-7")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--tag", default="_localgeom_v1")
    a = ap.parse_args()
    {"check": check, "diagnose": diagnose}[a.cmd](a)
