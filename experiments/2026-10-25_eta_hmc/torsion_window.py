"""Fixed T1 bias windows with small-step moves and window exchange at eta = 1 (TORSION_WINDOW_PLAN_DRAFT_2026-11-04.md).

Run: python torsion_window.py run SYSTEM-free --group A|B|CA|CB [--sweeps 200] [--tag _tw1_v1] [--commit-every 25]

Labels (eta = 1): b = 0 unbiased, b = 1..11 with W_b(T1) = kappa [1 - cos(T1 - c_b)], kappa = 80 kJ/mol,
c_b = -90, -110, ..., -170, +170, ..., +70 deg (along the observed arc through 180). Control groups use 12 unbiased labels.
Target of label (eta, b): U = V + eta dU + W_b(T1). Per walker and sweep, fixed order:
  1 short MM-HMC (eta_hmc proposal integrator, L = 20, dt = 2 fs; forces from V only; one model query):
        log R = -[dH + eta d(dU) + dW_b] / RT
  2 T1 single torsion, 2 x; 3 T5 single torsion, 1 x (theta ~ U(-20, 20) deg); 4 water single move, 2 x:
        two-stage (local_geom_mc.two_stage_step) with stage 1 on V + W_b, stage 2 exp(-eta d(dU)/RT)
  5 label exchanges (no energy evaluation): adjacent windows alternately by sweep parity, then b = 0 with window
        1 + sweep mod 11; label a holding x, b holding y:
        d_swap = (eta_a - eta_b)(dU_y - dU_x) + W_a(y) + W_b(x) - W_a(x) - W_b(y)
The pure functions (bias, acceptance, schedule, biased two-stage step) are what the CPU tests drive.
Records: per sweep and walker the label, V, dU, T1, T5, kernel outcomes, ligand coordinates and Omega; a progress file every
--commit-every sweeps (walker states, caches, RNG states); a resume continues from coordinates with new integrator seeds,
recorded as a continuation (HMC redraws momenta, so coordinates are the Markov state). No hash.
"""

import argparse
import json
import pathlib
import time

import numpy as np

KAPPA = 80.0                                                     # kJ/mol
CENTERS_DEG = [-90, -110, -130, -150, -170, 170, 150, 130, 110, 90, 70]
T1_ATOMS = (14, 3, 6, 16)
T5_ATOMS = (3, 6, 16, 2)
PANEL_FRAME = {"A": 0, "B": 4, "CA": 0, "CB": 4}
SEED = {"A": 910000, "B": 920000, "CA": 930000, "CB": 940000}
N_T1, N_T5, N_W = 2, 1, 2


# ---------------------------------------------------------------- pure parts (shared with the tests)

def labels_for(group):
    """[(eta, kappa, centre_rad)]; label 0 unbiased."""
    if group in ("A", "B"):
        return [(1.0, 0.0, 0.0)] + [(1.0, KAPPA, float(np.radians(c))) for c in CENTERS_DEG]
    return [(1.0, 0.0, 0.0)] * (len(CENTERS_DEG) + 1)


def bias(t1, label):
    """W_b(T1) in kJ/mol; t1 in rad."""
    _, kappa, c = label
    return kappa * (1.0 - np.cos(t1 - c))


def hmc_log_accept(dH, eta, dU_new, dU_old, W_new, W_old, rt):
    return -(dH + eta * (dU_new - dU_old) + (W_new - W_old)) / rt


def swap_log_accept(la, lb, cx, cy, rt):
    """Label la holds the state with cache cx = (dU, t1), lb holds cy."""
    d = (la[0] - lb[0]) * (cy[0] - cx[0]) + bias(cy[1], la) + bias(cx[1], lb) - bias(cx[1], la) - bias(cy[1], lb)
    return -d / rt


def swap_pairs(sweep, n_windows):
    """Adjacent windows (label indices 1..n) by parity, then label 0 with window 1 + sweep mod n; fixed, state-free."""
    ph = sweep % 2
    pairs = [(k, k + 1) for k in range(1 + ph, n_windows, 2)]
    pairs.append((0, 1 + sweep % n_windows))
    return pairs


def exchange(perm, caches, labels, sweep, rng, rt):
    """perm[l] = walker at label l; caches[w] = (dU, t1). Sequential swaps in the fixed order; returns perm, log."""
    perm = list(perm)
    log = []
    for a, b in swap_pairs(sweep, len(labels) - 1):
        wa, wb = perm[a], perm[b]
        lr = swap_log_accept(labels[a], labels[b], caches[wa], caches[wb], rt)
        lu = float(np.log(rng.random()))
        acc = lu < lr
        if acc:
            perm[a], perm[b] = wb, wa
        log.append((a, b, float(lr), lu, bool(acc)))
    return perm, log


def biased_step(state, k, draw, apply, V, dU, t1_of, label, rt, rng, admissible=None):
    """local_geom_mc.two_stage_step on U = V + eta dU + W_b: stage 1 on V + W_b, stage 2 on eta d(dU) only.
    state = {"x", "V", "dU", "details", "t1"}; returns (new_state, record)."""
    from local_geom_mc import two_stage_step
    Wx = bias(state["t1"], label)
    s = {"x": state["x"], "V": state["V"] + Wx, "dU": state["dU"], "details": state["details"]}
    cache = {}

    def Vb(y):
        t = t1_of(y)
        cache[id(y)] = t
        return V(y) + bias(t, label)
    new, r = two_stage_step(s, k, draw, apply, Vb, dU, label[0], rt, rng, admissible)
    if new is s:
        return state, r
    t = cache[id(new["x"])]
    return {"x": new["x"], "V": new["V"] - bias(t, label), "dU": new["dU"], "details": new["details"], "t1": t}, r


# ---------------------------------------------------------------- molecular run

SYSTEM = "CyclophilinD_ligand-8"
HERE = pathlib.Path(__file__).resolve().parent
PANEL = HERE / "logs" / "deep_shallow_panel_20261031.npz"


def dihedral_rad(x, t):
    from torsion_ncmc import dihedral
    return float(np.radians(dihedral(x, t)))


def run(a):
    import openmm
    import eta_hmc as eh
    import local_E
    from eta_exchange import DETAILS, initial_state, load_initial_panel
    from local_geom_mc import torsion_move, water_move
    from torsion_ncmc import panel_model
    labels = labels_for(a.group)
    K = len(labels)
    head, backbone = panel_model()
    ident = eh.model_identity(ckpt=head, backbone=backbone)
    want = json.loads(json.dumps({
        "schema": "torsion-window-1", "system": SYSTEM, "state": 0, "group": a.group, "labels": labels,
        "kappa": KAPPA, "centers_deg": CENTERS_DEG if a.group in ("A", "B") else None, "panel": str(PANEL),
        "panel_frame": PANEL_FRAME[a.group], "seed_base": SEED[a.group], "sweeps": a.sweeps,
        "kernel": {"hmc": {"L": 20, "dt_fs": 2.0, "n": 1}, "T1": N_T1, "T5": N_T5, "water": N_W, "theta_max_deg": 20.0,
                   "water_moves": "uniform in Omega; translation U(-0.01,0.01) nm/component or rotation about O U(-15,15) deg, 1/2 each"},
        "exchange": "adjacent windows by sweep parity, then label 0 with window 1 + sweep mod 11; sequential",
        "E_construction": "local_E v1", "dtype": "float64", **ident, "RT": eh.KT,
        "commit_every": a.commit_every}))
    out = eh.HERE / "runs" / SYSTEM / f"tw_{a.group}{a.tag}.npz"
    prog = out.with_name(out.stem + ".progress.npz")
    leg = local_E.Leg(SYSTEM)
    me = eh.ModelEnergy(leg, 0, "float64", ckpt=head, backbone=backbone)
    system, groups, box = eh.mm_system(SYSTEM)
    rec = {k: [] for k in ("perm", "swap", "V", "dU", "t1", "t5", "lig_xyz", "omega", "hmc_acc", "hmc_dH",
                           "geo_acc", "geo_stage1", "n_model", "n_V")}
    continuations, done, t_prior = [], 0, 0.0
    saved = out if out.exists() else prog if prog.exists() else None
    if saved is not None:
        z = np.load(saved, allow_pickle=False)
        old = json.loads(str(z["meta"]))
        diff = [k for k in want if old.get(k) != want[k]]
        if diff:
            raise SystemExit(f"{saved.name}: different contract ({', '.join(diff)}); use another --tag")
        if old["complete"]:
            print(f"{out.name}: complete, reused", flush=True)
            return
        done, t_prior, continuations = int(old["completed_sweeps"]), float(old["seconds"]), list(old["continuations"])
        for k in rec:
            rec[k] = list(z[k])
        states = [{"x": z["state_x"][w].astype(np.float64), "V": float(z["state_V"][w]), "dU": float(z["state_dU"][w]),
                   "t1": float(z["state_t1"][w]), "details": {k: z[f"state_{k}"][w] for k in DETAILS}} for w in range(K)]
        perm = list(map(int, z["perm"][-1]))
        rngs = [np.random.default_rng() for _ in range(K)]
        for w in range(K):
            rngs[w].bit_generator.state = json.loads(str(z[f"rng_w{w}"]))
        rng_x = np.random.default_rng()
        rng_x.bit_generator.state = json.loads(str(z["rng_x"]))
        continuations.append({"after_sweep": done, "mode": "coordinates, new integrator seeds"})
        print(f"{out.name}: resuming after sweep {done}", flush=True)
    stride = 7919 * len(continuations)
    ctxs, igs = [], []
    for w in range(K):
        ig = eh.proposal_integrator(20, 2.0)
        ig.setRandomNumberSeed(SEED[a.group] + 10 * w + 1 + stride)
        c = eh.make_context(system, box, ig)
        eh.gm.set_state(c, 0, len(groups))
        ctxs.append(c)
        igs.append(ig)
    calls0 = me.calls
    if saved is None:
        x0, _, info = load_initial_panel(PANEL, PANEL_FRAME[a.group])
        st0 = initial_state(ctxs[0], x0, me, info)               # one checked model query for all walkers
        ctxs[0].setPositions(st0["x"])
        V0 = ctxs[0].getState(getEnergy=True).getPotentialEnergy().value_in_unit(eh.KJ)
        t10 = dihedral_rad(st0["x"], T1_ATOMS)
        states = [{"x": st0["x"].copy(), "V": V0, "dU": st0["e"], "t1": t10,
                   "details": {k: np.asarray(v).copy() for k, v in st0["details"].items()}} for _ in range(K)]
        perm = list(range(K))
        rngs = [np.random.default_rng(SEED[a.group] + 10 * w) for w in range(K)]
        rng_x = np.random.default_rng(SEED[a.group] + 9)
    t0 = time.perf_counter()

    def dU_fn(y):
        v = me(y)
        return v, {k: np.asarray(me.last_components[k]).copy() for k in DETAILS}

    while done < a.sweeps:
        s = done
        inv = np.empty(K, int)
        inv[perm] = np.arange(K)
        row = {k: [] for k in ("V", "dU", "t1", "t5", "lig_xyz", "omega", "hmc_acc", "hmc_dH", "geo_acc", "geo_stage1",
                               "n_model", "n_V")}
        for w in range(K):
            lab = labels[inv[w]]
            c, ig, rng, st = ctxs[w], igs[w], rngs[w], states[w]
            nm, nv = 0, 0

            def V(y, c=c):
                nonlocal nv
                nv += 1
                c.setPositions(y)
                return c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(eh.KJ)
            # 1 short MM-HMC with bias and model terms in the acceptance
            c.setPositions(st["x"])
            ig.setGlobalVariableByName("draw", 1.0)
            ig.step(1)
            dH = ig.getGlobalVariableByName("Enew") - ig.getGlobalVariableByName("Eold")
            xn = eh.positions(c)
            if not np.isfinite(dH):
                raise FloatingPointError(f"sweep {s} walker {w}: non-finite HMC work")
            dUn, detn = dU_fn(xn)
            nm += 1
            t1n = dihedral_rad(xn, T1_ATOMS)
            lr = hmc_log_accept(dH, lab[0], dUn, st["dU"], bias(t1n, lab), bias(st["t1"], lab), eh.KT)
            hacc = bool(np.log(rng.random()) < lr)
            if hacc:
                st = {"x": xn, "V": ig.getGlobalVariableByName("Vnew"), "dU": dUn, "details": detn, "t1": t1n}
            geo_acc, geo_s1 = [], []
            for name, n in (("T1", N_T1), ("T5", N_T5)):
                draw, apply = torsion_move(name)
                for _ in range(n):
                    st, r = biased_step(st, 1, draw, apply, V, dU_fn, lambda y: dihedral_rad(y, T1_ATOMS), lab, eh.KT, rng)
                    nm += r["n_model"]
                    geo_acc.append(bool(r["acc"]))
                    geo_s1.append(bool(r["stage1"]))
            for _ in range(N_W):
                wat = np.asarray(st["details"]["cluster_system_indices"][35:]).reshape(50, 3)
                wsel = [int(v) for v in wat[int(rng.integers(50))]]
                draw, apply = water_move(wsel, "trans" if rng.random() < 0.5 else "rot")
                adm = (lambda x, y, o=wsel[0]: o in set(leg.carve(y, 50)[3][35:].tolist()))
                st, r = biased_step(st, 1, draw, apply, V, dU_fn, lambda y: dihedral_rad(y, T1_ATOMS), lab, eh.KT, rng, adm)
                nm += r["n_model"]
                geo_acc.append(bool(r["acc"]))
                geo_s1.append(bool(r["stage1"]))
            if not (np.isfinite(st["V"]) and np.isfinite(st["dU"])):
                raise FloatingPointError(f"sweep {s} walker {w}: non-finite state")
            states[w] = st
            for k, v in (("V", st["V"]), ("dU", st["dU"]), ("t1", st["t1"]), ("t5", dihedral_rad(st["x"], T5_ATOMS)),
                         ("lig_xyz", st["x"][:35]), ("omega", np.asarray(st["details"]["omega_water_resid"])),
                         ("hmc_acc", hacc), ("hmc_dH", dH), ("geo_acc", geo_acc), ("geo_stage1", geo_s1),
                         ("n_model", nm), ("n_V", nv)):
                row[k].append(v)
        perm, slog = exchange(perm, [(st["dU"], st["t1"]) for st in states], labels, s, rng_x, eh.KT)
        rec["perm"].append(np.array(perm))
        rec["swap"].append(np.array(slog, dtype=float))
        for k, v in row.items():
            rec[k].append(np.array(v))
        done += 1
        if done % a.commit_every == 0 or done == a.sweeps:
            meta = {**want, "completed_sweeps": done, "complete": done == a.sweeps, "continuations": continuations,
                    "seconds": t_prior + time.perf_counter() - t0, "model_calls_this_process": me.calls - calls0,
                    "date": time.strftime("%F %T")}
            data = {"meta": json.dumps(meta), **{k: np.array(v) for k, v in rec.items()},
                    "state_x": np.array([st["x"] for st in states]), "state_V": np.array([st["V"] for st in states]),
                    "state_dU": np.array([st["dU"] for st in states]), "state_t1": np.array([st["t1"] for st in states]),
                    **{f"state_{k}": np.array([st["details"][k] for st in states]) for k in DETAILS},
                    "rng_x": json.dumps(rng_x.bit_generator.state),
                    **{f"rng_w{w}": json.dumps(rngs[w].bit_generator.state) for w in range(K)}}
            tgt = out if done == a.sweeps else prog
            tmp = tgt.with_name(tgt.stem + ".tmp.npz")
            np.savez(tmp, **data)
            tmp.replace(tgt)
            if done == a.sweeps:
                prog.unlink(missing_ok=True)
            sw = np.array(rec["swap"][-a.commit_every:]) if rec["swap"] else None
            print(f"{out.name}: committed {done}/{a.sweeps}; model calls this process {me.calls - calls0}; "
                  f"HMC acc {np.mean(rec['hmc_acc'][-a.commit_every:]):.2f}; geo acc {np.mean(rec['geo_acc'][-a.commit_every:]):.2f}; "
                  f"swap acc {sw[..., 4].mean():.2f}; {meta['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--group", required=True, choices=list(PANEL_FRAME))
    ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--tag", default="_tw1_v1")
    ap.add_argument("--commit-every", type=int, default=25)
    a = ap.parse_args()
    run(a)
