"""Training data for the head retrain with response supervision (PLAN.md §1, user 2026-10-25).

    T(x) = C_bb(x) - X(x),  C_bb = E_bb(L+Omega) - E_bb(L) - E_bb(Omega),  X = explicit Coulomb cross
    sum_c D_c(x) -> T(x),   Delta(sum_c D_c) -> Delta T   (same trajectory, environment fixed)

Per selected S0 frame x0 (endpoint S0, all coupled): Omega = the 50 nearest whole waters of x0 (local_E v1 rule),
one fresh-velocity MM trajectory of 20 x 2 fs (the eta-HMC proposal integrator, no accept step); at each kept step t
the ligand of x_t is rigidly fitted back onto the ligand of x0, environment and Omega stay those of x0
(diag_internal_coupling.py construction). Labels in float64 with the frozen POLAR-1-M:
    C_bb   decomp.clusters.coupling_labels (three passes)
    X      decomp.polar_es.coulomb_cross on the whole cluster (one pass, the same densities as the C_bb whole pass)
Stored in eV (test_joint.py units); test_joint subtracts its own cached X from C_bb as for every cluster (E3D_ES=1).

Run: python make_data.py gen SYSTEM --reps 5-21 [--frames 30] [--steps 0,5,10,20] [--seed 0]   (one file per rep,
                                                                                               rerun skips a rep whose
                                                                                               stored settings match)
     python make_data.py pack NAME SYSTEM[:REPS] ...   -> data/resp_NAME.npz (input of test_joint.py E3D_RESP)
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-25_eta_hmc"))
import eta_hmc as eh  # noqa: E402
from diag_fast_dU import Parts, kabsch_onto  # noqa: E402

N_WATER = 50
DATA = HERE / "data"


def reps_of(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def load_bb():
    import torch
    from model_eval import BB
    return torch.load(BB, map_location="cpu", weights_only=False).to(torch.float64).to("cuda").eval()


def cross_labels(bb, ps, ns, fs, batch):
    """[n] explicit Coulomb cross (eV), float64, one whole-cluster pass each."""
    import torch
    from decomp.data import mace_batch
    from decomp.polar_es import coulomb_cross
    out = []
    for k in range(0, len(ps), batch):
        sl = slice(k, k + batch)
        b = mace_batch(ps[sl], ns[sl], bb, charges=[0] * len(ps[sl]), frag_ids=fs[sl], device="cuda",
                       dtype=torch.float64)
        with torch.no_grad(), coulomb_cross(bb, b["frag_id"]) as rec:
            bb(b, training=False, compute_force=False)
        out.append(rec["cross"].detach().cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def gen(a):
    from decomp.clusters import coupling_labels
    leg = eh.local_E.Leg(a.system)
    parts = Parts(leg, None)
    lig = np.array(leg.lig)
    steps = [int(s) for s in a.steps.split(",")]
    assert steps[0] == 0 and steps == sorted(steps)
    system, groups, box = eh.mm_system(a.system)
    ig = eh.proposal_integrator(1, 2.0)
    c = eh.make_context(system, box, ig)
    eh.gm.set_state(c, 0, len(groups))
    bb = load_bb()
    out_dir = DATA / a.system
    out_dir.mkdir(parents=True, exist_ok=True)
    for rep in reps_of(a.reps):
        xyz = np.load(eh.local_E.GW / "runs" / a.system / f"rep{rep}_S0.npz")["xyz"]
        frames = np.linspace(0, len(xyz) - 1, a.frames).round().astype(int)
        meta = {"system": a.system, "rep": rep, "state": 0, "frames": frames.tolist(), "steps": steps,
                "dt_fs": 2.0, "n_water": N_WATER, "seed": a.seed + rep, "constraint_tol": eh.TOL,
                "labels": "C_bb coupling_labels, X coulomb_cross; float64; eV"}
        path = out_dir / f"rep{rep}.npz"
        if path.exists():
            old = json.loads(str(np.load(path)["meta"]))
            if old == meta:
                print(f"  {a.system} rep{rep}: done, same settings, skipped", flush=True)
                continue
            print(f"  {a.system} rep{rep}: stored settings differ, recomputing", flush=True)
        ig.setRandomNumberSeed(a.seed + rep)
        t0 = time.perf_counter()
        P, W, SI, num, frag = [], [], [], None, None
        for fi in frames:
            c.setPositions(xyz[fi].astype(np.float64))
            c.applyConstraints(eh.TOL)
            x0 = eh.positions(c)
            s0 = leg.carve(x0, N_WATER)[3]
            om = np.unique(parts.resids[s0[len(lig):]])
            ig.setGlobalVariableByName("draw", 1.0)
            xs = {0: x0}
            for st in range(1, steps[-1] + 1):
                ig.step(1)
                ig.setGlobalVariableByName("draw", 0.0)
                if st in steps:
                    xs[st] = eh.positions(c)
            for st in steps:
                y = x0.copy()
                y[lig] = kabsch_onto(xs[st][lig], x0[lig])
                p, n, f, si = parts.carve_set(y, om)
                if num is None:
                    num, frag = n, f
                assert (n == num).all() and (f == frag).all()
                P.append(p)
                SI.append(si)
            W.append(om)
        t_md = time.perf_counter() - t0
        t1 = time.perf_counter()
        ps, ns, fs = P, [num] * len(P), [frag] * len(P)
        cbb = np.asarray(coupling_labels(bb, ps, ns, fs, device="cuda", batch=a.batch), np.float64)
        t_cbb = time.perf_counter() - t1
        t1 = time.perf_counter()
        x = cross_labels(bb, ps, ns, fs, a.batch)
        t_x = time.perf_counter() - t1
        n_t = len(steps)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, meta=json.dumps(meta), positions=np.array(P), numbers=num, frag=frag,
                 C_bb=cbb.reshape(-1, n_t), cross=x.reshape(-1, n_t), omega_resids=np.array(W),
                 sysidx=np.array(SI))
        os.replace(tmp, path)
        T = (cbb - x).reshape(-1, n_t) * eh.EV
        print(f"  {a.system} rep{rep}: {len(frames)} frames x {n_t} configs ({len(num)} atoms); "
              f"MM+carve {t_md:.1f} s, C_bb {t_cbb:.1f} s ({t_cbb / len(P):.2f} s/config), "
              f"X {t_x:.1f} s ({t_x / len(P):.2f} s/config); T mean {T.mean():+.2f} kJ/mol, "
              f"sd of T(t)-T(0) at steps {steps[1:]}: "
              + " ".join(f"{(T[:, k] - T[:, 0]).std(ddof=1) if len(T) > 1 else float('nan'):.2f}"
                         for k in range(1, n_t)), flush=True)


def pack(a):
    """Concatenate per-rep files: lists of clusters, trajectory id, step index, and the labels."""
    pos, num, frag, cbb, x, traj, step, src = [], [], [], [], [], [], [], []
    tid = 0
    for spec in a.parts:
        system, _, reps = spec.partition(":")
        for rep in (reps_of(reps) if reps else []):
            z = np.load(DATA / system / f"rep{rep}.npz")
            meta = json.loads(str(z["meta"]))
            n_t = len(meta["steps"])
            P = z["positions"]
            for k in range(len(P)):
                pos.append(P[k])
                num.append(z["numbers"])
                frag.append(z["frag"])
                traj.append(tid + k // n_t)
                step.append(meta["steps"][k % n_t])
                src.append(f"{system}:rep{rep}:frame{meta['frames'][k // n_t]}")
            tid += len(P) // n_t
            cbb.append(z["C_bb"].ravel())
            x.append(z["cross"].ravel())
    out = DATA / f"resp_{a.name}.npz"
    obj = lambda L: np.array(L + [None], dtype=object)[:-1]  # noqa: E731
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, positions=obj(pos), numbers=obj(num), frag=obj(frag), C_bb=np.concatenate(cbb),
             cross=np.concatenate(x), traj=np.array(traj), step=np.array(step), source=np.array(src),
             parts=np.array(a.parts))
    os.replace(tmp, out)
    print(f"  {out}: {len(pos)} configurations, {tid} trajectories")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("system")
    g.add_argument("--reps", required=True)
    g.add_argument("--frames", type=int, default=30)
    g.add_argument("--steps", default="0,5,10,20")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--batch", type=int, default=4)
    p = sub.add_parser("pack")
    p.add_argument("name")
    p.add_argument("parts", nargs="+")
    a = ap.parse_args()
    {"gen": gen, "pack": pack}[a.cmd](a)
