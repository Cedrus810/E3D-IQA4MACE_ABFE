"""Model on the saved MM frames: per-state ligand-environment coupling and the energy of each step (user, 2026-10-23).

Run: python step_eval.py eval SYSTEM [--n-water 50] [--batch 32] [--max-frames N]   (resumable per state)
     python step_eval.py check SYSTEM                                               (new path vs model_eval.d_ie)
     python step_eval.py steps SYSTEM [--n-water 50]

MM made the frames (groupwise_mm.py, runs/SYSTEM/rep1_S*.npz); the model is loaded once and scores the
frames of every state in batches. Per frame of S_k, for every ligand atom i still coupled in S_k:
    D_iE = sum_{a in E} D_ia (head, 4 channels summed)  +  E_es,i^LE (explicit Coulomb, per atom)
Atoms of the k dummy groups get no D_iE (NaN). A dummy atom does not see the environment, so its
ligand-environment edges are cut from the graph (its edges inside the ligand stay), as MM turns its
interactions with the environment off. SK (all dummy) has no coupled atom and is not evaluated.
    E_k = < sum_{i coupled in S_k} D_iE >_{S_k},   step k -> k+1:  dE_k = E_k - E_{k+1},   E_K = 0
Cluster per frame: ligand + the N nearest waters (ions left out), non-periodic, as the head was trained.
"""

import argparse
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp.data import mace_batch  # noqa: E402
from decomp.openmm_bridge import carve_from_arrays  # noqa: E402
from decomp.polar_es import coulomb_cross  # noqa: E402

import groupwise_mm as gm  # noqa: E402
from model_eval import DEV, EV, load_model, topology  # noqa: E402


def carve(xyz_nm, top, n_water):
    lig, _, box, numbers, resids, resnames, keep = top
    pos, num, frag, idx = carve_from_arrays(xyz_nm.astype(np.float64)[keep] * 10, numbers[keep], resids[keep],
                                            resnames[keep], n_solvent=n_water, box_ang=np.diag(box) * 10)
    assert (keep[idx[:len(lig)]] == np.array(lig)).all()
    return pos, num, frag


def couplings(model, clusters, dummy):
    """[B, n_lig] D_iE in eV for a batch of carved clusters; dummy = [n_lig] bool, NaN rows out."""
    n_lig = len(dummy)
    data = mace_batch([c[0] for c in clusters], [c[1] for c in clusters], model.backbone,
                      charges=[0] * len(clusters), frag_ids=[c[2] for c in clusters], device=DEV)
    frag = data["frag_id"]
    in_L = frag == 0
    dm = torch.as_tensor(np.concatenate([np.r_[dummy, np.zeros(len(c[1]) - n_lig, bool)] for c in clusters]),
                         device=DEV)
    src, dst = data["edge_index"]
    keep = ~((in_L[src] != in_L[dst]) & (dm[src] | dm[dst]))     # cut dummy-environment edges only
    sub = {**data, "edge_index": data["edge_index"][:, keep], "shifts": data["shifts"][keep],
           "unit_shifts": data["unit_shifts"][keep]}
    with torch.no_grad():                                  # the per-atom Coulomb split runs its own autograd
        with coulomb_cross(model.backbone, frag, per_atom=True) as rec:
            h = model.backbone(sub, training=False, compute_force=False)["node_feats"]
        _, D = model.head(h, sub["positions"], sub["edge_index"])
    D = D.sum(-1) if D.dim() > 1 else D
    s, d = sub["edge_index"]
    cr = in_L[s] != in_L[d]
    lig_end = torch.where(in_L[s], s, d)
    # each unordered pair appears twice in the edge list: half to its ligand atom each time
    c = torch.zeros(len(frag), dtype=D.dtype, device=DEV).index_add(0, lig_end[cr], 0.5 * D[cr])
    c = c + rec["per_atom"].to(D.dtype)
    rows = data["ptr"][:-1, None] + torch.arange(n_lig, device=DEV)[None, :]
    out = c[rows].detach().cpu().numpy().astype(np.float64)
    out[:, dummy] = np.nan
    return out


def evaluate(a):
    top = topology(a.system)
    lig, groups = top[0], top[1]
    pos_in_lig = {x: i for i, x in enumerate(lig)}
    K = len(groups)
    model = load_model()
    tag = f"N{a.n_water}" + ("_smoke" if a.max_frames else "")
    t0, nf = time.perf_counter(), 0
    for k in range(K):                                     # SK: no coupled atom
        out = gm.HERE / "runs" / a.system / f"steps_{tag}_S{k}.npz"
        if out.exists():
            print(f"{out.name} exists; skip", flush=True)
            continue
        dummy = np.zeros(len(lig), bool)
        for g in groups[:k]:
            dummy[[pos_in_lig[x] for x in g]] = True
        xyz = np.load(gm.HERE / "runs" / a.system / f"rep1_S{k}.npz")["xyz"]
        frames = np.arange(len(xyz))[:a.max_frames] if a.max_frames else np.arange(len(xyz))
        D, ts = [], time.perf_counter()
        for b in range(0, len(frames), a.batch):
            D.append(couplings(model, [carve(xyz[f], top, a.n_water) for f in frames[b:b + a.batch]], dummy))
        D = np.concatenate(D) * EV                         # kJ/mol
        nf += len(frames)
        np.savez(out, D=D, frames=frames, dummy=dummy, n_water=a.n_water, batch=a.batch)
        dt = time.perf_counter() - ts
        print(f"{a.system} S{k}: {len(frames)} frames in {dt:.1f} s ({dt / len(frames) * 1e3:.0f} ms/frame, "
              f"batch {a.batch}); coupled-atom sum mean {np.nansum(D, 1).mean():+.1f} kJ/mol -> {out.name}",
              flush=True)
    print(f"done: {nf} frames, {time.perf_counter() - t0:.0f} s; peak GPU mem "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


def check(a):
    """S0: batched new path vs model_eval.d_ie frame by frame; S1: batch of 4 vs batch of 1."""
    from decomp.openmm_bridge import AlchemicalForce
    from model_eval import d_ie
    top = topology(a.system)
    lig, groups = top[0], top[1]
    pos_in_lig = {x: i for i, x in enumerate(lig)}
    model = load_model()
    xyz0 = np.load(gm.HERE / "runs" / a.system / "rep1_S0.npz")["xyz"][:4]
    cl = [carve(x, top, a.n_water) for x in xyz0]
    new = couplings(model, cl, np.zeros(len(lig), bool))
    force = AlchemicalForce(model, cl[0][1], cl[0][2], mode="graph")
    old = np.array([d_ie(model, force, c[0]) for c in cl])
    print(f"S0, 4 frames: max |new - model_eval.d_ie| {np.abs(new - old).max() * EV:.2e} kJ/mol "
          f"(per-frame sums {np.round(new.sum(1) * EV, 2).tolist()})")
    dummy = np.zeros(len(lig), bool)
    dummy[[pos_in_lig[x] for x in groups[0]]] = True
    xyz1 = np.load(gm.HERE / "runs" / a.system / "rep1_S1.npz")["xyz"][:4]
    cl = [carve(x, top, a.n_water) for x in xyz1]
    b4 = couplings(model, cl, dummy)
    b1 = np.concatenate([couplings(model, [c], dummy) for c in cl])
    print(f"S1, 4 frames: max |batch 4 - batch 1| {np.nanmax(np.abs(b4 - b1)) * EV:.2e} kJ/mol; "
          f"dummy entries NaN: {bool(np.isnan(b4[:, dummy]).all())}; coupled sums "
          f"{np.round(np.nansum(b4, 1) * EV, 2).tolist()}")


def steps(a):
    from pymbar import timeseries
    _, lig, groups, _, _ = gm.leg(a.system)
    pos_in_lig = {x: i for i, x in enumerate(lig)}
    gidx = [[pos_in_lig[x] for x in g] for g in groups]
    K = len(groups)

    def mean_se(x):
        g = timeseries.statistical_inefficiency(x) if len(x) > 10 and x.std() > 0 else 1.0
        return float(x.mean()), float(x.std(ddof=1) * np.sqrt(g / len(x)))

    E, Eg = [], []
    for k in range(K):
        D = np.load(gm.HERE / "runs" / a.system / f"steps_N{a.n_water}_S{k}.npz")["D"]
        E.append(mean_se(np.nansum(D, 1)))
        Eg.append([mean_se(D[:, ix].sum(1))[0] if k <= g else None for g, ix in enumerate(gidx)])
    E.append((0.0, 0.0))
    print(f"{a.system}, N_water {a.n_water} (kJ/mol; D < 0 = attractive coupling)")
    print("  <D_gE> per group, coupled groups only (rows S0..S{}):".format(K - 1))
    for k, row in enumerate(Eg):
        print(f"   S{k}: " + " ".join(f"{x:+9.1f}" if x is not None else f"{'dummy':>9s}" for x in row)
              + f"   E_{k} {E[k][0]:+9.1f} +- {E[k][1]:.1f}")
    st = []
    for k in range(K):
        dE, se = E[k][0] - E[k + 1][0], float(np.hypot(E[k][1], E[k + 1][1]))
        st.append({"step": f"S{k}->S{k + 1}", "group": k + 1, "dE": dE, "se": se})
        print(f"  step S{k}->S{k + 1} (G{k + 1} -> dummy): dE {dE:+9.1f} +- {se:.1f}")
    print(f"  sum of steps = E_0 {E[0][0]:+.1f}")
    json.dump({"E": E, "group_means": Eg, "steps": st},
              open(gm.HERE / "runs" / a.system / f"steps_N{a.n_water}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "check", "steps"])
    ap.add_argument("system")
    ap.add_argument("--n-water", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    {"eval": evaluate, "check": check, "steps": steps}[a.cmd](a)
