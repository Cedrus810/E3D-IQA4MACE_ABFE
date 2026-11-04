"""eta=0 m=25 vs m=1 deep / shallow ladders: evidence for the three verdict layers (ETA0_MM_UPDATES_DECISION_2026-11-01.md
"运行后的裁决"). Read-only, NumPy only, no model or MD.

Run: python -B eta0_m25_verdict.py --out logs/eta0_m25_verdict_<date>.json
Inputs: ladder_S0_start{r}_start{deep,shallow}8{_v1 (m=1, PBS 22191), _m25_v1 (m=25)}.npz and the shared panel.
All blocks of --block sweeps kept; rejected steps kept (states are the actual chain states, never proposals).

1 Generation at eta=0 (the walker holding eta=0 during the sweep, before exchange):
    per-sweep ligand heavy-atom RMSD between its state after and before the sweep (A, superposed; 0 if unchanged);
    eta=0 label view: RMSD to the ladder's own start, Omega retained from the own start, <dU>;
    rotatable heavy-atom torsions (non-ring heavy-heavy bonds, from the System bonds): 60-degree bin changes along the
    eta=0 label series; water O within 3.5 A of ligand N/O on the saved full-box snapshots (every 10 sweeps).
2 Transport: eta=0 -> eta=1 arrivals and round trips; accepted-state versions first seen at eta=1 (a version is a
    walker's state after a changed sweep; recycling of already-seen versions is not new); eta=1 label view RMSD to the
    own start, Omega retained, torsion bin changes; sweeps since the eta=1 walker last held eta=0 (right-censored if never).
3 Initial-state dependence: eta=1 <dU> blocks, fraction below -90 kJ/mol (descriptive marker), RMSD to the deep frame,
    all-eta block means; deep vs shallow pooled eta=1 torsion-bin occupancy (total variation distance); final eta=1
    states pairwise RMSD. Water identities are not comparable across ladders from different source boxes.
No pass line: the verdict is read from these numbers against the decision table.
"""

import argparse
import json
import os
import re

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from compare_ladders import kabsch_rmsd, transport  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "CyclophilinD_ligand-8")
PANEL = os.path.join(HERE, "logs", "deep_shallow_panel_20261031.npz")
TAGS = {"m1": "8_v1", "m25": "8_m25_v1"}
DEEP_MARK = -90.0
NBIN = 6


def ligand_graph(meta, n_lig):
    """Heavy / N,O atoms by atomic number (HMR masses cannot tell CH2 carbons from H: 12.01 - 2 x 3.02 = 5.97)."""
    import eta_hmc  # noqa: F401  (sys.path for model_eval)
    from model_eval import topology
    lig, _, _, numbers = topology(meta["system"])[:4]
    z = np.asarray(numbers)[np.asarray(lig)][:n_lig]
    assert list(map(int, lig)) == list(meta["ligand_system_indices"])
    heavy = [i for i in range(n_lig) if z[i] > 1]
    polar = [i for i in range(n_lig) if z[i] in (7, 8)]
    hs = set(heavy)
    bonds = set()
    for p1, p2 in re.findall(r'<Bond[^>]*p1="(\d+)"[^>]*p2="(\d+)"', meta["system_definition"]):
        a, b = int(p1), int(p2)
        if a in hs and b in hs:
            bonds.add((min(a, b), max(a, b)))
    nb = {i: set() for i in heavy}
    for a, b in bonds:
        nb[a].add(b), nb[b].add(a)

    def connected_without(a, b):
        seen, todo = {a}, [a]
        while todo:
            u = todo.pop()
            for v in nb[u]:
                if (min(u, v), max(u, v)) == (min(a, b), max(a, b)) or v in seen:
                    continue
                if v == b:
                    return True
                seen.add(v), todo.append(v)
        return False

    tors = []
    for b, c in sorted(bonds):
        if len(nb[b]) < 2 or len(nb[c]) < 2 or connected_without(b, c):
            continue
        tors.append((min(nb[b] - {c}), b, c, min(nb[c] - {b})))
    return heavy, polar, tors


def dihedral(x, t):
    p0, p1, p2, p3 = (x[..., i, :] for i in t)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    return np.degrees(np.arctan2((np.cross(b1n, v) * w).sum(-1), (v * w).sum(-1)))


def bins(angles):
    return ((angles + 180.0) // (360.0 / NBIN)).astype(int) % NBIN


def blockmean(v, block):
    return [float(np.mean(v[b:b + block])) for b in range(0, len(v), block)]


def ladder(cfg, branch, r, panel, block):
    path = os.path.join(RUNS, f"ladder_S0_start{r}_start{branch}{TAGS[cfg]}.npz")
    z = np.load(path, allow_pickle=False)
    m = json.loads(str(z["meta"]))
    fi = m["panel_initialization"]["frame_index"]
    assert m["complete"] and fi == (0 if branch == "deep" else r + 3)
    perm, ldU, wel = z["perm"], z["label_dU"], z["walker_eta_local"]
    n, K = perm.shape
    lig = np.array(m["ligand_system_indices"])
    heavy, polar, tors = ligand_graph(m, len(lig))
    LX = z["lig_xyz"] * 10                                            # [sweep, walker, atom, 3] A
    own = panel["xyz_nm"][fi][lig] * 10
    deep = panel["xyz_nm"][0][lig] * 10
    own_om = set(panel["omega_water_resid"][fi])
    out = {"config": cfg, "branch": branch, "rep": r, "panel_frame": fi, "path": path,
           "eta0_mm_updates": int(m.get("eta0_mm_updates", 1)), "seconds": m.get("seconds"),
           "torsions": [list(map(int, t)) for t in tors]}
    # 1 generation: the walker that held eta=0 during sweep s, its state after s vs after s-1
    w0 = np.argmax(wel == 0.0, axis=1)
    move = np.array([0.0] + [kabsch_rmsd(LX[s, w0[s]][heavy], LX[s - 1, w0[s]][heavy]) for s in range(1, n)])
    changed = np.asarray(z["acc"], bool)[np.arange(n), w0]
    views = {}
    for name, k in (("eta0", 0), ("eta1", K - 1)):
        wk = perm[:, k]
        x = LX[np.arange(n), wk]
        om = z["omega_water_resid"][np.arange(n), wk]
        tb = np.stack([bins(dihedral(x, t)) for t in tors], 1) if tors else np.zeros((n, 0), int)
        views[name] = {"rmsd_own_A": blockmean(np.array([kabsch_rmsd(xi[heavy], own[heavy]) for xi in x]), block),
                       "rmsd_deep_A": blockmean(np.array([kabsch_rmsd(xi[heavy], deep[heavy]) for xi in x]), block),
                       "omega_retained_own": blockmean(np.array([len(own_om & set(o)) / 50 for o in om]), block),
                       "dU": blockmean(ldU[:, k], block),
                       "frac_below_mark": blockmean((ldU[:, k] < DEEP_MARK).astype(float), block),
                       "torsion_bin_changes": (tb[1:] != tb[:-1]).sum(0).tolist(),
                       "torsion_bin_occupancy": [np.bincount(tb[:, j], minlength=NBIN).tolist() for j in range(len(tors))]}
    # water geometry on saved snapshots, eta=0 and eta=1 label walkers
    xs = np.asarray(z["xyz_sweeps"]) - 1
    full = z["xyz"]                                                   # [snap, walker, particle, 3] nm
    idx = np.array(json.loads(str(z["meta"]))["ligand_system_indices"])
    for name, k in (("eta0", 0), ("eta1", K - 1)):
        cnt = []
        for q, s in enumerate(xs):
            w = perm[s, k]
            om = z["omega_water_resid"][s, w]
            sysidx = z["cluster_system_indices"][s, w][len(idx):]
            box = np.diag(np.asarray(m["box_nm"]))
            O = full[q, w][sysidx[::3]] * 10                          # water O first in each residue triple
            L = full[q, w][idx][polar] * 10
            d = O[:, None] - L[None]
            d -= np.round(d / (box * 10)) * box * 10
            cnt.append(int((np.linalg.norm(d, axis=-1).min(1) < 3.5).sum()))
        views[name]["waters_near_NO_snapshots"] = blockmean(np.array(cnt, float), max(1, block // 10))
    out["eta0_walker_move_rmsd_A"] = blockmean(move, block)
    out["eta0_walker_changed_frac"] = blockmean(changed.astype(float), block)
    if out["eta0_mm_updates"] > 1:
        sub = np.asarray(z["mm_sub_acc"])[np.arange(n), w0]
        out["eta0_substep_acc"] = blockmean(sub.mean(1), block)
    # 2 transport
    up, rt, label = transport(perm, K)
    ver = np.zeros(K, int)
    seen, new_top, since0, last0 = set(), np.zeros(n), np.full(n, np.nan), np.full(K, -1)
    acc = np.asarray(z["acc"], bool)
    for s in range(n):
        ver += acc[s]
        last0[w0[s]] = s
        t = perm[s, K - 1]
        key = (int(t), int(ver[t]))
        if key not in seen:
            seen.add(key)
            new_top[s] = 1
        since0[s] = s - last0[t] if last0[t] >= 0 else np.nan
    out.update({"arrivals_0_to_1": int(up), "round_trips": int(rt),
                "eta1_new_versions_per_block": [int(new_top[b:b + block].sum()) for b in range(0, n, block)],
                "eta1_sweeps_since_eta0_median": [float(np.nanmedian(since0[b:b + block]))
                                                  if np.isfinite(since0[b:b + block]).any() else None
                                                  for b in range(0, n, block)],
                "eta1_never_held_eta0_frac": blockmean(np.isnan(since0).astype(float), block),
                "all_eta_dU_blocks": [ldU[b:b + block].mean(0).tolist() for b in range(0, n, block)],
                "views": views, "final_eta1_lig_heavy_A": LX[-1, perm[-1, K - 1]][heavy].tolist()})
    return out


def tvd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def main(a):
    if os.path.exists(a.out):
        raise SystemExit(f"preserving {a.out}; choose another --out")
    panel = np.load(PANEL, allow_pickle=False)
    L = [ladder(cfg, br, r, panel, a.block) for cfg in ("m1", "m25") for br in ("deep", "shallow") for r in (1, 2, 3, 4)]
    tors = L[0]["torsions"]
    cross = {}
    for cfg in ("m1", "m25"):
        occ = {br: np.sum([x["views"]["eta1"]["torsion_bin_occupancy"] for x in L if x["config"] == cfg and x["branch"] == br], 0)
               for br in ("deep", "shallow")}
        cross[cfg] = {"eta1_torsion_tvd_deep_vs_shallow": [tvd(occ["deep"][j], occ["shallow"][j]) for j in range(len(tors))]}
        fin = [np.array(x["final_eta1_lig_heavy_A"]) for x in L if x["config"] == cfg]
        cross[cfg]["final_eta1_pairwise_rmsd_A"] = [[kabsch_rmsd(p, q) for q in fin] for p in fin]
    res = {"block_sweeps": a.block, "deep_mark_kJmol": DEEP_MARK, "torsion_bins_deg": 360 / NBIN, "torsions": tors,
           "ladders": L, "cross": cross, "note": "descriptive; no pass line"}
    json.dump(res, open(a.out, "w"), indent=1)

    def f(v, fmt="{:+.1f}"):
        return "/".join(fmt.format(x) for x in v)
    print(f"blocks of {a.block} sweeps; rotatable heavy-atom torsions {tors}")
    print("\n[1 generation at eta=0]  walker move RMSD A per sweep | changed frac | eta0 view: RMSD own, Omega kept, "
          "torsion bin changes (sum), waters near N/O, <dU>")
    for x in L:
        v = x["views"]["eta0"]
        print(f"  {x['config']:>3s} {x['branch'][0]}{x['rep']}  {f(x['eta0_walker_move_rmsd_A'], '{:.2f}')} | "
              f"{f(x['eta0_walker_changed_frac'], '{:.2f}')} | {f(v['rmsd_own_A'], '{:.2f}')} | "
              f"{f(v['omega_retained_own'], '{:.2f}')} | {sum(v['torsion_bin_changes']):4d} | "
              f"{f(v['waters_near_NO_snapshots'], '{:.1f}')} | {f(v['dU'])}")
    print("\n[2 transport]  arrivals 0->1, round trips | new eta1 versions per block | eta1 sweeps since eta0 (median) | "
          "eta1 view: RMSD own, Omega kept, torsion bin changes")
    for x in L:
        v = x["views"]["eta1"]
        s0 = "/".join("-" if q is None else f"{q:.0f}" for q in x["eta1_sweeps_since_eta0_median"])
        print(f"  {x['config']:>3s} {x['branch'][0]}{x['rep']}  {x['arrivals_0_to_1']:3d} {x['round_trips']:3d} | "
              f"{'/'.join(map(str, x['eta1_new_versions_per_block']))} | {s0} | {f(v['rmsd_own_A'], '{:.2f}')} | "
              f"{f(v['omega_retained_own'], '{:.2f}')} | {sum(v['torsion_bin_changes']):4d}")
    print("\n[3 initial-state dependence]  eta1 <dU> | frac < -90 | eta1 RMSD to deep frame")
    for x in L:
        v = x["views"]["eta1"]
        print(f"  {x['config']:>3s} {x['branch'][0]}{x['rep']}  {f(v['dU'])} | {f(v['frac_below_mark'], '{:.2f}')} | "
              f"{f(v['rmsd_deep_A'], '{:.2f}')}")
    for cfg in ("m1", "m25"):
        print(f"  {cfg}: eta1 torsion-bin TVD deep vs shallow per torsion "
              f"{[round(t, 2) for t in cross[cfg]['eta1_torsion_tvd_deep_vs_shallow']]}")
        pr = np.array(cross[cfg]["final_eta1_pairwise_rmsd_A"])
        print(f"  {cfg}: final eta1 RMSD deep-deep {pr[:4, :4][np.triu_indices(4, 1)].round(2).tolist()}, "
              f"shallow-shallow {pr[4:, 4:][np.triu_indices(4, 1)].round(2).tolist()}, deep-shallow min {pr[:4, 4:].min():.2f} "
              f"max {pr[:4, 4:].max():.2f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
