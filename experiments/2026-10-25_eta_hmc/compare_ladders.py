"""Exchange ladders vs matched no-exchange ladders (read-only on eta_exchange-1 records; no model, no MD).

Per ladder: exchange rate per adjacent pair; walker transport (one-way 0 -> top arrivals, round trips);
arrivals at the top label of walkers that visited eta=0 since their previous top stay; structural change of the
TOP-label view per sweep (ligand internal RMSD, ligand centre, Omega turnover at lags; a swap that brings
another walker counts as change); top-label <dU> (statistical inefficiency in sweeps, SE); across-ladder spread.
Budget is the same for both protocols: R(N+1) model calls per ladder. No pass threshold.

Run: python -B compare_ladders.py --tags _ex_v1,_ctl_v1 --out logs/compare_ladders_v1.json
"""

import argparse
import json
import os

import numpy as np

from eta_exchange import read_ladder

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
LAGS = (1, 10, 50, 100)


def kabsch_rmsd(a, b):
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt(((a0 @ (u @ np.diag([1, 1, d]) @ vt) - b0) ** 2).sum(1).mean()))


def transport(perm, K):
    """One-way bottom->top arrivals and round trips (top->bottom->top counted per walker)."""
    n = perm.shape[0]
    label = np.empty_like(perm)
    label[np.arange(n)[:, None], perm] = np.arange(K)[None, :]          # label[s, w]
    up, rt_count = 0, 0
    for w in range(K):
        last = None                                                      # last extreme visited: 'b' or 't'
        trips = []
        for s in range(n):
            l = label[s, w]
            if l == 0 and last != "b":
                last = "b"
                trips.append("b")
            elif l == K - 1 and last != "t":
                if last == "b":
                    up += 1
                last = "t"
                trips.append("t")
        seq = "".join(trips)
        rt_count += sum(seq[i - 2:i + 1] == "tbt" for i in range(2, len(seq)))
    return up, rt_count, label


def start_ligands(m):
    """Ligand coordinates of the K true starting frames (meta start_frames of rep start_rep, MM production)."""
    if "panel_initialization" in m:
        # Use the values embedded in the run contract, not a mutable external panel file.
        x = np.asarray(m["panel_initialization"]["positions_nm"], np.float64)
        lig = np.asarray(m["ligand_system_indices"])
        return [x[lig].copy() for _ in m["etas"]]
    gw = os.path.join(os.path.dirname(HERE), "2026-10-23_groupwise_dummy", "runs", m["system"])
    z = np.load(os.path.join(gw, f"rep{m['start_rep']}_S{m['state']}.npz"))
    lig = np.asarray(m["ligand_system_indices"])
    out = []
    for src in m["start_frames"]:
        x = z["last_positions"] if src == "last_positions" else z["xyz"][int(src[4:-1])]
        out.append(np.asarray(x, np.float64)[lig])
    return out


def excursions(z, perm, label, K, init, top_view):
    """Per arrival at the top label of a walker that visited eta=0 since its previous top stay:
    internal RMSD of the arriving structure to the nearest start frame and to the nearest structure previously seen at
    the top label; and the walker's own internal change across its last eta=0 stay. Baseline: same novelty for
    top-view structures produced by local moves (no walker change)."""
    n = perm.shape[0]
    lig = z["lig_xyz"]
    rows, seen_bottom, bottom_start = [], [False] * K, [None] * K
    for s in range(n):
        for w in range(K):
            if label[s, w] == 0:
                if not seen_bottom[w] or bottom_start[w] is None:
                    bottom_start[w] = [s, s]
                bottom_start[w][1] = s
                seen_bottom[w] = True
        wt = perm[s, K - 1]
        if s > 0 and perm[s - 1, K - 1] != wt and seen_bottom[wt]:
            hist = top_view[:s:2]
            b0, b1 = bottom_start[wt]
            rows.append({"sweep": s + 1, "walker": int(wt),
                         "rmsd_to_nearest_start": float(min(kabsch_rmsd(lig[s, wt], x) for x in init)),
                         "rmsd_to_nearest_previous_top": float(min(kabsch_rmsd(lig[s, wt], h) for h in hist)),
                         "bottom_stay_sweeps": int(b1 - b0 + 1),
                         "change_during_bottom_stay": float(kabsch_rmsd(lig[b1, wt], lig[b0, wt]))})
            seen_bottom[wt], bottom_start[wt] = False, None
    base = [min(kabsch_rmsd(top_view[s], h) for h in top_view[:s:2]) for s in range(2, n, 10)
            if perm[s - 1, K - 1] == perm[s, K - 1]]
    return {"top_arrivals_after_bottom": rows,
            "local_top_novelty_vs_previous_top_mean": float(np.mean(base)) if base else None}


def ladder_metrics(path, discard):
    from pymbar import timeseries
    m, z = read_ladder(path)
    K, n = len(m["etas"]), int(m["completed_sweeps"])
    perm, top = z["perm"], K - 1
    out = {"file": os.path.basename(str(path)), "start": m["start_rep"], "exchange": m["exchange"],
           "model_calls_ladder": m["model_calls_ladder"], "seconds": m["seconds"]}
    att, acc = z["ex_attempt"], z["ex_acc"]
    out["pair_exchange_rate"] = [float(acc[:, k].sum() / att[:, k].sum()) if att[:, k].any() else None for k in range(K - 1)]
    up, rtc, label = transport(perm, K)
    out["oneway_bottom_to_top"], out["round_trips_top_bottom_top"] = up, rtc
    # arrivals at top of walkers that visited eta=0 since their previous top stay
    fresh, seen_bottom = 0, {w: False for w in range(K)}
    for s in range(n):
        for w in range(K):
            if label[s, w] == 0:
                seen_bottom[w] = True
        wt = perm[s, top]
        if (s == 0 or perm[s - 1, top] != wt) and seen_bottom[wt]:
            fresh += 1
            seen_bottom[wt] = False
        elif s == 0 or perm[s - 1, top] == wt:
            pass
    out["top_arrivals_after_bottom_visit"] = fresh
    out["top_label_walker_changes"] = int((perm[1:, top] != perm[:-1, top]).sum())
    # local acceptance grouped by the eta each local move actually used (walker_eta_local), not the post-exchange perm
    out["local_acc_by_eta"] = [float(z["acc"][z["walker_eta_local"] == e].mean()) for e in m["etas"]]
    # top-label view: ligand coordinates and Omega identity of whichever walker holds the top label
    lig = z["lig_xyz"][np.arange(n), perm[:, top]]
    om = z["omega_water_resid"][np.arange(n), perm[:, top]]
    cen = lig.mean(1)
    for k in LAGS:
        idx = np.arange(0, n - k, max(1, k // 2))
        out[f"top_lig_internal_rmsd_nm_lag{k}"] = float(np.mean([kabsch_rmsd(lig[i + k], lig[i]) for i in idx]))
        out[f"top_lig_centre_move_nm_lag{k}"] = float(np.linalg.norm(cen[k:] - cen[:-k], axis=1).mean())
        out[f"top_omega_changed_lag{k}"] = float(np.mean([len(set(om[i + k]) - set(om[i])) for i in idx]))
    # novelty against the K true starting frames (all walkers), every 10th top-view structure
    init = start_ligands(m)
    out["top_rmsd_to_nearest_start_max_nm"] = float(max(min(kabsch_rmsd(lig[i], x) for x in init) for i in range(0, n, 10)))
    out.update(excursions(z, perm, label, K, init, lig))
    out["top_dU_block100_means"] = [float(z["label_dU"][i:i + 100, top].mean()) for i in range(0, n, 100)]
    y = z["label_dU"][int(n * discard):, top]
    g = float(timeseries.statistical_inefficiency(y)) if y.std() > 0 else float(len(y))
    out.update({"top_dU_mean": float(y.mean()), "top_dU_sd": float(y.std(ddof=1)), "top_g_sweeps": g,
                "top_n_over_g": len(y) / g, "top_dU_se": float(y.std(ddof=1) * np.sqrt(g / len(y)))})
    out["label_dU_mean"] = [float(z["label_dU"][int(n * discard):, k].mean()) for k in range(K)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--starts", default="1,2,3,4")
    ap.add_argument("--system", default="CyclophilinD_ligand-8")
    ap.add_argument("--discard", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"discard": a.discard, "note": "top label = eta 1; equal model-call budget per ladder; no pass threshold",
           "protocols": {}}
    for tag in a.tags.split(","):
        rows = [ladder_metrics(os.path.join(HERE, "runs", a.system, f"ladder_S0_start{s}{tag}.npz"), a.discard)
                for s in map(int, a.starts.split(","))]
        mus, ses = np.array([r["top_dU_mean"] for r in rows]), np.array([r["top_dU_se"] for r in rows])
        res["protocols"][tag] = {"ladders": rows, "top_means": mus.tolist(), "spread_top_means": float(np.ptp(mus)),
                                 "median_top_se": float(np.median(ses))}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for tag, p in res["protocols"].items():
        print(f"== {tag}: eta=1 <dU> by ladder {np.round(p['top_means'], 2).tolist()} spread {p['spread_top_means']:.2f} "
              f"median SE {p['median_top_se']:.2f}")
        for r in p["ladders"]:
            print(f"  rep{r['start']}: pair acc {[None if x is None else round(x, 3) for x in r['pair_exchange_rate']]} | "
                  f"0->1 {r['oneway_bottom_to_top']} RT {r['round_trips_top_bottom_top']} fresh-at-top {r['top_arrivals_after_bottom_visit']} "
                  f"top changes {r['top_label_walker_changes']} | local acc by eta {np.round(r['local_acc_by_eta'], 3).tolist()} | "
                  f"top internal lag1/10/50 {r['top_lig_internal_rmsd_nm_lag1']:.4f}/{r['top_lig_internal_rmsd_nm_lag10']:.4f}/"
                  f"{r['top_lig_internal_rmsd_nm_lag50']:.4f} max-to-nearest-start {r['top_rmsd_to_nearest_start_max_nm']:.4f} | "
                  f"centre lag50 {r['top_lig_centre_move_nm_lag50']:.4f} | Omega lag50 {r['top_omega_changed_lag50']:.1f} | "
                  f"top dU {r['top_dU_mean']:+.2f}±{r['top_dU_se']:.2f} g {r['top_g_sweeps']:.1f} n/g {r['top_n_over_g']:.1f} | "
                  f"label means {np.round(r['label_dU_mean'], 1).tolist()}")
            print(f"        top dU per 100 sweeps {np.round(r['top_dU_block100_means'], 1).tolist()}; "
                  f"local-move novelty vs previous top {r['local_top_novelty_vs_previous_top_mean']}")
            for e in r["top_arrivals_after_bottom"]:
                print(f"        arrival sweep {e['sweep']} walker {e['walker']}: to nearest start {e['rmsd_to_nearest_start']:.3f}, "
                      f"to nearest previous top {e['rmsd_to_nearest_previous_top']:.3f}, bottom stay {e['bottom_stay_sweeps']} sweeps "
                      f"changed {e['change_during_bottom_stay']:.3f} nm")


if __name__ == "__main__":
    main()
