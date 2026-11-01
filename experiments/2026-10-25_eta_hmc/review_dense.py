"""Review of extended dense ladders (read-only; no model, no MD).

Per ladder: dH before / after the continuation point; per-label dU block means (every eta, 100-sweep blocks);
per 100-sweep block at eta=1: walker changes, local acceptance (by the eta used), within-block sd (common-stall check);
visits to the low region dU < LOW (any label, per walker): entries, exits, sweeps spent, labels held.

Run: python -B review_dense.py --tag _dense8_v1x1000 --out logs/review_dense8_v1x1000.json
"""

import argparse
import json
import os

import numpy as np

from eta_exchange import read_ladder

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
LOW = -90.0                                     # kJ/mol; the earlier ~-100 region (rep1 of _ex_v1x1000b)
BLOCK = 100


def review(path):
    m, z = read_ladder(path)
    n, K = int(m["completed_sweeps"]), len(m["etas"])
    cut = [c["after_sweep"] for c in m.get("continuations", [])]
    out = {"file": os.path.basename(path), "start": m["start_rep"], "continuations": m.get("continuations", [])}
    if cut:
        c = cut[0]
        out["dH_before"] = [float(z["dH"][:c].mean()), float(z["dH"][:c].std()), float(np.abs(z["dH"][:c]).max())]
        out["dH_after"] = [float(z["dH"][c:].mean()), float(z["dH"][c:].std()), float(np.abs(z["dH"][c:]).max())]
    blocks = range(0, n, BLOCK)
    out["label_block_means"] = [[float(z["label_dU"][b:b + BLOCK, k].mean()) for b in blocks] for k in range(K)]
    top = K - 1
    p = z["perm"][:, top]
    rows = []
    for b in blocks:
        sl = slice(b, b + BLOCK)
        used = z["walker_eta_local"][sl] == m["etas"][-1]
        rows.append({"sweeps": [b + 1, min(b + BLOCK, n)],
                     "top_walker_changes": int((p[sl][1:] != p[sl][:-1]).sum() + (b > 0 and p[b] != p[b - 1])),
                     "top_local_acc": float(z["acc"][sl][used].mean()),
                     "top_dU_sd": float(z["label_dU"][sl, top].std()),
                     "pair_acc": [float(z["ex_acc"][sl, k].sum() / max(1, z["ex_attempt"][sl, k].sum())) for k in range(K - 1)]})
    out["top_blocks"] = rows
    # low-region visits per walker
    label = np.empty_like(z["perm"])
    label[np.arange(n)[:, None], z["perm"]] = np.arange(K)[None, :]
    low = z["dU"] < LOW
    visits = []
    for w in range(K):
        s = 0
        while s < n:
            if low[s, w]:
                e = s
                while e + 1 < n and low[e + 1, w]:
                    e += 1
                visits.append({"walker": w, "enter_sweep": s + 1, "leave_sweep": e + 2 if e + 1 < n else None,
                               "sweeps": e - s + 1, "min_dU": float(z["dU"][s:e + 1, w].min()),
                               "labels_held": sorted(set(np.asarray(m["etas"])[label[s:e + 1, w]].tolist()))})
                s = e + 1
            else:
                s += 1
    out["low_region_visits"] = visits
    out["low_region_fraction_at_top"] = float((z["label_dU"][:, top] < LOW).mean())
    out["min_dU_any"] = float(z["dU"].min())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--starts", default="1,2,3,4")
    ap.add_argument("--system", default="CyclophilinD_ligand-8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = [review(os.path.join(HERE, "runs", a.system, f"ladder_S0_start{s}{a.tag}.npz")) for s in map(int, a.starts.split(","))]
    with open(a.out, "w") as f:
        json.dump({"low_threshold": LOW, "block_sweeps": BLOCK, "ladders": res}, f, indent=1)
    for r in res:
        print(f"== rep{r['start']}: dH before {np.round(r.get('dH_before', []), 2).tolist()} after {np.round(r.get('dH_after', []), 2).tolist()}; "
              f"min dU any {r['min_dU_any']:.1f}; fraction of eta=1 time below {LOW}: {r['low_region_fraction_at_top']:.3f}")
        for k, bm in enumerate(r["label_block_means"]):
            print(f"   label {k}: {np.round(bm, 1).tolist()}")
        print("   eta=1 per block: changes " + str([b['top_walker_changes'] for b in r['top_blocks']]) +
              " | local acc " + str([round(b['top_local_acc'], 3) for b in r['top_blocks']]) +
              " | sd " + str([round(b['top_dU_sd'], 1) for b in r['top_blocks']]))
        print("   pair acc first/last block: " + str([round(x, 2) for x in r['top_blocks'][0]['pair_acc']]) + " / " +
              str([round(x, 2) for x in r['top_blocks'][-1]['pair_acc']]))
        for v in r["low_region_visits"]:
            print(f"   low visit walker {v['walker']}: sweeps {v['enter_sweep']}->{v['leave_sweep']} ({v['sweeps']}), min {v['min_dU']:.1f}, labels {v['labels_held']}")


if __name__ == "__main__":
    main()
