"""Compare eta-chain proposal kernels per model call (read-only on schema-3 records; no model, no MD).

Per chain: acceptance / rejection runs / proposal dU change; actual movement per model call (ligand rigid-body +
internal, Omega water turnover, Omega-water displacement between full-box snapshots); dU time series
(statistical inefficiency in model calls, n/g); across-start spread of <dU>. No pass threshold.

Run: python -B compare_proposals.py --tags _nvarb_ms,_nvarb_L5 --eta 1.0 --out logs/compare_L20_L5_eta1.json
"""

import argparse
import json
import os

import numpy as np

from eta_records import read_record

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
LAGS = (1, 10, 50)


def kabsch_rmsd(a, b):
    """RMSD of a onto b after optimal rigid superposition (nm)."""
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((a0 @ r - b0) ** 2).sum(1).mean()))


def chain_metrics(path, discard):
    from pymbar import timeseries
    m, z = read_record(path)
    acc, dU, lig = z["acc"], z["dU"], z["lig_xyz"]
    ia = np.flatnonzero(acc)
    runs = np.diff(np.r_[-1, ia, len(acc)]) - 1
    out = {"file": os.path.basename(str(path)), "L": m["L"], "dt_fs": m["dt_fs"], "start": m["start_rep"],
           "model_calls": int(m["cycles"]) + 1, "acc_rate": float(acc.mean()), "n_acc": int(len(ia)),
           "last_acc_cycle": int(ia.max() + 1) if len(ia) else 0, "longest_reject_run": int(runs.max()),
           "proposal_dE_mean": float(z["dE"].mean()), "proposal_dE_sd": float(z["dE"].std(ddof=1)),
           "proposal_dH_mean": float(z["dH"].mean()), "seconds_per_cycle": float(m["seconds"]) / int(m["cycles"])}
    # movement of the chain state (rejections count as zero moves): per lag in model calls
    com = lig.mean(1)
    for k in LAGS:
        out[f"lig_com_move_nm_lag{k}"] = float(np.linalg.norm(com[k:] - com[:-k], axis=1).mean())
        idx = np.arange(0, len(lig) - k, max(1, k // 2))
        out[f"lig_internal_rmsd_nm_lag{k}"] = float(np.mean([kabsch_rmsd(lig[i + k], lig[i]) for i in idx]))
    om = z["omega_water_resid"]
    out["omega_waters_changed_per_call"] = float(np.mean([len(set(om[i]) - set(om[i - 1])) for i in range(1, len(om))]))
    out["omega_waters_changed_lag50"] = float(np.mean([len(set(om[i + 50]) - set(om[i])) for i in range(0, len(om) - 50, 25)]))
    # Omega-water oxygen displacement between consecutive full-box snapshots (every coords_every calls)
    L = np.diag(np.asarray(m["box_nm"]))
    si, nl = z["cluster_system_indices"], len(m["ligand_system_indices"])
    xc, X = z["xyz_cycles"], z["xyz"]
    disp = []
    for a in range(1, len(xc)):
        ox = si[xc[a - 1] - 1][nl::3]                     # Omega water oxygens of the earlier snapshot state
        d = X[a][ox] - X[a - 1][ox]
        d -= L * np.round(d / L)
        disp.append(np.linalg.norm(d, axis=1).mean())
    out["omega_O_move_nm_per_snapshot"] = float(np.mean(disp))
    out["snapshot_interval_calls"] = int(m["coords_every"])
    n0 = int(len(dU) * discard)
    y = dU[n0:]
    g = float(timeseries.statistical_inefficiency(y)) if y.std() > 0 else float(len(y))
    out.update({"dU_mean": float(y.mean()), "dU_sd": float(y.std(ddof=1)), "g_calls": g, "n_over_g": len(y) / g,
                "dU_se": float(y.std(ddof=1) * np.sqrt(g / len(y)))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--starts", default="1,2,3,4")
    ap.add_argument("--system", default="CyclophilinD_ligand-8")
    ap.add_argument("--discard", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"eta": a.eta, "discard": a.discard, "note": "per model call; no pass threshold", "protocols": {}}
    for tag in a.tags.split(","):
        rows = [chain_metrics(os.path.join(HERE, "runs", a.system, f"chain_S0_eta{a.eta:.3f}_start{s}{tag}.npz"), a.discard)
                for s in map(int, a.starts.split(","))]
        mus, ses = np.array([r["dU_mean"] for r in rows]), np.array([r["dU_se"] for r in rows])
        res["protocols"][tag] = {"chains": rows, "start_means": mus.tolist(), "spread_of_start_means": float(np.ptp(mus)),
                                 "median_se": float(np.median(ses))}
        keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
        res["protocols"][tag]["mean_over_starts"] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for tag, p in res["protocols"].items():
        print(f"== {tag}: start means {np.round(p['start_means'], 2).tolist()} spread {p['spread_of_start_means']:.2f} "
              f"median SE {p['median_se']:.2f}")
        for r in p["chains"]:
            print(f"  s{r['start']} L{r['L']}: acc {r['acc_rate']:.3f} n_acc {r['n_acc']} last {r['last_acc_cycle']} "
                  f"maxrej {r['longest_reject_run']} | dE {r['proposal_dE_mean']:+.2f}±{r['proposal_dE_sd']:.2f} | "
                  f"COM lag1/10/50 {r['lig_com_move_nm_lag1']:.4f}/{r['lig_com_move_nm_lag10']:.4f}/{r['lig_com_move_nm_lag50']:.4f} "
                  f"int lag1/10/50 {r['lig_internal_rmsd_nm_lag1']:.4f}/{r['lig_internal_rmsd_nm_lag10']:.4f}/{r['lig_internal_rmsd_nm_lag50']:.4f} | "
                  f"Omega chg/call {r['omega_waters_changed_per_call']:.2f} lag50 {r['omega_waters_changed_lag50']:.1f} "
                  f"O move/{r['snapshot_interval_calls']} calls {r['omega_O_move_nm_per_snapshot']:.3f} | "
                  f"dU {r['dU_mean']:+.2f}±{r['dU_se']:.2f} g {r['g_calls']:.1f} n/g {r['n_over_g']:.1f} | {r['seconds_per_cycle']:.2f} s/cyc")


if __name__ == "__main__":
    main()
