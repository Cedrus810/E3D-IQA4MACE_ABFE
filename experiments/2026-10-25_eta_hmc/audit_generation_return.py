"""Offline generation/return audit; NumPy only, no model, MD or reweighting.

Local movement is attributed using walker_eta_local, before label exchange.
Accepted-state versions track recycling, not independent conformational samples.
All first-passage durations are finite-record diagnostics with censoring reported.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from compare_ladders import kabsch_rmsd
from eta_exchange import read_ladder, same_contract_value

HERE = Path(__file__).resolve().parent


def summary(values):
    a = np.asarray(values, float)
    if not a.size:
        return {"n": 0, "mean": None, "q0_q50_q90_q100": None}
    return {"n": int(a.size), "mean": float(a.mean()),
            "q0_q50_q90_q100": np.quantile(a, [0, .5, .9, 1]).tolist()}


def lineage(m, z):
    n, K = z["perm"].shape
    labels = np.empty((n, K), int)
    labels[np.arange(n)[:, None], z["perm"]] = np.arange(K)
    versions = np.zeros(K, int)
    last_origin = np.full(K, np.nan)
    seen_top = set()
    last_bottom = np.full(K, -1, int)
    # Start at first observed bottom visit, count only the first subsequent top arrival.
    pending = {}
    passages, top_rows = [], []
    for s in range(n):
        for w in range(K):
            if z["acc"][s, w]:
                versions[w] += 1
                last_origin[w] = z["walker_eta_local"][s, w]
            if labels[s, w] == 0:
                last_bottom[w] = s
                pending.setdefault(w, s)
            if labels[s, w] == K - 1 and w in pending:
                first = pending.pop(w)
                passages.append({"walker": w, "start_sweep": first + 1, "arrival_sweep": s + 1,
                                 "duration_sweeps": s - first})
        w = int(z["perm"][s, -1])
        key = (w, int(versions[w]))
        changed = s > 0 and w != int(z["perm"][s - 1, -1])
        top_rows.append({"sweep": s + 1, "walker_changed": changed,
                         "state_version_new_to_top": key not in seen_top,
                         "last_accepted_eta": float(last_origin[w]) if versions[w] else None,
                         "sweeps_since_bottom": int(s - last_bottom[w]) if last_bottom[w] >= 0 else None})
        seen_top.add(key)
    censored = [{"walker": w, "start_sweep": first + 1, "observed_duration_sweeps": n - 1 - first}
                for w, first in sorted(pending.items())]
    return labels, passages, censored, top_rows


def analyze(m, z):
    n, K = z["perm"].shape
    labels, passages, censored, top_rows = lineage(m, z)
    rmsd = np.full((n, K), np.nan)
    omega_changed = np.full((n, K), np.nan)
    for s in range(1, n):
        for w in range(K):
            rmsd[s, w] = kabsch_rmsd(z["lig_xyz"][s, w], z["lig_xyz"][s - 1, w])
            omega_changed[s, w] = len(set(z["omega_water_resid"][s, w]) - set(z["omega_water_resid"][s - 1, w]))
    windows = []
    for lo, hi in ((0, n // 2), (n // 2, n)):
        sl = slice(lo, hi)
        local = []
        for eta in m["etas"]:
            used = z["walker_eta_local"][sl] == eta
            ok = used & np.isfinite(rmsd[sl])
            local.append({"eta": eta, "attempts": int(used.sum()),
                          "accepted": int(z["acc"][sl][used].sum()),
                          "accepted_step_internal_rmsd_nm": summary(rmsd[sl][ok & z["acc"][sl].astype(bool)]),
                          "per_attempt_internal_rmsd_nm": summary(rmsd[sl][ok]),
                          "per_attempt_omega_changes": summary(omega_changed[sl][ok]),
                          "proposal_deltaU_kJ_mol": summary(z["dE"][sl][used])})
        rows = top_rows[lo:hi]
        swaps = [r for r in rows if r["walker_changed"]]
        arrivals = [p for p in passages if lo < p["arrival_sweep"] <= hi]
        windows.append({"sweeps": [lo + 1, hi], "local_generation": local,
                        "pair_attempts": z["ex_attempt"][sl].sum(0).astype(int).tolist(),
                        "pair_accepts": z["ex_acc"][sl].sum(0).astype(int).tolist(),
                        "pair_rates": (z["ex_acc"][sl].sum(0) / z["ex_attempt"][sl].sum(0)).tolist(),
                        "bottom_to_top_arrivals": len(arrivals),
                        "arrival_duration_sweeps": summary([p["duration_sweeps"] for p in arrivals]),
                        "top_walker_changes": len(swaps),
                        "top_changes_with_unseen_accepted_state_version": sum(r["state_version_new_to_top"] for r in swaps),
                        "top_new_versions_by_last_accepted_eta": {
                            str(e): sum(r["state_version_new_to_top"] and r["last_accepted_eta"] == e for r in swaps)
                            for e in m["etas"]},
                        "top_time_since_last_bottom_sweeps": summary([r["sweeps_since_bottom"] for r in rows if r["sweeps_since_bottom"] is not None]),
                        "top_sweeps_with_no_observed_bottom_ancestry": sum(r["sweeps_since_bottom"] is None for r in rows),
                        "walker_label_ranges": [{"walker": w,
                            "etas_visited": np.asarray(m["etas"])[np.unique(labels[sl, w])].tolist()}
                            for w in range(K)]})
    return {"start": m["start_rep"], "etas": m["etas"], "complete": m["complete"],
            "sweeps": n, "nominal_calls_excluding_rechecks": m["model_calls_ladder"],
            "windows": windows, "bottom_to_top_passages": passages, "right_censored_passages": censored,
            "min_deltaU_any": float(z["dU"].min()), "continuations": m.get("continuations", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_dense8_v1x1000")
    ap.add_argument("--system", default="CyclophilinD_ligand-8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    keys = ("head", "backbone", "E_construction", "T_K", "dtype", "box_nm",
            "ligand_system_indices", "groups", "dummy_mask", "n_water", "state", "system_definition")
    old_m, old_z = read_ladder(HERE / "runs" / a.system / "ladder_S0_start1_ex_v1x1000b.npz")
    historical = {"file": "ladder_S0_start1_ex_v1x1000b.npz", "min_deltaU_any": float(old_z["dU"].min()),
                  "start_frames": old_m["start_frames"], "stored_target_contract_comparison": []}
    del old_z
    for rep in range(1, 5):
        path = HERE / "runs" / a.system / f"ladder_S0_start{rep}{a.tag}.npz"
        m, z = read_ladder(path)
        row = analyze(m, z)
        row["file"] = path.name
        # Missing metadata is not treated as agreement.
        historical["stored_target_contract_comparison"].append({"start": rep,
            "compared_keys": [k for k in keys if k in old_m and k in m],
            "missing_keys": [k for k in keys if k not in old_m or k not in m],
            "different_keys": [k for k in keys if k in old_m and k in m and not same_contract_value(k, old_m[k], m[k])]})
        rows.append(row)
        for w in row["windows"]:
            zero = w["local_generation"][0]
            print(f"rep{rep} {w['sweeps']}: bottom local acc {zero['accepted']}/{zero['attempts']}; "
                  f"bottom RMSD/attempt {zero['per_attempt_internal_rmsd_nm']['mean']:.4f}; "
                  f"arrivals {w['bottom_to_top_arrivals']}; edges {np.round(w['pair_rates'], 3).tolist()}; "
                  f"top switches/new versions {w['top_walker_changes']}/{w['top_changes_with_unseen_accepted_state_version']}; "
                  f"top bottom-age median {w['top_time_since_last_bottom_sweeps']['q0_q50_q90_q100']}")
    result = {"note": "diagnostics only; state versions are not independent samples; movement excludes first sweep; "
                       "passages keep pre-window ancestry; censored durations are not failures or a rate estimate; "
                       "stored metadata agreement is not independent re-evaluation of historical deep energies",
              "historical_deep": historical, "ladders": rows}
    Path(a.out).write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
