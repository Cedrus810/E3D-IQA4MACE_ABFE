"""Read selected saved arrays; no model, MD, GPU, or production-data mutation.

Run from any directory with NumPy. Outputs only this review's JSON report.
This is a provenance/statistics audit, not an equilibrium acceptance test.
"""
from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ETA = ROOT / "experiments/2026-10-25_eta_hmc"
RUNS = ETA / "runs/CyclophilinD_ligand-8"


def inspect(rep):
    name = f"ladder_S0_start{rep}_ex_v1x1000.npz"
    path = RUNS / name
    log = ETA / "logs" / f"ladder_ex_v1x1000_start{rep}.log"
    row = {"rep": rep, "file": str(path.relative_to(ROOT)), "file_exists": path.exists()}
    if not path.exists():
        row["log_last_lines"] = log.read_text().splitlines()[-3:] if log.exists() else []
        return row
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        d, p, atoms = z["dU"], z["perm"], z["dU_atom"]
        label = z["label_dU"]
        n = len(label)
        top_mask = z["walker_eta_local"][-500:] == meta["etas"][-1]
        row.update({"complete": meta["complete"], "completed_sweeps": meta["completed_sweeps"],
                    "etas": meta["etas"], "head": meta["head"],
                    "model_calls_recorded": meta["model_calls_ladder"],
                    "label_mapping_max_abs_error": float(np.max(np.abs(label - d[np.arange(n)[:, None], p]))),
                    "atom_sum_max_abs_error": float(np.max(np.abs(atoms.sum(-1) - d))),
                    "eta1_block100_means": [float(label[i:i+100, -1].mean()) for i in range(0, n, 100)],
                    "eta1_last500_mean": float(label[-500:, -1].mean()),
                    "eta1_last500_local_accepted": int(z["acc"][-500:][top_mask].sum()),
                    "eta1_last500_local_attempted": int(top_mask.sum()),
                    "eta1_last500_distinct_scalar_energies": int(len(np.unique(label[-500:, -1]))),
                    "all_labels_last500_mean": label[-500:].mean(0).tolist(),
                    "pair_exchange_rates": (z["ex_acc"].sum(0) / z["ex_attempt"].sum(0)).tolist()})
        with np.load(RUNS / f"ladder_S0_start{rep}_ex_v1.npz", allow_pickle=False) as old:
            row["source_500_prefix_equal"] = {
                key: bool(np.array_equal(z[key][:500], old[key]))
                for key in ("dU", "label_dU", "perm", "acc", "dU_atom")}
    return row


def main():
    rows = [inspect(rep) for rep in (1, 2, 3, 4)]
    means = [r["eta1_last500_mean"] for r in rows if r.get("complete")]
    result = {"review_date_JST": "2026-10-30", "scope": "saved files only, not live scheduler status",
              "interpretation": "block means and spread are diagnostics, not a_j or equilibrium evidence",
              "cost_note": "recorded model-call count excludes resume energy rechecks; account separately",
              "ladders": rows, "completed_ladders": len(means),
              "spread_eta1_last500_complete_ladders": float(np.ptp(means)) if means else None}
    out = Path(__file__).with_name("saved_ladder_audit.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
