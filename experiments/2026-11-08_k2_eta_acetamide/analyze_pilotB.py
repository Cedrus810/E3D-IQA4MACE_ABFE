"""Pilot B1 (K2 exchange ladders) next to pilot A (independent chains), same query budget, discard 0.2.
Files opened by explicit path (the NFS directory listing on 181 can be stale). Report only."""

import json
import os
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ETA = Path("/home/ruigengji/E3D/experiments/2026-10-25_eta_hmc")
sys.path.insert(0, str(ETA))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import k2_exchange as kx  # noqa: E402
from k2_report import report_ladders  # noqa: E402

MM_REF = -11.56     # <dU_K2> on 2000 independent MM S0 frames (head_calibration step 2, k2_decision_checks.json)


def round_trips(labels, top):
    """Completed bottom->top->bottom trips of one walker's label sequence."""
    n, state = 0, None
    for lab in labels:
        if lab == 0:
            if state == "up":
                n += 1
            state = "down"
        elif lab == top and state == "down":
            state = "up"
    return n


def main(tag="B1", preps=(1, 2, 3, 4)):
    from pymbar import timeseries
    files = [ETA / f"runs/acetamide/k2ladder_S0_prep{p}_pilot{tag}.npz" for p in preps]
    out = {"discard_fraction": 0.2, "MM_reference_dU_K2_eta0": MM_REF, "ladders": []}
    for p, f in zip(preps, files):
        m, z = kx.read_ladder(f)
        n = int(m["completed_sweeps"])
        n0 = int(0.2 * n)
        etas = m["etas"]
        K = len(etas)
        lab = np.argsort(z["perm"], axis=1)                    # label of each walker after each sweep
        local = np.asarray(z["walker_eta_local"])[n0:]
        acc = np.asarray(z["acc"], bool)[n0:]
        per = []
        for k, e in enumerate(etas):
            du = z["label_dU"][n0:, k]
            g = float(timeseries.statistical_inefficiency(du)) if du.std() > 0 else 1.0
            here = local == e
            per.append({"eta": e, "mean_dU": float(du.mean()), "sd_dU": float(du.std(ddof=1)), "g": g,
                        "n_eff": len(du) / g, "local_acc": float(acc[here].mean()),
                        "distinct_walkers_post_discard": int(len(np.unique(z["perm"][n0:, k])))})
        rerun = HERE / f"logs/pilotB/{tag}rerun_prep{p}.log"
        log = (rerun if rerun.exists() else HERE / f"logs/pilotB/{tag}_prep{p}.log").read_text()
        sec = re.search(r"([\d.]+) s per K2 query", log)
        out["ladders"].append({
            "prep": p, "complete": m["complete"], "sweeps": n, "continuations": m["continuations"],
            "device": m["checkpoint_device"], "s_per_query": float(sec.group(1)) if sec else None,
            "exchange_acc": (z["ex_acc"].sum(0) / np.maximum(z["ex_attempt"].sum(0), 1)).tolist(),
            "round_trips_per_walker": [round_trips(lab[:, w], K - 1) for w in range(K)],
            "per_label": per})
    out[f"report_{tag}"] = report_ladders(files, discard=0.2, bootstrap=200, seed=1)
    a = json.loads((HERE / "report_pilotA.json").read_text())
    out["pilotA_per_start_a_BAR"] = ([s["a_BAR"] for s in a["report_prep2_4"]["per_start"]]
                                     + [s["a_BAR"] for s in a["report_prep1_without_eta0.3"]["per_start"]])
    out["pilotA_eta0_mean_dU"] = [c["mean_dU"] for c in a["chains"] if c["eta"] == 0.0]
    (HERE / f"report_pilot{tag}{'' if len(preps) == 4 else '_partial'}.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "B1"
    preps = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (1, 2, 3, 4)
    main(tag, preps)
