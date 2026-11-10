"""Pilot A summary: k2_report on prep2-4 (complete 11-point grid) and prep1 without eta=0.3 (supplement),
plus per-chain acceptance, statistical inefficiency g of dU, ligand RMS move per query, and adjacent-eta
work sd. Files are opened by explicit path (the NFS directory listing on 181 was stale). Report only."""

import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ETA = Path("/home/ruigengji/E3D/experiments/2026-10-25_eta_hmc")
sys.path.insert(0, str(ETA))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import k2_records as kr  # noqa: E402
from k2_report import report  # noqa: E402

ETAS = [round(0.1 * i, 1) for i in range(11)]
RT = 0.0083144626 * 298.15


def path(e, p):
    return ETA / f"runs/acetamide/k2chain_S0_eta{e:.3f}_prep{p}_pilotA.npz"


def main():
    from pymbar import timeseries
    out = {"discard_fraction": 0.2, "RT": RT, "chains": []}
    data = {}
    for p in (1, 2, 3, 4):
        for e in ETAS:
            f = path(e, p)
            if not f.exists():
                continue
            m, z = kr.read_record(f)
            n0 = int(0.2 * len(z["dU"]))
            du = z["dU"][n0:]
            lig = z["lig_xyz"][n0:].astype(np.float64)
            move = np.sqrt(((lig[1:] - lig[:-1]) ** 2).sum(-1).mean(-1)).mean()
            g = float(timeseries.statistical_inefficiency(du)) if du.std() > 0 else 1.0
            data[e, p] = du
            out["chains"].append({"eta": e, "prep": p, "acc": float(z["acc"][n0:].mean()), "mean_dU": float(du.mean()),
                                  "sd_dU": float(du.std(ddof=1)), "g_dU": g, "n_eff": len(du) / g,
                                  "mean_q": float(z["q"][n0:].mean()), "sd_q": float(z["q"][n0:].std(ddof=1)),
                                  "lig_rms_move_nm_per_query": float(move),
                                  "first_half_mean_dU": float(du[:len(du) // 2].mean()),
                                  "second_half_mean_dU": float(du[len(du) // 2:].mean())})
    out["adjacent_work_sd_RT"] = []
    for e0, e1 in zip(ETAS[:-1], ETAS[1:]):
        for p in (2, 3, 4):
            out["adjacent_work_sd_RT"].append({"eta0": e0, "eta1": e1, "prep": p,
                                               "sd_wF": float((e1 - e0) * data[e0, p].std() / RT),
                                               "sd_wR": float((e1 - e0) * data[e1, p].std() / RT)})
    main_files = [path(e, p) for p in (2, 3, 4) for e in ETAS]
    out["report_prep2_4"] = report(main_files, ETAS, ["prep2", "prep3", "prep4"], discard=0.2, bootstrap=200, seed=1)
    eta_p1 = [e for e in ETAS if e != 0.3]
    out["report_prep1_without_eta0.3"] = report([path(e, 1) for e in eta_p1], eta_p1, ["prep1"], discard=0.2,
                                                bootstrap=200, seed=2)
    (HERE / "report_pilotA.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
