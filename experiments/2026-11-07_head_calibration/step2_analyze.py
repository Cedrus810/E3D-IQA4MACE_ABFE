"""Summarize step 2: r = C_ML - C_bb on 10 independent acetamide S0 MM preparations (N=50). Report only."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RT = 0.008314462618 * 298.15
preps = range(1, 11)
D = {p: json.loads((HERE / f"mm/prep{p}_S0_r_s5.json").read_text())["rows"] for p in preps}
out = {"units": "kJ/mol", "RT": RT, "frame_spacing_ps": 10, "per_prep": {}}

def acf1(x):
    x = x - x.mean(); return float((x[:-1] * x[1:]).sum() / (x * x).sum())

def block_se(x, nb=10):
    b = np.array_split(x, nb); m = np.array([v.mean() for v in b]); return float(m.std(ddof=1) / np.sqrt(nb))

print(f"{'prep':>4} {'n':>4} {'mean r':>8} {'sd r':>6} {'blkSE':>6} {'acf1':>6} {'corr(r,bb-MM)':>13} {'corr(r,Cbb)':>11} {'mean bb-MM':>10} {'mean dU':>8}")
allr, allbm, allcbb, allp = [], [], [], []
for p in preps:
    rows = sorted(D[p], key=lambda r: r["frame"])
    r = np.array([x["r"] for x in rows]); bm = np.array([x["bb_minus_MM"] for x in rows])
    cbb = np.array([x["C_bb"] for x in rows]); du = np.array([x["dU"] for x in rows])
    s = {"n": len(r), "mean_r": float(r.mean()), "sd_r": float(r.std(ddof=1)), "block_SE_r": block_se(r),
         "acf_lag10ps": acf1(r), "corr_r_bbMM": float(np.corrcoef(r, bm)[0, 1]), "corr_r_Cbb": float(np.corrcoef(r, cbb)[0, 1]),
         "mean_bb_minus_MM": float(bm.mean()), "sd_bb_minus_MM": float(bm.std(ddof=1)), "mean_dU": float(du.mean()), "sd_dU": float(du.std(ddof=1)),
         "first_half_mean_r": float(r[:100].mean()), "second_half_mean_r": float(r[100:].mean())}
    out["per_prep"][p] = s
    print(f"{p:4d} {len(r):4d} {s['mean_r']:8.2f} {s['sd_r']:6.2f} {s['block_SE_r']:6.2f} {s['acf_lag10ps']:6.2f} {s['corr_r_bbMM']:13.2f} {s['corr_r_Cbb']:11.2f} {s['mean_bb_minus_MM']:10.2f} {s['mean_dU']:8.2f}")
    allr += list(r); allbm += list(bm); allcbb += list(cbb); allp += [p] * len(r)
r, bm, cbb, P = map(np.array, (allr, allbm, allcbb, allp))
means = np.array([out["per_prep"][p]["mean_r"] for p in preps])
pool = {"n": len(r), "mean_r": float(r.mean()), "sd_r": float(r.std(ddof=1)),
        "SE_from_prep_means": float(means.std(ddof=1) / np.sqrt(len(means))), "sd_of_prep_means": float(means.std(ddof=1)),
        "min_r": float(r.min()), "max_r": float(r.max()), "q05_q50_q95": np.percentile(r, [5, 50, 95]).tolist(),
        "corr_r_bbMM": float(np.corrcoef(r, bm)[0, 1]), "corr_r_Cbb": float(np.corrcoef(r, cbb)[0, 1]),
        "sd_bb_minus_MM": float(bm.std(ddof=1)), "slope_r_on_bbMM": float(np.polyfit(bm, r, 1)[0]),
        "slope_r_on_Cbb": float(np.polyfit(cbb, r, 1)[0])}
out["pooled"] = pool
cal, val = P <= 5, P > 5
c = float(r[cal].mean()); res = r[val] - c
out["K1_split"] = {"calibration_preps": "1-5", "validation_preps": "6-10", "c": c,
                   "validation_mean_residual": float(res.mean()), "validation_rmse_after_shift": float(np.sqrt((res ** 2).mean())),
                   "validation_sd_r": float(r[val].std(ddof=1)), "validation_rmse_no_shift": float(np.sqrt((r[val] ** 2).mean()))}
print(json.dumps({"pooled": pool, "K1_split": out["K1_split"]}, indent=1))
(HERE / "step2_summary.json").write_text(json.dumps(out, indent=2) + "\n")
