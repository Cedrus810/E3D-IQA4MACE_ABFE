"""Paired per-frame remainder differences, lampool heads vs the S13.35 operating head.

Run: python paired.py
Reads the per-frame tables of the test_scale.py logs (r = truth - sumD per frame
and n_solv; the same 20 test frames and carves for every head), and prints, per
test set and n, mean(r_new - r_old) +- SE of the paired difference (sd/sqrt(20)).
"""

import json

import numpy as np

E3D = "/home/ruigengji/E3D"
L = f"{E3D}/runs/polar-M/logs"
X = f"{E3D}/experiments/2026-10-13_intermediate_lambda"
OLD = {"thermal_test": f"{L}/scale_N_thermal_es_thermal_test_solvent.log",
       "lam0.5": f"{X}/scale_N_es_thermal200_lam0.5.log",
       "lam0.0": f"{X}/scale_N_es_thermal200_lam0.0.log",
       "frames_test": f"{L}/scale_N_thermal_es_frames_test_solvent.log"}
NEW = {c: {"thermal_test": f"{L}/scale_N_es_lampool_c{c}_thermal_test_solvent.log",
           "lam0.5": f"{L}/scale_N_es_lampool_c{c}_lam0.5_test_solvent.log",
           "lam0.0": f"{L}/scale_N_es_lampool_c{c}_lam0.0_test_solvent.log",
           "frames_test": f"{L}/scale_N_es_lampool_c{c}_frames_test_solvent.log"}
       for c in (20000, 10000)}


def table(path):
    """{(frame, n): r} from the per-frame rows (9 numeric columns)."""
    r = {}
    for line in open(path):
        w = line.split()
        if len(w) != 9:
            continue
        try:
            v = [float(x) for x in w]
        except ValueError:
            continue
        r[(int(v[0]), int(v[1]))] = v[4] - v[5]
    return r


out = {}
for c, sets in NEW.items():
    print(f"== lampool c{c} minus operating head (eV): mean diff +- paired SE  [r_old mean, r_new mean]")
    for s, p in sets.items():
        a, b = table(OLD[s]), table(p)
        assert a.keys() == b.keys(), (s, len(a), len(b))
        for n in sorted({k[1] for k in a}):
            ks = sorted(k for k in a if k[1] == n)
            ro = np.array([a[k] for k in ks]); rn = np.array([b[k] for k in ks])
            d = rn - ro
            se = d.std(ddof=1) / np.sqrt(len(d))
            out.setdefault(str(c), {}).setdefault(s, {})[n] = dict(
                old=float(ro.mean()), new=float(rn.mean()), diff=float(d.mean()), se=float(se))
            print(f"  {s:13s} n={n:3d}  {d.mean():+.3f} +- {se:.3f}  ({d.mean()/se:+5.1f} SE)"
                  f"  [{ro.mean():+.3f}, {rn.mean():+.3f}]")
print("== lampool c20000 minus c10000 (eV): mean diff +- paired SE")
for s in NEW[20000]:
    a, b = table(NEW[10000][s]), table(NEW[20000][s])
    for n in sorted({k[1] for k in a}):
        ks = sorted(k for k in a if k[1] == n)
        d = np.array([b[k] - a[k] for k in ks])
        se = d.std(ddof=1) / np.sqrt(len(d))
        out.setdefault("20000-10000", {}).setdefault(s, {})[n] = dict(diff=float(d.mean()), se=float(se))
        print(f"  {s:13s} n={n:3d}  {d.mean():+.3f} +- {se:.3f}  ({d.mean()/se:+5.1f} SE)")
json.dump(out, open(f"{E3D}/experiments/2026-10-14_lampool_result/paired.json", "w"), indent=1)
