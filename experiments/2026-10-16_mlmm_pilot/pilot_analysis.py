"""Pilot ladder diagnostics (v3 S8 route step 2): where the windows are short.

Run: python pilot_analysis.py [run_dir]   (default atenolol_n20)

From state.npz (swap counts) and records.json (u_kn, dU/dq per segment):
  - swap acceptance per neighbour pair
  - MBAR nearest-neighbour overlap (pymbar, all samples, no subsampling)
  - per neighbour pair: the reduced work w = u_j(x) - u_i(x) in both directions
    (mean, sd); sd >> 1 kT means poor overlap
  - per segment: <dU/dq> and its SE (naive) at each state, the TI integral, and
    each interval's share of the TI variance
"""

import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp.fe import mbar_free_energy  # noqa: E402
from decomp.mlmm_hremd import SEGMENTS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "atenolol_n20")
KT = 0.008314462618 * 298.15
z = np.load(f"{run}/state.npz", allow_pickle=True)
rec = json.load(open(f"{run}/records.json"))
qs = z["qs"]
K = len(qs)
u = np.array(rec["u"])                          # [n, conf/state k, state l], reduced
n = len(u)
acc = z["n_acc"] / np.maximum(z["n_att"], 1)
u_kn = u.transpose(1, 0, 2).reshape(K * n, K).T
m = mbar_free_energy(u_kn, [n] * K, kT=KT)
ov = m["overlap"]
print(f"{run}: {K} states, {n} samples per state\n")
print("pair (q_i - q_j)        swaps     NN overlap   w_fwd mean/sd (kT)   w_rev mean/sd (kT)")
for i in range(K - 1):
    wf = u[:, i, i + 1] - u[:, i, i]
    wr = u[:, i + 1, i] - u[:, i + 1, i + 1]
    print(f"{qs[i]:6.3f} - {qs[i + 1]:6.3f}   {int(z['n_acc'][i]):2d}/{int(z['n_att'][i]):2d} {acc[i]:.2f}   "
          f"{ov[i, i + 1]:.3f}        {wf.mean():+8.2f} / {wf.std():6.2f}     {wr.mean():+8.2f} / {wr.std():6.2f}")

print("\nsegment  q        <dU/dq> kJ/mol   SE      | interval share of TI variance")
for name, a, b in SEGMENTS:
    ks = [k for k, q in enumerate(qs) if a - 1e-9 <= q <= b + 1e-9]
    y = np.array([[rec["dudq"][t][k][name] for k in ks] for t in range(n)])
    mu, se = y.mean(0), y.std(0, ddof=1) / np.sqrt(n)
    dq = np.diff(qs[ks])
    var_int = (dq / 2) ** 2 * (se[:-1] ** 2 + se[1:] ** 2)
    share = var_int / var_int.sum() if var_int.sum() > 0 else var_int
    print(f"{name}: dG = {np.trapezoid(mu, qs[ks]):+.2f} kJ/mol, naive SE {np.sqrt(var_int.sum()):.2f}")
    for j, k in enumerate(ks):
        tail = f"   | {share[j]:.2f} to next" if j < len(ks) - 1 else ""
        print(f"   {qs[k]:6.3f}  {mu[j]:+12.2f}  {se[j]:7.2f}{tail}")
r = json.load(open(f"{run}/result.json"))
print(f"\nTI total {r['dG_TI']:+.2f}; MBAR {r.get('dG_MBAR', float('nan')):+.2f} kJ/mol; "
      f"per-atom sum {r.get('per_atom_sum', float('nan')):+.2f}, cut {r.get('dG_cut', float('nan')):+.2f}, "
      f"closure {r.get('closure', float('nan')):+.3f}")
pa = r.get("per_atom", {})
for key, v in pa.items():
    v = np.array(v)
    print(f"  per-atom {key:8s} sum {v.sum():+8.2f}  min {v.min():+7.2f}  max {v.max():+7.2f} kJ/mol")
