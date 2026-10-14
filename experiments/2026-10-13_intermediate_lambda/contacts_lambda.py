"""Ligand-water shortest contacts at lambda_elec = 1, 0.5, 0 (seed-23 test runs).

Run: python contacts_lambda.py
lambda 1 = ../2026-10-12_thermal_frames/thermal_test_solvent_contacts.npz.
Per element pair, quantiles over the 0.5-ps snapshots of one 950-ps run each,
and the fraction of snapshots whose contact is shorter than the shortest one in
the operating head's ligand clusters (thermal200, cluster_contacts.log).
"""

import json

import numpy as np

E3D = "/home/ruigengji/E3D"
X = f"{E3D}/experiments/2026-10-13_intermediate_lambda"
runs = {"1.0": f"{E3D}/experiments/2026-10-12_thermal_frames/thermal_test_solvent_contacts.npz",
        "0.5": f"{X}/lam0.5_test_solvent_contacts.npz",
        "0.0": f"{X}/lam0.0_test_solvent_contacts.npz"}
# per-cluster shortest contact, min over the 2,474 ligand clusters of thermal200
CLU = {"C-Ow": 2.69, "C-Hw": 1.77, "N-Ow": 2.62, "N-Hw": 1.69, "O-Ow": 2.46,
       "O-Hw": 1.54, "H-Ow": 1.66, "H-Hw": 1.46}

C = {k: np.load(v, allow_pickle=True) for k, v in runs.items()}
pairs = [str(p) for p in C["1.0"]["pairs"]]
print("shortest ligand-water contact per snapshot (A): p1 / median; "
      "'<clu' = fraction of snapshots below the thermal200 cluster minimum")
print(f"{'pair':6s}" + "".join(f" | lambda_elec {k:>3s}       " for k in runs))
out = {}
for k, p in enumerate(pairs):
    row = f"{p:6s}"
    for lam, c in C.items():
        x = c["contacts"][:, k]
        below = float((x < CLU[p]).mean())
        out.setdefault(p, {})[lam] = dict(p1=float(np.percentile(x, 1)),
                                         median=float(np.median(x)), below_clusters=below)
        row += f" | {np.percentile(x, 1):5.2f} / {np.median(x):5.2f}  {100*below:4.1f}%"
    print(row)
json.dump(out, open(f"{X}/contacts_lambda.json", "w"), indent=1)
