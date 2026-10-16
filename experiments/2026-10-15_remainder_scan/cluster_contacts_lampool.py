"""Shortest ligand-water contacts in the operating head's training clusters (v3 S13.41).

Run from the repo root: python experiments/2026-10-15_remainder_scan/cluster_contacts_lampool.py

The exact training set of the S13.37 head: solvent_clusters on
lampool400_solvent.npz, n = 10,000, size 2-24, rng default_rng(1),
ligand_frac 0.25 (run_polar.sh / test_joint.py). Per ligand cluster, the
shortest contact per (ligand element, water element); quantiles over clusters.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

from decomp.clusters import solvent_clusters  # noqa: E402

FR = "experiments/2026-10-13_intermediate_lambda/lampool400_solvent.npz"
pos, num, frag = solvent_clusters(FR, n=10000, rng=np.random.default_rng(1), size=(2, 24), ligand_frac=0.25)
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
mins = {}
nl = 0
for p, z, f in zip(pos, num, frag):
    if (f == 0).sum() != 41:
        continue
    nl += 1
    L, W = f == 0, f == 1
    D = np.linalg.norm(p[L][:, None] - p[W][None], axis=-1)
    zl, zw = z[L], z[W]
    for a in (6, 7, 8, 1):
        for b in (8, 1):
            mins.setdefault(f"{SYM[a]}-{SYM[b]}w", []).append(D[np.ix_(zl == a, zw == b)].min())
print(f"{FR}: {nl} ligand clusters; per-cluster shortest contact min / p0.1 / p1 / p5 / median (A)")
for k, v in mins.items():
    v = np.array(v)
    print(f"  {k:5s} " + "  ".join(f"{x:.2f}" for x in (v.min(), *np.percentile(v, [0.1, 1, 5, 50]))))
