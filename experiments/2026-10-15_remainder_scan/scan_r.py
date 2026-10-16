"""Remainder r(d) along a single-water approach, operating head (v3 S13.41).

Run from the repo root on a GPU: python experiments/2026-10-15_remainder_scan/scan_r.py

E7 at q = 2.0 (lambda = 0+, wall on) went NaN: an inner water fell onto a
ligand C while the head's coupling sum c grew to +3,145 kJ/mol. At
lambda = 0+, U = E_bb(L) + E_bb(E) + r, with r = truth - sum c, so r is the
only ML ligand-water interaction left, and -dr/dd pulls a water in when r
falls on approach. This measures r(d) on the S13.30 geometry: one water
moved rigidly toward one ligand atom along its own approach direction, the
probe atom (water O or H) from 3.5 to 0.7 A, frames_test_solvent.npz frame 0,
float64, 4 ligand atoms per element (seed 0) as in S13.30.

  truth(d) = E_bb(L + w) - E_bb(L) - E_bb(w)
  c(d)     = sum_i c_i of the anchored U (sumD + explicit L/E Coulomb, D6)
  r(d)     = truth - c

Reported per scan, against d_min (the closest L-w distance at that step):
the depth of r below its value at the start, the steepest inward pull
max(dr/dd) (kJ/mol/A), and where r first falls more than 1 kT below its
start. Saved to scan_r.json.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

import decomp  # noqa: F401,E402
from decomp.openmm_bridge import AnchoredForce  # noqa: E402
from test_anchored import e_bb, model64  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
CKPT = "runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
EV, KT = 96.48533212, 2.479                         # kJ/mol per eV, kT at 298 K in kJ/mol
model = model64(CKPT)
bb = model.backbone

d = np.load("frames_test_solvent.npz", allow_pickle=True)
P, Z, res, rn, box = d["positions"][0], d["numbers"], d["resids"], d["resnames"], d["box"]
lig = np.where(rn == "MOL")[0]
PL, ZL = P[lig], Z[lig]
E_L = e_bb(bb, PL, ZL)
wat_idx = [np.where(res == r)[0] for r in np.unique(res[rn == "HOH"])]


def nearest_water(x):
    best = None
    for idx in wat_idx:
        Pw = P[idx] - np.round((P[idx].mean(0) - x) / box) * box
        dd = np.linalg.norm(Pw - x, axis=1).min()
        if best is None or dd < best[0]:
            best = (dd, Pw, Z[idx])
    return best[1], best[2]


rng = np.random.default_rng(0)
grid = np.round(np.arange(3.5, 0.699, -0.05), 3)
rows = []
for X in (6, 7, 8, 1):
    atoms = np.where(ZL == X)[0]
    for probe in (8, 1):
        pick = atoms if len(atoms) <= 4 else rng.choice(atoms, 4, replace=False)
        for i in pick:
            Pw0, Zw = nearest_water(PL[i])
            k = min([j for j in range(3) if Zw[j] == probe], key=lambda j: np.linalg.norm(Pw0[j] - PL[i]))
            u = (Pw0[k] - PL[i]) / np.linalg.norm(Pw0[k] - PL[i])
            E_w = e_bb(bb, Pw0, Zw)
            pos_all = np.concatenate([PL, Pw0]); num = np.concatenate([ZL, Zw])
            frag = np.r_[np.zeros(len(ZL), int), np.ones(3, int)]
            f = AnchoredForce(model, num, frag, es=True)
            f.set_lambda(0.5)                         # continuous segment: dU/dp = -sum c
            curve = []
            for rr in grid:
                Pw = Pw0 + (PL[i] + rr * u - Pw0[k])
                pos_all[len(ZL):] = Pw
                dm = np.linalg.norm(PL[:, None] - Pw[None], axis=-1)
                a, b = np.unravel_index(dm.argmin(), dm.shape)
                truth = e_bb(bb, pos_all, num) - E_L - E_w
                f.energy_and_forces(pos_all, need_dudl=True)
                c = -f.last_dudl_total
                curve.append(dict(r=float(rr), dmin=float(dm.min()),
                                  pair=f"{SYM[int(ZL[a])]}-{SYM[int(Zw[b])]}w",
                                  truth=truth, c=c, rem=truth - c))
            rem = np.array([x["rem"] for x in curve]) * EV
            dmin = np.array([x["dmin"] for x in curve])
            slope = np.gradient(rem, dmin)               # dr/dd, kJ/mol/A; > 0 pulls inward
            below = np.where(rem < rem[0] - KT)[0]
            row = dict(lig_atom=int(i), X=SYM[X], probe=SYM[probe], start=float(dmin[0]),
                       depth=float(rem.min() - rem[0]), d_at_min=float(dmin[rem.argmin()]),
                       max_pull=float(slope.max()), d_at_pull=float(dmin[slope.argmax()]),
                       d_1kT=float(dmin[below[0]]) if below.size else None, curve=curve)
            rows.append(row)
            print(f"{SYM[X]}{i:<3d}<-w{SYM[probe]}  start d {row['start']:.2f}  r falls 1 kT below start at d "
                  f"{row['d_1kT'] if row['d_1kT'] is not None else float('nan'):.2f}  deepest "
                  f"{row['depth']:+9.1f} kJ/mol at d {row['d_at_min']:.2f}  max pull {row['max_pull']:8.1f} "
                  f"kJ/mol/A at d {row['d_at_pull']:.2f} ({curve[int(slope.argmax())]['pair']})", flush=True)
json.dump(rows, open(f"{OUT}/scan_r.json", "w"), indent=1)

print("\nper (ligand element, probe): largest d at which r is 1 kT below its start; deepest r")
for X in "CNOH":
    for pr in "OH":
        v = [r for r in rows if r["X"] == X and r["probe"] == pr]
        if v:
            dk = [r["d_1kT"] for r in v if r["d_1kT"] is not None]
            print(f"  {X}<-w{pr}  n={len(v)}  d(1 kT) max {max(dk) if dk else float('nan'):.2f}  "
                  f"deepest {min(r['depth'] for r in v):+.1f} kJ/mol")
