"""D15, part 2: where does the MLP stop being physical at short range?

A single water is moved rigidly toward one ligand atom of Atenolol along its
own approach direction (frames_test_solvent.npz frame 0, held out), the probe
atom (water O or water H) from 3.5 to 0.7 A. Along the scan, float64:

  truth(r) = E_bb(L + w) - E_bb(L) - E_bb(w)     backbone coupling
  sumD(r)  = sum_a D_ia (+ nothing else)         head N's attributed coupling

A physical potential is repulsive all the way in. The learned one has a
maximum somewhere: inside it the energy falls again -- no wall (STATUS:
minimising with the learned potential ran to -5e5 eV). r_peak, where truth is
largest, is the inner limit of physical behaviour; the wall must keep every
ligand-water contact outside it. d_min is the actual shortest L-w distance at
each r (another pair can be closer than the probe).

Run from the repo root on a GPU.
"""
import os, sys, json
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import decomp  # noqa: F401
from decomp.data import mace_batch
from decomp.openmm_bridge import AnchoredForce
from test_anchored import model64

dev = "cuda"
OUT = os.path.dirname(os.path.abspath(__file__))
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
model = model64("runs/polar-M/ckpt/joint_organic_n50000_c10000_frames_solvent_h256_s60000_N.pt")
bb = model.backbone


def E(pos, num):
    b = mace_batch([pos], [num], bb, charges=[0], device=dev, dtype=torch.float64)
    with torch.no_grad():
        return float(bb(b, training=False, compute_force=False)["energy"].sum())


d = np.load("frames_test_solvent.npz", allow_pickle=True)
P, Z, res, rn = d["positions"][0], d["numbers"], d["resids"], d["resnames"]
box = d["box"]
lig = np.where(rn == "MOL")[0]
PL, ZL = P[lig], Z[lig]
E_L = E(PL, ZL)
wat_res = [r for r in np.unique(res[rn == "HOH"])]
wat_idx = [np.where(res == r)[0] for r in wat_res]


def nearest_water(x):
    best = None
    for idx in wat_idx:
        Pw = P[idx] - np.round((P[idx].mean(0) - x) / box) * box      # minimum image
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
            k = [j for j in range(3) if Zw[j] == probe]
            k = min(k, key=lambda j: np.linalg.norm(Pw0[j] - PL[i]))
            u = (Pw0[k] - PL[i]) / np.linalg.norm(Pw0[k] - PL[i])
            E_w = E(Pw0, Zw)
            pos_all = np.concatenate([PL, Pw0]); num = np.concatenate([ZL, Zw])
            frag = np.r_[np.zeros(len(ZL), int), np.ones(3, int)]
            f = AnchoredForce(model, num, frag); f.set_lambda(0.5)
            curve = []
            for r in grid:
                Pw = Pw0 + (PL[i] + r * u - Pw0[k])
                pos_all[len(ZL):] = Pw
                dm = np.linalg.norm(PL[:, None] - Pw[None], axis=-1)
                a, b = np.unravel_index(dm.argmin(), dm.shape)
                truth = E(pos_all, num) - E_L - E_w
                f.energy_and_forces(pos_all, need_dudl=True)
                curve.append((float(r), float(dm.min()), f"{SYM[int(ZL[a])]}-{SYM[int(Zw[b])]}",
                              truth, -f.last_dudl_total / 0.5 * 0.5))
            c = np.array([x[3] for x in curve]); s = np.array([x[4] for x in curve])
            ip, jp = int(c.argmax()), int(s.argmax())
            row = dict(lig_atom=int(i), X=SYM[X], probe=SYM[probe], r0=float(np.linalg.norm(Pw0[k] - PL[i])),
                       r_peak=curve[ip][0], dmin_peak=curve[ip][1], pair_peak=curve[ip][2],
                       truth_peak=float(c[ip]), r_peak_head=curve[jp][0], dmin_peak_head=curve[jp][1],
                       curve=curve)
            rows.append(row)
            print(f"{SYM[X]}{i:<3d}<-w{SYM[probe]}  start {row['r0']:.2f} A   backbone peak at r {row['r_peak']:.2f} "
                  f"(d_min {row['dmin_peak']:.2f} {row['pair_peak']}, {row['truth_peak']:+.2f} eV)   "
                  f"head peak at r {row['r_peak_head']:.2f} (d_min {row['dmin_peak_head']:.2f})", flush=True)
json.dump(rows, open(f"{OUT}/scan.json", "w"), indent=1)

print("\nper (ligand element, probe): d_min at the backbone's peak -- the inner limit of physical behaviour")
for X in "CNOH":
    for pr in "OH":
        v = [r["dmin_peak"] for r in rows if r["X"] == X and r["probe"] == pr]
        if v:
            print(f"  {X}-{pr}  n={len(v)}  d_min at peak: max {max(v):.2f}  mean {np.mean(v):.2f}  min {min(v):.2f}")
