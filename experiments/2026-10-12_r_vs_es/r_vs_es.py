"""Is the anchored remainder r = truth - sumD the long-range electrostatics the
head cannot see?  (v3 S2.3 / D6)

POLAR-1's node_feats are the short-range stack only (test_polar.py), so the
head never sees the GTO Coulomb term.  Per carve of frames_test_solvent.npz
(the same carves as run_remainder.sh / learn_d7.zsh), in float64, from the
backbone alone:

  cross  = E_es(rho_L + rho_E) - E_es(rho_L) - E_es(rho_E), rho = full-system
           densities: the ligand-environment Coulomb at fixed charges, exactly
           pair-resolvable (bilinear)
  d_es   = E_es(all) - E_es(L alone) - E_es(E alone): electrostatic part of
           the coupling, cross + charge response
  d_el   = same for POLAR's local electron energy

r per carve is read from the two scan logs (heads D and N), not recomputed.
Run from the repo root; GPU, a few minutes.
"""
import os, re, sys, json
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import decomp  # noqa: F401
from decomp.data import mace_batch
from decomp.openmm_bridge import carve_from_arrays

OUT = os.path.dirname(os.path.abspath(__file__))
LOGS = {"D": "runs/polar-M/logs/remainder_frames.log",
        "N": "runs/polar-M/logs/scale_N_frames_test_solvent.log"}
SIZES = (5, 10, 20, 50, 100)
dev = "cuda"

bb = torch.load("/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", map_location="cpu",
                weights_only=False).to(torch.float64).to(dev).eval()

# Wrap the Coulomb block: same return value, plus the L/E cross term computed
# from the very densities the forward pass produced.
orig = bb.coulomb_energy
rec = {}
class Wrap(torch.nn.Module):
    def __init__(self, inner):
        super().__init__(); self.inner = inner
    def forward(self, **kw):
        e = self.inner(**kw)
        m = rec.get("ligmask")
        if m is not None:
            sf = kw["source_feats"]
            mL = m.to(sf.dtype).view(-1, *([1] * (sf.dim() - 1)))
            eL = self.inner(**{**kw, "source_feats": sf * mL})
            eE = self.inner(**{**kw, "source_feats": sf * (1 - mL)})
            rec["cross"] = float((e - eL - eE).sum())
        return e
bb.coulomb_energy = Wrap(orig)

def run(pos, num, ligmask=None):
    rec.clear(); rec["ligmask"] = ligmask
    b = mace_batch([pos], [num], bb, charges=[0], device=dev, dtype=torch.float64)
    with torch.no_grad():
        o = bb(b, training=False, compute_force=False)
    return (float(o["energy"]), float(o["electrostatic_energy"].sum()),
            float(o["electron_energy"].sum()), rec.get("cross"))

def parse(path):
    r = {}
    for line in open(path):
        t = line.split()
        if len(t) == 9 and t[0].isdigit():
            r[(int(t[0]), int(t[1]))] = (float(t[4]), float(t[5]))   # truth, sumD
    return r

heads = {k: parse(v) for k, v in LOGS.items()}
d = np.load("frames_test_solvent.npz", allow_pickle=True)
rows = []
for f, P in enumerate(d["positions"]):
    for n in SIZES:
        pos, num, frag, _ = carve_from_arrays(P, d["numbers"], d["resids"], d["resnames"],
                                              n_solvent=n, box_ang=d["box"])
        L = frag == 0
        Ea, esa, ela, cross = run(pos, num, torch.as_tensor(L, device=dev))
        El, esl, ell, _ = run(pos[L], num[L])
        Ee, ese, ele, _ = run(pos[~L], num[~L])
        truth = Ea - El - Ee
        row = dict(frame=f, n=n, truth=truth, cross=cross,
                   d_es=esa - esl - ese, d_el=ela - ell - ele)
        for h, tab in heads.items():
            tl, sd = tab[(f, n)]
            row[f"r_{h}"] = tl - sd
            row[f"truth_log_{h}"] = tl
        rows.append(row)
        print(f"{f:3d}{n:5d}  truth {truth:8.4f} (log {row['truth_log_D']:8.4f})  "
              f"cross {cross:8.4f}  d_es {row['d_es']:8.4f}  d_el {row['d_el']:8.4f}  "
              f"r_D {row['r_D']:7.4f}  r_N {row['r_N']:7.4f}", flush=True)
json.dump(rows, open(f"{OUT}/rows.json", "w"), indent=1)

kT = 0.025693
print("\nper size, over 20 frames: corr(r, x), and sd of r after removing b*x "
      "(least squares, with intercept), in kT")
for h in heads:
    print(f"\n head {h}")
    print(f"{'n':>4s}{'sd(r)':>8s}" + "".join(f"{x:>22s}" for x in ("cross", "d_es", "d_es+d_el")))
    for n in SIZES:
        R = [x for x in rows if x["n"] == n]
        r = np.array([x[f"r_{h}"] for x in R])
        out = f"{n:4d}{r.std(ddof=1)/kT:8.2f}"
        for key in ("cross", "d_es", "dsum"):
            v = np.array([x["d_es"] + x["d_el"] if key == "dsum" else x[key] for x in R])
            c = np.corrcoef(r, v)[0, 1]
            b = np.polyfit(v, r, 1)
            res = r - np.polyval(b, v)
            out += f"   c {c:+.2f} b {b[0]:+.2f} sd {res.std(ddof=2)/kT:5.2f}"
        print(out)
print("\nmagnitudes (mean over frames, eV):")
for n in SIZES:
    R = [x for x in rows if x["n"] == n]
    m = lambda k: np.mean([x[k] for x in R])
    s = lambda k: np.std([x[k] for x in R], ddof=1)
    print(f"{n:4d}  truth {m('truth'):7.3f}  cross {m('cross'):7.3f}+-{s('cross'):.3f}  "
          f"d_es {m('d_es'):7.3f}+-{s('d_es'):.3f}  d_el {m('d_el'):7.3f}+-{s('d_el'):.3f}")
