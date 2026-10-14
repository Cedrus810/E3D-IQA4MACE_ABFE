"""D6 smoke: decomp/polar_es.py and the E3D_ES target path, on one carve and a
few DES370K dimers. fp64 for the identities. Run from the repo root."""
import os, sys, json
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import decomp  # noqa: F401
from decomp.data import mace_batch
from decomp.openmm_bridge import carve_from_arrays
from decomp.polar_es import coulomb_cross, es_coupling

dev = "cuda"
bb = torch.load("/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", map_location="cpu",
                weights_only=False).to(torch.float64).to(dev).eval()
d = np.load("frames_test_solvent.npz", allow_pickle=True)
pos, num, frag, _ = carve_from_arrays(d["positions"][0], d["numbers"], d["resids"],
                                      d["resnames"], n_solvent=20, box_ang=d["box"])
b = mace_batch([pos], [num], bb, charges=[0], frag_ids=[frag], device=dev, dtype=torch.float64)

# 1. wrapper leaves the energy untouched
with torch.no_grad():
    e0 = float(bb(b, training=False, compute_force=False)["energy"])
    with coulomb_cross(bb, b["frag_id"], per_atom=True) as rec:
        e1 = float(bb(b, training=False, compute_force=False)["energy"])
cross = float(rec["cross"].sum()); pa = rec["per_atom"].cpu().numpy()
print(f"[1] energy with/without wrapper  |dE| = {abs(e1-e0):.1e} eV")
assert abs(e1 - e0) < 1e-9
assert bb.coulomb_energy.__class__.__name__ != "_Recorder", "wrapper not restored"

# 2. matches the S13.27 analysis for this carve
ref = [r for r in json.load(open("experiments/2026-10-12_r_vs_es/rows.json"))
       if r["frame"] == 0 and r["n"] == 20][0]
print(f"[2] cross {cross:.6f} vs r_vs_es {ref['cross']:.6f}  |d| = {abs(cross-ref['cross']):.1e}")
assert abs(cross - ref["cross"]) < 1e-6

# 3. Euler split: ligand rows and environment rows each sum to cross
L = frag == 0
print(f"[3] per-atom: sum_L {pa[L].sum():.6f}  sum_E {pa[~L].sum():.6f}  cross {cross:.6f}"
      f"  ligand atoms {pa[L].min():+.4f} .. {pa[L].max():+.4f}")
assert abs(pa[L].sum() - cross) < 1e-8 and abs(pa[~L].sum() - cross) < 1e-8

# 4. d cross / dR against central finite differences (density response included)
def cross_at(P):
    bp = mace_batch([P], [num], bb, charges=[0], frag_ids=[frag], device=dev, dtype=torch.float64)
    bp["positions"] = bp["positions"].requires_grad_(True)
    with coulomb_cross(bb, bp["frag_id"]) as r:
        bb(bp, training=False, compute_force=False)
    return r["cross"].sum(), bp["positions"]
c, p = cross_at(pos)
g, = torch.autograd.grad(c, p)
h, worst = 1e-4, 0.0
for atom in (0, 5, int(np.where(~L)[0][0])):
    for k in range(3):
        P1, P2 = pos.copy(), pos.copy(); P1[atom, k] += h; P2[atom, k] -= h
        with torch.no_grad():
            fd = (float(cross_at(P1)[0]) - float(cross_at(P2)[0])) / (2 * h)
        worst = max(worst, abs(fd - float(g[atom, k])))
print(f"[4] d cross/dR autograd vs FD (h={h} A, 9 components): max |diff| = {worst:.1e} eV/A"
      f"  (|g| max {float(g.abs().max()):.3f})")
assert worst < 1e-5

# 5. es_coupling: cross agrees, d_es agrees with r_vs_es
cr, des = es_coupling(bb, pos, num, frag, device=dev)
print(f"[5] es_coupling cross {cr:.6f}  d_es {des:.6f} (r_vs_es {ref['d_es']:.6f})")
assert abs(cr - cross) < 1e-9 and abs(des - ref["d_es"]) < 1e-6

# 6. E3D_ES target path: forces from the split autograd equal MACE's own
os.environ["E3D_ES"] = "1"
import test_joint as tj
tj.ES = True
z = np.load("data/des370k_organic_test.npz", allow_pickle=True)
P = [z["positions"][i] for i in range(4)]; Z = [z["numbers"][i] for i in range(4)]
Fr = [z["frags"][i] for i in range(4)]
bb32 = bb.to(torch.float32)
bt = mace_batch(P, Z, bb32, charges=[0] * 4, frag_ids=Fr, device=dev)
bt["positions"] = bt["positions"].requires_grad_(True)
teacher = tj.MACEDecomposition(bb32).to(dev).eval()
E_es, F_es, I_es, X = tj.make_targets(bb32, teacher, bt)
tj.ES = False
E_0, F_0, I_0 = tj.make_targets(bb32, teacher, bt)
with torch.no_grad():
    out = bb32({**bt, "positions": bt["positions"].detach()}, training=False, compute_force=False)
es = out["electrostatic_energy"]
print(f"[6] E: (E_0 - E_es) - E_es_backbone  max |d| = {float(((E_0 - E_es) - es).abs().max()):.1e} eV")
print(f"    X (dimer L/E Coulomb, eV): {X.cpu().numpy().round(4)}")
assert float(((E_0 - E_es) - es).abs().max()) < 1e-4
print("ALL CHECKS PASSED")
