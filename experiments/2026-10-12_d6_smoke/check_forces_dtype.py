"""Is MACE's own force for PolarMACE the exact -dE/dR that make_targets(ES) uses?
Same 4 DES370K dimers, fp32 and fp64, and fp32 run twice (nondeterminism floor)."""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import decomp  # noqa: F401
from decomp.data import mace_batch
dev = "cuda"
z = np.load("data/des370k_organic_test.npz", allow_pickle=True)
P = [z["positions"][i] for i in range(4)]; Z = [z["numbers"][i] for i in range(4)]
bb0 = torch.load("/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", map_location="cpu", weights_only=False)
for dt in (torch.float32, torch.float64):
    bb = bb0.to(dt).to(dev).eval()
    b = mace_batch(P, Z, bb, charges=[0] * 4, device=dev, dtype=dt)
    def mace_F():
        p = b["positions"].clone().requires_grad_(True)
        return bb({**b, "positions": p}, compute_force=True)["forces"].detach()
    p = b["positions"].clone().requires_grad_(True)
    e = bb({**b, "positions": p}, training=False, compute_force=False)["energy"]
    g, = torch.autograd.grad(e.sum(), p)
    F1, F2 = mace_F(), mace_F()
    print(f"{str(dt):14s} max|F_mace - (-dE/dR)| = {float((F1 + g).abs().max()):.1e}   "
          f"max|F_mace run1 - run2| = {float((F1 - F2).abs().max()):.1e}   |F| max {float(F1.abs().max()):.2f}")
