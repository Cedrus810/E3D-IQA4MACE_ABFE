"""Env smoke: does MACE-POLAR-1-M + the trained 20-frame head load and run here?

Run from the repo root:  <env>/bin/python experiments/2026-10-10_env_smoke/smoke_polar.py
CPU only, water dimer, seconds. Read-only on /home/ruigengji/MLP/mace and runs/.
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import decomp  # noqa: F401  -- before e3nn
import e3nn, mace
from decomp.data import mace_batch
from decomp.mace_adapter import MACEDecomposition

print(f"python {sys.executable}")
print(f"torch {torch.__version__}  e3nn {e3nn.__version__}  mace {getattr(mace, '__version__', '?')}")
import graph_longrange as g; print(f"graph_longrange {g.__version__}")

POS = np.array([[0.00, 0.00, 0.00], [0.96, 0.00, 0.00], [-0.24, 0.93, 0.00],
                [0.00, 0.00, 2.90], [0.96, 0.00, 3.00], [-0.24, 0.93, 3.00]])
NUM = np.array([8, 1, 1, 8, 1, 1]); FRAG = np.array([0, 0, 0, 1, 1, 1])
PATH = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"
CK = "runs/polar-M/ckpt/joint_organic_n50000_c10000_frames_solvent_h256_s60000_D.pt"

t = time.time()
bb = torch.load(PATH, map_location="cpu", weights_only=False).to(torch.float64).eval()
data = mace_batch([POS], [NUM], bb, charges=[0], frag_ids=[FRAG], device="cpu", dtype=torch.float64)
e_bb = bb(data, training=False, compute_force=False)["energy"].item()
print(f"[1] backbone load+forward ok  E = {e_bb:.9f} eV  ({time.time()-t:.1f} s)")

from ase import Atoms
from mace.calculators import MACECalculator
at = Atoms("OHHOHH", positions=POS)
at.info.update(charge=0, spin=1, external_field=[0.0, 0.0, 0.0])
at.calc = MACECalculator(model_paths=PATH, device="cpu", default_dtype="float64", model_type="PolarMACE")
ref = at.get_potential_energy()
print(f"[2] MACECalculator          E = {ref:.9f} eV  |dE| = {abs(e_bb-ref):.2e}")

bb32 = torch.load(PATH, map_location="cpu", weights_only=False).to(torch.float32).eval()
torch.manual_seed(0)
model = MACEDecomposition(bb32, hidden=256, n_pair_channels=4).eval()
model.head.load_state_dict(torch.load(CK, map_location="cpu"))
d32 = mace_batch([POS], [NUM], bb32, charges=[0], frag_ids=[FRAG], device="cpu", dtype=torch.float32)
out = model(d32, compute_force=True)
print(f"[3] head ckpt load+forward ok  E_head = {out['energy'].item():.6f} eV  "
      f"|F|max = {out['forces'].abs().max().item():.4f}  finite = {bool(torch.isfinite(out['forces']).all())}")
print("SMOKE OK")
