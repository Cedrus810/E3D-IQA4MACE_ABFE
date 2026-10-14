"""Stage 0.2 smoke: wall time of one ML force call (backbone fwd+bwd + head)
on carves of the Atenolol solvent leg, POLAR-1-M + 20-frame <=24 head, GPU.
This is the ML part of an ML/MM step; the MM part and the PythonForce round
trip are not included. Run from the repo root."""
import sys, time, os
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import decomp  # noqa: F401
from run_abfe import load_model
from decomp.openmm_bridge import AlchemicalForce, carve_from_arrays

CK = "runs/polar-M/ckpt/joint_organic_n50000_c10000_frames_solvent_h256_s60000_D.pt"
model = load_model(CK, "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", device="cuda")
d = np.load("md50_solvent.npz", allow_pickle=True)
print(f"{'n_solv':>7s}{'atoms':>7s}{'ms/call':>9s}{'sd':>7s}{'peak GiB':>9s}")
for n in (20, 50, 100):
    pos, num, frag, _ = carve_from_arrays(d["positions"], d["numbers"], d["resids"],
                                          d["resnames"], n_solvent=n, box_ang=d["box"])
    f = AlchemicalForce(model, num, frag, mode="graph")
    f.set_lambda(0.5)
    for _ in range(3):
        f.energy_and_forces(pos)                     # warm-up
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(20):
        t = time.perf_counter(); f.energy_and_forces(pos); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    print(f"{n:7d}{len(num):7d}{np.mean(ts):9.1f}{np.std(ts):7.1f}"
          f"{torch.cuda.max_memory_allocated()/2**30:9.2f}", flush=True)
