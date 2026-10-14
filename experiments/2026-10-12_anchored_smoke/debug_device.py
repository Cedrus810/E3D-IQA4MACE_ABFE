import sys, os; sys.path.insert(0, os.getcwd())
import numpy as np, torch
import decomp  # noqa
from run_abfe import load_model
from decomp.openmm_bridge import AnchoredForce
d = np.load("experiments/2026-10-12_anchored_smoke/atenolol_w5.npz")
m = load_model("runs/polar-M/ckpt/joint_organic_n50000_c10000_frames_solvent_h256_s60000_N.pt",
               "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", device="cuda")
f = AnchoredForce(m, d["numbers"], d["frag_id"])
print("force.device", f.device, "bb", next(m.backbone.parameters()).device)
p = torch.tensor(d["positions"], dtype=torch.float32).requires_grad_(True)
data = f._graph(p.detach().numpy())
data["positions"] = p.to(f.device)
print({k: (v.device if torch.is_tensor(v) else type(v)) for k, v in data.items()})
from decomp.hremd import TorchReplica
import traceback
try:
    r = TorchReplica(f, d["numbers"], d["positions"] * 0.1)
    print("TorchReplica ok")
except Exception:
    traceback.print_exc()
    import inspect
    print(inspect.getsource(type(f).torch_energy))
    print(type(f).__mro__)
