"""Split the coupling at the last good state of diag_q.py q = 2.0 (v3 S13.41).

Run: python diag_parts.py [diag_q2.0_last.npz]

On the ML region (L + 20 inner waters, unwrapped as MLMMForce does), float64,
CPU: truth = E_bb(L u I) - E_bb(L) - E_bb(I); c_i split into the head's pair
part (sumD) and the explicit L/E Coulomb (cross, D6); r = truth - sum c. Also
the same at the starting frame (thermal200 frame 0), and the ligand atoms with
the largest c_i with their closest water atoms.
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm  # noqa: E402
from decomp.lambda_mask import _fragment, anchored_energy  # noqa: E402
from decomp.mace_adapter import MACEDecomposition  # noqa: E402

E3D = "/home/ruigengji/E3D"
LEG = f"{E3D}/em_solvent.npz"
BB = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"
CKPT = f"{E3D}/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
START = f"{E3D}/experiments/2026-10-12_thermal_frames/thermal200_solvent.npz"
EV = 96.48533212
torch.set_num_threads(8)

bb = torch.load(BB, map_location="cpu", weights_only=False).to(torch.float64).eval()
model = MACEDecomposition(bb, hidden=256, n_pair_channels=4)
model.head.load_state_dict(torch.load(CKPT, map_location="cpu"))
model = model.to(torch.float64).eval()
for p in model.parameters():
    p.requires_grad_(False)

d = np.load(LEG, allow_pickle=True)
num = np.asarray(d["numbers"])
pos0 = np.asarray(np.load(START, allow_pickle=True)["positions"][0], dtype=np.float64)
system, mlf, info = mlmm.build_mlmm_system(LEG, model, n_inner=20, es=True, device="cpu", positions=pos0)
box = info["box"]
last = np.load(sys.argv[1] if len(sys.argv) > 1 else "diag_q2.0_last.npz")["positions"]
af = mlf.anch
Zml = num[mlf.ml_idx]
nl = len(info["lig"])


def split(P, label):
    X = mlf.ml_positions(P, box)
    data = af._graph(X)
    lam = torch.zeros(len(Zml), dtype=torch.float64)
    _, c_all, parts = anchored_energy(model, data, lam, af.frag_id, es=True, cover=af.cover)
    _, c_pair, _ = anchored_energy(model, data, lam, af.frag_id, es=False, cover=af.cover)
    E_all = float(parts["E_bb"])
    E_L = float(bb(_fragment(data, af.frag_id, 0, 0), training=False, compute_force=False)["energy"].sum())
    E_I = float(bb(_fragment(data, af.frag_id, 1, 0), training=False, compute_force=False)["energy"].sum())
    truth = E_all - E_L - E_I
    cA, cP = c_all[:nl].detach().numpy(), c_pair[:nl].detach().numpy()
    print(f"{label}: truth {truth * EV:9.1f}  sum c {cA.sum() * EV:9.1f} = sumD {cP.sum() * EV:9.1f} "
          f"+ cross {(cA - cP).sum() * EV:9.1f}  ->  r {(truth - cA.sum()) * EV:9.1f} kJ/mol", flush=True)
    D = np.linalg.norm(X[:nl, None] - X[None, nl:], axis=-1)
    for i in np.argsort(-cA)[:4]:
        j = int(np.argmin(D[i]))
        print(f"   ligand atom {i:2d} Z{Zml[i]}: c_i {cA[i] * EV:8.1f} (pair {cP[i] * EV:8.1f}, cross "
              f"{(cA[i] - cP[i]) * EV:8.1f}) kJ/mol; closest water atom Z{Zml[nl + j]} at {D[i, j]:.2f} A; "
              f"waters within 2.5 A: {int((D[i] < 2.5).sum())}", flush=True)
    return X


split(pos0, "start (thermal200 frame 0)")
split(np.asarray(last, dtype=np.float64), f"last good state before the NaN ({sys.argv[1] if len(sys.argv) > 1 else 'diag_q2.0_last.npz'})")
