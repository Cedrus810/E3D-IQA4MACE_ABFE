"""D6 smoke, part 2: the E3D_ES target path of test_joint.make_targets.
fp32 as in training, so the yardstick is fp32 rounding of |E_bb| (~1e4 eV):
2^-23 * |E_bb| per pass. Run from the repo root."""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import decomp  # noqa: F401
from decomp.data import mace_batch
import test_joint as tj

dev = "cuda"
bb = torch.load("/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", map_location="cpu",
                weights_only=False).to(torch.float32).to(dev).eval()
z = np.load("data/des370k_organic_test.npz", allow_pickle=True)
P = [z["positions"][i] for i in range(4)]; Z = [z["numbers"][i] for i in range(4)]
Fr = [z["frags"][i] for i in range(4)]
bt = mace_batch(P, Z, bb, charges=[0] * 4, frag_ids=Fr, device=dev)
bt["positions"] = bt["positions"].requires_grad_(True)
teacher = tj.MACEDecomposition(bb).to(dev).eval()

tj.ES = True;  E1, F1, I1, X = tj.make_targets(bb, teacher, bt)
tj.ES = False; E0, F0, I0 = tj.make_targets(bb, teacher, bt)

pos = bt["positions"].detach().clone().requires_grad_(True)
out = bb({**bt, "positions": pos}, training=False, compute_force=False)
es = out["electrostatic_energy"]
g_es, = torch.autograd.grad(es.sum(), pos)
tol_E = 4 * 2 ** -23 * float(out["energy"].abs().max())
dE = float(((E0 - E1) - es.detach()).abs().max())
dF = float(((F0 - F1) - (-g_es)).abs().max())
print(f"E: max |(E0 - E_ES) - E_es| = {dE:.1e} eV   fp32 yardstick 4 eps |E_bb| = {tol_E:.1e} eV")
print(f"F: max |(F0 - F_ES) - F_es| = {dF:.1e} eV/A   (|F0| max {float(F0.abs().max()):.2f}, |F_es| max {float(g_es.abs().max()):.3f})")
print(f"E_es per dimer {es.detach().cpu().numpy().round(4)}   X {X.cpu().numpy().round(4)}")
print(f"I unchanged: {float((I1 - I0).abs().max()):.1e}")
# fp32 forces are not bit-reproducible on GPU: MACE's own forces differ by
# ~2.5e-4 eV/A between two identical calls (check_forces_dtype.py; 6e-13 in
# fp64, where the two paths also agree to 6e-13). That run-to-run floor is the
# yardstick, measured here on the same batch.
_, F0b, _ = tj.make_targets(bb, teacher, bt)
floor = float((F0 - F0b).abs().max())
print(f"F run-to-run floor (two identical fp32 calls): {floor:.1e} eV/A")
ok = dE < tol_E and dF < 3 * max(floor, 1e-6)
print("PASSED" if ok else "FAILED")
