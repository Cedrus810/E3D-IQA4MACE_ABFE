"""Add-on to check_d19.py: W1 / W3 with the wall actually touching, and the FD step of [4].

The acetamide start frame has no water inside any wall radius, so check_d19's W1 / W3
derivatives were 0 = 0. Here the water nearest ligand atom 0 is moved rigidly toward it
until its closest atom sits at 0.85 r0.
  [2w] dU/dq vs central FD in q at q 0.5 (W1) and 3.5 (W3, s_w = 4 - q)
  [3w] per-atom sums vs totals at the same q
  [4h] ML force vs FD at q 1.5 with h = 1e-4 and 5e-5 A: O(h^2) truncation shrinks 4x
"""

import sys

import numpy as np
import openmm
import torch
from openmm import unit

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm  # noqa: E402
from decomp.mlmm_hremd import Evaluator, segments_of  # noqa: E402
from run_abfe import load_model  # noqa: E402

E3D = "/home/ruigengji/E3D"
LEG = f"{E3D}/experiments/2026-10-15_acetamide_leg/em_acetamide.npz"
KJ = unit.kilojoule_per_mole
torch.set_num_threads(16)
plat = openmm.Platform.getPlatformByName("Reference")
P = np.asarray(np.load(LEG, allow_pickle=True)["positions"], dtype=np.float64)
model = load_model(f"{E3D}/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt",
                   "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", device="cpu").to(torch.float64)
sysA, mlfA, info = mlmm.build_mlmm_system(LEG, model, device="cpu", positions=P, shell_mm=True)
box, lig, num, W = info["box"], info["lig"], info["numbers"], info["water"]

# push the water nearest ligand atom a0 to 0.85 r0 of its closest atom
a0 = int(lig[0])
mi = lambda d: d - box * np.round(d / box)  # noqa: E731
dO = [np.linalg.norm(mi(P[w[0]] - P[a0])) for w in W]
w = W[int(np.argmin(dO))]
r0 = {int(x): mlmm.WALL_R0[int(num[a0])][0 if int(num[x]) == 8 else 1] for x in w}
X = P.copy()
u = mi(P[w[0]] - P[a0]); u /= np.linalg.norm(u)
for step in range(400):
    ratio = min(np.linalg.norm(mi(X[x] - P[a0])) / r0[x] for x in w)
    if ratio <= 0.85:
        break
    X[w] -= 0.005 * u
print(f"water {w.tolist()} moved {np.linalg.norm(mi(X[w[0]] - P[w[0]])):.2f} A; closest r/r0 {ratio:.3f}", flush=True)

qs = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0])
ev = Evaluator(sysA, mlfA, info, qs, plat)
sysB, mlfB, _ = mlmm.build_mlmm_system(LEG, model, device="cpu", positions=P, shell_mm=True)
ctx = openmm.Context(sysB, openmm.VerletIntegrator(0.001), plat)
ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])


def full(q, Y, groups=None):
    mlmm.set_path(ctx, mlfB, q, lig, shell_mm=True)
    ctx.setPositions(Y * unit.angstrom)
    kw = {} if groups is None else {"groups": groups}
    return ctx.getState(getEnergy=True, **kw).getPotentialEnergy().value_in_unit(KJ)


U, parts = ev.all_states(X)
print(f"[1w] all_states vs Context: max |dU| {np.abs(U - np.array([full(q, X) for q in qs])).max():.2e} kJ/mol", flush=True)
for q in (0.5, 3.5):
    der, at = ev.derivatives(X, q, parts, per_atom=True)
    seg = segments_of(q, shell_mm=True)[0]
    fd = (full(q + 1e-3, X) - full(q - 1e-3, X)) / 2e-3
    s = sum(v.sum() for v in at.values())
    print(f"[2w] q {q:g} ({seg}): dU/dq {der[seg]:+.6f} vs FD {fd:+.6f} kJ/mol (|d| {abs(der[seg] - fd):.1e})", flush=True)
    print(f"[3w] q {q:g} ({seg}): per-atom sum {s:+.6f} vs total {der[seg]:+.6f} (|d| {abs(s - der[seg]):.1e})", flush=True)

inner_atoms = [int(x) for x in info["inner_atoms"]]
atoms = [int(lig[0]), int(lig[3]), inner_atoms[0], inner_atoms[1], inner_atoms[30], inner_atoms[31]]
mlmm.set_path(ctx, mlfB, 1.5, lig, shell_mm=True)
ctx.setPositions(P * unit.angstrom)
F = ctx.getState(getForces=True, groups={mlmm.G_ML}).getForces(asNumpy=True).value_in_unit(KJ / unit.nanometer) / 10
for h in [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ("1e-4", "5e-5"))]:
    err = []
    for x in atoms:
        for c in range(3):
            Xp, Xm = P.copy(), P.copy()
            Xp[x, c] += h; Xm[x, c] -= h
            err.append(abs(-(full(1.5, Xp, {mlmm.G_ML}) - full(1.5, Xm, {mlmm.G_ML})) / (2 * h) - F[x, c]))
    print(f"[4h] ML force at q 1.5 vs FD, h {h:g} A: max |d| {max(err):.2e}, median {np.median(err):.2e} kJ/mol/A",
          flush=True)
