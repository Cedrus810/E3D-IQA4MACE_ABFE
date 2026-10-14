"""The anchored Hamiltonian's exact checks (RESEARCH_PLAN_v3.md S3, checks 1-5).

Run: python test_anchored.py HEAD_CKPT [ES_HEAD_CKPT] [structure.npz] [n_solvent]

One carve of the Atenolol leg, float64 throughout, so every identity is held to
float64 arithmetic rather than to fp32 GPU noise (S13.28: 2.5e-4 eV/A between
two identical fp32 force calls). The checks are arithmetic: they hold for any
head, trained or not, so a 200-step head is as good a test as a converged one.

  [1] U(p=0) = E_bb(R)                       lambda = 1
  [2] U(p=2) = E_bb(R_L) + E_bb(R_E)         lambda = 0, fragments built independently
  [3] dU/dp: per-atom sum = total = -(sumD [+ cross]); autograd = finite difference
  [4] U(p=1) - U(p=2) = r = truth - sumD     the cut-step work, against test_scale
  [5] forces = -dU/dR by finite differences, on both segments; finite on a p grid
  [6] es=True: sum c_i - (es=False) = the explicit cross Coulomb (es_coupling)
"""

import sys

import numpy as np
import torch

import decomp  # noqa: F401
from decomp.clusters import coupling_labels
from decomp.data import mace_batch
from decomp.mace_adapter import MACEDecomposition
from decomp.openmm_bridge import AnchoredForce, carve_from_arrays
from decomp.polar_es import es_coupling

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BB = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"


def model64(ckpt):
    bb = torch.load(BB, map_location="cpu", weights_only=False).to(torch.float64).to(DEV).eval()
    torch.manual_seed(0)
    m = MACEDecomposition(bb, hidden=256, n_pair_channels=4).to(DEV).eval()
    m.head.load_state_dict(torch.load(ckpt, map_location=DEV))
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def e_bb(bb, pos, num):
    b = mace_batch([pos], [num], bb, charges=[0], device=DEV, dtype=torch.float64)
    with torch.no_grad():
        return float(bb(b, training=False, compute_force=False)["energy"].sum())


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {name:52s} {got:+.9f} vs {want:+.9f}  |d| {abs(got-want):.1e}  "
          f"{'OK' if ok else 'FAIL'}", flush=True)
    return ok


def fd_forces(f, pos, comps, h=1e-4):
    worst = 0.0
    _, F = f.energy_and_forces(pos)
    for a, k in comps:
        P1, P2 = pos.copy(), pos.copy()
        P1[a, k] += h; P2[a, k] -= h
        fd = -(f.energy_and_forces(P1)[0] - f.energy_and_forces(P2)[0]) / (2 * h)
        worst = max(worst, abs(fd - F[a, k]))
    return worst, float(np.abs(F).max())


def main(ckpt, ckpt_es=None, structure="frames_test_solvent.npz", n=10):
    d = np.load(structure, allow_pickle=True)
    P = d["positions"][0] if d["positions"].ndim == 3 else d["positions"]
    pos, num, frag, _ = carve_from_arrays(P, d["numbers"], d["resids"], d["resnames"],
                                          n_solvent=n, box_ang=d["box"])
    L = frag == 0
    model = model64(ckpt)
    bb = model.backbone
    print(f"{structure} frame 0, n_solvent {n}: {len(num)} atoms, ligand {L.sum()}  [{ckpt}]")
    f = AnchoredForce(model, num, frag)
    ok = True

    U = {}
    for p in (0.0, 0.5, 1.0, 1.5, 2.0):
        f.set_lambda(p)
        U[p] = f.energy_and_forces(pos)[0]

    print("[1] lambda = 1")
    ok &= check("U(p=0) - E_bb(R)", U[0.0], e_bb(bb, pos, num), 1e-8)
    print("[2] lambda = 0")
    ok &= check("U(p=2) - E_bb(L) - E_bb(E)", U[2.0],
                e_bb(bb, pos[L], num[L]) + e_bb(bb, pos[~L], num[~L]), 1e-8)

    print("[3] dU/dp")
    for p in (0.5, 1.5):
        f.set_lambda(p)
        f.energy_and_forces(pos, need_dudl=True)
        tot, per = f.last_dudl_total, f.last_dudl
        h = 1e-4
        f.set_lambda(p + h); up = f.energy_and_forces(pos)[0]
        f.set_lambda(p - h); um = f.energy_and_forces(pos)[0]
        ok &= check(f"p={p}: autograd dU/dp vs finite difference", tot, (up - um) / (2 * h), 1e-6)
        if p < 1:
            ok &= check(f"p={p}: sum of per-atom dU/dp = total", float(per.sum()), tot, 1e-10)
            ok &= check(f"p={p}: per-atom dU/dp zero off the ligand",
                        float(np.abs(per[~L]).max()), 0.0, 0.0)
            sumD = -tot
        else:
            ok &= check(f"p={p}: cut dU/dp = U(0) - U(0+)", tot, U[2.0] - U[1.0], 1e-8)

    print("[4] cut-step work")
    truth = float(coupling_labels(bb, [pos], [num], [frag], device=DEV)[0])
    ok &= check("U(0+) - U(0) = truth - sumD", U[1.0] - U[2.0], truth - sumD, 1e-8)
    ok &= check("dU/dp linear on the cut: U(1.5) = mean of ends", U[1.5],
                0.5 * (U[1.0] + U[2.0]), 1e-8)

    print("[5] forces")
    comps = [(0, 0), (int(np.where(L)[0][-1]), 2), (int(np.where(~L)[0][0]), 1)]
    for p in (0.5, 1.5):
        f.set_lambda(p)
        w, Fm = fd_forces(f, pos, comps)
        ok &= check(f"p={p}: max |F - FD| (3 components), |F|max {Fm:.2f}", w, 0.0, 1e-5)
    Fmax = []
    for p in np.linspace(0, 2, 9):
        f.set_lambda(float(p))
        F = f.energy_and_forces(pos)[1]
        assert np.isfinite(F).all(), p
        Fmax.append(np.abs(F).max())
    print(f"  max |F| over p = 0..2 (9 points): {np.round(Fmax, 2)}  finite")

    if ckpt_es:
        print(f"[6] es=True  [{ckpt_es}]")
        m2 = model64(ckpt_es)
        fe, f0 = AnchoredForce(m2, num, frag, es=True), AnchoredForce(m2, num, frag)
        for g in (fe, f0):
            g.set_lambda(0.5); g.energy_and_forces(pos, need_dudl=True)
        cross, _ = es_coupling(m2.backbone, pos, num, frag, device=DEV)
        ok &= check("sum c (es) - sum c (no es) = cross", -(fe.last_dudl_total - f0.last_dudl_total),
                    cross, 1e-8)
        fe.set_lambda(0.5)
        w, Fm = fd_forces(fe, pos, comps)
        ok &= check(f"p=0.5 es: max |F - FD|, |F|max {Fm:.2f}", w, 0.0, 1e-5)
        fe.set_lambda(0.0)
        ok &= check("es: U(p=0) = E_bb(R)", fe.energy_and_forces(pos)[0], e_bb(m2.backbone, pos, num), 1e-8)

    print("\nall passed" if ok else "\nFAILED")
    return ok


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    sys.exit(0 if main(a[0], a[1] if len(a) > 1 and a[1] != "-" else None,
                       a[2] if len(a) > 2 else "frames_test_solvent.npz",
                       int(a[3]) if len(a) > 3 else 10) else 1)
