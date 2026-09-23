"""Is MACE-POLAR-1 a usable backbone for the decomposition head?

Run: python test_polar.py [S|M|L]

POLAR-1 is `mace.modules.extensions.PolarMACE`: a short-range MACE stack plus an
explicit long-range electrostatic term. Three things about it are not true of
`mace-omol-0`, and each one can be wrong silently:

  1. It needs `fermi_level`, `external_field`, `pbc`, `rcell` and `volume` in the
     input dict. `decomp.data.mace_batch` supplies them.
  2. It builds a k-grid from `cell` before it knows the system is aperiodic, so
     a cell is required even with pbc off -- but the energy must not depend on
     which one. `mace_batch` uses 2 * r_max instead of MACE's own
     (max|r| + 1) * 5 * r_max; this file is what licenses that.
  3. Its `node_feats` are the SHORT-RANGE stack only. The interaction loop runs
     to completion before the electrostatics block, so the long-range term
     reaches the total energy but never the features the head reads.

Environment: mace-torch 0.3.16 with graph_electrostatics **v0.4.0**. v0.4.4 (PR
#4, merged 2026-09-15) normalised source features to [N, 1, M] and added
`force_pbc_evaluator`, neither of which mace 0.3.16 knows about -- it fails with
`TypeError: ... unexpected keyword argument 'force_pbc_evaluator'`. The docs'
"0.3.16 + v0.4.4" pairing does not work.

Read-only on /home/ruigengji/MLP/mace; downloads nothing.
"""

import sys

import numpy as np
import torch

import decomp  # noqa: F401  -- must precede e3nn, see decomp/__init__.py
from decomp.data import mace_batch
from decomp.mace_adapter import MACEDecomposition, mace_node_irreps

MDIR = "/home/ruigengji/MLP/mace"

# A water dimer: small enough to finite-difference, polar enough that the
# long-range term is not noise.
POS = np.array([[0.00, 0.00, 0.00], [0.96, 0.00, 0.00], [-0.24, 0.93, 0.00],
                [0.00, 0.00, 2.90], [0.96, 0.00, 3.00], [-0.24, 0.93, 3.00]])
NUM = np.array([8, 1, 1, 8, 1, 1])
FRAG = np.array([0, 0, 0, 1, 1, 1])


def batch(positions, n=1, model=None, dtype=torch.float64, device="cpu"):
    return mace_batch([positions] * n, [NUM] * n, model, charges=[0] * n,
                      frag_ids=[FRAG] * n, device=device, dtype=dtype)


def energy(model, data):
    return model(data, training=False, compute_force=False)["energy"]


def reference_energy(path):
    """The official MACECalculator, which uses MACE's own 120 A box."""
    from ase import Atoms
    from mace.calculators import MACECalculator

    at = Atoms("OHHOHH", positions=POS)
    at.info["charge"] = 0
    at.info["spin"] = 1
    at.info["external_field"] = [0.0, 0.0, 0.0]
    at.calc = MACECalculator(model_paths=path, device="cpu",
                             default_dtype="float64", model_type="PolarMACE")
    return at.get_potential_energy()


def main(tag="M"):
    path = f"{MDIR}/MACE-POLAR-1-{tag}.model"
    m = torch.load(path, map_location="cpu",
                   weights_only=False).to(torch.float64).eval()
    irreps = mace_node_irreps(m)
    print(f"MACE-POLAR-1-{tag}  {sum(p.numel() for p in m.parameters())/1e6:.1f}M "
          f"params  r_max {float(m.r_max)}  node_feats {irreps.dim}d  {irreps}")
    if not any(ir.l > 0 for _, ir in irreps):
        print("  WARNING: scalars only -- the head's pair branch loses all "
              "angular information on this backbone")

    # 1. our batch reproduces the official calculator, whose box is 20x larger.
    #    This is the whole licence for the small cell in mace_batch.
    ref = reference_energy(path)
    got = energy(m, batch(POS, model=m)).item()
    print(f"  vs MACECalculator   {got:.9f} vs {ref:.9f}  "
          f"|dE| = {abs(got-ref):.2e} eV")
    assert abs(got - ref) < 1e-7, "small-cell energy disagrees with MACE's own box"

    # 2. and explicitly: the box may not matter at all.
    r_max = float(m.r_max)
    es = []
    for factor in (2.0, 5.0, 20.0):
        b = batch(POS, model=m)
        box = factor * r_max
        eye = torch.eye(3, dtype=torch.float64)
        b["cell"] = (eye * box).repeat(1, 1)
        b["rcell"] = (eye * (2 * np.pi / box)).repeat(1, 1)
        b["volume"] = torch.full((1,), box ** 3, dtype=torch.float64)
        es.append(energy(m, b).item())
    print(f"  box independence    max spread over 2/5/20 x r_max = "
          f"{max(es)-min(es):.2e} eV")
    assert max(es) - min(es) < 1e-8, "energy depends on the dummy cell"

    # 3. batching: three copies of one dimer must each give the single answer.
    e3 = energy(m, batch(POS, n=3, model=m))
    print(f"  batching            max|E_i - E_1| = "
          f"{(e3 - e3[0]).abs().max():.2e} eV")
    assert (e3 - e3[0]).abs().max() < 1e-9, "graphs in a batch are not independent"

    # 4. rotation invariance -- and POLAR is only approximately invariant.
    #    The short-range stack is exact (1e-10), but the real-space electrostatic
    #    evaluator finite-differences the field along the fixed Cartesian axes
    #    (RealSpaceFiniteDifferenceElectrostaticFeatures, via
    #    DisplacedGTOExternalFieldBlock), which is covariant only to O(h^2). The
    #    residue below is POLAR's, not the head's, and it ends up in the E/F
    #    targets the head is trained on. Budget, not bug -- but measure it.
    torch.manual_seed(1)
    rot = [abs(energy(m, batch(POS @ torch.linalg.qr(
        torch.randn(3, 3, dtype=torch.float64))[0].numpy().T, model=m)).item() - got)
        for _ in range(4)]
    print(f"  rotation anisotropy max|dE| = {max(rot):.2e} eV over 4 rotations "
          f"({max(rot)/0.043364104*1e3:.3f} millical/mol) -- POLAR's own")
    assert max(rot) < 1e-3, "rotation anisotropy far beyond POLAR's finite-difference budget"

    # 5. the head attaches, and its forces are the gradient of its energy.
    torch.manual_seed(0)
    head = MACEDecomposition(m, hidden=32).to(torch.float64).eval()
    b = batch(POS, model=m)
    b["positions"] = b["positions"].requires_grad_(True)
    out = head(b)
    F = out["forces"]

    h = 1e-5
    fd = torch.zeros_like(F)
    for i in range(POS.shape[0]):
        for k in range(3):
            plus, minus = POS.copy(), POS.copy()
            plus[i, k] += h
            minus[i, k] -= h
            bp, bm = batch(plus, model=m), batch(minus, model=m)
            bp["positions"] = bp["positions"].requires_grad_(True)
            bm["positions"] = bm["positions"].requires_grad_(True)
            fd[i, k] = -(head(bp, compute_force=False)["energy"].item()
                         - head(bm, compute_force=False)["energy"].item()) / (2 * h)
    err = (F - fd).abs().max().item()
    print(f"  head forces         max|F_auto - F_fd| = {err:.2e} eV/A")
    assert err < 1e-5, "head forces are not conservative on this backbone"

    # 6. what the head cannot see: the long-range term is outside node_feats.
    full = m(batch(POS, model=m), training=False, compute_force=False)
    lr = full["electrostatic_energy"].item()
    print(f"  long-range term     {lr:+.6f} eV of E = {got:.6f} eV  "
          f"({100*abs(lr/got):.3f}%) -- in the energy, NOT in node_feats")

    print("all passed")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "M")
