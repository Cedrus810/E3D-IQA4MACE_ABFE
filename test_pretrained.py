"""Attach the head to the locally stored pretrained MACE checkpoints.

Read-only: loads existing files from /home/ruigengji/MLP/mace, downloads nothing.
"""

import sys
import torch

import decomp  # noqa: F401  -- compat shim, must precede e3nn
from decomp.mace_adapter import MACEDecomposition, mace_node_irreps
from e3nn import o3

MDIR = "/home/ruigengji/MLP/mace"
torch.manual_seed(0)


def load(name):
    m = torch.load(f"{MDIR}/{name}", map_location="cpu", weights_only=False)
    m.eval()
    return m


def make_data(model, pos, z):
    """Minimal non-periodic MACE input for one structure."""
    zs = model.atomic_numbers.tolist()
    r_max = float(model.r_max)
    n = pos.shape[0]
    kw = {"dtype": pos.dtype, "device": pos.device}
    attrs = torch.zeros(n, len(zs), **kw)
    attrs[torch.arange(n, device=pos.device), [zs.index(int(x)) for x in z]] = 1.0
    d = torch.cdist(pos, pos)
    adj = d < r_max
    adj.fill_diagonal_(False)   # never filter self-pairs by distance: see decomp/data.py
    ei = adj.nonzero().t().contiguous()
    return {
        "positions": pos, "node_attrs": attrs, "edge_index": ei,
        "shifts": torch.zeros(ei.shape[1], 3, **kw),
        "unit_shifts": torch.zeros(ei.shape[1], 3, **kw),
        "cell": torch.zeros(3, 3, **kw),
        "batch": torch.zeros(n, dtype=torch.long, device=pos.device),
        "ptr": torch.tensor([0, n], device=pos.device),
        "head": torch.zeros(1, dtype=torch.long, device=pos.device),
    }


def water_dimer(dtype):
    pos = torch.tensor([
        [0.000, 0.000, 0.000], [0.758, 0.587, 0.000], [-0.758, 0.587, 0.000],
        [2.800, 0.100, 0.300], [3.400, 0.600, -0.200], [3.100, -0.700, 0.500],
    ], dtype=dtype)
    return pos, torch.tensor([8, 1, 1, 8, 1, 1])


def survey(names):
    print("node_feats irreps of each pretrained checkpoint:")
    for nm in names:
        try:
            m = load(nm)
            ir = mace_node_irreps(m)
            p = next(m.parameters())
            print(f"  {nm:38s} r_max={float(m.r_max):.1f} dtype={str(p.dtype).split('.')[-1]:7s}"
                  f" n_elem={len(m.atomic_numbers):3d} irreps={str(ir):46s} dim={ir.dim}")
            del m
        except Exception as e:
            print(f"  {nm:38s} FAILED: {type(e).__name__}: {str(e)[:70]}")


def full_check(name):
    """Rotation/translation invariance, pair symmetry, force conservation."""
    m = load(name)
    dtype = next(m.parameters()).dtype
    print(f"\n{name}  (dtype {str(dtype).split('.')[-1]})")

    model = MACEDecomposition(m).eval()   # head is built at the backbone's dtype
    pos, z = water_dimer(dtype)

    E0 = model(make_data(m, pos, z), compute_force=False)["energy"]
    R = o3.rand_matrix().to(dtype)
    E1 = model(make_data(m, pos @ R.T, z), compute_force=False)["energy"]
    E2 = model(make_data(m, pos + torch.tensor([3.1, -2.0, 0.7], dtype=dtype), z),
               compute_force=False)["energy"]
    print(f"  rotation |dE| = {(E0-E1).abs().max():.2e}   "
          f"translation |dE| = {(E0-E2).abs().max():.2e}")

    data = make_data(m, pos, z)
    D = model(data, compute_force=False)["D"]
    ei, n = data["edge_index"], pos.shape[0]
    key, rev = ei[0] * n + ei[1], ei[1] * n + ei[0]
    order = torch.argsort(key)
    print(f"  pair symmetry  max|D_ij - D_ji| = "
          f"{(D - D[order[torch.searchsorted(key[order], rev)]]).abs().max():.2e}")

    if dtype == torch.float64:
        def energy(p):
            return model(make_data(m, p, z), compute_force=False)["energy"][0]
        F = model(make_data(m, pos.clone(), z))["forces"]
        eps, worst = 1e-5, 0.0
        for i in (0, 4):
            for a in range(3):
                d = torch.zeros_like(pos); d[i, a] = eps
                fd = -(energy(pos + d) - energy(pos - d)) / (2 * eps)
                worst = max(worst, abs(fd.item() - F[i, a].item()))
        print(f"  force conservation  max|F_auto - F_fd| = {worst:.2e}")
    else:
        print("  force conservation  skipped (float32: finite differences too noisy)")
    del m, model


if __name__ == "__main__":
    names = [
        "MACE-OFF24_medium.model", "mace-omol-0-extra-large-4M.model",
        "MACE-POLAR-1-S.model", "mace-mpa-0-medium.model",
    ]
    survey(names)
    for nm in sys.argv[1:] or ["MACE-OFF24_medium.model"]:
        full_check(nm)
