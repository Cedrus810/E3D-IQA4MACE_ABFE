"""Self-check for the MACE backbone adapter. Run: python test_mace_adapter.py

MACE is node-centric (E = sum_i E_i, no native pair term), so this is the real
test of the head's backbone-agnostic claim.
"""

import numpy as np
import torch

import decomp  # noqa: F401  -- compat shim, must precede e3nn
from decomp.mace_adapter import MACEDecomposition, mace_node_irreps
from e3nn import o3
from mace.modules import MACE, gate_dict, interaction_classes

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

R_MAX = 5.0
Z = [1, 6, 8]


def make_mace(hidden, n_inter=2, max_ell=3):
    return MACE(
        r_max=R_MAX, num_bessel=8, num_polynomial_cutoff=6, max_ell=max_ell,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=n_inter, num_elements=len(Z),
        hidden_irreps=o3.Irreps(hidden), MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.zeros(len(Z)), avg_num_neighbors=8.0,
        atomic_numbers=Z, correlation=3, gate=gate_dict["silu"],
    )


def make_data(pos, species):
    """Minimal non-periodic MACE input. species indexes into Z."""
    n = pos.shape[0]
    node_attrs = torch.zeros(n, len(Z))
    node_attrs[torch.arange(n), species] = 1.0
    d = torch.cdist(pos, pos)
    adj = d < R_MAX
    adj.fill_diagonal_(False)   # never filter self-pairs by distance: see decomp/data.py
    ei = adj.nonzero().t().contiguous()
    return {
        "positions": pos,
        "node_attrs": node_attrs,
        "edge_index": ei,
        "shifts": torch.zeros(ei.shape[1], 3),
        "unit_shifts": torch.zeros(ei.shape[1], 3),
        "cell": torch.zeros(3, 3),
        "batch": torch.zeros(n, dtype=torch.long),
        "ptr": torch.tensor([0, n]),
        "head": torch.zeros(1, dtype=torch.long),
    }


def water_dimer():
    pos = torch.tensor([
        [0.000, 0.000, 0.000], [0.758, 0.587, 0.000], [-0.758, 0.587, 0.000],
        [2.800, 0.100, 0.300], [3.400, 0.600, -0.200], [3.100, -0.700, 0.500],
    ])
    return pos, torch.tensor([2, 0, 0, 2, 0, 0])   # O H H O H H


def test_irreps_derived_not_assumed():
    for hidden, n_inter, max_ell in [
        ("32x0e+32x1o", 2, 3), ("16x0e+16x1o+16x2e", 2, 3),
        ("24x0e+24x1o", 3, 2), ("8x0e", 2, 1),
    ]:
        m = make_mace(hidden, n_inter, max_ell)
        ir = mace_node_irreps(m)
        pos, sp = water_dimer()
        h = m(make_data(pos, sp), compute_force=False)["node_feats"]
        assert h.shape[-1] == ir.dim, f"{hidden}/{n_inter}: {h.shape[-1]} != {ir.dim}"
        print(f"  hidden={hidden:22s} L={n_inter} lmax={max_ell} -> {str(ir):42s} dim {ir.dim}")


def test_rotation_translation_invariance():
    model = MACEDecomposition(make_mace("16x0e+16x1o+16x2e"))
    pos, sp = water_dimer()
    E0 = model(make_data(pos, sp), compute_force=False)["energy"]

    R = o3.rand_matrix()
    E1 = model(make_data(pos @ R.T, sp), compute_force=False)["energy"]
    E2 = model(make_data(pos + torch.tensor([3.1, -2.0, 0.7]), sp), compute_force=False)["energy"]

    r_err = (E0 - E1).abs().max().item()
    t_err = (E0 - E2).abs().max().item()
    assert r_err < 1e-9, f"total energy not rotation invariant: {r_err:.3e}"
    assert t_err < 1e-9, f"total energy not translation invariant: {t_err:.3e}"
    print(f"  rotation |dE| = {r_err:.2e}   translation |dE| = {t_err:.2e}")


def test_pair_symmetry_through_mace():
    model = MACEDecomposition(make_mace("16x0e+16x1o+16x2e"))
    pos, sp = water_dimer()
    data = make_data(pos, sp)
    D = model(data, compute_force=False)["D"]
    ei = data["edge_index"]

    n = pos.shape[0]
    key, rev = ei[0] * n + ei[1], ei[1] * n + ei[0]
    order = torch.argsort(key)
    D_rev = D[order[torch.searchsorted(key[order], rev)]]
    err = (D - D_rev).abs().max().item()
    assert err < 1e-11, f"D_ij != D_ji through MACE: {err:.3e}"
    print(f"  pair symmetry through MACE  max|D_ij - D_ji| = {err:.2e}")


def test_force_conservation_through_mace():
    model = MACEDecomposition(make_mace("16x0e+16x1o"))
    pos, sp = water_dimer()

    def energy(p):
        return model(make_data(p, sp), compute_force=False)["energy"][0]

    F = model(make_data(pos.clone(), sp))["forces"]

    eps, worst = 1e-5, 0.0
    for i in (0, 1, 4):
        for a in range(3):
            d = torch.zeros_like(pos); d[i, a] = eps
            fd = -(energy(pos + d) - energy(pos - d)) / (2 * eps)
            worst = max(worst, abs(fd.item() - F[i, a].item()))
    assert worst < 1e-6, f"forces not conservative through MACE: {worst:.3e}"
    print(f"  force conservation through MACE  max|F_auto - F_fd| = {worst:.2e}")


if __name__ == "__main__":
    print("MACEDecomposition self-check")
    test_irreps_derived_not_assumed()
    test_rotation_translation_invariance()
    test_pair_symmetry_through_mace()
    test_force_conservation_through_mace()
    print("all passed")
