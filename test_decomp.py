"""Self-check for the decomposition head. Run: python test_decomp.py

Asserts the four properties the head claims by construction, plus that the
interaction loss picks out the right edges.
"""

import torch

import decomp  # noqa: F401  -- applies the e3nn/torch compat shim, must precede e3nn
from decomp.head import DecompositionHead, total_energy, poly_cutoff
from decomp.losses import interaction_loss
from e3nn import o3

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

IRREPS = "8x0e+4x1o+2x2e"
R_MAX = 5.0


def build(n=12, box=6.0):
    pos = torch.rand(n, 3) * box
    h = torch.randn(n, o3.Irreps(IRREPS).dim)
    d = torch.cdist(pos, pos)
    adj = d < R_MAX
    adj.fill_diagonal_(False)   # never filter self-pairs by distance: see decomp/data.py
    edge_index = adj.nonzero().t().contiguous()    # bidirectional by construction
    return h, pos, edge_index


def test_pair_symmetry():
    """D_ij == D_ji. The head builds this in; it is not averaged in afterwards."""
    head = DecompositionHead(IRREPS, r_max=R_MAX)
    h, pos, ei = build()
    _, D = head(h, pos, ei)

    # locate the reverse of every edge
    n = pos.shape[0]
    key = ei[0] * n + ei[1]
    rev = ei[1] * n + ei[0]
    order = torch.argsort(key)
    pos_of = torch.searchsorted(key[order], rev)
    D_rev = D[order[pos_of]]

    err = (D - D_rev).abs().max().item()
    assert err < 1e-12, f"D_ij != D_ji, max |D_ij - D_ji| = {err:.3e}"
    print(f"  pair symmetry      max|D_ij - D_ji| = {err:.2e}")


def test_rotation_invariance():
    """E_intra and D are 0e: invariant when positions and features co-rotate."""
    head = DecompositionHead(IRREPS, r_max=R_MAX)
    h, pos, ei = build()
    E0, D0 = head(h, pos, ei)

    R = o3.rand_matrix()
    Dmat = o3.Irreps(IRREPS).D_from_matrix(R)
    E1, D1 = head(h @ Dmat.T, pos @ R.T, ei)

    e_err = (E0 - E1).abs().max().item()
    d_err = (D0 - D1).abs().max().item()
    assert e_err < 1e-10, f"E_intra not rotation invariant: {e_err:.3e}"
    assert d_err < 1e-10, f"D not rotation invariant: {d_err:.3e}"
    print(f"  rotation invariance max|dE_intra| = {e_err:.2e}  max|dD| = {d_err:.2e}")


def test_energy_conservation():
    """Forces are -dE/dR by autograd and agree with central differences."""
    head = DecompositionHead(IRREPS, r_max=R_MAX)
    h, pos, ei = build()
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    ebatch = torch.zeros(ei.shape[1], dtype=torch.long)

    def energy(p):
        Ei, D = head(h, p, ei)
        return total_energy(Ei, D, batch, ebatch, 1)[0]

    p = pos.clone().requires_grad_(True)
    F = -torch.autograd.grad(energy(p), p)[0]

    eps, worst = 1e-5, 0.0
    for i in (0, 3, 7):
        for a in range(3):
            d = torch.zeros_like(pos); d[i, a] = eps
            fd = -(energy(pos + d) - energy(pos - d)) / (2 * eps)
            worst = max(worst, abs(fd.item() - F[i, a].item()))
    assert worst < 1e-6, f"autograd force != finite difference, max err {worst:.3e}"
    print(f"  force conservation  max|F_auto - F_fd| = {worst:.2e}")


def test_smooth_cutoff():
    """D -> 0 at r_max with no step, so atoms entering the cutoff do not kick."""
    head = DecompositionHead(IRREPS, r_max=R_MAX)
    h = torch.randn(2, o3.Irreps(IRREPS).dim)
    ei = torch.tensor([[0, 1], [1, 0]])

    rs = torch.linspace(R_MAX - 0.5, R_MAX + 0.1, 40)
    vals = []
    for r in rs:
        pos = torch.tensor([[0.0, 0.0, 0.0], [r.item(), 0.0, 0.0]])
        vals.append(head(h, pos, ei)[1][0].item())
    vals = torch.tensor(vals)

    assert vals[rs >= R_MAX].abs().max() < 1e-14, "D does not vanish beyond r_max"
    jump = (vals[1:] - vals[:-1]).abs().max().item()
    assert jump < 1e-2, f"D jumps at the cutoff: {jump:.3e}"
    assert poly_cutoff(torch.tensor([R_MAX]), R_MAX).item() == 0.0
    print(f"  smooth cutoff       max step near r_max = {jump:.2e}, D(r>=r_max) = 0")


def test_scalar_layouts():
    """0e components need not be first or contiguous, and the layout varies per
    backbone config. MACE concatenates per-layer features, so hidden_irreps,
    num_interactions and keep_last_layer_irreps all change it. Derive the
    indices from the irreps; never assume a layout."""
    layouts = [
        "4x1o+6x0e+2x2e+3x0e",                        # scalars scattered, not first
        "32x0e+32x1o+32x0e",                          # MACE 2 layers, max_ell 1
        "16x0e+16x1o+16x2e+16x0e+16x1o+16x2e",        # MACE 2 layers, last kept
        "8x0e+8x1o+8x0e+8x1o+8x0e",                   # MACE 3 layers
        "12x0e",                                       # scalars only
        "5x1o+7x0e",                                   # no scalars at all at front
    ]
    for irreps_str in layouts:
        irreps = o3.Irreps(irreps_str)

        # independent reimplementation of the expected indices
        expect, off = [], 0
        for mul, ir in irreps:
            d = mul * ir.dim
            if ir.l == 0 and ir.p == 1:
                expect += list(range(off, off + d))
            off += d

        head = DecompositionHead(irreps_str, r_max=R_MAX)
        assert head.scalar_idx.tolist() == expect, f"{irreps_str}: wrong scalar indices"
        assert head.n_scalar == len(expect)

        n = 10
        pos = torch.rand(n, 3) * 6.0
        h = torch.randn(n, irreps.dim)
        d = torch.cdist(pos, pos)
        adj = d < R_MAX
        adj.fill_diagonal_(False)   # never filter self-pairs by distance
        ei = adj.nonzero().t().contiguous()

        E0, D0 = head(h, pos, ei)
        R = o3.rand_matrix()
        Dm = irreps.D_from_matrix(R)
        E1, D1 = head(h @ Dm.T, pos @ R.T, ei)

        e_err = (E0 - E1).abs().max().item()
        d_err = (D0 - D1).abs().max().item()
        assert e_err < 1e-10, f"{irreps_str}: E_intra not invariant ({e_err:.3e})"
        assert d_err < 1e-10, f"{irreps_str}: D not invariant ({d_err:.3e})"
        print(f"  layout {irreps_str:38s} n_scalar={head.n_scalar:3d} "
              f"max|dE|={e_err:.1e} max|dD|={d_err:.1e}")


def test_interaction_loss_selects_cross_edges():
    """L_int must sum exactly the A-B cross terms, each unordered pair once."""
    head = DecompositionHead(IRREPS, r_max=R_MAX)
    h, pos, ei = build(n=10)
    _, D = head(h, pos, ei)

    frag = torch.tensor([0] * 5 + [1] * 5)
    ebatch = torch.zeros(ei.shape[1], dtype=torch.long)
    _, pred = interaction_loss(D, ei, frag, ebatch, torch.zeros(1), 1)

    manual = sum(
        D[k].item()
        for k in range(ei.shape[1])
        if frag[ei[0, k]] != frag[ei[1, k]] and ei[0, k] < ei[1, k]
    )
    err = abs(pred.item() - manual)
    assert err < 1e-12, f"cross-fragment sum wrong: {pred.item()} vs {manual}"
    print(f"  L_int cross-edges   sum D_AB = {pred.item():+.6f}  (err {err:.1e})")


if __name__ == "__main__":
    print("DecompositionHead self-check")
    test_pair_symmetry()
    test_rotation_invariance()
    test_energy_conservation()
    test_smooth_cutoff()
    test_scalar_layouts()
    test_interaction_loss_selects_cross_edges()
    print("all passed")
