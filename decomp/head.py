"""Backbone-agnostic energy-decomposition head.

    E = sum_i E_intra(i) + sum_{i<j} D_ij

Takes equivariant node features from any e3nn-based backbone (Allegro, NequIP,
MACE, ...) and emits an intra-atomic energy per node and a pair energy per edge.

Design constraints (RESEARCH_PLAN_v2.md S4.1):
  * D_ij == D_ji by construction, not by post-hoc averaging.
  * D_ij -> 0 smoothly at r_max, so forces stay continuous under MD.
  * Both branches feed the total energy; forces come from autograd. No side outputs.
"""

import torch
from torch import nn

from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.math import soft_one_hot_linspace


def poly_cutoff(r, r_max, p=6):
    """Polynomial envelope, 1 at r=0 -> 0 at r=r_max with zero 1st/2nd derivative."""
    x = r / r_max
    out = (
        1.0
        - ((p + 1) * (p + 2) / 2) * x**p
        + p * (p + 2) * x ** (p + 1)
        - (p * (p + 1) / 2) * x ** (p + 2)
    )
    return torch.where(x < 1.0, out, torch.zeros_like(out))


class DecompositionHead(nn.Module):
    """Equivariant node features -> (E_intra per atom, D_ij per edge).

    Args:
        irreps_node: irreps of the backbone's node features, e.g. "64x0e+32x1o+16x2e".
        lmax_pair:   max spherical-harmonic degree in the pair channel. Only EVEN
                     degrees are used: r_hat_ij -> -r_hat_ji under swap, so odd
                     degrees would break D_ij == D_ji.
        r_max:       cutoff radius, must match the backbone's.
    """

    def __init__(self, irreps_node, lmax_pair=2, n_radial=8, r_max=5.0,
                 hidden=64, n_pair_channels=1, irreps_proj=None):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.r_max = float(r_max)
        self.n_radial = n_radial

        # Optional projection before the pair tensor product. The TP runs PER
        # EDGE, so its cost scales with the backbone's feature width -- and
        # mace-omol-0 emits 19456 dims (three layers concatenated, largely
        # redundant). Projecting first decouples head capacity from backbone
        # width: without it, raising `hidden` past ~128 is unaffordable.
        self.proj = None
        if irreps_proj is not None:
            self.proj = o3.Linear(self.irreps_node, o3.Irreps(irreps_proj))
            self.irreps_node = o3.Irreps(irreps_proj)

        # --- node branch: scalars of h_i -> E_intra ---
        # 0e components need not be contiguous or first: MACE concatenates its
        # per-layer features, giving e.g. "32x0e+32x1o+32x0e". Gather them by
        # their actual slices, or the node MLP silently eats vector components
        # and E_intra stops being rotation invariant.
        scalar_idx = [
            torch.arange(sl.start, sl.stop)
            for (_, ir), sl in zip(self.irreps_node, self.irreps_node.slices())
            if ir.l == 0 and ir.p == 1
        ]
        if not scalar_idx:
            raise ValueError(f"backbone features carry no 0e scalars: {self.irreps_node}")
        self.register_buffer("scalar_idx", torch.cat(scalar_idx), persistent=False)
        self.n_scalar = int(self.scalar_idx.numel())
        self.node_mlp = FullyConnectedNet(
            [self.n_scalar, hidden, hidden, 1], act=torch.nn.functional.silu
        )

        # --- pair branch ---
        # even l only => invariant under i<->j swap
        self.irreps_sh = o3.Irreps([(1, (l, 1)) for l in range(0, lmax_pair + 1, 2)])
        # contract h_i+h_j against Y(r_hat) down to scalars; angular information
        # survives the contraction, swap symmetry does not have to be patched in.
        self.irreps_tp_out = o3.Irreps(f"{hidden}x0e")
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_node, self.irreps_sh, self.irreps_tp_out
        )
        # n_pair_channels > 1 emits one D per channel (e.g. the four SAPT
        # components). The physical pair energy is their sum, so the alchemical
        # layer keeps using D.sum(-1) and nothing downstream changes.
        self.n_pair_channels = n_pair_channels
        self.pair_mlp = FullyConnectedNet(
            [hidden + n_radial, hidden, hidden, n_pair_channels],
            act=torch.nn.functional.silu,
        )

    def forward(self, h, pos, edge_index):
        """
        Args:
            h:          [N, irreps_node.dim] backbone node features.
            pos:        [N, 3] positions. Pass with requires_grad for forces.
            edge_index: [2, E] BIDIRECTIONAL edge list (both ij and ji present).

        Returns:
            E_intra: [N]  intra-atomic energy, referenced to isolated atoms.
            D:       [E]  pair energy per DIRECTED edge, D_ij == D_ji.
                          Total pair energy is 0.5 * D.sum().
        """
        src, dst = edge_index[0], edge_index[1]
        if self.proj is not None:
            h = self.proj(h)

        E_intra = self.node_mlp(h[:, self.scalar_idx]).squeeze(-1)

        edge_vec = pos[dst] - pos[src]
        r = edge_vec.norm(dim=-1)

        h_sym = h[src] + h[dst]                       # symmetric under swap
        Y = o3.spherical_harmonics(                   # even l only -> symmetric
            self.irreps_sh, edge_vec, normalize=True, normalization="component"
        )
        radial = soft_one_hot_linspace(               # depends on |r| -> symmetric
            r, start=0.0, end=self.r_max, number=self.n_radial,
            basis="smooth_finite", cutoff=True,
        )

        feat = torch.cat([self.tp(h_sym, Y), radial], dim=-1)
        D = self.pair_mlp(feat) * poly_cutoff(r, self.r_max).unsqueeze(-1)
        return E_intra, D.squeeze(-1) if self.n_pair_channels == 1 else D


def total_energy(E_intra, D, batch, edge_batch, n_graphs):
    """Sum the two branches per structure. D is over directed edges, hence the 0.5."""
    E = torch.zeros(n_graphs, dtype=E_intra.dtype, device=E_intra.device)
    E = E.index_add(0, batch, E_intra)
    E = E.index_add(0, edge_batch, 0.5 * D)
    return E
