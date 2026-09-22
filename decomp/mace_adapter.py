"""Attach a DecompositionHead to a MACE backbone.

MACE is node-centric: its energy is sum_i E_i with no native pair term, so D_ij
must be constructed rather than read off. This is the generalization test for the
head's backbone-agnostic claim (RESEARCH_PLAN_v2.md S4.1, WP4).

Everything is derived from the model at runtime. Node-feature irreps depend on
hidden_irreps, num_interactions, max_ell and keep_last_layer_irreps, so nothing
about the layout may be assumed.
"""

import contextlib

import torch
from torch import nn
from e3nn import o3

from .head import DecompositionHead, total_energy


@contextlib.contextmanager
def _default_dtype(dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def mace_node_irreps(model):
    """Irreps of MACE's `node_feats` output.

    MACE returns torch.cat of each layer's product-basis output, so the irreps
    are those outputs concatenated in order. Not simplified: the concatenation
    is what indexes the tensor.
    """
    return o3.Irreps("+".join(str(p.linear.irreps_out) for p in model.products))


class MACEDecomposition(nn.Module):
    """MACE features -> (E_intra, D_ij) -> total energy and conservative forces.

    The backbone's own readouts are bypassed; energy comes entirely from the head.
    """

    def __init__(self, mace_model, lmax_pair=2, n_radial=8, hidden=64,
                 irreps_proj=None, n_pair_channels=1):
        super().__init__()
        self.backbone = mace_model
        self.r_max = float(mace_model.r_max)

        # Build the head at the backbone's dtype. e3nn bakes its Wigner-3j and
        # normalisation constants into buffers at construction time, so building
        # in float32 and calling .to(float64) afterwards keeps float32-rounded
        # constants and costs ~1e-7 relative equivariance -- measurable, and
        # exactly float32 eps. Constructing at the target dtype gives 2e-16.
        ref = next(mace_model.parameters())
        with _default_dtype(ref.dtype):
            self.head = DecompositionHead(
                mace_node_irreps(mace_model),
                lmax_pair=lmax_pair,
                n_radial=n_radial,
                r_max=self.r_max,
                hidden=hidden,
                irreps_proj=irreps_proj,
                n_pair_channels=n_pair_channels,
            )
        # Device moves are exact, unlike dtype casts, so this one is safe to do
        # after construction.
        self.head.to(ref.device)

    def forward(self, data, compute_force=True):
        pos = data["positions"]
        if compute_force and not pos.requires_grad:
            pos = pos.requires_grad_(True)
            data = {**data, "positions": pos}

        h = self.backbone(data, training=self.training, compute_force=False)["node_feats"]
        E_intra, D = self.head(h, pos, data["edge_index"])
        D_tot = D.sum(-1) if D.dim() > 1 else D   # channels are additive

        batch = data["batch"]
        edge_batch = batch[data["edge_index"][0]]
        n_graphs = int(batch.max()) + 1
        E = total_energy(E_intra, D_tot, batch, edge_batch, n_graphs)

        out = {"energy": E, "E_intra": E_intra, "D": D}
        if compute_force:
            out["forces"] = -torch.autograd.grad(
                E.sum(), pos, create_graph=self.training
            )[0]
        return out
