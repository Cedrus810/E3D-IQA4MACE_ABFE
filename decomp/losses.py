"""Mixed-supervision loss (RESEARCH_PLAN_v2.md S4.3).

    L = w_E L_E + w_F L_F + m_IQA w_I L_IQA + m_int w_int L_int

L_IQA pins the one-body/two-body split (node gauge).
L_int pins sum_{i in A, b in B} D_ib against a physical observable -- the
quantity the alchemy actually scales. IQA supervision alone does not constrain
it, which is why both terms are needed.
"""

import torch


def _per_graph_mean(x, index, n_graphs):
    """Mean of x within each graph. Empty graphs give 0."""
    total = torch.zeros(n_graphs, dtype=x.dtype, device=x.device).index_add(0, index, x)
    count = torch.zeros(n_graphs, dtype=x.dtype, device=x.device).index_add(
        0, index, torch.ones_like(x)
    )
    return total / count.clamp(min=1.0)


def _masked_mean(per_graph, mask):
    """Mean over labelled structures only; 0 (not NaN) when nothing is labelled."""
    if mask is None:
        return per_graph.mean()
    m = mask.to(per_graph.dtype)
    return (per_graph * m).sum() / m.sum().clamp(min=1.0)


def energy_loss(E_pred, E_ref, n_atoms):
    """Per-atom normalised, as in arXiv:2609.00674."""
    return (((E_pred - E_ref) / n_atoms) ** 2).mean()


def force_loss(F_pred, F_ref, batch, n_graphs):
    """Mean over 3N components, then over structures."""
    se = ((F_pred - F_ref) ** 2).sum(dim=-1) / 3.0
    return _per_graph_mean(se, batch, n_graphs).mean()


def iqa_node_loss(E_intra_pred, E_intra_ref, batch, n_graphs, mask=None):
    """Node gauge. E_intra_ref must be referenced to isolated atoms:
    dE_intra(i) = E_intra(i) - E_iso_atom(Z_i)."""
    se = (E_intra_pred - E_intra_ref) ** 2
    return _masked_mean(_per_graph_mean(se, batch, n_graphs), mask)


def cross_fragment_sum(D, edge_index, frag_id, edge_batch, n_graphs):
    """Sum D over edges crossing the fragment boundary, per structure.

    Each unordered cross pair appears twice in a bidirectional edge list, hence
    the 0.5. D may carry a trailing channel axis, which is preserved.
    """
    src, dst = edge_index[0], edge_index[1]
    cross = (frag_id[src] != frag_id[dst]).to(D.dtype)
    if D.dim() == 1:
        out = torch.zeros(n_graphs, dtype=D.dtype, device=D.device)
        return out.index_add(0, edge_batch, 0.5 * cross * D)
    out = torch.zeros(n_graphs, D.shape[-1], dtype=D.dtype, device=D.device)
    return out.index_add(0, edge_batch, 0.5 * cross.unsqueeze(-1) * D)


def interaction_loss(D, edge_index, frag_id, edge_batch, E_int_ref, n_graphs, mask=None):
    """Counterpoise-corrected fragment interaction energy."""
    E_int_pred = cross_fragment_sum(D, edge_index, frag_id, edge_batch, n_graphs)
    if E_int_pred.dim() > 1:
        E_int_pred = E_int_pred.sum(-1)
    return _masked_mean((E_int_pred - E_int_ref) ** 2, mask), E_int_pred


def sapt_loss(D, edge_index, frag_id, edge_batch, sapt_ref, n_graphs, mask=None):
    """Multi-channel supervision against the SAPT decomposition.

    D must carry one channel per component. The components decompose the SAME
    interaction energy but decay very differently -- electrostatics ~1/r,
    exchange ~exp(-r), dispersion ~1/r^6 -- so matching all of them constrains
    how D_ij is distributed over the boundary, not merely its sum. Supervising
    the total alone lets any wrong combination of components add up correctly.

    Each channel is normalised by its own mean square: the components differ by
    3x in magnitude (exchange is the largest), and without normalisation the
    loss would be dominated by exchange alone.

    sapt_ref: [n_graphs, n_channels]. Rows with NaN (6.1% of DES370K lacks SAPT)
    are masked out per structure.
    """
    pred = cross_fragment_sum(D, edge_index, frag_id, edge_batch, n_graphs)
    if pred.dim() == 1:
        raise ValueError("sapt_loss needs a head with n_pair_channels > 1")
    n = min(pred.shape[-1], sapt_ref.shape[-1])
    pred, ref = pred[:, :n], sapt_ref[:, :n]

    ok = torch.isfinite(ref).all(-1)
    if mask is not None:
        ok = ok & mask.bool()
    ref = torch.nan_to_num(ref)

    scale = (ref[ok].pow(2).mean(0) if ok.any() else torch.ones(n, device=ref.device))
    per_channel = ((pred - ref) ** 2 / scale.clamp(min=1e-12))
    return _masked_mean(per_channel.mean(-1), ok), pred


def _scale(t):
    """Mean square of the target: divides a squared-error term to make it O(1)."""
    return t.detach().pow(2).mean().clamp(min=1e-12)


def total_loss(
    pred, ref, batch, edge_batch, n_graphs, n_atoms,
    w_E=1.0, w_F=1.0, w_I=0.1, w_int=0.1,
    mask_iqa=None, mask_int=None, edge_index=None, frag_id=None,
    normalize=True,
):
    """Returns (scalar loss, dict of components).

    normalize=True divides each squared-error term by the mean square of its own
    target, so every term starts at O(1) and the weights mean the same thing
    across datasets. Without it, weights silently depend on label units: with
    forces of order 14 and intra energies of order 0.35, w_I=0.1 puts the IQA
    term at ~1e-6 of the total loss and it contributes no usable gradient.
    The published w_I=0.1 was tuned for that paper's label scales, not yours.
    """
    out = {}
    sE = _scale(ref["E"] / n_atoms) if normalize else 1.0
    loss = w_E * energy_loss(pred["E"], ref["E"], n_atoms) / sE
    out["E"] = (loss / max(w_E, 1e-12)).detach()

    l_f = force_loss(pred["F"], ref["F"], batch, n_graphs)
    l_f = l_f / _scale(ref["F"]) if normalize else l_f
    loss = loss + w_F * l_f
    out["F"] = l_f.detach()

    if ref.get("E_intra") is not None and w_I > 0:
        l_i = iqa_node_loss(pred["E_intra"], ref["E_intra"], batch, n_graphs, mask_iqa)
        l_i = l_i / _scale(ref["E_intra"]) if normalize else l_i
        loss = loss + w_I * l_i
        out["IQA"] = l_i.detach()

    if ref.get("E_int") is not None and w_int > 0:
        l_x, _ = interaction_loss(
            pred["D"], edge_index, frag_id, edge_batch, ref["E_int"], n_graphs, mask_int
        )
        l_x = l_x / _scale(ref["E_int"]) if normalize else l_x
        loss = loss + w_int * l_x
        out["int"] = l_x.detach()

    return loss, out
