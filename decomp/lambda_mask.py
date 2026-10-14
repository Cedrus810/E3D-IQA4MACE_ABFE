"""Atom-wise alchemical coupling (RESEARCH_PLAN_v2.md S5).

    U(R, lambda) = sum_i E_intra(i) + D_LL + D_EE + sum_{i in L} lambda_i sum_{a in E} D_ia

Three modes, because they give DIFFERENT endpoints:

  "edge"      scale only the LE edge energies. This is v1's design. E_intra and
              D_LL still see the environment through the backbone's features, so
              at lambda=0 the ligand stays environment-polarised and
              U(R,0) != U_L + U_E. Kept as a control, not for use.

  "graph"     scale the LE edge energies AND cut the graph across the boundary,
              so at lambda=0 each subsystem is computed as if the other were
              absent. Endpoint exact for any backbone. Costs the v1 requirement
              that ligand-internal energy be lambda-invariant -- an aesthetic
              preference, not a thermodynamic one: TI needs dU/dlambda, which
              autograd supplies including the extra term.

  "bodyorder" the S5.3 generalisation. A term over atom set S scales by
              prod_{i in S and L} lambda_i when S touches both L and E, else 1.
              Reduces to "graph" at two-body order; needed for MACE.
"""

import torch


def lambda_edge_weight(lam, frag_id, edge_index, ligand=0):
    """Per-edge coupling factor.

    An edge inside one fragment is untouched. A cross-boundary edge with one
    ligand end scales by that atom's lambda; with two (impossible for a single
    L/E split, possible for multi-fragment) by the product.
    """
    src, dst = edge_index
    in_L = frag_id == ligand
    l_src, l_dst = in_L[src], in_L[dst]
    cross = l_src != l_dst
    lam_src = torch.where(l_src, lam[src], torch.ones_like(lam[src]))
    lam_dst = torch.where(l_dst, lam[dst], torch.ones_like(lam[dst]))
    w = lam_src * lam_dst
    return torch.where(cross, w, torch.ones_like(w))


def alchemical_energy(model, data, lam, frag_id, mode="graph", ligand=0):
    """U(R, lambda) for one batch. `lam` is per atom; only ligand entries matter.

    Returns (U per graph, E_intra, D) so the caller can inspect the partition.
    """
    if mode not in ("edge", "graph", "bodyorder"):
        raise ValueError(mode)

    ei = data["edge_index"]
    w = lambda_edge_weight(lam, frag_id, ei, ligand)

    if mode == "edge":
        # v1: features still span the boundary; only the edge energy is scaled.
        h = model.backbone(data, training=model.training,
                           compute_force=False)["node_feats"]
        E_intra, D = model.head(h, data["positions"], ei)
    else:
        # Cut the graph as lambda closes. At lambda=0 the boundary edges are
        # gone from the message passing too, so each side is computed alone.
        keep = w > 0
        sub = {**data, "edge_index": ei[:, keep],
               "shifts": data["shifts"][keep],
               "unit_shifts": data["unit_shifts"][keep]}
        h = model.backbone(sub, training=model.training,
                           compute_force=False)["node_feats"]
        E_intra, D = model.head(h, data["positions"], ei)

    D = D.sum(-1) if D.dim() > 1 else D
    batch = data["batch"]
    ng = int(batch.max()) + 1
    U = torch.zeros(ng, dtype=E_intra.dtype, device=E_intra.device)
    U = U.index_add(0, batch, E_intra)
    U = U.index_add(0, batch[ei[0]], 0.5 * w * D)
    return U, E_intra, D


def isolated_energy(model, data, frag_id, which, ligand=0):
    """Energy of one fragment computed with the other absent -- the reference
    the lambda=0 endpoint must reproduce."""
    sel = (frag_id == which)
    idx = sel.nonzero().squeeze(-1)
    remap = torch.full_like(frag_id, -1)
    remap[idx] = torch.arange(len(idx), device=frag_id.device)
    ei = data["edge_index"]
    keep = sel[ei[0]] & sel[ei[1]]
    sub_ei = remap[ei[:, keep]]
    sub = {k: v for k, v in data.items() if k not in
           ("positions", "node_attrs", "edge_index", "shifts", "unit_shifts",
            "batch", "ptr", "frag_id")}
    sub.update({
        "positions": data["positions"][idx],
        "node_attrs": data["node_attrs"][idx],
        "edge_index": sub_ei,
        "shifts": data["shifts"][keep],
        "unit_shifts": data["unit_shifts"][keep],
        "batch": torch.zeros(len(idx), dtype=torch.long, device=idx.device),
        "ptr": torch.tensor([0, len(idx)], device=idx.device),
    })
    h = model.backbone(sub, training=False, compute_force=False)["node_feats"]
    E_intra, D = model.head(h, sub["positions"], sub_ei)
    D = D.sum(-1) if D.dim() > 1 else D
    return E_intra.sum() + 0.5 * D.sum()


# ---------------------------------------------------------------- anchored (v3 S3)

def _fragment(data, frag_id, which, charge):
    """One fragment of a single-graph batch as its own MACE input dict."""
    sel = (frag_id == which)
    idx = sel.nonzero().squeeze(-1)
    remap = torch.full_like(frag_id, -1)
    remap[idx] = torch.arange(len(idx), device=frag_id.device)
    ei = data["edge_index"]
    keep = sel[ei[0]] & sel[ei[1]]
    sub = {k: v for k, v in data.items() if k not in
           ("positions", "node_attrs", "edge_index", "shifts", "unit_shifts",
            "batch", "ptr", "frag_id", "total_charge")}
    sub.update({
        "positions": data["positions"][idx],
        "node_attrs": data["node_attrs"][idx],
        "edge_index": remap[ei[:, keep]],
        "shifts": data["shifts"][keep],
        "unit_shifts": data["unit_shifts"][keep],
        "batch": torch.zeros(len(idx), dtype=torch.long, device=idx.device),
        "ptr": torch.tensor([0, len(idx)], device=idx.device),
        "total_charge": torch.tensor([int(charge)], dtype=torch.long,
                                     device=idx.device),
    })
    return sub


def anchored_energy(model, data, lam, frag_id, ligand=0, es=False, cut=None,
                    charges=(0, 0)):
    """The anchored Hamiltonian of RESEARCH_PLAN_v3.md S3, one graph.

        U(R, lambda) = E_bb(R) - sum_{i in L} (1 - lambda_i) c_i(R)        lambda > 0, full graph
        c_i          = sum_{a in E} D_ia  (+ E_es,i^LE with es=True, S2.3)
        U(R, 0)      = E_bb(R_L) + E_bb(R_E)

    The total comes from the backbone, so both endpoints are exact whatever
    the head; the head only decides how much of the coupling is attributed.
    `cut` = s in [0, 1] gives the cut segment at lambda = 0+,
    U_s = (1 - s) U(0+) + s U(0) (S3: the remainder spread is 3-13 kT, S13.25,
    so this segment needs windows). cut=None is the continuous segment.

    dU/dlambda_i = c_i exactly: D does not depend on lambda. Forces with
    es=True are exact for uniform lambda over the ligand (the diagonal path):
    the per-atom Coulomb split carries no position gradient, the cross term
    times (1 - mean lambda) carries all of it, with zero value.

    charges = (ligand, environment) total charges for the fragments at
    lambda = 0. Returns (U [1], c [n_nodes], parts dict).
    """
    import contextlib
    from .polar_es import coulomb_cross

    if int(data["batch"].max()) != 0:
        raise ValueError("anchored_energy takes one graph")
    bb = model.backbone
    in_L = frag_id == ligand
    ctx = (coulomb_cross(bb, frag_id, ligand, per_atom=True) if es
           else contextlib.nullcontext({}))
    with ctx as rec:
        out = bb(data, training=False, compute_force=False)
    E_bb = out["energy"].sum()
    _, D = model.head(out["node_feats"], data["positions"], data["edge_index"])
    D = D.sum(-1) if D.dim() > 1 else D

    src, dst = data["edge_index"]
    cross = in_L[src] != in_L[dst]
    lig_end = torch.where(in_L[src], src, dst)
    # each unordered pair appears twice in the edge list: half to its ligand atom each time
    c = torch.zeros(len(frag_id), dtype=D.dtype, device=D.device).index_add(
        0, lig_end[cross], 0.5 * D[cross])
    es_grad = torch.zeros((), dtype=D.dtype, device=D.device)
    if es:
        c = c + torch.where(in_L, rec["per_atom"].to(D.dtype), torch.zeros_like(c))
        x = rec["cross"].sum()
        es_grad = (1 - lam[in_L].mean().detach()) * (x - x.detach())
    w = torch.where(in_L, 1 - lam, torch.zeros_like(lam))
    U_plus = E_bb - (w * c).sum() - es_grad
    parts = {"E_bb": E_bb, "sum_c": (c * in_L).sum()}
    if cut is None:
        return U_plus.reshape(1), c, parts

    U0 = sum(bb(_fragment(data, frag_id, k, q), training=False,
                compute_force=False)["energy"].sum()
             for k, q in ((ligand, charges[0]), (1 - ligand, charges[1])))
    parts.update({"U_plus": U_plus, "U0": U0})
    return ((1 - cut) * U_plus + cut * U0).reshape(1), c, parts
