"""Training step for E3D-IQA on a MACE backbone.

    L = w_E L_E + w_F L_F + w_I L_IQA  [+ w_int L_int]

The defining experiment (arXiv:2609.00674): w_I = 0 leaves the node/edge split
free and it does NOT converge to IQA on its own; w_I > 0 reorganises it at
almost unchanged E/F accuracy. w_I = 0.1 is the published optimum.
"""

import torch

from .losses import total_loss


def train_step(model, batch, opt, weights, freeze_backbone=True):
    """One optimisation step. `batch` is a MACE data dict plus reference labels.

    freeze_backbone: with a pretrained OMol25 MACE the representation is already
    there; only the decomposition head needs to learn the split. Much cheaper
    and far less data-hungry than training the backbone from scratch.
    """
    model.train()
    if freeze_backbone:
        model.backbone.eval()
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    out = model(batch, compute_force=True)
    n_graphs = int(batch["batch"].max()) + 1
    n_atoms = torch.bincount(batch["batch"], minlength=n_graphs).to(out["energy"].dtype)

    loss, parts = total_loss(
        pred={"E": out["energy"], "F": out["forces"],
              "E_intra": out["E_intra"], "D": out["D"]},
        ref=batch["ref"],
        batch=batch["batch"],
        edge_batch=batch["batch"][batch["edge_index"][0]],
        n_graphs=n_graphs,
        n_atoms=n_atoms,
        edge_index=batch["edge_index"],
        frag_id=batch.get("frag_id"),
        mask_iqa=batch.get("mask_iqa"),
        mask_int=batch.get("mask_int"),
        **weights,
    )

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss), {k: float(v) for k, v in parts.items()}


def trainable_parameters(model, freeze_backbone=True):
    return model.head.parameters() if freeze_backbone else model.parameters()
