"""The defining E3D-IQA experiment, reimplemented in e3nn on a MACE backbone.

Run: python test_negative_control.py

The original code is not published, so this is a from-scratch e3nn reproduction.
A randomly initialised teacher defines one particular (E_intra, D_ij) gauge.
Students are trained on the teacher's labels:

    w_I = 0     energy+force only  -> fits E/F but should NOT recover the split
    w_I = 0.1   + L_IQA            -> should recover the split at similar E/F

arXiv:2609.00674 reports 12.210 eV -> 0.229 eV intra error (53x) for this
contrast. Here the labels are synthetic, so this tests the machinery, not the
chemistry. No external data, no downloads.
"""

import numpy as np
import torch

import decomp  # noqa: F401
from decomp.mace_adapter import MACEDecomposition
from decomp.train import train_step, trainable_parameters
from e3nn import o3
from mace.modules import MACE, gate_dict, interaction_classes

DEV = "cuda" if torch.cuda.is_available() else "cpu"
Z = [1, 8]
R_MAX = 5.0


def make_mace():
    return MACE(
        r_max=R_MAX, num_bessel=8, num_polynomial_cutoff=6, max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=len(Z),
        hidden_irreps=o3.Irreps("16x0e+16x1o"), MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros(len(Z)), avg_num_neighbors=8.0,
        atomic_numbers=Z, correlation=2, gate=gate_dict["silu"],
    ).to(DEV)


def water_dimers(n_struct, seed=1):
    """Jittered water dimers -- real chemistry, so the graph is not pathological."""
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0., 0., 0.], [0.758, 0.587, 0.], [-0.758, 0.587, 0.],
                         [2.9, 0.1, 0.3], [3.5, 0.6, -0.2], [3.2, -0.7, 0.5]])
    return [base + 0.12 * torch.randn(6, 3, generator=g) for _ in range(n_struct)]


def collate(structs, species):
    """Batch a list of [n,3] geometries into one MACE data dict."""
    pos, attrs, ei, batch, ptr, off = [], [], [], [], [0], 0
    for i, p in enumerate(structs):
        n = p.shape[0]
        a = torch.zeros(n, len(Z)); a[torch.arange(n), species] = 1.0
        d = torch.cdist(p, p)
        adj = d < R_MAX
        adj.fill_diagonal_(False)   # never filter self-pairs by distance: see decomp/data.py
        e = adj.nonzero().t()
        pos.append(p); attrs.append(a); ei.append(e + off)
        batch.append(torch.full((n,), i, dtype=torch.long))
        off += n; ptr.append(off)
    pos = torch.cat(pos).to(DEV)
    ei = torch.cat(ei, 1).contiguous().to(DEV)
    batch = torch.cat(batch).to(DEV)
    return {
        "positions": pos, "node_attrs": torch.cat(attrs).to(DEV), "edge_index": ei,
        "shifts": torch.zeros(ei.shape[1], 3, device=DEV),
        "unit_shifts": torch.zeros(ei.shape[1], 3, device=DEV),
        "cell": torch.zeros(3, 3, device=DEV), "batch": batch,
        "ptr": torch.tensor(ptr, device=DEV),          # one offset per structure + 1
        "head": torch.zeros(len(structs), dtype=torch.long, device=DEV),
    }


def run(w_I, batch, ref, steps=400, lr=5e-3, seed=7):
    torch.manual_seed(seed)
    student = MACEDecomposition(make_mace()).to(DEV)
    opt = torch.optim.Adam(trainable_parameters(student, False), lr=lr)
    b = {**batch, "ref": ref}
    for _ in range(steps):
        train_step(student, b, opt,
                   dict(w_E=1.0, w_F=1.0, w_I=w_I, w_int=0.0), freeze_backbone=False)
    student.eval()
    out = student(b, compute_force=True)
    n = int(b["batch"].max()) + 1
    na = torch.bincount(b["batch"], minlength=n).to(out["energy"].dtype)
    return {
        "E/atom MAE": ((out["energy"] - ref["E"]).abs() / na).mean().item(),
        "F MAE": (out["forces"] - ref["F"]).abs().mean().item(),
        "E_intra MAE": (out["E_intra"] - ref["E_intra"]).abs().mean().item(),
    }


if __name__ == "__main__":
    print(f"E3D-IQA negative control, e3nn reimplementation on MACE  (device {DEV})")
    species = torch.tensor([1, 0, 0, 1, 0, 0])          # O H H O H H
    batch = collate(water_dimers(16), species)

    torch.manual_seed(123)
    teacher = MACEDecomposition(make_mace()).to(DEV).eval()
    out = teacher({**batch}, compute_force=True)
    ref = {"E": out["energy"].detach(), "F": out["forces"].detach(),
           "E_intra": out["E_intra"].detach(), "D": out["D"].detach()}
    print(f"  teacher gauge: |E_intra| mean {ref['E_intra'].abs().mean():.4f}, "
          f"|D| mean {ref['D'].abs().mean():.4f}\n")

    rows = {f"w_I = {w}": run(w, batch, ref) for w in (0.0, 0.1)}
    keys = list(next(iter(rows.values())))
    print(f"  {'':12s} " + "".join(f"{k:>16s}" for k in keys))
    for name, r in rows.items():
        print(f"  {name:12s} " + "".join(f"{r[k]:16.5f}" for k in keys))

    a, b_ = rows["w_I = 0.0"]["E_intra MAE"], rows["w_I = 0.1"]["E_intra MAE"]
    print(f"\n  L_IQA reorganises the node branch by {a / max(b_, 1e-12):.1f}x "
          f"({a:.4f} -> {b_:.4f})")
    print("  paper, real IQA labels: 12.210 -> 0.229 eV (53x)")
    assert b_ < a / 3, "L_IQA failed to reorganise the decomposition"
    print("  passed")
