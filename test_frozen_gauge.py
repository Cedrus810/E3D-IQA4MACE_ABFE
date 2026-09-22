"""Can the IQA gauge be imposed on a PRETRAINED backbone without retraining it?

Run: python test_frozen_gauge.py

This is the question the paper cannot answer: E3D-IQA was trained from scratch,
so it cannot separate "the representation must reorganise" from "only the
readout must change". The answer decides whether training from scratch on
OMol25 (v2 plan WP5, the most expensive work package) is necessary at all.

Setup: a real pretrained MACE (MACE-OFF24_medium) supplies the features. A
randomly initialised teacher head on those same features defines one particular
(E_intra, D_ij) gauge -- so the target gauge is expressible from frozen
features by construction. Students then try to find it:

    frozen,    w_I = 0     E/F only, backbone frozen
    frozen,    w_I = 0.1   + L_IQA,  backbone frozen     <- the question
    finetuned, w_I = 0.1   + L_IQA,  backbone trainable  <- the reference

If frozen w_I=0.1 reaches the finetuned result, WP5 is unnecessary.

Read-only on /home/ruigengji/MLP/mace; downloads nothing.
"""

import torch

import decomp  # noqa: F401
from decomp.mace_adapter import MACEDecomposition
from decomp.train import train_step, trainable_parameters
from test_negative_control import collate, water_dimers
from test_pretrained import load

DEV = "cuda" if torch.cuda.is_available() else "cpu"
import os
N_STRUCT = int(os.environ.get("NSTRUCT", 32))
STEPS = int(os.environ.get("STEPS", 3000))


def fresh_backbone():
    return load("MACE-OFF24_medium.model").to(torch.float32).to(DEV)


def run(w_I, freeze, batch, ref, lr=1e-3, seed=7, trace=False):
    torch.manual_seed(seed)
    student = MACEDecomposition(fresh_backbone()).to(DEV)
    opt = torch.optim.Adam(trainable_parameters(student, freeze), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    b = {**batch, "ref": ref}
    hist = []
    for i in range(STEPS):
        loss, _ = train_step(student, b, opt,
                             dict(w_E=1.0, w_F=1.0, w_I=w_I, w_int=0.0),
                             freeze_backbone=freeze)
        sched.step()
        if trace and i % (STEPS // 6) == 0:
            hist.append(f"{i}:{loss:.4f}")
    if trace:
        print(f"      loss trace  {'  '.join(hist)}")
    student.eval()
    out = student(b, compute_force=True)
    n = int(b["batch"].max()) + 1
    na = torch.bincount(b["batch"], minlength=n).to(out["energy"].dtype)
    n_train = sum(p.numel() for p in trainable_parameters(student, freeze))
    return {
        "E/atom MAE": ((out["energy"] - ref["E"]).abs() / na).mean().item(),
        "F MAE": (out["forces"] - ref["F"]).abs().mean().item(),
        "E_intra MAE": (out["E_intra"] - ref["E_intra"]).abs().mean().item(),
        "D MAE": (out["D"] - ref["D"]).abs().mean().item(),
        "trainable": n_train,
    }


if __name__ == "__main__":
    print(f"Frozen-backbone gauge test  (device {DEV}, MACE-OFF24_medium, "
          f"{N_STRUCT} structures, {STEPS} steps)")
    species = torch.tensor([1, 0, 0, 1, 0, 0])          # O H H O H H -> indices into [H, O]
    batch = collate(water_dimers(N_STRUCT), species)
    # MACE-OFF24 has 10 elements; rebuild node_attrs against its own element table
    bb = fresh_backbone()
    zs = bb.atomic_numbers.tolist()
    z_real = torch.tensor([8, 1, 1, 8, 1, 1]).repeat(N_STRUCT)
    attrs = torch.zeros(batch["positions"].shape[0], len(zs), device=DEV)
    attrs[torch.arange(len(z_real), device=DEV),
          torch.tensor([zs.index(int(x)) for x in z_real], device=DEV)] = 1.0
    batch["node_attrs"] = attrs

    torch.manual_seed(123)
    teacher = MACEDecomposition(bb).to(DEV).eval()
    out = teacher({**batch}, compute_force=True)
    ref = {"E": out["energy"].detach(), "F": out["forces"].detach(),
           "E_intra": out["E_intra"].detach(), "D": out["D"].detach()}
    print(f"  teacher gauge: |E_intra| mean {ref['E_intra'].abs().mean():.4f}, "
          f"|D| mean {ref['D'].abs().mean():.4f}\n")
    del teacher, bb
    torch.cuda.empty_cache() if DEV == "cuda" else None

    # predict-zero baseline: if a trained MAE is not far BELOW this, nothing was learned
    n = int(batch["batch"].max()) + 1
    na = torch.bincount(batch["batch"], minlength=n).to(ref["E"].dtype)
    zero = {"E/atom MAE": (ref["E"].abs() / na).mean().item(),
            "F MAE": ref["F"].abs().mean().item(),
            "E_intra MAE": ref["E_intra"].abs().mean().item(),
            "D MAE": ref["D"].abs().mean().item(), "trainable": 0}

    cases = {"frozen    w_I=0   ": (0.0, True), "frozen    w_I=1   ": (1.0, True),
             "frozen    w_I=10  ": (10.0, True), "frozen    w_I=100 ": (100.0, True),
             "finetuned w_I=10  ": (10.0, False)}
    rows = {"predict zero      ": zero}
    for name, (w, fr) in cases.items():
        print(f"  {name}")
        rows[name] = run(w, fr, batch, ref, trace=True)

    keys = ["E/atom MAE", "F MAE", "E_intra MAE", "D MAE"]
    print()
    print(f"  {'':19s} " + "".join(f"{k:>14s}" for k in keys) + f"{'trainable':>14s}")
    for name, r in rows.items():
        print(f"  {name:19s} " + "".join(f"{r[k]:14.5f}" for k in keys)
              + f"{r['trainable']:14,d}")

    z = zero["E_intra MAE"]
    f0 = rows["frozen    w_I=0   "]["E_intra MAE"]
    f1 = min(rows[k]["E_intra MAE"] for k in rows if k.startswith("frozen    w_I=") and "=0 " not in k)
    ft = rows["finetuned w_I=10  "]["E_intra MAE"]

    # A comparison between runs means nothing unless the runs actually learned.
    converged = f1 < 0.5 * z and ft < 0.5 * z
    print(f"\n  learned anything?  frozen w_I=0.1 {f1:.4f} vs predict-zero {z:.4f}"
          f"   finetuned {ft:.4f}")
    if not converged:
        print("  => NOT CONVERGED. No conclusion can be drawn about WP5 from this run.")
    else:
        print(f"  frozen: L_IQA reorganises by {f0 / max(f1, 1e-12):.1f}x "
              f"({f0:.4f} -> {f1:.4f})")
        print(f"  frozen vs finetuned: {f1:.4f} vs {ft:.4f} ({f1 / max(ft,1e-12):.2f}x)")
        print("  => WP5 " + ("UNNECESSARY" if f1 < 2 * ft else "still justified"))
