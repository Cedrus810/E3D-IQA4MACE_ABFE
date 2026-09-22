"""Does S13.2 carry from MACE-OFF24 (640-dim) to an OMol25-pretrained backbone?

Run: python test_omol_transfer.py

Frozen `mace-omol-0-extra-large-4M`: 51.3M params, 19456-dim node features,
82 elements, r_max 6.0, and two per-graph categorical embeddings (total_spin,
total_charge). Deliberately small -- 8 structures, 600 steps, frozen only.
Fine-tuning 51M parameters on one 2080 Ti is not the question here.

Read-only on /home/ruigengji/MLP/mace; downloads nothing.
"""

import time

import torch

import decomp  # noqa: F401
from decomp.mace_adapter import MACEDecomposition, mace_node_irreps
from decomp.train import train_step, trainable_parameters
from test_negative_control import water_dimers

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "/home/ruigengji/MLP/mace/mace-omol-0-extra-large-4M.model"
N_STRUCT, STEPS, R_MAX = 8, 3000, 6.0


def backbone():
    m = torch.load(MODEL, map_location="cpu", weights_only=False)
    return m.to(torch.float32).to(DEV).eval()


def collate(structs, z_mol, m):
    """Batch geometries into a MACE data dict, including OMol25's charge/spin."""
    zs = m.atomic_numbers.tolist()
    idx = [zs.index(int(x)) for x in z_mol]
    pos, attrs, ei, batch, ptr, off = [], [], [], [], [0], 0
    for i, p in enumerate(structs):
        n = p.shape[0]
        a = torch.zeros(n, len(zs)); a[torch.arange(n), idx] = 1.0
        d = torch.cdist(p, p)
        adj = d < R_MAX
        adj.fill_diagonal_(False)   # never filter self-pairs by distance: see decomp/data.py
        ei.append(adj.nonzero().t() + off)
        pos.append(p); attrs.append(a)
        batch.append(torch.full((n,), i, dtype=torch.long))
        off += n; ptr.append(off)
    pos = torch.cat(pos).to(DEV)
    e = torch.cat(ei, 1).contiguous().to(DEV)
    ns = len(structs)
    return {
        "positions": pos, "node_attrs": torch.cat(attrs).to(DEV), "edge_index": e,
        "shifts": torch.zeros(e.shape[1], 3, device=DEV),
        "unit_shifts": torch.zeros(e.shape[1], 3, device=DEV),
        "cell": torch.zeros(3, 3, device=DEV),
        "batch": torch.cat(batch).to(DEV),
        "ptr": torch.tensor(ptr, device=DEV),
        "head": torch.zeros(ns, dtype=torch.long, device=DEV),
        # Neutral closed-shell: multiplicity 1, charge 0. Shape is [n_graphs],
        # NOT [n_graphs, 1]: the embedding is gathered per atom as emb[batch],
        # so a trailing singleton dim broadcasts to [n_atoms, n_atoms, 1024].
        "total_spin": torch.ones(ns, dtype=torch.long, device=DEV),
        "total_charge": torch.zeros(ns, dtype=torch.long, device=DEV),
    }


def run(w_I, batch, ref, lr=1e-3, seed=7):
    torch.manual_seed(seed)
    student = MACEDecomposition(backbone()).to(DEV)
    opt = torch.optim.Adam(trainable_parameters(student, True), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    b = {**batch, "ref": ref}
    t0 = time.perf_counter()
    for i in range(STEPS):
        loss, _ = train_step(student, b, opt,
                             dict(w_E=1.0, w_F=1.0, w_I=w_I, w_int=0.0),
                             freeze_backbone=True)
        sched.step()
        if i % (STEPS // 4) == 0:
            print(f"      step {i:4d}  loss {loss:.4f}", flush=True)
    student.eval()
    out = student(b, compute_force=True)
    n = int(b["batch"].max()) + 1
    na = torch.bincount(b["batch"], minlength=n).to(out["energy"].dtype)
    r = {"E/atom MAE": ((out["energy"] - ref["E"]).abs() / na).mean().item(),
         "F MAE": (out["forces"] - ref["F"]).abs().mean().item(),
         "E_intra MAE": (out["E_intra"] - ref["E_intra"]).abs().mean().item(),
         "D MAE": (out["D"] - ref["D"]).abs().mean().item()}
    print(f"      {time.perf_counter()-t0:.0f}s, peak GPU "
          f"{torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
    del student, opt
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return r


if __name__ == "__main__":
    print(f"OMol25 backbone transfer test  (device {DEV})", flush=True)
    m = backbone()
    ir = mace_node_irreps(m)
    print(f"  mace-omol-0-extra-large-4M: {sum(p.numel() for p in m.parameters())/1e6:.1f}M "
          f"params, node_feats dim {ir.dim}", flush=True)
    print(f"  {N_STRUCT} water dimers, {STEPS} steps, backbone FROZEN\n", flush=True)

    z_mol = torch.tensor([8, 1, 1, 8, 1, 1])
    batch = collate(water_dimers(N_STRUCT), z_mol, m)

    torch.manual_seed(123)
    teacher = MACEDecomposition(m).to(DEV).eval()
    out = teacher({**batch}, compute_force=True)
    ref = {k: v.detach() for k, v in
           zip(("E", "F", "E_intra", "D"),
               (out["energy"], out["forces"], out["E_intra"], out["D"]))}
    del teacher, m
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    n = int(batch["batch"].max()) + 1
    na = torch.bincount(batch["batch"], minlength=n).to(ref["E"].dtype)
    zero = {"E/atom MAE": (ref["E"].abs() / na).mean().item(),
            "F MAE": ref["F"].abs().mean().item(),
            "E_intra MAE": ref["E_intra"].abs().mean().item(),
            "D MAE": ref["D"].abs().mean().item()}

    rows = {"predict zero  ": zero}
    for name, w in {"frozen w_I=0  ": 0.0, "frozen w_I=10 ": 10.0}.items():
        print(f"  {name}", flush=True)
        rows[name] = run(w, batch, ref)

    keys = ["E/atom MAE", "F MAE", "E_intra MAE", "D MAE"]
    print(f"\n  {'':15s} " + "".join(f"{k:>14s}" for k in keys))
    for name, r in rows.items():
        print(f"  {name:15s} " + "".join(f"{r[k]:14.5f}" for k in keys))

    z, f0, f1 = zero["E_intra MAE"], rows["frozen w_I=0  "]["E_intra MAE"], \
                rows["frozen w_I=10 "]["E_intra MAE"]
    print(f"\n  E_intra: predict-zero {z:.4f}  w_I=0 {f0:.4f}  w_I=10 {f1:.4f}")
    if f1 < 0.5 * z:
        print(f"  => gauge IS imposable on frozen OMol25 features ({f0/max(f1,1e-12):.1f}x)")
    else:
        print("  => NOT CONVERGED or features insufficient. No conclusion.")
    dz, d1 = zero["D MAE"], rows["frozen w_I=10 "]["D MAE"]
    print(f"  D_ij: {d1:.4f} vs |D| {dz:.4f}  ({100*d1/dz:.0f}% relative) "
          f"-- S13.2 saw 30%")
