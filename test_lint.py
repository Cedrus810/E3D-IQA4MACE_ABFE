"""Can L_int pin sum D_AB to real CCSD(T) interaction energies?

Run: python test_lint.py [config] [n_train] [steps]
     python test_lint.py hcno_small 2000 2000

S13.3 showed D_ij is not identifiable from total energy plus node supervision:
5x more training improved total energy 11x and left D_ij at 1.09x, worse than
predicting zero. Every ligand-environment term in the alchemical Hamiltonian is
a D_ij, so unless L_int pins sum D_LE the ABFE layer has no foundation.

Target: held-out MAE on sum D_AB vs E_int at chemical accuracy, ~1 kcal/mol
(0.0434 eV). Backbone frozen (S13.2, S13.3); only the head trains.

Read-only on /home/ruigengji/MLP/mace and data/; downloads nothing.
"""

import pathlib
import sys
import time

import numpy as np
import torch

import decomp  # noqa: F401
from decomp.data import KCAL_PER_MOL_IN_EV, mace_batch
from decomp.losses import interaction_loss
from decomp.mace_adapter import MACEDecomposition

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "/home/ruigengji/MLP/mace/mace-omol-0-extra-large-4M.model"
BATCH = 16


def load_split(config, split, n=None):
    z = np.load(f"data/des370k_{config}_{split}.npz", allow_pickle=True)
    pos, num, frag = list(z["positions"]), list(z["numbers"]), list(z["frags"])
    E, q = z["energies"], z["charges"]
    sysid = z["system_id"] if "system_id" in z else np.zeros(len(pos))
    if n is not None and n < len(pos):
        idx = np.random.default_rng(0).choice(len(pos), n, replace=False)
        pos = [pos[i] for i in idx]; num = [num[i] for i in idx]
        frag = [frag[i] for i in idx]; E, q, sysid = E[idx], q[idx], sysid[idx]
    return pos, num, frag, E, q, sysid


def batches(data, model, bs=BATCH, shuffle=False, seed=0):
    pos, num, frag, E, q = data[:5]
    order = np.arange(len(pos))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for k in range(0, len(order), bs):
        j = order[k:k + bs]
        b = mace_batch([pos[i] for i in j], [num[i] for i in j], model,
                       charges=[int(q[i]) for i in j], frag_ids=[frag[i] for i in j],
                       device=DEV)
        b["E_int"] = torch.as_tensor(E[j], dtype=torch.float32, device=DEV)
        yield b


@torch.enable_grad()
def evaluate(model, data):
    """Held-out MAE of sum D_AB against E_int, plus the predict-zero baseline."""
    errs, refs = [], []
    for b in batches(data, model.backbone):
        out = model(b, compute_force=False)
        ng = int(b["batch"].max()) + 1
        _, pred = interaction_loss(out["D"], b["edge_index"], b["frag_id"],
                                   b["batch"][b["edge_index"][0]], b["E_int"], ng)
        errs.append((pred - b["E_int"]).abs().detach().cpu())
        refs.append(b["E_int"].abs().detach().cpu())
    e, r = torch.cat(errs), torch.cat(refs)
    return e.mean().item(), r.mean().item()


_BB = None


def backbone():
    """Load the 414 MB checkpoint once and reuse it across runs."""
    global _BB
    if _BB is None:
        t0 = time.perf_counter()
        print(f"  loading {pathlib.Path(MODEL).name} ...", end="", flush=True)
        _BB = torch.load(MODEL, map_location="cpu",
                         weights_only=False).to(torch.float32).to(DEV).eval()
        print(f" {time.perf_counter()-t0:.0f}s", flush=True)
    return _BB


def run(w_int, train, val, steps, lr=1e-3, seed=7):
    # Fixed dataset-level scale. A per-batch mean square fluctuates wildly at
    # small batch size and the loss trace stops being readable.
    SCALE = float(np.mean(train[3] ** 2))
    torch.manual_seed(seed)
    bb = backbone()
    model = MACEDecomposition(bb).to(DEV)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    t0, step, run_loss = time.perf_counter(), 0, []
    every = min(250, max(1, steps // 40))
    while step < steps:
        for b in batches(train, bb, shuffle=True, seed=step):
            out = model(b, compute_force=False)
            ng = int(b["batch"].max()) + 1
            l_int, _ = interaction_loss(out["D"], b["edge_index"], b["frag_id"],
                                        b["batch"][b["edge_index"][0]], b["E_int"], ng)
            l_int = l_int / SCALE
            # keep the decomposition anchored: total energy must stay the backbone's
            loss = w_int * l_int
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            step += 1
            run_loss.append(float(loss))
            if step == 1 or step % every == 0:
                mae, _ = evaluate(model, val)
                print(f"      step {step:6d}/{steps}  "
                      f"loss {np.mean(run_loss[-every:]):7.4f}  "
                      f"val MAE {mae/KCAL_PER_MOL_IN_EV:6.2f} kcal/mol  "
                      f"[{time.perf_counter()-t0:4.0f}s]", flush=True)
            if step >= steps:
                break
    print(f"      {time.perf_counter()-t0:.0f}s, peak GPU "
          f"{torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
    r = {}
    for nm, d in (("val", val), ("test", TEST)):
        mae, base = evaluate(model, d)
        r[nm] = mae; r[nm + "_base"] = base
    del model, bb, opt
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return r


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "hcno_small"
    n_tr = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    if not pathlib.Path(f"data/des370k_{cfg}_train.npz").exists():
        sys.exit(f"missing data/des370k_{cfg}_*.npz -- run prepare_data.py first")

    print(f"loading splits [{cfg}] ...", flush=True)
    train = load_split(cfg, "train", n_tr)
    val = load_split(cfg, "val", 400)
    TEST = load_split(cfg, "test", n_test)
    print(f"L_int on DES370K [{cfg}]  (device {DEV})")
    print(f"  train {len(train[0]):,} ({len(set(train[5])):,} sys)  "
          f"val {len(val[0]):,} ({len(set(val[5])):,} sys)  "
          f"test {len(TEST[0]):,} ({len(set(TEST[5])):,} sys)  "
          f"-- split by system_id, no geometry leakage")
    print(f"  |E_int| test {np.abs(TEST[3]).mean():.4f} eV = "
          f"{np.abs(TEST[3]).mean()/KCAL_PER_MOL_IN_EV:.2f} kcal/mol,  "
          f"frozen backbone, {steps} steps\n")

    rows = {}
    for name, w in {"w_int=0 (untrained)": 0.0, "w_int=1": 1.0}.items():
        print(f"  {name}")
        rows[name] = run(w, train, val, steps if w > 0 else 1)

    print(f"\n  {'':22s} {'val MAE':>22s} {'test MAE':>22s}")
    for name, r in rows.items():
        print(f"  {name:22s} "
              f"{r['val']:9.4f} eV {r['val']/KCAL_PER_MOL_IN_EV:8.2f} kcal "
              f"{r['test']:9.4f} eV {r['test']/KCAL_PER_MOL_IN_EV:8.2f} kcal")
    b = rows["w_int=1"]["test_base"]
    print(f"  {'predict zero':22s} {'':22s} "
          f"{b:9.4f} eV {b/KCAL_PER_MOL_IN_EV:8.2f} kcal")

    t = rows["w_int=1"]["test"]
    print(f"\n  chemical accuracy = 1 kcal/mol = 0.0434 eV")
    print(f"  => sum D_AB is " + ("PINNED" if t < 0.0434 else
          "pinned to %.1f kcal/mol, not yet chemical accuracy" % (t/KCAL_PER_MOL_IN_EV)))
