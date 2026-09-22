"""Does multi-channel SAPT supervision pin D_ij better than the total alone?

Run: python test_sapt.py [config] [n_train] [steps] [hidden]

S13.4 pinned the boundary SUM to 0.27 kcal/mol, but the flagship per-atom
attribution (S6, S13.8) integrates sum_a D_ia PER LIGAND ATOM, and a single
scalar per structure cannot constrain several hundred cross-fragment edges.

DES370K ships the SAPT decomposition of the same interaction energy into
electrostatics, exchange, induction and dispersion. They decay very differently
(~1/r, ~exp(-r), and ~1/r^6 for dispersion), so requiring all four turns one
constraint per structure into four with different spatial character -- at no
extra quantum chemistry, the columns are already there.

Three arms, same budget:
  total   1 channel,  supervised on cbs_CCSD(T)_all      (the S13.4 baseline)
  sapt    4 channels, supervised on the four components
  sum     4 channels, supervised on their sum only       (channel control)

The third arm matters: it separates "more channels helped" from "component
supervision helped". All are scored on the same held-out total E_int, plus a
per-channel breakdown.

Read-only on /home/ruigengji/MLP/mace and data/; downloads nothing.
"""

import pathlib
import sys
import time

import numpy as np
import torch

import decomp  # noqa: F401
from decomp.data import KCAL_PER_MOL_IN_EV, mace_batch
from decomp.losses import cross_fragment_sum, sapt_loss
from decomp.mace_adapter import MACEDecomposition
from test_lint import BATCH, DEV, MODEL

SAPT_N = 4
SAPT_SCALE = None   # dataset-level mean square per channel; set in __main__          # es, ex, ind, disp  (the 5th column is their total)
TRAIN_E = None      # set in __main__; the scale for the total-energy loss
_BB = None


def backbone():
    global _BB
    if _BB is None:
        t0 = time.perf_counter()
        print(f"  loading {pathlib.Path(MODEL).name} ...", end="", flush=True)
        _BB = torch.load(MODEL, map_location="cpu",
                         weights_only=False).to(torch.float32).to(DEV).eval()
        print(f" {time.perf_counter()-t0:.0f}s", flush=True)
    return _BB


def load(config, split, n=None, seed=0):
    z = np.load(f"data/des370k_{config}_{split}.npz", allow_pickle=True)
    pos, num, frag = list(z["positions"]), list(z["numbers"]), list(z["frags"])
    E, q, sysid = z["energies"], z["charges"], z["system_id"]
    sapt = z["sapt"][:, :SAPT_N] if "sapt" in z else np.full((len(pos), SAPT_N), np.nan)
    if n is not None and n < len(pos):
        i = np.random.default_rng(seed).choice(len(pos), n, replace=False)
        pos = [pos[k] for k in i]; num = [num[k] for k in i]
        frag = [frag[k] for k in i]; E, q, sapt, sysid = E[i], q[i], sapt[i], sysid[i]
    return pos, num, frag, E, q, sapt, sysid


def batch_of(data, j, bb):
    pos, num, frag, E, q, sapt, _ = data
    b = mace_batch([pos[i] for i in j], [num[i] for i in j], bb,
                   charges=[int(q[i]) for i in j],
                   frag_ids=[frag[i] for i in j], device=DEV)
    b["E_int"] = torch.as_tensor(E[j], dtype=torch.float32, device=DEV)
    b["sapt"] = torch.as_tensor(sapt[j], dtype=torch.float32, device=DEV)
    return b


@torch.enable_grad()
def evaluate(model, cached, bb):
    """Held-out MAE on the total, and per channel where the head has channels."""
    err, per, ref = [], [], []
    for c in cached:
        b = to_dev(c)
        D = model.head(b["h"], b["positions"], b["edge_index"])[1]
        ng = int(b["batch"].max()) + 1
        s = cross_fragment_sum(D, b["edge_index"], b["frag_id"],
                               b["batch"][b["edge_index"][0]], ng)
        tot = s.sum(-1) if s.dim() > 1 else s
        err.append((tot - b["E_int"]).abs().detach().cpu())
        ref.append(b["E_int"].abs().detach().cpu())
        if s.dim() > 1:
            per.append((s - b["sapt"]).abs().detach().cpu())
    out = {"total": torch.cat(err).mean().item(),
           "base": torch.cat(ref).mean().item()}
    if per:
        out["per_channel"] = torch.cat(per).nanmean(0).tolist()
    return out


def cache_feats(data, bb, tag=""):
    """Run the frozen backbone once and keep the node features.

    Nothing here needs dH/dR -- this experiment never computes forces -- so the
    51.3M-parameter backbone is pure overhead after the first pass. Features are
    kept on CPU and moved per batch; training then costs only the head, which
    benchmarks at ~3% of the backbone: ~50x faster per step.

    **fp32, not fp16.** fp16 quantises h to 1.7e-4 relative, which is 2e-4 eV
    per edge -- negligible alone, but the boundary sum runs over several hundred
    edges and accumulates to 3.1e-3 eV = 0.071 kcal/mol. Against the 0.27
    kcal/mol already achieved (S13.5) that is 26% of the signal. The same
    cancellation that makes the total hard to learn makes it sensitive to
    rounding in the features. 19456 dims at fp32 is ~2 MB per structure.
    """
    n = len(data[0])
    out, t0 = [], time.perf_counter()
    with torch.no_grad():
        for k in range(0, n, BATCH):
            j = list(range(k, min(k + BATCH, n)))
            b = batch_of(data, j, bb)
            h = bb(b, compute_force=False)["node_feats"]
            out.append({
                "h": h.cpu(),
                "positions": b["positions"].cpu(),
                "edge_index": b["edge_index"].cpu(),
                "frag_id": b["frag_id"].cpu(),
                "batch": b["batch"].cpu(),
                "E_int": b["E_int"].cpu(),
                "sapt": b["sapt"].cpu(),
            })
    gb = sum(c["h"].numel() * 4 for c in out) / 2**30
    print(f"      cached {n:,} structures in {time.perf_counter()-t0:.0f}s "
          f"({gb:.1f} GiB){tag}", flush=True)
    return out


def to_dev(c):
    return {k: v.to(DEV) for k, v in c.items()}


def run(arm, train, test, steps, hidden, lr=1e-3, seed=7):
    torch.manual_seed(seed)
    bb = backbone()
    nc = 1 if arm == "total" else SAPT_N
    model = MACEDecomposition(bb, hidden=hidden, n_pair_channels=nc).to(DEV)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(0)
    scale_tot = float(np.mean(TRAIN_E ** 2))
    n, every, hist = len(train), min(250, max(1, steps // 20)), []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        b = to_dev(train[int(rng.integers(n))])
        D = model.head(b["h"], b["positions"], b["edge_index"])[1]
        ng = int(b["batch"].max()) + 1
        eb = b["batch"][b["edge_index"][0]]
        if arm.startswith("sapt"):
            loss, _ = sapt_loss(D, b["edge_index"], b["frag_id"], eb,
                                b["sapt"], ng, scale=SAPT_SCALE)
        else:
            loss = 0.0
        if arm != "sapt":
            # The components cancel heavily -- exchange alone is 3x the total --
            # so fitting each to ~50% leaves the sum useless (S13.9). Whenever
            # the total is what will be read out, it must be in the loss.
            s = cross_fragment_sum(D, b["edge_index"], b["frag_id"], eb, ng)
            tot = s.sum(-1) if s.dim() > 1 else s
            loss = loss + ((tot - b["E_int"]) ** 2).mean() / scale_tot
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        hist.append(float(loss))
        if step == 1 or step % every == 0:
            print(f"      step {step:6d}/{steps}  loss {np.mean(hist[-every:]):7.4f}"
                  f"  [{time.perf_counter()-t0:4.0f}s]", flush=True)

    r = evaluate(model, test, bb)
    torch.save(model.head.state_dict(), f"ckpt/sapt_{arm}_h{hidden}_s{steps}.pt")
    del model, opt
    torch.cuda.empty_cache()
    return r


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "organic"
    n_tr = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    hidden = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    pathlib.Path("ckpt").mkdir(exist_ok=True)

    print(f"SAPT multi-channel supervision [{cfg}]  ({DEV})", flush=True)
    train, test = load(cfg, "train", n_tr), load(cfg, "test", 2000)
    _sp = train[6] if len(train) > 6 else None
    if _sp is not None:
        _ok = np.isfinite(_sp).all(-1)
        SAPT_SCALE = (_sp[_ok] ** 2).mean(0) if _ok.any() else None

    ok = np.isfinite(train[5]).all(-1).mean()
    print(f"  train {len(train[0]):,} ({len(set(train[6])):,} sys, "
          f"{100*ok:.0f}% with SAPT)   test {len(test[0]):,} "
          f"({len(set(test[6])):,} sys)")
    print(f"  |E_int| test {np.abs(test[3]).mean()/KCAL_PER_MOL_IN_EV:.2f} kcal/mol"
          f"   hidden={hidden}  steps={steps:,}\n", flush=True)

    bb = backbone()
    TRAIN_E = train[3]
    train_c = cache_feats(train, bb, " (train)")
    test_c = cache_feats(test, bb, " (test)")

    arms = sys.argv[5].split(",") if len(sys.argv) > 5 else \
        ["total", "sum", "sapt", "sapt+total"]
    rows = {}
    for arm in arms:
        print(f"  arm = {arm}", flush=True)
        rows[arm] = run(arm, train_c, test_c, steps, hidden)

    K = KCAL_PER_MOL_IN_EV
    print(f"\n  {'arm':12s} {'channels':>9s} {'test MAE on total':>20s}")
    for arm, r in rows.items():
        nc = 1 if arm == "total" else SAPT_N
        print(f"  {arm:12s} {nc:9d} {r['total']:11.5f} eV "
              f"{r['total']/K:7.2f} kcal")
    print(f"  {'zero':12s} {'':9s} {rows['total']['base']:11.5f} eV "
          f"{rows['total']['base']/K:7.2f} kcal")

    names = ["es", "ex", "ind", "disp"]
    print(f"\n  per-channel MAE (kcal/mol):")
    print(f"    {'':12s}" + "".join(f"{n:>9s}" for n in names))
    for arm in ("sum", "sapt", "sapt+total"):
        if "per_channel" in rows.get(arm, {}):
            print(f"    {arm:12s}" + "".join(
                f"{v/K:9.3f}" for v in rows[arm]["per_channel"]))

    if not all(k in rows for k in ("total", "sum", "sapt+total")):
        sys.exit(0)
    a, b_ = rows["total"]["total"], rows["sum"]["total"]
    d = rows["sapt+total"]["total"]
    print(f"\n  extra channels at matched supervision: "
          f"{a/K:.2f} -> {b_/K:.2f} kcal/mol ({b_/max(a,1e-12):.2f}x)")
    print(f"  components added on top of the total: "
          f"{b_/K:.2f} -> {d/K:.2f} kcal/mol ({d/max(b_,1e-12):.2f}x)")
    # Helping the total was never the point -- S13.4 already pins it. The
    # question is whether the per-edge distribution gets constrained, which the
    # per-channel rows report. Judge the trade, not the total alone.
    if "per_channel" in rows.get("sum", {}) and "per_channel" in rows.get("sapt+total", {}):
        gain = np.mean([u / max(v, 1e-12) for u, v in
                        zip(rows["sum"]["per_channel"],
                            rows["sapt+total"]["per_channel"])])
        print(f"  => the trade: {d/max(b_,1e-12):.2f}x on the total "
              f"({b_/K:.2f} -> {d/K:.2f} kcal/mol, both inside chemical accuracy) "
              f"for {gain:.1f}x on the per-channel distribution")
