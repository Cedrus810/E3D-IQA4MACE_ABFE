"""Do the four constraints cooperate, or fight?

Run: python test_joint.py [config] [n_train] [steps]

S13.4 trained on L_int alone, so the decomposition reproduced interaction
energies but was not a potential: nothing held the total energy. This adds the
anchor and asks whether pinning sum D_AB costs total-energy fidelity, the way
IQA supervision costs a few percent in S13.1.

Targets, with the backbone frozen throughout:
  E, F     the backbone's OWN energy and forces, minus per-element atomic
           references. This is RESEARCH_PLAN_v2.md S13.6 option D: keep MACE's
           energy exactly, make the head's decomposition reproduce it.
  E_int    DES370K CCSD(T)/CBS. Real.
  E_intra  a fixed random teacher head. Synthetic placeholder until real IQA
           labels exist -- it fixes *a* node gauge, not the IQA one.

Conditions:
  A  L_E + L_F                    can the head carry the backbone's energy?
  B  L_E + L_F + L_int            does L_int cost energy accuracy?
  C  L_E + L_F + L_IQA + L_int    all four at once

Read-only on /home/ruigengji/MLP/mace and data/; downloads nothing.
"""

import json
import os
import pathlib
import re
import sys
import time

import numpy as np
import torch

import decomp  # noqa: F401
from decomp.data import KCAL_PER_MOL_IN_EV, mace_batch
from decomp.losses import cross_fragment_sum, interaction_loss, sapt_loss
from decomp.clusters import coupling_labels, solvent_clusters
from decomp.mace_adapter import MACEDecomposition
from test_lint import BATCH, DEV, MODEL, OUT

SAPT_N = 4
SAPT_SCALE = None   # dataset-level mean square per channel; set in __main__      # es, ex, ind, disp


def load_split(config, split, n=None, seed=0):
    """Like test_lint.load_split, plus the SAPT columns."""
    z = np.load(f"data/des370k_{config}_{split}.npz", allow_pickle=True)
    pos, num, frag = list(z["positions"]), list(z["numbers"]), list(z["frags"])
    E, q = z["energies"], z["charges"]
    sysid = z["system_id"] if "system_id" in z else np.zeros(len(pos))
    sapt = z["sapt"][:, :SAPT_N] if "sapt" in z else np.full((len(pos), SAPT_N), np.nan)
    if n is not None and n < len(pos):
        i = np.random.default_rng(seed).choice(len(pos), n, replace=False)
        pos = [pos[k] for k in i]; num = [num[k] for k in i]
        frag = [frag[k] for k in i]
        E, q, sysid, sapt = E[i], q[i], sysid[i], sapt[i]
    return pos, num, frag, E, q, sysid, sapt

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


def atomic_ref(bb, node_attrs):
    """Per-element reference energy, so the target is O(10) eV not O(-10000)."""
    e = bb.atomic_energies_fn.atomic_energies
    e = e[0] if e.dim() > 1 else e
    return (node_attrs * e.to(node_attrs.dtype)).sum(-1)


def make_targets(bb, teacher, b):
    """Backbone's own E/F (referenced), plus the teacher's node gauge."""
    pos = b["positions"].clone().requires_grad_(True)
    d = {**b, "positions": pos}
    out = bb(d, compute_force=True)
    ref = atomic_ref(bb, b["node_attrs"])
    n = int(b["batch"].max()) + 1
    E = out["energy"].detach() - torch.zeros(n, device=DEV).index_add(0, b["batch"], ref)
    F = out["forces"].detach()
    with torch.enable_grad():
        t = teacher(d, compute_force=False)
    return E, F, t["E_intra"].detach()


def _atom_batches(sizes, max_n, max_atoms):
    """Consecutive index groups capped by count AND by total atoms.

    DES370K dimers are <= 34 atoms, so BATCH * 34 never binds on them and a
    dimer-only run groups exactly as before. Backbone-labelled clusters run to
    113 atoms; without the atom cap a batch of 16 is 5x a dimer batch and
    POLAR-1-L, already at 11 GiB on dimers, cannot hold it.
    """
    out, cur, na = [], [], 0
    for i, s in enumerate(sizes):
        if cur and (len(cur) == max_n or na + s > max_atoms):
            out.append(cur)
            cur, na = [], 0
        cur.append(i)
        na += s
    if cur:
        out.append(cur)
    return out


def precompute(bb, teacher, data, tag="", cache_path=None):
    """Targets depend only on frozen modules and fixed geometries, so compute
    them once. Doing it inside the training loop runs the backbone three times
    per step instead of one -- the dominant cost, and pure waste.

    Cached to disk as well, so resuming a run or training one condition at a
    time does not repeat the ~4.5 minutes of setup.
    """
    if cache_path is not None and pathlib.Path(cache_path).exists():
        t0 = time.perf_counter()
        blob = torch.load(cache_path, map_location=DEV, weights_only=False)
        print(f"      loaded {len(blob)} cached target batches in "
              f"{time.perf_counter()-t0:.0f}s{tag}", flush=True)
        return blob
    pos_l, num_l, frag_l, E_l, q_l = data[:5]
    out, t0 = [], time.perf_counter()
    for j in _atom_batches([len(p) for p in pos_l], BATCH, BATCH * 34):
        b = mace_batch([pos_l[i] for i in j], [num_l[i] for i in j], bb,
                       charges=[int(q_l[i]) for i in j],
                       frag_ids=[frag_l[i] for i in j], device=DEV)
        b["positions"] = b["positions"].requires_grad_(True)
        E, F, I = make_targets(bb, teacher, b)
        out.append((j, E, F, I))
    print(f"      precomputed {len(pos_l):,} targets in "
          f"{time.perf_counter()-t0:.0f}s{tag}", flush=True)
    if cache_path is not None:
        torch.save([(j, E.cpu(), F.cpu(), I.cpu()) for j, E, F, I in out], cache_path)
        out = [(j, E.to(DEV), F.to(DEV), I.to(DEV)) for j, E, F, I in out]
    return out


def run(tag, w_E, w_F, w_I, w_int, w_sapt, train, test, steps, teacher,
        lr=1e-3, seed=7, hidden=64, cache=None, ckpt=None):
    torch.manual_seed(seed)
    bb = backbone()
    nc = SAPT_N if w_sapt else 1
    model = MACEDecomposition(bb, hidden=hidden, n_pair_channels=nc).to(DEV)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    pos_l, num_l, frag_l, E_l, q_l = train[:5]
    clu = train[7] if len(train) > 7 else None
    ms_dimer = float(np.mean(E_l[~clu] ** 2)) if clu is not None \
        else float(np.mean(E_l ** 2))
    ms_clu = float(np.mean(E_l[clu] ** 2)) if clu is not None and clu.any() else 1.0
    rng = np.random.default_rng(0)
    # Absolute, not steps//8: at 32,000 steps that was one line every
    # 4,000 steps -- 15 minutes of silence that looks like a hang.
    t0, step, every = time.perf_counter(), 0, min(250, max(1, steps // 8))
    run_loss = []   # print a moving average: a single batch swings 2x on its own
    n_batches = len(cache)

    while step < steps:
        j, E_ref, F_ref, I_ref = cache[int(rng.integers(n_batches))]
        b = mace_batch([pos_l[i] for i in j], [num_l[i] for i in j], bb,
                       charges=[int(q_l[i]) for i in j],
                       frag_ids=[frag_l[i] for i in j], device=DEV)
        b["positions"] = b["positions"].requires_grad_(True)
        Eint_ref = torch.as_tensor(E_l[j], dtype=torch.float32, device=DEV)
        sapt_ref = torch.as_tensor(train[6][j], dtype=torch.float32, device=DEV)

        out = model(b, compute_force=True)
        ng = int(b["batch"].max()) + 1
        na = torch.bincount(b["batch"], minlength=ng).float()

        loss = w_E * (((out["energy"] - E_ref) / na) ** 2).mean() / 0.01
        loss = loss + w_F * ((out["forces"] - F_ref) ** 2).mean() / F_ref.pow(2).mean().clamp(min=1e-12)
        if w_I:
            loss = loss + w_I * ((out["E_intra"] - I_ref) ** 2).mean() / I_ref.pow(2).mean().clamp(min=1e-12)
        if w_int:
            eb = b["batch"][b["edge_index"][0]]
            if clu is None:
                li, _ = interaction_loss(out["D"], b["edge_index"], b["frag_id"],
                                         eb, Eint_ref, ng)
                loss = loss + w_int * li / ms_dimer
            else:
                # Separate normalisers. Cluster couplings are ~1 eV and dimer
                # E_int ~0.02 eV, so one shared mean-square would put the
                # CCSD(T) anchor 3 orders of magnitude below the backbone
                # labels and quietly drop the only real data in the term.
                m = torch.as_tensor(clu[j], dtype=torch.bool, device=DEV)
                ld, _ = interaction_loss(out["D"], b["edge_index"], b["frag_id"],
                                         eb, Eint_ref, ng, mask=~m)
                lc, _ = interaction_loss(out["D"], b["edge_index"], b["frag_id"],
                                         eb, Eint_ref, ng, mask=m)
                loss = loss + w_int * (ld / ms_dimer + lc / ms_clu)
        if w_sapt:
            # The components pin how D is spread over the boundary (S13.10,
            # S13.11); the total above pins the cancelling residue. Both, or
            # neither works.
            ls, _ = sapt_loss(out["D"], b["edge_index"], b["frag_id"],
                              b["batch"][b["edge_index"][0]], sapt_ref, ng,
                              scale=SAPT_SCALE)
            loss = loss + w_sapt * ls

        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        step += 1
        run_loss.append(float(loss))
        if step == 1 or step % every == 0:
            print(f"      step {step:6d}/{steps}  "
                  f"loss {np.mean(run_loss[-every:]):8.4f} "
                  f"[{time.perf_counter()-t0:4.0f}s]", flush=True)

    r = evaluate(model, bb, teacher, test)
    if ckpt is not None:
        torch.save(model.head.state_dict(), ckpt)
        print(f"      head saved -> {ckpt}", flush=True)
    return r


@torch.enable_grad()
def evaluate(model, bb, teacher, data):
    pos_l, num_l, frag_l, E_l, q_l = data[:5]
    eE, eF, eI, eX, eC = [], [], [], [], []
    for k in range(0, len(pos_l), BATCH):
        j = list(range(k, min(k + BATCH, len(pos_l))))
        b = mace_batch([pos_l[i] for i in j], [num_l[i] for i in j], bb,
                       charges=[int(q_l[i]) for i in j],
                       frag_ids=[frag_l[i] for i in j], device=DEV)
        b["positions"] = b["positions"].requires_grad_(True)
        E_ref, F_ref, I_ref = make_targets(bb, teacher, b)
        out = model(b, compute_force=True)
        ng = int(b["batch"].max()) + 1
        na = torch.bincount(b["batch"], minlength=ng).float()
        eb = b["batch"][b["edge_index"][0]]
        sc = cross_fragment_sum(out["D"], b["edge_index"], b["frag_id"], eb, ng)
        pred = sc.sum(-1) if sc.dim() > 1 else sc
        if sc.dim() > 1:
            sref = torch.as_tensor(data[6][j], dtype=torch.float32, device=DEV)
            eC.append((sc - sref).abs().detach().cpu())
        eE.append(((out["energy"] - E_ref).abs() / na).detach().cpu())
        eF.append((out["forces"] - F_ref).abs().mean(-1).detach().cpu())
        eI.append((out["E_intra"] - I_ref).abs().detach().cpu())
        eX.append((pred - torch.as_tensor(E_l[j], dtype=torch.float32,
                                          device=DEV)).abs().detach().cpu())
    r = {"E/atom": torch.cat(eE).mean().item(),
         "F": torch.cat(eF).mean().item(),
         "E_intra": torch.cat(eI).mean().item(),
         "E_int": torch.cat(eX).mean().item()}
    if eC:
        r["per_channel"] = torch.cat(eC).nanmean(0).tolist()
    return r


if __name__ == "__main__":
    # Arguments are recognised by shape, not position, so
    #   test_joint.py organic C
    #   test_joint.py organic 50000 32000 256 C
    # both work and a misplaced selector cannot be parsed as a number.
    argv = sys.argv[1:]
    want = "ABCD"
    rest = []
    for a in argv:
        if re.fullmatch(r"[ABCDabcd]+", a):
            want = a.upper()
        else:
            rest.append(a)
    cfg = rest[0] if len(rest) > 0 else "hcno_small"
    n_tr = int(rest[1]) if len(rest) > 1 else 20000
    steps = int(rest[2]) if len(rest) > 2 else 6000
    hidden = int(rest[3]) if len(rest) > 3 else 64
    print(f"  cases={want}  cfg={cfg}  n_train={n_tr:,}  steps={steps:,}  "
          f"hidden={hidden}", flush=True)

    print(f"Joint training [{cfg}]  (device {DEV})", flush=True)
    train = load_split(cfg, "train", n_tr)
    _sp = train[6] if len(train) > 6 else None
    if _sp is not None:
        _ok = np.isfinite(_sp).all(-1)
        SAPT_SCALE = (_sp[_ok] ** 2).mean(0) if _ok.any() else None

    test = load_split(cfg, "test", 1000)
    print(f"  train {len(train[0]):,} ({len(set(train[5])):,} sys)  "
          f"test {len(test[0]):,} ({len(set(test[5])):,} sys)", flush=True)

    bb = backbone()
    torch.manual_seed(123)
    teacher = MACEDecomposition(bb).to(DEV).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Many-body structures, labelled by the backbone itself. Dimers pin only
    # the SUM over a few dozen cross edges, which leaves a per-edge bias free
    # to grow with the system -- measured at -2.0 meV/edge, i.e. -3.1 eV of a
    # -1.5 eV coupling on a 341-atom carve. See decomp/clusters.py.
    n_clu = int(os.environ.get("E3D_CLUSTERS", 0))
    clu = np.zeros(len(train[0]), dtype=bool)
    if n_clu:
        t0 = time.perf_counter()
        cp, cn, cf = solvent_clusters(n=n_clu, rng=np.random.default_rng(1))
        # ~0.19 s per label in float64 (three backbone passes each), so 10,000
        # is half an hour every restart unless it lands on disk.
        lab = OUT / f"data/clusters_{n_clu}.npy"
        if lab.exists():
            ce = np.load(lab)
        else:
            ce = coupling_labels(bb, cp, cn, cf, device=DEV)
            (OUT / "data").mkdir(parents=True, exist_ok=True)
            np.save(lab, ce)
        train = (train[0] + cp, train[1] + cn, train[2] + cf,
                 np.concatenate([train[3], ce]),
                 np.concatenate([train[4], np.zeros(n_clu, dtype=train[4].dtype)]),
                 np.concatenate([train[5], -np.arange(1, n_clu + 1)]),
                 np.concatenate([train[6], np.full((n_clu, SAPT_N), np.nan)]),
                 np.concatenate([clu, np.ones(n_clu, dtype=bool)]))
        print(f"  + {n_clu:,} backbone-labelled clusters in "
              f"{time.perf_counter()-t0:.0f}s: coupling "
              f"{ce.min():+.2f} .. {ce.max():+.2f} eV, "
              f"|mean| {np.abs(ce).mean():.2f}", flush=True)
    else:
        train = tuple(train) + (clu,)

    # The cluster count belongs in every output name: the cached E/F targets
    # and the trained head both depend on it, and reusing a dimer-only cache
    # under a cluster run is silent and wrong.
    tag = f"_c{n_clu}" if n_clu else ""
    stem = f"{cfg}_n{n_tr}{tag}_h{hidden}_s{steps}"
    for d in ("ckpt", "logs", "data"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    results_path = OUT / f"logs/joint_{stem}.json"
    rows = json.loads(results_path.read_text()) if results_path.exists() else {}
    if rows:
        print(f"  resuming: {sorted(rows)} already done", flush=True)

    cache = precompute(bb, teacher, train, " (train)",
                       cache_path=OUT / f"data/targets_{cfg}_n{n_tr}{tag}.pt")
    cases = {"A  E+F        ": (1, 1, 0, 0, 0),
             "B  +L_int     ": (1, 1, 0, 1, 0),
             "C  +L_IQA+int ": (1, 1, 0.1, 1, 0),
             "D  +SAPT      ": (1, 1, 0.1, 1, 1)}
    for tag, w in cases.items():
        if tag.strip()[0] not in want:
            continue
        if tag in rows:
            print(f"  {tag}  cached, skipping", flush=True)
            continue
        print(f"  {tag}  hidden={hidden}", flush=True)
        rows[tag] = run(tag, *w, train, test, steps, teacher,
                        hidden=hidden, cache=cache,
                        ckpt=OUT / f"ckpt/joint_{stem}_{tag.strip()[0]}.pt")
        results_path.write_text(json.dumps(rows, indent=2))

    if not all(k in rows for k in cases):
        print(f"\n  {sorted(rows)} done, rest pending -- rerun to continue")
        sys.exit(0)

    keys = ["E/atom", "F", "E_intra", "E_int"]
    print(f"\n  {'':15s}" + "".join(f"{k:>13s}" for k in keys) + f"{'E_int':>12s}")
    for tag, r in rows.items():
        print(f"  {tag:15s}" + "".join(f"{r[k]:13.5f}" for k in keys)
              + f"{r['E_int']/KCAL_PER_MOL_IN_EV:9.2f} kcal")

    K = KCAL_PER_MOL_IN_EV
    if "D  +SAPT      " in rows and "per_channel" in rows["D  +SAPT      "]:
        names = ["es", "ex", "ind", "disp"]
        print("\n  per-channel MAE (kcal/mol), arm D:")
        print("    " + "".join(f"{n:>9s}" for n in names))
        print("    " + "".join(f"{v/K:9.3f}"
                               for v in rows["D  +SAPT      "]["per_channel"]))
    for lo, hi, what in (("A  E+F        ", "B  +L_int     ", "L_int"),
                         ("B  +L_int     ", "C  +L_IQA+int ", "L_IQA"),
                         ("C  +L_IQA+int ", "D  +SAPT      ", "SAPT")):
        if lo in rows and hi in rows:
            x, y = rows[lo]["E/atom"], rows[hi]["E/atom"]
            print(f"  cost of {what:6s} on total energy: {x:.5f} -> {y:.5f} "
                  f"eV/atom ({100*(y/max(x,1e-12)-1):+.0f}%)")
    print("  E_int: " + "   ".join(
        f"{t.strip()[0]} {r['E_int']/K:.2f}" for t, r in rows.items()) + " kcal/mol")
