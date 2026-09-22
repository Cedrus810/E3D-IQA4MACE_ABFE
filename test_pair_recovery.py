"""Does component supervision pin the pair distribution? (S13.9 follow-up)

Run: python test_pair_recovery.py [n_struct] [steps]

S13.9 asked whether SAPT components constrain how D_ij is spread over the
boundary, but measured the total interaction energy instead -- because no
reference per-pair decomposition exists to compare against. The teacher/student
construction removes that obstacle: a fixed random 4-channel teacher head IS a
reference decomposition, representable by the student by construction, so
D_ij recovery can be measured directly.

Four arms, identical budget and identical architecture (4 channels throughout,
so the comparison is supervision, not parameterisation):

    total        supervise the summed boundary total only
    components   supervise the 4 channel boundary sums
    both         supervise components and total
    oracle       supervise D_ij edge by edge -- the ceiling

The question is whether `components` moves D_ij recovery toward `oracle`
relative to `total`. S13.2/S13.3/S13.5 put total-only recovery at 28-36% of
|D| regardless of data, capacity or weight; if component supervision does not
move that, the per-atom route needs something other than SAPT.

No external data, no downloads.
"""

import sys
import time

import numpy as np
import torch

import decomp  # noqa: F401
from decomp.data import mace_batch
from decomp.losses import cross_fragment_sum
from decomp.mace_adapter import MACEDecomposition
from e3nn import o3
from mace.modules import MACE, gate_dict, interaction_classes

DEV = "cuda" if torch.cuda.is_available() else "cpu"
Z = [1, 8]
R_MAX = 5.0
NC = 4


def small_mace():
    return MACE(
        r_max=R_MAX, num_bessel=8, num_polynomial_cutoff=6, max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=len(Z),
        hidden_irreps=o3.Irreps("16x0e+16x1o"), MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros(len(Z)), avg_num_neighbors=8.0,
        atomic_numbers=Z, correlation=2, gate=gate_dict["silu"],
    ).to(DEV)


def dimers(n, seed=1):
    g = np.random.default_rng(seed)
    base = np.array([[0., 0., 0.], [0.758, 0.587, 0.], [-0.758, 0.587, 0.],
                     [2.9, 0.1, 0.3], [3.5, 0.6, -0.2], [3.2, -0.7, 0.5]])
    return [base + 0.12 * g.standard_normal((6, 3)) for _ in range(n)]


def run(arm, bb, teacher_D, batches, steps, lr=5e-3, seed=7):
    torch.manual_seed(seed)
    model = MACEDecomposition(bb, hidden=64, n_pair_channels=NC).to(DEV)
    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        loss = 0.0
        for b, Dref in zip(batches, teacher_D):
            D = model(b, compute_force=False)["D"]
            ng = int(b["batch"].max()) + 1
            eb = b["batch"][b["edge_index"][0]]
            if arm == "oracle":
                loss = loss + ((D - Dref) ** 2).mean() / Dref.pow(2).mean()
                continue
            s = cross_fragment_sum(D, b["edge_index"], b["frag_id"], eb, ng)
            sref = cross_fragment_sum(Dref, b["edge_index"], b["frag_id"], eb, ng)
            if arm in ("total", "both"):
                loss = loss + (((s.sum(-1) - sref.sum(-1)) ** 2).mean()
                               / sref.sum(-1).pow(2).mean().clamp(min=1e-12))
            if arm in ("components", "both"):
                loss = loss + (((s - sref) ** 2).mean(0)
                               / sref.pow(2).mean(0).clamp(min=1e-12)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()

    with torch.no_grad():
        num = den = cross_num = cross_den = 0.0
        for b, Dref in zip(batches, teacher_D):
            D = model(b, compute_force=False)["D"]
            num += (D - Dref).abs().sum().item(); den += Dref.abs().sum().item()
            src, dst = b["edge_index"]
            c = b["frag_id"][src] != b["frag_id"][dst]
            cross_num += (D[c] - Dref[c]).abs().sum().item()
            cross_den += Dref[c].abs().sum().item()
    del model, opt
    torch.cuda.empty_cache() if DEV == "cuda" else None
    return {"all": num / den, "cross": cross_num / cross_den}


if __name__ == "__main__":
    n_struct = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    torch.manual_seed(0)
    print(f"Pair-distribution recovery under different supervision  ({DEV})")

    bb = small_mace()
    z = np.array([8, 1, 1, 8, 1, 1]); fr = np.array([0, 0, 0, 1, 1, 1])
    confs = dimers(n_struct)
    batches = [mace_batch(confs[k:k + 16], [z] * len(confs[k:k + 16]), bb,
                          frag_ids=[fr] * len(confs[k:k + 16]), device=DEV)
               for k in range(0, n_struct, 16)]

    torch.manual_seed(123)
    teacher = MACEDecomposition(bb, hidden=64, n_pair_channels=NC).to(DEV).eval()
    with torch.no_grad():
        teacher_D = [teacher(b, compute_force=False)["D"].detach() for b in batches]
    scale = torch.cat(teacher_D).abs().mean().item()
    print(f"  {n_struct} dimers, {NC} channels, {steps} steps, "
          f"teacher |D| = {scale:.4f}\n")
    del teacher

    rows = {}
    for arm in ("total", "components", "both", "oracle"):
        t0 = time.perf_counter()
        rows[arm] = run(arm, bb, teacher_D, batches, steps)
        print(f"  {arm:11s} D recovery: all edges {100*rows[arm]['all']:6.1f}%   "
              f"cross-boundary {100*rows[arm]['cross']:6.1f}%   "
              f"[{time.perf_counter()-t0:4.0f}s]", flush=True)

    t, c, o = (rows["total"]["cross"], rows["components"]["cross"],
               rows["oracle"]["cross"])
    print(f"\n  cross-boundary relative error, the quantity per-atom "
          f"attribution needs:")
    print(f"    total-only  {100*t:.1f}%   components {100*c:.1f}%   "
          f"both {100*rows['both']['cross']:.1f}%   oracle {100*o:.1f}%")
    closed = (t - c) / max(t - o, 1e-12)
    print(f"  => component supervision closes {100*closed:.0f}% of the gap "
          f"between total-only and the oracle")
