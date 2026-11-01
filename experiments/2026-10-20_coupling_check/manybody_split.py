"""Two-body / many-body split of the ligand - inner-shell coupling (v3 S13.57 follow-up).

truth = E(L u I) - E(L) - E(I)                                   POLAR-1-M, float64, CPU
two   = sum_w [E(L + w) - E(L) - E(w)]                           ligand - one water, each water alone
many  = truth - two                                              L-I-I and higher (incl. the ligand-induced
                                                                 change of POLAR's water-water energy)
E_MM  = GAFF2/TIP3P L-I Coulomb + LJ (pairwise, non-periodic)    as in coupling_check.py

Hypothesis (S13.57): the D19 excess (-95 kJ/mol against MM on D19 frames) sits in `many`.
Frames: the same sources as coupling_check.py; the MM MD is regenerated with the same seed and
saved to frames.npz this time (positions may differ from the first run: CUDA MD is not bitwise
reproducible, so the MM rows are compared as an ensemble, not frame by frame).
"""
import json
import pathlib
import sys

import numpy as np
import openmm
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from coupling_check import BB, CKPT, EV, LEG, ONE_4PI_EPS0, ROOT, checkpoint_frames, mm_frames  # noqa: E402

sys.path.insert(0, str(ROOT))
from decomp.fires import Fires, heavy_selection, water_indices  # noqa: E402
from decomp.mlmm import MLMMForce  # noqa: E402
from decomp.openmm_bridge import AlchemicalForce  # noqa: E402
from run_abfe import load_model  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="state.npz files to split at q 0 and 1 instead of the default set; skips the MM MD")
    ap.add_argument("--out", default="manybody_split.json")
    args = ap.parse_args()
    torch.set_num_threads(8)
    d = np.load(LEG, allow_pickle=True)
    system = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
    numbers, box = np.asarray(d["numbers"]), np.asarray(d["box"], float)
    lig = np.where(d["resids"] == d["resids"][0])[0]
    W = water_indices(d["resnames"], d["resids"])
    centre, wsel = heavy_selection(lig, numbers, W)
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    prm = np.array([[p._value for p in nb.getParticleParameters(i)] for i in range(system.getNumParticles())])
    model = load_model(str(CKPT), BB, device="cpu").double()
    bb = model.backbone

    if args.checkpoints:
        frames = []
        for ck in args.checkpoints:
            frames += checkpoint_frames(ROOT / ck)
    else:
        mm = mm_frames(system, np.asarray(d["positions"], float), box)
        np.savez(HERE / "frames.npz", positions=np.stack(mm), box=box)
        frames = [(f"MM {k}", p) for k, p in enumerate(mm)]
        for ck in ("experiments/2026-10-19_d19_acetamide/segA/state.npz",
                   "experiments/2026-10-19_d19_acetamide/full/state.npz",
                   "experiments/2026-10-16_mm_path_control/fires_seed1/state.npz"):
            frames += checkpoint_frames(ROOT / ck)

    cache = {}

    def energy(X, Z):
        key = tuple(Z)
        if key not in cache:
            cache[key] = AlchemicalForce(model, Z, np.zeros(len(Z), int), device="cpu")
        af = cache[key]
        with torch.no_grad():
            return float(bb(af._graph(X), training=False, compute_force=False)["energy"].sum()) * EV

    rows = []
    nl = len(lig)
    for name, P in frames:
        fires = Fires(centre, wsel, n_inner=20, device="cpu")
        inner = fires.select_inner(P, box)
        mlf = MLMMForce(model, numbers, lig, W[inner], fires, es=True, device="cpu")
        X = mlf.ml_positions(P, box)
        Z = numbers[mlf.ml_idx]
        XL, ZL = X[:nl], Z[:nl]
        E_all, E_L, E_I = energy(X, Z), energy(XL, ZL), energy(X[nl:], Z[nl:])
        truth = E_all - E_L - E_I
        two_w = []
        for k in range(len(inner)):
            sl = slice(nl + 3 * k, nl + 3 * k + 3)
            Xw, Zw = X[sl], Z[sl]
            two_w.append(energy(np.vstack([XL, Xw]), np.r_[ZL, Zw]) - E_L - energy(Xw, Zw))
        two = float(np.sum(two_w))
        ia = mlf.ml_idx
        r = np.linalg.norm(X[:nl, None] / 10 - X[None, nl:] / 10, axis=-1)
        qL, sL, eL = prm[ia[:nl]].T
        qI, sI, eI = prm[ia[nl:]].T
        x6 = (0.5 * (sL[:, None] + sI[None]) / r) ** 6
        e_mm = float(ONE_4PI_EPS0 * (qL[:, None] * qI[None] / r).sum()
                     + (4 * np.sqrt(eL[:, None] * eI[None]) * (x6 * x6 - x6)).sum())
        rows.append(dict(frame=name, truth=truth, two=two, many=truth - two, mm=e_mm,
                         two_per_water=two_w, min_LI_A=float(10 * r.min())))
        print("%-18s truth %8.1f = two %8.1f + many %7.1f | MM %8.1f | two-MM %7.1f  min L-I %.2f A"
              % (name, truth, two, truth - two, e_mm, two - e_mm, 10 * r.min()), flush=True)

    out = dict(rows=rows, summary={})
    groups = ((("all given", lambda n: True),) if args.checkpoints else
              (("MM ensemble", lambda n: n.startswith("MM ")),
               ("D19 checkpoints", lambda n: "seg" in n or "full" in n),
               ("E8 checkpoint", lambda n: "fires_seed1" in n)))
    for label, sel in groups:
        R = [x for x in rows if sel(x["frame"])]
        s = {k: (float(np.mean([x[k] for x in R])), float(np.std([x[k] for x in R], ddof=1)))
             for k in ("truth", "two", "many", "mm", "min_LI_A")}
        s["two_minus_mm"] = float(np.mean([x["two"] - x["mm"] for x in R]))
        s["n"] = len(R)
        out["summary"][label] = s
        print(f"\n{label} (n = {len(R)}): mean +- sd, kJ/mol")
        for k, v in s.items():
            print("  %-13s %s" % (k, "%8.1f +- %.1f" % v if isinstance(v, tuple) else "%8.1f" % v))
    (HERE / args.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
