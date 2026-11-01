"""Steps 1-2 of the D19 review (v3 S13.57 end, user 2026-10-21).

Step 1: the many-body split with the PRODUCTION inner identity (picked on the leg's start frame, as
every run used here did), plus how many of those waters are among the 20 nearest on each frame.
Step 2: on the same configurations, the full old-path model and D19 at q = 1 (lambda = 1, wall on),
both built by build_mlmm_system from the same start frame and evaluated as OpenMM Contexts (CUDA,
double; ML float64 on the CPU), so PME, exceptions, wall and FIRES are handled by the production
code in both. Everything but the shell's own energy is common:
    dU = U_D19 - U_old = -E_POLAR(I) + E_TIP3P(I-I, PME)      (+ terms identical in both, which cancel)
Asks: does dU (relative to MM configurations) offset the many-body advantage of the D19 frames?
MM frames (frames.npz, no FIRES) have no production identity; they use the nearest 20 per frame
and are built per frame, as the structural reference.
"""
import json
import pathlib
import sys

import numpy as np
import openmm
import torch
from openmm import unit

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from coupling_check import BB, CKPT, EV, LEG, ROOT, checkpoint_frames  # noqa: E402

sys.path.insert(0, str(ROOT))
from decomp import mlmm  # noqa: E402
from decomp.fires import Fires, heavy_selection, water_indices  # noqa: E402
from decomp.openmm_bridge import AlchemicalForce  # noqa: E402
from run_abfe import load_model  # noqa: E402

PLATFORM = openmm.Platform.getPlatformByName("CUDA")


class Model:
    """One path (old or D19) built from one identity frame; energies at q = 1."""

    def __init__(self, model, id_positions, shell_mm):
        self.system, self.mlf, self.info = mlmm.build_mlmm_system(
            str(LEG), model, n_inner=20, device="cpu", positions=id_positions, shell_mm=shell_mm)
        self.ctx = openmm.Context(self.system, openmm.VerletIntegrator(0.001), PLATFORM,
                                  {"Precision": "double"})
        box = self.info["box"]
        self.ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[k] * box[k] / 10)) for k in range(3)])
        mlmm.set_path(self.ctx, self.mlf, 1.0, self.info["lig"], shell_mm=shell_mm)

    def energies(self, P):
        self.ctx.setPositions(P * unit.angstrom)
        tot = self.ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        nb = mlmm.group_energy(self.ctx, mlmm.G_NB)
        return tot, nb


def main():
    torch.set_num_threads(8)
    d = np.load(LEG, allow_pickle=True)
    numbers, box = np.asarray(d["numbers"]), np.asarray(d["box"], float)
    start = np.asarray(d["positions"], float)
    lig = np.where(d["resids"] == d["resids"][0])[0]
    nl = len(lig)
    W = water_indices(d["resnames"], d["resids"])
    centre, wsel = heavy_selection(lig, numbers, W)
    model = load_model(str(CKPT), BB, device="cpu").double()
    bb = model.backbone
    cache = {}

    def energy(X, Z):
        key = tuple(Z)
        if key not in cache:
            cache[key] = AlchemicalForce(model, Z, np.zeros(len(Z), int), device="cpu")
        with torch.no_grad():
            return float(bb(cache[key]._graph(X), training=False, compute_force=False)["energy"].sum()) * EV

    def split(mlf, P):
        X = mlf.ml_positions(P, box)
        Z = numbers[mlf.ml_idx]
        E_all, E_L, E_I = energy(X, Z), energy(X[:nl], Z[:nl]), energy(X[nl:], Z[nl:])
        two = sum(energy(np.vstack([X[:nl], X[nl + 3 * k: nl + 3 * k + 3]]),
                         np.r_[Z[:nl], Z[nl + 3 * k: nl + 3 * k + 3]]) - E_L
                  - energy(X[nl + 3 * k: nl + 3 * k + 3], Z[nl + 3 * k: nl + 3 * k + 3])
                  for k in range(20))
        truth = E_all - E_L - E_I
        return dict(truth=truth, two=two, many=truth - two, E_I=E_I)

    def nearest(P):
        return Fires(centre, wsel, n_inner=20, device="cpu").select_inner(P, box)

    rows = []

    def evaluate(name, P, old, new, identity):
        e_old, nb_old = old.energies(P)
        e_new, nb_new = new.energies(P)
        s = split(new.mlf, P)
        shared = len(np.intersect1d(identity, nearest(P)))
        row = dict(frame=name, dU=e_new - e_old, minus_E_I=-s["E_I"], dNB=nb_new - nb_old,
                   check=(e_new - e_old) - (-s["E_I"] + nb_new - nb_old), shared_with_nearest=shared,
                   **{k: s[k] for k in ("truth", "two", "many")})
        rows.append(row)
        print("%-18s shared %2d | truth %7.1f two %7.1f many %6.1f | dU %12.1f = -E_I %12.1f + dNB %7.1f (check %.1e)"
              % (name, shared, s["truth"], s["two"], s["many"], row["dU"], row["minus_E_I"], row["dNB"],
                 row["check"]), flush=True)

    # production identity: the leg's start frame, as all four runs
    prod = nearest(start)
    old, new = Model(model, start, False), Model(model, start, True)
    assert np.array_equal(np.sort(old.info["inner"]), np.sort(prod)) and np.array_equal(np.sort(new.info["inner"]), np.sort(prod))
    for ck in ("experiments/2026-10-19_d19_acetamide/segA/state.npz",
               "experiments/2026-10-19_d19_acetamide/full/state.npz",
               "experiments/2026-10-17_set_run/acetamide/state.npz",
               "experiments/2026-10-16_mm_path_control/fires_seed1/state.npz"):
        for name, P in checkpoint_frames(ROOT / ck):
            evaluate(name, P, old, new, prod)
    del old, new

    # MM frames: nearest 20 per frame (no production identity without FIRES)
    mm = np.load(HERE / "frames.npz")["positions"]
    for k, P in enumerate(mm):
        old_k, new_k = Model(model, P, False), Model(model, P, True)
        evaluate(f"MM {k}", P, old_k, new_k, nearest(P))
        del old_k, new_k

    ref = np.mean([r["dU"] for r in rows if r["frame"].startswith("MM ")])
    ref_many = np.mean([r["many"] for r in rows if r["frame"].startswith("MM ")])
    print("\nrelative to the MM-frame mean: dU_rel = dU - <dU>_MM ; many_rel = many - <many>_MM (kJ/mol)")
    groups = (("D19 (segA, full)", lambda n: n.startswith(("segA", "full"))),
              ("old path (acetamide)", lambda n: n.startswith("acetamide")),
              ("E8 (fires_seed1)", lambda n: n.startswith("fires_seed1")),
              ("MM frames", lambda n: n.startswith("MM ")))
    summary = {}
    for label, sel in groups:
        R = [r for r in rows if sel(r["frame"])]
        s = dict(n=len(R),
                 dU_rel=[round(r["dU"] - ref, 1) for r in R] if len(R) <= 4 else
                 (float(np.mean([r["dU"] for r in R]) - ref), float(np.std([r["dU"] for r in R], ddof=1))),
                 many_rel=[round(r["many"] - ref_many, 1) for r in R] if len(R) <= 4 else
                 (float(np.mean([r["many"] for r in R]) - ref_many), float(np.std([r["many"] for r in R], ddof=1))),
                 truth=float(np.mean([r["truth"] for r in R])), many=float(np.mean([r["many"] for r in R])),
                 shared_with_nearest=[r["shared_with_nearest"] for r in R] if len(R) <= 4 else
                 float(np.mean([r["shared_with_nearest"] for r in R])))
        summary[label] = s
        print(f"  {label:22s} {s}")
    (HERE / "cross_eval.json").write_text(json.dumps(dict(rows=rows, summary=summary), indent=2) + "\n")


if __name__ == "__main__":
    main()
