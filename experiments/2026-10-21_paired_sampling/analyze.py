"""Analyse the paired trajectories (paired_sampling.py), v3 S13.57-13.58 follow-up.

Per saved frame (every 0.5 ps, 20 per trajectory), with the trajectory's own (production) inner
identity: truth / two / many (POLAR-1-M float64, CPU) and dU = U_D19 - U_old at q = 1 from both
models built on the start frame (OpenMM CUDA double), as cross_eval.py. dU_rel and many_rel are
relative to the MM-frame means of cross_eval.json (GAFF2/TIP3P frames, nearest 20).
"""
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
CC = HERE.parent / "2026-10-20_coupling_check"
sys.path.insert(0, str(CC))
sys.path.insert(0, str(HERE))
from coupling_check import BB, CKPT, EV, LEG  # noqa: E402
from cross_eval import Model  # noqa: E402
from paired_sampling import starts  # noqa: E402

ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from decomp.openmm_bridge import AlchemicalForce  # noqa: E402
from run_abfe import load_model  # noqa: E402


def main():
    torch.set_num_threads(8)
    d = np.load(LEG, allow_pickle=True)
    numbers, box = np.asarray(d["numbers"]), np.asarray(d["box"], float)
    nl = int(np.sum(d["resids"] == d["resids"][0]))
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
        two = 0.0
        for k in range(20):
            w = slice(nl + 3 * k, nl + 3 * k + 3)
            two += energy(np.vstack([X[:nl], X[w]]), np.r_[Z[:nl], Z[w]]) - E_L - energy(X[w], Z[w])
        truth = E_all - E_L - E_I
        return truth, two, truth - two

    cx = json.load(open(CC / "cross_eval.json"))
    mm = [r for r in cx["rows"] if r["frame"].startswith("MM ")]
    ref_dU, ref_many = np.mean([r["dU"] for r in mm]), np.mean([r["many"] for r in mm])

    out = {}
    for sname, P0 in starts():
        old, new = Model(model, P0, False), Model(model, P0, True)
        for mname in ("old", "d19"):
            z = np.load(HERE / "traj" / f"{sname}_{mname}.npz")
            assert np.array_equal(np.sort(z["inner"]), np.sort(new.info["inner"])), "identity mismatch"
            rows = []
            for P in z["positions"].astype(np.float64):
                truth, two, many = split(new.mlf, P)
                dU = new.energies(P)[0] - old.energies(P)[0]
                rows.append(dict(truth=truth, two=two, many=many, dU_rel=dU - ref_dU, many_rel=many - ref_many))
            out[f"{sname}_{mname}"] = rows
            t = (np.arange(len(rows)) + 1) * float(z["every"]) * float(z["dt_fs"]) / 1000
            late = rows[len(rows) // 2:]
            print("%-10s many %s" % (f"{sname}/{mname}", " ".join("%6.0f" % r["many"] for r in rows)), flush=True)
            print("%-10s dU_rel %s" % ("", " ".join("%6.0f" % r["dU_rel"] for r in rows)), flush=True)
            print("%-10s last 5 ps: truth %7.1f  two %7.1f  many %7.1f  dU_rel %7.1f   (t = %.1f..%.1f ps)"
                  % ("", *(np.mean([r[k] for r in late]) for k in ("truth", "two", "many", "dU_rel")),
                     t[len(rows) // 2], t[-1]), flush=True)
        del old, new

    print("\npaired, last 5 ps means (D19 - old), kJ/mol")
    summ = {}
    for sname, _ in starts():
        a, b = out[f"{sname}_old"], out[f"{sname}_d19"]
        late = lambda R, k: float(np.mean([r[k] for r in R[len(R) // 2:]]))  # noqa: E731
        s = {k: (late(a, k), late(b, k), late(b, k) - late(a, k)) for k in ("truth", "two", "many", "dU_rel")}
        summ[sname] = s
        print("  %-5s " % sname + "  ".join("%s old %7.1f d19 %7.1f diff %+7.1f" % (k, *v) for k, v in s.items()))
    (HERE / "analyze.json").write_text(json.dumps(dict(ref=dict(dU=ref_dU, many=ref_many), traj=out,
                                                       summary=summ), indent=2) + "\n")


if __name__ == "__main__":
    main()
