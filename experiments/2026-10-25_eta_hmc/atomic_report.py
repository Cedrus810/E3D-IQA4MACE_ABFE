"""Atomic eta TI and its residual to scalar BAR, from schema-2 chain files.

No model / OpenMM import. Incompatible targets and missing atomic records are
rejected. Records are sampled by cycle, including rejections. Block bootstrap
resamples joint atom rows WITHIN each chain, preserving atom covariances; its
quantiles are diagnostics, not calibrated scientific confidence intervals.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from eta_records import read_record

TARGET = ("system", "state", "leg", "T_K", "E_construction", "dtype", "backbone", "head",
          "ligand_system_indices", "groups", "dummy_mask", "box_nm", "system_xml", "softcore_alpha",
          "n_water", "energy_units", "xyz_units", "system_definition", "context_parameters")


def moving_block_sample(x, length, rng):
    n = len(x)
    starts = rng.integers(0, n - length + 1, size=int(np.ceil(n / length)))
    return np.concatenate([x[s:s + length] for s in starts], axis=0)[:n]


def integrate(etas, chains, kt):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from pymbar import other_estimators
    means = np.stack([np.concatenate(chains[e]).mean(axis=0) for e in etas])
    atom = np.trapezoid(means, etas, axis=0)
    bar = 0.0
    for lo, hi in zip(etas[:-1], etas[1:]):
        x = np.concatenate(chains[lo]).sum(axis=1)
        y = np.concatenate(chains[hi]).sum(axis=1)
        result = other_estimators.bar((hi - lo) * x / kt, -(hi - lo) * y / kt)
        bar += float(result["Delta_f"]) * kt
    return atom, bar, means


def report(files, expect_etas, expect_starts, discard=0.2, bootstrap=0, block_length=None, seed=0):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from pymbar import timeseries
    etas = list(map(float, expect_etas))
    starts = list(map(int, expect_starts))
    if not (len(etas) >= 2 and etas[0] == 0 and etas[-1] == 1 and np.all(np.diff(etas) > 0)):
        raise ValueError("expected eta grid must increase from 0 to 1")
    if not starts or len(set(starts)) != len(starts) or not 0 <= discard < 1 or bootstrap < 0:
        raise ValueError("invalid starts / discard / bootstrap")
    if block_length is not None and block_length < 1:
        raise ValueError("block length must be positive")
    rows, target, details = {}, None, []
    for f in files:
        meta, z = read_record(f)
        if not meta["complete"]:
            raise ValueError(f"{f}: incomplete chain; analyze committed progress separately")
        ident = {k: meta[k] for k in TARGET}
        if target is None:
            target = ident
        elif target != ident:
            raise ValueError("incompatible targets / atom mappings; analyze them separately")
        key = (float(meta["eta"]), int(meta["start_rep"]))
        if key in rows or key[0] not in etas or key[1] not in starts:
            raise ValueError(f"duplicate or unexpected eta / start {key}")
        n0 = int(len(z["dU"]) * discard)
        x = z["dU_atom"][n0:]
        if len(x) < 2:
            raise ValueError("too few post-discard cycles")
        series = np.c_[x, x.sum(axis=1)]
        gs = [float(timeseries.statistical_inefficiency(s)) if np.std(s) > 0 else 1.0 for s in series.T]
        length = min(len(x), int(block_length) if block_length else max(1, int(np.ceil(max(gs)))))
        rows[key] = x
        details.append({"file": str(f), "eta": key[0], "start": key[1], "cycles": len(x),
                        "block_length": length, "full_blocks": len(x) // length,
                        "max_g_atom_or_total": max(gs), "mean_dU_atom": x.mean(axis=0).tolist(),
                        "mean_dU": float(x.sum(axis=1).mean()),
                        "L": meta["L"], "dt_fs": meta["dt_fs"]})
    if target is None:
        raise ValueError("no schema-2 chain files")
    missing = [(e, s) for e in etas for s in starts if (e, s) not in rows]
    base = {"kind": "atomic eta estimator diagnostics; not molecular convergence acceptance",
            "target": target, "expected_etas": etas, "expected_starts": starts,
            "complete": not missing, "missing": missing, "discard_fraction": discard,
            "chains": details, "sampling_weight": "every post-discard cycle, including rejected repeats"}
    if missing:
        # Do not manufacture full-range atom corrections for an incomplete grid.
        return base
    kt = 0.0083144626 * float(target["T_K"])
    pools = {e: [rows[e, s] for s in starts] for e in etas}
    atom, bar, means = integrate(etas, pools, kt)
    scalar_ti = float(np.trapezoid(means.sum(axis=1), etas))
    base.update({"A_atom_TI": atom.tolist(), "a_TI": scalar_ti, "a_BAR": bar,
                 "r_eta_BAR_minus_atomic_TI": float(bar - atom.sum()),
                 "atomic_sum_minus_scalar_TI": float(atom.sum() - scalar_ti),
                 "per_eta_mean_atom": means.tolist(),
                 "BAR_point_estimator": "all post-discard cycles; correlation handled only in optional block resampling"})
    base["per_start"] = []
    for s in starts:
        a, b, _ = integrate(etas, {e: [rows[e, s]] for e in etas}, kt)
        base["per_start"].append({"start": s, "A_atom_TI": a.tolist(), "a_TI": float(a.sum()),
                                  "a_BAR": b, "r_eta": float(b - a.sum())})
    if bootstrap:
        rng = np.random.default_rng(seed)
        lengths = {(r["eta"], r["start"]): r["block_length"] for r in details}
        aa, bb = [], []
        for _ in range(bootstrap):
            resampled = {e: [moving_block_sample(rows[e, s], lengths[e, s], rng) for s in starts] for e in etas}
            a, b, _ = integrate(etas, resampled, kt)
            aa.append(a)
            bb.append(b)
        aa, bb = np.asarray(aa), np.asarray(bb)
        base["block_bootstrap"] = {"replicates": bootstrap, "seed": seed,
                "status": "empirical resampling diagnostics; not calibrated confidence intervals",
                "chains_with_fewer_than_two_full_blocks": [r["file"] for r in details if r["full_blocks"] < 2],
                "A_atom_quantiles_2p5_50_97p5": np.percentile(aa, [2.5, 50, 97.5], axis=0).tolist(),
                "a_TI_quantiles_2p5_50_97p5": np.percentile(aa.sum(axis=1), [2.5, 50, 97.5]).tolist(),
                "a_BAR_quantiles_2p5_50_97p5": np.percentile(bb, [2.5, 50, 97.5]).tolist(),
                "r_eta_quantiles_2p5_50_97p5": np.percentile(bb - aa.sum(axis=1), [2.5, 50, 97.5]).tolist()}
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--expect-etas", required=True)
    p.add_argument("--expect-starts", required=True)
    p.add_argument("--discard", type=float, default=0.2)
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--block-length", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="new diagnostic JSON path; does not overwrite historical analyses")
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new diagnostic output path")
    r = report(a.files, a.expect_etas.split(","), a.expect_starts.split(","),
               a.discard, a.bootstrap, a.block_length, a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    print(f"complete={r['complete']}; {out}")


if __name__ == "__main__":
    main()
