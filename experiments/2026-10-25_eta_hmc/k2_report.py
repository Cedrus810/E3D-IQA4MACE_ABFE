"""K2 atomic eta TI, global remainder and numerical residual (K2_DECISION_2026-11-07.md §3-§4).

    A_i^H = int <d_i>_eta deta,   R = int <q>_eta deta,   a^TI = int <dU>_eta deta = sum_i A_i^H + R
    eps_eta = a^BAR - (sum_i A_i^H + R)      (numerical; kept separate from the structural remainder R)
R is reported on its own and never apportioned to atoms. Rows are joint per cycle (atoms and q together,
rejected repeats included); block bootstrap resamples them jointly within each chain. Diagnostics only.
No model / OpenMM import.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from atomic_report import integrate, moving_block_sample
from k2_records import read_record

TARGET = ("system", "state", "leg", "T_K", "E_construction", "attribution_definition", "dtype", "backbone", "head",
          "realspace_backend", "realspace_modules", "total_charge", "omega_rule", "ligand_system_indices", "groups",
          "dummy_mask", "box_nm", "system_xml", "softcore_alpha", "n_water", "energy_units", "xyz_units",
          "system_definition", "context_parameters")


def split(v, na):
    v = np.asarray(v)
    return v[..., :na], v[..., na]


def report(files, expect_etas, expect_starts, discard=0.2, bootstrap=0, block_length=None, seed=0):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from pymbar import timeseries
    etas = list(map(float, expect_etas))
    starts = [str(x) for x in expect_starts]
    if not (len(etas) >= 2 and etas[0] == 0 and etas[-1] == 1 and np.all(np.diff(etas) > 0)):
        raise ValueError("expected eta grid must increase from 0 to 1")
    if not starts or len(set(starts)) != len(starts) or not 0 <= discard < 1 or bootstrap < 0:
        raise ValueError("invalid starts / discard / bootstrap")
    rows, target, details, na = {}, None, [], None
    for f in files:
        meta, z = read_record(f)
        if not meta["complete"]:
            raise ValueError(f"{f}: incomplete chain")
        ident = {k: meta.get(k) for k in TARGET}
        if target is None:
            target, na = ident, len(meta["ligand_system_indices"])
        elif target != ident:
            raise ValueError("incompatible K2 targets / mappings; analyze them separately")
        key = (float(meta["eta"]), str(meta["start_label"]))
        if key in rows or key[0] not in etas or key[1] not in starts:
            raise ValueError(f"duplicate or unexpected eta / start {key}")
        n0 = int(len(z["dU"]) * discard)
        x = np.c_[z["d_atom"][n0:], z["q"][n0:]]                      # joint rows: atoms then q
        if len(x) < 2:
            raise ValueError("too few post-discard cycles")
        if not np.allclose(x.sum(axis=1), z["dU"][n0:], rtol=1e-12, atol=1e-8):
            raise ValueError("record closure failed")
        series = np.c_[x, x.sum(axis=1)]
        gs = [float(timeseries.statistical_inefficiency(s)) if np.std(s) > 0 else 1.0 for s in series.T]
        length = min(len(x), int(block_length) if block_length else max(1, int(np.ceil(max(gs)))))
        rows[key] = x
        details.append({"file": str(f), "eta": key[0], "start": key[1], "cycles": len(x), "acceptance": float(z["acc"][n0:].mean()),
                        "block_length": length, "full_blocks": len(x) // length, "max_g": max(gs),
                        "mean_d_atom": x[:, :na].mean(axis=0).tolist(), "mean_q": float(x[:, na].mean()),
                        "mean_dU": float(x.sum(axis=1).mean())})
    if target is None:
        raise ValueError("no K2 chain files")
    missing = [(e, s) for e in etas for s in starts if (e, s) not in rows]
    base = {"kind": "K2 eta estimator diagnostics; not molecular convergence acceptance", "target": target,
            "expected_etas": etas, "expected_starts": starts, "complete": not missing, "missing": missing,
            "discard_fraction": discard, "chains": details,
            "sampling_weight": "every post-discard cycle, including rejected repeats"}
    if missing:
        return base
    kt = 0.0083144626 * float(target["T_K"])
    cols, bar, means = integrate(etas, {e: [rows[e, s] for s in starts] for e in etas}, kt)
    A, R = split(cols, na)
    a_ti = float(np.trapezoid(means.sum(axis=1), etas))
    base.update({"A_H_atom_TI": A.tolist(), "R_TI": float(R), "a_TI": a_ti, "a_BAR": bar,
                 "eps_eta_BAR_minus_atom_plus_R_TI": float(bar - A.sum() - R),
                 "atom_plus_R_minus_scalar_TI": float(A.sum() + R - a_ti),
                 "per_eta_mean_d_atom": means[:, :na].tolist(), "per_eta_mean_q": means[:, na].tolist()})
    base["per_start"] = []
    for s in starts:
        c, b, _ = integrate(etas, {e: [rows[e, s]] for e in etas}, kt)
        a_, r_ = split(c, na)
        base["per_start"].append({"start": s, "A_H_atom_TI": a_.tolist(), "R_TI": float(r_),
                                  "a_TI": float(c.sum()), "a_BAR": b, "eps_eta": float(b - c.sum())})
    if bootstrap:
        rng = np.random.default_rng(seed)
        lengths = {(r["eta"], r["start"]): r["block_length"] for r in details}
        cc, bb = [], []
        for _ in range(bootstrap):
            res = {e: [moving_block_sample(rows[e, s], lengths[e, s], rng) for s in starts] for e in etas}
            c, b, _ = integrate(etas, res, kt)
            cc.append(c)
            bb.append(b)
        cc, bb = np.asarray(cc), np.asarray(bb)
        q = [2.5, 50, 97.5]
        base["block_bootstrap"] = {"replicates": bootstrap, "seed": seed,
                                   "status": "empirical resampling diagnostics; not calibrated confidence intervals",
                                   "A_H_atom_quantiles": np.percentile(cc[:, :na], q, axis=0).tolist(),
                                   "R_quantiles": np.percentile(cc[:, na], q).tolist(),
                                   "a_TI_quantiles": np.percentile(cc.sum(axis=1), q).tolist(),
                                   "a_BAR_quantiles": np.percentile(bb, q).tolist(),
                                   "eps_eta_quantiles": np.percentile(bb - cc.sum(axis=1), q).tolist()}
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
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new output path")
    r = report(a.files, a.expect_etas.split(","), a.expect_starts.split(","), a.discard, a.bootstrap,
               a.block_length, a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    print(f"complete={r['complete']}; {out}")


if __name__ == "__main__":
    main()
