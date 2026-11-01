"""Same-frame residual check of the nvar heads (a: 60k, b: 104k) on the _propdiag proposals (no MD, no training).

Learned channels re-evaluated with head_compare.evaluate (same coordinates, Omega, mask, backbone, float64).
T = C_bb - X, X and C_MM reused from backbone_compare_propdiag.npz; old / v2 reused from head_compare_propdiag.npz.
    dU = (C_bb - C_MM) + r,   r = learned - T
Acceptance of the FIXED proposals along the recorded history: energy diagnostic, not a sampling result.

Run (project root, after head_compare.py and backbone_compare.py):
    python -B experiments/2026-10-28_solvent_contract/diagnostics/nvar_compare.py
"""

import json
import sys

import numpy as np

import head_compare as hc
from head_compare import eh, local_E, read_record

CK = "/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_"
NVAR = {"nvar_a": CK + "s60000_nvar_N.pt", "nvar_b": CK + "s104000_nvar_N.pt"}
OUT = hc.HERE / "nvar_compare_propdiag"


def stats(d):
    return {"mean": float(d.mean()), "quantiles_5_25_50_75_95": np.percentile(d, hc.Q).tolist(),
            "frac_positive": float((d > 0).mean())}


def main():
    m, z = read_record(hc.CHAIN)
    leg = local_E.Leg(m["system"])
    res = hc.evaluate(m, z, leg, NVAR)
    np.savez(OUT.with_suffix(".npz"), **{f"{h}_{k}": v for h, d in res.items() for k, v in d.items()})
    b = np.load(hc.HERE / "backbone_compare_propdiag.npz")
    h = np.load(hc.HERE / "head_compare_propdiag.npz")
    T, X, cbb, mm = b["T"], b["X"], b["C_bb"], b["C_MM"]
    for k, d in res.items():
        if np.abs(d["C_ML_coul"].sum(1) - X).max() > 1e-6:
            raise ValueError(f"{k}: explicit Coulomb differs from the reused X")
    learned = {"old": h["old_C_ML_learned"].sum(1), "v2": h["v2_C_ML_learned"].sum(1),
               **{k: d["C_ML_learned"].sum(1) for k, d in res.items()}}

    acc = z["acc"]
    s = int(np.flatnonzero(acc).max())
    prev = np.array([max([i for i in np.flatnonzero(acc) if i < t], default=-1) for t in range(len(acc))])
    rej = np.arange(s + 1, len(acc))
    t = np.arange(1, len(acc))
    pos = {a: n for n, a in enumerate(m["ligand_system_indices"])}
    gidx = [[pos[a] for a in g] for g in m["groups"]]
    out = {"heads": {**{k: v for k, v in hc.HEADS.items()}, **NVAR}, "stalled_proposal": s + 1, "units": "kJ/mol",
           "backbone_minus_MM_rejected_mean": float(((cbb - mm)[rej] - (cbb - mm)[s]).mean()),
           "note": "same-frame energy diagnostic on one stalled chain; not equilibrium-tail or mixing evidence"}
    for k, L in learned.items():
        r = L - T
        dU = cbb - mm + r
        dE = dU[t] - dU[prev[t]]
        p = np.minimum(1.0, np.exp(-(z["dH"][t] + m["eta"] * dE) / eh.KT))
        row = {"step5to6_dlearned": float(L[s] - L[s - 1]), "step5to6_dresidual": float(r[s] - r[s - 1]),
               "rejected_dlearned": stats(L[rej] - L[s]), "rejected_dresidual": stats(r[rej] - r[s]),
               "residual_all60": {"mean": float(r.mean()), "sd": float(r.std(ddof=1)), "at_stall": float(r[s])},
               "rejected_dE_mean": float(dE[s:].mean()),
               "rejected_p_acc": {"median": float(np.median(p[s:])), "max": float(p[s:].max()), "sum": float(p[s:].sum())},
               "p_acc_proposals_2_to_6": p[:s].tolist()}
        if k in res:
            Lat = res[k]["C_ML_learned"]
            row["step5to6_dlearned_groups"] = [float(Lat[s, g].sum() - Lat[s - 1, g].sum()) for g in gidx]
            row["rejected_dlearned_group_means"] = [float((Lat[rej][:, g].sum(1) - Lat[s, g].sum()).mean()) for g in gidx]
        out[k] = row
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
