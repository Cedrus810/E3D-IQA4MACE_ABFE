"""Same-frame old head vs v2 on the _propdiag chain (diagnostic; no MD, no training).

Frames: the 60 saved proposals of chain_S0_eta0.500_start1_propdiag.npz (proposal 6 = the stalled state).
Coordinates, Omega (re-carved from the same full box, checked against the record), dummy mask, backbone and
float64 are those of the chain; only the head file changes. MM terms and dH come from the chain record.
Acceptance probabilities are for these FIXED proposals along the recorded old-head history, not a v2 chain.

Run (project root): python -B experiments/2026-10-28_solvent_contract/diagnostics/head_compare.py
"""

import json
import pathlib
import sys

import numpy as np

ETA = pathlib.Path("/home/ruigengji/E3D/experiments/2026-10-25_eta_hmc")
sys.path.insert(0, str(ETA))
import eta_hmc as eh  # noqa: E402
import local_E  # noqa: E402
from eta_records import read_record  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
CK = "/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_"
HEADS = {"old": CK + "h256_s60000_N.pt", "v2": CK + "resp_v2_h256_s60000_N.pt"}
CHAIN = ETA / "runs/CyclophilinD_ligand-8/chain_S0_eta0.500_start1_propdiag.npz"
OUT = HERE / "head_compare_propdiag"
Q = [5, 25, 50, 75, 95]


def evaluate(m, z, leg, heads=HEADS):
    import torch
    res = {}
    for name, ck in heads.items():
        me = eh.ModelEnergy(leg, m["state"], m["dtype"], ckpt=ck, backbone=m["backbone"])
        rows = {k: [] for k in ("C_ML_learned", "C_ML_coul", "dU_atom")}
        for t, x in enumerate(z["prop_xyz"]):
            me(x)
            lc = me.last_components
            if not np.array_equal(lc["cluster_system_indices"], z["prop_cluster_system_indices"][t]):
                raise ValueError(f"proposal {t + 1}: Omega differs from the chain record")
            if not np.allclose(lc["C_MM"], z["prop_C_MM"][t], rtol=0, atol=1e-9):
                raise ValueError(f"proposal {t + 1}: C_MM differs from the chain record")
            for k in rows:
                rows[k].append(lc[k])
        res[name] = {k: np.array(v) for k, v in rows.items()}
        print(f"{name}: {me.calls} calls, {me.seconds / me.calls * 1e3:.0f} ms/call", flush=True)
        del me
        torch.cuda.empty_cache()
    return res


def main():
    m, z = read_record(CHAIN)
    if pathlib.Path(m["head"]).name != pathlib.Path(HEADS["old"]).name:
        raise ValueError("chain was not run with the old head")
    if not np.array_equal(z["xyz_cycles"], np.arange(1, m["cycles"] + 1)):
        raise ValueError("need full-box proposals at every cycle")
    leg = local_E.Leg(m["system"])
    res = evaluate(m, z, leg)
    np.savez(OUT.with_suffix(".npz"), **{f"{h}_{k}": v for h, d in res.items() for k, v in d.items()})

    acc = z["acc"]
    s = int(np.flatnonzero(acc).max())              # 0-based index of the stalled state (proposal 6)
    prev = np.array([max([i for i in np.flatnonzero(acc) if i < t], default=-1) for t in range(len(acc))])
    lig = m["ligand_system_indices"]
    pos = {a: n for n, a in enumerate(lig)}
    gidx = [[pos[a] for a in g] for g in m["groups"]]
    rej = np.arange(s + 1, len(acc))
    out = {"chain": str(CHAIN), "heads": HEADS, "stalled_proposal": s + 1, "eta": m["eta"], "kT": eh.KT,
           "note": "same-frame energy diagnostic; acceptance of fixed proposals is not v2 mixing evidence"}
    old_rec = z["prop_C_ML_learned"]
    out["old_recompute_max_abs_learned_vs_record"] = float(np.abs(res["old"]["C_ML_learned"] - old_rec).max())
    out["coul_max_abs_v2_minus_old"] = float(np.abs(res["v2"]["C_ML_coul"] - res["old"]["C_ML_coul"]).max())
    for h, d in res.items():
        L = d["C_ML_learned"]
        dU = d["dU_atom"].sum(1)
        r = {"step5to6_learned": float(L[s].sum() - L[s - 1].sum()),
             "step5to6_learned_groups": [float(L[s, g].sum() - L[s - 1, g].sum()) for g in gidx],
             "step5to6_ddU": float(dU[s] - dU[s - 1])}
        dL = L[rej].sum(1) - L[s].sum()
        dLg = [L[rej][:, g].sum(1) - L[s, g].sum() for g in gidx]
        r["rejected_dLearned"] = {"mean": float(dL.mean()), "quantiles_5_25_50_75_95": np.percentile(dL, Q).tolist(),
                                  "frac_positive": float((dL > 0).mean()),
                                  "group_means": [float(x.mean()) for x in dLg]}
        # fixed proposals t >= 2 (proposal 1's start state was not saved as a full box)
        t = np.arange(1, len(acc))
        dE = dU[t] - dU[prev[t]]
        p = np.minimum(1.0, np.exp(-(z["dH"][t] + m["eta"] * dE) / eh.KT))
        r["dE_proposals_2_to_6"] = dE[:s].tolist()
        r["p_acc_proposals_2_to_6"] = p[:s].tolist()
        pr = p[s:]
        r["rejected_dE_mean"] = float(dE[s:].mean())
        r["rejected_p_acc"] = {"median": float(np.median(pr)), "max": float(pr.max()), "sum": float(pr.sum()),
                               "quantiles_5_25_50_75_95": np.percentile(pr, Q).tolist()}
        if h == "old":
            r["max_abs_dE_vs_record"] = float(np.abs(dE - z["dE"][t]).max())
        out[h] = r
    (OUT.with_suffix(".json")).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
