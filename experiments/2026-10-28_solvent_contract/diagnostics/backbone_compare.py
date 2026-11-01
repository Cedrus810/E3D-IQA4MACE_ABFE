"""Same-frame T = C_bb - X on the _propdiag proposals (diagnostic; no MD, no training).

C_bb: decomp.clusters.coupling_labels on the same carved Omega cluster (checked against the record), float64.
X: explicit Coulomb summed over ligand atoms, as recorded (prop_C_ML_coul; head-independent, head_compare.py).
The learned channels fit T, so  learned - T  is the head fitting residual on these frames.
Also: the fixed-proposal acceptance if the coupling were C_bb itself (dU_bb = C_bb - C_MM, MM terms and dH
from the record) -- an energy diagnostic, not a sampling result.

Run (project root, after head_compare.py): python -B experiments/2026-10-28_solvent_contract/diagnostics/backbone_compare.py
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
CHAIN = ETA / "runs/CyclophilinD_ligand-8/chain_S0_eta0.500_start1_propdiag.npz"
HEADS = HERE / "head_compare_propdiag.npz"
OUT = HERE / "backbone_compare_propdiag"
Q = [5, 25, 50, 75, 95]


def c_bb(m, z, leg):
    import torch
    from decomp.clusters import coupling_labels
    from model_eval import BB
    if str(pathlib.Path(BB).resolve()) != m["backbone"]:
        raise ValueError("backbone differs from the chain")
    bb = torch.load(BB, map_location="cpu", weights_only=False).to(torch.float64).to("cuda").eval()
    ps, ns, fs = [], [], []
    for t, x in enumerate(z["prop_xyz"]):
        p, n, f, s = leg.carve(x, m["n_water"])
        if not np.array_equal(s, z["prop_cluster_system_indices"][t]):
            raise ValueError(f"proposal {t + 1}: Omega differs from the chain record")
        ps.append(p)
        ns.append(n)
        fs.append(f)
    return np.asarray(coupling_labels(bb, ps, ns, fs, device="cuda", batch=1), np.float64) * eh.EV


def stats(d):
    return {"mean": float(d.mean()), "quantiles_5_25_50_75_95": np.percentile(d, Q).tolist(),
            "frac_positive": float((d > 0).mean())}


def main():
    m, z = read_record(CHAIN)
    leg = local_E.Leg(m["system"])
    cbb = c_bb(m, z, leg)
    h = np.load(HEADS)
    X = z["prop_C_ML_coul"].sum(1)
    if np.abs(h["v2_C_ML_coul"].sum(1) - X).max() > 1e-6:
        raise ValueError("explicit Coulomb differs between record and head_compare")
    T = cbb - X
    learned = {k: h[f"{k}_C_ML_learned"].sum(1) for k in ("old", "v2")}
    mm = z["prop_C_MM"].sum((1, 2))
    np.savez(OUT.with_suffix(".npz"), C_bb=cbb, X=X, T=T, C_MM=mm, **{f"learned_{k}": v for k, v in learned.items()})

    acc = z["acc"]
    s = int(np.flatnonzero(acc).max())
    prev = np.array([max([i for i in np.flatnonzero(acc) if i < t], default=-1) for t in range(len(acc))])
    rej = np.arange(s + 1, len(acc))
    out = {"chain": str(CHAIN), "stalled_proposal": s + 1, "units": "kJ/mol",
           "note": "same-frame energy diagnostic; C_bb is a scalar per frame (no atom / group split)",
           "level_at_stall": {"C_bb": float(cbb[s]), "X": float(X[s]), "T": float(T[s]),
                              **{f"learned_{k}": float(v[s]) for k, v in learned.items()}},
           "step5to6": {"dC_bb": float(cbb[s] - cbb[s - 1]), "dX": float(X[s] - X[s - 1]), "dT": float(T[s] - T[s - 1]),
                        **{f"dlearned_{k}": float(v[s] - v[s - 1]) for k, v in learned.items()},
                        **{f"dresidual_{k}": float((v - T)[s] - (v - T)[s - 1]) for k, v in learned.items()}},
           "rejected_vs_stall": {"dT": stats(T[rej] - T[s]), "dC_bb": stats(cbb[rej] - cbb[s]),
                                 **{f"dlearned_{k}": stats(v[rej] - v[s]) for k, v in learned.items()},
                                 **{f"dresidual_{k}": stats((v - T)[rej] - (v - T)[s]) for k, v in learned.items()}},
           "residual_learned_minus_T_all60": {k: {"mean": float((v - T).mean()), "sd": float((v - T).std(ddof=1)),
                                                  "at_stall": float((v - T)[s])} for k, v in learned.items()}}
    dU_bb = cbb - mm
    t = np.arange(1, len(acc))
    dE = dU_bb[t] - dU_bb[prev[t]]
    p = np.minimum(1.0, np.exp(-(z["dH"][t] + m["eta"] * dE) / eh.KT))
    out["fixed_proposals_with_C_bb"] = {"dE_proposals_2_to_6": dE[:s].tolist(), "p_acc_proposals_2_to_6": p[:s].tolist(),
                                        "rejected_dE_mean": float(dE[s:].mean()),
                                        "rejected_p_acc": {"median": float(np.median(p[s:])), "max": float(p[s:].max()),
                                                           "sum": float(p[s:].sum())}}
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
