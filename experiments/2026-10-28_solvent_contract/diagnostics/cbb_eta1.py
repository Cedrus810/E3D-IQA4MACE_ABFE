"""Split the 40 fs proposal change of dU on the nvar b eta=1 chains into backbone - MM and head residual
(diagnostic; no MD, no training).

    dE = dU(proposal) - dU(current) = [dC_bb - dC_MM] + dr,   r = learned - T,  T = C_bb - X

C_bb: decomp.clusters.coupling_labels on the same carved Omega cluster (checked against the record), float64.
learned, X, C_MM, dE: from the chain record. Full boxes exist every 10 cycles (states and proposals); a pair is
(proposal at snapshot cycle t, current state = state at snapshot t-10), used only if no acceptance in between
(checked on the ligand coordinates). Per-chain C_bb files are reused when their frame cycles match.

Run (project root): python -B experiments/2026-10-28_solvent_contract/diagnostics/cbb_eta1.py
"""

import json
import pathlib
import sys
import time

import numpy as np

ETA = pathlib.Path("/home/ruigengji/E3D/experiments/2026-10-25_eta_hmc")
sys.path.insert(0, str(ETA))
import eta_hmc as eh  # noqa: E402
import local_E  # noqa: E402
from eta_records import read_record  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUTDIR = HERE / "cbb_nvarb_ms"
CHAINS = {s: ETA / f"runs/CyclophilinD_ligand-8/chain_S0_eta1.000_start{s}_nvarb_ms.npz" for s in (1, 2, 3, 4)}


def c_bb_frames(bb, leg, m, frames, cl_idx):
    from decomp.clusters import coupling_labels
    out = []
    for x, s_ref in zip(frames, cl_idx):
        p, n, f, s = leg.carve(x, m["n_water"])
        if s_ref is not None and not np.array_equal(s, s_ref):
            raise ValueError("Omega differs from the chain record")
        out.append(float(coupling_labels(bb, [p], [n], [f], device="cuda", batch=1)[0]) * eh.EV)
    return np.array(out)


def compute(bb, leg, start, path):
    m, z = read_record(path)
    out = OUTDIR / f"cbb_eta1.000_start{start}.npz"
    xc = z["xyz_cycles"]
    if out.exists():
        old = np.load(out)
        if np.array_equal(old["xyz_cycles"], xc):
            print(f"start{start}: reuse {out.name}", flush=True)
            return
    t0 = time.perf_counter()
    # chain state at cycle c is the state after cycle c (index c-1); its Omega is cluster_system_indices[c-1]
    state = c_bb_frames(bb, leg, m, z["xyz"], z["cluster_system_indices"][xc - 1])
    prop = c_bb_frames(bb, leg, m, z["prop_xyz"], z["prop_cluster_system_indices"][xc - 1])
    np.savez(out, xyz_cycles=xc, C_bb_state=state, C_bb_prop=prop)
    print(f"start{start}: {2 * len(xc)} frames in {time.perf_counter() - t0:.0f} s -> {out.name}", flush=True)


def analyse():
    rows = []
    for s, path in CHAINS.items():
        m, z = read_record(path)
        c = np.load(OUTDIR / f"cbb_eta1.000_start{s}.npz")
        xc = list(z["xyz_cycles"])
        lig = np.asarray(m["ligand_system_indices"])
        for k in range(1, len(xc)):
            t, tp = xc[k], xc[k - 1]                     # proposal cycle t; previous snapshot cycle
            if t - tp != 10 or z["acc"][tp:t - 1].any():
                continue                                  # state changed between snapshot and proposal
            if not np.array_equal(z["lig_xyz"][t - 2], z["xyz"][k - 1][lig]):
                raise ValueError("snapshot is not the current state")
            i, j = t - 1, t - 2                           # proposal index, current-state index
            d = {"start": s, "cycle": int(t),
                 "dC_bb": c["C_bb_prop"][k] - c["C_bb_state"][k - 1],
                 "dX": z["prop_C_ML_coul"][i].sum() - z["C_ML_coul"][j].sum(),
                 "dlearned": z["prop_C_ML_learned"][i].sum() - z["C_ML_learned"][j].sum(),
                 "dMM": z["prop_C_MM"][i].sum() - z["C_MM"][j].sum(),
                 "dE": float(z["dE"][i])}
            rows.append(d)
    R = {k: np.array([r[k] for r in rows], float) for k in ("dC_bb", "dX", "dlearned", "dMM", "dE")}
    bm = R["dC_bb"] - R["dMM"]                            # backbone - MM part
    dr = R["dlearned"] - (R["dC_bb"] - R["dX"])          # head residual part
    if not np.allclose(bm + dr, R["dE"], atol=1e-6):
        raise ValueError("decomposition does not reproduce dE")
    cov = np.cov(np.stack([bm, dr]))
    res = {"n_pairs": len(rows), "pairs_per_start": {s: sum(r["start"] == s for r in rows) for s in CHAINS},
           "units": "kJ/mol", "note": "pairs only where no acceptance since the previous snapshot (biased to stuck stretches)",
           "dE": {"mean": float(R["dE"].mean()), "sd": float(R["dE"].std(ddof=1))},
           "backbone_minus_MM": {"mean": float(bm.mean()), "sd": float(np.sqrt(cov[0, 0]))},
           "head_residual": {"mean": float(dr.mean()), "sd": float(np.sqrt(cov[1, 1]))},
           "corr_bbMM_residual": float(np.corrcoef(bm, dr)[0, 1]),
           "var_share": {"backbone_minus_MM": float(cov[0].sum() / cov.sum()), "head_residual": float(cov[1].sum() / cov.sum())},
           "sd_dC_bb": float(R["dC_bb"].std(ddof=1)), "sd_dMM": float(R["dMM"].std(ddof=1)),
           "corr_dC_bb_dMM": float(np.corrcoef(R["dC_bb"], R["dMM"])[0, 1])}
    for s in CHAINS:
        sel = np.array([r["start"] == s for r in rows])
        res[f"start{s}"] = {"n": int(sel.sum()), "dE_mean": float(R["dE"][sel].mean()),
                            "bbMM_mean": float(bm[sel].mean()), "bbMM_sd": float(bm[sel].std(ddof=1)),
                            "resid_mean": float(dr[sel].mean()), "resid_sd": float(dr[sel].std(ddof=1))}
    np.savez(OUTDIR / "pairs.npz", **R, backbone_minus_MM=bm, head_residual=dr,
             start=np.array([r["start"] for r in rows]), cycle=np.array([r["cycle"] for r in rows]))
    (OUTDIR / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


def main():
    import torch
    from model_eval import BB
    OUTDIR.mkdir(exist_ok=True)
    m, _ = read_record(CHAINS[1])
    if str(pathlib.Path(BB).resolve()) != m["backbone"]:
        raise ValueError("backbone differs from the chains")
    leg = local_E.Leg(m["system"])
    bb = torch.load(BB, map_location="cpu", weights_only=False).to(torch.float64).to("cuda").eval()
    for s, path in CHAINS.items():
        compute(bb, leg, s, path)
    analyse()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
