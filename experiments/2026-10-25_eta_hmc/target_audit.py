"""Deep / shallow S0 panel: where does dU = C_ML - C_MM come from? (REVIEW_2026-11-01_path_overview.md §7 step 1)

Run: python target_audit.py --panel logs/deep_shallow_panel_20261031.npz --out logs/target_audit_<date>.json

On each saved panel frame (exact full-box coordinates, S0, local_E v1 Omega of 50 waters), with the panel's head and
backbone, float64:
    C_ML  head coupling (learned channels + explicit Coulomb X_head), re-queried and compared with the cache
    C_bb  backbone label E(L+Omega) - E(L) - E(Omega)   (decomp.clusters.coupling_labels, as the training labels)
    X     explicit Coulomb cross of the whole cluster   (head_retrain/make_data.cross_labels)
    C_MM  bare Coulomb + cutoff LJ over the same pairs  (cache)
    dU = (C_ML - C_bb) + (C_bb - C_MM): head fit error relative to the backbone + backbone-MM mismatch.
The backbone label is defined only as a total (no per-atom C_bb); per-group splits are given for C_ML, C_MM and dU.
Geometry: ligand heavy-atom RMSD to the deepest frame after superposition, Omega waters with O within 3.5 A of a
ligand N/O. Reported only; no thresholds. Same-frame diagnostic of the frozen target: not physical accuracy of the
backbone, not basin weights, not an a_0 estimate.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-25_head_retrain"))
import eta_hmc as eh  # noqa: E402
import local_E  # noqa: E402


def rmsd_fit(a, b):
    """RMSD of a onto b after optimal superposition (Kabsch), same units."""
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def main(a):
    import torch
    from decomp.clusters import coupling_labels
    from make_data import cross_labels

    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit(f"preserving {out}; choose another --out")
    z = np.load(a.panel, allow_pickle=False)
    m = json.loads(str(z["meta"]))
    c = m["contract"]
    leg = local_E.Leg(c["system"])
    me = eh.ModelEnergy(leg, c["state"], c["dtype"], ckpt=c["head"], backbone=c["backbone"])
    bb = torch.load(c["backbone"], map_location="cpu", weights_only=False).to(torch.float64).to("cuda").eval()
    groups = c["groups"]
    lig = np.array(c["ligand_system_indices"])
    nums = None
    ps, ns, fs, rows = [], [], [], []
    for i, src in enumerate(m["frames"]):
        p, n, f, s = leg.carve(np.asarray(z["xyz_nm"][i], np.float64), c["n_water"])
        if not np.array_equal(np.asarray(s), z["cluster_system_indices"][i]):
            raise SystemExit(f"frame {i}: Omega differs from the panel cache")
        dU = me(z["xyz_nm"][i])
        lc = me.last_components
        ps.append(p), ns.append(n), fs.append(f)
        nums = n
        rows.append({"tag": src["tag"], "rep": src["rep"], "walker": src["walker"],
                     "dU": float(dU), "dU_cache": float(z["expected_dU"][i]),
                     "C_ML": float(lc["C_ML"].sum()), "C_ML_learned": float(lc["C_ML_learned"].sum()),
                     "X_head": float(lc["C_ML_coul"].sum()),
                     "C_MM": float(lc["C_MM"].sum()), "C_MM_es": float(lc["C_MM"][:, 0].sum()),
                     "C_MM_lj": float(lc["C_MM"][:, 1].sum()),
                     "groups": [{"dU": float(lc["dU_atom"][g].sum()), "C_ML": float(lc["C_ML"][g].sum()),
                                 "C_MM": float(lc["C_MM"][g].sum())} for g in groups]})
    cbb = np.asarray(coupling_labels(bb, ps, ns, fs, device="cuda", batch=a.batch), np.float64) * eh.EV
    x = np.asarray(cross_labels(bb, ps, ns, fs, a.batch), np.float64) * eh.EV
    heavy = [k for k, e in enumerate(nums[fs[0] == 0]) if e > 1]
    polar = [k for k, e in enumerate(nums[fs[0] == 0]) if e in (7, 8)]
    deep = int(np.argmin([r["dU"] for r in rows]))
    for i, r in enumerate(rows):
        p, n, f = ps[i], ns[i], fs[i]
        L, W = p[f == 0], p[f == 1]
        wO = W[n[f == 1] == 8]
        r.update({"C_bb": float(cbb[i]), "X_label": float(x[i]),
                  "head_minus_bb": r["C_ML"] - float(cbb[i]), "bb_minus_MM": float(cbb[i]) - r["C_MM"],
                  "learned_minus_T": r["C_ML_learned"] - (float(cbb[i]) - float(x[i])),
                  "lig_heavy_rmsd_to_deep_A": rmsd_fit(L[heavy], ps[deep][fs[deep] == 0][heavy]),
                  "omega_O_within_3p5A_of_lig_NO": int((np.linalg.norm(wO[:, None] - L[polar][None], axis=2)
                                                        .min(1) < 3.5).sum())})
    res = {"panel": a.panel, "contract": {k: c[k] for k in ("system", "state", "head", "backbone", "E_construction",
                                                           "dtype", "n_water", "groups")},
           "units": "kJ/mol, A", "deep_frame": deep, "model_calls": me.calls, "frames": rows,
           "note": "same-frame decomposition of the frozen target; not physical accuracy, basin weights or a_0"}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"{c['system']} S0 panel, head {pathlib.Path(c['head']).name}; kJ/mol")
    print(f"{'frame':>16s} {'dU':>9s} {'|cache|':>8s} {'C_ML':>9s} {'C_bb':>9s} {'C_MM':>9s} {'ML-bb':>8s} {'bb-MM':>8s} "
          f"{'Xh-X':>8s} {'MMes':>8s} {'MMlj':>7s} {'rmsd':>5s} {'nW':>3s}  dU by group")
    for r in rows:
        print(f"{r['tag'][:10]:>10s} rep{r['rep']} {r['dU']:+9.2f} {abs(r['dU'] - r['dU_cache']):8.1e} {r['C_ML']:+9.2f} "
              f"{r['C_bb']:+9.2f} {r['C_MM']:+9.2f} {r['head_minus_bb']:+8.2f} {r['bb_minus_MM']:+8.2f} "
              f"{r['X_head'] - r['X_label']:+8.1e} {r['C_MM_es']:+8.2f} {r['C_MM_lj']:+7.2f} "
              f"{r['lig_heavy_rmsd_to_deep_A']:5.2f} {r['omega_O_within_3p5A_of_lig_NO']:3d}  "
              + " ".join(f"{g['dU']:+7.2f}" for g in r["groups"]))
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    main(ap.parse_args())
