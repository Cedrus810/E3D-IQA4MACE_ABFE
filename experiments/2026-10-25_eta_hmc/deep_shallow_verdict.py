"""Deep / shallow initial-state ladders: the evidence for the 2026-11-01 verdict table (DEEP_SHALLOW_STARTS §裁决判据).

Run: python -B deep_shallow_verdict.py --out logs/deep_shallow_verdict_<date>.json
Read-only on the eight _startdeep8_v1 / _startshallow8_v1 records and the panel; NumPy only, no model, no MD.

Per ladder, blocks of --block sweeps, all blocks kept (no burn-in chosen after the fact):
    <dU> at every eta label;
    eta=1 structure: ligand heavy-atom RMSD (A, after superposition) to the deep panel frame and to the ladder's own start;
    eta=1 Omega: fraction of the 50 water identities shared with the deep frame's Omega and with the own start's Omega;
    eta=1 deep-label occupancy: fraction of sweeps with dU < -90 kJ/mol (the old audit's descriptive marker, no gate);
    walker sources: distinct walkers seen at eta=1, share of the most frequent one, eta=0 -> eta=1 arrivals.
Final eta=1 states of all eight ladders: pairwise ligand RMSD and Omega overlap, to see whether both branches ended in
one common region. Reported only; no pass line.
"""

import argparse
import json
import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from compare_ladders import kabsch_rmsd, transport  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "CyclophilinD_ligand-8")
PANEL = os.path.join(HERE, "logs", "deep_shallow_panel_20261031.npz")
DEEP_MARK = -90.0


def heavy_atoms(meta, n_lig):
    # HMR can leave CH2 carbons at 5.97 Da; masses do not identify elements.
    import eta_hmc  # noqa: F401  (sys.path for model_eval)
    from model_eval import topology

    lig, _, _, numbers = topology(meta["system"])[:4]
    lig = np.asarray(lig, int)
    if len(lig) != n_lig or lig.tolist() != list(meta["ligand_system_indices"]):
        raise ValueError("topology ligand mapping differs from the saved ladder")
    return np.flatnonzero(np.asarray(numbers)[lig] > 1).tolist()


def ladder(branch, r, panel, block):
    path = os.path.join(RUNS, f"ladder_S0_start{r}_start{branch}8_v1.npz")
    z = np.load(path, allow_pickle=False)
    m = json.loads(str(z["meta"]))
    assert m["complete"] and m["panel_initialization"]["frame_index"] == (0 if branch == "deep" else r + 3)
    perm, ldU = z["perm"], z["label_dU"]
    n, K = perm.shape
    lig = np.array(m["ligand_system_indices"])
    heavy = heavy_atoms(m, len(lig))
    fi = m["panel_initialization"]["frame_index"]
    deep_lig = panel["xyz_nm"][0][lig][heavy] * 10
    own_lig = panel["xyz_nm"][fi][lig][heavy] * 10
    deep_om, own_om = set(panel["omega_water_resid"][0]), set(panel["omega_water_resid"][fi])
    top = perm[:, K - 1]                                         # walker at eta=1 per sweep
    lx = z["lig_xyz"][np.arange(n), top][:, heavy] * 10
    om = z["omega_water_resid"][np.arange(n), top]
    r_deep = np.array([kabsch_rmsd(x, deep_lig) for x in lx])
    r_own = np.array([kabsch_rmsd(x, own_lig) for x in lx])
    sh_deep = np.array([len(deep_om & set(o)) / 50 for o in om])
    sh_own = np.array([len(own_om & set(o)) / 50 for o in om])
    d1 = ldU[:, K - 1]
    up, rt, _ = transport(perm, K)
    blocks = []
    for b in range(0, n, block):
        sl = slice(b, min(b + block, n))
        w, c = np.unique(top[sl], return_counts=True)
        blocks.append({"sweeps": [b + 1, sl.stop], "mean_dU_by_eta": ldU[sl].mean(0).tolist(),
                       "eta1_rmsd_to_deep_A": float(r_deep[sl].mean()), "eta1_rmsd_to_own_start_A": float(r_own[sl].mean()),
                       "eta1_omega_shared_with_deep": float(sh_deep[sl].mean()),
                       "eta1_omega_shared_with_own_start": float(sh_own[sl].mean()),
                       "eta1_frac_dU_below_mark": float((d1[sl] < DEEP_MARK).mean()),
                       "eta1_distinct_walkers": int(len(w)), "eta1_top_walker_share": float(c.max() / c.sum())})
    below = d1 < DEEP_MARK
    return {"branch": branch, "rep": r, "path": path, "panel_frame": fi, "etas": m["etas"],
            "seed_base": m["seed_base"], "model_calls": m.get("model_calls_ladder"), "seconds": m.get("seconds"),
            "arrivals_0_to_1": int(up), "round_trips": int(rt),
            "eta1_entries_below_mark": int((~below[:-1] & below[1:]).sum()),
            "eta1_exits_below_mark": int((below[:-1] & ~below[1:]).sum()),
            "blocks": blocks, "final_eta1": {"dU": float(d1[-1]), "lig_heavy_A": lx[-1].tolist(),
                                             "omega": om[-1].tolist(), "walker": int(top[-1])}}


def main(a):
    if os.path.exists(a.out):
        raise SystemExit(f"preserving {a.out}; choose another --out")
    panel = np.load(PANEL, allow_pickle=False)
    L = [ladder(br, r, panel, a.block) for br in ("deep", "shallow") for r in (1, 2, 3, 4)]
    fin = [np.array(x["final_eta1"]["lig_heavy_A"]) for x in L]
    pair_rmsd = [[kabsch_rmsd(p, q) for q in fin] for p in fin]
    pair_om = [[len(set(x["final_eta1"]["omega"]) & set(y["final_eta1"]["omega"])) / 50 for y in L] for x in L]
    res = {"deep_mark_kJmol": DEEP_MARK, "block_sweeps": a.block, "ladders": L,
           "final_eta1_pairwise_lig_rmsd_A": pair_rmsd, "final_eta1_pairwise_omega_shared": pair_om,
           "note": "descriptive; no pass line; dU mark is the old audit's label, not a gate"}
    json.dump(res, open(a.out, "w"), indent=1)
    etas = L[0]["etas"]
    print(f"blocks of {a.block} sweeps; kJ/mol, A; eta1 = top label")
    for x in L:
        print(f"\n{x['branch']} rep{x['rep']} (panel frame {x['panel_frame']}): arrivals 0->1 {x['arrivals_0_to_1']}, "
              f"round trips {x['round_trips']}, eta1 entries/exits below {DEEP_MARK:g}: "
              f"{x['eta1_entries_below_mark']}/{x['eta1_exits_below_mark']}, final eta1 dU {x['final_eta1']['dU']:+.1f}")
        print("   sweeps   " + " ".join(f"{e:>7g}" for e in etas) + "  | rmsdD rmsdS  omD  omS  f<-90 nW top")
        for b in x["blocks"]:
            print(f"  {b['sweeps'][0]:4d}-{b['sweeps'][1]:<4d}" + " ".join(f"{v:+7.1f}" for v in b["mean_dU_by_eta"])
                  + f"  | {b['eta1_rmsd_to_deep_A']:5.2f} {b['eta1_rmsd_to_own_start_A']:5.2f} "
                    f"{b['eta1_omega_shared_with_deep']:4.2f} {b['eta1_omega_shared_with_own_start']:4.2f} "
                    f"{b['eta1_frac_dU_below_mark']:5.2f} {b['eta1_distinct_walkers']:2d} {b['eta1_top_walker_share']:4.2f}")
    names = [f"{x['branch'][0]}{x['rep']}" for x in L]
    print("\nfinal eta1 states: pairwise ligand heavy-atom RMSD (A) | Omega shared fraction")
    for i, nm in enumerate(names):
        print(f"  {nm}: " + " ".join(f"{v:4.2f}" for v in pair_rmsd[i]) + "  | " + " ".join(f"{v:4.2f}" for v in pair_om[i]))
    print(f"-> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
