"""Diagnosis (user 2026-10-25): why does dU = C^ML - C^MM change by 5-9 kT within one 40 fs MM proposal?

From rep1..rep4 S0 last frames (CypD-8), N_TRAJ fresh-velocity MM trajectories of 20 x 2 fs (the HMC
proposal, no accept step). At every step: C^ML and C^MM (Coulomb, LJ) per ligand atom and the Omega waters.
For the pair (x0, x20) the change of dU is split by counterfactual hybrids (float64 model, deterministic):
    lig      x0 with the ligand at x20                     (ligand motion only)
    lig_H    x0 with only the ligand H atoms at x20
    lig_rig  x0 with the x0 ligand rigidly fitted onto x20 (ligand rigid-body motion only)
    wat      x0 with every non-ligand atom at x20          (environment motion only)
    omega    x20 evaluated with the Omega water set of x0  (water-set swap effect = full - this)
Run: python diag_fast_dU.py [--n-traj 5]
"""

import argparse
import json
import sys
import time

import numpy as np

import eta_hmc as eh

NAME, STATE = "CyclophilinD_ligand-8", 0


def kabsch_onto(P, Q):
    """P rigidly moved onto Q (both [n, 3])."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (P - pc) @ R.T + qc


class Parts:
    def __init__(self, leg, me):
        self.leg, self.me = leg, me
        lig, _, box, numbers, resids, resnames, keep = leg.top
        self.box = np.diag(box)
        self.numbers, self.resids = numbers, resids
        self.wat_atoms = {r: np.where(resids == r)[0] for r in np.unique(resids[keep]) if resnames[np.where(resids == r)[0][0]] == "HOH"}

    def carve_set(self, x, waters):
        """ligand + the given water residues, each imaged next to the ligand (carve_from_arrays rule)."""
        lig = np.array(self.leg.lig)
        X = x * 10
        L = X[lig]
        pos, idx = [L], [lig]
        for r in waters:
            a = self.wat_atoms[r]
            dv = X[a][:, None] - L[None]
            shift = -np.round(dv.reshape(-1, 3).mean(0) / (self.box * 10)) * self.box * 10
            pos.append(X[a] + shift)
            idx.append(a)
        pos, s = np.concatenate(pos), np.concatenate(idx)
        frag = np.r_[np.zeros(len(lig), int), np.ones(len(s) - len(lig), int)]
        return pos, self.numbers[s], frag, s

    def __call__(self, x, waters=None):
        leg, me = self.leg, self.me
        if waters is None:
            p, n, f, s = leg.carve(x, 50)
        else:
            p, n, f, s = self.carve_set(x, waters)
        mm = leg.c_mm(p, s, me.dm)                                       # [n_lig, 2]
        ml = eh.couplings_dt(me.model, [(p, n, f)], me.dm, me.dtype)[0] * eh.EV
        w = np.unique(self.resids[s[len(leg.lig):]])
        return {"ml": ml, "mm": mm, "dU": ml.sum() - mm.sum(), "waters": w}


def main(a):
    leg = eh.local_E.Leg(NAME)
    me = eh.ModelEnergy(leg, STATE, "float64")
    parts = Parts(leg, me)
    system, groups, box = eh.mm_system(NAME)
    lig = np.array(leg.lig)
    ligset = set(lig.tolist())
    n_con_lig = sum(1 for k in range(system.getNumConstraints())
                    if set(system.getConstraintParameters(k)[:2]) & ligset)
    H = lig[leg.top[3][lig] == 1]                                         # by element, not by (HMR) mass
    print(f"{NAME}: ligand {len(lig)} atoms, {len(H)} H; constraints touching the ligand {n_con_lig} "
          f"(of {system.getNumConstraints()})")
    ig = eh.proposal_integrator(1, 2.0)
    c = eh.make_context(system, box, ig)
    eh.gm.set_state(c, STATE, len(groups))
    rng_seed = 0
    rows, series = [], []
    t0 = time.perf_counter()
    for rep in (1, 2, 3, 4):
        for k in range(a.n_traj):
            c.setPositions(eh.start_positions(NAME, rep, STATE))
            c.applyConstraints(eh.TOL)
            ig.setGlobalVariableByName("draw", 1.0)
            xs = [eh.positions(c)]
            for step in range(20):
                ig.step(1)
                ig.setGlobalVariableByName("draw", 0.0)
                xs.append(eh.positions(c))
            ev = [parts(x) for x in xs]
            x0, x20 = xs[0], xs[-1]
            p0, p20 = ev[0], ev[-1]
            hyb = {}
            y = x0.copy(); y[lig] = x20[lig]; hyb["lig"] = parts(y)
            y = x0.copy(); y[H] = x20[H]; hyb["lig_H"] = parts(y)
            y = x0.copy(); y[lig] = kabsch_onto(x0[lig], x20[lig]); hyb["lig_rig"] = parts(y)
            y = x20.copy(); y[lig] = x0[lig]; hyb["wat"] = parts(y)
            hyb["omega"] = parts(x20, waters=p0["waters"])
            r = {"rep": rep, "traj": k, "dU_t": [e["dU"] for e in ev],
                 "ml_t": [e["ml"].sum() for e in ev], "mm_t": [e["mm"].sum() for e in ev],
                 "omega_changed": int(len(np.setxor1d(p0["waters"], p20["waters"])) // 2),
                 "d_full": p20["dU"] - p0["dU"], "d_ml": p20["ml"].sum() - p0["ml"].sum(),
                 "d_mm_coul": p20["mm"][:, 0].sum() - p0["mm"][:, 0].sum(),
                 "d_mm_lj": p20["mm"][:, 1].sum() - p0["mm"][:, 1].sum(),
                 **{f"d_{h}": hyb[h]["dU"] - p0["dU"] for h in ("lig", "lig_H", "lig_rig", "wat")},
                 **{f"dml_{h}": hyb[h]["ml"].sum() - p0["ml"].sum() for h in ("lig", "lig_H", "lig_rig", "wat")},
                 **{f"dmm_{h}": hyb[h]["mm"].sum() - p0["mm"].sum() for h in ("lig", "lig_H", "lig_rig", "wat")},
                 "d_omega_swap": p20["dU"] - hyb["omega"]["dU"],
                 "atom_dml": (p20["ml"] - p0["ml"]).tolist(), "atom_dmm": (p20["mm"].sum(1) - p0["mm"].sum(1)).tolist(),
                 "lig_rms_move_nm": float(np.sqrt(((x20[lig] - x0[lig]) ** 2).sum(-1).mean())),
                 "H_rms_move_nm": float(np.sqrt(((x20[H] - x0[H]) ** 2).sum(-1).mean()))}
            rows.append(r)
            print(f"  rep{rep} traj{k}: d dU {r['d_full']:+7.2f} (ML {r['d_ml']:+7.2f}, MM coul {r['d_mm_coul']:+7.2f}, "
                  f"lj {r['d_mm_lj']:+6.2f}) | lig {r['d_lig']:+7.2f} H {r['d_lig_H']:+7.2f} rigid {r['d_lig_rig']:+7.2f} "
                  f"env {r['d_wat']:+7.2f} | omega swap {r['d_omega_swap']:+6.2f} ({r['omega_changed']} waters) "
                  f"| {time.perf_counter() - t0:.0f} s", flush=True)
    kT = eh.KT
    def sd(key):
        return float(np.std([r[key] for r in rows], ddof=1))
    print(f"\nsd over {len(rows)} trajectories of the 40 fs change (kJ/mol; kT {kT:.3f}):")
    for key, lab in (("d_full", "dU total"), ("d_ml", "  C^ML part"), ("d_mm_coul", "  C^MM Coulomb part"),
                     ("d_mm_lj", "  C^MM LJ part"), ("d_lig", "ligand moves only"), ("d_lig_H", "  ligand H only"),
                     ("d_lig_rig", "  ligand rigid body only"), ("d_wat", "environment moves only"),
                     ("d_omega_swap", "Omega water-set swap")):
        print(f"  {lab:28s} {sd(key):7.2f}  ({sd(key) / kT:.2f} kT)")
    for h in ("lig", "lig_H", "wat"):
        print(f"  {h:8s} ML part sd {sd('dml_' + h):6.2f}   MM part sd {sd('dmm_' + h):6.2f}")
    T = np.array([r["dU_t"] for r in rows])
    dt = T - T[:, :1]
    print("  sd of dU(t) - dU(0) at t = 2,4,8,12,20,40 fs: " +
          " ".join(f"{dt[:, s].std(ddof=1):.2f}" for s in (1, 2, 4, 6, 10, 20)))
    A = np.array([r["atom_dml"] for r in rows])
    B = np.array([r["atom_dmm"] for r in rows])
    order = np.argsort(-A.std(0))[:8]
    names = {i: ("H" if i in set(H.tolist()) else "heavy") for i in lig}
    print("  ligand atoms with the largest sd of the 40 fs change of C^ML_i (index in ligand, type, sd ML, sd MM):")
    for i in order:
        print(f"    {i:3d} {names[lig[i]]:5s}  {A[:, i].std(ddof=1):6.2f}  {B[:, i].std(ddof=1):6.2f}")
    print(f"  mean ligand RMS move {np.mean([r['lig_rms_move_nm'] for r in rows]):.4f} nm, "
          f"H {np.mean([r['H_rms_move_nm'] for r in rows]):.4f} nm; model calls {me.calls}, "
          f"{me.seconds / max(me.calls, 1) * 1e3:.0f} ms/call")
    json.dump(rows, open(eh.HERE / "runs" / "diag_fast_dU.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=5)
    main(ap.parse_args())
