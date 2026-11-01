"""Coupling response to ligand internal deformation only (user 2026-10-25): head vs frozen backbone vs MM.

Configurations: from rep1..rep4 S0 last frames (CypD-8), N_TRAJ fresh-velocity MM trajectories of 20 x 2 fs.
At every step t the ligand of x_t is rigidly fitted back onto the ligand of x0 (rigid-body motion removed,
constraint lengths kept); environment and the Omega water set stay those of x0. On each y_t:
    C_head = sum_i D_iE (head, 4 channels) + explicit Coulomb per atom   (local_E v1 C^ML, float64)
    C_bb   = E_bb(L + Omega) - E_bb(L) - E_bb(Omega)                      (frozen POLAR-1-M, float64)
    C_MM   = bare Coulomb + cutoff LJ over the same pairs                 (local_E v1 C^MM)
Split of the internal-deformation part of dU: (C_head - C_bb) + (C_bb - C_MM).
Run: python diag_internal_coupling.py [--n-traj 5]
"""

import argparse
import json
import time

import numpy as np

import eta_hmc as eh
from diag_fast_dU import NAME, STATE, Parts, kabsch_onto


def main(a):
    from decomp.clusters import coupling_labels
    leg = eh.local_E.Leg(NAME)
    me = eh.ModelEnergy(leg, STATE, "float64")
    parts = Parts(leg, me)
    system, groups, box = eh.mm_system(NAME)
    lig = np.array(leg.lig)
    ligset = set(lig.tolist())
    cons = [system.getConstraintParameters(k) for k in range(system.getNumConstraints())]
    cons = [(i, j, d.value_in_unit(eh.unit.nanometer)) for i, j, d in cons if i in ligset or j in ligset]
    ig = eh.proposal_integrator(1, 2.0)
    c = eh.make_context(system, box, ig)
    eh.gm.set_state(c, STATE, len(groups))
    rows, worst_con = [], 0.0
    t0 = time.perf_counter()
    for rep in (1, 2, 3, 4):
        for k in range(a.n_traj):
            c.setPositions(eh.start_positions(NAME, rep, STATE))
            c.applyConstraints(eh.TOL)
            ig.setGlobalVariableByName("draw", 1.0)
            xs = [eh.positions(c)]
            for _ in range(20):
                ig.step(1)
                ig.setGlobalVariableByName("draw", 0.0)
                xs.append(eh.positions(c))
            x0 = xs[0]
            om = parts(x0)["waters"]
            head, bb, mm, rmsd = [], [], [], []
            for x in xs:
                y = x0.copy()
                y[lig] = kabsch_onto(x[lig], x0[lig])
                worst_con = max(worst_con, max(abs(np.linalg.norm(y[i] - y[j]) - d) for i, j, d in cons))
                p, n, f, s = parts.carve_set(y, om)
                head.append(float(np.nansum(eh.couplings_dt(me.model, [(p, n, f)], me.dm, me.dtype)[0]) * eh.EV))
                bb.append(float(coupling_labels(me.model.backbone, [p], [n], [f], device="cuda")[0]) * eh.EV)
                mm.append(float(leg.c_mm(p, s, me.dm).sum()))
                rmsd.append(float(np.sqrt(((y[lig] - x0[lig]) ** 2).sum(-1).mean())))
            rows.append({"rep": rep, "traj": k, "head": head, "bb": bb, "mm": mm, "lig_internal_rmsd_nm": rmsd})
            H, B, M = np.array(head), np.array(bb), np.array(mm)
            print(f"  rep{rep} traj{k}: 0->40 fs  d head {H[-1] - H[0]:+7.2f}  d bb {B[-1] - B[0]:+7.2f}  "
                  f"d MM {M[-1] - M[0]:+7.2f}  | head-bb at t0 {H[0] - B[0]:+7.2f}  internal RMSD {rmsd[-1]:.4f} nm"
                  f"  | {time.perf_counter() - t0:.0f} s", flush=True)
    H = np.array([r["head"] for r in rows])
    B = np.array([r["bb"] for r in rows])
    M = np.array([r["mm"] for r in rows])
    kT = eh.KT
    steps = (1, 2, 4, 6, 10, 20)
    print(f"\n{len(rows)} trajectories; ligand internal deformation only, environment and Omega fixed at x0; "
          f"max constraint-length error {worst_con:.1e} nm")
    print("  sd over trajectories of X(t) - X(0), kJ/mol, at t = " + ", ".join(f"{2 * s} fs" for s in steps))
    series = {"C_head": H, "C_bb": B, "C_MM": M, "C_head - C_bb": H - B, "C_bb - C_MM": B - M,
              "C_head - C_MM (= dU, internal)": H - M}
    out = {"n": len(rows), "max_constraint_error_nm": worst_con, "t_fs": [2 * s for s in steps]}
    for lab, X in series.items():
        d = X - X[:, :1]
        v = [float(d[:, s].std(ddof=1)) for s in steps]
        out[lab] = v
        print(f"    {lab:32s} " + " ".join(f"{x:7.2f}" for x in v) + f"   ({v[-1] / kT:.2f} kT at 40 fs)")
    for (la, A), (lb, Bv) in ((("head", H), ("bb", B)), (("bb", B), ("MM", M)), (("head", H), ("MM", M))):
        da, db = A[:, 20] - A[:, 0], Bv[:, 20] - Bv[:, 0]
        r = float(np.corrcoef(da, db)[0, 1])
        slope = float((da * db).sum() / (db * db).sum())
        out[f"corr_40fs_{la}_vs_{lb}"] = r
        out[f"slope_40fs_{la}_on_{lb}"] = slope
        print(f"  40 fs change: corr({la}, {lb}) {r:+.3f}, slope {la}/{lb} {slope:+.3f}")
    # every step pooled (all t > 0): fast-part correlation without choosing one lag
    for (la, A), (lb, Bv) in ((("head", H), ("bb", B)), (("bb", B), ("MM", M))):
        da, db = (A[:, 1:] - A[:, :1]).ravel(), (Bv[:, 1:] - Bv[:, :1]).ravel()
        r = float(np.corrcoef(da, db)[0, 1])
        out[f"corr_allt_{la}_vs_{lb}"] = r
        print(f"  all t pooled: corr({la}, {lb}) {r:+.3f}")
    print(f"  mean C_head {H.mean():+.2f}, C_bb {B.mean():+.2f}, C_MM {M.mean():+.2f} kJ/mol; "
          f"internal RMSD at 40 fs {np.mean([r['lig_internal_rmsd_nm'][-1] for r in rows]):.4f} nm")
    json.dump({"summary": out, "rows": rows}, open(eh.HERE / "runs" / "diag_internal_coupling.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=5)
    main(ap.parse_args())
