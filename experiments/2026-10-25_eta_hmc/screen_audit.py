"""Full-target energies of the single-candidate torsion proposals rejected by the MM pre-screen
(FULL_TARGET_SCREEN_AUDIT_DECISION_2026-11-02.md). Read-only on S0_localgeom_v1; no new candidates, no MD.

Run: python screen_audit.py run    --out logs/screen_audit_<date>     (GPU: <= 269 model queries, <= 408 MM energies)
     python screen_audit.py report --out logs/screen_audit_<date>     (CPU: tables from the saved per-candidate file)

Per frame (branch 1 chain, k = 1): rebuild the chain from x_start and the recorded what / move / acc (a rejection keeps
the source); the source of step s is the chain state before s. One model query on x_start (checked against the panel
cache and the chain's start_dU); V(x_start) and V(y) for all 50 candidates (checked against start_V / V_y);
model queries only for the candidates the MM stage rejected (the others reuse their recorded dU_y).
    dV = V(y) - V(src), d(dU) = dU(y) - dU(src), eta = 1
    log p_DA   = min(0, -dV/RT) + min(0, -d(dU)/RT)
    log p_full = min(0, -(dV + d(dU))/RT)
Omega of source and candidate from the deterministic carve (no model). Model channels of the source are known only when
the source is x_start (accepted later states kept no components); candidate channels are recorded for every new query.
"""

import argparse
import json
import pathlib
import time

import numpy as np

SYSTEM = "CyclophilinD_ligand-8"
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "runs" / SYSTEM / "S0_localgeom_v1"
BINS = [(0, 5), (5, 10), (10, 15), (15, 20.0001)]               # |theta| deg, left-closed; the last includes 20


def run(a):
    import eta_hmc as eh
    import local_E
    from compare_ladders import kabsch_rmsd
    from deep_shallow_panel import compare_frame
    from eta_exchange import DETAILS, load_initial_panel
    from local_geom_mc import six_torsions, torsion_move
    from torsion_ncmc import dihedral, panel_model
    out = pathlib.Path(a.out)
    if out.with_suffix(".json").exists():
        raise SystemExit(f"preserving {out}.json; choose another --out")
    head, backbone = panel_model()
    leg = local_E.Leg(SYSTEM)
    me = eh.ModelEnergy(leg, 0, "float64", ckpt=head, backbone=backbone)
    system, groups, box = eh.mm_system(SYSTEM)
    import openmm
    c = eh.make_context(system, box, openmm.VerletIntegrator(0.001))
    eh.gm.set_state(c, 0, len(groups))
    heavy, _, tors = six_torsions()
    n_V = 0

    def V(x):
        nonlocal n_V
        n_V += 1
        c.setPositions(x)
        return c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(eh.KJ)

    def omega(x):
        return set(np.asarray(leg.carve(x, 50)[3][35:]).tolist())

    rows, checks = [], []
    t0 = time.perf_counter()
    for f in range(8):
        z = np.load(SRC / f"f{f}_b1.npz")
        m = json.loads(str(z["meta"]))
        R = json.loads(str(z["records"]))
        if m["k"] != 1 or not m["complete"] or len(R) != 50:
            raise SystemExit(f"f{f}_b1: not a complete single-candidate chain")
        x = np.array(z["x_start"], np.float64)
        # start recheck: model identity, Omega, atoms vs panel cache and the chain cache
        e0 = me(x)
        det0 = {k: np.asarray(me.last_components[k]).copy() for k in DETAILS}
        _, _, info = load_initial_panel(eh.HERE / "logs" / "deep_shallow_panel_20261031.npz", f)
        cmp = compare_frame(e0, det0, info["expected_dU"], info["expected_components"])
        V0 = V(x)
        chk = {"frame": f, "dU_start": e0, "dU_start_chain": m["start_dU"], "d_dU": e0 - m["start_dU"],
               "panel_cache_reproduced": bool(cmp["cache_reproduced"]), "V_start": V0, "dV_start": V0 - m["start_V"]}
        checks.append(chk)
        if not cmp["cache_reproduced"] or abs(e0 - m["start_dU"]) > 1e-3:
            raise SystemExit(f"frame {f}: start state does not reproduce: {chk}")
        src = {"x": x, "V": m["start_V"], "dU": m["start_dU"], "is_start": True, "omega": omega(x)}
        for s, r in enumerate(R):
            name, th = r["what"], float(r["move"])
            _, apply = torsion_move(name)
            y = apply(src["x"], th)
            Vy = V(y)
            hist = r["stage1"]
            if hist:
                dUy, det = r["dU_y"], None
            else:
                dUy = me(y)
                det = {k: np.asarray(me.last_components[k]).copy() for k in DETAILS}
                if not np.isfinite(dUy):
                    raise SystemExit(f"frame {f} step {s}: non-finite model energy")
            om_y = omega(y)
            dV, ddU = Vy - src["V"], dUy - src["dU"]
            row = {"frame": f, "step": s, "torsion": name, "theta_deg": float(np.degrees(th)),
                   "stage1_hist": bool(hist), "acc_hist": bool(r["acc"]), "model_from_history": bool(hist),
                   "V_y": Vy, "V_y_recorded": r["V_y"], "dV_y_check": Vy - r["V_y"], "dV": dV, "d_dU": ddU,
                   "dE": dV + ddU, "log_p_DA": min(0.0, -dV / eh.KT) + min(0.0, -ddU / eh.KT),
                   "log_p_full": min(0.0, -(dV + ddU) / eh.KT), "log_a1_hist": r["log_a1"],
                   "rmsd18_A": kabsch_rmsd(y[heavy] * 10, src["x"][heavy] * 10),
                   "dtors_deg": [float((dihedral(y[:35], t) - dihedral(src["x"][:35], t) + 180) % 360 - 180) for t in tors],
                   "omega_changed": int(len(om_y ^ src["omega"]) // 2), "source_is_start": src["is_start"]}
            if det is not None:
                row["cand_learned"] = float(det["C_ML_learned"].sum())
                row["cand_coul"] = float(det["C_ML_coul"].sum())
                row["cand_C_MM"] = float(det["C_MM"].sum())
                if src["is_start"]:
                    row["d_learned"] = row["cand_learned"] - float(det0["C_ML_learned"].sum())
                    row["d_coul"] = row["cand_coul"] - float(det0["C_ML_coul"].sum())
                    row["d_C_MM"] = row["cand_C_MM"] - float(det0["C_MM"].sum())
            rows.append(row)
            if r["acc"]:
                src = {"x": y, "V": Vy, "dU": dUy, "is_start": False, "omega": om_y}
            if abs(src["dU"] - r["dU"]) > 1e-3 and r["acc"]:
                raise SystemExit(f"frame {f} step {s}: rebuilt state dU differs from the record")
        if np.abs(src["x"] - z["x_final"]).max() > 0:
            raise SystemExit(f"frame {f}: rebuilt final state differs from x_final")
        print(f"frame {f}: start recheck ok (d dU {chk['d_dU']:+.1e}, dV {chk['dV_start']:+.1e}); "
              f"new queries {sum(1 for q in rows if q['frame'] == f and not q['model_from_history'])}", flush=True)
    res = {"source": str(SRC), "model_calls": me.calls, "mm_evals": n_V, "seconds": time.perf_counter() - t0,
           "max_abs_dV_y_check": max(abs(q["dV_y_check"]) for q in rows), "start_checks": checks, "rows": rows,
           "RT": eh.KT, "eta": 1.0, "date": time.strftime("%F %T")}
    out.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"model calls {me.calls}, MM evals {n_V}, max |V_y - recorded| {res['max_abs_dV_y_check']:.2e} kJ/mol, "
          f"{res['seconds']:.0f} s -> {out}.json", flush=True)


def report(a):
    res = json.loads(pathlib.Path(a.out).with_suffix(".json").read_text())
    rt = res["RT"]
    R = res["rows"]
    print(f"model calls {res['model_calls']}, MM evals {res['mm_evals']}, max |V_y - recorded| {res['max_abs_dV_y_check']:.2e}; "
          f"start rechecks max |d dU| {max(abs(c['d_dU']) for c in res['start_checks']):.1e}, "
          f"max |dV| {max(abs(c['dV_start']) for c in res['start_checks']):.1e}")

    def summ(sel, label):
        if not sel:
            print(f"  {label:34s} n   0")
            return
        pDA = np.exp([q["log_p_DA"] for q in sel])
        pF = np.exp([q["log_p_full"] for q in sel])
        d2 = np.array([q["rmsd18_A"] ** 2 for q in sel])
        dV = np.array([q["dV"] for q in sel])
        ddU = np.array([q["d_dU"] for q in sel])
        offset = np.mean((dV > 0) & (ddU < 0))
        print(f"  {label:34s} n {len(sel):3d}  sum p_DA {pDA.sum():6.2f}  sum p_full {pF.sum():6.2f}  "
              f"sum p_DA*d2 {np.sum(pDA * d2):6.3f}  sum p_full*d2 {np.sum(pF * d2):6.3f}  "
              f"dV med {np.median(dV):+6.1f}  d(dU) med {np.median(ddU):+6.1f}  dV>0&d(dU)<0 {offset:.2f}  RMSD med {np.median(np.sqrt(d2)):.2f}")

    print("\nall 400 candidates (same historical sources), by group; d2 = candidate RMSD18^2 (A^2)")
    summ(R, "all")
    summ([q for q in R if q["stage1_hist"]], "MM stage passed (139 historical)")
    summ([q for q in R if not q["stage1_hist"]], "MM stage rejected (261 new)")
    for t in ("T1", "T5"):
        summ([q for q in R if q["torsion"] == t], t)
    for lo, hi in BINS:
        summ([q for q in R if lo <= abs(q["theta_deg"]) < hi], f"|theta| [{lo},{min(hi, 20):g}]")
        summ([q for q in R if lo <= abs(q["theta_deg"]) < hi and not q["stage1_hist"]], f"  of which MM-rejected")
    for f in range(8):
        summ([q for q in R if q["frame"] == f], f"frame {f}")
    rej = [q for q in R if not q["stage1_hist"]]
    pF = np.exp([q["log_p_full"] for q in rej])
    big = [q for q in rej if abs(q["theta_deg"]) >= 10]
    print(f"\nMM-rejected: p_full > 0.1 for {int((pF > 0.1).sum())}, > 0.5 for {int((pF > 0.5).sum())}; "
          f"of |theta| >= 10: n {len(big)}, p_full > 0.1 for {sum(np.exp(q['log_p_full']) > 0.1 for q in big)}")
    pDA = np.exp([q["log_p_DA"] for q in R])
    pFa = np.exp([q["log_p_full"] for q in R])
    print(f"cost: direct full-target = 400 model queries (expected accepts {pFa.sum():.1f}); "
          f"two-stage expected queries sum p1 = {np.sum(np.exp([min(0, -q['dV'] / rt) for q in R])):.1f} (expected accepts {pDA.sum():.1f}); "
          f"historical two-stage queries 139 (accepts {sum(q['acc_hist'] for q in R)}); initial rechecks 8 not included")
    ch = [q for q in rej if "d_learned" in q]
    if ch:
        print(f"channels (MM-rejected candidates whose source is x_start, n {len(ch)}): d learned med {np.median([q['d_learned'] for q in ch]):+.1f}, "
              f"d coul med {np.median([q['d_coul'] for q in ch]):+.1f}, d C_MM med {np.median([q['d_C_MM'] for q in ch]):+.1f}")
    print(f"Omega changed (waters swapped) among all candidates: {sum(q['omega_changed'] > 0 for q in R)}")


def decompose(a):
    """MM only, no model: for the same 400 rebuilt (source, candidate) pairs, split dV into ligand-environment coupling
    (V - V_K, S_K = all groups dummy: environment-environment terms do not see the ligand), the ligand torsion term
    (PeriodicTorsionForce), the bond + angle terms (kept by a rigid rotation about a bond; checked) and the remaining
    intra-ligand nonbonded part (V_K - torsion - bond/angle). Writes <out>_decomp.json."""
    import openmm
    import eta_hmc as eh
    from local_geom_mc import torsion_move
    out = pathlib.Path(a.out)
    res = json.loads(out.with_suffix(".json").read_text())
    system, groups, box = eh.mm_system(SYSTEM)
    K = len(groups)
    for f_ in system.getForces():
        f_.setForceGroup(1 if isinstance(f_, openmm.PeriodicTorsionForce) else
                         2 if isinstance(f_, (openmm.HarmonicBondForce, openmm.HarmonicAngleForce)) else 0)
    c = eh.make_context(system, box, openmm.VerletIntegrator(0.001))
    n = [0]

    def parts(x):
        c.setPositions(x)
        out_ = {}
        for state, key in ((0, "V"), (K, "V_K")):
            eh.gm.set_state(c, state, K)
            out_[key] = c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(eh.KJ)
            n[0] += 1
        out_["tors"] = c.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(eh.KJ)
        out_["bond_angle"] = c.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(eh.KJ)
        return out_

    rows = res["rows"]
    dec = []
    i = 0
    for f in range(8):
        z = np.load(SRC / f"f{f}_b1.npz")
        R = json.loads(str(z["records"]))
        src = np.array(z["x_start"], np.float64)
        ps = parts(src)
        for s, r in enumerate(R):
            _, apply = torsion_move(r["what"])
            y = apply(src, float(r["move"]))
            py = parts(y)
            q = rows[i]
            assert (q["frame"], q["step"]) == (f, s)
            d = {k: py[k] - ps[k] for k in py}
            dec.append({"frame": f, "step": s, "torsion": q["torsion"], "theta_deg": q["theta_deg"],
                        "stage1_hist": q["stage1_hist"], "dV": d["V"], "dV_audit": q["dV"],
                        "d_coupling": d["V"] - d["V_K"], "d_tors": d["tors"], "d_bond_angle": d["bond_angle"],
                        "d_intra_nb": d["V_K"] - d["tors"] - d["bond_angle"], "d_dU": q["d_dU"]})
            i += 1
            if r["acc"]:
                src, ps = y, py
    out_d = pathlib.Path(str(out) + "_decomp.json")
    out_d.write_text(json.dumps({"mm_evals": n[0], "rows": dec}, indent=1) + "\n")
    D = dec
    print(f"MM energy evaluations {n[0]} (+ force-group reads), 0 model queries; max |dV - audit dV| "
          f"{max(abs(q['dV'] - q['dV_audit']) for q in D):.2e}; max |d bond+angle| {max(abs(q['d_bond_angle']) for q in D):.2e} kJ/mol")
    print("medians, kJ/mol:  dV = d coupling + d torsion + d intra-ligand nonbonded (+ d bond/angle ~ 0); d(dU) for reference")
    sel = {"all 400": D, "MM-rejected": [q for q in D if not q["stage1_hist"]]}
    for lo, hi in BINS:
        sel[f"MM-rejected |theta| [{lo},{min(hi, 20):g}]"] = [q for q in D if not q["stage1_hist"] and lo <= abs(q["theta_deg"]) < hi]
    for t in ("T1", "T5"):
        sel[f"MM-rejected {t} |theta|>=10"] = [q for q in D if not q["stage1_hist"] and q["torsion"] == t and abs(q["theta_deg"]) >= 10]
    for k, v in sel.items():
        med = lambda key: np.median([q[key] for q in v])  # noqa: E731
        print(f"  {k:36s} n {len(v):3d}  dV {med('dV'):+7.1f} = coupling {med('d_coupling'):+7.1f}, torsion {med('d_tors'):+6.1f}, "
              f"intra-nb {med('d_intra_nb'):+7.1f};  d(dU) {med('d_dU'):+6.1f}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "decompose"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": run, "report": report, "decompose": decompose}[a.cmd](a)
