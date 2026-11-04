"""Torsion + surrounding-water collective move on the original S0 target (TORSION_WATER_COLLECTIVE_MC_REVIEW_2026-11-02.md).

Run: python collective_mc.py check                       real panel frames: inverse, invariants, constraints, Gram, PBC (CPU)
     python collective_mc.py mmcheck --out logs/collective_mm_<date>   paired MM pre-check on the 400 historical proposals
                                                          (<= 1616 full-box MM energies, 0 model queries, 0 MD)

Move (one torsion, angle theta): the moving side rotates by theta about the axis a -> b (a, b fixed). Every whole water,
from its mass-weighted centre c taken at the periodic image nearest to a, gets cylinder coordinates (r, z) about that axis
and the weight
    w = S((r - r_core)/l) S((z_lo - z)/l) S((z - z_hi)/l),  S(t) = 1 (t <= 0), 1 - 3t^2 + 2t^3 (0 < t < 1), 0 (t >= 1)
    r_core = r_L + h, z_lo = z_min - h, z_hi = z_max + h      (r_L, z_min, z_max over the moving side: rotation invariants)
and rotates rigidly (all three atoms) by alpha = theta w about the same axis; the displacement is written back to the
water's own continuous coordinates (no per-atom wrapping). Ions and the fixed side do not move. r, z of every water centre
and the side invariants are unchanged by the move, so -theta inverts it; the Jacobian is 1 (shear in the cylinder angle,
rigid internal rotation, block-triangular overall). Applicability: the whole support (r <= r_core + l,
z_lo - l <= z <= z_hi + l) must lie within half the smallest box length of a, so each water has one image there.
Acceptance (local_geom_mc.two_stage_step): MM stage on the full S0 V, then exp(-eta d(dU)/RT); the model gives energy only.
"""

import argparse
import json
import pathlib
import time

import numpy as np

H_CORE = 0.30                                         # nm
L_TRANS = 0.30                                        # nm


class NotApplicable(ValueError):
    """The support of the water weight is not inside one periodic image around the axis anchor."""


def smooth(t):
    t = np.asarray(t, float)
    return np.where(t <= 0, 1.0, np.where(t >= 1, 0.0, 1 - 3 * t ** 2 + 2 * t ** 3))


def rodrigues(v, n, ang):
    """Rotate vectors v [..., 3] about unit n by angles ang (broadcast over leading dims)."""
    ang = np.asarray(ang, float)[..., None]
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(n, v) * s + n * (v @ n)[..., None] * (1 - c)


def geometry(x, a, b, side, waters, masses, box, h=H_CORE, ell=L_TRANS):
    """Axis, side invariants, water images, cylinder coordinates and weights for configuration x."""
    p = x[a]
    n = x[b] - p
    n = n / np.linalg.norm(n)
    v = x[side] - p
    zs = v @ n
    rs = np.linalg.norm(v - zs[:, None] * n, axis=1)
    rL, zmin, zmax = float(rs.max()), float(zs.min()), float(zs.max())
    rcore, zlo, zhi = rL + h, zmin - h, zmax + h
    half = 0.5 * float(np.min(box))
    reach = float(np.hypot(rcore + ell, max(abs(zlo - ell), abs(zhi + ell))))
    if reach >= half:
        raise NotApplicable(f"support reaches {reach:.3f} nm from the anchor, half box {half:.3f} nm")
    m = masses[waters]                                                  # [W, 3]
    X = x[waters]                                                       # [W, 3, 3]
    c = (m[..., None] * X).sum(1) / m.sum(1)[:, None]
    shift = -np.asarray(box) * np.round((c - p) / np.asarray(box))
    vc = c + shift - p
    zc = vc @ n
    rc = np.linalg.norm(vc - zc[:, None] * n, axis=1)
    w = smooth((rc - rcore) / ell) * smooth((zlo - zc) / ell) * smooth((zc - zhi) / ell)
    return {"p": p, "n": n, "rL": rL, "zmin": zmin, "zmax": zmax, "shift": shift, "rc": rc, "zc": zc, "w": w,
            "reach": reach, "half_box": half}


def collective_apply(x, theta, a, b, side, waters, masses, box, h=H_CORE, ell=L_TRANS, with_info=False):
    from local_geom_mc import rotate_about
    g = geometry(x, a, b, side, waters, masses, box, h, ell)
    axis = x[b] - x[a]
    out = rotate_about(x, x[a], axis / np.linalg.norm(axis), theta, side)   # bit-identical to local_geom_mc.torsion_move
    k = np.flatnonzero(g["w"] > 0)
    if k.size:
        X = x[waters[k]] + g["shift"][k][:, None, :]                    # image frame of each moved water
        Xn = g["p"] + rodrigues(X - g["p"], g["n"], (theta * g["w"][k])[:, None])
        out[waters[k]] = x[waters[k]] + (Xn - X)
    return (out, g) if with_info else out


def torsion_only_apply(x, theta, a, b, side):
    """The historical single-torsion candidate (local_geom_mc.torsion_move), bit for bit."""
    from local_geom_mc import rotate_about
    axis = x[b] - x[a]
    return rotate_about(x, x[a], axis / np.linalg.norm(axis), theta, side)


def gram(xyz, cons, masses=None):
    from local_geom_mc import gram as g
    return g(xyz, cons, masses)


# ---------------------------------------------------------------- molecular parts

SYSTEM = "CyclophilinD_ligand-8"
HERE = pathlib.Path(__file__).resolve().parent
PANEL = HERE / "logs" / "deep_shallow_panel_20261031.npz"
SRC = HERE / "runs" / SYSTEM / "S0_localgeom_v1"


def system_tables():
    import openmm
    import eta_hmc as eh
    from model_eval import topology
    system, groups, box = eh.mm_system(SYSTEM)
    masses = np.array([system.getParticleMass(i).value_in_unit(openmm.unit.dalton) for i in range(system.getNumParticles())])
    cons = [(int(i), int(j)) for i, j, _ in (system.getConstraintParameters(k) for k in range(system.getNumConstraints()))]
    resn = np.asarray(topology(SYSTEM)[5])
    resid = np.asarray(topology(SYSTEM)[4])
    w_atoms = np.flatnonzero(resn == "HOH")
    waters = w_atoms.reshape(-1, 3)
    assert (np.diff(resid[waters], axis=1) == 0).all() and (masses[waters[:, 0]] > 15).all()   # O first, one residue each
    return system, groups, np.diag(np.asarray(box, float)), masses, cons, waters


def torsion_spec(name):
    from torsion_ncmc import SIDES, TORSIONS
    return TORSIONS[name][1], TORSIONS[name][2], np.asarray(SIDES[name])


def check(a):
    from eta_exchange import load_initial_panel
    system, groups, box, masses, cons, waters = system_tables()
    rng = np.random.default_rng(3)
    cons_arr = np.asarray(cons)
    lig_cons = [c for c in cons if c[0] < 35]
    res = []
    for f in range(8):
        x, _, _ = load_initial_panel(PANEL, f)
        oh = np.linalg.norm(x[waters[:, 1]] - x[waters[:, 0]], axis=1)
        row = {"frame": f, "water_OH_range_nm": [float(oh.min()), float(oh.max())]}
        for name in ("T1", "T5"):
            ai, bi, side = torsion_spec(name)
            th = float(rng.uniform(-np.radians(20), np.radians(20)))
            y, g = collective_apply(x, th, ai, bi, side, waters, masses, box, with_info=True)
            g2 = geometry(y, ai, bi, side, waters, masses, box)
            back = collective_apply(y, -th, ai, bi, side, waters, masses, box)
            moved = np.flatnonzero(g["w"] > 0)
            dl = np.abs(np.linalg.norm(y[cons_arr[:, 0]] - y[cons_arr[:, 1]], axis=1)
                        - np.linalg.norm(x[cons_arr[:, 0]] - x[cons_arr[:, 1]], axis=1)).max()
            wc = [c for c in cons if c[0] in set(waters[moved].ravel())]
            dG = 0.0
            for block in (lig_cons, wc):
                if block:
                    dG = max(dG, np.abs(gram(y, block, masses) - gram(x, block, masses)).max(),
                             np.abs(gram(y, block) - gram(x, block)).max())
            # periodic: the same configuration with every water shifted by its own random box vector
            sh = rng.integers(-1, 2, (len(waters), 3)) * box
            xs = x.copy()
            xs[waters] += sh[:, None, :]
            ys = collective_apply(xs, th, ai, bi, side, waters, masses, box)
            dpbc = float(np.abs((ys - xs) - (y - x)).max())
            row[name] = {"theta_deg": float(np.degrees(th)), "inverse_nm": float(np.abs(back - x).max()),
                         "axis_moved_nm": float(max(np.abs(y[ai] - x[ai]).max(), np.abs(y[bi] - x[bi]).max())),
                         "invariants_change": float(max(abs(g2["rL"] - g["rL"]), abs(g2["zmin"] - g["zmin"]), abs(g2["zmax"] - g["zmax"]))),
                         "weights_change": float(np.abs(g2["w"] - g["w"]).max()),
                         "constraint_length_change_nm": float(dl), "gram_change": float(dG),
                         "pbc_image_change_nm": dpbc, "n_w1": int((g["w"] >= 1).sum()),
                         "n_transition": int(((g["w"] > 0) & (g["w"] < 1)).sum()), "reach_nm": g["reach"], "half_box_nm": g["half_box"]}
        res.append(row)
        print(f"frame {f}: water O-H {row['water_OH_range_nm'][0]:.5f}..{row['water_OH_range_nm'][1]:.5f} nm; " + "; ".join(
            f"{n} {r['theta_deg']:+.1f} deg: inverse {r['inverse_nm']:.1e}, axis {r['axis_moved_nm']:.0e}, invariants {r['invariants_change']:.1e}, "
            f"w {r['weights_change']:.1e}, constraints {r['constraint_length_change_nm']:.1e}, Gram {r['gram_change']:.1e}, "
            f"PBC {r['pbc_image_change_nm']:.1e}; waters w=1 {r['n_w1']}, transition {r['n_transition']}; reach {r['reach_nm']:.2f} < {r['half_box_nm']:.2f}"
            for n, r in ((k, row[k]) for k in ("T1", "T5"))), flush=True)
    pathlib.Path(a.out + "_check.json").write_text(json.dumps(res, indent=1) + "\n")


def mmcheck(a):
    import openmm
    import eta_hmc as eh
    from compare_ladders import kabsch_rmsd
    out = pathlib.Path(a.out + ".json")
    if out.exists():
        raise SystemExit(f"preserving {out}; choose another --out")
    system, groups, box, masses, cons, waters = system_tables()
    K = len(groups)
    c = eh.make_context(system, np.diag(box), openmm.VerletIntegrator(0.001))
    n_eval = [0]
    heavy = list(range(18))
    wat_set = waters

    def E(x):
        c.setPositions(x)
        e = {}
        for st, key in ((0, "V0"), (K, "VK")):
            eh.gm.set_state(c, st, K)
            e[key] = c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(eh.KJ)
            n_eval[0] += 1
        return e

    rows = []
    t0 = time.perf_counter()
    for f in range(8):
        z = np.load(SRC / f"f{f}_b1.npz")
        R = json.loads(str(z["records"]))
        src = np.array(z["x_start"], np.float64)
        es = E(src)
        for s, r in enumerate(R):
            name, th = r["what"], float(r["move"])
            ai, bi, side = torsion_spec(name)
            yR = torsion_only_apply(src, th, ai, bi, side)
            yC, g = collective_apply(src, th, ai, bi, side, waters, masses, box, with_info=True)
            if np.abs(yC[:35] - yR[:35]).max() != 0:
                raise RuntimeError("ligand coordinates of the two candidates differ")
            eR, eC = E(yR), E(yC)
            mv = np.flatnonzero(g["w"] > 0)
            still = np.setdiff1d(np.arange(len(waters)), mv)
            O = yC[waters[:, 0]]
            dmin_ww = np.inf
            if mv.size:
                d = O[mv][:, None] - O[still][None]
                d -= np.round(d / box) * box
                dmin_ww = float(np.linalg.norm(d, axis=-1).min())
            dl = yC[:35][:, None] - yC[waters[:, 0]][None]
            dl -= np.round(dl / box) * box
            dmin_lw = float(np.linalg.norm(dl, axis=-1).min())
            dwat = float(np.abs(yC[waters[mv]] - src[waters[mv]]).max()) if mv.size else 0.0
            d_int = eR["VK"] - es["VK"]
            d_env = eC["VK"] - eR["VK"]
            dC_C = (eC["V0"] - eC["VK"]) - (es["V0"] - es["VK"])
            dC_R = (eR["V0"] - eR["VK"]) - (es["V0"] - es["VK"])
            dV_C, dV_R = eC["V0"] - es["V0"], eR["V0"] - es["V0"]
            rows.append({"frame": f, "step": s, "torsion": name, "theta_deg": float(np.degrees(th)), "acc_hist": bool(r["acc"]),
                         "dV_R": dV_R, "dV_C": dV_C, "d_internal": d_int, "d_environment": d_env,
                         "dC_LE_R": dC_R, "dC_LE_C": dC_C, "closure_C": dV_C - (d_int + d_env + dC_C),
                         "dV_R_hist_check": eR["V0"] - r["V_y"],
                         "log_a1_R": min(0.0, -dV_R / eh.KT), "log_a1_C": min(0.0, -dV_C / eh.KT),
                         "n_w1": int((g["w"] >= 1).sum()), "n_transition": int(((g["w"] > 0) & (g["w"] < 1)).sum()),
                         "min_OO_moved_still_nm": dmin_ww, "min_lig_O_nm": dmin_lw, "max_water_disp_nm": dwat,
                         "rmsd18_A": kabsch_rmsd(yR[heavy] * 10, src[heavy] * 10)})
            if r["acc"]:
                src, es = yR, eR                                       # the historical chain moved to y_R
        print(f"frame {f}: done ({n_eval[0]} MM evaluations so far)", flush=True)
    if np.abs(src - z["x_final"]).max() != 0:
        raise RuntimeError("rebuilt final state differs")
    res = {"mm_evals": n_eval[0], "model_queries": 0, "md_steps": 0, "seconds": time.perf_counter() - t0, "RT": eh.KT,
           "h_nm": H_CORE, "l_nm": L_TRANS, "rows": rows, "date": time.strftime("%F %T")}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"MM evaluations {n_eval[0]}, {res['seconds']:.0f} s -> {out}", flush=True)


def report(a):
    res = json.loads(pathlib.Path(a.out + ".json").read_text())
    R = res["rows"]
    rt = res["RT"]
    print(f"MM evaluations {res['mm_evals']}, model queries 0, MD 0; {res['seconds']:.0f} s; "
          f"max |closure| {max(abs(q['closure_C']) for q in R):.1e}; max |V(y_R) - historical V_y| {max(abs(q['dV_R_hist_check']) for q in R):.1e} kJ/mol")
    bins = [(0, 5), (5, 10), (10, 15), (15, 20.0001)]

    def line(sel, lab):
        if not sel:
            return
        md = lambda k: np.median([q[k] for q in sel])  # noqa: E731
        aR = np.exp([q["log_a1_R"] for q in sel])
        aC = np.exp([q["log_a1_C"] for q in sel])
        d2 = np.array([q["rmsd18_A"] ** 2 for q in sel])
        dd = np.array([q["dV_C"] - q["dV_R"] for q in sel])
        print(f"  {lab:24s} n {len(sel):3d}  dV_R {md('dV_R'):+7.1f}  dV_C {md('dV_C'):+7.1f}  med(dV_C-dV_R) {np.median(dd):+7.1f}  "
              f"[internal {md('d_internal'):+5.1f}, environment {md('d_environment'):+6.1f}, coupling R {md('dC_LE_R'):+7.1f} / C {md('dC_LE_C'):+7.1f}]  "
              f"sum a1 R {aR.sum():6.2f} C {aC.sum():6.2f}  sum a1*d2 R {np.sum(aR * d2):.3f} C {np.sum(aC * d2):.3f}  "
              f"waters w=1 {md('n_w1'):.0f}, trans {md('n_transition'):.0f}; min O-O moved/still {md('min_OO_moved_still_nm'):.3f} nm")
    print("medians per candidate (kJ/mol); a1 = min(1, exp(-dV0/RT)); d2 = ligand RMSD18^2 (A^2), identical for R and C")
    line(R, "all")
    for t in ("T1", "T5"):
        line([q for q in R if q["torsion"] == t], t)
        for lo, hi in bins:
            line([q for q in R if q["torsion"] == t and lo <= abs(q["theta_deg"]) < hi], f"  {t} |theta| [{lo},{min(hi, 20):g}]")
    for f in range(8):
        line([q for q in R if q["frame"] == f], f"frame {f}")
    big = [q for q in R if abs(q["theta_deg"]) >= 10]
    fr = sorted({q["frame"] for q in big if np.exp(q["log_a1_C"]) > 0.1})
    print(f"|theta|>=10: n {len(big)}; a1 > 0.1 R {sum(np.exp(q['log_a1_R']) > 0.1 for q in big)}, C {sum(np.exp(q['log_a1_C']) > 0.1 for q in big)} "
          f"(C from frames {fr}); short contacts min O-O < 0.25 nm in C: {sum(q['min_OO_moved_still_nm'] < 0.25 for q in R)} of {len(R)}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "mmcheck", "report"])
    ap.add_argument("--out", default="logs/collective_20261104")
    a = ap.parse_args()
    {"check": check, "mmcheck": mmcheck, "report": report}[a.cmd](a)
