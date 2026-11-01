"""Where is the fast part of C_head - C_bb under ligand internal deformation? (user 2026-10-25)

Same construction as diag_internal_coupling.py (new trajectories; the earlier configurations were not saved,
so C_head / C_bb / C_MM are recomputed here and must reproduce that table's magnitudes). Configurations are
saved this time. Per configuration (float64):
    head: per ligand atom i, the 4 learned channels (es, ex, ind, disp; es trained with the explicit Coulomb
          removed, E3D_ES=1) and the explicit Coulomb cross_i;  C_head = sum_i (sum_c D_ic + cross_i)
    backbone: C_bb (coupling_labels); d_es = E_es(L+O) - E_es(L) - E_es(O) and cross_bb (polar_es.es_coupling);
          short-range part bb_sr = C_bb - d_es
Split:  C_head - C_bb = [sum_c D_c - bb_sr]  +  [cross - d_es]
                        learned vs backbone short range   explicit Coulomb vs backbone electrostatics
Run: python diag_head_channels.py [--n-traj 5]
"""

import argparse
import json
import time

import numpy as np
import torch

import eta_hmc as eh
from diag_fast_dU import NAME, STATE, Parts, kabsch_onto

CH = ("es", "ex", "ind", "disp")


def head_parts(model, cluster, n_lig):
    """[n_lig, 4] learned channels and [n_lig] explicit Coulomb, kJ/mol (S0: no dummy)."""
    from decomp.data import mace_batch
    from decomp.polar_es import coulomb_cross
    p, n, f = cluster
    data = mace_batch([p], [n], model.backbone, charges=[0], frag_ids=[f], device="cuda", dtype=torch.float64)
    frag = data["frag_id"]
    in_L = frag == 0
    with torch.no_grad():
        with coulomb_cross(model.backbone, frag, per_atom=True) as rec:
            h = model.backbone(data, training=False, compute_force=False)["node_feats"]
        _, D = model.head(h, data["positions"], data["edge_index"])
    s, d = data["edge_index"]
    cr = in_L[s] != in_L[d]
    lig_end = torch.where(in_L[s], s, d)
    c4 = torch.zeros(len(frag), 4, dtype=D.dtype, device="cuda").index_add(0, lig_end[cr], 0.5 * D[cr])
    return (c4[:n_lig].cpu().numpy() * eh.EV, rec["per_atom"][:n_lig].cpu().numpy().astype(np.float64) * eh.EV)


def main(a):
    from decomp.clusters import coupling_labels
    from decomp.polar_es import es_coupling
    leg = eh.local_E.Leg(NAME)
    me = eh.ModelEnergy(leg, STATE, "float64")
    parts = Parts(leg, me)
    system, groups, box = eh.mm_system(NAME)
    lig = np.array(leg.lig)
    n_lig = len(lig)
    ig = eh.proposal_integrator(1, 2.0)
    c = eh.make_context(system, box, ig)
    eh.gm.set_state(c, STATE, len(groups))
    T = []                                                        # per trajectory dict of arrays [21, ...]
    confs = []
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
            r = {k_: [] for k_ in ("D", "cross", "C_bb", "d_es", "cross_bb", "C_MM")}
            ligpos = []
            for x in xs:
                y = x0.copy()
                y[lig] = kabsch_onto(x[lig], x0[lig])
                p, n, f, s = parts.carve_set(y, om)
                D, cross = head_parts(me.model, (p, n, f), n_lig)
                cb, des = es_coupling(me.model.backbone, p, n, f, device="cuda")
                r["D"].append(D)
                r["cross"].append(cross)
                r["C_bb"].append(float(coupling_labels(me.model.backbone, [p], [n], [f], device="cuda")[0]) * eh.EV)
                r["d_es"].append(des * eh.EV)
                r["cross_bb"].append(cb * eh.EV)
                r["C_MM"].append(float(leg.c_mm(p, s, me.dm).sum()))
                ligpos.append(y[lig].astype(np.float32))
            r = {k_: np.array(v) for k_, v in r.items()}
            T.append(r)
            confs.append({"rep": rep, "traj": k, "x0_omega_resids": om, "lig_xyz": np.array(ligpos)})
            Ch = r["D"].sum((1, 2)) + r["cross"].sum(1)
            print(f"  rep{rep} traj{k}: 0->40 fs d C_head {Ch[-1] - Ch[0]:+7.2f}  d C_bb {r['C_bb'][-1] - r['C_bb'][0]:+7.2f}"
                  f"  | cross head-path vs es_coupling max diff {np.abs(r['cross'].sum(1) - r['cross_bb']).max():.1e}"
                  f"  | {time.perf_counter() - t0:.0f} s", flush=True)
    D = np.array([r["D"] for r in T])                          # [traj, 21, n_lig, 4]
    X = np.array([r["cross"] for r in T])                      # [traj, 21, n_lig]
    Cbb = np.array([r["C_bb"] for r in T])
    des = np.array([r["d_es"] for r in T])
    Cmm = np.array([r["C_MM"] for r in T])
    Ch = D.sum((2, 3)) + X.sum(2)
    bb_sr = Cbb - des
    kT = eh.KT

    def dsd(Y, s=20):
        d = Y[:, s] - Y[:, 0]
        return float(d.std(ddof=1))

    def pooled(Y):
        return (Y[:, 1:] - Y[:, :1]).reshape(-1)

    print(f"\n{len(T)} trajectories, ligand internal deformation only (environment, Omega fixed at x0)")
    print(f"  means: C_head {Ch.mean():+.2f}  C_bb {Cbb.mean():+.2f}  (bb short range {bb_sr.mean():+.2f}, bb es {des.mean():+.2f})"
          f"  learned sum {D.sum((2, 3)).mean():+.2f}  explicit Coulomb {X.sum(2).mean():+.2f}  C_MM {Cmm.mean():+.2f}")
    rows = {"C_head": Ch, "C_bb": Cbb, "C_MM": Cmm, "C_head - C_bb": Ch - Cbb,
            "  learned sum - bb short range": D.sum((2, 3)) - bb_sr, "  explicit Coulomb - bb es": X.sum(2) - des,
            "learned sum": D.sum((2, 3)), "bb short range": bb_sr, "explicit Coulomb": X.sum(2), "bb es (d_es)": des}
    for i, ch in enumerate(CH):
        rows[f"  head channel {ch}"] = D[:, :, :, i].sum(2)
    print("  sd of X(t)-X(0) over trajectories (kJ/mol):      40 fs     pooled t")
    out = {}
    for lab, Y in rows.items():
        out[lab.strip()] = [dsd(Y), float(pooled(Y).std(ddof=1))]
        print(f"    {lab:34s} {dsd(Y):8.2f}  {pooled(Y).std(ddof=1):8.2f}")
    print("  correlations of the changes (all t pooled):")
    for la, A, lb, B in (("learned sum", D.sum((2, 3)), "bb short range", bb_sr),
                         ("explicit Coulomb", X.sum(2), "bb es", des),
                         ("C_head", Ch, "C_bb", Cbb), ("C_bb", Cbb, "C_MM", Cmm)):
        a_, b_ = pooled(A), pooled(B)
        r_ = float(np.corrcoef(a_, b_)[0, 1])
        sl = float((a_ * b_).sum() / (b_ * b_).sum())
        out[f"corr {la} vs {lb}"] = [r_, sl]
        print(f"    {la:18s} vs {lb:16s} corr {r_:+.3f}  slope {sl:+.3f}")
    for i, ch in enumerate(CH):
        a_ = pooled(D[:, :, :, i].sum(2))
        r_ = float(np.corrcoef(a_, pooled(bb_sr))[0, 1])
        print(f"    head {ch:5s} vs bb short range corr {r_:+.3f}")
    # per atom: which ligand atoms carry the fast change of the learned sum and of the explicit Coulomb
    names = leg.top[3][lig]
    da = (D.sum(3)[:, 1:] - D.sum(3)[:, :1]).reshape(-1, n_lig)
    dx = (X[:, 1:] - X[:, :1]).reshape(-1, n_lig)
    dch = np.stack([(D[:, 1:, :, i] - D[:, :1, :, i]).reshape(-1, n_lig) for i in range(4)], -1)
    order = np.argsort(-da.std(0))[:10]
    sym = {1: "H", 6: "C", 7: "N", 8: "O"}
    print("  ligand atoms by sd of the change of the learned sum (pooled t): atom, element, group, sd learned, "
          "sd es/ex/ind/disp, sd explicit Coulomb")
    gid = {a_: g + 1 for g, ix in enumerate(leg.gidx) for a_ in ix}
    for i in order:
        print(f"    {i:3d} {sym.get(int(names[i]), names[i]):2s} G{gid[i]}  {da[:, i].std():6.2f}   "
              + "/".join(f"{dch[:, i, c_].std():.2f}" for c_ in range(4)) + f"   {dx[:, i].std():6.2f}")
    out["atoms_by_learned_sd"] = [[int(i), int(names[i]), int(gid[i]), float(da[:, i].std()),
                                   [float(dch[:, i, c_].std()) for c_ in range(4)], float(dx[:, i].std())] for i in order]
    json.dump(out, open(eh.HERE / "runs" / "diag_head_channels.json", "w"), indent=1)
    np.savez(eh.HERE / "runs" / "diag_head_channels_data.npz", D=D, cross=X, C_bb=Cbb, d_es=des, C_MM=Cmm,
             lig_xyz=np.array([c_["lig_xyz"] for c_ in confs]),
             start=np.array([[c_["rep"], c_["traj"]] for c_ in confs]))
    print(f"  saved per-configuration data and ligand coordinates (environment = rep S0 last frame)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=5)
    main(ap.parse_args())
