"""K2 endpoint energy query (DESIGN §7.1/§7.5/§10; experiments/2026-11-07_head_calibration/K2_DECISION_2026-11-07.md).

    C_bb,0 = E_bb(L+Omega) - E_bb(L) - E_bb(Omega)          (frozen POLAR-1-M, same carved cluster)
    dU_K2  = C_bb,0 - C^MM_Omega                            (scalar target; head-independent)
    cH_i   = learned_i + explicit Coulomb_i                 (head atom readout, attribution only)
    d_i    = cH_i - cMM_i,   q = C_bb,0 - sum_i cH_i,   sum_i d_i + q = dU_K2

Defined endpoints only: full ligand S0 (evaluated) and all-dummy S_K (zero by definition, no model call).
Partial-dummy C_bb,j is undefined: refused. The legacy head-total target (local_E v1, eta_hmc.ModelEnergy)
is untouched; K2 component keys deliberately differ (no "dU_atom", whose sum was the old scalar), and the
K2 identity carries E_construction=local_E_bb_v1, so eta_records.check_contract refuses mixing both ways.
"""

import os
import pathlib
import time

import numpy as np

import eta_hmc as eh

TARGET = "local_E_bb_v1"
ATTRIBUTION = "head_raw_plus_global_remainder_v1"
N_WATER = 50
OMEGA_RULE = "ligand + 50 nearest complete waters, periodic imaging (local_E v1 carve)"
K2_DETAILS = ("C_H", "C_H_learned", "C_H_coul", "C_MM", "d_atom", "C_bb", "q",
              "cluster_system_indices", "omega_water_resid")


def backend_identity(backbone):
    """Real-space long-range evaluator actually present in the loaded backbone (read from the modules)."""
    rows = []
    for path, mod in backbone.named_modules():
        name = type(mod).__name__
        if "RealSpace" in name or name in ("AnalyticEnergy", "AnalyticFeatures"):
            rows.append(f"{path}:{name}:offset={getattr(mod, 'offset', None)}")
    if not rows:
        raise ValueError("no real-space long-range evaluator found in the backbone")
    kinds = {r.split(":")[1] for r in rows}
    if kinds <= {"RealSpaceFiniteDiffereneEnergy", "RealSpaceFiniteDifferenceElectrostaticFeatures"}:
        label = "graph_longrange_stencil"
    elif kinds <= {"AnalyticEnergy", "AnalyticFeatures"}:
        label = "analytic_l01_v1"
    else:
        raise ValueError(f"mixed / unknown real-space evaluators: {sorted(kinds)}")
    return {"realspace_backend": label, "realspace_modules": sorted(rows)}


def k2_identity(head, backbone, model, dtype_name, total_charge):
    """Contract fields for K2 chains / caches. Paths are identities, mtimes provenance (eta_records)."""
    bb, hd = str(pathlib.Path(backbone).resolve()), str(pathlib.Path(head).resolve())
    return {"E_construction": TARGET, "attribution_definition": ATTRIBUTION,
            "backbone": bb, "head": hd, "backbone_mtime": os.path.getmtime(bb), "head_mtime": os.path.getmtime(hd),
            **backend_identity(model.backbone), "dtype": dtype_name, "total_charge": int(total_charge),
            "n_water": N_WATER, "omega_rule": OMEGA_RULE}


def endpoint_kind(dummy):
    if not dummy.any():
        return "S0"
    if dummy.all():
        return "SK"
    raise NotImplementedError("K2 C_bb for partial-dummy endpoints is not defined (K2 contract §2)")


def compose(c_h_atom, learned_atom, coul_atom, c_mm, c_bb, dummy, sysidx, omega):
    """Pure K2 bookkeeping in kJ/mol. c_mm is [n_lig, 2] (Coulomb, LJ). Returns (dU_K2, components)."""
    c_h_atom, learned_atom, coul_atom = (np.asarray(v, np.float64) for v in (c_h_atom, learned_atom, coul_atom))
    c_mm = np.asarray(c_mm, np.float64)
    if not (np.isfinite(c_h_atom).all() and np.isfinite(c_mm).all() and np.isfinite(c_bb)):
        raise FloatingPointError("non-finite K2 component")
    if np.any(c_h_atom[dummy] != 0) or np.any(c_mm[dummy] != 0):
        raise ValueError("dummy rows must be zero")
    d = c_h_atom - c_mm.sum(axis=1)
    q = float(c_bb) - float(c_h_atom.sum())
    du = float(c_bb) - float(c_mm.sum())
    comp = {"C_H": c_h_atom, "C_H_learned": learned_atom, "C_H_coul": coul_atom, "C_MM": c_mm, "d_atom": d,
            "C_bb": np.asarray(float(c_bb)), "q": np.asarray(q),
            "cluster_system_indices": np.asarray(sysidx, np.int64), "omega_water_resid": np.asarray(omega, np.int64)}
    return du, comp


class ModelEnergyK2:
    """dU_K2(x) in kJ/mol for one full-box frame (nm). Explicit head and backbone are required."""

    def __init__(self, leg, j, head, backbone, dtype_name="float64", device="cuda"):
        import torch
        from model_eval import load_model
        if head is None or backbone is None:
            raise ValueError("K2 requires explicit head and backbone paths")
        if dtype_name != "float64":
            raise ValueError("K2 C_bb is a difference of ~1e4 eV totals; float64 only")
        self.leg, self.dm, self.device = leg, leg.dummy(j), device
        self.kind = endpoint_kind(self.dm)
        self.total_charge = int(np.rint(leg.q[np.asarray(leg.lig)].sum()))
        if self.total_charge != 0:     # decomp.clusters.coupling_labels evaluates every fragment at charge 0
            raise NotImplementedError(f"ligand net charge {self.total_charge}: charged K2 C_bb not implemented")
        self.dtype = getattr(torch, dtype_name)
        self.model = load_model(ckpt=head, backbone=backbone, device=device).to(self.dtype)
        self.identity = k2_identity(head, backbone, self.model, dtype_name, self.total_charge)
        self.calls, self.seconds = 0, 0.0

    def c_bb(self, p, n, f):
        from decomp.clusters import coupling_labels
        return float(coupling_labels(self.model.backbone, [p], [n], [f], device=self.device, batch=1)[0] * eh.EV)

    def __call__(self, xyz_nm):
        t = time.perf_counter()
        na = len(self.dm)
        p, n, f, s = self.leg.carve(np.asarray(xyz_nm, np.float64), N_WATER)
        omega = np.unique(self.leg.top[4][s[na:]])
        if self.kind == "SK":           # no coupled atom: every coupling is zero by definition, no model call
            z = np.zeros(na)
            du, comp = compose(z, z, z, np.zeros((na, 2)), 0.0, self.dm, s, omega)
        else:
            mm = self.leg.c_mm(p, s, self.dm)
            tot, lrn, cou = eh.couplings_dt(self.model, [(p, n, f)], self.dm, self.dtype, split=True, device=self.device)
            du, comp = compose(tot[0] * eh.EV, lrn[0] * eh.EV, cou[0] * eh.EV, mm, self.c_bb(p, n, f), self.dm, s, omega)
            self.calls += 1
        self.last_components = comp
        self.seconds += time.perf_counter() - t
        return du
