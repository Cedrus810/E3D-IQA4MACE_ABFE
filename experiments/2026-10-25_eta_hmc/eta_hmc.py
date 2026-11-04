"""Endpoint MM->model correction by MM-HMC proposals and model-energy Metropolis (DESIGN §10, user 2026-10-25).

    U_{j,eta}(x) = V_j(x) + eta * dU_j(x),   dU_j = C^ML - C^MM  (local_E v1, frozen)
    proposal: fresh Maxwell-Boltzmann momenta, L velocity-Verlet steps on V_j with RATTLE-type
              constraints (reversible, volume preserving), no thermostat, no CM-motion remover
    accept:   min{1, exp[-beta (dV + dK + eta * d dU)]}; on reject the old positions are restored
The model never gives a force; it is called once per proposal, on the proposed configuration.

Run: python eta_hmc.py repeat   SYSTEM [--frames 10] [--repeats 5]        step 1: energy-query repeatability
     python eta_hmc.py reverse  SYSTEM [--L 20] [--dt-fs 2]               step 2a: reversibility, dH
     python eta_hmc.py analytic SYSTEM [--L 20] [--cycles 2000]           step 2b: accept/reject vs Langevin,
                                                                           on a cheap known target V + f
     python eta_hmc.py chain    SYSTEM --eta E --start REP [--L 20] [--cycles N] [--tag T]   step 3 (model)
     python eta_hmc.py analyze  SYSTEM [--tag T] [--expect-etas 0.0,...,1.0] [--expect-starts 1,2,3,4]
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import openmm
from openmm import unit

E3D = pathlib.Path("/home/ruigengji/E3D")
sys.path.insert(0, str(E3D))
sys.path.insert(0, str(E3D / "experiments/2026-10-23_groupwise_dummy"))
sys.path.insert(0, str(E3D / "experiments/2026-10-24_local_E"))
import groupwise_mm as gm  # noqa: E402
import local_E  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
KJ = unit.kilojoule_per_mole
KT = gm.KT                                            # kJ/mol, 300 K
TOL = 1e-7                                            # constraint tolerance for HMC
# Context readback vs cached accepted state. CUDA mixed precision stores positions as float32 + correction; a
# restore/readback can differ by up to one float32 rounding (ulp at 8 nm = 9.5e-7 nm; measured 1.4e-14 typical,
# 7.1e-7 nm accumulated over 275 restores when readbacks were fed back, 2026-10-30). A wrong state is >= 1e-2 nm.
POS_TOL = 1e-6                                        # nm
EV = 96.485332


# ---------------------------------------------------------------- model energy (local_E v1)

def couplings_dt(model, clusters, dummy, dtype, split=False, device="cuda"):
    """step_eval.couplings with an explicit dtype; float32 reproduces it.
    split=True also returns the learned-channel and explicit-Coulomb parts (same forward), eV.
    device defaults to the sampling device; CPU is available for fixed-frame audits."""
    import torch
    from decomp.data import mace_batch
    from decomp.polar_es import coulomb_cross
    n_lig = len(dummy)
    data = mace_batch([c[0] for c in clusters], [c[1] for c in clusters], model.backbone,
                      charges=[0] * len(clusters), frag_ids=[c[2] for c in clusters], device=device, dtype=dtype)
    frag = data["frag_id"]
    in_L = frag == 0
    dm = torch.as_tensor(np.concatenate([np.r_[dummy, np.zeros(len(c[1]) - n_lig, bool)] for c in clusters]),
                         device=device)
    src, dst = data["edge_index"]
    keep = ~((in_L[src] != in_L[dst]) & (dm[src] | dm[dst]))
    sub = {**data, "edge_index": data["edge_index"][:, keep], "shifts": data["shifts"][keep],
           "unit_shifts": data["unit_shifts"][keep]}
    with torch.no_grad():
        with coulomb_cross(model.backbone, frag, per_atom=True) as rec:
            h = model.backbone(sub, training=False, compute_force=False)["node_feats"]
        _, D = model.head(h, sub["positions"], sub["edge_index"])
    D = D.sum(-1) if D.dim() > 1 else D
    s, d = sub["edge_index"]
    cr = in_L[s] != in_L[d]
    lig_end = torch.where(in_L[s], s, d)
    learned = torch.zeros(len(frag), dtype=D.dtype, device=device).index_add(0, lig_end[cr], 0.5 * D[cr])
    coul = rec["per_atom"].to(D.dtype)
    c = learned + coul
    rows = data["ptr"][:-1, None] + torch.arange(n_lig, device=device)[None, :]
    parts = [t[rows].detach().cpu().numpy().astype(np.float64) for t in ((c, learned, coul) if split else (c,))]
    for p in parts:
        p[:, dummy] = np.nan
    return tuple(parts) if split else parts[0]


class ModelEnergy:
    """dU_j(x) in kJ/mol for one full-box frame (nm), local_E v1."""

    def __init__(self, leg, j, dtype_name="float64", ckpt=None, backbone=None, device="cuda"):
        import torch
        from model_eval import load_model
        self.leg, self.dm, self.device = leg, leg.dummy(j), device
        self.dtype = getattr(torch, dtype_name)
        self.model = load_model(ckpt=ckpt, backbone=backbone, device=device).to(self.dtype)
        self.calls, self.seconds = 0, 0.0

    def __call__(self, xyz_nm):
        t = time.perf_counter()
        p, n, f, s = self.leg.carve(np.asarray(xyz_nm, np.float64), 50)
        mm = self.leg.c_mm(p, s, self.dm)
        ml_atom, ml_learned, ml_coul = (np.zeros(len(self.dm), dtype=np.float64) for _ in range(3))
        if self.dm.all():
            ml = 0.0
        else:                       # dummy columns are NaN by construction; every active atom must be finite
            tot, lrn, cou = couplings_dt(self.model, [(p, n, f)], self.dm, self.dtype, split=True, device=self.device)
            act = tot[0, ~self.dm]
            if not np.isfinite(act).all():
                raise FloatingPointError(f"non-finite C^ML on active ligand atoms {np.where(~np.isfinite(act))[0]}")
            ml_atom[~self.dm] = act * EV
            ml_learned[~self.dm], ml_coul[~self.dm] = lrn[0, ~self.dm] * EV, cou[0, ~self.dm] * EV
            ml = act.sum()
        if not np.isfinite(mm).all():
            raise FloatingPointError("non-finite C^MM")
        self.last_components = {
            "C_ML": ml_atom, "C_ML_learned": ml_learned, "C_ML_coul": ml_coul, "C_MM": np.asarray(mm, np.float64),
            "dU_atom": ml_atom - mm.sum(axis=1),
            "cluster_system_indices": np.asarray(s, np.int64),
            "omega_water_resid": np.unique(self.leg.top[4][s[len(self.dm):]]).astype(np.int64)}
        self.calls += 1
        self.seconds += time.perf_counter() - t
        return ml * EV - mm.sum()


def model_identity(ckpt=None, backbone=None):
    """Backbone / head files ModelEnergy loads (model_eval.BB, CKPT). Paths decide reuse; mtimes are provenance."""
    import os
    import model_eval
    bb, head = str(pathlib.Path(backbone or model_eval.BB).resolve()), str(pathlib.Path(ckpt or model_eval.CKPT).resolve())
    return {"backbone": bb, "head": head,
            "backbone_mtime": os.path.getmtime(bb), "head_mtime": os.path.getmtime(head)}


# ---------------------------------------------------------------- MM proposal

def proposal_integrator(L, dt_fs, temperature_K=gm.T):
    ig = openmm.CustomIntegrator(dt_fs * 1e-3)
    for g, v in (("kT", gm.thermal_energy(temperature_K)), ("draw", 1.0), ("ke", 0.0), ("Eold", 0.0), ("Enew", 0.0), ("Vold", 0.0), ("Vnew", 0.0)):
        ig.addGlobalVariable(g, v)
    ig.addPerDofVariable("x1", 0)
    ig.addComputePerDof("v", "draw*sqrt(kT/m)*gaussian + (1-draw)*v")
    ig.addConstrainVelocities()
    ig.addComputeSum("ke", "0.5*m*v*v")
    ig.addComputeGlobal("Vold", "energy")
    ig.addComputeGlobal("Eold", "ke + energy")
    for _ in range(L):
        ig.addComputePerDof("v", "v + 0.5*dt*f/m")
        ig.addComputePerDof("x", "x + dt*v")
        ig.addComputePerDof("x1", "x")
        ig.addConstrainPositions()
        ig.addComputePerDof("v", "v + 0.5*dt*f/m + (x - x1)/dt")
        ig.addConstrainVelocities()
    ig.addComputeSum("ke", "0.5*m*v*v")
    ig.addComputeGlobal("Vnew", "energy")
    ig.addComputeGlobal("Enew", "ke + energy")
    ig.setConstraintTolerance(TOL)
    return ig


def mm_system(name, extra=None):
    """groupwise_mm System (group switches), CM-motion remover removed (it would break detailed balance)."""
    _, system, groups, _, box = gm.build(name)
    for i in reversed(range(system.getNumForces())):
        if isinstance(system.getForce(i), openmm.CMMotionRemover):
            system.removeForce(i)
    if extra is not None:
        system.addForce(extra)
    return system, groups, box


def make_context(system, box, integrator, platform="CUDA"):
    plat = openmm.Platform.getPlatformByName(platform)
    c = openmm.Context(system, integrator, plat, {"Precision": "mixed"} if platform == "CUDA" else {})
    c.setPeriodicBoxVectors(*[openmm.Vec3(*row) for row in np.asarray(box)])
    return c


def start_positions(name, rep, j):
    return np.load(local_E.GW / "runs" / name / f"rep{rep}_S{j}.npz")["last_positions"].astype(np.float64)


def positions(c):
    return c.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)


def hmc(c, ig, x0, eta, cycles, energy, rng, record=None, log_every=0, capture=None, initial=None):
    """Metropolis on U = V + eta * energy(x). energy(x) is evaluated once per proposal and once at x0."""
    rt = ig.getGlobalVariableByName("kT")
    if initial is None:
        c.setPositions(x0)
        c.applyConstraints(TOL)
        x = positions(c)
        e = energy(x)
        details = capture() if capture is not None else None
    else:
        # Context / integrator and RNG have already been restored. Do not
        # reset positions, project constraints or redraw momenta here.
        held = positions(c)
        if np.abs(held - initial["x"]).max() > POS_TOL:
            raise ValueError("checkpoint positions differ from the saved accepted state")
        # The chain state stays the exact cached x (where e was evaluated); the Context is within one rounding.
        x = np.array(initial["x"], np.float64)
        e, details = float(initial["e"]), initial.get("details")
    if not np.isfinite(e):
        raise FloatingPointError("non-finite initial target energy")
    out = {k: [] for k in ("acc", "dH", "dE", "V", "e", "prop_V")}
    rec, detail_rows, prop_rec, prop_rows = [], [], [], []
    for t in range(cycles):
        ig.setGlobalVariableByName("draw", 1.0)
        ig.step(1)
        dH = ig.getGlobalVariableByName("Enew") - ig.getGlobalVariableByName("Eold")
        xn = positions(c)
        en = energy(xn)
        proposed_details = capture() if capture is not None else None
        if not np.isfinite(dH) or not np.isfinite(en):
            raise FloatingPointError("non-finite proposal energy / integration work")
        logp = -(dH + eta * (en - e)) / rt
        acc = np.log(rng.random()) < logp
        prop = en - e
        Vn = ig.getGlobalVariableByName("Vnew")
        if acc:
            x, e, V = xn, en, Vn
            details = proposed_details
        else:
            # Restore from the exact cached x every time; never feed a readback back in (rounding would accumulate).
            c.setPositions(x)
            V = ig.getGlobalVariableByName("Vold")
        for k, v in zip(("acc", "dH", "dE", "V", "e", "prop_V"), (acc, dH, prop, V, e, Vn)):
            out[k].append(v)
        if record is not None:
            rec.append(record(x))
            prop_rec.append(record(xn))      # proposal diagnostics; never enters the chain averages
        if capture is not None:
            # On rejection this is the old state, including its Omega and
            # atomic energy; never the rejected proposal's decomposition.
            detail_rows.append(details)
            prop_rows.append(proposed_details)
        if log_every and (t + 1) % log_every == 0:
            a = np.mean(out["acc"][-log_every:])
            print(f"    cycle {t + 1}: acc {a:.2f}  <dH> {np.mean(out['dH'][-log_every:]):+.2f}  e {e:+.2f}", flush=True)
    out = {k: np.array(v) for k, v in out.items()}
    if record is not None:
        out["rec"], out["prop_rec"] = np.array(rec), np.array(prop_rec)
    if capture is not None:
        out["details"] = {k: np.stack([r[k] for r in detail_rows]) for k in details} if detail_rows else {}
        out["prop_details"] = {k: np.stack([r[k] for r in prop_rows]) for k in details} if prop_rows else {}
    out["final_state"] = {"x": x.copy(), "e": e, "details": details}
    return out


def mm_updates(c, ig, x, m, rng):
    """m MM-HMC substeps on V alone (the eta = 0 target; no model query): fresh momenta each, Metropolis on its own
    dH, rejection restores the exact cached x before the next substep (ETA0_MM_UPDATES_DECISION_2026-11-01.md).
    Each substep keeps exp(-V/kT); so does their composition. Returns (x, V, dH[m], acc[m])."""
    rt = ig.getGlobalVariableByName("kT")
    dHs, accs = np.full(m, np.nan), np.zeros(m, bool)
    V = None
    for t in range(m):
        ig.setGlobalVariableByName("draw", 1.0)
        ig.step(1)
        dH = ig.getGlobalVariableByName("Enew") - ig.getGlobalVariableByName("Eold")
        if V is None:
            V = ig.getGlobalVariableByName("Vold")
        if not np.isfinite(dH):
            raise FloatingPointError("non-finite MM integration work")
        dHs[t] = dH
        accs[t] = np.log(rng.random()) < -dH / rt
        if accs[t]:
            x, V = positions(c), ig.getGlobalVariableByName("Vnew")
        else:
            c.setPositions(x)
    return x, V, dHs, accs


def stat_ineff(x):
    from pymbar import timeseries
    return float(timeseries.statistical_inefficiency(x)) if np.std(x) > 0 else 1.0


def mean_se(x):
    return float(np.mean(x)), float(np.std(x, ddof=1) * np.sqrt(stat_ineff(x) / len(x)))


# ---------------------------------------------------------------- step 1

def repeat(a):
    leg = local_E.Leg(a.system)
    j = a.state
    dm = leg.dummy(j)
    from step_eval import couplings
    xyz = np.load(local_E.GW / "runs" / a.system / f"rep1_S{j}.npz")["xyz"][:a.frames].astype(np.float64)
    res = {}
    for dt in ("float32", "float64"):
        me = ModelEnergy(leg, j, dt)
        v = np.array([[me(x) for _ in range(a.repeats)] for x in xyz])          # [frames, repeats]
        res[dt] = v
        sp = (v.max(1) - v.min(1))
        print(f"{dt}: dU per frame (first repeat) {np.round(v[:, 0], 3).tolist()}")
        print(f"{dt}: repeat spread max {sp.max():.3e}, mean {sp.mean():.3e} kJ/mol over {a.frames} frames x "
              f"{a.repeats}; {me.seconds / me.calls * 1e3:.0f} ms per call", flush=True)
        if dt == "float32":                                                     # identical to step_eval.couplings
            cl = [leg.carve(x, 50) for x in xyz[:2]]
            ref = couplings(me.model, [(p, n, f) for p, n, f, _ in cl], dm)
            new = couplings_dt(me.model, [(p, n, f) for p, n, f, _ in cl], dm, me.dtype)
            print(f"float32 couplings_dt vs step_eval.couplings, batch 2: max |diff| "
                  f"{np.nanmax(np.abs(ref - new)) * EV:.2e} kJ/mol per atom")
        del me
    d = res["float64"].mean(1) - res["float32"].mean(1)
    print(f"float64 - float32 (repeat means): max |diff| {np.abs(d).max():.3e}, mean {d.mean():+.3e} kJ/mol")
    np.savez(HERE / "runs" / f"repeat_{a.system}_S{j}.npz", **res)


# ---------------------------------------------------------------- step 2a

def reverse(a):
    KT = gm.thermal_energy(gm.temperature(a.system))
    system, groups, box = mm_system(a.system)
    K = len(groups)
    ig = proposal_integrator(a.L, a.dt_fs, gm.temperature(a.system))
    c = make_context(system, box, ig)
    gm.set_state(c, a.state, K)
    rng = np.random.default_rng(1)
    errs, dHs = [], []
    for rep in (1, 2, 3, 4):
        c.setPositions(start_positions(a.system, rep, a.state))
        c.applyConstraints(TOL)
        x0 = positions(c)
        for trial in range(5):
            ig.setGlobalVariableByName("draw", 1.0)
            ig.step(1)
            dH = ig.getGlobalVariableByName("Enew") - ig.getGlobalVariableByName("Eold")
            V1 = ig.getGlobalVariableByName("Vnew")
            v = c.getState(getVelocities=True).getVelocities(asNumpy=True)
            c.setVelocities(-v)
            ig.setGlobalVariableByName("draw", 0.0)
            ig.step(1)
            dH2 = ig.getGlobalVariableByName("Enew") - ig.getGlobalVariableByName("Eold")
            x2 = positions(c)
            err = np.abs(x2 - x0).max()
            errs.append(err)
            dHs.append(dH)
            print(f"  start rep{rep} trial {trial}: dH fwd {dH:+.4f}  back {dH2:+.4f} (sum {dH + dH2:+.2e})  "
                  f"V {V1:.1f}  max|x_back - x0| {err:.2e} nm", flush=True)
            c.setPositions(x0)
    print(f"L {a.L} x {a.dt_fs} fs: max |x_back - x0| {max(errs):.2e} nm; dH mean {np.mean(dHs):+.3f}, "
          f"sd {np.std(dHs):.3f} kJ/mol; <exp(-dH/kT)> {np.mean(np.exp(-np.array(dHs) / KT)):.3f}")


# ---------------------------------------------------------------- step 2b

def analytic(a):
    """Proposal on V, target V + f with f = k (d - d0)^2 on a ligand 1-3 distance (fast CV).
    Reference: Langevin on V + f. Also eta = 0 vs Langevin on V (no f)."""
    T = gm.temperature(a.system)
    KT = gm.thermal_energy(T)
    leg = local_E.Leg(a.system)
    _, system0, groups, _, box = gm.build(a.system)
    K = len(groups)
    hb = [f for f in system0.getForces() if isinstance(f, openmm.HarmonicAngleForce)][0]
    lig = set(leg.lig)
    heavy = {i for i in leg.lig if system0.getParticleMass(i).value_in_unit(unit.dalton) > 9.5}
    angle = None
    for k in range(hb.getNumAngles()):
        i, j_, l_, _, _ = hb.getAngleParameters(k)
        if {i, j_, l_} <= heavy:
            angle = (i, l_)
            break
    i1, i2 = angle
    ref_xyz = np.load(local_E.GW / "runs" / a.system / f"rep1_S{a.state}.npz")["xyz"].astype(np.float64)
    d_ref = np.linalg.norm(ref_xyz[:, i1] - ref_xyz[:, i2], axis=-1)
    mu, s = d_ref.mean(), d_ref.std()
    d0, kf = mu - 2 * s, KT / (2 * s * s)
    print(f"CV: ligand 1-3 distance atoms {i1}-{i2}; Langevin S{a.state} rep1 frames: mean {mu:.5f} sd {s:.5f} nm; "
          f"f = {kf:.1f} (d - {d0:.5f})^2 kJ/mol")

    def f_of(x):
        return kf * (np.linalg.norm(x[i1] - x[i2]) - d0) ** 2

    def d_of(x):
        return np.linalg.norm(x[i1] - x[i2])

    # reference Langevin on V + f
    bond = openmm.CustomBondForce(f"{kf}*(r-{d0})^2")
    bond.addBond(i1, i2, [])
    sysf, _, _ = mm_system(a.system, extra=bond)
    lg = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picosecond)
    lg.setRandomNumberSeed(11)
    c = make_context(sysf, box, lg)
    gm.set_state(c, a.state, K)
    c.setPositions(start_positions(a.system, 1, a.state))
    c.setVelocitiesToTemperature(T * unit.kelvin, 11)
    lg.step(5000)
    dl = []
    for _ in range(a.lang_samples):
        lg.step(50)
        dl.append(d_of(positions(c)))
    dl = np.array(dl)
    del c
    # HMC: proposal V only
    system, _, _ = mm_system(a.system)
    ig = proposal_integrator(a.L, a.dt_fs, T)
    c = make_context(system, box, ig)
    gm.set_state(c, a.state, K)
    rng = np.random.default_rng(7)
    rows = {}
    for eta, ref in ((1.0, dl), (0.0, d_ref)):
        t = time.perf_counter()
        o = hmc(c, ig, start_positions(a.system, 1, a.state), eta, a.cycles, f_of, rng, record=d_of)
        dh = o["rec"][a.cycles // 10:]
        m_h, se_h = mean_se(dh)
        m_r, se_r = mean_se(ref)
        rows[eta] = {"acc": float(o["acc"].mean()), "hmc": [m_h, se_h, float(dh.std())], "ref": [m_r, se_r, float(ref.std())]}
        print(f"eta {eta}: acc {o['acc'].mean():.3f}; d HMC {m_h:.5f} +- {se_h:.5f} (sd {dh.std():.5f}) vs "
              f"{'Langevin V+f' if eta else 'Langevin V (rep1 frames)'} {m_r:.5f} +- {se_r:.5f} (sd {ref.std():.5f}); "
              f"z {(m_h - m_r) / np.hypot(se_h, se_r):+.2f}; {time.perf_counter() - t:.0f} s", flush=True)
    json.dump(rows, open(HERE / "runs" / f"analytic_{a.system}_S{a.state}.json", "w"), indent=1)


# ---------------------------------------------------------------- step 3

def chain(a):
    from eta_records import run_recorded
    if not 0 <= a.eta <= 1:
        raise ValueError("eta must be in [0,1]")
    if a.dtype != "float64":
        raise ValueError("model MC acceptance requires the validated float64 energy query")
    leg = local_E.Leg(a.system)
    system, groups, box = mm_system(a.system)
    K = len(groups)
    ig = proposal_integrator(a.L, a.dt_fs, leg.temperature_K)
    seed = 1000 * a.start + int(round(a.eta * 1000))
    ig.setRandomNumberSeed(seed + 1)
    c = make_context(system, box, ig)
    gm.set_state(c, a.state, K)
    me = ModelEnergy(leg, a.state, a.dtype, ckpt=a.head, backbone=a.backbone)
    rng = np.random.default_rng(seed)
    out = HERE / "runs" / a.system / f"chain_S{a.state}_eta{a.eta:.3f}_start{a.start}{a.tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    ident = model_identity(ckpt=a.head, backbone=a.backbone)
    want = {"system": a.system, "state": a.state, "eta": a.eta, "start_rep": a.start, "L": a.L, "dt_fs": a.dt_fs,
            "dtype": a.dtype, "T_K": leg.temperature_K, "E_construction": "local_E v1", **ident,
            "leg": "solvent", "n_water": 50, "n_particles": system.getNumParticles(),
            "ligand_system_indices": list(map(int, leg.lig)), "groups": [list(map(int, g)) for g in leg.groups],
            "dummy_mask": leg.dummy(a.state).tolist(), "box_nm": np.asarray(box, float).tolist(),
            "system_xml": str(gm.leg(a.system)[0]), "softcore_alpha": gm.ALPHA,
            "system_definition": openmm.XmlSerializer.serialize(system),
            "context_parameters": dict(c.getParameters()),
            "integrator_definition": openmm.XmlSerializer.serialize(ig),
            "openmm_version": openmm.__version__,
            "acceptance_seed": seed, "proposal_seed": seed + 1, "energy_units": "kJ/mol", "xyz_units": "nm"}
    meta = run_recorded(c, ig, me, rng, start_positions(a.system, a.start, a.state), a.eta, a.cycles,
                        out, want, hmc, checkpoint_every=a.checkpoint_every,
                        coords_every=a.coords_every, log_every=a.log_every)
    print(f"{out.name}: complete={meta['complete']}; {meta['completed_cycles']} cycles; "
          f"full-box snapshots every {a.coords_every} cycles, atomic energies every cycle", flush=True)


TARGET = ("system", "state", "T_K", "E_construction", "dtype", "backbone", "head",
          "system_definition", "context_parameters", "ligand_system_indices", "dummy_mask", "box_nm")


def analyze(a):
    """Per (eta, start): acceptance, <dU>, g(dU) in model calls, ligand displacement per model call;
    across starts; adjacent-eta BAR (reduced work = d_eta * dU / kT) and trapezoid TI for a_j.
    Chains are split by TARGET (older chains have no backbone / head recorded: None). Coverage is checked
    against --expect-etas / --expect-starts (default: the eta x start grid found): a sum that misses eta 0 or 1,
    or any expected (eta, start), is printed and saved as incomplete, not as a_j."""
    import re
    pat = re.compile(rf"chain_S{a.state}_eta\d+\.\d+_start\d+{re.escape(a.tag)}\.npz")
    files = sorted(f for f in (HERE / "runs" / a.system).glob(f"chain_S{a.state}_eta*_start*{a.tag}.npz")
                   if pat.fullmatch(f.name))
    groups = {}
    for f in files:
        m = json.loads(str(np.load(f)["meta"]))
        groups.setdefault(json.dumps({k: m.get(k) for k in TARGET}, sort_keys=True), []).append(f)
    if len(groups) > 1:
        print(f"{len(groups)} incompatible targets among {len(files)} chains; analysed separately")
    out = []
    for key, fs in groups.items():
        target = json.loads(key)
        print("target: " + ", ".join(f"{k}={target[k]}" for k in TARGET[:7]))
        out.append({"target": target, "files": [f.name for f in fs], **analyze_target(a, fs)})
    json.dump(out, open(HERE / "runs" / a.system / f"analysis_S{a.state}{a.tag}.json", "w"), indent=1)


def analyze_target(a, files):
    from pymbar import other_estimators
    rows = {}
    KT = None
    for f in files:
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        rt = gm.thermal_energy(m["T_K"])
        if KT is not None and rt != KT:
            raise ValueError("cannot combine chains at different temperatures")
        KT = rt
        n0 = int(len(z["dU"]) * a.discard)
        dU, acc, X = z["dU"][n0:], z["acc"][n0:], z["lig_xyz"][n0:].astype(np.float64)
        disp = np.sqrt(((X[1:] - X[:-1]) ** 2).sum(-1).mean(-1))                 # ligand RMS move per call, nm
        mu, se = mean_se(dU)
        r = {"eta": m["eta"], "start": m["start_rep"], "n": len(dU), "acc": float(acc.mean()), "mean_dU": mu, "se": se,
             "sd_dU": float(dU.std()), "g_calls": stat_ineff(dU), "lig_rms_move_nm_per_call": float(disp.mean()),
             "ms_per_cycle": 1e3 * m["seconds"] / m["cycles"], "dU": dU}
        rows.setdefault(m["eta"], []).append(r)
    etas = sorted(rows)
    print(f"{a.system} S{a.state}: first {a.discard:.0%} of every chain discarded (kJ/mol, kT {KT:.4f})")
    print("  eta  start    n   acc   <dU> +- SE        sd    g(calls)  lig move/call (nm)")
    summary = []
    for e in etas:
        for r in sorted(rows[e], key=lambda r: r["start"]):
            print(f"  {e:.2f}  {r['start']:>4d} {r['n']:5d}  {r['acc']:.2f}  {r['mean_dU']:+8.2f} +- {r['se']:5.2f}  "
                  f"{r['sd_dU']:6.2f}  {r['g_calls']:7.1f}   {r['lig_rms_move_nm_per_call']:.4f}")
        mus = np.array([r["mean_dU"] for r in rows[e]])
        ses = np.array([r["se"] for r in rows[e]])
        pooled = np.concatenate([r["dU"] for r in rows[e]])
        summary.append({"eta": e, "n_starts": len(mus), "start_means": mus.tolist(), "start_se": ses.tolist(),
                        "spread_of_start_means": float(mus.max() - mus.min()) if len(mus) > 1 else 0.0,
                        "pooled_mean": float(pooled.mean())})
        if len(mus) > 1:
            print(f"        across starts: means {np.round(mus, 2).tolist()}, spread {mus.max() - mus.min():.2f}, "
                  f"typical SE {np.median(ses):.2f}")
    pairs = []
    for e0, e1 in zip(etas[:-1], etas[1:]):
        x0 = np.concatenate([r["dU"] for r in rows[e0]])
        x1 = np.concatenate([r["dU"] for r in rows[e1]])
        g0 = max(np.median([r["g_calls"] for r in rows[e0]]), 1.0)
        g1 = max(np.median([r["g_calls"] for r in rows[e1]]), 1.0)
        wF, wR = (e1 - e0) * x0[::int(np.ceil(g0))] / KT, -(e1 - e0) * x1[::int(np.ceil(g1))] / KT
        b = other_estimators.bar(wF, wR)
        pairs.append({"eta0": e0, "eta1": e1, "df_kT": float(b["Delta_f"]), "ddf_kT": float(b["dDelta_f"]),
                      "sd_wF_kT": float(wF.std()), "sd_wR_kT": float(wR.std()), "nF": len(wF), "nR": len(wR)})
        print(f"  BAR {e0:.2f}->{e1:.2f}: {b['Delta_f'] * KT:+8.2f} +- {b['dDelta_f'] * KT:.2f} kJ/mol  "
              f"sd(w) {wF.std():.2f}/{wR.std():.2f} kT  n {len(wF)}/{len(wR)}")
    found = {(round(e, 6), r["start"]) for e in etas for r in rows[e]}
    exp_etas = [float(x) for x in a.expect_etas.split(",")] if a.expect_etas else etas
    exp_starts = ([int(x) for x in a.expect_starts.split(",")] if a.expect_starts
                  else sorted({s for _, s in found}))
    missing = [(e, s) for e in exp_etas for s in exp_starts if (round(e, 6), s) not in found]
    full_range = bool(etas) and np.isclose(etas[0], 0.0) and np.isclose(etas[-1], 1.0)
    complete = full_range and not missing
    if missing:
        print(f"  INCOMPLETE: {len(missing)} of {len(exp_etas) * len(exp_starts)} expected (eta, start) missing: "
              + ", ".join(f"({e:.2f}, {s})" for e, s in missing))
    res = {"complete": bool(complete), "eta_range": [etas[0], etas[-1]] if etas else None,
           "missing": missing, "per_eta": summary}
    if pairs:
        tot = KT * sum(p["df_kT"] for p in pairs)
        means = [np.mean(np.concatenate([r["dU"] for r in rows[e]])) for e in etas]
        ti = float(np.trapezoid(means, etas))
        if not full_range:
            label = f"eta {etas[0]:.2f}->{etas[-1]:.2f} only (not a_j: eta 0 and 1 not both present)"
        elif missing:
            label = "a_j from an INCOMPLETE set of chains"
        else:
            label = "a_j"
        print(f"  {label}: BAR sum {tot:+.2f} kJ/mol;  trapezoid TI {ti:+.2f} kJ/mol over eta {etas}")
        res.update({"label": label, "BAR_sum": tot, "TI": ti, "pairs": pairs})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["repeat", "reverse", "analytic", "chain", "analyze"])
    ap.add_argument("system")
    ap.add_argument("--state", type=int, default=0)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--dt-fs", type=float, default=2.0)
    ap.add_argument("--cycles", type=int, default=2000)
    ap.add_argument("--lang-samples", type=int, default=2000)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--tag", default="")
    ap.add_argument("--head", default=None, help="explicit existing head checkpoint; preserve old models and use a new tag")
    ap.add_argument("--backbone", default=None, help="explicit existing backbone file")
    ap.add_argument("--checkpoint-every", type=int, default=50, help="commit a restartable chunk every N cycles")
    ap.add_argument("--coords-every", type=int, default=10, help="save full-box accepted-state snapshots every N cycles")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--discard", type=float, default=0.2)
    ap.add_argument("--expect-etas", default="")           # analyze: comma list, e.g. 0.0,0.1,...,1.0
    ap.add_argument("--expect-starts", default="")         # analyze: comma list, e.g. 1,2,3,4
    a = ap.parse_args()
    (HERE / "runs").mkdir(exist_ok=True)
    {"repeat": repeat, "reverse": reverse, "analytic": analytic, "chain": chain, "analyze": analyze}[a.cmd](a)
