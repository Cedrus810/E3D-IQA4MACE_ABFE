"""Local coupling replacement at the report endpoints (DESIGN_2026-10-23_groupwise_dummy.md §7.1, §9 priority 1-2).

Run: python local_E.py check SYSTEM [--frames 4]                        (identity checks, few frames)
     python local_E.py eval SYSTEM [--n-water 50] [--batch 16] [--max-frames N]   (all endpoints; resumable)
     python local_E.py analyze SYSTEM                                     (delta U statistics, a_j)

At endpoint j (first j groups dummy) and a frame x of that endpoint's own MM production:
    E_j(x)  = V_j(x) - C^MM_j,Omega(x) + C^ML_j,Omega(x)
    dU_j(x) = C^ML_j,Omega(x) - C^MM_j,Omega(x)
    a_j     = -RT log < exp(-dU_j / RT) >_{V_j}
Omega(x): the N water molecules nearest to any ligand atom (all ligand atoms, whatever the state),
whole molecules, imaged next to the ligand (openmm_bridge.carve_from_arrays); ions are not in Omega.
The rule does not depend on j. Replaced pair set: (ligand atoms coupled at j) x (atoms of Omega).
  C^ML: step_eval.couplings, i.e. head (4 channels) + explicit Coulomb per coupled ligand atom on the
        non-periodic cluster ligand + Omega, dummy-environment edges cut.
  C^MM: on the same carved geometry and the same pairs,
        Coulomb  ONE_4PI_EPS0 q_i q_a / r          (bare, every pair; no PME split)
        LJ       4 eps ((sig/r)^12 - (sig/r)^6), r < r_c, no switch (the form V uses at on_g = 1)
V keeps everything else, including PME minus the bare Coulomb of these pairs, the LJ tail correction,
ligand-ion terms, ligand internal terms and water-water terms: the same rule and the same value in E_j
and V_j at every frame. At j = K no ligand atom is coupled, so C^ML = C^MM = 0 and a_K = 0.
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
GW = E3D / "experiments/2026-10-23_groupwise_dummy"
sys.path.insert(0, str(E3D))
sys.path.insert(0, str(GW))
import groupwise_mm as gm  # noqa: E402
from decomp.openmm_bridge import carve_from_arrays  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
KJ = unit.kilojoule_per_mole
RT = gm.KT                                            # kJ/mol at gm.T (300 K, IBS legs)


def one_4pi_eps0():
    """OpenMM's Coulomb constant (kJ/mol nm e^-2), read from a two-charge NoCutoff system."""
    s = openmm.System()
    s.addParticle(1.0)
    s.addParticle(1.0)
    nb = openmm.NonbondedForce()
    nb.addParticle(1.0, 1.0, 0.0)
    nb.addParticle(1.0, 1.0, 0.0)
    s.addForce(nb)
    c = openmm.Context(s, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    c.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(1, 0, 0)])
    return c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)


class Leg:
    def __init__(self, name):
        from model_eval import topology
        self.name = name
        xml, self.lig, self.groups, _, _ = gm.leg(name)
        self.top = topology(name)                     # lig, groups, box, numbers, resids, resnames, keep
        ref = openmm.XmlSerializer.deserialize(open(xml).read())
        nb = [f for f in ref.getForces() if isinstance(f, openmm.NonbondedForce)][0]
        p = [nb.getParticleParameters(i) for i in range(ref.getNumParticles())]
        self.q = np.array([x[0].value_in_unit(unit.elementary_charge) for x in p])
        self.sig = np.array([x[1].value_in_unit(unit.nanometer) for x in p])
        self.eps = np.array([x[2].value_in_unit(KJ) for x in p])
        self.rc = nb.getCutoffDistance().value_in_unit(unit.nanometer)
        self.nb_method = nb.getNonbondedMethod()
        self.K = len(self.groups)
        pos_in_lig = {x: i for i, x in enumerate(self.lig)}
        self.gidx = [[pos_in_lig[x] for x in g] for g in self.groups]
        self.k_e = one_4pi_eps0()

    def dummy(self, j):
        """[n_lig] bool: ligand atoms dummy at endpoint j (first j groups)."""
        d = np.zeros(len(self.lig), bool)
        for g in self.gidx[:j]:
            d[g] = True
        return d

    def carve(self, xyz_nm, n_water):
        """(pos A, numbers, frag, system indices) of ligand + Omega."""
        lig, _, box, numbers, resids, resnames, keep = self.top
        pos, num, frag, idx = carve_from_arrays(xyz_nm.astype(np.float64)[keep] * 10, numbers[keep],
                                                resids[keep], resnames[keep], n_solvent=n_water,
                                                box_ang=np.diag(box) * 10)
        sysidx = keep[idx]
        assert (sysidx[:len(lig)] == np.array(lig)).all()
        return pos, num, frag, sysidx

    def c_mm(self, pos_ang, sysidx, dummy):
        """[n_lig, 2] (Coulomb, LJ) kJ/mol per ligand atom over Omega; dummy rows 0."""
        n_lig = len(self.lig)
        L, W = sysidx[:n_lig], sysidx[n_lig:]
        r = np.linalg.norm(pos_ang[:n_lig, None] - pos_ang[None, n_lig:], axis=-1) / 10.0
        coul = self.k_e * self.q[L][:, None] * self.q[W][None, :] / r
        sig = 0.5 * (self.sig[L][:, None] + self.sig[W][None, :])
        eps = np.sqrt(self.eps[L][:, None] * self.eps[W][None, :])
        sr6 = (sig / r) ** 6
        lj = np.where(r < self.rc, 4 * eps * (sr6 * sr6 - sr6), 0.0)
        out = np.stack([coul.sum(1), lj.sum(1)], -1)
        out[dummy] = 0.0
        return out


def check(a):
    """Identity checks on a few frames per endpoint; nothing is written but the log."""
    import torch
    from model_eval import load_model
    from step_eval import couplings
    leg = Leg(a.system)
    n_lig = len(leg.lig)
    print(f"{a.system}: K {leg.K}, n_lig {n_lig}, cutoff {leg.rc} nm, NonbondedForce method {leg.nb_method}, "
          f"ONE_4PI_EPS0 {leg.k_e:.6f}")

    # 1. LJ formula, combination rule, cutoff and imaging vs OpenMM's own group force (whole box, LRC off)
    ref, system, groups, _, box = gm.build(a.system)
    lj_forces = [f for f in system.getForces() if isinstance(f, openmm.CustomNonbondedForce)]
    for g, f in enumerate(lj_forces):
        f.setForceGroup(g + 1)
        f.setUseLongRangeCorrection(False)
    c = gm.context(system, box, openmm.VerletIntegrator(0.001))
    boxd = np.diag(np.asarray(box))
    ligset = set(leg.lig)
    env = np.array([i for i in range(system.getNumParticles()) if i not in ligset])
    for j in (0, leg.K - 1, leg.K):
        xyz = np.load(GW / "runs" / a.system / f"rep1_S{j}.npz")["xyz"][0].astype(np.float64)
        gm.set_state(c, j, leg.K)
        c.setPositions(xyz)
        for g in range(leg.K):
            e_omm = c.getState(getEnergy=True, groups={g + 1}).getPotentialEnergy().value_in_unit(KJ)
            A = np.array(leg.groups[g])
            d = xyz[A][:, None] - xyz[env][None, :]
            d -= boxd * np.round(d / boxd)
            r = np.linalg.norm(d, axis=-1)
            sig = 0.5 * (leg.sig[A][:, None] + leg.sig[env][None, :])
            eps = np.sqrt(leg.eps[A][:, None] * leg.eps[env][None, :])
            sr6 = (sig / r) ** 6
            e_me = 0.0 if g < j else float(np.where(r < leg.rc, 4 * eps * (sr6 * sr6 - sr6), 0.0).sum())
            print(f"  [LJ] S{j} G{g + 1} ({'dummy' if g < j else 'on'}): OpenMM {e_omm:+.6f}  pairs {e_me:+.6f}  "
                  f"diff {e_omm - e_me:+.2e} kJ/mol")
    del c

    # 2. Coulomb of the replaced pairs vs an independent OpenMM NoCutoff cluster; carved vs minimum-image r
    model = load_model()
    for j in range(leg.K + 1):
        xyz = np.load(GW / "runs" / a.system / f"rep1_S{j}.npz")["xyz"][:a.frames]
        dm = leg.dummy(j)
        cl = [leg.carve(x, a.n_water) for x in xyz]
        mm = np.array([leg.c_mm(p, s, dm) for p, _, _, s in cl])            # [F, n_lig, 2]
        pos, _, _, sysidx = cl[0]
        s_ = openmm.System()
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        for k, i in enumerate(sysidx):
            s_.addParticle(1.0)
            nb.addParticle(0.0 if (k < n_lig and dm[k]) else leg.q[i], 1.0, 0.0)
        s_.addForce(nb)
        cc = openmm.Context(s_, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))

        def e_with(mask):
            for k, i in enumerate(sysidx):
                on = mask[k] and not (k < n_lig and dm[k])
                nb.setParticleParameters(k, leg.q[i] if on else 0.0, 1.0, 0.0)
            nb.updateParametersInContext(cc)
            cc.setPositions(pos / 10.0)
            return cc.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
        isL = np.arange(len(sysidx)) < n_lig
        e_int = e_with(np.ones(len(sysidx), bool)) - e_with(isL) - e_with(~isL)
        dmi = pos[:n_lig, None] / 10 - pos[None, n_lig:] / 10
        dw = xyz[0].astype(np.float64)[sysidx[:n_lig]][:, None] - xyz[0].astype(np.float64)[sysidx[n_lig:]][None]
        dw -= boxd * np.round(dw / boxd)
        dr = np.abs(np.linalg.norm(dmi, axis=-1) - np.linalg.norm(dw, axis=-1))
        near = np.linalg.norm(dw, axis=-1) < leg.rc
        print(f"  [Coul] S{j}: pair sum {mm[0, :, 0].sum():+.6f}  OpenMM NoCutoff {e_int:+.6f}  "
              f"diff {mm[0, :, 0].sum() - e_int:+.2e} kJ/mol; |r_carved - r_minimage| max within r_c "
              f"{dr[near].max():.2e} nm, all pairs {dr.max():.2e} nm")

        # 3. C^ML: new export vs the legacy steps_N50 file, batch F vs batch 1; dummy rows; S_K empty
        if dm.all():
            print(f"  [S{j}] all dummy: coupled atoms {int((~dm).sum())}; C^MM sum {mm[..., :].sum():+.1f} "
                  f"(C^ML not evaluated: no coupled atom) -> dU = 0, a_K = 0 by construction")
            continue
        clm = [(p, n, f) for p, n, f, _ in cl]
        ml = couplings(model, clm, dm) * 96.485332                     # eV -> kJ/mol
        ml1 = np.concatenate([couplings(model, [x], dm) for x in clm]) * 96.485332
        leg_file = GW / "runs" / a.system / f"steps_N{a.n_water}_S{j}.npz"
        old = np.load(leg_file)["D"][:a.frames] if leg_file.exists() else None
        msg = (f"  [ML] S{j}: dummy rows NaN {bool(np.isnan(ml[:, dm]).all())}, coupled finite "
               f"{bool(np.isfinite(ml[:, ~dm]).all())}; max |batch {a.frames} - batch 1| "
               f"{np.nanmax(np.abs(ml - ml1)):.2e}")
        if old is not None:
            msg += f"; max |new - legacy {leg_file.name}| {np.nanmax(np.abs(ml - old)):.2e} kJ/mol"
        print(msg)
        dU = np.nansum(ml, 1) - mm.sum((1, 2))
        print(f"  [dU] S{j}: C^ML {np.round(np.nansum(ml, 1), 2).tolist()}  C^MM "
              f"{np.round(mm.sum((1, 2)), 2).tolist()}  dU {np.round(dU, 2).tolist()} kJ/mol")
    print(f"peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def evaluate(a):
    import torch
    from model_eval import CKPT, BB, load_model
    from step_eval import couplings
    leg = Leg(a.system)
    n_lig = len(leg.lig)
    out_dir = HERE / "runs" / a.system
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"N{a.n_water}" + (f"_first{a.max_frames}" if a.max_frames else "")
    model = None
    for j in range(leg.K + 1):
        src = GW / "runs" / a.system / f"rep1_S{j}.npz"
        z = np.load(src)
        frames = np.arange(len(z["xyz"]))[:a.max_frames] if a.max_frames else np.arange(len(z["xyz"]))
        dm = leg.dummy(j)
        meta = {"system": a.system, "rep": 1, "endpoint": j, "K": leg.K, "n_water": a.n_water,
                "source": str(src), "frames": frames.tolist(), "dummy": dm.tolist(),
                "groups": [list(map(int, g)) for g in leg.groups], "T_K": gm.T, "backbone": BB, "head": CKPT,
                "E_construction": "local_E v1: V - C^MM(bare Coulomb + cutoff LJ, Omega pairs) + C^ML(head+ES)"}
        out = out_dir / f"endpoint_{tag}_S{j}.npz"
        if out.exists():                                  # resume: reuse only if the content matches
            old = json.loads(str(np.load(out)["meta"]))
            if old == meta:
                print(f"{out.name}: same endpoint, frames, mask and model; reused", flush=True)
                continue
            print(f"{out.name}: metadata differs, recomputing", flush=True)
        t = time.perf_counter()
        ml = np.zeros((len(frames), n_lig))
        mm = np.zeros((len(frames), n_lig, 2))
        waters = np.zeros((len(frames), a.n_water), np.int64)
        for b in range(0, len(frames), a.batch):
            cl = [leg.carve(z["xyz"][f], a.n_water) for f in frames[b:b + a.batch]]
            for k, (p, _, _, s) in enumerate(cl):
                mm[b + k] = leg.c_mm(p, s, dm)
                waters[b + k] = np.unique(leg.top[4][s[n_lig:]])  # residue indices of Omega
            if not dm.all():
                if model is None:
                    model = load_model()
                ml[b:b + len(cl)] = couplings(model, [(p, n, f) for p, n, f, _ in cl], dm) * 96.485332
        ml[:, dm] = 0.0
        V = z["u"][j][frames] * gm.KT
        dU = ml.sum(1) - mm.sum((1, 2))
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, meta=json.dumps(meta), frames=frames, V=V, C_ML=ml, C_MM=mm, dU=dU,
                 omega_water_resid=waters)
        tmp.replace(out)
        dt = time.perf_counter() - t
        print(f"{a.system} S{j}: {len(frames)} frames in {dt:.1f} s ({dt / len(frames) * 1e3:.0f} ms/frame); "
              f"<C^ML> {ml.sum(1).mean():+.2f}  <C^MM> {mm.sum((1, 2)).mean():+.2f}  <dU> {dU.mean():+.2f}  "
              f"sd(dU) {dU.std(ddof=1) if len(dU) > 1 else 0:.2f} kJ/mol -> {out.name}", flush=True)
    if model is not None:
        print(f"peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


def logmeanexp(x):
    m = x.max()
    return m + np.log(np.mean(np.exp(x - m)))


def analyze(a):
    from pymbar import timeseries
    leg = Leg(a.system)
    tag = f"N{a.n_water}"
    res = []
    print(f"{a.system}: endpoint correction a_j = -RT log <exp(-dU/RT)>_Vj, RT {RT:.4f} kJ/mol (kJ/mol)")
    for j in range(leg.K + 1):
        z = np.load(HERE / "runs" / a.system / f"endpoint_{tag}_S{j}.npz")
        dU = z["dU"]
        x = -dU / RT
        aj = -RT * logmeanexp(x)
        w = np.exp(x - x.max())
        w /= w.sum()
        g = timeseries.statistical_inefficiency(dU) if dU.std() > 0 else 1.0
        blocks = [-RT * logmeanexp(b) for b in np.array_split(x, 5)]
        q = np.percentile(dU, [0, 5, 50, 95, 100])
        r = {"endpoint": j, "n": len(dU), "mean_dU": float(dU.mean()), "sd_dU": float(dU.std(ddof=1)),
             "sd_dU_over_RT": float(dU.std(ddof=1) / RT), "pct_0_5_50_95_100": q.tolist(), "a_j": float(aj),
             "a_j_5_blocks": blocks, "max_weight": float(w.max()), "kish_n_eff": float(1 / (w ** 2).sum()),
             "stat_inefficiency": float(g), "mean_C_ML": float(z["C_ML"].sum(1).mean()),
             "mean_C_MM_coul": float(z["C_MM"][..., 0].sum(1).mean()),
             "mean_C_MM_lj": float(z["C_MM"][..., 1].sum(1).mean())}
        res.append(r)
        print(f"  S{j}: n {r['n']}  <dU> {r['mean_dU']:+8.2f}  sd {r['sd_dU']:6.2f} ({r['sd_dU_over_RT']:.2f} RT)  "
              f"a_j {aj:+8.2f}  blocks {' '.join(f'{b:+7.2f}' for b in blocks)}  "
              f"max w {r['max_weight']:.3f}  Kish n_eff {r['kish_n_eff']:.1f}  g {g:.1f}")
    json.dump(res, open(HERE / "runs" / a.system / f"analysis_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "eval", "analyze"])
    ap.add_argument("system")
    ap.add_argument("--n-water", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    {"check": check, "eval": evaluate, "analyze": analyze}[a.cmd](a)
