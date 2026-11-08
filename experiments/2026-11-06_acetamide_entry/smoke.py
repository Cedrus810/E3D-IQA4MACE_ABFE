"""Fixed engineering smoke: 5 tiny MM windows and 3 four-cycle eta chains.

CPU/Reference; at most 74 molecular integration steps, 17 target calls, 3 reference forwards.
Outputs are engineering records, never a converged free-energy result.
"""

import contextlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import openmm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "experiments/2026-10-25_eta_hmc"),
               str(ROOT / "experiments/2026-10-26_mm_bridge")]
import eta_hmc as eh
import bridge_mm as bm
import atom_deriv as ad
from eta_records import read_record, run_recorded
from atomic_report import report


def write(path, obj):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def ctx(system, box, ig):
    return eh.make_context(system, box, ig, platform="Reference")


def ef(c):
    s = c.getState(getEnergy=True, getForces=True)
    return s.getPotentialEnergy().value_in_unit(eh.KJ), s.getForces(asNumpy=True).value_in_unit(eh.KJ / eh.unit.nanometer)


def mm_checks():
    ref, grouped, groups, x, box = eh.gm.build("acetamide")
    _, bridge, _, _, _ = bm.build("acetamide")
    cr, cg, cb = [ctx(s, box, openmm.VerletIntegrator(.001)) for s in (ref, grouped, bridge)]
    for c in (cr, cg, cb):
        c.setPositions(x)
    e_ref, f_ref = ef(cr)
    e_group, f_group = ef(cg)
    checks = [{"name": "S0_original_vs_grouped", "dE": e_group - e_ref,
               "max_abs_dF": float(np.max(np.abs(f_group - f_ref)))}]
    for state in range(len(groups) + 1):
        eh.gm.set_state(cg, state, len(groups))
        bm.set_params(cb, bm.endpoint(len(groups), state))
        eg, fg = ef(cg)
        eb, fb = ef(cb)
        checks.append({"name": f"S{state}_grouped_vs_bridge", "dE": eb - eg,
                       "max_abs_dF": float(np.max(np.abs(fb - fg)))})
    del cr, cg, cb
    # Original NonbondedForce and the split pair/exception implementation have
    # different analytical dispersion tails. Isolate that fixed-box energy
    # baseline without changing either production System.
    no_tail = []
    for system in (ref, grouped):
        clone = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
        for force in clone.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setUseDispersionCorrection(False)
            elif isinstance(force, openmm.CustomNonbondedForce):
                force.setUseLongRangeCorrection(False)
        c = ctx(clone, box, openmm.VerletIntegrator(.001))
        c.setPositions(x)
        no_tail.append(ef(c))
        del c
    checks.append({"name": "S0_original_vs_grouped_without_tail", "dE": no_tail[1][0] - no_tail[0][0],
                   "max_abs_dF": float(np.max(np.abs(no_tail[1][1] - no_tail[0][1])))})
    for g in range(len(groups)):
        sa, _ = ad.build_atoms("acetamide", g)
        sr, _ = ad.build_ref("acetamide", g)
        ca, cr = [ctx(s, box, openmm.VerletIntegrator(.001)) for s in (sa, sr)]
        ca.setPositions(x)
        cr.setPositions(x)
        for el, on in ((.5, 1.), (0., .5)):
            p = bm.params(len(groups), g, el, on)
            ad.set_params(ca, ad.atom_params(p, g, groups[g]))
            ad.set_params(cr, p)
            item = {"group": g, "el": el, "on": on, "dE_sync": ad.energy(ca) - ad.energy(cr)}
            if on == 1:
                item["derivative_sum_difference"] = float(ad.d_el(ca, [f"el_{g}_a{i}" for i in groups[g]], el).sum()
                                                           - ad.d_el(cr, [f"el_{g}"], el)[0])
            else:
                item["derivative_sum_difference"] = float(ad.d_on(ca, [f"on_{g}_a{i}" for i in groups[g]]).sum()
                    - cr.getState(getParameterDerivatives=True, groups={2}).getEnergyParameterDerivatives()[f"on_{g}"])
            checks.append({"name": "atomic_sync", **item})
        del ca, cr
    # Algebraically equivalent double-precision constructions; this is an implementation tolerance.
    for check in checks:
        quantities = ("max_abs_dF",) if check["name"] == "S0_original_vs_grouped" else ("dE", "max_abs_dF", "dE_sync", "derivative_sum_difference")
        for k in quantities:
            if k in check and (not np.isfinite(check[k]) or abs(check[k]) > 1e-6):
                raise ValueError(f"MM equivalence failed: {check}")
    return checks


def bridge_smoke():
    args = SimpleNamespace(system="acetamide", rep=1, tag="_entry_smoke", groups="1,2", elec="1,0", vdw="1,0",
                           equil_ps=0., prod_ps=.02, frame_ps=.002)
    with mock.patch.object(bm, "HERE", HERE / "mm"), mock.patch.object(bm, "context", side_effect=ctx):
        bm.run(args)
        d = bm.run_dir(args)
        files = sorted(d.glob("w*.npz"))
        if len(files) != 5:
            raise ValueError("expected exactly five bridge windows")
        times = []
        for path in files:
            with np.load(path, allow_pickle=False) as z:
                m = json.loads(str(z["meta"]))
                if m["T_K"] != 298.15 or m["kT_kJmol"] != eh.gm.thermal_energy(298.15) or z["xyz"].shape[0] != 10:
                    raise ValueError("bridge temperature or frame count mismatch")
                times.append(z["times_ps"].copy())
        bm.run(args)  # completed windows are reused, without extra MD
        for path, ts in zip(files, times):
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z["times_ps"], ts)
        config_before = (d / "config.json").read_text()
        with mock.patch.object(bm.gm, "temperature", return_value=300.):
            try:
                bm.run(args)
            except ValueError as error:
                if "T_K" not in str(error):
                    raise
            else:
                raise ValueError("wrong-temperature reuse was accepted")
        if (d / "config.json").read_text() != config_before:
            raise ValueError("wrong-temperature attempt changed the source config")
        bm.analyze(args)
    return {"path": str(d), "windows": 5, "integration_steps": 50, "completed_reuse_passed": True,
            "wrong_temperature_rejected_without_overwrite": True,
            "scope": "engineering only; two-point charge/LJ schedule and ten frames per window"}


def model_smoke():
    from decomp.clusters import coupling_labels
    panel_path = ROOT / "experiments/2026-10-25_eta_hmc/logs/deep_shallow_panel_20261031.npz"
    with np.load(panel_path, allow_pickle=False) as z:
        ident = json.loads(str(z["meta"]))["contract"]
    leg = eh.local_E.Leg("acetamide")
    _, _, _, x, box = eh.gm.build("acetamide")
    energy = eh.ModelEnergy(leg, 0, ckpt=ident["head"], backbone=ident["backbone"], device="cpu")
    d0 = energy(x)
    atom0 = energy.last_components["dU_atom"].copy()
    repeated = energy(x)
    if abs(d0 - repeated) > 1e-7 or np.max(np.abs(atom0 - energy.last_components["dU_atom"])) > 1e-7:
        raise ValueError("fixed-frame model repeatability failed")
    p, n, f, si = leg.carve(x, 50)
    cbb = float(coupling_labels(energy.model.backbone, [p], [n], [f], device="cpu", batch=1)[0] * eh.EV)
    cml, cmm = float(energy.last_components["C_ML"].sum()), float(energy.last_components["C_MM"].sum())
    if not np.isfinite(cbb) or abs(d0 - (cml - cbb) - (cbb - cmm)) > 1e-8:
        raise ValueError("head/backbone/MM decomposition failed")
    if np.any(leg.c_mm(p, si, leg.dummy(leg.K))):
        raise ValueError("full-dummy local MM coupling is not zero")
    files = []
    for eta in (0., .5, 1.):
        system, groups, _ = eh.mm_system("acetamide")
        ig = eh.proposal_integrator(2, 2., leg.temperature_K)
        ig.setRandomNumberSeed(8100 + int(100 * eta))
        c = ctx(system, box, ig)
        eh.gm.set_state(c, 0, len(groups))
        contract = {"system": "acetamide", "state": 0, "leg": "solvent", "T_K": leg.temperature_K,
                    "eta": eta, "start_rep": 1, "L": 2, "dt_fs": 2., "dtype": "float64",
                    "E_construction": "local_E v1", "head": ident["head"], "backbone": ident["backbone"],
                    "system_definition": openmm.XmlSerializer.serialize(system),
                    "context_parameters": dict(c.getParameters()), "box_nm": np.asarray(box).tolist(),
                    "n_particles": system.getNumParticles(), "n_water": 50,
                    "ligand_system_indices": list(leg.lig), "groups": leg.groups, "dummy_mask": leg.dummy(0).tolist(),
                    "system_xml": str(eh.gm.leg("acetamide")[0]), "softcore_alpha": eh.gm.ALPHA,
                    "energy_units": "kJ/mol", "xyz_units": "nm", "scope": "engineering smoke, shared starting structure"}
        out = HERE / f"chain_eta{eta:.1f}_smoke.npz"
        meta = run_recorded(c, ig, energy, np.random.default_rng(9100 + int(100 * eta)), x, eta, 4, out,
                            contract, eh.hmc, checkpoint_every=2, coords_every=1)
        if not meta["complete"]:
            raise ValueError("incomplete engineering chain")
        _, data = read_record(out)
        if data["dU_atom"].shape != (4, 9) or data["xyz"].shape != (4, 2634, 3):
            raise ValueError("small-system atomic or full-box records have wrong shape")
        files.append(out)
        del c, ig
    if energy.calls != 17:
        raise ValueError(f"target query budget mismatch: {energy.calls}")
    atomic = report(files, [0., .5, 1.], [1], discard=0., bootstrap=0, block_length=1)
    atomic["scope"] = "engineering smoke; complete eta grid is not convergence or scientific acceptance"
    write(HERE / "atomic_engineering_report.json", atomic)
    return {"target_queries": energy.calls, "reference_forwards": 3, "integration_steps": 24,
            "fixed_frame_dU": d0, "fixed_frame_C_bb": cbb, "head_minus_backbone": cml - cbb,
            "backbone_minus_MM": cbb - cmm, "model_repeat_difference": repeated - d0,
            "atomic_records_and_report_passed": True}


def main():
    out = HERE / "smoke.json"
    if out.exists() or (HERE / "mm").exists() or any(HERE.glob("chain*_smoke*.npz")):
        raise FileExistsError("preserving existing engineering outputs; inspect before another run")
    started = time.perf_counter()
    result = {"complete": False, "scope": "bounded engineering validation only", "temperature_K": 298.15,
              "platform": "Reference", "model_device": "cpu", "max_MD_steps": 74,
              "max_target_queries": 17, "max_reference_forwards": 3}
    write(out, result)
    try:
        result["MM_checks"] = mm_checks()
        print("MM endpoint and atomic derivative checks passed", flush=True)
        write(out, result)
        result["bridge"] = bridge_smoke()
        write(out, result)
        result["model"] = model_smoke()
        result["complete"] = True
        result["seconds"] = time.perf_counter() - started
        write(out, result)
        print(json.dumps({"complete": True, "MD_steps": 74, "target_queries": 17, "seconds": result["seconds"]}), flush=True)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["seconds"] = time.perf_counter() - started
        write(out, result)
        raise


if __name__ == "__main__":
    main()
