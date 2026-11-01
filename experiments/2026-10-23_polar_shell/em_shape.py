"""Smoke: POLAR-shell shape states (q 4.5, 5) from the leg start frame, energy-minimised first.

Per q: EM (OpenMM LocalEnergyMinimizer), then 200 steps checked for NaN every 10, then timing
(5 warm-up + 3 x 20 steps) as in concurrency.py. FP32, fast forces, fragment batching. Diagnostic only.
"""

import argparse
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def steepest_descent(context, max_iter, tol_kj_mol_nm, step_nm=0.002, max_step_nm=0.02):
    """Steepest descent on the full Context energy, step capped by the largest atomic displacement.
    OpenMM's LocalEnergyMinimizer falls back to the CPU platform at large forces, which rejects this
    system (non-identical exclusions across nonbonded forces)."""
    import numpy as np
    from openmm import unit
    state = context.getState(getPositions=True, getEnergy=True, getForces=True)
    x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    U = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    F = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    for _ in range(max_iter):
        fmax = np.linalg.norm(F, axis=1).max()
        if fmax < tol_kj_mol_nm:
            break
        context.setPositions(x + F * (step_nm / fmax))
        context.applyConstraints(1e-6)
        trial = context.getState(getPositions=True, getEnergy=True, getForces=True)
        U_new = trial.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if np.isfinite(U_new) and U_new < U:
            x = trial.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            U, F = U_new, trial.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
            step_nm = min(step_nm * 1.2, max_step_nm)
        else:
            step_nm *= 0.5
            if step_nm < 1e-7:
                break
    context.setPositions(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qs", default="4.5,5")
    parser.add_argument("--leg", default="experiments/2026-10-15_acetamide_leg/em_acetamide.npz")
    parser.add_argument("--T", type=float, default=298.15)
    parser.add_argument("--em-tol", type=float, default=10.0, help="kJ/mol/nm, OpenMM default")
    parser.add_argument("--em-iter", type=int, default=1000)
    parser.add_argument("--check-steps", type=int, default=200)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--precision", default="mixed", help="CUDA precision; single may make the minimiser fall back to CPU")
    args = parser.parse_args()
    import numpy as np
    import openmm
    import torch
    from openmm import unit
    from decomp import mlmm
    from run_abfe import load_model

    leg = ROOT / args.leg
    ckpt = ROOT / "runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
    start = np.asarray(np.load(leg, allow_pickle=True)["positions"], dtype=np.float64)
    model = load_model(str(ckpt), "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", device="cuda")
    system, force, info = mlmm.build_mlmm_system(str(leg), model, n_inner=20, device="cuda", positions=start,
                                                 shell_mm=False, polar_shell=True,
                                                 fast_forces=True, batch_fragments=True)
    integrator = openmm.LangevinMiddleIntegrator(args.T * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    integrator.setRandomNumberSeed(7)
    parser_precision = args.precision
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"),
                             {"Precision": parser_precision})
    box = np.asarray(info["box"], dtype=np.float64)
    context.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[a] * box[a] / 10)) for a in range(3)])
    lig = np.asarray(info["lig"])
    fires = force.fires

    def snapshot():
        state = context.getState(getPositions=True, getEnergy=True)
        P = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
        U, F = force.evaluate(P, box)
        return P, dict(U_total=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                       U_ml_fires=U, max_F=float(np.linalg.norm(F, axis=1).max()),
                       max_F_lig=float(np.linalg.norm(F[lig], axis=1).max()),
                       fires=dict(fires.last))

    results = []
    for q in [float(v) for v in args.qs.split(",")]:
        mlmm.set_path(context, force, q, info["lig"], shell_mm=False, polar_shell=True)
        context.setPositions(start * unit.angstrom)
        _, before = snapshot()
        t = time.perf_counter()
        mlmm.steepest_descent(context, args.em_iter, args.em_tol)
        em_s = time.perf_counter() - t
        P_em, after = snapshot()
        print(f"q={q:g} EM {em_s:.0f} s: U_total {before['U_total']:.1f} -> {after['U_total']:.1f}, "
              f"ML+FIRES {before['U_ml_fires']:.1f} -> {after['U_ml_fires']:.1f}, "
              f"max|F| {before['max_F']:.3g} -> {after['max_F']:.3g} kJ/mol/A, FIRES after {after['fires']}",
              flush=True)
        context.setVelocitiesToTemperature(args.T * unit.kelvin, 7)
        status, done = "ok", 0
        try:
            while done < args.check_steps:
                integrator.step(10)
                done += 10
                U = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                if not np.isfinite(U):
                    status = f"NaN energy by step {done}"
                    break
        except openmm.OpenMMException as exc:
            status = f"failed in steps {done + 1}-{done + 10}: {exc}"
        _, end = snapshot() if status == "ok" else (None, None)
        print(f"q={q:g} MD check: {status}" + (f"; after {done} steps max|F| {end['max_F']:.3g}, "
                                                f"FIRES {end['fires']}" if end else ""), flush=True)
        case = dict(q=q, em_s=em_s, before=before, after_em=after, md_check=status, after_md=end)
        if status == "ok":
            integrator.step(5)
            samples = []
            for _ in range(3):
                torch.cuda.synchronize()
                t = time.perf_counter()
                integrator.step(20)
                context.getState(getPositions=True)
                torch.cuda.synchronize()
                samples.append(1000 * (time.perf_counter() - t) / 20)
            case["ms_per_step"] = statistics.median(samples)
            case["samples_ms"] = samples
            force_samples = []
            P = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            for _ in range(3):
                torch.cuda.synchronize()
                t = time.perf_counter()
                force.evaluate(np.asarray(P), box)
                torch.cuda.synchronize()
                force_samples.append(1000 * (time.perf_counter() - t))
            case["force_ms"] = statistics.median(force_samples)
            print(f"q={q:g}: {case['ms_per_step']:.1f} ms/step, force call {case['force_ms']:.1f} ms", flush=True)
        results.append(case)
        np.save(args.out.with_name(f"em_shape_q{q:g}.npy"), P_em)
        args.out.write_text(json.dumps(results, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
