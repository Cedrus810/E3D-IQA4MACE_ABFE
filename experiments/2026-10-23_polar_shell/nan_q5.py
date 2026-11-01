"""Smoke: at which step does POLAR-shell q=5 go NaN from the leg start frame, per force variant?

Same start / seed as concurrency.py rank 0. Steps in chunks of 5; reports the first failing chunk and
the max |F| of the ML force at the last good frame. Diagnostic only.
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--variants", default="fast_batch_fp32,reference_fp32,fast_batch_fp64")
    args = parser.parse_args()
    import numpy as np
    import openmm
    import torch
    from openmm import unit
    from decomp import mlmm
    from run_abfe import load_model

    leg = ROOT / "experiments/2026-10-15_acetamide_leg/em_acetamide.npz"
    ckpt = ROOT / "runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
    start = np.asarray(np.load(leg, allow_pickle=True)["positions"], dtype=np.float64)
    for variant in args.variants.split(","):
        fast = variant.startswith("fast")
        dtype = torch.float64 if variant.endswith("fp64") else torch.float32
        model = load_model(str(ckpt), "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", device="cuda").to(dtype)
        system, force, info = mlmm.build_mlmm_system(str(leg), model, n_inner=20, device="cuda", positions=start,
                                                     shell_mm=False, polar_shell=True,
                                                     fast_forces=fast, batch_fragments=fast)
        integrator = openmm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
        integrator.setRandomNumberSeed(7)
        context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"))
        box = np.asarray(info["box"], dtype=np.float64)
        context.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[a] * box[a] / 10)) for a in range(3)])
        mlmm.set_path(context, force, args.q, info["lig"], shell_mm=False, polar_shell=True)
        context.setPositions(start * unit.angstrom)
        context.setVelocitiesToTemperature(298.15 * unit.kelvin, 7)
        done, last = 0, None
        try:
            while done < args.steps:
                integrator.step(5)
                done += 5
                state = context.getState(getPositions=True, getEnergy=True)
                last = (np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)),
                        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            status = f"OK through {done} steps, U {last[1]:.1f} kJ/mol"
        except openmm.OpenMMException as exc:
            status = f"FAILED in steps {done + 1}-{done + 5}: {exc}"
        detail = ""
        if last is not None:
            U, F = force.evaluate(last[0], box)
            lig = np.asarray(info["lig"])
            per = np.linalg.norm(F, axis=1)
            detail = (f"; last good frame: ML+FIRES U {U:.1f}, max|F| {per.max():.1f} kJ/mol/A "
                      f"(ligand max {per[lig].max():.1f})")
        print(f"{variant} q={args.q:g}: {status}{detail}", flush=True)
        del context, integrator, model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
