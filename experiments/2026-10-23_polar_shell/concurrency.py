"""Per-process ML/MM step time when N copies share one GPU (no replica batching).

Each copy builds the same POLAR-shell system as profile_mlmm.py (production defaults: FP32,
fast forces, fragment batching), waits at a file barrier until all N copies are built, then
times the same q states in the same order. Output: <out-dir>/rank<i>.json. No free energy.
"""

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--qs", default="0,2,3.5,5")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import openmm
    import torch
    from openmm import unit
    from decomp import mlmm
    from run_abfe import load_model

    torch.set_num_threads(args.threads)
    leg = ROOT / "experiments/2026-10-15_acetamide_leg/em_acetamide.npz"
    ckpt = ROOT / "runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
    model = load_model(str(ckpt), "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", device="cuda")
    start = np.asarray(np.load(leg, allow_pickle=True)["positions"], dtype=np.float64)
    system, force, info = mlmm.build_mlmm_system(str(leg), model, n_inner=20, device="cuda", positions=start,
                                                 shell_mm=False, polar_shell=True,
                                                 fast_forces=True, batch_fragments=True)
    integrator = openmm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    integrator.setRandomNumberSeed(7 + args.rank)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"))
    box = np.asarray(info["box"], dtype=np.float64)
    context.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[a] * box[a] / 10)) for a in range(3)])
    mlmm.set_path(context, force, 0.0, info["lig"], shell_mm=False, polar_shell=True)
    context.setPositions(start * unit.angstrom)
    integrator.step(2)                                   # CUDA/JIT warm-up before the barrier
    torch.cuda.synchronize()

    (args.out_dir / f"ready{args.rank}").touch()
    while len(list(args.out_dir.glob("ready*"))) < args.nproc:
        time.sleep(0.2)
    began = time.time()
    cases = []
    for q in [float(v) for v in args.qs.split(",")]:
        mlmm.set_path(context, force, q, info["lig"], shell_mm=False, polar_shell=True)
        context.setPositions(start * unit.angstrom)
        context.setVelocitiesToTemperature(298.15 * unit.kelvin, 7 + args.rank)
        integrator.step(args.warmup)
        samples = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            t = time.perf_counter()
            integrator.step(args.steps)
            context.getState(getPositions=True)
            torch.cuda.synchronize()
            samples.append(1000 * (time.perf_counter() - t) / args.steps)
        cases.append(dict(q=q, ms_per_step=statistics.median(samples), samples_ms=samples,
                          ended_s=time.time() - began))
        print(f"rank {args.rank}/{args.nproc} q={q:g}: {statistics.median(samples):.1f} ms/step", flush=True)
    result = dict(rank=args.rank, nproc=args.nproc, threads=args.threads, began_unix=began,
                  ended_unix=time.time(), gpu=torch.cuda.get_device_name(),
                  peak_mem_mib=torch.cuda.max_memory_allocated() / 2**20, cases=cases)
    (args.out_dir / f"rank{args.rank}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
