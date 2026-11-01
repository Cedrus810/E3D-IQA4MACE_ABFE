"""Fixed-frame POLAR-1-M CPU equivalence checks and paired force-call timings.

No integration or free-energy estimation. Run in openmm_dev; --out must be new.
"""

import argparse
import hashlib
import json
import pathlib
import statistics
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from decomp import mlmm
from run_abfe import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--qs", help="optional comma-separated subset of path states")
    parser.add_argument("--variant", choices=("optimized", "scalar", "batched", "reference"), default="optimized")
    parser.add_argument("--paths", help="optional comma-separated subset: legacy,D19,POLAR-shell")
    args = parser.parse_args()
    if args.out.exists() or args.repeats < 1:
        parser.error("use a new output path and a positive repeat count")
    torch.set_num_threads(args.threads)
    leg = ROOT / "experiments/2026-10-15_acetamide_leg/em_acetamide.npz"
    checkpoint = ROOT / "runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
    backbone = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"
    print(f"Loading POLAR-1-M and operating head on CPU, {args.dtype}", flush=True)
    model = load_model(str(checkpoint), backbone, device="cpu").to(getattr(torch, args.dtype))
    positions = np.asarray(np.load(leg, allow_pickle=True)["positions"], dtype=np.float64)
    source_files = [ROOT / "decomp" / filename for filename in
                    ("lambda_mask.py", "openmm_bridge.py", "mlmm.py", "fires.py", "polar_es.py")]
    report = dict(device="cpu", dtype=args.dtype, variant=args.variant, torch=torch.__version__, threads=args.threads,
                  source_sha256={str(filename): hashlib.sha256(filename.read_bytes()).hexdigest()
                                 for filename in source_files},
                  purpose="fixed-frame equivalence and CPU force latency, not GPU speed or scientific acceptance",
                  cases=[])
    for name, options in (("legacy", {}), ("D19", dict(shell_mm=True)), ("POLAR-shell", dict(polar_shell=True))):
        if args.paths and name not in args.paths.split(","):
            continue
        _, force, info = mlmm.build_mlmm_system(str(leg), model, device="cpu", n_inner=20, **options)
        coordinates = (0.0, 1.5, 2.5, 3.0, 3.5) + ((4.5,) if name == "POLAR-shell" else ())
        if args.qs:
            coordinates = tuple(float(value) for value in args.qs.split(","))
        for coordinate in coordinates:
            _, _, path, switch = mlmm.segment(coordinate, **options)
            force.set_p(path)
            force.set_t(switch)
            if name == "POLAR-shell":
                force.fires.set_shape(max(0.0, coordinate - 4.0))
            force.anch.fast_forces = False
            force.anch.batch_fragments = False
            reference = force.evaluate(positions, info["box"])
            force.anch.fast_forces = args.variant in ("optimized", "scalar")
            force.anch.batch_fragments = args.variant in ("optimized", "batched")
            actual = force.evaluate(positions, info["box"])
            error_energy = abs(actual[0] - reference[0])
            error_force = float(np.max(np.abs(actual[1] - reference[1])))
            if args.dtype == "float64":
                energy_tolerance, force_tolerance = 2e-5, 1e-5
            else:
                data = force.anch._graph(force.ml_positions(positions, info["box"]))
                with torch.no_grad():
                    absolute = abs(float(model.backbone(data, training=False, compute_force=False)["energy"].sum()))
                energy_tolerance = 4 * torch.finfo(torch.float32).eps * absolute * mlmm.EV_TO_KJ_PER_MOL
                force_tolerance = 0.05
            passed = error_energy <= energy_tolerance and error_force <= force_tolerance
            if not passed:
                report["cases"].append(dict(path=name, q=coordinate, passed=False,
                                             energy_error_kj_mol=error_energy, force_error_kj_mol_ang=error_force,
                                             energy_tolerance=energy_tolerance, force_tolerance=force_tolerance))
                args.out.write_text(json.dumps(report, indent=2) + "\n")
                raise AssertionError(f"{name} q={coordinate:g}: force/energy equivalence failed; see {args.out}")
            timings = {"reference": [], "optimized": []}
            for repeat in range(args.repeats):
                order = ("reference", "optimized") if repeat % 2 == 0 else ("optimized", "reference")
                for variant in order:
                    force.anch.fast_forces = variant == "optimized" and args.variant in ("optimized", "scalar")
                    force.anch.batch_fragments = variant == "optimized" and args.variant in ("optimized", "batched")
                    started = time.perf_counter()
                    force.evaluate(positions, info["box"])
                    timings[variant].append(time.perf_counter() - started)
            ratio = statistics.median(timings["reference"]) / statistics.median(timings["optimized"])
            report["cases"].append(dict(path=name, q=coordinate, passed=True, energy_error_kj_mol=error_energy,
                                         force_error_kj_mol_ang=error_force, timings_s=timings,
                                         energy_tolerance=energy_tolerance, force_tolerance=force_tolerance,
                                         cpu_latency_ratio=ratio))
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{name} q={coordinate:g}: energy error {error_energy:.2e}, "
                  f"force error {error_force:.2e}; CPU ratio {ratio:.2f}", flush=True)
    print(f"PASS -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
