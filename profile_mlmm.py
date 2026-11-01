"""Bounded ML/MM timing and nested profiling, without resuming a production run.

See PERFORMANCE_PLAN_2026-10-20.md for the measurement and acceptance protocol.
"""

import argparse
import contextlib
import functools
import hashlib
import json
import os
import pathlib
import socket
import statistics
import time
from unittest.mock import patch


class StageTimer:
    """Nested wall times; CUDA synchronization is diagnostic and adds overhead."""

    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.stack = []
        self.rows = {}

    @contextlib.contextmanager
    def stage(self, name):
        self.synchronize()
        frame = dict(path="/".join([entry["name"] for entry in self.stack] + [name]),
                     name=name, children=0.0)
        self.stack.append(frame)
        started = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            elapsed = time.perf_counter() - started
            self.stack.pop()
            row = self.rows.setdefault(frame["path"], dict(calls=0, total_s=0.0, self_s=0.0))
            row["calls"] += 1
            row["total_s"] += elapsed
            row["self_s"] += elapsed - frame["children"]
            if self.stack:
                self.stack[-1]["children"] += elapsed

    def wrap(self, name, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            label = name(*args, **kwargs) if callable(name) else name
            with self.stage(label):
                return function(*args, **kwargs)
        return wrapped

    def report(self):
        return [dict(stage=name, **values, mean_ms=1000 * values["total_s"] / values["calls"])
                for name, values in sorted(self.rows.items())]


@contextlib.contextmanager
def instrument(force, timer, torch):
    from decomp.polar_es import _Recorder

    anchored = force.anch
    backbone = anchored.model.backbone
    full_size = len(anchored.frag_id)

    def backbone_label(data, *args, **kwargs):
        if len(data["ptr"]) > 2:
            return "backbone.fragments_batched"
        size = len(data["positions"])
        return "backbone.full" if size == full_size else f"backbone.fragment_{size}"

    targets = [(force, "evaluate", "force.evaluate"),
               (force, "ml_positions", "coordinates.unwrap"),
               (force, "fragment_energies", "fragments.force"),
               (anchored, "energy_and_forces", "anchored.force"),
               (anchored, "_graph", "graph.build"),
               (anchored, "_U", "anchored.energy"),
               (backbone, "forward", backbone_label),
               (anchored.model.head, "forward", "head"),
               (backbone.coulomb_energy, "forward", "coulomb.kernel"),
               (_Recorder, "forward", "coulomb.recorder"),
               (_Recorder, "_cross", "coulomb.cross"),
               (force.fires, "energy_and_forces", "fires.force"),
               (torch.autograd, "grad", "autograd.grad")]
    with contextlib.ExitStack() as stack:
        for target, attribute, name in targets:
            stack.enter_context(patch.object(target, attribute,
                                             timer.wrap(name, getattr(target, attribute))))
        yield


def timed(function, synchronize, repeats):
    samples = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        function()
        synchronize()
        samples.append(time.perf_counter() - started)
    return dict(samples_s=samples, median_s=statistics.median(samples),
                min_s=min(samples), max_s=max(samples))


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def arguments():
    root = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg-npz", type=pathlib.Path,
                        default=root / "experiments/2026-10-15_acetamide_leg/em_acetamide.npz")
    parser.add_argument("--ckpt", type=pathlib.Path, default=root / (
        "runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"))
    parser.add_argument("--backbone", type=pathlib.Path,
                        default=pathlib.Path("/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"))
    parser.add_argument("--start", help="original inner-water selection frame, stack.npz:index")
    parser.add_argument("--snapshot", type=pathlib.Path,
                        help="read positions from state.npz at each exact q; inner identities still use --start/leg")
    parser.add_argument("--qs", default="0,1.5,2,2.5,3,3.5,4")
    parser.add_argument("--n-inner", type=positive_int, default=20)
    parser.add_argument("--steps", type=positive_int, default=10)
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--profile-calls", type=positive_int, default=3)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--threads", type=positive_int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--platform", default="CUDA")
    paths = parser.add_mutually_exclusive_group()
    paths.add_argument("--old-path", action="store_true", help="profile pre-D19 SW path")
    paths.add_argument("--polar-shell", action="store_true", help="profile POLAR-shell + shape path")
    parser.add_argument("--reference-forces", action="store_true")
    parser.add_argument("--batch-fragments", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ml-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--trace", action="store_true", help="also export a separate CPU/CUDA torch trace per q")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True, help="must be a new directory")
    args = parser.parse_args()
    try:
        args.qs = [float(value) for value in args.qs.split(",")]
    except ValueError:
        parser.error("--qs must be comma-separated numbers")
    limit = 5.0 if args.old_path or args.polar_shell else 4.0
    if not args.qs or any(not 0 <= value <= limit for value in args.qs):
        parser.error(f"--qs must be in [0, {limit:g}]")
    if not 0 < args.dt <= 2.0:
        parser.error("--dt must be in (0, 2] fs for this timing protocol")
    if args.out_dir.exists():
        parser.error("--out-dir already exists; use a fresh directory")
    return args


def main():
    args = arguments()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import openmm
    import torch
    from openmm import unit
    from decomp import mlmm
    from run_abfe import load_model

    torch.set_num_threads(args.threads)
    platform = openmm.Platform.getPlatformByName(args.platform)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no timing was run")
    synchronize = torch.cuda.synchronize if args.device == "cuda" else lambda: None
    with np.load(args.leg_npz, allow_pickle=True) as leg:
        start = np.asarray(leg["positions"], dtype=np.float64)
    if args.start:
        filename, index = args.start.rsplit(":", 1)
        with np.load(filename, allow_pickle=True) as frames:
            start = np.asarray(frames["positions"][int(index)], dtype=np.float64)
    snapshots = None
    if args.snapshot:
        with np.load(args.snapshot, allow_pickle=True) as saved:
            shell_mm = not (args.old_path or args.polar_shell)
            saved_polar = bool(saved["polar_shell"]) if "polar_shell" in saved.files else False
            if bool(saved["shell_mm"]) != shell_mm or saved_polar != args.polar_shell:
                raise ValueError("snapshot and requested path have different shell_mm settings")
            snapshots = {}
            for value in args.qs:
                matches = np.flatnonzero(np.isclose(saved["qs"], value, atol=1e-8, rtol=0))
                if len(matches) != 1:
                    raise ValueError(f"snapshot must contain exactly one state at q={value:g}")
                snapshots[value] = np.asarray(saved["P"][matches[0]], dtype=np.float64)
    args.out_dir.mkdir(parents=True)
    root = pathlib.Path(__file__).resolve().parent
    source_paths = [pathlib.Path(__file__), root / "run_abfe.py"]
    source_paths += [root / "decomp" / name for name in (
        "mlmm.py", "openmm_bridge.py", "lambda_mask.py", "polar_es.py", "fires.py", "head.py")]
    metadata = dict(host=socket.gethostname(), torch=torch.__version__, openmm=openmm.__version__,
                    gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                    arguments={key: str(value) if isinstance(value, pathlib.Path) else value
                               for key, value in vars(args).items()},
                    source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in source_paths}, inputs={})
    input_paths = [args.leg_npz, args.ckpt, args.backbone]
    if args.snapshot:
        input_paths.append(args.snapshot)
    if args.start:
        input_paths.append(pathlib.Path(args.start.rsplit(":", 1)[0]))
    for path in input_paths:
        stat = path.stat()
        metadata["inputs"][str(path.resolve())] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    result = dict(metadata=metadata, cases=[], interpretation=(
        "Uninstrumented step/force timing is the performance baseline. Nested stages synchronize CUDA "
        "at every boundary and add overhead. Inclusive stage times overlap; sum self_s only. "
        "No free energy or scientific acceptance is measured. Weight files are identified by path/stat, not hash."))
    report = args.out_dir / "profile.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    model = load_model(str(args.ckpt), str(args.backbone), device=args.device).to(getattr(torch, args.ml_dtype))
    system, force, info = mlmm.build_mlmm_system(
        str(args.leg_npz), model, n_inner=args.n_inner, device=args.device,
        positions=start, shell_mm=not (args.old_path or args.polar_shell), polar_shell=args.polar_shell,
        fast_forces=not args.reference_forces,
        batch_fragments=args.batch_fragments and not args.reference_forces)
    integrator = openmm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1 / unit.picosecond,
                                                args.dt * unit.femtosecond)
    integrator.setRandomNumberSeed(7)
    context = openmm.Context(system, integrator, platform)
    box = np.asarray(info["box"], dtype=np.float64)
    context.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[axis] * box[axis] / 10))
                                    for axis in range(3)])
    metadata["inner_water_indices"] = np.asarray(info["inner"]).tolist()
    for case_index, value in enumerate(args.qs):
        positions = start if snapshots is None else snapshots[value]
        mlmm.set_path(context, force, value, info["lig"], shell_mm=not (args.old_path or args.polar_shell),
                      polar_shell=args.polar_shell)
        context.setPositions(positions * unit.angstrom)
        context.setVelocitiesToTemperature(298.15 * unit.kelvin, 7)
        integrator.step(args.warmup)

        def step_block():
            integrator.step(args.steps)
            context.getState(getPositions=True)

        baseline = timed(step_block, synchronize, args.repeats)
        reference = force.evaluate(positions, box)
        force_time = timed(lambda: force.evaluate(positions, box), synchronize, args.repeats)
        timer = StageTimer(synchronize)
        with instrument(force, timer, torch):
            for _ in range(args.profile_calls):
                profiled = force.evaluate(positions, box)
        case = dict(q=value, step=baseline, force=force_time, stages=timer.report(),
                    ms_per_step=1000 * baseline["median_s"] / args.steps,
                    ns_per_day=args.dt * 1e-6 * args.steps * 86400 / baseline["median_s"],
                    instrumentation_delta=dict(energy_kj_mol=abs(profiled[0] - reference[0]),
                                               max_force_kj_mol_ang=float(np.max(np.abs(
                                                   profiled[1] - reference[1])))))
        if args.trace:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if args.device == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities) as trace:
                force.evaluate(positions, box)
                synchronize()
            trace.export_chrome_trace(str(args.out_dir / f"trace_{case_index}_q{value:g}.json"))
        result["cases"].append(case)
        report.write_text(json.dumps(result, indent=2) + "\n")
        print(f"q={value:g}: {case['ms_per_step']:.2f} ms/step, "
              f"force {force_time['median_s'] * 1000:.2f} ms; "
              f"{case['ns_per_day']:.3f} ns/day", flush=True)
    print(f"-> {report}", flush=True)


if __name__ == "__main__":
    main()
