"""Step 3 of the D19 review (v3 S13.57 end, user 2026-10-21): short paired sampling at lambda = 1.

Same start, same inner identity (picked on the start frame, as production), same seed and
velocities; the only difference is the shell's own energy: old path (POLAR shell) vs D19 (TIP3P
shell). State q = 1 (lambda = 1, wall on), production settings (LangevinMiddle 298.15 K, 1/ps, 2 fs,
CUDA default precision, head float32 on the GPU). Several starts.

Asks: does D19 consistently move into the strong many-body shells (S13.57) and the old model not?
The frames are analysed afterwards (analyze.py: truth / two / many with the production identity,
and U_D19 - U_old on every frame).

Resumable: one npz per (start, model); finished ones are skipped on a rerun.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import openmm
from openmm import unit

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from decomp import mlmm  # noqa: E402
from run_abfe import load_model  # noqa: E402

LEG = ROOT / "experiments/2026-10-15_acetamide_leg/em_acetamide.npz"
CKPT = ROOT / "runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
BB = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"


def starts():
    d = np.load(LEG, allow_pickle=True)
    out = [("leg", np.asarray(d["positions"], float))]
    z = np.load(ROOT / "experiments/2026-10-16_mm_path_control/fires_seed1/state.npz", allow_pickle=True)
    out.append(("e8q1", np.asarray(z["P"][int(np.argmin(np.abs(z["qs"] - 1.0)))], float)))
    mm = np.load(ROOT / "experiments/2026-10-20_coupling_check/frames.npz")["positions"]
    out += [("mm5", np.asarray(mm[5], float)), ("mm15", np.asarray(mm[15], float))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000, help="2 fs steps per trajectory (5000 = 10 ps)")
    ap.add_argument("--every", type=int, default=250, help="save every N steps (250 = 0.5 ps)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="traj", help="output subdirectory (smoke: another name)")
    ap.add_argument("--only", nargs="*", default=None, help="start names to run (default all)")
    args = ap.parse_args()
    out_dir = HERE / args.out
    out_dir.mkdir(exist_ok=True)
    model = load_model(str(CKPT), BB, device="cuda")
    plat = openmm.Platform.getPlatformByName("CUDA")
    T = 298.15
    for sname, P0 in starts():
        if args.only and sname not in args.only:
            continue
        for mname, shell_mm in (("old", False), ("d19", True)):
            f = out_dir / f"{sname}_{mname}.npz"
            if f.exists():
                print(f"skip {f.name} (done)", flush=True)
                continue
            system, mlf, info = mlmm.build_mlmm_system(str(LEG), model, n_inner=20, es=True, device="cuda",
                                                       positions=P0, shell_mm=shell_mm)
            integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
            integ.setRandomNumberSeed(args.seed)
            ctx = openmm.Context(system, integ, plat)
            box = np.asarray(info["box"], float)
            ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])
            mlmm.set_path(ctx, mlf, 1.0, info["lig"], shell_mm=shell_mm)
            ctx.setPositions(P0 * unit.angstrom)
            ctx.setVelocitiesToTemperature(T * unit.kelvin, args.seed)
            pos, ufires, t_wall = [], [], time.perf_counter()
            for k in range(args.steps // args.every):
                integ.step(args.every)
                st = ctx.getState(getPositions=True)
                pos.append(st.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
                ufires.append(float(mlf.last.get("U_fires", np.nan)))
                if k == 0:
                    print(f"{sname}/{mname}: first block {time.perf_counter() - t_wall:.0f} s "
                          f"({(time.perf_counter() - t_wall) / args.every * 1000:.0f} ms/step)", flush=True)
            wall = time.perf_counter() - t_wall
            np.savez(f, positions=np.asarray(pos, np.float32), U_fires=np.asarray(ufires), start=sname,
                     shell_mm=shell_mm, inner=np.asarray(info["inner"]), box=box, every=args.every,
                     dt_fs=2.0, seed=args.seed, wall_s=wall)
            print(f"{sname}/{mname}: {args.steps} steps in {wall:.0f} s "
                  f"({wall / args.steps * 1000:.0f} ms/step) -> {f.name}", flush=True)
            del ctx, integ, system, mlf
    (out_dir / "done.json").write_text(json.dumps(dict(steps=args.steps, every=args.every, seed=args.seed)) + "\n")


if __name__ == "__main__":
    main()
