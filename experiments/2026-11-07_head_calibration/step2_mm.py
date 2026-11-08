"""Head calibration step 2, MM part: independent acetamide S0 trajectories (PLAN_DRAFT.md section 2, step 2).

S0 only (all groups coupled), same System / Langevin 298.15 K, 1/ps, 2 fs as groupwise_mm.run.
Each preparation starts from the minimized input with its own velocity / Langevin seed.
Usage: python step2_mm.py PREP EQUIL_PS PROD_PS SAVE_PS [PLATFORM] [OUT_DIR]
Resumable: rerunning the same command continues from the last saved chunk (positions + velocities;
a resumed piece is recorded in `continuations`). Different settings for an existing file raise.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
import openmm
from openmm import unit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/2026-10-23_groupwise_dummy"))
import groupwise_mm as gm  # noqa: E402

NAME = "acetamide"
CHUNK_PS = 20.0


def save(path, d):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **d)
    tmp.replace(path)


def main(prep, equil_ps, prod_ps, save_ps, platform="CUDA", out_dir=None):
    out_dir = Path(out_dir) if out_dir else HERE / "mm"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"prep{prep}_S0.npz"
    T = gm.temperature(NAME)
    settings = {"prep": prep, "equil_ps": equil_ps, "prod_ps": prod_ps, "save_ps": save_ps, "T_K": T,
                "dt_ps": gm.DT_PS, "friction_per_ps": 1.0, "seed": 1000 * prep + 11, "state": "S0"}
    _, system, groups, pos, box = gm.build(NAME)
    integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1.0 / unit.picosecond, gm.DT_PS * unit.picosecond)
    c = openmm.Context(system, integ, openmm.Platform.getPlatformByName(platform))
    c.setPeriodicBoxVectors(*[openmm.Vec3(*row) for row in np.asarray(box)])
    gm.set_state(c, 0, len(groups))
    stride = int(round(save_ps / gm.DT_PS))
    n_frames = int(round(prod_ps / save_ps))
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        old = json.loads(str(d["settings"]))
        if old != settings:
            raise ValueError(f"{path}: settings differ {old} vs {settings}; preserve it and use another PREP")
        if bool(d["complete"]):
            print(f"{path.name}: complete", flush=True)
            return
        integ.setRandomNumberSeed(settings["seed"] + 7919 * (len(json.loads(str(d["continuations"]))) + 1))
        c.setPositions(d["last_positions"])
        c.setVelocities(d["last_velocities"])
        conts = json.loads(str(d["continuations"])) + [{"at_frame": int(len(d["xyz"])), "equil_done": bool(d["equil_done"]),
                                                        "platform": platform}]
        xyz, times = list(d["xyz"]), list(d["time_ps"])
        equil_done = bool(d["equil_done"])
    else:
        integ.setRandomNumberSeed(settings["seed"])
        c.setPositions(pos)
        c.setVelocitiesToTemperature(T * unit.kelvin, settings["seed"])
        conts, xyz, times, equil_done = [{"at_frame": 0, "equil_done": False, "platform": platform}], [], [], False

    def checkpoint(complete=False):
        st = c.getState(getPositions=True, getVelocities=True)
        save(path, {"settings": json.dumps(settings), "continuations": json.dumps(conts),
                    "xyz": np.asarray(xyz, np.float32).reshape(-1, system.getNumParticles(), 3),
                    "time_ps": np.asarray(times, np.float64), "box": np.asarray(box, np.float64),
                    "equil_done": equil_done, "complete": complete,
                    "last_positions": st.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                    "last_velocities": st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)})

    t0 = time.perf_counter()
    if not equil_done:
        integ.step(int(round(equil_ps / gm.DT_PS)))
        equil_done = True
        print(f"prep{prep}: equil {equil_ps} ps in {time.perf_counter() - t0:.1f} s", flush=True)
        checkpoint()
    per_chunk = max(1, int(round(CHUNK_PS / save_ps)))
    t1, steps = time.perf_counter(), 0
    while len(xyz) < n_frames:
        for _ in range(min(per_chunk, n_frames - len(xyz))):
            integ.step(stride)
            steps += stride
            xyz.append(c.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
            times.append(len(times) * save_ps + save_ps)
        checkpoint()
        el = time.perf_counter() - t1
        print(f"prep{prep}: {len(xyz)}/{n_frames} frames, {steps} steps in {el:.1f} s "
              f"({el / steps * 1e3:.3f} ms/step)", flush=True)
    checkpoint(complete=True)
    print(f"prep{prep}: done in {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), float(a[1]), float(a[2]), float(a[3]), *(a[4:6]))
