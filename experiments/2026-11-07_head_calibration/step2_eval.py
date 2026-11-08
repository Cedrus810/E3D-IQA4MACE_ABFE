"""Head calibration step 2, model part: r = C_ML - C_bb at the production Omega (N=50) on saved MM frames.

Usage: python step2_eval.py MM_NPZ STRIDE [DEVICE] [THREADS]
Evaluates frames 0, STRIDE, 2*STRIDE, ... of a step2_mm.py trajectory (complete or not; only saved frames).
One nvar-b target query (via ModelEnergy, local_E v1) + 3 backbone forwards per frame, float64.
Output next to the input: <stem>_r_s<STRIDE>.json; rerunning skips frames already written.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIDELITY = ROOT / "experiments/2026-11-06_acetamide_fidelity"
sys.path[:0] = [str(ROOT / "experiments/2026-10-25_eta_hmc")]
import eta_hmc as eh  # noqa: E402
from decomp.clusters import coupling_labels  # noqa: E402


def write(path, data):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main(npz, stride, device="cpu", threads=8):
    npz, stride, threads = Path(npz).resolve(), int(stride), int(threads)
    torch.set_num_threads(threads)
    out = npz.with_name(f"{npz.stem}_r_s{stride}.json")
    heads = json.loads((FIDELITY / "evaluation.partial.json").read_text())
    with np.load(npz, allow_pickle=False) as z:
        xyz, times = z["xyz"].astype(np.float64), z["time_ps"].copy()
        settings = json.loads(str(z["settings"]))
    data = json.loads(out.read_text()) if out.exists() else {
        "source": str(npz), "mm_settings": settings, "stride": stride, "N": 50, "units": "kJ/mol",
        "head": heads["heads"]["nvar_b"], "backbone": heads["backbone"], "dtype": "float64",
        "device": device, "rows": []}
    if data["mm_settings"] != settings or data["stride"] != stride:
        raise ValueError(f"{out}: existing output belongs to different settings; preserve it")
    done = {r["frame"] for r in data["rows"]}
    leg = eh.local_E.Leg("acetamide")
    energy = eh.ModelEnergy(leg, 0, ckpt=data["head"], backbone=data["backbone"], device=device)
    for f in range(0, len(xyz), stride):
        if f in done:
            continue
        t = time.perf_counter()
        du = energy(xyz[f])
        comp = energy.last_components
        p, num, frag, si = leg.carve(xyz[f], 50)
        c_bb = float(coupling_labels(energy.model.backbone, [p], [num], [frag], device=device, batch=1)[0] * eh.EV)
        c_ml = float(comp["C_ML"].sum())
        row = {"frame": f, "time_ps": float(times[f]), "dU": du, "C_ML": c_ml,
               "C_ML_learned": float(comp["C_ML_learned"].sum()), "C_ML_coul": float(comp["C_ML_coul"].sum()),
               "C_MM": float(comp["C_MM"].sum()), "C_bb": c_bb, "r": c_ml - c_bb,
               "bb_minus_MM": c_bb - float(comp["C_MM"].sum()),
               "omega_water_resid": comp["omega_water_resid"].tolist(), "seconds": time.perf_counter() - t}
        data["rows"].append(row)
        write(out, data)
        print(f"{npz.stem} frame {f} t={times[f]:.0f} ps r={row['r']:.3f} dU={du:.3f} "
              f"bb-MM={row['bb_minus_MM']:.3f} {row['seconds']:.1f}s", flush=True)
    data["complete_for_saved_frames"] = True
    write(out, data)


if __name__ == "__main__":
    main(*sys.argv[1:])
