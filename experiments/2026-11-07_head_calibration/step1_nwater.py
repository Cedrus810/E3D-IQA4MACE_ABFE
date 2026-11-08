"""Head calibration step 1: r = C_ML - C_bb versus the number of Omega waters N.

Diagnostic only (PLAN_DRAFT.md section 2, step 1). local_E v1 keeps N=50; other N are probes.
5 frames (input + 4 saved MM frames of the fidelity panel) x N in {10,16,24,32,40,50}:
30 nvar-b target queries + 90 backbone forwards, CPU float64, 0 MD, existing stencil backend.
Rerunning skips (frame, N) rows already in the output.
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

N_WATERS = (10, 16, 24, 32, 40, 50)
FRAME_IDS = ("input", "saved_mm:0", "saved_mm:3", "saved_mm:6", "saved_mm:9")
OUT = HERE / "step1_nwater.json"


def write(data):
    tmp = OUT.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(OUT)


def main():
    torch.set_num_threads(8)
    source = json.loads((FIDELITY / "evaluation.partial.json").read_text())
    cached = {f["id"]: f for f in source["frames"]}
    with np.load(FIDELITY / "panel.npz", allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        ids = [f["id"] for f in meta["frames"]]
        xyz = {fid: z["xyz_nm"][ids.index(fid)].copy() for fid in FRAME_IDS}
    data = json.loads(OUT.read_text()) if OUT.exists() else {
        "scope": "diagnostic N-water probe; local_E v1 unchanged (N=50); stencil backend; no MD",
        "head": source["heads"]["nvar_b"], "backbone": source["backbone"],
        "dtype": "float64", "device": "cpu", "units": "kJ/mol", "rows": []}
    done = {(r["frame"], r["N"]) for r in data["rows"]}
    leg = eh.local_E.Leg("acetamide")
    energy = eh.ModelEnergy(leg, 0, ckpt=data["head"], backbone=data["backbone"], device="cpu")
    dm = leg.dummy(0)
    for fid in FRAME_IDS:
        for n_w in N_WATERS:
            if (fid, n_w) in done:
                continue
            t = time.perf_counter()
            p, num, frag, si = leg.carve(xyz[fid], n_w)
            total, learned, explicit = eh.couplings_dt(energy.model, [(p, num, frag)], dm, torch.float64,
                                                       split=True, device="cpu")
            c_ml = float(total[0].sum() * eh.EV)
            c_learned = float(learned[0].sum() * eh.EV)
            c_coul = float(explicit[0].sum() * eh.EV)
            c_bb = float(coupling_labels(energy.model.backbone, [p], [num], [frag], device="cpu", batch=1)[0] * eh.EV)
            c_mm = leg.c_mm(p, si, dm)
            row = {"frame": fid, "N": n_w, "n_cluster_atoms": int(len(p)),
                   "water_resid": sorted(int(v) for v in np.unique(leg.top[4][si[len(dm):]])),
                   "C_ML": c_ml, "C_ML_learned": c_learned, "C_ML_coul": c_coul, "C_bb": c_bb,
                   "C_MM_coul": float(c_mm[:, 0].sum()), "C_MM_lj": float(c_mm[:, 1].sum()),
                   "r": c_ml - c_bb, "dU": c_ml - float(c_mm.sum()), "T": c_bb - c_coul,
                   "learned_minus_T": c_learned - (c_bb - c_coul),
                   "seconds": time.perf_counter() - t}
            if n_w == 50:   # reproduction of the fidelity panel's N=50 numbers, reported not gated
                row["N50_cache_dU_diff"] = row["dU"] - cached[fid]["dU"]
                row["N50_cache_Cbb_diff"] = row["C_bb"] - cached[fid]["C_bb"]
            data["rows"].append(row)
            write(data)
            print(f"{fid:11s} N={n_w:2d} atoms={len(p):3d} r={row['r']:9.3f} Cbb={c_bb:10.3f} "
                  f"dU={row['dU']:9.3f} {row['seconds']:.1f}s", flush=True)
    data["complete"] = len(data["rows"]) == len(FRAME_IDS) * len(N_WATERS)
    write(data)


if __name__ == "__main__":
    main()
