"""Five target queries and nine reference forwards on an isolated model clone; no MD."""

import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIDELITY = ROOT / "experiments/2026-11-06_acetamide_fidelity"
sys.path[:0] = [str(ROOT / "experiments/2026-10-25_eta_hmc"), str(HERE)]
import eta_hmc as eh
from analytic_gto import replace_in_clone
from decomp.clusters import coupling_labels
from eta_exchange import RESTART_ATOL


def write(path, value):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    out = HERE / "diagnostic.json"
    if out.exists():
        raise FileExistsError("preserving the existing diagnostic")
    torch.set_num_threads(4)
    start = time.perf_counter()
    source = json.loads((FIDELITY / "evaluation.partial.json").read_text())
    with np.load(FIDELITY / "panel.npz", allow_pickle=False) as z:
        x0, x90 = z["xyz_nm"][0].copy(), z["xyz_nm"][-1].copy()
    result = {"complete": False, "scope": "isolated candidate, no production target change", "MD_steps": 0,
              "target_queries": 0, "reference_forwards": 0, "current": [], "candidate": []}
    write(out, result)
    try:
        leg = eh.local_E.Leg("acetamide")
        original = eh.ModelEnergy(leg, 0, ckpt=source["heads"]["nvar_b"], backbone=source["backbone"], device="cpu")
        for name, x, row in (("input", x0, source["frames"][0]), ("rotation90", x90, source["frames"][-1])):
            value = original(x)
            result["target_queries"] += 1
            atom = original.last_components["C_ML"]
            error = float(np.max(np.abs(atom - row["heads"]["nvar_b"]["atom"])))
            if abs(value - row["dU"]) > RESTART_ATOL or error > RESTART_ATOL:
                raise ValueError("original target cache changed before candidate comparison")
            result["current"].append({"id": name, "dU": value, "C_ML_atom": atom.tolist(), "cache_atom_error": error})
            write(out, result)
        candidate = copy.deepcopy(original)
        candidate.calls = 0
        result["replacements"] = replace_in_clone(candidate.model.backbone)
        parameters = dict(original.model.named_parameters())
        candidate_parameters = dict(candidate.model.named_parameters())
        identical = (parameters.keys() == candidate_parameters.keys()
                     and all(torch.equal(p, candidate_parameters[name]) for name, p in parameters.items()))
        if not identical:
            raise ValueError("candidate changed learned parameters")
        result["learned_parameter_tensors_identical"] = True
        for name, x in (("input", x0), ("rotation90", x90), ("cluster_rotation37", None)):
            if time.perf_counter() - start > 300:
                raise TimeoutError("candidate diagnostic wall budget exhausted")
            if x is not None:
                du = candidate(x)
                atom = candidate.last_components["C_ML"].copy()
                coul = candidate.last_components["C_ML_coul"].copy()
                p, n, f, si = leg.carve(x, 50)
                mm = candidate.last_components["C_MM"].copy()
            else:
                p, n, f, si = leg.carve(x0, 50)
                angle = np.radians(37.)
                R = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)], [0., np.sin(angle), np.cos(angle)]])
                p = (p - p[:9].mean(0)) @ R.T + p[:9].mean(0)
                total, _, explicit = eh.couplings_dt(candidate.model, [(p, n, f)], leg.dummy(0), torch.float64, split=True, device="cpu")
                atom, coul = total[0] * eh.EV, explicit[0] * eh.EV
                mm = leg.c_mm(p, si, leg.dummy(0))
                du = float(atom.sum() - mm.sum())
            result["target_queries"] += 1
            cbb = float(coupling_labels(candidate.model.backbone, [p], [n], [f], device="cpu", batch=1)[0] * eh.EV)
            result["reference_forwards"] += 3
            result["candidate"].append({"id": name, "dU": du, "C_bb": cbb, "C_ML_atom": atom.tolist(),
                                        "coul_atom": coul.tolist(), "C_MM": float(mm.sum()),
                                        "head_minus_backbone": float(atom.sum() - cbb)})
            write(out, result)
            print(f"{name}: dU {du:.9f}, Cbb {cbb:.9f}, head-bb {atom.sum()-cbb:.6f}", flush=True)
        base = result["candidate"][0]
        controls = []
        for row in result["candidate"][1:]:
            errors = {"id": row["id"], "dU_difference": row["dU"] - base["dU"],
                      "C_bb_difference": row["C_bb"] - base["C_bb"], "C_MM_difference": row["C_MM"] - base["C_MM"],
                      "max_atom_difference": float(np.max(np.abs(np.array(row["C_ML_atom"]) - base["C_ML_atom"])))}
            controls.append(errors)
            if any(abs(v) > RESTART_ATOL for k, v in errors.items() if k != "id"):
                raise ValueError(f"candidate symmetry check failed: {errors}")
        result["symmetry_checks"] = controls
        if result["target_queries"] != 5 or result["reference_forwards"] != 9:
            raise ValueError("candidate diagnostic counts differ from the contract")
        result["complete"] = True
        result["seconds"] = time.perf_counter() - start
        write(out, result)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["seconds"] = time.perf_counter() - start
        write(out, result)
        raise


if __name__ == "__main__":
    main()
