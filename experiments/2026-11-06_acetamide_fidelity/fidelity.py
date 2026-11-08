"""Fixed acetamide fidelity probes. Twenty-five target queries, seventy-five reference forwards, no MD."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENTRY = ROOT / "experiments/2026-11-06_acetamide_entry"
sys.path[:0] = [str(ROOT / "experiments/2026-10-25_eta_hmc"),
               str(ROOT / "experiments/2026-10-25_head_retrain")]


def write(path, data):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def rotate(x, indices, center, axis, angle):
    y = x.copy()
    k = np.eye(3)[axis]
    v = y[indices] - center
    c, s = np.cos(angle), np.sin(angle)
    y[indices] = center + c * v + s * np.cross(k, v) + (1 - c) * (v @ k)[:, None] * k
    return y


def response(prediction, reference):
    p, r = np.asarray(prediction, float), np.asarray(reference, float)
    if p.shape != r.shape or p.size == 0 or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("response vectors must be finite, aligned and nonempty")
    denom = float(r @ r)
    return {"n": int(p.size), "rms_reference": float(np.sqrt(np.mean(r * r))),
            "rmse_error": float(np.sqrt(np.mean((p - r) ** 2))),
            "slope_through_origin": float((p @ r) / denom) if denom > 1e-20 else None,
            "max_abs_error": float(np.max(np.abs(p - r)))}


def learned_atoms(D, frag, edges, n_lig):
    """Independent directed-edge reduction, checked against ModelEnergy's original readout."""
    values = np.asarray(D, float)
    if values.ndim == 2:
        values = values.sum(axis=1)
    src, dst = np.asarray(edges, int)
    cross = frag[src] != frag[dst]
    ends = np.where(frag[src] == 0, src, dst)
    return np.bincount(ends[cross], weights=.5 * values[cross], minlength=len(frag))[:n_lig]


def nearest_water(carved_positions, system_indices, n_lig):
    """Choose on imaged cluster geometry; raw box coordinates can select another water."""
    water = np.asarray(system_indices[n_lig:]).reshape(-1, 3)
    p = np.asarray(carved_positions)
    distances = np.linalg.norm(p[n_lig::3, None] - p[:n_lig][None], axis=-1).min(axis=1)
    return water[int(np.argmin(distances))]


def make_panel(leg, xml, x, box):
    import openmm
    n_lig = len(leg.lig)
    lig = np.asarray(leg.lig)
    rows = [{"id": "input", "kind": "input", "xyz": x.copy()}]
    p, _, _, si = leg.carve(x, 50)
    selected = nearest_water(p, si, n_lig)
    for kind, selected_indices in (("ligand_translation", lig), ("water_translation", selected)):
        for axis in range(3):
            for sign in (-1, 1):
                y = x.copy()
                y[selected_indices, axis] += sign * .005
                rows.append({"id": f"{kind}:{axis}:{sign}", "kind": kind, "axis": axis, "sign": sign, "xyz": y})
    for axis in range(3):
        for sign in (-1, 1):
            rows.append({"id": f"ligand_rotation:{axis}:{sign}", "kind": "ligand_rotation", "axis": axis,
                         "sign": sign, "xyz": rotate(x, lig, x[lig].mean(0), axis, sign * np.radians(5))})
    source = ENTRY / "mm/runs/acetamide/rep1_entry_smoke/w000_S0.npz"
    with np.load(source, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if not meta["complete"] or meta["T_K"] != leg.temperature_K or meta["endpoint"] != "S0":
            raise ValueError("wrong saved MM source")
        for i in (0, 3, 6, 9):
            y = z["xyz"][i].astype(np.float64)
            if y.shape != x.shape or not np.allclose(z["box"], box, rtol=0, atol=2e-7):
                raise ValueError("saved MM frame mapping or box mismatch")
            rows.append({"id": f"saved_mm:{i}", "kind": "saved_mm", "source": str(source),
                         "frame": i, "time_ps": float(z["times_ps"][i]), "xyz": y})
    if not np.allclose(box, np.eye(3) * box[0, 0], rtol=0, atol=1e-12):
        raise ValueError("the fixed 90-degree control requires a cubic box")
    translated = x + box[0]
    rows.append({"id": "control:translation", "kind": "control", "xyz": translated})
    rows.append({"id": "control:rotation", "kind": "control", "xyz": rotate(x, np.arange(len(x)),
                  np.diag(box) / 2, 2, np.pi / 2)})
    system = openmm.XmlSerializer.deserialize(xml.read_text())
    pairs = np.array([system.getConstraintParameters(i)[:2] for i in range(system.getNumConstraints())], int)
    d0 = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
    max_change = 0.
    for row in rows:
        if row["kind"] != "saved_mm":
            y = row["xyz"]
            change = float(np.max(np.abs(np.linalg.norm(y[pairs[:, 0]] - y[pairs[:, 1]], axis=1) - d0)))
            if change > 1e-12:
                raise ValueError("a rigid probe changed a constrained distance")
            max_change = max(max_change, change)
    if len(rows) != 25:
        raise ValueError("fixed panel must contain 25 frames")
    return rows, {"selected_water_system_indices": selected.tolist(), "max_constraint_distance_change_nm": max_change,
                  "n_particles": len(x), "ligand_indices": lig.tolist(), "box_nm": box.tolist(),
                  "temperature_K": leg.temperature_K, "n_water": 50}


def run():
    out, partial = HERE / "evaluation.json", HERE / "evaluation.partial.json"
    if out.exists() or partial.exists() or (HERE / "panel.npz").exists():
        raise FileExistsError("preserving the existing panel/evaluation; inspect before another run")
    import torch
    import eta_hmc as eh
    from decomp.clusters import coupling_labels
    from eta_exchange import RESTART_ATOL
    from validate import HEADS

    torch.set_num_threads(4)
    started = time.perf_counter()
    with np.load(ROOT / "experiments/2026-10-25_eta_hmc/logs/deep_shallow_panel_20261031.npz", allow_pickle=False) as z:
        ident = json.loads(str(z["meta"]))["contract"]
    if HEADS["nvar_b"] != ident["head"]:
        raise ValueError("head table differs from the frozen current target")
    leg = eh.local_E.Leg("acetamide")
    xml, _, _, x, box = eh.gm.leg("acetamide")
    frames, metadata = make_panel(leg, xml, x, box)
    np.savez(HERE / "panel.npz", xyz_nm=np.stack([f["xyz"] for f in frames]),
             meta=json.dumps({**metadata, "frames": [{k: v for k, v in f.items() if k != "xyz"} for f in frames]}))
    result = {"complete": False, "scope": "fixed correlated geometry probes; backbone fidelity only",
              "heads": HEADS, "backbone": ident["backbone"], "panel": metadata,
              "budget": {"target_queries": 25, "reference_forwards": 75, "head_only_readouts": 100, "wall_seconds": 600},
              "target_queries": 0, "reference_forwards": 0, "head_only_readouts": 0, "MD_steps": 0, "frames": []}
    write(partial, result)
    try:
        energy = eh.ModelEnergy(leg, 0, ckpt=ident["head"], backbone=ident["backbone"], device="cpu")
        heads = {}
        for name, checkpoint in HEADS.items():
            head = copy.deepcopy(energy.model.head)
            head.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
            heads[name] = head.eval()
        captured = {}
        def capture(module, args):
            captured["args"] = args
        handle = energy.model.head.register_forward_pre_hook(capture)
        old_input = json.loads((ENTRY / "smoke.json").read_text())["model"]["fixed_frame_dU"]
        for frame in frames:
            if time.perf_counter() - started >= 600:
                raise TimeoutError("fixed static-panel wall budget exhausted")
            value = energy(frame["xyz"])
            result["target_queries"] = energy.calls
            if frame["id"] == "input" and abs(value - old_input) > RESTART_ATOL:
                raise ValueError("initial frame no longer reproduces the committed target")
            p, n, frag, si = leg.carve(frame["xyz"], 50)
            cbb = float(coupling_labels(energy.model.backbone, [p], [n], [frag], device="cpu", batch=1)[0] * eh.EV)
            result["reference_forwards"] += 3
            h, positions, edges = captured["args"]
            npedges = edges.detach().cpu().numpy()
            coul = energy.last_components["C_ML_coul"].copy()
            by_head = {}
            with torch.no_grad():
                for name, head in heads.items():
                    _, D = head(h, positions, edges)
                    atom = learned_atoms(D.detach().cpu().numpy(), frag, npedges, len(leg.lig)) * eh.EV + coul
                    by_head[name] = {"C_ML": float(atom.sum()), "atom": atom.tolist(), "error": float(atom.sum() - cbb)}
                    result["head_only_readouts"] += 1
            check = float(np.max(np.abs(np.asarray(by_head["nvar_b"]["atom"]) - energy.last_components["C_ML"])))
            if check > RESTART_ATOL:
                raise ValueError("cached-feature readout differs from actual target atoms")
            row = {k: v for k, v in frame.items() if k != "xyz"}
            row.update({"dU": value, "C_bb": cbb, "C_MM": float(energy.last_components["C_MM"].sum()),
                        "C_MM_atom": energy.last_components["C_MM"].tolist(), "coul_atom": coul.tolist(),
                        "heads": by_head, "cached_feature_atom_error": check,
                        "cluster_system_indices": si.tolist(), "omega": energy.last_components["omega_water_resid"].tolist()})
            if not np.isfinite(cbb) or abs(value - by_head["nvar_b"]["C_ML"] + row["C_MM"]) > RESTART_ATOL:
                raise ValueError("non-finite reference or target decomposition mismatch")
            result["frames"].append(row)
            result["seconds"] = time.perf_counter() - started
            write(partial, result)
            print(f"{frame['id']}: dU {value:.3f}; head-bb " + " ".join(f"{name} {r['error']:.3f}" for name, r in by_head.items())
                  + f"; {result['seconds']:.1f}s", flush=True)
        handle.remove()
        base = result["frames"][0]
        for row in result["frames"]:
            if row["kind"] == "control":
                if row["omega"] != base["omega"]:
                    raise ValueError("symmetry control changed Omega identity")
                errors = [abs(row["C_bb"] - base["C_bb"]), abs(row["C_MM"] - base["C_MM"])]
                errors += [float(np.max(np.abs(np.asarray(row["heads"][name]["atom"]) - base["heads"][name]["atom"]))) for name in heads]
                if max(errors) > RESTART_ATOL:
                    raise ValueError("a symmetry control failed")
        result["complete"] = True
        write(out, result)
        partial.unlink()
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["seconds"] = time.perf_counter() - started
        write(partial, result)
        raise


def summarize():
    source = HERE / "evaluation.json"
    if not source.exists():
        source = HERE / "evaluation.partial.json"
    result = json.loads(source.read_text())
    if len(result["frames"]) != 25:
        raise ValueError("all 25 frames must be evaluated; incomplete acceptance is reported explicitly")
    for key, expected in (("target_queries", 25), ("reference_forwards", 75), ("head_only_readouts", 100)):
        if result[key] != expected:
            raise ValueError("evaluation count differs from the fixed panel")
    rows, base = result["frames"], result["frames"][0]
    selected = [r for r in rows if r["kind"] != "control"]
    groups = ("ligand_translation", "ligand_rotation", "water_translation", "saved_mm")
    summary = {"scope": result["scope"], "source": str(source), "acceptance_passed": result["complete"],
               "acceptance_error": result.get("error"), "heads": {}, "fixed_panel_frames": len(selected), "controls": [],
               "max_cached_feature_atom_error": max(r["cached_feature_atom_error"] for r in rows)}
    selection_audit = HERE / "panel_selection_audit.json"
    if selection_audit.exists():
        summary["panel_selection_audit"] = json.loads(selection_audit.read_text())
    for row in rows:
        if row["kind"] == "control":
            summary["controls"].append({"id": row["id"], "C_bb_difference": row["C_bb"] - base["C_bb"],
                "C_MM_difference": row["C_MM"] - base["C_MM"],
                "heads": {name: {"total_difference": row["heads"][name]["C_ML"] - base["heads"][name]["C_ML"],
                    "max_atom_difference": float(np.max(np.abs(np.array(row["heads"][name]["atom"]) - base["heads"][name]["atom"])))}
                    for name in result["heads"]}})
    for name in result["heads"]:
        errors = np.array([r["heads"][name]["error"] for r in selected])
        stats = {"input_error": base["heads"][name]["error"], "fixed_panel_bias": float(errors.mean()),
                 "error_min": float(errors.min()), "error_max": float(errors.max()),
                 "RMSE_after_mean_removal": float(np.sqrt(np.mean((errors - errors.mean()) ** 2))), "response": {}}
        for kind in groups:
            sel = [r for r in rows if r["kind"] == kind]
            pred = [r["heads"][name]["C_ML"] - base["heads"][name]["C_ML"] for r in sel]
            ref = [r["C_bb"] - base["C_bb"] for r in sel]
            stats["response"][kind] = response(pred, ref)
        summary["heads"][name] = stats
    write(HERE / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "summary"))
    args = parser.parse_args()
    run() if args.command == "run" else summarize()
