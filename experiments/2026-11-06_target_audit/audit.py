"""Frozen twenty-frame target audit. No trajectories, training or MM Context evaluations.

python audit.py cached     # validate saved components, export the fixed panel
python audit.py reference --device cpu  # at most 20 target queries + 60 backbone forwards
python audit.py summary --device cpu    # paired source-to-final decomposition
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
E3D = HERE.parents[1]
ETA = E3D / "experiments/2026-10-25_eta_hmc"
PANEL = ETA / "logs/deep_shallow_panel_20261031.npz"
RUNS = ETA / "runs/CyclophilinD_ligand-8"
FIELDS = ("C_ML", "C_ML_learned", "C_ML_coul", "C_MM", "dU_atom",
          "cluster_system_indices", "omega_water_resid")
sys.path[:0] = [str(E3D), str(ETA)]


def write_json(path, value):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def components(details, value):
    """Verify the same-frame decomposition before calculating descriptive summaries."""
    d = {k: np.asarray(v) for k, v in details.items()}
    if any(not np.isfinite(v).all() for v in d.values()) or not np.isfinite(value):
        raise ValueError("non-finite saved component")
    n = len(d["C_ML"])
    if any(d[k].shape != (n,) for k in ("C_ML", "C_ML_learned", "C_ML_coul", "dU_atom")):
        raise ValueError("invalid per-atom shape")
    if d["C_MM"].shape != (n, 2):
        raise ValueError("C_MM must retain Coulomb and LJ columns")
    errors = [np.max(np.abs(d["C_ML"] - d["C_ML_learned"] - d["C_ML_coul"])),
              np.max(np.abs(d["dU_atom"] - d["C_ML"] + d["C_MM"].sum(axis=1))),
              abs(d["dU_atom"].sum() - value)]
    if max(errors) > 1e-8:
        raise ValueError("saved decomposition does not close")
    if len(np.unique(d["omega_water_resid"])) != 50:
        raise ValueError("expected fifty unique water identities")
    return {"C_ML": float(d["C_ML"].sum()), "learned": float(d["C_ML_learned"].sum()),
            "coulomb": float(d["C_ML_coul"].sum()), "C_MM_es": float(d["C_MM"][:, 0].sum()),
            "C_MM_lj": float(d["C_MM"][:, 1].sum()), "dU": float(value),
            "max_closure_error": float(max(errors))}


def load_frames():
    frames = []
    with np.load(PANEL, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        contract = meta["contract"]
        if contract["state"] != 0 or contract["E_construction"] != "local_E v1" or contract["n_water"] != 50:
            raise ValueError("unexpected source target")
        for i in range(8):
            frames.append({"id": f"panel:{i}", "origin": "panel", "source": str(PANEL), "frame": i,
                           "xyz_nm": z["xyz_nm"][i].copy(), "expected_dU": float(z["expected_dU"][i]),
                           "details": {k: z[k][i].copy() for k in FIELDS}})
    for group in ("A", "B", "CA", "CB"):
        path = RUNS / f"tw_{group}_tw1_v1.npz"
        with np.load(path, allow_pickle=False) as z:
            m = json.loads(str(z["meta"]))
            if not m["complete"] or m["completed_sweeps"] != 200:
                raise ValueError(f"{group}: incomplete source")
            for key in ("head", "backbone", "E_construction", "dtype", "system", "state"):
                if m[key] != contract[key]:
                    raise ValueError(f"{group}: source target differs at {key}")
            if not np.array_equal(np.sort(z["perm"][-1]), np.arange(12)):
                raise ValueError(f"{group}: invalid final label permutation")
            for label in (0, 1, 11):
                w = int(z["perm"][-1, label])
                frames.append({"id": f"{group}:label{label}:walker{w}", "origin": group,
                               "source": str(path), "label": label, "walker": w,
                               "xyz_nm": z["state_x"][w].copy(), "expected_dU": float(z["state_dU"][w]),
                               "details": {k: z[f"state_{k}"][w].copy() for k in FIELDS}})
    for frame in frames:
        if frame["xyz_nm"].shape != (contract["n_particles"], 3) or not np.isfinite(frame["xyz_nm"]).all():
            raise ValueError("invalid coordinates")
        frame["cached"] = components(frame["details"], frame["expected_dU"])
    return contract, frames


def cached():
    contract, frames = load_frames()
    result = {"scope": "fixed diagnostic states; no equilibrium pooling or free-energy estimate",
              "source_target": {k: contract[k] for k in ("system", "state", "head", "backbone", "T_K", "dtype", "E_construction")},
              "n_frames": len(frames), "new_model_queries": 0, "new_md_steps": 0,
              "frames": [{k: f[k] for k in f if k not in ("xyz_nm", "details")} for f in frames]}
    write_json(HERE / "cached.json", result)
    print(json.dumps({"frames": len(frames), "max_closure_error": max(f["cached"]["max_closure_error"] for f in frames)}))


def reference(device):
    # Check preserved results before importing the model stack.
    out = HERE / f"reference_{device}.json"
    partial = HERE / f"reference_{device}.partial.json"
    if out.exists() or partial.exists():
        raise FileExistsError("preserving existing reference output; inspect it before another run")
    import torch
    import eta_hmc as eh
    from deep_shallow_panel import compare_frame
    from decomp.clusters import coupling_labels
    from decomp.mace_adapter import MACEDecomposition

    contract, frames = load_frames()
    torch.set_num_threads(4)
    started = time.perf_counter()
    result = {"scope": "head fidelity to backbone on fixed selected states; not physical validation or equilibrium sampling",
              "device": device, "head": contract["head"], "backbone": contract["backbone"],
              "complete": False, "budget": {"target_queries": 20, "reference_forwards": 60, "wall_seconds": 600},
              "target_queries": 0, "reference_forwards": 0, "md_steps": 0, "frames": []}
    write_json(partial, result)
    bb = torch.load(contract["backbone"], map_location="cpu", weights_only=False).to(torch.float64).to(device).eval()
    torch.manual_seed(0)
    model = MACEDecomposition(bb, hidden=256, n_pair_channels=4).to(device).eval()
    model.head.load_state_dict(torch.load(contract["head"], map_location=device))
    for p in model.parameters():
        p.requires_grad_(False)
    leg = eh.local_E.Leg(contract["system"])
    try:
        for f in frames:
            if time.perf_counter() - started >= 600:
                raise TimeoutError("static audit reached its fixed wall budget")
            pos, numbers, frag, si = leg.carve(f["xyz_nm"], 50)
            mm = leg.c_mm(pos, si, leg.dummy(0))
            tot, learned, coul = eh.couplings_dt(model, [(pos, numbers, frag)], leg.dummy(0), torch.float64,
                                                split=True, device=device)
            result["target_queries"] += 1
            details = {"C_ML": tot[0] * eh.EV, "C_ML_learned": learned[0] * eh.EV,
                       "C_ML_coul": coul[0] * eh.EV, "C_MM": mm,
                       "dU_atom": tot[0] * eh.EV - mm.sum(axis=1), "cluster_system_indices": si,
                       "omega_water_resid": np.unique(leg.top[4][si[len(leg.lig):]])}
            du = float(details["dU_atom"].sum())
            comparison = compare_frame(du, details, f["expected_dU"], f["details"])
            row = {k: f[k] for k in f if k not in ("xyz_nm", "details")}
            row["cache_check"] = comparison
            if not comparison["cache_reproduced"]:
                result["frames"].append(row)
                raise ValueError(f"{f['id']}: cache mismatch; inspect partial output")
            cbb = float(coupling_labels(bb, [pos], [numbers], [frag], device=device, batch=1)[0] * eh.EV)
            result["reference_forwards"] += 3
            cml = float(details["C_ML"].sum())
            cmm = float(mm.sum())
            row.update({"C_bb": cbb, "head_minus_backbone": cml - cbb, "backbone_minus_MM": cbb - cmm,
                        "current": components(details, du), "current_per_atom": {k: details[k].tolist() for k in FIELDS}})
            if not np.isfinite(cbb) or abs(du - (cml - cbb) - (cbb - cmm)) > 1e-8:
                raise ValueError("non-finite reference or failed target decomposition")
            result["frames"].append(row)
            result["seconds"] = time.perf_counter() - started
            write_json(partial, result)
            print(f"{f['id']}: dU={du:.3f}, head-bb={cml-cbb:.3f}, bb-MM={cbb-cmm:.3f}; {result['seconds']:.1f}s", flush=True)
        result["complete"] = True
        write_json(out, result)
        partial.unlink()
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["seconds"] = time.perf_counter() - started
        write_json(partial, result)
        raise


def summary(device):
    path = HERE / f"reference_{device}.json"
    result = json.loads(path.read_text())
    rows = result["frames"]
    if not result["complete"] or len(rows) != 20 or not all(r["cache_check"]["cache_reproduced"] for r in rows):
        raise ValueError("a complete cache-consistent reference panel is required")
    if result["target_queries"] != 20 or result["reference_forwards"] != 60:
        raise ValueError("reference evaluation counts differ from the fixed panel")
    grouped = {}
    for origin in ("panel", "A", "B", "CA", "CB"):
        sel = [r for r in rows if r["origin"] == origin]
        grouped[origin] = {"n": len(sel)}
        for key in ("head_minus_backbone", "backbone_minus_MM", "C_bb"):
            values = [r[key] for r in sel]
            grouped[origin][key] = {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}
    by_id = {r["id"]: r for r in rows}
    paired = []
    for origin, start in (("A", 0), ("B", 4), ("CA", 0), ("CB", 4)):
        src = by_id[f"panel:{start}"]
        end = next(r for r in rows if r["origin"] == origin and r["label"] == 0)
        diff = {"dU": end["current"]["dU"] - src["current"]["dU"],
                "head_minus_backbone": end["head_minus_backbone"] - src["head_minus_backbone"],
                "backbone_minus_MM": end["backbone_minus_MM"] - src["backbone_minus_MM"]}
        if abs(diff["dU"] - diff["head_minus_backbone"] - diff["backbone_minus_MM"]) > 1e-8:
            raise ValueError("paired difference identity failed")
        paired.append({"origin": origin, "initial": src["id"], "final": end["id"], "difference": diff})
    output = {"source": str(path), "scope": "descriptive fixed-frame and paired endpoint differences, not equilibrium statistics",
              "all_caches_reproduced": True,
              "max_abs_cache_dU_difference": max(abs(r["cache_check"]["deltaU_difference"]) for r in rows),
              "grouped": grouped, "paired_b0_changes": paired}
    write_json(HERE / "summary.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cached", "reference", "summary"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "cached":
        cached()
    elif args.command == "reference":
        reference(args.device)
    else:
        summary(args.device)
