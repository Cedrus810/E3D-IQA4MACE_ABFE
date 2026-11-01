"""Export/re-evaluate exact cached final configurations. No MD or Context checkpoint load.

export uses saved arrays only; rescore loads the frozen model once and queries eight frames.
The comparison tolerance is the existing eta_exchange.RESTART_ATOL, not a new sampling gate.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from eta_exchange import DETAILS, RESTART_ATOL, same_contract_value

HERE = Path(__file__).resolve().parent
CONTRACT = ("system", "state", "leg", "head", "backbone", "E_construction", "T_K", "dtype", "box_nm",
            "ligand_system_indices", "groups", "dummy_mask", "n_water", "n_particles", "system_definition",
            "energy_units", "xyz_units")


def export_panel(path):
    rows, configs, expected, reference = [], [], [], None
    fields = {k: [] for k in DETAILS}
    for tag in ("_ex_v1x1000b", "_dense8_v1x1000"):
        for rep in range(1, 5):
            source = HERE / "runs/CyclophilinD_ligand-8" / f"ladder_S0_start{rep}{tag}.npz"
            with np.load(source, allow_pickle=False) as z:
                m = json.loads(str(z["meta"]))
                if not m["complete"] or m["completed_sweeps"] != 1000:
                    raise ValueError(f"source not complete at 1000: {source}")
                contract = {k: m[k] for k in CONTRACT}
                if reference is None:
                    reference = contract
                diff = [k for k in CONTRACT if not same_contract_value(k, reference[k], contract[k])]
                if diff:
                    raise ValueError(f"source target differs: {source}: {diff}")
                w = int(z["perm"][-1, -1])
                configs.append(z["last_positions"][w].copy())
                expected.append(float(z["dU"][-1, w]))
                for k in DETAILS:
                    fields[k].append(z[k][-1, w].copy())
                rows.append({"source": str(source), "tag": tag, "rep": rep, "sweep": 1000,
                             "walker": w, "eta": m["etas"][-1], "start_frames": m["start_frames"],
                             "head_mtime_at_source": m.get("head_mtime"),
                             "backbone_mtime_at_source": m.get("backbone_mtime")})
    meta = {"schema": "e3d-deep-shallow-panel-1", "contract": reference, "frames": rows,
            "note": "Selected by protocol and final eta=1 state, not reweighted or pooled as equilibrium samples"}
    np.savez(path, meta=json.dumps(meta), xyz_nm=np.array(configs), expected_dU=np.array(expected),
             **{k: np.array(v) for k, v in fields.items()})
    print(json.dumps({"panel": str(path), "frames": len(rows), "cached_deltaU": expected}, indent=1))


def compare_frame(value, components, expected, details):
    errors = {k: float(np.max(np.abs(np.asarray(components[k], float) - details[k]))) for k in DETAILS}
    delta = float(value - expected)
    finite = np.isfinite(value) and all(np.isfinite(np.asarray(components[k])).all() for k in DETAILS)
    identity = all(np.array_equal(components[k], details[k]) for k in ("cluster_system_indices", "omega_water_resid"))
    ok = bool(finite and identity and abs(delta) <= RESTART_ATOL and
              all(errors[k] <= RESTART_ATOL for k in DETAILS))
    return {"current_deltaU": float(value), "cached_deltaU": float(expected), "deltaU_difference": delta,
            "component_max_abs_difference": errors, "same_cluster_and_omega": identity, "cache_reproduced": ok}


def rescore_panel(path, out):
    # GPU/model imports are confined to this command. export and pure comparisons remain CPU/NumPy only.
    import eta_hmc as eh
    import local_E

    with np.load(path, allow_pickle=False) as z:
        m = json.loads(str(z["meta"]))
        if m.get("schema") != "e3d-deep-shallow-panel-1":
            raise ValueError("wrong panel schema")
        c = m["contract"]
        if c["state"] != 0 or c["dtype"] != "float64" or c["E_construction"] != "local_E v1":
            raise ValueError("this diagnostic is restricted to frozen CypD S0 / float64 / local_E v1")
        me = eh.ModelEnergy(local_E.Leg(c["system"]), c["state"], c["dtype"], ckpt=c["head"], backbone=c["backbone"])
        ident = eh.model_identity(ckpt=c["head"], backbone=c["backbone"])
        rows = []
        for i, src in enumerate(m["frames"]):
            value = me(z["xyz_nm"][i])
            result = compare_frame(value, me.last_components, z["expected_dU"][i], {k: z[k][i] for k in DETAILS})
            rows.append({**src, **result})
            print(f"frame {i}: {src['tag']} rep{src['rep']} dU {value:.6f}; "
                  f"cache reproduced {result['cache_reproduced']}", flush=True)
        result = {"panel": str(path), "current_model_identity": ident, "existing_restart_atol": RESTART_ATOL,
                  "model_calls": me.calls, "model_seconds": me.seconds, "frames": rows,
                  "all_caches_reproduced": all(r["cache_reproduced"] for r in rows),
                  "note": "same-frame cache/target consistency only, not physical accuracy or basin population"}
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    if not result["all_caches_reproduced"]:
        raise SystemExit("Source energies/atoms differ: inspect saved diagnostics before new sampling")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("export", "rescore"))
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.command == "export":
        export_panel(Path(a.panel))
    else:
        if not a.out:
            ap.error("rescore requires --out")
        rescore_panel(Path(a.panel), a.out)
