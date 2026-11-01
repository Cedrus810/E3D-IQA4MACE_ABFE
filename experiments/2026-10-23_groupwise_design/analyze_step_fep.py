"""Analyze existing groupwise MM arrays and optional saved ML decompositions.

No MD, model loading, reference-value fitting, or automatic sampling acceptance.
Saved ML scores are identified as legacy outputs, not newly validated predictions.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from step_fep import KB, estimate_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--n-water", type=int, default=50)
    parser.add_argument("--include-ml", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / "experiments/2026-10-23_groupwise_dummy"
    groups = json.loads((source / "groups.json").read_text())[args.system]["groups"]
    ligand = json.loads((root / "data/solvent_legs_ibs" / args.system / "rep1/ligand_indices_solvent.json").read_text())["ligand_indices"]
    lookup = {atom: index for index, atom in enumerate(ligand)}
    kt = KB * args.temperature
    state_files = [source / "runs" / args.system / f"rep{args.rep}_S{k}.npz" for k in range(len(groups) + 1)]
    u = []
    for k, path in enumerate(state_files):
        with np.load(path, allow_pickle=False) as z:
            if int(z["state"].item()) != k:
                parser.error(f"state mismatch in {path}")
            energy = z["u"].astype(float)
            if energy.ndim != 2 or energy.shape[0] != len(groups) + 1:
                parser.error(f"wrong state energy matrix in {path}")
            u.append(energy)
    records = []
    for k, group in enumerate(groups, start=1):
        forward = kt * (u[k - 1][k] - u[k - 1][k - 1])
        reverse = kt * (u[k][k - 1] - u[k][k])
        d = None
        ml_file = None
        if args.include_ml:
            # Existing score filenames encode rep1 only: never silently use them for another rep.
            if args.rep != 1:
                parser.error("legacy ML filenames are rep1-only")
            ml_file = source / "runs" / args.system / f"steps_N{args.n_water}_S{k - 1}.npz"
            with np.load(ml_file, allow_pickle=False) as z:
                if not np.array_equal(z["frames"], np.arange(len(forward))):
                    parser.error(f"ML frames do not cover the same forward sample in {ml_file}")
                d = z["D"][:, [lookup[a] for a in group["system_index"]]]
        result = estimate_step(forward, args.temperature, d, reverse)
        result.update(step=k, source_state=k - 1, destination_state=k,
                      group_atom_ids=group["system_index"],
                      source_file=str(state_files[k - 1]), destination_file=str(state_files[k]),
                      ml_file=str(ml_file) if ml_file else None)
        records.append(result)
    report = {
        "system": args.system, "rep": args.rep, "temperature_K": args.temperature,
        "method": "actual_MM_step_EXP_with_optional_ML_work_decomposition",
        "legacy_source_metadata": "temperature supplied explicitly; original files lack a full parameter manifest",
        "legacy_ml_scores": args.include_ml,
        "physical_validation": "not_completed",
        "steps": records,
        "total_forward_exp_kj_mol": sum(r["delta_g_kj_mol"] for r in records),
        "total_reverse_exp_kj_mol": sum(r["delta_g_reverse_exp_kj_mol"] for r in records),
    }
    if args.include_ml:
        report["total_atom_path_contributions_kj_mol"] = sum(sum(r["atom_delta_g_kj_mol"]) for r in records)
        report["total_residual_path_contribution_kj_mol"] = sum(r["residual_delta_g_kj_mol"] for r in records)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{args.system}: forward {report['total_forward_exp_kj_mol']:+.6f}, "
          f"reverse {report['total_reverse_exp_kj_mol']:+.6f} kJ/mol -> {args.out}")


if __name__ == "__main__":
    main()
