"""Check reusable acetamide input identity without molecular energy or trajectory calls."""

import json
from pathlib import Path
import sys

import numpy as np
import openmm
from openmm import app, unit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from audit import write_json


def main():
    source = ROOT / "experiments/2026-10-15_acetamide_leg"
    xml = source / "system_solvent.xml"
    cif = source / "topology_solvent.cif"
    npz = source / "em_acetamide.npz"
    system = openmm.XmlSerializer.deserialize(xml.read_text())
    top = app.PDBxFile(str(cif))
    atoms = list(top.topology.atoms())
    lig = [a.index for a in atoms if a.residue.index == 0]
    waters = [r for r in top.topology.residues() if r.name == "HOH"]
    groups = json.loads((ROOT / "experiments/2026-10-23_groupwise_dummy/groups.json").read_text())["acetamide"]
    flat = [i for g in groups["groups"] for i in g["system_index"]]
    if sorted(flat) != lig or len(set(flat)) != len(flat):
        raise ValueError("group mapping does not partition the ligand")
    with np.load(npz, allow_pickle=False) as z:
        xyz = z["positions"].copy()
        numbers = z["numbers"].copy()
        box = np.diag(z["box"] / 10)
        if len(atoms) != system.getNumParticles() or xyz.shape != (len(atoms), 3):
            raise ValueError("particle count mismatch")
        if not np.array_equal(numbers, [a.element.atomic_number for a in atoms]):
            raise ValueError("element mapping mismatch")
        if not np.array_equal(z["resids"], [a.residue.index for a in atoms]):
            raise ValueError("residue mapping mismatch")
        if not np.array_equal(z["resnames"], [a.residue.name for a in atoms]):
            raise ValueError("residue names mismatch")
    cifxyz = top.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    dxyz = float(np.max(np.abs(cifxyz - xyz)))
    if not np.allclose(cifxyz, xyz, rtol=0, atol=1e-4):
        raise ValueError("CIF/NPZ coordinates differ beyond CIF text rounding")
    xmlbox = np.asarray([v.value_in_unit(unit.nanometer) for v in system.getDefaultPeriodicBoxVectors()])
    if not np.allclose(xmlbox, box, rtol=0, atol=1e-10):
        raise ValueError("XML/NPZ box mismatch")
    if any(isinstance(f, openmm.MonteCarloBarostat) for f in system.getForces()):
        raise ValueError("expected fixed-box NVT input")
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    q = [nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in lig]
    if abs(sum(q)) > 1e-8:
        raise ValueError("expected a neutral ligand")
    result = {"inputs": {"xml": str(xml), "cif": str(cif), "npz": str(npz)},
              "input_identity_passed": True, "system": "acetamide", "temperature_K": 298.15,
              "temperature_source": str(source / "prep_acetamide.py"),
              "n_particles": len(atoms), "n_waters": len(waters), "n_constraints": system.getNumConstraints(),
              "ligand_indices": lig, "groups": groups, "ligand_charges_e": q,
              "ligand_masses_Da": [system.getParticleMass(i).value_in_unit(unit.dalton) for i in lig],
              "box_nm": box.tolist(), "max_cif_npz_difference_A": dxyz,
              "nonbonded_method": str(nb.getNonbondedMethod()),
              "cutoff_nm": nb.getCutoffDistance().value_in_unit(unit.nanometer),
              "has_CM_motion_remover": any(isinstance(f, openmm.CMMotionRemover) for f in system.getForces()),
              "model_cluster_atoms_for_50_waters": len(lig) + 150,
              "new_model_queries": 0, "new_MM_energy_evaluations": 0, "new_MD_steps": 0,
              "current_entrypoint_gaps": [
                  "groupwise_mm.leg and model_eval.topology resolve only the IBS input layout",
                  "MM bridge, endpoint kernels and analyses use 300 K constants; acetamide requires 298.15 K throughout",
                  "fresh independent S0 solvent starts under the current protocol are still required"],
              "scope": "reusable input identity only; no new-protocol free energy or sampling acceptance"}
    write_json(HERE / "acetamide_input.json", result)
    print(json.dumps({k: result[k] for k in ("input_identity_passed", "n_particles", "n_waters", "n_constraints",
                                             "max_cif_npz_difference_A", "model_cluster_atoms_for_50_waters")}))


if __name__ == "__main__":
    main()
