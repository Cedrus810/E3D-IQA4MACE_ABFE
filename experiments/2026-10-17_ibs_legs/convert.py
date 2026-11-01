"""IBS solvent legs (data/solvent_legs_ibs, REF_solvent_legs_from_ABFE_IBS_2026-10-15.md) -> E3D leg files.

Run: python convert.py      -> <system>.npz in this directory, one per system (rep1's pre-equilibrated frame)

E3D leg format (as em_solvent.npz): positions A, numbers, resnames, resids, box A, source = system XML.
Coordinates and box from pre_equil_final_state.npz (the NPT-equilibrated start IBS ran from), not the
cif (built before pre-equilibration, REF S4.5). Ligand residue LIG, first in the topology (checked).
"""
import json, os
import numpy as np
from openmm import app

SRC = "/home/ruigengji/E3D/data/solvent_legs_ibs"
OUT = os.path.dirname(os.path.abspath(__file__))
SYSTEMS = ["HSP90_ligand-17", "HSP90_ligand-19", "HSP90_ligand-27",
           "CyclophilinD_ligand-4", "CyclophilinD_ligand-8", "PWWP1_ligand-8"]
for s in SYSTEMS:
    d = f"{SRC}/{s}/rep1"
    top = app.PDBxFile(f"{d}/topology_solvent.cif").topology
    atoms = list(top.atoms())
    z = np.load(f"{d}/pre_equil_final_state.npz")
    box = np.asarray(z["box_nm"], dtype=float)
    assert np.allclose(box, np.diag(np.diag(box))), "orthorhombic only"
    lig = json.load(open(f"{d}/ligand_indices_solvent.json"))
    lig = lig if isinstance(lig, list) else list(lig.values())[0]
    resids = np.array([a.residue.index for a in atoms])
    assert atoms[0].residue.name == "LIG" and list(lig) == list(np.where(resids == resids[0])[0])
    np.savez(f"{OUT}/{s}.npz", positions=np.asarray(z["positions_nm"], dtype=np.float64) * 10.0,
             numbers=np.array([a.element.atomic_number for a in atoms]),
             resnames=np.array([a.residue.name for a in atoms]), resids=resids,
             box=np.diag(box) * 10.0, source=np.array(f"{SRC}/{s}/system_solvent.xml"))
    print(f"{s}: {len(atoms)} atoms, ligand {len(lig)}, box {np.diag(box)[0] * 10:.3f} A -> {s}.npz", flush=True)
