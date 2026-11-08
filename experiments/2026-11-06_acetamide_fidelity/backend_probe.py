"""Analytic two-site probes of the installed finite-displacement Coulomb evaluator.

No learned-model forward, molecular energy call, trajectory, or dependency modification.
"""

import json
from pathlib import Path

import numpy as np
import torch

from graph_longrange.realspace_electrostatics import RealSpaceFiniteDiffereneEnergy

HERE = Path(__file__).resolve().parent
EV = 96.485332


def main():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        p = torch.tensor([[0., 0., 0.], [2., .7, -.3]])
        # The library maps columns [3,1,2] to Cartesian dipoles [x,y,z].
        charges = torch.tensor([.3, -.3])
        dipoles = torch.tensor([[.2, -.1, .05], [-.1, .03, .15]])
        R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        batch = torch.zeros(2, dtype=torch.long)
        density = torch.cat([charges[:, None], dipoles[:, [1, 2, 0]]], dim=1)
        rotated = torch.cat([charges[:, None], (dipoles @ R.T)[:, [1, 2, 0]]], dim=1)
        rows = []
        with torch.no_grad():
            for offset in (.02, .01, .005):
                block = RealSpaceFiniteDiffereneEnergy(1, 1., offset=offset)
                e0 = float(block(density, p, batch)[0])
                er = float(block(rotated, p @ R.T, batch)[0])
                rows.append({"offset_A": offset, "original_eV": e0, "rotated_eV": er,
                             "rotation_error_kJmol": (er - e0) * EV})
            monopoles = RealSpaceFiniteDiffereneEnergy(0, 1.)
            l0_error = float(monopoles(charges[:, None], p @ R.T, batch)[0] - monopoles(charges[:, None], p, batch)[0]) * EV
        if abs(l0_error) > 1e-10 or abs(rows[0]["rotation_error_kJmol"]) < 1e-6:
            raise ValueError("finite-displacement negative control did not distinguish monopoles and dipoles")
        if not abs(rows[2]["rotation_error_kJmol"]) < abs(rows[1]["rotation_error_kJmol"]) < abs(rows[0]["rotation_error_kJmol"]):
            raise ValueError("offset refinement did not reduce the toy rotation error")
        result = {"scope": "installed evaluator mechanism on two analytic sites; not a quantitative model-error attribution",
                  "default_dtype": "float64", "monopole_rotation_error_kJmol": l0_error, "dipole_probe": rows,
                  "learned_model_forwards": 0, "MD_steps": 0, "dependency_changes": 0}
        path = HERE / "backend_probe.json"
        if path.exists():
            raise FileExistsError("preserving the existing probe")
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        torch.set_default_dtype(previous)


if __name__ == "__main__":
    main()
