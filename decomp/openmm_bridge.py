"""Run the alchemical Hamiltonian inside OpenMM (WP8).

Adapted from openmmml/models/macepotential.py, which already solves the parts
that are easy to get wrong: the MACE input dict (including OMol25's
`total_charge`/`total_spin`), the official neighbour list, and the unit scaling.

The one thing worth noticing there: it uses `openmm.PythonForce`, not
`TorchForce`. So nothing needs to be TorchScript-able -- the model stays an
ordinary torch module and lambda is an ordinary Python attribute that can be
changed between windows. That removes what looked like the hard part of WP8.
"""

import numpy as np
import torch

# OpenMM: kJ/mol and nm.  MACE: eV and Angstrom.
EV_TO_KJ_PER_MOL = 96.48533212
NM_TO_ANG = 10.0


class AlchemicalForce:
    """Callable for `openmm.PythonForce`: state -> (energy kJ/mol, forces kJ/mol/nm).

    `lam` is a plain attribute. Set it between windows; nothing needs rebuilding.
    """

    def __init__(self, model, numbers, frag_id, mode="graph",
                 charge=0, multiplicity=1, ligand=0, device=None):
        from mace.tools import utils
        from mace.tools.torch_tools import to_one_hot
        from mace.tools.utils import atomic_numbers_to_indices

        self.model = model
        self.bb = model.backbone
        self.mode, self.ligand = mode, ligand
        self.device = device or next(self.bb.parameters()).device
        self.dtype = next(self.bb.parameters()).dtype
        self.r_max = float(self.bb.r_max)
        n = len(numbers)

        z_table = utils.AtomicNumberTable([int(z) for z in self.bb.atomic_numbers])
        self.node_attrs = to_one_hot(
            torch.tensor(atomic_numbers_to_indices(np.asarray(numbers), z_table=z_table),
                         dtype=torch.long, device=self.device).unsqueeze(-1),
            num_classes=len(z_table)).to(self.dtype)
        self.frag_id = torch.as_tensor(np.asarray(frag_id), dtype=torch.long,
                                       device=self.device)
        self.lam = torch.ones(n, dtype=self.dtype, device=self.device)
        self._const = {
            "ptr": torch.tensor([0, n], dtype=torch.long, device=self.device),
            "batch": torch.zeros(n, dtype=torch.long, device=self.device),
            "pbc": torch.tensor([False] * 3, dtype=torch.bool, device=self.device),
            "cell": torch.zeros(3, 3, dtype=self.dtype, device=self.device),
            "total_charge": torch.tensor([float(charge)], dtype=torch.long,
                                         device=self.device),
            "total_spin": torch.tensor([float(multiplicity)], dtype=torch.long,
                                       device=self.device),
        }
        self.n_dudl_calls = 0
        self.last_dudl = None

    def set_lambda(self, value):
        """Scalar for the diagonal path, or per-atom for a sequential one."""
        v = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        self.lam = v.expand_as(self.lam).clone() if v.dim() == 0 else v.to(self.device)

    def _graph(self, positions_ang):
        from mace.data.neighborhood import get_neighborhood
        cell = np.identity(3, dtype=np.float64)
        ei, shifts, _, _ = get_neighborhood(positions_ang, self.r_max,
                                            [False] * 3, cell)
        d = dict(self._const)
        d["positions"] = torch.tensor(positions_ang, dtype=self.dtype,
                                      device=self.device, requires_grad=True)
        d["node_attrs"] = self.node_attrs
        d["edge_index"] = torch.tensor(ei, dtype=torch.int64, device=self.device)
        d["shifts"] = torch.tensor(shifts, dtype=self.dtype, device=self.device)
        d["unit_shifts"] = torch.zeros_like(d["shifts"])
        return d

    def energy_and_forces(self, positions_ang, need_dudl=False):
        from .lambda_mask import alchemical_energy
        data = self._graph(positions_ang)
        lam = self.lam.clone().requires_grad_(need_dudl)
        U, _, _ = alchemical_energy(self.model, data, lam, self.frag_id,
                                    mode=self.mode, ligand=self.ligand)
        outs = [data["positions"]] + ([lam] if need_dudl else [])
        grads = torch.autograd.grad(U.sum(), outs)
        F = -grads[0].detach()
        if need_dudl:
            self.last_dudl = grads[1].detach().cpu().numpy()
            self.n_dudl_calls += 1
        return float(U.sum()), F.cpu().numpy()

    def __call__(self, state):
        """OpenMM PythonForce entry point."""
        pos = state.getPositions(asNumpy=True).value_in_unit_system(
            __import__("openmm").unit.md_unit_system) * NM_TO_ANG
        U, F = self.energy_and_forces(np.asarray(pos, dtype=np.float64))
        return U * EV_TO_KJ_PER_MOL, F * EV_TO_KJ_PER_MOL / NM_TO_ANG


def make_system(numbers, force, masses_amu=None):
    """A bare System carrying only the alchemical force -- enough for gas-phase
    stage A of the validation ladder."""
    import openmm
    from ase.data import atomic_masses
    system = openmm.System()
    for i, z in enumerate(numbers):
        m = atomic_masses[int(z)] if masses_amu is None else masses_amu[i]
        system.addParticle(float(m))
    f = openmm.PythonForce(force)
    f.setUsesPeriodicBoundaryConditions(False)
    system.addForce(f)
    return system
