"""Run the alchemical Hamiltonian inside OpenMM (WP8).

Adapted from openmmml/models/macepotential.py, which already solves the parts
that are easy to get wrong: the MACE input dict (including OMol25's
`total_charge`/`total_spin`), the official neighbour list, and the unit scaling.

The one thing worth noticing there: it uses `openmm.PythonForce`, not
`TorchForce`. So nothing needs to be TorchScript-able -- the model stays an
ordinary torch module and lambda is an ordinary Python attribute that can be
changed between windows. That removes what looked like the hard part of WP8.
"""

import pathlib

import numpy as np
import torch

from .data import polar_fields

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
            # POLAR only, and empty otherwise, so this path stays bit-identical
            # for the backbones S13.14 was measured on. Two batch builders that
            # disagree about the input dict is how a backbone works on dimers
            # and dies here, so it is the same helper mace_batch uses.
            **polar_fields(self.bb, self.r_max, 1,
                           dtype=self.dtype, device=self.device),
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

    def torch_energy(self, p):
        """U in eV, differentiable in p (Angstrom torch tensor) -- what
        hremd.TorchReplica integrates."""
        from .lambda_mask import alchemical_energy
        data = self._graph(p.detach().cpu().numpy())
        data["positions"] = p.to(self.device)       # TorchReplica keeps p on the CPU
        U, _, _ = alchemical_energy(self.model, data, self.lam, self.frag_id,
                                    mode=self.mode, ligand=self.ligand)
        return U.sum()

    def __call__(self, state):
        """OpenMM PythonForce entry point."""
        pos = state.getPositions(asNumpy=True).value_in_unit_system(
            __import__("openmm").unit.md_unit_system) * NM_TO_ANG
        U, F = self.energy_and_forces(np.asarray(pos, dtype=np.float64))
        return U * EV_TO_KJ_PER_MOL, F * EV_TO_KJ_PER_MOL / NM_TO_ANG


class AnchoredForce(AlchemicalForce):
    """The anchored Hamiltonian (RESEARCH_PLAN_v3.md S3) as an OpenMM / HREMD force.

    One path coordinate p in [0, 2] runs both segments, so a single replica
    ladder carries the whole decoupling:

        p in [0, 1]   lambda = 1 - p on every ligand atom (diagonal path), full graph
        p in [1, 2]   cut segment, s = p - 1: U_s = (1 - s) U(0+) + s U(0)

    `set_lambda` takes p, so hremd.HREMD needs no change of interface: pass the
    p ladder as its `lambdas`. TI over p then gives dG_decouple directly, and
    `last_dudl` holds dU/dp per atom -- minus c_i on the continuous segment
    (the per-atom attribution), zero on the cut segment, whose dU/dp
    (U(0) - U(0+), not attributable) is in `last_dudl_total`.
    """

    def __init__(self, model, numbers, frag_id, es=False, charge=0,
                 ligand_charge=0, multiplicity=1, ligand=0, device=None, cover=None,
                 cap=None):
        """cover: None, or a {(Z_lig, Z_env): r_cov} table (lambda_mask.COVER_R)
        for the S3 cover switch on the head's pair terms. cap: None, or the
        margin m (eV) of the S3 backbone cap on the attributed coupling."""
        super().__init__(model, numbers, frag_id, mode="anchored", charge=charge,
                         multiplicity=multiplicity, ligand=ligand, device=device)
        self.es = es
        self.cap = cap
        from .lambda_mask import COVER_DELTA, cover_table
        self.cover = None if cover is None else (
            torch.as_tensor(np.asarray(numbers), dtype=torch.long, device=self.device),
            cover_table(cover, device=self.device), COVER_DELTA)
        self.charges = (ligand_charge, charge - ligand_charge)
        self.set_lambda(0.0)

    def set_lambda(self, value):
        p = float(value)
        if not 0.0 <= p <= 2.0:
            raise ValueError(f"path coordinate {p} outside [0, 2]")
        self.p = p
        self.cut = None if p <= 1.0 else p - 1.0
        lam = 1.0 - p if p <= 1.0 else 0.0
        self.lam = torch.full_like(self.lam, lam)

    def _U(self, data, lam):
        from .lambda_mask import anchored_energy
        return anchored_energy(self.model, data, lam, self.frag_id,
                               ligand=self.ligand, es=self.es, cut=self.cut,
                               charges=self.charges, cover=self.cover, cap=self.cap)

    def torch_energy(self, p):
        data = self._graph(p.detach().cpu().numpy())
        data["positions"] = p.to(self.device)       # TorchReplica keeps p on the CPU
        return self._U(data, self.lam)[0].sum()

    def energy_and_forces(self, positions_ang, need_dudl=False):
        data = self._graph(positions_ang)
        lam = self.lam.clone().requires_grad_(need_dudl and self.cut is None)
        U, c, parts = self._U(data, lam)
        outs = [data["positions"]] + ([lam] if lam.requires_grad else [])
        grads = torch.autograd.grad(U.sum(), outs)
        F = -grads[0].detach()
        if need_dudl:
            if self.cut is None:            # dU/dp = -dU/dlambda_i, lambda = 1 - p
                g = -grads[1].detach()
                self.last_dudl_total = float(g.sum())
            else:                           # dU/ds = U(0) - U(0+), global
                g = torch.zeros_like(self.lam)
                self.last_dudl_total = float(parts["U0"] - parts["U_plus"])
            self.last_dudl = g.cpu().numpy()
            self.n_dudl_calls += 1
        return float(U.sum()), F.cpu().numpy()


def from_abfe_output(out_dir, leg="solvent", ligand_resname="MOL"):
    """Load a prepared leg from an ABFE_IBS output directory.

    Those runs already ship `system_{leg}.xml` and `topology_{leg}.cif` -- a
    solvated, equilibrated, parameterised box. Reuse them rather than rebuilding
    the system: the topology, box and solvation are not what this project is
    testing.

    Returns (system, topology, positions_angstrom, numbers, frag_id) with
    frag_id = 0 for the ligand and 1 for everything else.
    """
    import numpy as np
    import openmm
    import openmm.app as app
    from openmm import unit

    out_dir = pathlib.Path(out_dir)
    suffix = "" if leg == "complex" else f"_{leg}"
    sys_xml = out_dir / f"system{suffix}.xml"
    cif = out_dir / f"topology{suffix}.cif"
    system = openmm.XmlSerializer.deserialize(sys_xml.read_text())
    st = app.PDBxFile(str(cif))
    top = st.topology
    pos = np.asarray(st.positions.value_in_unit(unit.angstrom))
    numbers = np.array([a.element.atomic_number for a in top.atoms()])
    frag = np.array([0 if a.residue.name == ligand_resname else 1
                     for a in top.atoms()])
    if not (frag == 0).any():
        raise ValueError(f"no residue named {ligand_resname!r} in {cif}")
    return system, top, pos, numbers, frag


def carve_ml_region(topology, positions_ang, n_solvent=100, ligand_resname="MOL",
                    box_ang=None):
    """Ligand plus its `n_solvent` nearest whole solvent residues.

    The full solvated box is out of reach for a full-ML treatment (S13.13: the
    4,076-atom Atenolol leg extrapolates to ~55 GiB), so the ML region is carved
    around the ligand. Whole residues are kept: half a water molecule is not a
    chemical species the backbone has ever seen.

    Returns (positions_ang, numbers, frag_id, atom_indices) with frag_id = 0 for
    the ligand and 1 for the solvent.
    """
    import numpy as np

    residues = list(topology.residues())
    lig_res = [r for r in residues if r.name == ligand_resname]
    if not lig_res:
        raise ValueError(f"no residue named {ligand_resname!r}")
    lig_idx = np.array([a.index for r in lig_res for a in r.atoms()])
    lig_pos = positions_ang[lig_idx]

    others = [r for r in residues if r.name != ligand_resname]
    d, shifts = [], []
    for r in others:
        idx = np.array([a.index for a in r.atoms()])
        dv = positions_ang[idx][:, None] - lig_pos[None, :]
        if box_ang is not None:
            # Minimum image, on the whole residue at once so it is not split.
            # Without this a ligand near a box face picks its "nearest" waters
            # from the far side of the box.
            com = dv.reshape(-1, 3).mean(0)
            shift = -np.round(com / box_ang) * box_ang
            dv = dv + shift
            shifts.append(shift)
        else:
            shifts.append(np.zeros(3))
        # min distance to any ligand atom, not centre to centre: a water hydrogen
        # bonded to the ligand can sit further out by centroid than one that is not
        d.append(np.linalg.norm(dv, axis=-1).min())
    order = np.argsort(d)[:n_solvent]

    pos = positions_ang.copy()
    for k in order:
        idx = np.array([a.index for a in others[k].atoms()])
        pos[idx] = pos[idx] + shifts[k]
    positions_ang = pos
    keep = list(lig_idx) + [a.index for k in order for a in others[k].atoms()]
    keep = np.array(keep)
    numbers = np.array([list(topology.atoms())[i].element.atomic_number
                        for i in keep])
    frag = np.concatenate([np.zeros(len(lig_idx), dtype=int),
                           np.ones(len(keep) - len(lig_idx), dtype=int)])
    return positions_ang[keep], numbers, frag, keep


def carve_from_arrays(positions_ang, numbers, resids, resnames, n_solvent=100,
                      ligand_resname="MOL", box_ang=None):
    """carve_ml_region without an OpenMM Topology -- from plain arrays.

    Lets the ML region be carved out of a minimised snapshot saved as .npz,
    which is the practical path: minimise the full periodic box with OpenMM and
    its own force field (fast, stable, hard cores), then score the carved region
    with the learned potential. Minimising a carved cluster with an MLIP instead
    drives atoms into each other -- outside its training domain there is no wall.
    """
    import numpy as np

    positions_ang = np.asarray(positions_ang)
    resids = np.asarray(resids)
    resnames = np.asarray(resnames)
    lig_idx = np.where(resnames == ligand_resname)[0]
    if lig_idx.size == 0:
        raise ValueError(f"no atoms in residue {ligand_resname!r}")
    lig_pos = positions_ang[lig_idx]

    other_res = [r for r in np.unique(resids) if r not in set(resids[lig_idx])]
    d, shifts, groups = [], [], []
    for r in other_res:
        idx = np.where(resids == r)[0]
        dv = positions_ang[idx][:, None] - lig_pos[None, :]
        if box_ang is not None:
            shift = -np.round(dv.reshape(-1, 3).mean(0) / box_ang) * box_ang
            dv = dv + shift
        else:
            shift = np.zeros(3)
        d.append(np.linalg.norm(dv, axis=-1).min())
        shifts.append(shift); groups.append(idx)

    order = np.argsort(d)[:n_solvent]
    pos = positions_ang.copy()
    for k in order:
        pos[groups[k]] = pos[groups[k]] + shifts[k]
    keep = np.concatenate([lig_idx] + [groups[k] for k in order])
    frag = np.concatenate([np.zeros(len(lig_idx), dtype=int),
                           np.ones(len(keep) - len(lig_idx), dtype=int)])
    return pos[keep], np.asarray(numbers)[keep], frag, keep


def strip_ligand_environment_nonbonded(system, ligand_idx, force_groups=None):
    """Remove the MM ligand-environment nonbonded interaction.

    In an ML/MM split the learned decomposition supplies the ligand-environment
    coupling, so the MM NonbondedForce must not also supply it. Ligand-internal
    and environment-internal MM terms are left alone: only the cross terms are
    replaced. Implemented with exceptions rather than by deleting the force, so
    water-water electrostatics and PME keep working.
    """
    import openmm
    lig = set(int(i) for i in ligand_idx)
    for f in (system.getForce(i) for i in range(system.getNumForces())):
        if not isinstance(f, openmm.NonbondedForce):
            continue
        env = [i for i in range(f.getNumParticles()) if i not in lig]
        for i in sorted(lig):
            qi, si, ei = f.getParticleParameters(i)
            for j in env:
                f.addException(i, j, 0.0 * qi * qi, si, 0.0 * ei, replace=True)
    return system


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
