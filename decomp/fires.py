"""FIRES: keep the n ML waters nearest the ligand (RESEARCH_PLAN_v3.md S6.3.2, D11, D16).

Rowley & Roux, JCTC 8, 3526 (2012), eq 3: the configurational integral of N
indistinguishable waters factorises into the n nearest the solute and the
N - n outside the outermost of them, so sampling with the inner set held
strictly nearer than the outer set gives exact averages, whatever the
potential. Their restraint (eq 4-5) is a half-harmonic push on every outer
water closer than R_inner = max over inner waters.

Here, two changes, both decided in D16:

- Ordering distance d_w = softmin_beta over (water O, ligand heavy atom)
  pairs of the minimum-image distance (`heavy_selection`). Atenolol is
  12 x 6 x 10 A, so a centre-of-mass sphere would not follow it. Over all
  atoms (the closest-atom metric of `carve_from_arrays`) the restraint lands
  on H, stacks, and is unstable at 2 fs: omega dt >= 1.8 and a NaN at 57-63 ps,
  against <= 0.66 and stable on heavy atoms (v3 S13.38; user, 2026-10-14).
- Restraint U = (k/2) sum_{w in I} sum_{v in O} relu(d_w - d_v)^2, a pairwise
  variant of eq 5. It equals eq 5 when only the outermost inner water is
  beyond a given outer one, is zero on the allowed region like eq 5, and is
  C1 everywhere; eq 5's max hands the reaction force to another water when
  the outermost inner one changes during a violation, so its force jumps.

Units: positions in Angstrom, k in kJ/mol/A^2, energy in kJ/mol. `__call__`
is an `openmm.PythonForce` computation (kJ/mol, kJ/mol/nm).
"""

import numpy as np
import torch

NM_TO_ANG = 10.0
K_PAPER = 500.0 * 4.184          # 500 kcal/mol/A^2 (Rowley & Roux), in kJ/mol/A^2


def softmin(x, beta, dim=-1):
    """-(1/beta) log sum exp(-beta x): below min(x) by at most ln(len)/beta."""
    return -torch.logsumexp(-beta * x, dim=dim) / beta


class Fires:
    """FIRES restraint on whole waters around a ligand, orthorhombic PBC.

    lig_idx: ligand atom indices. water_idx: [n_water, 3] atom indices of each
    water. inner: indices into water_idx of the n inner (ML) waters; set them
    with `select_inner` on a starting frame, then they never change.
    """

    def __init__(self, lig_idx, water_idx, n_inner, beta=20.0, k=K_PAPER,
                 device="cpu", dtype=torch.float64):
        self.lig = torch.as_tensor(np.asarray(lig_idx), dtype=torch.long, device=device)
        self.wat = torch.as_tensor(np.asarray(water_idx), dtype=torch.long, device=device)
        self.n_inner, self.beta, self.k = int(n_inner), float(beta), float(k)
        self.device, self.dtype = device, dtype
        self.inner = None
        self.last = {}

    def distances(self, pos, box):
        """d_w for every water: [n_water]. pos [N, 3] A tensor, box [3] A tensor."""
        dv = pos[self.wat][:, :, None, :] - pos[self.lig][None, None, :, :]
        dv = dv - box * torch.round(dv / box)
        r = torch.sqrt((dv ** 2).sum(-1))                    # [n_water, 3, n_lig]
        return softmin(r.reshape(r.shape[0], -1), self.beta)

    def select_inner(self, pos_ang, box_ang):
        pos, box = self._t(pos_ang), self._t(box_ang)
        with torch.no_grad():
            d = self.distances(pos, box)
        order = torch.argsort(d)
        self.inner = order[:self.n_inner].clone()
        mask = torch.zeros(len(d), dtype=torch.bool, device=self.device)
        mask[self.inner] = True
        self.inner_mask = mask
        return self.inner.cpu().numpy()

    def energy(self, pos, box):
        """U_F in kJ/mol, differentiable in pos."""
        d = self.distances(pos, box)
        di, do = d[self.inner_mask], d[~self.inner_mask]
        gap = torch.relu(di[:, None] - do[None, :])          # inner farther than outer
        U = 0.5 * self.k * (gap ** 2).sum()
        with torch.no_grad():
            self.last = dict(U=float(U), n_viol=int((gap > 0).sum()),
                             max_depth=float(gap.max()) if gap.numel() else 0.0,
                             r_inner=float(di.max()), d_outer_min=float(do.min()))
        return U

    def energy_and_forces(self, pos_ang, box_ang):
        pos = self._t(pos_ang).requires_grad_(True)
        U = self.energy(pos, self._t(box_ang))
        (g,) = torch.autograd.grad(U, pos)
        return float(U.detach()), -g.detach().cpu().numpy()   # kJ/mol, kJ/mol/A

    def __call__(self, state):
        """openmm.PythonForce entry point."""
        from openmm import unit
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        box = np.diag(state.getPeriodicBoxVectors(asNumpy=True)
                      .value_in_unit(unit.angstrom))
        U, F = self.energy_and_forces(np.asarray(pos), box)
        return U, F * NM_TO_ANG                                # kJ/mol/A -> kJ/mol/nm

    def _t(self, x):
        return torch.as_tensor(np.asarray(x), dtype=self.dtype, device=self.device)


def heavy_selection(lig_idx, numbers, water_idx):
    """(ligand heavy atoms, [n_water, 1] water oxygens): the D16 ordering atoms.

    water_idx rows are (O, H, H) as `water_indices` returns them for TIP3P.
    """
    lig_idx, numbers, water_idx = np.asarray(lig_idx), np.asarray(numbers), np.asarray(water_idx)
    assert (numbers[water_idx[:, 0]] == 8).all(), "first atom of each water must be O"
    return lig_idx[numbers[lig_idx] > 1], water_idx[:, :1]


def water_indices(resnames, resids):
    """[n_water, 3] atom indices of each HOH residue, in residue order."""
    resnames, resids = np.asarray(resnames), np.asarray(resids)
    w = np.where(resnames == "HOH")[0]
    groups = {}
    for i in w:
        groups.setdefault(int(resids[i]), []).append(int(i))
    out = np.array([groups[r] for r in sorted(groups)])
    assert out.shape[1] == 3, out.shape
    return out
