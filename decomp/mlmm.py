"""ML/MM solvent leg with FIRES and the MM wall (RESEARCH_PLAN_v3.md S6.3, D10, D14-D16).

Atoms: L (ligand), I (n inner waters: ML, identity fixed by FIRES), O (outer
waters: MM). Who computes what (S6.3.1):

    L internal, L-I, I-I   ML: the anchored U of S3 on L u I (`AnchoredForce`), with the
                           cover switch on the head's pair terms and the backbone
                           cap on sum c (S3, S13.41)
    L-O                    MM Coulomb (PME) + LJ, linear in lambda (D10)
    I-O, O-O, O internal   MM, unchanged (TIP3P, rigid)
    wall L x (I u O)       MM, WCA-shifted repulsion, switch s_w (W1, W3)
    FIRES I vs O           restraint on heavy-atom ordering (decomp/fires.py)

One path coordinate q in [0, 4] runs every segment (S6.3.4):

    q in [0, 1]  W1   lambda = 1,     s_w = q            p = 0
    q in [1, 2]  W2   lambda = 2 - q, s_w = 1            p = q - 1   (anchored, continuous)
    q in [2, 3]  cut  lambda = 0,     s_w = 1            p = q - 1   (anchored cut segment)
    q in [3, 4]  W3   lambda = 0,     s_w = 4 - q        p = 2       (wall softcored off)

where p is `AnchoredForce`'s path coordinate. Force groups: 0 bonded and
the rest, 1 NonbondedForce, 2 L-O LJ, 3 wall, 4 ML + FIRES (PythonForce).

The L-O Coulomb is exactly linear in each lambda_i: ligand charges enter as
particle-parameter offsets lambda_i q_i on zero base charges, and every L-L
and L-I pair is an exception with zero charge product, so PME's exclusion
correction removes their reciprocal part (the S13.36 mechanism). NonbondedForce
has no parameter derivatives in OpenMM 8.5, so dU/dlambda is evaluated as
E(lambda = 1) - E(lambda = 0) on the current positions, which linearity makes
exact (check E2).
"""

import numpy as np
import openmm
import torch
from openmm import unit

from .fires import Fires, heavy_selection, water_indices
from .openmm_bridge import AnchoredForce, EV_TO_KJ_PER_MOL, NM_TO_ANG

# D15 radius, revised by the user 2026-10-15 (v3 S13.41): per (ligand element,
# water element) the larger of the shortest contact in the operating head's
# training clusters (lampool400, 2,474 ligand clusters) and DES370K p1. At
# lambda = 0+ only the remainder r acts, and r is trusted only where the
# backbone-labelled clusters reach; the first radius (DES370K p1: H-Ow 1.32,
# H-Hw 1.18, ...) let ligand H be crushed to 1.1-1.25 A, r -> -1,243 kJ/mol.
# Angstrom. Rows: ligand element; columns: water O, water H.
WALL_R0 = {1: (1.66, 1.39), 6: (2.69, 1.83), 7: (2.62, 1.69), 8: (2.46, 1.54)}
G_NB, G_LJ, G_WALL, G_ML = 1, 2, 3, 4
# S3 backbone cap margin (user, 2026-10-15; v3 S13.41): ~3 sd of the remainder
# at n = 20 (0.15-0.17 eV, S13.37), so ordinary configurations are untouched. eV.
CAP_M = 0.5


def segment(q):
    """(lambda, s_w, p) at path coordinate q in [0, 4]."""
    q = float(q)
    if not 0.0 <= q <= 4.0:
        raise ValueError(f"path coordinate {q} outside [0, 4]")
    if q <= 1.0:
        return 1.0, q, 0.0
    if q <= 2.0:
        return 2.0 - q, 1.0, q - 1.0
    if q <= 3.0:
        return 0.0, 1.0, q - 1.0
    return 0.0, 4.0 - q, 2.0


class MLMMForce:
    """openmm.PythonForce computation: anchored ML on L u I plus FIRES.

    The ML coordinates are taken from the full system and unwrapped around
    the ligand: the ligand by minimum image onto its first heavy atom, each
    inner water by minimum image of its oxygen onto the ligand centroid.
    """

    def __init__(self, model, numbers, lig_idx, inner_water_idx, fires, es=True,
                 device=None, cover=True, cap=CAP_M):
        self.lig = np.asarray(lig_idx)
        self.inner_w = np.asarray(inner_water_idx)                # [n, 3]
        self.ml_idx = np.concatenate([self.lig, self.inner_w.ravel()])
        frag = np.r_[np.zeros(len(self.lig), int), np.ones(self.inner_w.size, int)]
        from .lambda_mask import COVER_R
        self.anch = AnchoredForce(model, np.asarray(numbers)[self.ml_idx], frag, es=es,
                                  device=device, cover=COVER_R if cover else None, cap=cap)
        self.fires = fires
        self.anchor = int(self.lig[np.asarray(numbers)[self.lig] > 1][0])
        self.need_dudl = False
        self.last = {}

    def set_p(self, p):
        self.anch.set_lambda(p)

    def ml_positions(self, pos, box):
        """L u I coordinates (A), unwrapped around the ligand."""
        lp = pos[self.lig] - pos[self.anchor]
        lp = lp - box * np.round(lp / box) + pos[self.anchor]
        c = lp.mean(0)
        w = pos[self.inner_w]                                       # [n, 3, 3]
        shift = -box * np.round((w[:, 0] - c) / box)
        return np.concatenate([lp, (w + shift[:, None, :]).reshape(-1, 3)])

    def evaluate(self, pos, box):
        """(U kJ/mol, F kJ/mol/A over all atoms, parts)."""
        U_ml, F_ml = self.anch.energy_and_forces(self.ml_positions(pos, box),
                                                 need_dudl=self.need_dudl)
        U_f, F_f = self.fires.energy_and_forces(pos, box)
        F = F_f.copy()
        F[self.ml_idx] += F_ml * EV_TO_KJ_PER_MOL
        self.last = dict(U_ml=U_ml * EV_TO_KJ_PER_MOL, U_fires=U_f, **self.fires.last)
        return U_ml * EV_TO_KJ_PER_MOL + U_f, F

    def __call__(self, state):
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        box = np.diag(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom))
        U, F = self.evaluate(np.asarray(pos, dtype=np.float64), box)
        return U, F * NM_TO_ANG


def _strip_ligand_bonded(system, lig):
    """Zero every bonded MM term that touches only ligand atoms (ML supplies them)."""
    L = set(int(i) for i in lig)
    for f in system.getForces():
        if isinstance(f, openmm.HarmonicBondForce):
            for k in range(f.getNumBonds()):
                i, j, r0, _ = f.getBondParameters(k)
                if i in L and j in L:
                    f.setBondParameters(k, i, j, r0, 0.0)
        elif isinstance(f, openmm.HarmonicAngleForce):
            for k in range(f.getNumAngles()):
                i, j, l, a0, _ = f.getAngleParameters(k)
                if i in L and j in L and l in L:
                    f.setAngleParameters(k, i, j, l, a0, 0.0)
        elif isinstance(f, openmm.PeriodicTorsionForce):
            for k in range(f.getNumTorsions()):
                i, j, l, m, n, ph, _ = f.getTorsionParameters(k)
                if {i, j, l, m} <= L:
                    f.setTorsionParameters(k, i, j, l, m, n, ph, 0.0)
        elif type(f).__name__.startswith(("Custom", "CMAP")) and not isinstance(
                f, openmm.CMMotionRemover):
            raise NotImplementedError(f"unhandled force {type(f).__name__} in the prepared leg")


def _nonbonded(nb, lig, inner_atoms):
    """NonbondedForce edits; returns the ligand's (q e, sigma nm, eps kJ/mol)."""
    L = [int(i) for i in lig]
    q, sig, eps = {}, {}, {}
    for i in L:
        qi, si, ei = nb.getParticleParameters(i)
        q[i] = qi.value_in_unit(unit.elementary_charge)
        sig[i] = si.value_in_unit(unit.nanometer)
        eps[i] = ei.value_in_unit(unit.kilojoule_per_mole)
    have = {}
    for k in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(k)
        have[(min(i, j), max(i, j))] = k

    def zero_pair(i, j):
        key = (min(i, j), max(i, j))
        if key in have:
            nb.setExceptionParameters(have[key], key[0], key[1], 0.0, 1.0, 0.0)
        else:
            have[key] = nb.addException(key[0], key[1], 0.0, 1.0, 0.0)

    for a, i in enumerate(L):                       # L-L: ML
        for j in L[a + 1:]:
            zero_pair(i, j)
    I = [int(i) for i in inner_atoms]
    for i in L:                                     # L-I: ML
        for j in I:
            zero_pair(i, j)
    for a, i in enumerate(I):                       # I-I across waters: ML (intra already excluded)
        for j in I[a + 1:]:
            zero_pair(i, j)
    for i in L:                                     # charges -> lambda_i offsets; LJ -> group 2
        nb.setParticleParameters(i, 0.0, sig[i], 0.0)
        nb.addGlobalParameter(f"lam_{i}", 1.0)
        nb.addParticleParameterOffset(f"lam_{i}", i, q[i], 0.0, 0.0)
    return q, sig, eps


def _ligand_outer_lj(system, nb, lig, outer_atoms, sig, eps, dispersion_correction):
    """L-O Lennard-Jones, linear in the global lam_lj, NonbondedForce's cutoff."""
    f = openmm.CustomNonbondedForce(
        "lam_lj*4*e*((s/r)^12-(s/r)^6); s=0.5*(s1+s2); e=sqrt(e1*e2)")
    f.addGlobalParameter("lam_lj", 1.0)
    f.addEnergyParameterDerivative("lam_lj")
    f.addPerParticleParameter("s")
    f.addPerParticleParameter("e")
    for i in range(system.getNumParticles()):
        if i in sig:
            f.addParticle([sig[i], eps[i]])
        else:
            _, si, ei = nb.getParticleParameters(i)
            f.addParticle([si.value_in_unit(unit.nanometer),
                           ei.value_in_unit(unit.kilojoule_per_mole)])
    f.addInteractionGroup([int(i) for i in lig], [int(i) for i in outer_atoms])
    f.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    f.setCutoffDistance(nb.getCutoffDistance())
    f.setUseSwitchingFunction(nb.getUseSwitchingFunction())
    if nb.getUseSwitchingFunction():
        f.setSwitchingDistance(nb.getSwitchingDistance())
    f.setUseLongRangeCorrection(dispersion_correction)
    f.setForceGroup(G_LJ)
    return f


def _wall(system, lig, water_atoms, numbers, eps_w, alpha):
    """U = s_w eps_w g(r_eff / r0), g(x) = x^-12 - 2 x^-6 + 1 for x < 1 (WCA-shifted),
    r_eff = (alpha (1 - s_w) r0^6 + r^6)^(1/6) (softcore, only felt for s_w < 1)."""
    els = sorted(WALL_R0)
    t_of = {z: k for k, z in enumerate(els)}            # ligand types 0..3, water O 4, water H 5
    nt = len(els) + 2
    tab = np.zeros((nt, nt))
    for z, (rO, rH) in WALL_R0.items():
        for a, b, r in ((t_of[z], nt - 2, rO), (t_of[z], nt - 1, rH)):
            tab[a, b] = tab[b, a] = r / 10.0                 # nm
    f = openmm.CustomNonbondedForce(
        "s_w*eps_w*step(r0-reff)*((r0/reff)^12-2*(r0/reff)^6+1);"
        "reff=(alpha_w*(1-s_w)*r0^6+r^6)^(1/6); r0=r0tab(t1,t2)")
    f.addTabulatedFunction("r0tab", openmm.Discrete2DFunction(nt, nt, tab.ravel(order="F")))
    f.addGlobalParameter("s_w", 0.0)
    f.addGlobalParameter("eps_w", float(eps_w))
    f.addGlobalParameter("alpha_w", float(alpha))
    f.addEnergyParameterDerivative("s_w")
    f.addPerParticleParameter("t")
    L, Wset = set(int(i) for i in lig), set(int(i) for i in water_atoms)
    for i in range(system.getNumParticles()):
        z = int(numbers[i])
        if i in L:
            if z not in t_of:
                raise ValueError(f"no wall radius for ligand element Z={z}")
            f.addParticle([float(t_of[z])])
        elif i in Wset:
            f.addParticle([float(nt - 2 if z == 8 else nt - 1)])
        else:
            f.addParticle([0.0])
    f.addInteractionGroup(sorted(L), sorted(Wset))
    f.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    f.setCutoffDistance(max(r for r0 in WALL_R0.values() for r in r0) / 10.0 + 0.01)
    f.setForceGroup(G_WALL)
    return f


def build_mlmm_system(leg_npz, model, n_inner=20, eps_w=25.0, alpha=0.5, es=True,
                      device=None, dispersion_correction=True, positions=None, cover=True,
                      cap=None):
    """The ML/MM solvent leg of S6.3.

    leg_npz: an em_solvent.npz-style file (positions A, numbers, resnames,
    resids, box A, source = the prepared system XML). positions overrides the
    starting frame (it also picks the inner waters). eps_w in kJ/mol (to be
    set from the lambda = 0 smoke, S6.3.3). Returns (system, ml_force, info).
    """
    d = np.load(leg_npz, allow_pickle=True)
    system = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
    pos = np.asarray(d["positions"] if positions is None else positions, dtype=np.float64)
    numbers, box = np.asarray(d["numbers"]), np.asarray(d["box"], dtype=np.float64)
    lig = np.where(d["resids"] == d["resids"][0])[0]
    W = water_indices(d["resnames"], d["resids"])
    fires = Fires(*heavy_selection(lig, numbers, W), n_inner=n_inner,
                  device=device or "cpu")
    inner = fires.select_inner(pos, box)
    outer = np.setdiff1d(np.arange(len(W)), inner)
    inner_atoms, outer_atoms = W[inner].ravel(), W[outer].ravel()

    _strip_ligand_bonded(system, lig)
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    assert len(nb) == 1, "expected one NonbondedForce"
    nb = nb[0]
    nb.setUseDispersionCorrection(dispersion_correction)
    # The L-I and I-I exceptions join different molecules, which the raw
    # (unwrapped) coordinates can place a box apart; OpenMM's default computes
    # exceptions without PBC (check E6).
    nb.setExceptionsUsePeriodicBoundaryConditions(True)
    nb.setForceGroup(G_NB)
    q, sig, eps = _nonbonded(nb, lig, inner_atoms)
    system.addForce(_ligand_outer_lj(system, nb, lig, outer_atoms, sig, eps,
                                     dispersion_correction))
    system.addForce(_wall(system, lig, W.ravel(), numbers, eps_w, alpha))
    mlf = MLMMForce(model, numbers, lig, W[inner], fires, es=es, device=device, cover=cover,
                    cap=CAP_M if cap is None else cap)
    pf = openmm.PythonForce(mlf)
    pf.setUsesPeriodicBoundaryConditions(True)
    pf.setForceGroup(G_ML)
    system.addForce(pf)
    info = dict(lig=lig, water=W, inner=inner, inner_atoms=inner_atoms,
                outer_atoms=outer_atoms, q=q, box=box, numbers=numbers, positions=pos)
    return system, mlf, info


def set_path(ctx, mlf, q, lig):
    """Put context + ML force at path coordinate q (S6.3.4)."""
    lam, s_w, p = segment(q)
    for i in lig:
        ctx.setParameter(f"lam_{int(i)}", lam)
    ctx.setParameter("lam_lj", lam)
    ctx.setParameter("s_w", s_w)
    mlf.set_p(p)
    return lam, s_w, p


def group_energy(ctx, group):
    return ctx.getState(getEnergy=True, groups={group}).getPotentialEnergy() \
        .value_in_unit(unit.kilojoule_per_mole)


def mm_far_field_dudl(ctx, lig, lam, h=0.5):
    """dU/dlambda of the MM L-O terms at the current positions, kJ/mol.

    The NonbondedForce energy is a quadratic in the ligand charges, so a
    central difference is exact for any h: the quadratic part is the ligand's
    interaction with its own periodic images, ~0.06 kJ/mol here (check E2).
    The L-O LJ is linear and comes from its parameter derivative. Leaves the
    context at lambda.
    """
    def nb_at(x):
        for i in lig:
            ctx.setParameter(f"lam_{int(i)}", x)
        return group_energy(ctx, G_NB)
    d_nb = (nb_at(lam + h) - nb_at(lam - h)) / (2 * h)
    nb_at(lam)
    st = ctx.getState(getParameterDerivatives=True, groups={G_LJ})
    return d_nb + st.getEnergyParameterDerivatives()["lam_lj"]


def mm_far_field_per_atom_coulomb(ctx, lig, lam, h=0.5):
    """Per-ligand-atom dU/dlambda_i of the L-O Coulomb on the diagonal path
    (all lambda_j = lam), kJ/mol, by exact central differences in lambda_i.
    Sums to the NonbondedForce part of mm_far_field_dudl. Single-atom charge
    switching would not do: it charges the cell, and PME's neutralising
    background and the atom's own images enter (check E2).
    """
    out = []
    for i in lig:
        ctx.setParameter(f"lam_{int(i)}", lam + h)
        ep = group_energy(ctx, G_NB)
        ctx.setParameter(f"lam_{int(i)}", lam - h)
        em = group_energy(ctx, G_NB)
        ctx.setParameter(f"lam_{int(i)}", lam)
        out.append((ep - em) / (2 * h))
    return np.array(out)
