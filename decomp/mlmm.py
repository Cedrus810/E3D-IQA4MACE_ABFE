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

One path coordinate q in [0, 5] runs every segment (S6.3.4; SW inserted
2026-10-17, user -- before that q in [3, 4] was a W3 with ML waters):

    q in [0, 1]  W1   lambda = 1,     s_w = q            p = 0
    q in [1, 2]  W2   lambda = 2 - q, s_w = 1            p = q - 1   (anchored, continuous)
    q in [2, 3]  cut  lambda = 0,     s_w = 1            p = q - 1   (anchored cut segment)
    q in [3, 4]  SW   lambda = 0,     s_w = 1, t = q - 3 inner waters ML -> MM (D18):
                      U_ML = E_bb(L) + (1 - t) E_bb(I); the I-I MM nonbonded pairs x t
    q in [4, 5]  W3   lambda = 0,     s_w = 5 - q, t = 1  all-MM solvent, wall softcored off;
                      the decoupled ligand keeps its ML internal energy E_bb(L)

With shell_mm=True (D19 option 3, user 2026-10-18; v3 S6.3.4a) ML carries
only the ligand and its coupling to the shell, `E_bb(L) + truth - sum (1 - lambda_i) c_i`;
the inner waters' own energy and the I-I pairs are ordinary MM (TIP3P). The
shell is then plain water at lambda = 0, SW and dG_ref are not needed, and
the path is q in [0, 4]:

    q in [0, 1]  W1   lambda = 1,     s_w = q            p = 0
    q in [1, 2]  W2   lambda = 2 - q, s_w = 1            p = q - 1
    q in [2, 3]  cut  lambda = 0,     s_w = 1            p = q - 1   (U(0) = E_bb(L))
    q in [3, 4]  W3   lambda = 0,     s_w = 4 - q        all-MM solvent; the ligand keeps E_bb(L)

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

from .fires import Fires, ShapeFires, heavy_selection, water_indices
from .openmm_bridge import AnchoredForce, EV_TO_KJ_PER_MOL, NM_TO_ANG

# D15 radius, revised by the user 2026-10-15 (v3 S13.41): per (ligand element,
# water element) the larger of the shortest contact in the operating head's
# training clusters (lampool400, 2,474 ligand clusters) and DES370K p1. At
# lambda = 0+ only the remainder r acts, and r is trusted only where the
# backbone-labelled clusters reach; the first radius (DES370K p1: H-Ow 1.32,
# H-Hw 1.18, ...) let ligand H be crushed to 1.1-1.25 A, r -> -1,243 kJ/mol.
# Angstrom. Rows: ligand element; columns: water O, water H.
WALL_R0 = {1: (1.66, 1.39), 6: (2.69, 1.83), 7: (2.62, 1.69), 8: (2.46, 1.54)}
# Ions (user, 2026-10-23): the wall also keeps MM ions out of the ghost ligand, where the unscaled
# ligand-ion LJ in dU/dlambda diverges (v3 S13.54). Ions are MM only, so D15's ML-trust rule does
# not apply; r0 = ION_WALL_SCALE * sigma_ij with sigma_ij = (min ligand sigma of the element +
# ion sigma) / 2. 0.79 is the smallest r0 / sigma_ij of the heavy-element water-O radii above
# (C 0.83, N 0.83, O 0.79 on HSP90-27's GAFF2/TIP3P). Its cost at lambda = 1 is in W1's FEP.
ION_WALL_SCALE = 0.79
G_NB, G_LJ, G_WALL, G_ML = 1, 2, 3, 4
# S3 backbone cap margin (user, 2026-10-15; v3 S13.41): ~3 sd of the remainder
# at n = 20 (0.15-0.17 eV, S13.37), so ordinary configurations are untouched. eV.
CAP_M = 0.5


def segment(q, ref=False, shell_mm=False, polar_shell=False):
    """(lambda, s_w, p, t) at path coordinate q in [0, 5]; t = inner-water ML -> MM switch.

    ref=True is D18's neat-solvent reference leg dG_ref (q in [3, 4] only): the same
    switch with the wall off, around a single ghost atom (build with fires_center="point").
    shell_mm=True is the D19 path, q in [0, 4] with W3 on [3, 4] (t = 1: ML on the ligand only).
    polar_shell=True retains POLAR water through W3 on [3, 4], then changes
    only FIRES shape on [4, 5]. Its water-conversion switch t stays zero."""
    q = float(q)
    if polar_shell:
        if ref or shell_mm:
            raise ValueError("POLAR-shell path cannot be combined with shell_mm or ref")
        if not 0.0 <= q <= 5.0:
            raise ValueError(f"path coordinate {q} outside [0, 5] (polar_shell)")
        if q <= 3.0:
            return segment(q)
        return 0.0, max(0.0, 4.0 - q), 2.0, 0.0
    if shell_mm:
        if ref:
            raise ValueError("D19 (shell_mm) has no dG_ref leg")
        if not 0.0 <= q <= 4.0:
            raise ValueError(f"path coordinate {q} outside [0, 4] (shell_mm)")
        if q <= 3.0:
            return segment(q)
        return 0.0, 4.0 - q, 2.0, 1.0
    if ref:
        if not 3.0 <= q <= 4.0:
            raise ValueError(f"reference leg runs on q in [3, 4], got {q}")
        return 0.0, 0.0, 2.0, q - 3.0
    if not 0.0 <= q <= 5.0:
        raise ValueError(f"path coordinate {q} outside [0, 5]")
    if q <= 1.0:
        return 1.0, q, 0.0, 0.0
    if q <= 2.0:
        return 2.0 - q, 1.0, q - 1.0, 0.0
    if q <= 3.0:
        return 0.0, 1.0, q - 1.0, 0.0
    if q <= 4.0:
        return 0.0, 1.0, 2.0, q - 3.0
    return 0.0, 5.0 - q, 2.0, 1.0


class MLMMForce:
    """openmm.PythonForce computation: anchored ML on L u I plus FIRES.

    The ML coordinates are taken from the full system and unwrapped around
    the ligand: the ligand by minimum image onto its first heavy atom, each
    inner water by minimum image of its oxygen onto the ligand centroid.
    """

    def __init__(self, model, numbers, lig_idx, inner_water_idx, fires, es=True,
                 device=None, cover=True, cap=CAP_M, shell_mm=False,
                 fast_forces=True, batch_fragments=False):
        self.lig = np.asarray(lig_idx)
        self.inner_w = np.asarray(inner_water_idx)                # [n, 3]
        self.ml_idx = np.concatenate([self.lig, self.inner_w.ravel()])
        frag = np.r_[np.zeros(len(self.lig), int), np.ones(self.inner_w.size, int)]
        from .lambda_mask import COVER_R
        self.anch = AnchoredForce(model, np.asarray(numbers)[self.ml_idx], frag, es=es,
                                  device=device, cover=COVER_R if cover else None, cap=cap,
                                  drop_env=shell_mm, fast_forces=fast_forces,
                                  batch_fragments=batch_fragments)
        self.shell_mm = shell_mm
        self.fires = fires
        self.anchor = int(self.lig[np.asarray(numbers)[self.lig] > 1][0])
        self.need_dudl = False
        self.last = {}
        self.t = 0.0

    def set_p(self, p):
        self.anch.set_lambda(p)

    def set_t(self, t):
        """Inner-water ML -> MM switch (segment SW); only used at p = 2."""
        self.t = float(t)

    def fragment_energies(self, X, grad=True):
        """(E_bb(L), E_bb(I)) in eV on the unwrapped ML region X (A), and the
        position gradient of E_bb(L) + (1 - t) E_bb(I) if grad. With grad and
        t = 1 (W3) E_bb(I) is not needed and comes back as nan."""
        from .lambda_mask import _fragment, fragment_energies
        af = self.anch
        data = af._graph(X)
        bb = af.model.backbone
        frags = ((af.ligand, af.charges[0]), (1 - af.ligand, af.charges[1]))
        if grad and self.t >= 1.0:
            frags = frags[:1]
        if len(frags) == 2:
            E = fragment_energies(bb, data, af.frag_id, af.ligand, af.charges,
                                  batch=af.batch_fragments)
        else:
            E = [bb(_fragment(data, af.frag_id, which, charge), training=False,
                    compute_force=False)["energy"].sum() for which, charge in frags]
        if not grad:
            return float(E[0]), float(E[1]), None
        if len(E) == 1:
            (g,) = torch.autograd.grad(E[0], data["positions"])
            return float(E[0]), float("nan"), g.detach().cpu().numpy()
        U = E[0] + (1.0 - self.t) * E[1]
        (g,) = torch.autograd.grad(U, data["positions"])
        return float(E[0]), float(E[1]), g.detach().cpu().numpy()

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
        X = self.ml_positions(pos, box)
        if self.anch.p >= 2.0 and self.t > 0.0:              # SW / W3: fragments only
            E_L, E_I, g = self.fragment_energies(X)
            U_ml = E_L if self.t >= 1.0 else E_L + (1.0 - self.t) * E_I    # E_I is nan (not computed) at t = 1
            F_ml = -g
        else:
            U_ml, F_ml = self.anch.energy_and_forces(X, need_dudl=self.need_dudl)
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


class FiresOnlyForce:
    """PythonForce computation for the all-MM path control (v3 S7.2): FIRES only.

    Same interface as MLMMForce where mlmm_hremd uses it (anch is None: no ML).
    """

    def __init__(self, fires, lig_idx, inner_water_idx):
        self.fires = fires
        self.lig = np.asarray(lig_idx)
        self.inner_w = np.asarray(inner_water_idx)
        self.ml_idx = np.concatenate([self.lig, self.inner_w.ravel()])
        self.anch = None
        self.need_dudl = False
        self.last = {}

    def set_p(self, p):
        pass

    def evaluate(self, pos, box):
        if self.fires is None:
            return 0.0, np.zeros_like(pos)
        U, F = self.fires.energy_and_forces(pos, box)
        self.last = dict(U_ml=0.0, U_fires=U, **self.fires.last)
        return U, F

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


def _nonbonded(nb, lig, inner_atoms, shell_mm=False):
    """NonbondedForce edits; returns the ligand's (q e, sigma nm, eps kJ/mol).
    shell_mm=True (D19) leaves the I-I pairs as ordinary MM pairs."""
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
    if shell_mm:
        I = []                                      # D19: I-I stays ordinary MM, no t_sw
    # I-I across waters: ML, but switched back on as MM by t_sw in segment SW (D18):
    # chargeProd and epsilon are t_sw x their MM values, sigma the MM value, so the
    # exception energy is linear in t_sw and equals the ordinary PME pair at t_sw = 1
    # (direct 1/r plus the reciprocal part the exclusion correction leaves).
    if I:
        nb.addGlobalParameter("t_sw", 0.0)
    prm = {i: nb.getParticleParameters(i) for i in I}
    for a, i in enumerate(I):                       # (intra-water pairs are already excluded)
        for j in I[a + 1:]:
            key = (min(i, j), max(i, j))
            if key in have:
                continue
            qi, si, ei = prm[i]
            qj, sj, ej = prm[j]
            qq = (qi * qj).value_in_unit(unit.elementary_charge ** 2)
            sg = 0.5 * (si + sj).value_in_unit(unit.nanometer)
            ee = np.sqrt(ei.value_in_unit(unit.kilojoule_per_mole) * ej.value_in_unit(unit.kilojoule_per_mole))
            k = nb.addException(key[0], key[1], 0.0, sg, 0.0)
            have[key] = k
            nb.addExceptionParameterOffset("t_sw", k, qq, 0.0, ee)
    for i in L:                                     # charges -> lambda_i offsets; LJ -> group 2
        nb.setParticleParameters(i, 0.0, sig[i], 0.0)
        nb.addGlobalParameter(f"lam_{i}", 1.0)
        nb.addParticleParameterOffset(f"lam_{i}", i, q[i], 0.0, 0.0)
    return q, sig, eps


def _nonbonded_mm(nb, lig):
    """All-MM control (v3 S7.2): ligand charges -> lambda_i offsets, ligand LJ -> group 2,
    and every ligand-ligand pair without an exception gets one with its full q_i q_j and
    LJ, so the ligand's own nonbonded terms do not depend on lambda (as
    lambda_frames.scaled_system and the acetamide abfe_mm.py). L-I and I-I stay
    ordinary MM pairs. Returns the ligand's (q e, sigma nm, eps kJ/mol)."""
    L = [int(i) for i in lig]
    q, sig, eps = {}, {}, {}
    for i in L:
        qi, si, ei = nb.getParticleParameters(i)
        q[i] = qi.value_in_unit(unit.elementary_charge)
        sig[i] = si.value_in_unit(unit.nanometer)
        eps[i] = ei.value_in_unit(unit.kilojoule_per_mole)
    have = {tuple(sorted(nb.getExceptionParameters(k)[:2])) for k in range(nb.getNumExceptions())}
    for a, i in enumerate(L):
        for j in L[a + 1:]:
            if (i, j) not in have:
                nb.addException(i, j, q[i] * q[j], 0.5 * (sig[i] + sig[j]), np.sqrt(eps[i] * eps[j]))
    for i in L:
        nb.setParticleParameters(i, 0.0, sig[i], 0.0)
        nb.addGlobalParameter(f"lam_{i}", 1.0)
        nb.addParticleParameterOffset(f"lam_{i}", i, q[i], 0.0, 0.0)
    return q, sig, eps


def _ligand_outer_lj(system, nb, lig, outer_atoms, sig, eps, dispersion_correction):
    """L-O Lennard-Jones, linear in the global lam_lj, NonbondedForce's cutoff.

    r is floored at 1e-3 nm (0.01 A): at lam_lj = 0 (W3, the decoupled end) a
    water atom can sit on a ligand atom, and below ~1e-4 nm (s/r)^12 overflows
    float32 so that 0 x inf = NaN (`experiments/2026-10-16_mm_path_control/
    overlap_nan.py`; the cause of the all-MM control's NaNs at q 3.99 / 4,
    S13.48). For lam_lj > 0 the wall keeps every pair far above the floor."""
    f = openmm.CustomNonbondedForce(
        "lam_lj*4*e*((s/rr)^12-(s/rr)^6); rr=max(r, 0.001); s=0.5*(s1+s2); e=sqrt(e1*e2)")
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


def _wall(system, lig, water_atoms, numbers, eps_w, alpha, ions=(), lig_sigma=None, nb=None):
    """U = s_w eps_w g(r_eff / r0), g(x) = x^-12 - 2 x^-6 + 1 for x < 1 (WCA-shifted),
    r_eff = (alpha (1 - s_w) r0^6 + r^6)^(1/6) (softcore, only felt for s_w < 1).
    ions: non-ligand, non-water atoms, walled per ION_WALL_SCALE (lig_sigma {i: nm}, nb for ion sigma)."""
    els = sorted(WALL_R0)
    t_of = {z: k for k, z in enumerate(els)}            # ligand types 0..3, water O 4, water H 5, ions 6..
    ions = [int(i) for i in ions]
    ion_els = sorted(set(int(numbers[i]) for i in ions))
    ion_set = set(ions)
    nt = len(els) + 2 + len(ion_els)
    tab = np.zeros((nt, nt))
    for z, (rO, rH) in WALL_R0.items():
        for a, b, r in ((t_of[z], len(els), rO), (t_of[z], len(els) + 1, rH)):
            tab[a, b] = tab[b, a] = r / 10.0                 # nm
    ion_type = {}
    for k, zi in enumerate(ion_els):
        ion_type[zi] = len(els) + 2 + k
        first = next(i for i in ions if int(numbers[i]) == zi)
        s_ion = nb.getParticleParameters(first)[1].value_in_unit(unit.nanometer)
        for z in els:
            members = [i for i in lig if int(numbers[i]) == z]
            if members:
                s_lig = min(lig_sigma[int(i)] for i in members)
                tab[t_of[z], ion_type[zi]] = tab[ion_type[zi], t_of[z]] = \
                    ION_WALL_SCALE * 0.5 * (s_lig + s_ion)
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
            f.addParticle([float(len(els) if z == 8 else len(els) + 1)])
        elif i in ion_set:
            f.addParticle([float(ion_type[z])])
        else:
            f.addParticle([0.0])
    f.addInteractionGroup(sorted(L), sorted(Wset | ion_set))
    f.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    f.setCutoffDistance(float(tab.max()) + 0.01)           # = the water-only value when no ions
    f.setForceGroup(G_WALL)
    return f


def build_mlmm_system(leg_npz, model, n_inner=20, eps_w=25.0, alpha=0.5, es=True,
                      device=None, dispersion_correction=True, positions=None, cover=True,
                      cap=None, mm_only=False, use_fires=True, fires_center=None, shell_mm=False,
                      polar_shell=False, fast_forces=True, batch_fragments=False):
    """The ML/MM solvent leg of S6.3.

    mm_only=True builds the all-MM path control of S7.2 on the same path:
    ligand bonded terms kept, the ligand's own nonbonded terms kept at full
    strength by exceptions, L-I treated like L-O (Coulomb by the lambda_i
    offsets, LJ in the lam_lj group, which then covers every non-ligand atom),
    I-I ordinary MM, and the PythonForce carries FIRES only (n_inner MM waters,
    so relabelling is a symmetry and FIRES is exact). model may be None.
    use_fires=False (mm_only only) leaves FIRES out: no PythonForce at all.
    fires_center="point" orders the waters by distance to the ligand's first heavy atom
    only (the dG_ref leg of D18: a droplet around a point, not around the ghost's shape).
    shell_mm=True is D19 option 3 (v3 S6.3.4a): ML carries only the ligand and its coupling,
    the I-I pairs and the inner waters' own energy are MM; path q in [0, 4] (`segment`).
    polar_shell=True is the option-2 candidate with POLAR water retained and
    a shape-to-point FIRES bridge. fast_forces selects equivalent scalar and
    endpoint propagation; batch_fragments combines independent fragment graphs.

    leg_npz: an em_solvent.npz-style file (positions A, numbers, resnames,
    resids, box A, source = the prepared system XML). positions overrides the
    starting frame (it also picks the inner waters). eps_w in kJ/mol (to be
    set from the lambda = 0 smoke, S6.3.3). Returns (system, ml_force, info).
    """
    if polar_shell and (shell_mm or mm_only or not use_fires or fires_center is not None):
        raise ValueError("POLAR-shell path requires ML and shape FIRES, without shell_mm or ref")
    d = np.load(leg_npz, allow_pickle=True)
    system = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
    pos = np.asarray(d["positions"] if positions is None else positions, dtype=np.float64)
    numbers, box = np.asarray(d["numbers"]), np.asarray(d["box"], dtype=np.float64)
    lig = np.where(d["resids"] == d["resids"][0])[0]
    W = water_indices(d["resnames"], d["resids"])
    centre, wsel = heavy_selection(lig, numbers, W)
    if fires_center == "point":
        centre = np.asarray(centre)[:1]
    restraint = ShapeFires if polar_shell else Fires
    fires = restraint(centre, wsel, n_inner=n_inner, device=device or "cpu")
    inner = fires.select_inner(pos, box)
    outer = np.setdiff1d(np.arange(len(W)), inner)
    inner_atoms, outer_atoms = W[inner].ravel(), W[outer].ravel()

    if not mm_only:
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
    if mm_only:
        q, sig, eps = _nonbonded_mm(nb, lig)
        lj_env = np.setdiff1d(np.arange(system.getNumParticles()), lig)
    else:
        q, sig, eps = _nonbonded(nb, lig, inner_atoms, shell_mm=shell_mm)
        # every MM atom outside the ML region: outer waters and any ions (the IBS legs
        # carry NaCl; with waters only, ligand-ion LJ was silently dropped)
        lj_env = np.setdiff1d(np.arange(system.getNumParticles()),
                              np.concatenate([lig, inner_atoms]))
    system.addForce(_ligand_outer_lj(system, nb, lig, lj_env, sig, eps,
                                     dispersion_correction))
    ions = np.setdiff1d(np.arange(system.getNumParticles()), np.concatenate([lig, W.ravel()]))
    system.addForce(_wall(system, lig, W.ravel(), numbers, eps_w, alpha, ions=ions, lig_sigma=sig, nb=nb))
    if mm_only:
        mlf = FiresOnlyForce(fires if use_fires else None, lig, W[inner])
    else:
        mlf = MLMMForce(model, numbers, lig, W[inner], fires, es=es, device=device, cover=cover,
                        cap=CAP_M if cap is None else cap, shell_mm=shell_mm,
                        fast_forces=fast_forces, batch_fragments=batch_fragments)
    if not (mm_only and not use_fires):
        pf = openmm.PythonForce(mlf)
        pf.setUsesPeriodicBoundaryConditions(True)
        pf.setForceGroup(G_ML)
        system.addForce(pf)
    info = dict(lig=lig, water=W, inner=inner, inner_atoms=inner_atoms,
                outer_atoms=outer_atoms, lj_env=lj_env, q=q, box=box, numbers=numbers,
                positions=pos, mm_only=mm_only, shell_mm=shell_mm, polar_shell=polar_shell)
    return system, mlf, info


def steepest_descent(context, max_iter=1000, tol_kj_mol_nm=10.0, step_nm=0.002, max_step_nm=0.02):
    """Steepest descent on the full Context energy, step capped by the largest atomic displacement,
    constraints re-applied after each step. OpenMM's LocalEnergyMinimizer moves to the CPU platform
    at large forces, which rejects these systems (non-identical exclusions across nonbonded forces).
    Returns (U before, U after) in kJ/mol."""
    state = context.getState(getPositions=True, getEnergy=True, getForces=True)
    x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    U = U0 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    F = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    for _ in range(max_iter):
        fmax = np.linalg.norm(F, axis=1).max()
        if fmax < tol_kj_mol_nm:
            break
        context.setPositions(x + F * (step_nm / fmax))
        context.applyConstraints(1e-6)
        trial = context.getState(getPositions=True, getEnergy=True, getForces=True)
        U_new = trial.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if np.isfinite(U_new) and U_new < U:
            x = trial.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            U, F = U_new, trial.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
            step_nm = min(step_nm * 1.2, max_step_nm)
        else:
            step_nm *= 0.5
            if step_nm < 1e-7:
                break
    context.setPositions(x)
    return U0, U


def set_path(ctx, mlf, q, lig, ref=False, shell_mm=False, polar_shell=False):
    """Put context + ML force at path coordinate q (S6.3.4). Returns (lambda, s_w, p, t)."""
    lam, s_w, p, t = segment(q, ref=ref, shell_mm=shell_mm, polar_shell=polar_shell)
    for i in lig:
        ctx.setParameter(f"lam_{int(i)}", lam)
    ctx.setParameter("lam_lj", lam)
    ctx.setParameter("s_w", s_w)
    if "t_sw" in ctx.getParameters():            # absent in the all-MM control: SW is a no-op there
        ctx.setParameter("t_sw", t)
    mlf.set_p(p)
    if hasattr(mlf, "set_t"):
        mlf.set_t(t)
    if polar_shell:
        mlf.fires.set_shape(max(0.0, float(q) - 4.0))
    return lam, s_w, p, t


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
