"""Exact checks E1, E2, E3, E5, E6 of decomp/mlmm.py (RESEARCH_PLAN_v3.md S6.3.5).

Run: python check_mlmm.py [HEAD_CKPT]

MM parts on the Reference platform (double precision); ML in float64 on the
CPU. em_solvent.npz (the Atenolol solvent leg), n = 20 inner waters.

  E1  partition: at lambda = 1, s_w = 0, with dispersion correction off on both
      sides, MM(ours) = MM(ref) - MM_ref(A only) - bonded_ref(L), A = L u I.
      Exceptions remove A-A pairs at minimum image only, so the A-A
      interactions with periodic images stay in ours; their size is estimated
      as MM_ref(A only, 35 A box) - MM_ref(A only, L -> infinity), the limit
      extrapolated from 70 and 140 A boxes as E_inf + c / L^3 (a neutral,
      dipolar A). Run with a tight Ewald tolerance (argument 2): at the
      default 5e-4 the large-box PME error swamps the estimate.
  E2  far field: the NonbondedForce energy is linear in lambda; the per-atom
      sum of single-lambda_i energies equals the total; mm_far_field_dudl
      equals a central finite difference of the MM energy in lambda.
  E3  ML inside the full system: unwrapped L u I distances equal minimum-image
      distances; the PythonForce energy at p = 0 equals E_bb(L u I) and at
      p = 2 equals E_bb(L) + E_bb(I), each computed directly.
  E5  wall on two particles: zero above r0; force continuous at r0 (neighbour
      step halves when the step halves); softcore finite at r -> 0 for s_w < 1.
  E6  PBC: total energy (all groups, ML included) unchanged when the whole
      system, one inner water, or the ligand is shifted by a box vector.
"""

import copy
import sys
import time

import numpy as np
import openmm
import torch
from openmm import unit

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm  # noqa: E402
from decomp.mace_adapter import MACEDecomposition  # noqa: E402

E3D = "/home/ruigengji/E3D"
LEG = f"{E3D}/em_solvent.npz"
BB = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"
CKPT = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else
        f"{E3D}/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt")
EWALD_TOL = float(sys.argv[2]) if len(sys.argv) > 2 else None    # E1 only: tighten PME on every side
torch.set_num_threads(8)


def load_model64():
    bb = torch.load(BB, map_location="cpu", weights_only=False).to(torch.float64).eval()
    m = MACEDecomposition(bb, hidden=256, n_pair_channels=4)
    m.head.load_state_dict(torch.load(CKPT, map_location="cpu"))
    m = m.to(torch.float64).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def ctx_of(system, pos, box, platform="Reference"):
    integ = openmm.VerletIntegrator(1e-4)
    c = openmm.Context(system, integ, openmm.Platform.getPlatformByName(platform))
    c.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])
    c.setPositions(pos * unit.angstrom)
    return c, integ


def energy(c, groups=None):
    kw = {} if groups is None else {"groups": set(groups)}
    return c.getState(getEnergy=True, **kw).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


t0 = time.perf_counter()
d = np.load(LEG, allow_pickle=True)
model = load_model64()
system, mlf, info = mlmm.build_mlmm_system(LEG, model, n_inner=20, es=True, device="cpu",
                                           dispersion_correction=False)
pos, box, lig, num = info["positions"], info["box"], info["lig"], info["numbers"]
if EWALD_TOL:
    [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0].setEwaldErrorTolerance(EWALD_TOL)
A = np.concatenate([lig, info["inner_atoms"]])
print(f"built: {system.getNumParticles()} atoms, ligand {len(lig)}, inner atoms "
      f"{len(info['inner_atoms'])}, forces {[type(f).__name__ for f in system.getForces()]} "
      f"[{time.perf_counter() - t0:.0f}s]", flush=True)

# ---------- E1
ref = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
nbr = [f for f in ref.getForces() if isinstance(f, openmm.NonbondedForce)][0]
nbr.setUseDispersionCorrection(False)
if EWALD_TOL:
    nbr.setEwaldErrorTolerance(EWALD_TOL)
for f in ref.getForces():
    f.setForceGroup(1 if isinstance(f, openmm.NonbondedForce) else 0)
cr, _ = ctx_of(ref, pos, box)
E_ref_nb, E_ref_bonded = energy(cr, [1]), energy(cr, [0])


def a_only(box_ang):
    s = copy.deepcopy(ref)
    nb = [f for f in s.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    keep = set(A.tolist())
    for i in range(s.getNumParticles()):
        if i not in keep:
            _, sg, _ = nb.getParticleParameters(i)
            nb.setParticleParameters(i, 0.0, sg, 0.0)
    c, _ = ctx_of(s, pos, box_ang)
    return energy(c, [1])


E_A35, E_A70, E_A140 = a_only(box), a_only(2 * box), a_only(4 * box)
E_Ainf = E_A140 - (E_A70 - E_A140) / 7                      # E(L) = E_inf + c / L^3, from 70 and 140 A
co, _ = ctx_of(system, pos, box)                           # MM groups only: the PythonForce (group 4) is not evaluated
for i in lig:
    co.setParameter(f"lam_{int(i)}", 1.0)
co.setParameter("lam_lj", 1.0)
co.setParameter("s_w", 0.0)
E_ours = energy(co, [0, 1, 2])
expect = E_ref_nb + E_ref_bonded - E_A35 - E_ref_bonded


def removed_pairs():
    """What the exceptions remove, summed pair by pair in numpy at minimum image:
    for an A-A pair with a ref exception, that exception's Coulomb + LJ; for any
    other A-A pair, the full Coulomb q_i q_j / r plus LJ inside the cutoff."""
    K = 138.935456                                        # kJ/mol nm / e^2
    rc = nbr.getCutoffDistance().value_in_unit(unit.nanometer)
    exc = {}
    for k in range(nbr.getNumExceptions()):
        i, j, qq, sg, ep = nbr.getExceptionParameters(k)
        exc[(min(i, j), max(i, j))] = (qq.value_in_unit(unit.elementary_charge ** 2),
                                       sg.value_in_unit(unit.nanometer),
                                       ep.value_in_unit(unit.kilojoule_per_mole))
    par = [nbr.getParticleParameters(int(i)) for i in A]
    qv = np.array([x[0].value_in_unit(unit.elementary_charge) for x in par])
    sv = np.array([x[1].value_in_unit(unit.nanometer) for x in par])
    ev = np.array([x[2].value_in_unit(unit.kilojoule_per_mole) for x in par])
    X = pos[A] / 10
    bnm = box / 10
    tot = 0.0
    for a_ in range(len(A)):
        dv = X[a_ + 1:] - X[a_]
        dv -= bnm * np.round(dv / bnm)
        r = np.sqrt((dv ** 2).sum(-1))
        for b_, rr in enumerate(r, start=a_ + 1):
            key = (min(int(A[a_]), int(A[b_])), max(int(A[a_]), int(A[b_])))
            if key in exc:
                qq, sg, ep = exc[key]
            else:
                qq, sg, ep = qv[a_] * qv[b_], 0.5 * (sv[a_] + sv[b_]), np.sqrt(ev[a_] * ev[b_])
                if rr >= rc:
                    ep = 0.0
            tot += K * qq / rr + 4 * ep * ((sg / rr) ** 12 - (sg / rr) ** 6)
    return tot


E_removed = removed_pairs()
expect2 = E_ref_nb - E_removed
img = E_A35 - E_Ainf
print(f"E1 exact: MM(ours) {E_ours:.4f} vs MM(ref) - (A-A pairs removed, numpy at minimum image) "
      f"{expect2:.4f} kJ/mol: diff {E_ours - expect2:+.2e} (removed {E_removed:.4f})", flush=True)
print(f"E1 MM(ours) {E_ours:.4f}  vs  MM(ref) - MM_ref(A) - bonded_ref(L) {expect:.4f} kJ/mol: "
      f"diff {E_ours - expect:+.4f}; A-A image term estimate {img:+.4f} "
      f"(diff - image {E_ours - expect - img:+.4f}; A-only at 35/70/140 A {E_A35:.4f}/{E_A70:.4f}/{E_A140:.4f}); ligand bonded removed {E_ref_bonded:.2f}; "
      f"wall at s_w = 0: {energy(co, [3]):.2e}", flush=True)

# ---------- E2
lams = np.linspace(0, 1, 5)
E_l = []
for lam in lams:
    for i in lig:
        co.setParameter(f"lam_{int(i)}", lam)
    E_l.append(energy(co, [1]))
E_l = np.array(E_l)
lin = E_l - (E_l[0] + lams * (E_l[-1] - E_l[0]))
lam = 0.6
for i in lig:
    co.setParameter(f"lam_{int(i)}", lam)
per = mlmm.mm_far_field_per_atom_coulomb(co, lig, lam)
co.setParameter("lam_lj", lam)
d_lj = co.getState(getParameterDerivatives=True, groups={2}).getEnergyParameterDerivatives()["lam_lj"]
dudl = mlmm.mm_far_field_dudl(co, lig, lam)
h = 1e-3


def mm_at(x):
    for i in lig:
        co.setParameter(f"lam_{int(i)}", x)
    co.setParameter("lam_lj", x)
    return energy(co, [1, 2])


fd = (mm_at(lam + h) - mm_at(lam - h)) / (2 * h)
print(f"E2 NonbondedForce vs lambda (0..1, 5 points): max deviation from linear {np.abs(lin).max():.2e} "
      f"kJ/mol (L-O Coulomb at lambda = 1: {E_l[-1] - E_l[0]:.3f}); per-atom Coulomb sum + LJ "
      f"{per.sum() + d_lj:.4f} vs dudl (diff {per.sum() + d_lj - dudl:+.2e}); "
      f"mm_far_field_dudl {dudl:.4f} vs FD {fd:.4f} (diff {dudl - fd:+.2e})", flush=True)

# ---------- E3
P = mlf.ml_positions(pos, box)
dv = pos[A][:, None] - pos[A][None]
dv -= box * np.round(dv / box)
r_mi = np.sqrt((dv ** 2).sum(-1))
r_ml = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
close = r_mi < box.min() / 2 - 8
print(f"E3 unwrapped L u I: max |r - r_min_image| over {int(close.sum())} pairs {np.abs(r_ml - r_mi)[close].max():.2e} A; "
      f"ML extent {np.ptp(P, axis=0).round(1)} A", flush=True)
from decomp.lambda_mask import _fragment  # noqa: E402
af = mlf.anch
for p_, label in ((0.0, "p = 0 vs E_bb(L u I)"), (2.0, "p = 2 vs E_bb(L) + E_bb(I)")):
    mlf.set_p(p_)
    U, _ = af.energy_and_forces(P)
    data = af._graph(P)
    if p_ == 0.0:
        ref_E = float(model.backbone(data, training=False, compute_force=False)["energy"].sum())
    else:
        ref_E = sum(float(model.backbone(_fragment(data, af.frag_id, k, 0), training=False,
                                         compute_force=False)["energy"].sum()) for k in (0, 1))
    print(f"   {label}: {U:.8f} vs {ref_E:.8f} eV, |d| {abs(U - ref_E):.1e}", flush=True)

# ---------- E5
s2 = openmm.System()
for m in (16.0, 1.0):
    s2.addParticle(m)
s2.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * 3.0)) for i in range(3)])
wf = mlmm._wall(s2, [0], [1], np.array([8, 1]), eps_w=25.0, alpha=0.5)
wf.setForceGroup(0)
s2.addForce(wf)
r0 = mlmm.WALL_R0[8][1]
c2 = openmm.Context(s2, openmm.VerletIntegrator(1e-4), openmm.Platform.getPlatformByName("Reference"))


def wall_at(r, s_w):
    c2.setParameter("s_w", s_w)
    c2.setPositions([openmm.Vec3(1, 1, 1), openmm.Vec3(1 + r / 10, 1, 1)] * unit.nanometer)
    st = c2.getState(getEnergy=True, getForces=True)
    return (st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)[1, 0] / 10)


above = max(abs(wall_at(r, 1.0)[0]) for r in np.linspace(r0 + 1e-6, r0 + 1.0, 50))
steps = []
for npt in (201, 401):
    F_ = [wall_at(r, 1.0)[1] for r in np.linspace(r0 - 0.05, r0 + 0.05, npt)]
    steps.append(np.abs(np.diff(F_)).max())
soft = [wall_at(1e-4, s)[0] for s in (0.0, 0.5, 0.9, 0.99)]
print(f"E5 wall (ligand O - water H, r0 {r0} A): max |U| above r0 {above:.1e}; force step at 201/401 points "
      f"{steps[0]:.3f}/{steps[1]:.3f} kJ/mol/A (ratio {steps[0] / steps[1]:.2f}); "
      f"U at r = 1e-4 A for s_w 0, 0.5, 0.9, 0.99: {', '.join(f'{u:.3g}' for u in soft)} kJ/mol; "
      f"at s_w = 1, r = 0.8 r0: {wall_at(0.8 * r0, 1.0)[0]:.2f} (= eps_w g(0.8) = {25 * 7.92:.1f})", flush=True)

# ---------- E6
mlf.set_p(0.5)
cf, _ = ctx_of(system, pos, box)
for i in lig:
    cf.setParameter(f"lam_{int(i)}", 0.5)
cf.setParameter("lam_lj", 0.5)
cf.setParameter("s_w", 1.0)
E_base = energy(cf)
res = []
w0 = info["water"][info["inner"][0]]
for label, shift in (("whole system", np.ones((len(pos), 1)) * box * np.array([1, -1, 2])),
                     ("one inner water", np.zeros_like(pos)),
                     ("ligand", np.zeros_like(pos))):
    if label == "one inner water":
        shift[w0] = box * np.array([0, 1, 0])
    if label == "ligand":
        shift[lig] = box * np.array([-1, 0, 0])
    cf.setPositions((pos + shift) * unit.angstrom)
    res.append(f"{label} {abs(energy(cf) - E_base):.1e}")
print(f"E6 total energy at q = 1.5 (ML, FIRES, wall, MM) {E_base:.3f} kJ/mol; |dU| after a box-vector "
      f"shift of: {'; '.join(res)} kJ/mol [{time.perf_counter() - t0:.0f}s]", flush=True)
