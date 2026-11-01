"""HREMD over the full ML/MM path of RESEARCH_PLAN_v3.md S6.3.4 (`run_abfe.py --mode mlmm`).

One replica per path state q in [0, 4] (W1 | W2 | cut | W3, `mlmm.segment`), each
its own OpenMM System/Context built by `mlmm.build_mlmm_system` from the same
start frame, so every replica has the same inner waters (FIRES keeps them) and
the Hamiltonians differ only in q. Neighbour swaps exchange positions and
velocities.

**All-state energies from one ML forward.** For a configuration R, with
E_bb, the capped coupling S_eff = sum_i c_i and the fragment sum U0 (all
q-independent; `lambda_mask.anchored_energy`, cut branch at lambda = 0):

    U_ML(q) = E_bb                         W1 (lambda = 1)
            = E_bb - (q - 1) S_eff          W2 (lambda = 2 - q)
            = (1 - s)(E_bb - S_eff) + s U0  cut (s = q - 2)
            = U0                            W3

and the MM groups are cheap: the NonbondedForce is a quadratic in the
common ligand lambda (three PME evaluations fix it), the L-O LJ is lam_lj x
its lambda = 1 value, the wall is one group-3 evaluation per distinct s_w.
FIRES and the bonded group do not depend on q. So a whole u_kn row costs one
ML forward (no backward) plus MM, and the exchange uses the same rows.

**Derivatives for TI** (kJ/mol per unit q), per segment, so a state on a
segment boundary carries both one-sided values:

    W1   dU/dq = dU_wall/ds_w                    (s_w = q)
    W2   dU/dq = -(S_eff + dE_NB/dlambda + E_LJ(lambda=1))
    cut  dU/dq = U0 - (E_bb - S_eff)
    W3   dU/dq = -dU_wall/ds_w                   (s_w = 4 - q)

Per ligand atom (the attribution, S6.3.4): on W2 the ML c_i, the MM Coulomb
dE_NB/dlambda_i (exact central differences, `mlmm.mm_far_field_per_atom_coulomb`)
and the L-O LJ pair sum of atom i; on W1/W3 the wall's dU/ds_w of atom i.
The LJ and wall per-atom sums are computed in torch from the same pair terms
as OpenMM; the LJ long-range correction (a constant at fixed volume) is the
difference between OpenMM's parameter derivative and the pair sums, and is
split over ligand atoms in proportion to their sum of 4 eps sigma^6 (a
convention; the total is OpenMM's). Every per-atom set is checked against its
total in `check_mlmm_hremd.py`.

**D19 path** (shell_mm=True, user 2026-10-18; v3 S6.3.4a): q in [0, 4],
W1 | W2 | cut | W3, no SW. E_bb and U0 are then the reduced ones of
`anchored_energy(drop_env=True)` (E_bb - E_bb(I) and E_bb(L)), so the same
formulas hold with W3 at U_ML = E_bb(L) and s_w = 4 - q.

The option-2 candidate (polar_shell=True) keeps E_bb(I) through W3 at q in
[3, 4], then bridges shape FIRES to point FIRES at q in [4, 5]. The shape
work is global and the endpoint is a POLAR droplet in MM solvent, not bulk MM.

The evaluator runs OpenMM at double precision (MM only; the replicas run
the platform default). Checkpoint every iteration (`state.npz` +
`records.json` in out_dir): rerunning with the same out_dir resumes.
"""

import json
import os
import time

import numpy as np
import openmm
import torch
from openmm import unit

from . import mlmm
from .lambda_mask import anchored_energy

KB = 0.008314462618
EV = 96.48533212
# SW (inner waters ML -> MM, D18) inserted before W3 on 2026-10-17 (user): q in [0, 5]
SEGMENTS = (("W1", 0.0, 1.0), ("W2", 1.0, 2.0), ("cut", 2.0, 3.0), ("SW", 3.0, 4.0), ("W3", 4.0, 5.0))
# D19 option 3 (user 2026-10-18): the shell is MM water, no SW
SEGMENTS_D19 = (("W1", 0.0, 1.0), ("W2", 1.0, 2.0), ("cut", 2.0, 3.0), ("W3", 3.0, 4.0))
SEGMENTS_POLAR = SEGMENTS_D19 + (("shape", 4.0, 5.0),)


def _segs(shell_mm, polar_shell=False):
    if polar_shell:
        if shell_mm:
            raise ValueError("POLAR-shell and shell_mm paths are mutually exclusive")
        return SEGMENTS_POLAR
    return SEGMENTS_D19 if shell_mm else SEGMENTS


def path_states(n_w1=2, n_w2=12, n_cut=8, n_sw=6, n_w3=6, shell_mm=False,
                polar_shell=False, n_shape=6):
    """q values: n points per segment including both ends (shared ends once); n_sw unused with shell_mm."""
    ns = ((n_w1, n_w2, n_cut, n_w3, n_shape) if polar_shell else
          (n_w1, n_w2, n_cut, n_w3) if shell_mm else (n_w1, n_w2, n_cut, n_sw, n_w3))
    qs = []
    for (name, a, b), n in zip(_segs(shell_mm, polar_shell), ns):
        pts = np.linspace(a, b, max(n, 2))
        qs.extend(pts if not qs else pts[1:])
    return np.round(np.array(qs), 6)


def segments_of(q, shell_mm=False, polar_shell=False):
    return [name for name, a, b in _segs(shell_mm, polar_shell) if a - 1e-9 <= q <= b + 1e-9]


class Evaluator:
    """Energies at every state and derivatives at one state for a configuration."""

    def __init__(self, system, mlf, info, qs, platform, properties=None, ref=False):
        self.mlf, self.info, self.qs = mlf, info, np.asarray(qs, float)
        self.ref = ref
        self.shell_mm = bool(info.get("shell_mm", False))
        self.polar_shell = bool(info.get("polar_shell", False))
        self.lig = info["lig"]
        self.nl = len(self.lig)
        self.box = np.asarray(info["box"], float)
        self.ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), platform, properties or {})
        self.ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * self.box[i] / 10)) for i in range(3)])
        self.kT = None
        af = mlf.anch
        self.af = af                                  # None in the all-MM control
        dev = af.device if af is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        # torch pair data for the per-atom LJ (ligand x outer) and wall (ligand x all water)
        lj = [f for f in system.getForces() if f.getForceGroup() == mlmm.G_LJ][0]
        wall = [f for f in system.getForces() if f.getForceGroup() == mlmm.G_WALL][0]
        prm = np.array([lj.getParticleParameters(i) for i in range(system.getNumParticles())])  # sigma nm, eps
        self.outer = np.asarray(info.get("lj_env", info["outer_atoms"]))
        self.water = np.asarray(info["water"]).ravel()
        t = lambda x: torch.as_tensor(x, dtype=torch.float64, device=dev)  # noqa: E731
        sl, el = prm[self.lig, 0], prm[self.lig, 1]
        so, eo = prm[self.outer, 0], prm[self.outer, 1]
        self.lj_sig = t(0.5 * (sl[:, None] + so[None]) * 10.0)              # A
        self.lj_eps = t(np.sqrt(el[:, None] * eo[None]))                    # kJ/mol
        self.lj_rc = lj.getCutoffDistance().value_in_unit(unit.angstrom)
        c6 = (4 * self.lj_eps * self.lj_sig ** 6).sum(1)
        self.lrc_w = (c6 / c6.sum()).cpu().numpy()
        num = info["numbers"]
        r0 = np.array([[mlmm.WALL_R0[int(z)][0 if int(num[w]) == 8 else 1] for w in self.water]
                       for z in num[self.lig]])
        self.w_r0 = t(r0)
        self.w_eps = float(self.ctx.getParameter("eps_w"))
        self.w_alpha = float(self.ctx.getParameter("alpha_w"))
        self.w_rc = wall.getCutoffDistance().value_in_unit(unit.angstrom)
        self.dev = dev

    # ------------------------------------------------------------------ pieces

    def _group(self, g):
        return self.ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    def _set_lam(self, lam):
        for i in self.lig:
            self.ctx.setParameter(f"lam_{int(i)}", lam)

    def ml_parts(self, P):
        """(E_bb, S_eff, U0, c_i, E_bb(L), E_bb(I)) in kJ/mol at configuration P (A).
        With D19 (drop_env) E_bb and U0 are E_bb - E_bb(I) and E_bb(L)."""
        af = self.af
        if af is None:                                # all-MM control: no ML terms
            return 0.0, 0.0, 0.0, np.zeros(self.nl), 0.0, 0.0
        with torch.no_grad():
            data = af._graph(self.mlf.ml_positions(P, self.box))
            lam = torch.zeros_like(af.lam)
            # the cut branch computes both fragments, E_L and E_E, in the same pass
            _, c, parts = anchored_energy(af.model, data, lam, af.frag_id, ligand=af.ligand, es=af.es,
                                          cut=0.5, charges=af.charges, cover=af.cover, cap=af.cap,
                                          drop_env=af.drop_env, batch_fragments=af.batch_fragments)
        E_L, E_I = float(parts["E_L"]), float(parts["E_E"])
        E_bb, U0 = float(parts["E_bb"]) * EV, float(parts["U0"]) * EV
        # S from sum_c, not E_bb - U_plus: in float32 that difference of two
        # ~6.6e4 eV numbers resolves only 0.75 kJ/mol (smoke closure -0.2 kJ/mol)
        S = float(parts["sum_c"]) * EV
        return E_bb, S, U0, c[: self.nl].double().cpu().numpy() * EV, E_L * EV, E_I * EV

    def _pairs(self, P, idx_env):
        d = torch.as_tensor(P[idx_env][None] - P[self.lig][:, None], dtype=torch.float64, device=self.dev)
        b = torch.as_tensor(self.box, dtype=torch.float64, device=self.dev)
        d = d - b * torch.round(d / b)
        return d.norm(dim=-1)

    def lj_per_atom(self, P):
        """Per-ligand-atom L-O LJ at lam_lj = 1 (pair sums within the cutoff), kJ/mol."""
        r = self._pairs(P, self.outer)
        x = (self.lj_sig / r) ** 6
        e = torch.where(r < self.lj_rc, 4 * self.lj_eps * (x * x - x), torch.zeros_like(r))
        return e.sum(1).cpu().numpy()

    def wall_dsw_per_atom(self, P, s_w):
        """Per-ligand-atom dU_wall/ds_w at s_w (same form as mlmm._wall), kJ/mol."""
        r = self._pairs(P, self.water)
        s = torch.full((self.nl, 1), float(s_w), dtype=torch.float64, device=self.dev, requires_grad=True)
        reff = (self.w_alpha * (1 - s) * self.w_r0 ** 6 + r ** 6) ** (1 / 6)
        y = self.w_r0 / reff
        g = torch.where((reff < self.w_r0) & (r < self.w_rc), y ** 12 - 2 * y ** 6 + 1, torch.zeros_like(r))
        (d,) = torch.autograd.grad((s * self.w_eps * g).sum(), s)     # one s per ligand atom
        return d[:, 0].detach().cpu().numpy()

    # ------------------------------------------------------------------ public

    def all_states(self, P):
        """U(q_k) for every state k, kJ/mol (q-independent groups included), plus the parts."""
        self.ctx.setPositions(P * unit.angstrom)
        if "t_sw" in self.ctx.getParameters():
            self.ctx.setParameter("t_sw", 0.0)
        E_bb, S, U0, c, E_L, E_I = self.ml_parts(P)
        fz = self.mlf.fires
        if fz is None:
            E_f = 0.0
        else:
            with torch.no_grad():
                E_f = float(fz.energy(fz._t(P), fz._t(self.box)))
        shape_delta = 0.0
        if self.polar_shell:
            with torch.no_grad():
                shape_energy, point_energy = fz.shape_energies(fz._t(P), fz._t(self.box))
            E_f = float(shape_energy)
            shape_delta = float(point_energy - shape_energy)
        g0 = self._group(0)
        nb = {}
        for lam in (0.0, 0.5, 1.0):
            self._set_lam(lam)
            nb[lam] = self._group(mlmm.G_NB)
        a = nb[0.0]
        cq = 2 * (nb[1.0] + nb[0.0] - 2 * nb[0.5])
        b = nb[1.0] - a - cq
        has_t = "t_sw" in self.ctx.getParameters()
        if has_t:                                     # NB at lambda 0 with the I-I MM pairs on
            self._set_lam(0.0)
            self.ctx.setParameter("t_sw", 1.0)
            nb_t1 = self._group(mlmm.G_NB)
            self.ctx.setParameter("t_sw", 0.0)
        else:
            nb_t1 = a
        self.ctx.setParameter("lam_lj", 1.0)
        lj1 = self._group(mlmm.G_LJ)
        walls = {}
        U = np.zeros(len(self.qs))
        for k, q in enumerate(self.qs):
            lam, s_w, _, t = mlmm.segment(q, ref=self.ref, shell_mm=self.shell_mm,
                                          polar_shell=self.polar_shell)
            if s_w not in walls:
                self.ctx.setParameter("s_w", s_w)
                walls[s_w] = self._group(mlmm.G_WALL)
            if q <= 1.0:
                ml = E_bb
            elif q <= 2.0:
                ml = E_bb - (q - 1.0) * S
            elif q <= 3.0:
                s = q - 2.0
                ml = (1 - s) * (E_bb - S) + s * U0
            else:                                     # SW and W3 (t = 1; D19: W3 only)
                ml = E_L + (1 - t) * E_I if self.af is not None else 0.0
            nb_q = a + b * lam + cq * lam ** 2 + t * (nb_t1 - a)
            restraint = E_f + max(0.0, q - 4.0) * shape_delta if self.polar_shell else E_f
            U[k] = ml + restraint + g0 + nb_q + lam * lj1 + walls[s_w]
        return U, dict(E_bb=E_bb, S=S, U0=U0, c=c, nb=(a, b, cq), lj1=lj1, E_L=E_L, E_I=E_I,
                       nb_t1=nb_t1, shape_delta=shape_delta)

    def derivatives(self, P, q, parts, per_atom=True):
        """{segment: dU/dq} at state q, and per-atom arrays (kJ/mol per unit q)."""
        self.ctx.setPositions(P * unit.angstrom)
        out, atoms = {}, {}
        for seg in segments_of(q, self.shell_mm, self.polar_shell):
            if seg in ("W1", "W3"):
                sw = q if seg == "W1" else (4.0 if self.shell_mm or self.polar_shell else 5.0) - q
                self.ctx.setParameter("s_w", sw)
                st = self.ctx.getState(getParameterDerivatives=True, groups={mlmm.G_WALL})
                d = st.getEnergyParameterDerivatives()["s_w"]
                sign = 1.0 if seg == "W1" else -1.0
                out[seg] = sign * d
                if per_atom:
                    atoms[seg] = sign * self.wall_dsw_per_atom(P, sw)
            elif seg == "W2":
                lam2 = 2.0 - q
                a, b, cq = parts["nb"]
                out[seg] = -(parts["S"] + b + 2 * cq * lam2 + parts["lj1"])
                if per_atom:
                    self._set_lam(lam2)
                    coul = mlmm.mm_far_field_per_atom_coulomb(self.ctx, self.lig, lam2)
                    ljp = self.lj_per_atom(P)
                    ljp = ljp + self.lrc_w * (parts["lj1"] - ljp.sum())
                    atoms["W2_ml"] = -parts["c"]
                    atoms["W2_coul"] = -coul
                    atoms["W2_lj"] = -ljp
            elif seg == "cut":
                out[seg] = parts["U0"] - (parts["E_bb"] - parts["S"])
            elif seg == "shape":
                out[seg] = parts["shape_delta"]
            else:                                     # SW: dU/dt, global (not attributed)
                out[seg] = -parts["E_I"] + (parts["nb_t1"] - parts["nb"][0])
        return out, atoms


class MLMMLadder:
    def __init__(self, leg_npz, model, qs, start_positions, n_inner=20, T=298.15, dt_fs=2.0,
                 friction=1.0, seed=0, device="cuda", platform="CUDA", log=print, mm_only=False,
                 use_fires=True, eval_precision="double", ref=False, shell_mm=False,
                 polar_shell=False, fast_forces=True, batch_fragments=False):
        self.qs = np.asarray(qs, float)
        self.shell_mm = shell_mm
        self.polar_shell = polar_shell
        self.ref = ref
        if self.qs.ndim != 1 or len(self.qs) < 2 or not np.all(np.diff(self.qs) > 0):
            raise ValueError("ladder requires at least two strictly increasing path states")
        for value in self.qs:
            mlmm.segment(value, ref=ref, shell_mm=shell_mm, polar_shell=polar_shell)
        self.K = len(self.qs)
        self.T, self.kT = T, KB * T
        self.log = log
        plat = openmm.Platform.getPlatformByName(platform)
        self.reps = []
        t0 = time.perf_counter()
        for k, q in enumerate(self.qs):
            system, mlf, info = mlmm.build_mlmm_system(leg_npz, model, n_inner=n_inner, es=True,
                                                       device=device, positions=start_positions,
                                                       mm_only=mm_only, use_fires=use_fires,
                                                       fires_center="point" if ref else None,
                                                       shell_mm=shell_mm, polar_shell=polar_shell,
                                                       fast_forces=fast_forces, batch_fragments=batch_fragments)
            integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, friction / unit.picosecond,
                                                    dt_fs * unit.femtosecond)
            integ.setRandomNumberSeed(seed * 1000 + k + 1)
            ctx = openmm.Context(system, integ, plat)
            box = np.asarray(info["box"], float)
            ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])
            mlmm.set_path(ctx, mlf, q, info["lig"], ref=ref, shell_mm=shell_mm, polar_shell=polar_shell)
            ctx.setPositions(start_positions * unit.angstrom)
            ctx.setVelocitiesToTemperature(T * unit.kelvin, seed * 1000 + k + 1)
            self.reps.append(dict(ctx=ctx, integ=integ, mlf=mlf, info=info))
            if k == 0:
                inner0 = info["inner"]
            elif not np.array_equal(info["inner"], inner0):
                raise RuntimeError("replicas picked different inner waters")
        system, mlf, info = mlmm.build_mlmm_system(leg_npz, model, n_inner=n_inner, es=True, device=device,
                                                   positions=start_positions, mm_only=mm_only,
                                                   use_fires=use_fires, fires_center="point" if ref else None,
                                                   shell_mm=shell_mm, polar_shell=polar_shell,
                                                   fast_forces=fast_forces, batch_fragments=batch_fragments)
        # double precision for the evaluator: the per-atom Coulomb is a central
        # difference of PME energies of ~5e4 kJ/mol, which float32 resolves to
        # ~4e-3 kJ/mol per atom (the smoke's per-atom closure was off by 0.07)
        # (on a 2080 Ti, double-precision PME makes the evaluator the cost of an
        # all-MM iteration, ~80 s for 27 states; "mixed" is enough for totals)
        self.ev = Evaluator(system, mlf, info, self.qs, plat,
                            {"Precision": eval_precision} if platform == "CUDA" else None, ref=ref)
        self.info = info
        self.perm = np.arange(self.K)            # replica k carries the configuration that started in state perm[k]
        self.n_att = np.zeros(self.K - 1)
        self.n_acc = np.zeros(self.K - 1)
        self.it = 0
        self.rng = np.random.default_rng(seed)
        self.rec = dict(u=[], dudq=[], atoms=[])
        log(f"  built {self.K} replicas + evaluator in {time.perf_counter() - t0:.0f}s", flush=True)

    def _get(self, k):
        st = self.reps[k]["ctx"].getState(getPositions=True, getVelocities=True)
        return (np.asarray(st.getPositions(asNumpy=True).value_in_unit(unit.angstrom), dtype=np.float64),
                np.asarray(st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)))

    def _set(self, k, P, V):
        self.reps[k]["ctx"].setPositions(P * unit.angstrom)
        self.reps[k]["ctx"].setVelocities(V * unit.nanometer / unit.picosecond)

    def iterate(self, steps, record=True, per_atom=True):
        for k, r in enumerate(self.reps):
            try:
                r["integ"].step(steps)
            except Exception as exc:                 # say which state blew up
                raise RuntimeError(f"replica {k} (q {self.qs[k]:g}, segment {segments_of(self.qs[k], self.shell_mm, self.polar_shell)}) "
                                   f"failed at iteration {self.it + 1}: {exc}") from exc
        conf = [self._get(k) for k in range(self.K)]
        U, parts = zip(*[self.ev.all_states(P) for P, _ in conf])
        u = np.array(U) / self.kT                        # [conf k, state l]
        if record:
            dq, at = [], []
            for k in range(self.K):
                d, a = self.ev.derivatives(conf[k][0], self.qs[k], parts[k], per_atom=per_atom)
                dq.append(d); at.append(a)
            self.rec["u"].append(u)
            self.rec["dudq"].append(dq)
            self.rec["atoms"].append(at)
        start = self.rng.integers(2)
        for i in range(start, self.K - 1, 2):
            delta = (u[i, i + 1] + u[i + 1, i]) - (u[i, i] + u[i + 1, i + 1])
            self.n_att[i] += 1
            if delta <= 0 or self.rng.random() < np.exp(-delta):
                Pi, Vi = conf[i]
                Pj, Vj = conf[i + 1]
                self._set(i, Pj, Vj); self._set(i + 1, Pi, Vi)
                conf[i], conf[i + 1] = conf[i + 1], conf[i]
                self.perm[i], self.perm[i + 1] = self.perm[i + 1], self.perm[i]
                self.n_acc[i] += 1
        self.it += 1

    # ------------------------------------------------------------------ checkpoint

    def save(self, out_dir, n_equil):
        conf = [self._get(k) for k in range(self.K)]
        np.savez(f"{out_dir}/state.tmp.npz", P=np.array([c[0] for c in conf]), V=np.array([c[1] for c in conf]),
                 perm=self.perm, n_att=self.n_att, n_acc=self.n_acc, it=self.it, qs=self.qs,
                 rng=json.dumps(self.rng.bit_generator.state), n_equil=n_equil, shell_mm=self.shell_mm,
                 polar_shell=self.polar_shell, ref=self.ref, inner=self.info["inner"])
        os.replace(f"{out_dir}/state.tmp.npz", f"{out_dir}/state.npz")
        with open(f"{out_dir}/records.tmp.json", "w") as fh:
            json.dump(_jsonable(self.rec), fh)
        os.replace(f"{out_dir}/records.tmp.json", f"{out_dir}/records.json")

    def load(self, out_dir):
        z = np.load(f"{out_dir}/state.npz", allow_pickle=True)
        if z["qs"].shape != self.qs.shape or not np.allclose(z["qs"], self.qs):
            raise RuntimeError("checkpoint was written for a different ladder")
        for name in ("shell_mm", "polar_shell", "ref"):
            saved = bool(z[name]) if name in z.files else False
            if saved != bool(getattr(self, name)):
                raise RuntimeError(f"checkpoint has a different Hamiltonian: {name}={saved}")
        if "inner" in z.files and not np.array_equal(z["inner"], self.info["inner"]):
            raise RuntimeError("checkpoint has different inner-water identities")
        for k in range(self.K):
            self._set(k, z["P"][k], z["V"][k])
        self.perm, self.n_att, self.n_acc, self.it = z["perm"], z["n_att"], z["n_acc"], int(z["it"])
        self.rng.bit_generator.state = json.loads(str(z["rng"]))
        with open(f"{out_dir}/records.json") as fh:
            rec = json.load(fh)
        self.rec = dict(u=[np.array(x) for x in rec["u"]], dudq=rec["dudq"],
                        atoms=[[{k: np.array(v) for k, v in a.items()} for a in row] for row in rec["atoms"]])

    def init_from(self, run_dir, only_q=None):
        """Start each state from the configuration (and velocities) of the nearest-q state
        of an earlier run's checkpoint, e.g. a pilot, so slow segments (the W3 refill)
        do not start from the coupled frame. Same leg and start frame (inner waters)."""
        z = np.load(f"{run_dir}/state.npz", allow_pickle=True)
        saved_polar = bool(z["polar_shell"]) if "polar_shell" in z.files else False
        if saved_polar != self.polar_shell and only_q is None:
            raise RuntimeError("initialising a POLAR-shell path from another path requires an explicit init_q")
        old_q = np.asarray(z["qs"], float)
        if only_q is not None:                         # every state from that one old state
            j = int(np.argmin(np.abs(old_q - only_q)))
            if abs(old_q[j] - only_q) > 1e-6:
                raise RuntimeError(f"{run_dir} has no state at q {only_q}")
            pick = [j] * len(self.qs)
        else:
            pick = [int(np.argmin(np.abs(old_q - q))) for q in self.qs]
        for k, j in enumerate(pick):
            self._set(k, z["P"][j], z["V"][j])
        self.log("  initialised from " + run_dir + ": " +
                 ", ".join(f"{q:g}<-{old_q[j]:g}" for q, j in zip(self.qs, pick)), flush=True)

    def minimize(self):
        """Steepest-descent EM of every replica at its own state (user, 2026-10-23): a state far from
        the start frame's (e.g. POLAR-shell q >= 4.5, S13.61) otherwise goes NaN within 5 steps."""
        for k, r in enumerate(self.reps):
            t = time.perf_counter()
            U0, U1 = mlmm.steepest_descent(r["ctx"])
            self.log(f"  EM q {self.qs[k]:g}: {U0:.1f} -> {U1:.1f} kJ/mol ({time.perf_counter() - t:.0f} s)",
                     flush=True)

    def run(self, out_dir, n_iter, steps, n_equil=0, per_atom=True, save_every=1, init_from=None,
            init_q=None, em=True):
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(f"{out_dir}/state.npz"):
            self.load(out_dir)
            self.log(f"  resumed at iteration {self.it}", flush=True)
        else:
            if init_from:
                self.init_from(init_from, only_q=init_q)
            if em:
                self.minimize()
        while self.it < n_equil + n_iter:
            t = time.perf_counter()
            self.iterate(steps, record=self.it >= n_equil, per_atom=per_atom)
            if self.it % save_every == 0 or self.it == n_equil + n_iter:
                self.save(out_dir, n_equil)
            acc = self.n_acc / np.maximum(self.n_att, 1)
            self.log(f"  iter {self.it}/{n_equil + n_iter} ({'equil' if self.it <= n_equil else 'prod'}) "
                     f"{time.perf_counter() - t:.0f}s; swap acceptance min / mean {acc.min():.2f} / {acc.mean():.2f}",
                     flush=True)
        return analyze(self.qs, self.rec, self.kT, self.n_acc / np.maximum(self.n_att, 1),
                       shell_mm=self.shell_mm, polar_shell=self.polar_shell)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def analyze(qs, rec, kT, acceptance=None, shell_mm=False, polar_shell=False):
    """TI per segment (trapezoid over that segment's states), per-atom terms, MBAR total."""
    from .fe import mbar_free_energy, report_overlap
    qs = np.asarray(qs, float)
    n = len(rec["u"])
    out = dict(qs=qs.tolist(), n_samples=n, shell_mm=shell_mm, polar_shell=polar_shell)
    if polar_shell:
        target = "gas-phase ML ligand plus POLAR shell around one point in MM solvent"
        out["target_endpoint_reference"] = target
        out["path_complete"] = bool(len(qs) > 1 and np.isclose(qs[0], 0) and np.isclose(qs[-1], 5))
        out["endpoint_reference"] = target if len(qs) and np.isclose(qs[-1], 5) else "partial POLAR-shell path"
        out["reference_note"] = "This is not the bulk-MM endpoint; a reference-state conversion is needed for that comparison."
    if n == 0:
        return out
    total, per_atom = 0.0, {}
    for name, a, b in _segs(shell_mm, polar_shell):
        ks = [k for k, q in enumerate(qs) if a - 1e-9 <= q <= b + 1e-9]
        if len(ks) < 2:
            continue
        y = np.array([[rec["dudq"][t][k][name] for k in ks] for t in range(n)])     # [n, len(ks)]
        mean = y.mean(0)
        g = np.trapezoid(mean, qs[ks])
        err = np.sqrt(np.trapezoid(y.std(0, ddof=1) ** 2 / n, qs[ks]) * (qs[ks[-1]] - qs[ks[0]])) if n > 1 else np.nan
        out[f"dG_{name}"], out[f"dG_{name}_naive_err"] = float(g), float(err)
        out[f"dudq_{name}"] = mean.tolist()
        total += g
        keys = ["W2_ml", "W2_coul", "W2_lj"] if name == "W2" else ([name] if name in ("W1", "W3") else [])
        for key in keys:
            if all(key in rec["atoms"][t][k] for t in range(n) for k in ks):
                ya = np.array([[rec["atoms"][t][k][key] for k in ks] for t in range(n)]).mean(0)   # [len(ks), nl]
                per_atom[key] = np.trapezoid(ya, qs[ks], axis=0).tolist()
    out["dG_TI"] = float(total)
    out["per_atom"] = per_atom
    if per_atom:
        s = sum(np.sum(v) for v in per_atom.values())
        out["per_atom_sum"] = float(s)
        out["closure"] = float(total - s - out.get("dG_cut", 0.0) - out.get("dG_SW", 0.0)
                               - out.get("dG_shape", 0.0))
    u = np.array(rec["u"])                                                       # [n, conf, state]
    K = len(qs)
    u_kn = u.transpose(1, 0, 2).reshape(K * n, K).T
    # MBAR is exact under a constant shift of each state's reduced energy; the SW states carry
    # -E_bb(I) ~ 4e6 kJ/mol (~1.6e6 kT apart), which pymbar cannot condition, so shift each state
    # by its median over all samples and add the shifts back to the free energies.
    shift = np.median(u_kn, axis=1)
    try:
        m = mbar_free_energy(u_kn - shift[:, None], [n] * K, kT=kT)
        out.update(dG_MBAR=float(m["dG"] + (shift[-1] - shift[0]) * kT), dG_MBAR_err=float(m["dG_err"]),
                   overlap_min=float(report_overlap(m["overlap"])["min"]))
    except Exception as exc:  # noqa: BLE001
        out["mbar_error"] = f"{type(exc).__name__}: {exc}"
    if acceptance is not None:
        out["acceptance"] = np.asarray(acceptance).tolist()
    return out
