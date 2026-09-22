"""Hamiltonian replica exchange over the alchemical coupling (WP8).

One replica per lambda window, neighbour swaps, and the two arrays the analysis
needs: u_kn for MBAR and dU/dlambda for TI -- the latter per ligand atom, which
is where the per-atom attribution (S6.1) comes from at no extra sampling cost.

lambda lives as an attribute on each replica's `AlchemicalForce`, so every
replica needs its own force instance: the Hamiltonians stay put and the
configurations move between them.
"""

import numpy as np

KB_KJ = 0.008314462618      # kJ/mol/K


class TorchReplica:
    """Minimal replica backed by decomp.sampling.Langevin, in nm.

    OpenMM is the right engine for PBC, PME, constraints, barostats and
    restraints -- all of which matter at condensed-phase scale. At dimer scale
    it is pure overhead: profiling a 6-atom system gives 0.4 ms for the
    neighbour list, 30 ms for the MACE forward+backward, and ~150 ms once the
    PythonForce round trip is included. The methodological questions (lambda
    spacing, overlap, softcore, the message-masking control, attribution) all
    live at dimer scale, so they run here and WP9 runs on OpenMM.
    """

    def __init__(self, force, numbers, positions_nm, T=298.15, dt_fs=1.0,
                 friction=1.0, seed=0, device="cpu"):
        import torch
        from .sampling import Langevin, masses
        self.force = force
        self.torch = torch
        pos = torch.tensor(np.asarray(positions_nm) * 10.0,
                           dtype=torch.float32, device=device)

        def energy(p):
            """eV, differentiable in p (Angstrom)."""
            from .lambda_mask import alchemical_energy
            data = force._graph(p.detach().cpu().numpy())
            data["positions"] = p
            U, _, _ = alchemical_energy(force.model, data, force.lam,
                                        force.frag_id, mode=force.mode,
                                        ligand=force.ligand)
            return U.sum()

        self.dyn = Langevin(energy, pos, masses(numbers, pos.device),
                            T=T, dt=dt_fs, gamma=friction, seed=seed)

    def step(self, n):
        for _ in range(n):
            self.dyn.step()

    def get_positions_nm(self):
        return self.dyn.pos.detach().cpu().numpy() * 0.1

    def set_positions_nm(self, pos_nm):
        self.dyn.pos = self.torch.tensor(
            np.asarray(pos_nm) * 10.0, dtype=self.dyn.pos.dtype,
            device=self.dyn.pos.device)


class _OpenMMReplica:
    def __init__(self, ctx, integrator):
        self.ctx, self.integrator = ctx, integrator

    def step(self, n):
        self.integrator.step(n)

    def get_positions_nm(self):
        from openmm import unit
        return self.ctx.getState(getPositions=True).getPositions(
            asNumpy=True).value_in_unit(unit.nanometer)

    def set_positions_nm(self, pos_nm):
        from openmm import unit
        self.ctx.setPositions(pos_nm * unit.nanometer)


class HREMD:
    def __init__(self, systems, forces, positions, lambdas, T=298.15,
                 dt_fs=1.0, friction=1.0, platform=None, box=None,
                 backend="openmm", numbers=None, record_every=1):
        """systems/forces: one per window, same topology, different lambda.

        backend='torch' skips OpenMM (see TorchReplica); `systems` is then
        ignored and `numbers` is required.
        record_every > 1 records u_kn less often: that array costs n^2 energy
        evaluations per recording and MBAR wants decorrelated samples anyway.
        """
        assert len(forces) == len(lambdas)
        self.forces, self.lambdas = forces, np.asarray(lambdas, dtype=float)
        self.n = len(lambdas)
        self.kT = KB_KJ * T
        self.backend = backend
        self.record_every = max(1, int(record_every))
        self.replicas = []
        if backend == "torch":
            if numbers is None:
                raise ValueError("backend='torch' needs `numbers`")
            for i, (lam, f) in enumerate(zip(self.lambdas, forces)):
                f.set_lambda(float(lam))
                self.replicas.append(TorchReplica(
                    f, numbers, positions, T=T, dt_fs=dt_fs,
                    friction=friction, seed=i))
        else:
            import openmm
            from openmm import unit
            assert len(systems) == len(lambdas)
            plat = (openmm.Platform.getPlatformByName(platform) if platform
                    else None)
            for sysm, lam, f in zip(systems, self.lambdas, forces):
                f.set_lambda(float(lam))
                integ = openmm.LangevinMiddleIntegrator(
                    T * unit.kelvin, friction / unit.picosecond,
                    dt_fs * unit.femtosecond)
                ctx = (openmm.Context(sysm, integ, plat) if plat
                       else openmm.Context(sysm, integ))
                if box is not None:
                    ctx.setPeriodicBoxVectors(*box)
                ctx.setPositions(positions)
                ctx.setVelocitiesToTemperature(T * unit.kelvin)
                self.replicas.append(_OpenMMReplica(ctx, integ))
        # replica i currently carries the configuration of state perm[i]
        self.perm = np.arange(self.n)
        self.n_attempt = np.zeros(self.n - 1)
        self.n_accept = np.zeros(self.n - 1)
        self.u_kn, self.dudl, self.dudl_atom = [], [], []

    # ----------------------------------------------------------------- sampling

    def _positions(self, k):
        return self.replicas[k].get_positions_nm()

    def _reduced(self, k_state, pos_nm):
        """u = U/kT of configuration pos under window k_state's Hamiltonian."""
        f = self.forces[k_state]
        U, _ = f.energy_and_forces(np.asarray(pos_nm) * 10.0)
        return U * 96.48533212 / self.kT

    def _record(self):
        """One row of u_kn (all states x this configuration) plus dU/dlambda."""
        for k in range(self.n):
            pos = self._positions(k)
            self.u_kn.append([self._reduced(j, pos) for j in range(self.n)])
            f = self.forces[k]
            f.energy_and_forces(np.asarray(pos) * 10.0, need_dudl=True)
            g = f.last_dudl
            self.dudl_atom.append(g.copy())
            self.dudl.append(float(g.sum()))

    def _exchange(self, rng):
        """Neighbour swaps, alternating even/odd pairs."""
        start = rng.integers(2)
        for i in range(start, self.n - 1, 2):
            xi, xj = self._positions(i), self._positions(i + 1)
            uii, ujj = self._reduced(i, xi), self._reduced(i + 1, xj)
            uij, uji = self._reduced(i, xj), self._reduced(i + 1, xi)
            delta = (uij + uji) - (uii + ujj)
            self.n_attempt[i] += 1
            if delta <= 0 or rng.random() < np.exp(-delta):
                self.replicas[i].set_positions_nm(xj)
                self.replicas[i + 1].set_positions_nm(xi)
                self.perm[i], self.perm[i + 1] = self.perm[i + 1], self.perm[i]
                self.n_accept[i] += 1

    def run(self, n_iter, steps_per_iter=500, n_equil=0, seed=0, log_every=10):
        rng = np.random.default_rng(seed)
        for it in range(n_equil + n_iter):
            for rep in self.replicas:
                rep.step(steps_per_iter)
            self._exchange(rng)
            if it >= n_equil and (it - n_equil) % self.record_every == 0:
                self._record()
            if log_every and (it + 1) % log_every == 0:
                acc = self.n_accept.sum() / max(self.n_attempt.sum(), 1)
                print(f"      iter {it+1}/{n_equil+n_iter}  "
                      f"mean swap acceptance {acc:.2f}", flush=True)
        return self.results()

    # ----------------------------------------------------------------- analysis

    def results(self):
        from .fe import mbar_free_energy, report_overlap, ti, ti_per_atom
        n_frames = len(self.u_kn) // self.n
        u = np.array(self.u_kn).reshape(n_frames, self.n, self.n)
        u_kn = u.transpose(1, 0, 2).reshape(self.n * n_frames, self.n).T
        n_k = [n_frames] * self.n

        d = np.array(self.dudl).reshape(n_frames, self.n).mean(0)
        da = np.array(self.dudl_atom).reshape(n_frames, self.n, -1).mean(0)
        dG_ti, err_ti = ti(self.lambdas, d)
        out = {
            "lambdas": self.lambdas,
            "dudl": d,
            "dG_TI": dG_ti,
            "dG_TI_per_atom": ti_per_atom(self.lambdas, da),
            "acceptance": self.n_accept / np.maximum(self.n_attempt, 1),
            "n_frames": n_frames,
        }
        try:
            m = mbar_free_energy(u_kn, n_k, kT=self.kT)
            out.update({"dG_MBAR": m["dG"], "dG_MBAR_err": m["dG_err"],
                        "overlap": report_overlap(m["overlap"]),
                        "n_eff": m["n_eff"]})
        except Exception as exc:                       # keep TI even if MBAR fails
            out["mbar_error"] = f"{type(exc).__name__}: {exc}"
        return out
