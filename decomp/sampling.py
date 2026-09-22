"""Langevin sampling on the alchemical Hamiltonian (WP8).

Written in torch rather than through OpenMM: the model already is a torch module
returning conservative forces, and `openmm-torch`/`openmm-ml` are not installed.
For the dimer-scale systems of the validation ladder's stage A this is enough,
and it keeps the alchemical Hamiltonian in one place. Condensed-phase work
(stage B onward) needs a real MD engine and that is a separate decision (S11 R6).

Units: eV, Angstrom, amu, femtosecond.
"""

import numpy as np
import torch

KB = 8.617333262e-5          # eV/K
# 1 eV/A/amu = 9.6485e-3 A/fs^2
EV_PER_A_PER_AMU = 9.6485332e-3


def masses(numbers, device):
    from ase.data import atomic_masses
    return torch.tensor([atomic_masses[int(z)] for z in numbers],
                        dtype=torch.float32, device=device)


class Langevin:
    """BAOAB Langevin integrator at fixed lambda.

    BAOAB rather than a naive scheme because the configurational averages here
    feed straight into a free energy: its configurational sampling error is
    O(dt^2) where leapfrog-Langevin is O(dt), and dU/dlambda is exactly a
    configurational average.
    """

    def __init__(self, energy_fn, pos, mass, T=300.0, dt=0.5, gamma=1.0, seed=0):
        self.energy_fn = energy_fn          # pos -> scalar U
        self.pos = pos.clone()
        self.m = mass.unsqueeze(-1)
        self.T, self.dt, self.gamma = T, dt, gamma
        g = torch.Generator(device=pos.device).manual_seed(seed)
        self.g = g
        sigma = torch.sqrt(KB * T / self.m)
        self.vel = sigma * torch.randn(pos.shape, generator=g, device=pos.device)
        self.vel /= np.sqrt(EV_PER_A_PER_AMU)      # into A/fs
        self._f = self.force(self.pos)

    def force(self, pos):
        p = pos.detach().requires_grad_(True)
        U = self.energy_fn(p)
        return -torch.autograd.grad(U, p)[0].detach()

    def step(self):
        dt, m = self.dt, self.m
        a = EV_PER_A_PER_AMU
        self.vel += 0.5 * dt * self._f / m * a                      # B
        self.pos = self.pos + 0.5 * dt * self.vel                   # A
        c1 = np.exp(-self.gamma * dt * 1e-3)                        # O (gamma in 1/ps)
        c2 = np.sqrt(1 - c1 ** 2)
        sigma = torch.sqrt(KB * self.T / m / a)
        self.vel = c1 * self.vel + c2 * sigma * torch.randn(
            self.pos.shape, generator=self.g, device=self.pos.device)
        self.pos = self.pos + 0.5 * dt * self.vel                   # A
        self._f = self.force(self.pos)
        self.vel += 0.5 * dt * self._f / m * a                      # B
        return self.pos


def sample(energy_fn, pos0, numbers, n_steps, n_equil=0, stride=10,
           T=300.0, dt=0.5, gamma=1.0, seed=0, callback=None):
    """Run Langevin and return the sampled configurations.

    callback(pos) is called on each saved frame; use it to record dU/dlambda
    without storing trajectories.
    """
    dyn = Langevin(energy_fn, pos0, masses(numbers, pos0.device),
                   T=T, dt=dt, gamma=gamma, seed=seed)
    frames, out = [], []
    for i in range(n_equil + n_steps):
        dyn.step()
        if i >= n_equil and (i - n_equil) % stride == 0:
            if callback is None:
                frames.append(dyn.pos.detach().clone())
            else:
                out.append(callback(dyn.pos.detach()))
    return out if callback is not None else frames
