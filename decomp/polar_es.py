"""Explicit ligand-environment Coulomb channel for MACE-POLAR-1 (RESEARCH_PLAN_v3.md S2.3, D6).

POLAR-1's long-range term is the Coulomb energy of Gaussian-smeared atomic
densities after the field-update iterations, `E_es = 1/2 rho^T K rho`. The head
never sees it -- PolarMACE's `node_feats` are the short-range stack only
(test_polar.py) -- and the anchored remainder tracks its L/E part with slope ~1
(v3 S13.27). At fixed densities the L/E part is exactly pair-resolved:

    cross = E_es(rho_L + rho_E) - E_es(rho_L) - E_es(rho_E)

computed here from the very densities the backbone's forward pass produced, by
calling its `coulomb_energy` block again with one fragment's sources zeroed.
No pair matrix is needed. `cross` is linear in rho_L, so by Euler's theorem the
per-ligand-atom split is `rho_i . d cross / d rho_i`, exact, one autograd call;
the ligand rows sum to `cross`, and so do the environment rows.

Non-periodic only: with pbc all-False PolarMACE takes the real-space evaluator
(data.polar_fields), so no k-space term enters.
"""

import contextlib

import torch
from torch import nn


class _Recorder(nn.Module):
    """Stands in for `PolarMACE.coulomb_energy`: same return value, plus the
    fragment cross term from the same call's densities."""

    def __init__(self, inner, frag_id, ligand, per_atom):
        super().__init__()
        self.inner = inner
        self.frag_id, self.ligand, self.per_atom = frag_id, ligand, per_atom
        self.rec = {}

    def _cross(self, kw, sf):
        m = (self.frag_id == self.ligand).to(sf.dtype).view(-1, *([1] * (sf.dim() - 1)))
        e = self.inner(**{**kw, "source_feats": sf})
        eL = self.inner(**{**kw, "source_feats": sf * m})
        eE = self.inner(**{**kw, "source_feats": sf * (1 - m)})
        return e, e - eL - eE

    def forward(self, **kw):
        sf = kw["source_feats"]
        e, cross = self._cross(kw, sf)
        self.rec["cross"] = cross                      # [n_graphs], keeps the graph
        if self.per_atom:
            with torch.enable_grad():
                s = sf.detach().requires_grad_(True)
                kw_d = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kw.items()}
                _, c = self._cross(kw_d, s)
                g, = torch.autograd.grad(c.sum(), s)
            self.rec["per_atom"] = (s * g).detach().reshape(len(s), -1).sum(-1)
        return e


@contextlib.contextmanager
def coulomb_cross(bb, frag_id, ligand=0, per_atom=False):
    """Inside this block every forward of the POLAR backbone `bb` also records
    the ligand-environment Coulomb cross term.

        with coulomb_cross(bb, data["frag_id"]) as rec:
            out = bb(data, ...)
        rec["cross"]       # [n_graphs], differentiable w.r.t. positions
        rec["per_atom"]    # [n_nodes], detached; only with per_atom=True

    `per_atom` is for recording dU/dlambda_i on the diagonal path. Forces need
    only d cross / dR (all lambda_i equal there), which `rec["cross"]` carries.
    """
    if not hasattr(bb, "coulomb_energy"):
        raise TypeError(f"{type(bb).__name__} has no explicit Coulomb block")
    inner = bb.coulomb_energy
    rec_mod = _Recorder(inner, frag_id, ligand, per_atom)
    bb.coulomb_energy = rec_mod
    try:
        yield rec_mod.rec
    finally:
        bb.coulomb_energy = inner


def es_coupling(bb, positions, numbers, frag, device="cuda", ligand=0):
    """(cross, d_es) for one structure, in float64, from three backbone passes.

    cross  L/E Coulomb at the full system's densities (the explicit channel)
    d_es   E_es(all) - E_es(L alone) - E_es(E alone): the electrostatic part of
           the backbone's coupling, i.e. cross plus the densities' response

    For scoring a head trained with E3D_ES=1 against the backbone's coupling.
    """
    import numpy as np
    from .data import mace_batch

    was = next(bb.parameters()).dtype
    bb.to(torch.float64)
    try:
        def es(p, z, f=None):
            b = mace_batch([p], [z], bb, charges=[0], device=device, dtype=torch.float64,
                           frag_ids=None if f is None else [f])
            if f is None:
                with torch.no_grad():
                    return float(bb(b, training=False, compute_force=False)
                                 ["electrostatic_energy"].sum()), None
            with torch.no_grad(), coulomb_cross(bb, b["frag_id"], ligand) as rec:
                e = float(bb(b, training=False, compute_force=False)
                          ["electrostatic_energy"].sum())
            return e, float(rec["cross"].sum())

        frag = np.asarray(frag)
        L = frag == ligand
        e_all, cross = es(positions, numbers, frag)
        e_L, _ = es(positions[L], np.asarray(numbers)[L])
        e_E, _ = es(positions[~L], np.asarray(numbers)[~L])
    finally:
        bb.to(was)
    return cross, e_all - e_L - e_E
