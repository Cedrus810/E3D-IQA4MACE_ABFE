"""Path-defined bookkeeping for DESIGN §7.5; energies are kJ/mol.

These functions consume already estimated ensemble means / MM atomic TI.
They do not estimate missing ensembles, split a scalar BAR result into atoms,
or establish molecular convergence. No model, OpenMM or GPU is imported.
"""

import numpy as np


def integrate_endpoint_atoms(etas, mean_delta_u_atom):
    """A[j,i] = integral_0^1 <delta_u[j,i]>_(j,eta) d eta.

    Input means have shape [eta, atom] for ONE endpoint and ONE frozen target.
    Rejection repeats must remain in the underlying MC samples. Quadrature
    error and sample error must be assessed externally.
    """
    eta = np.asarray(etas, dtype=float)
    means = np.asarray(mean_delta_u_atom, dtype=float)
    if eta.ndim != 1 or len(eta) < 2 or means.ndim != 2 or means.shape[0] != len(eta):
        raise ValueError("need eta[n_eta>=2] and means[n_eta,n_atom]")
    if means.shape[1] == 0 or not np.isfinite(eta).all() or not np.isfinite(means).all():
        raise ValueError("eta and nonempty atomic means must be finite")
    if not (eta[0] == 0 and eta[-1] == 1 and np.all(np.diff(eta) > 0)):
        raise ValueError("endpoint correction requires strictly increasing eta from 0 to 1")
    return np.trapezoid(means, eta, axis=0)


def assemble_step_atom_matrix(mm_step_atom, endpoint_atom):
    """H[k,i] = B[k,i] + A[k+1,i] - A[k,i].

    B: [step, atom], atomic TI contributions along a specified MM path.
    A: [endpoint, atom], endpoint eta TI above; there is one shared record
    per endpoint. H can contain contributions on atoms outside removed group k.
    This function does not enforce closure to an independently estimated BAR
    result: any such discrepancy must be reported, never redistributed.
    """
    b, a = np.asarray(mm_step_atom, float), np.asarray(endpoint_atom, float)
    if b.ndim != 2 or a.shape != (b.shape[0] + 1, b.shape[1]) or min(b.shape) == 0:
        raise ValueError("need B[K,N] and A[K+1,N] with nonempty steps and atoms")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("missing / non-finite atomic contributions cannot be filled with zero")
    h = b + np.diff(a, axis=0)
    return {"step_atom": h, "step_total": h.sum(axis=1),
            "atom_total": h.sum(axis=0), "total": float(h.sum()),
            "telescoped_atom": b.sum(axis=0) + a[-1] - a[0]}
