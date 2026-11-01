"""Stepwise free energy of the ACTUAL MM states; ML only decomposes the work.

For pre-removal production frames x, w(x) = U_after^MM(x)-U_before^MM(x).
Delta G = -kT log mean(exp(-w/kT)). No model target potential is introduced.
The ML atom work estimates are -D_iE; their difference from w is retained
inside the same exponential through an explicit residual component.

All energies below are kJ/mol. No dynamics, learned density weights,
MBAR, effective-sample-size gate, or physical convergence claim is made.
"""

import numpy as np

KB = 0.008314462618


def _kt(temperature_K):
    if not np.isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("temperature_K must be finite and positive")
    return KB * temperature_K


def exponential_free_energy(work_kj, temperature_K):
    """Stable Zwanzig exponential average, retaining every supplied frame."""
    kt = _kt(temperature_K)
    work = np.asarray(work_kj, dtype=float)
    if work.ndim != 1 or len(work) == 0 or not np.all(np.isfinite(work)):
        raise ValueError("work must be a nonempty finite one-dimensional array")
    offset = work.min()
    return float(offset - kt * np.log(np.exp(-(work - offset) / kt).mean()))


def estimate_step(work_forward_kj, temperature_K, d_before_kj=None,
                  work_reverse_kj=None, epsabs=1e-8, epsrel=1e-8):
    """Evaluate one physical MM step and optionally its ML atom attribution.

    d_before_kj has shape [pre-removal frames, atoms in the group being removed].
    No ML output on post-removal penetration frames is needed.
    epsabs/epsrel control only numerical integration error, not sampling quality.
    Reverse EXP is reported separately and never averaged with the forward result.
    """
    kt = _kt(temperature_K)
    work = np.asarray(work_forward_kj, dtype=float)
    delta = exponential_free_energy(work, temperature_K)
    result = {
        "estimand": "free_energy_difference_between_actual_MM_states",
        "method": "forward_exponential_average",
        "temperature_K": float(temperature_K),
        "n_forward": len(work),
        "delta_g_kj_mol": delta,
        "mean_forward_work_kj_mol": float(work.mean()),
        "sampling_convergence": "not_established_by_this_calculation",
    }
    if work_reverse_kj is not None:
        reverse = np.asarray(work_reverse_kj, dtype=float)
        delta_reverse = -exponential_free_energy(reverse, temperature_K)
        result.update(n_reverse=len(reverse), delta_g_reverse_exp_kj_mol=delta_reverse,
                      forward_minus_reverse_kj_mol=delta - delta_reverse)
    if d_before_kj is None:
        return result

    from scipy.integrate import quad_vec

    d = np.asarray(d_before_kj, dtype=float)
    if d.ndim != 2 or d.shape[0] != len(work) or d.shape[1] == 0 or not np.all(np.isfinite(d)):
        raise ValueError("d_before_kj must be finite [forward frames, group atoms]")
    if not np.isfinite(epsabs) or not np.isfinite(epsrel) or epsabs <= 0 or epsrel <= 0:
        raise ValueError("numerical integration tolerances must be positive and finite")
    atom_work = -d
    residual = work - atom_work.sum(axis=1)
    components = np.column_stack([atom_work, residual])
    centered = (work - work.min()) / kt

    def integrand(t):
        probability = np.exp(-t * centered)
        probability /= probability.sum()
        return probability @ components

    # Resolve the narrow region near t=0 when the true work range is large.
    # These are numerical breakpoints, never simulated MD states.
    span = centered.max()
    points = None
    if span > 1:
        points = np.exp2(np.arange(-int(np.ceil(np.log2(span))), 0, dtype=float))
    values, error, info = quad_vec(integrand, 0.0, 1.0, epsabs=epsabs, epsrel=epsrel,
                                   norm="max", points=points, full_output=True)
    result.update(
        atom_delta_g_kj_mol=values[:-1].tolist(),
        residual_delta_g_kj_mol=float(values[-1]),
        mean_residual_work_kj_mol=float(residual.mean()),
        component_sum_kj_mol=float(values.sum()),
        closure_error_kj_mol=float(values.sum() - delta),
        numerical_integration_error_estimate_kj_mol=float(error),
        numerical_integration_success=bool(info.success),
        numerical_integration_evaluations=int(info.neval),
        numerical_epsabs_kj_mol=float(epsabs), numerical_epsrel=float(epsrel),
        attribution="straight_path_derivative_of_the_actual_work_log_exponential_average",
    )
    return result
