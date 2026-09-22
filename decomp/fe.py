"""Free-energy estimation: TI along the diagonal path, and MBAR (WP8).

The pymbar compatibility wrappers and the finiteness checks are adapted from
/home/ruigengji/ABFE_IBS/Atenolol-rank11/abfe_core.py, which has already been
through the pymbar 3.x/4.x API churn and through NaN-in-the-overlap-matrix
debugging. Reusing them beats rediscovering both.

What is NOT reused from there: everything MM-specific -- GROMACS topologies,
charge treatment, co-alchemical ions, barostats. The alchemical Hamiltonian
here is a learned decomposition (S5), not a force field.
"""

import numpy as np

try:
    import pymbar
    HAS_PYMBAR = True
except ImportError:
    HAS_PYMBAR = False

KB_KJ = 0.008314462618      # kJ/mol/K
KB_EV = 8.617333262e-5      # eV/K


# ----------------------------------------------------------------- pymbar shims

def build_mbar(u_kn, n_k, **kwargs):
    """pymbar 3.x/4.x tolerant MBAR constructor: drop unsupported kwargs in order."""
    if not HAS_PYMBAR:
        raise ImportError("pymbar required")
    drop_order = ["solver_protocol", "initialize", "relative_tolerance",
                  "solver_tolerance", "initial_f_k", "verbose"]
    current, variants = dict(kwargs), [dict(kwargs)]
    for key in drop_order:
        if key in current:
            current = dict(current); current.pop(key, None)
            variants.append(dict(current))
    last = None
    for candidate in variants:
        try:
            return pymbar.MBAR(u_kn, n_k, **candidate)
        except TypeError as exc:
            last = exc
    raise last if last is not None else RuntimeError("MBAR construction failed")


def _extract(result, primary, fallbacks=()):
    for name in (primary,) + tuple(fallbacks):
        if isinstance(result, dict) and name in result:
            return np.asarray(result[name], dtype=float)
        if hasattr(result, name):
            return np.asarray(getattr(result, name), dtype=float)
    return None


def free_energy_differences(mbar, uncertainty=True):
    """Returns (Delta_f, dDelta_f) across pymbar API versions."""
    methods = []
    if hasattr(mbar, "compute_free_energy_differences"):
        methods.append(("compute_free_energy_differences",
                        {"compute_uncertainty": uncertainty}))
    if hasattr(mbar, "compute_free_energy"):
        methods.append(("compute_free_energy", {}))
    res, last = None, None
    for name, kw in methods:
        try:
            res = getattr(mbar, name)(**kw); break
        except TypeError:
            try:
                res = getattr(mbar, name)(); break
            except Exception as exc:
                last = exc
        except Exception as exc:
            last = exc
    if res is None:
        raise last if last is not None else AttributeError("no MBAR free-energy method")
    df = _extract(res, "Delta_f", ("delta_f", "free_energy"))
    if df is None:
        raise KeyError("no free-energy matrix in the pymbar result")
    ddf = _extract(res, "dDelta_f", ("d_delta_f", "error", "uncertainty"))
    return df, (ddf if ddf is not None else np.full_like(df, np.nan))


def overlap_matrix(mbar):
    """Overlap between lambda windows. Raises on NaN rather than letting it
    flow silently into a convergence criterion."""
    res = mbar.compute_overlap()
    m = np.asarray(res["matrix"] if isinstance(res, dict) else res, dtype=float)
    if not np.all(np.isfinite(m)):
        raise RuntimeError(f"non-finite overlap matrix: {m}")
    return m


# ------------------------------------------------------------------- estimators

def statistical_inefficiency(series, max_lag=1000):
    """g = 1 + 2*sum_t C(t); the number of correlated frames per independent one."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 3:
        return 1.0
    x = x - x.mean()
    var = float(np.mean(x * x))
    if var <= 0:
        return 1.0
    g = 1.0
    for t in range(1, min(max_lag, n - 1)):
        c = float(np.mean(x[:-t] * x[t:])) / var
        if c <= 0:
            break
        g += 2.0 * c * (1.0 - t / n)
    return max(g, 1.0)


def ti(lambdas, dudl, dudl_err=None):
    """Thermodynamic integration by trapezoid: dG = integral <dU/dlambda> dlambda.

    lambdas must be ordered; the sign follows their direction, so passing them
    from 1 to 0 gives the decoupling free energy directly.
    """
    lam = np.asarray(lambdas, dtype=float)
    y = np.asarray(dudl, dtype=float)
    dG = float(np.trapezoid(y, lam))
    if dudl_err is None:
        return dG, float("nan")
    # trapezoid weights, errors added in quadrature
    w = np.zeros_like(lam)
    d = np.diff(lam)
    w[:-1] += d / 2; w[1:] += d / 2
    return dG, float(np.sqrt(np.sum((w * np.asarray(dudl_err)) ** 2)))


def ti_per_atom(lambdas, dudl_per_atom):
    """Per-atom dG_i along the diagonal path (S6.1).

    dudl_per_atom: [n_lambda, n_atoms] of < dU/dlambda_i >. Their sum is dU/dt
    along the diagonal, so sum_i dG_i = dG exactly -- verified in S13.7.
    """
    lam = np.asarray(lambdas, dtype=float)
    g = np.asarray(dudl_per_atom, dtype=float)
    return np.trapezoid(g, lam, axis=0)


def mbar_free_energy(u_kn, n_k, kT=1.0):
    """MBAR free energies and the overlap matrix. u_kn must be reduced (u/kT)."""
    mbar = build_mbar(u_kn, n_k)
    df, ddf = free_energy_differences(mbar)
    return {"dG": float(df[0, -1]) * kT,
            "dG_err": float(ddf[0, -1]) * kT,
            "overlap": overlap_matrix(mbar),
            "n_eff": np.asarray(mbar.compute_effective_sample_number(), dtype=float)}


def report_overlap(ov, threshold=0.03):
    """Nearest-neighbour overlap is the practical window-spacing diagnostic."""
    nn = np.array([ov[i, i + 1] for i in range(len(ov) - 1)])
    bad = np.where(nn < threshold)[0]
    return {"nearest_neighbour": nn, "min": float(nn.min()),
            "poor_windows": bad.tolist(), "ok": len(bad) == 0}
