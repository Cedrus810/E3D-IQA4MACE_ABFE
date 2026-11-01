"""Offline two-endpoint prototype; all input energies are reduced (beta * U).

No dynamics, model loading, outlier removal, or calibrated uncertainty estimates.
Columns contain decorrelated samples from source A followed by source B.
Rows of u_source/u_target are A (coupled), B (decoupled).
components[:, n].sum() must equal u_target[A,n] - u_target[B,n].
"""

import argparse
import json
from pathlib import Path

import numpy as np


def logsumexp(x, axis=None):
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else float(out.item())


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -np.asarray(x)))


def normalized(logw):
    return np.exp(logw - logsumexp(logw))


def weight_stats(w):
    return {"ess": float(1.0 / np.dot(w, w)),
            "ess_fraction": float(1.0 / (len(w) * np.dot(w, w))),
            "max_weight": float(w.max())}


def analyze(u_source, u_target, components, n_source, *, min_ess=50,
            min_fraction=0.02, max_weight=0.05, min_overlap=0.03,
            consistency_tol=0.1, quadrature_tol=1e-5, order=32):
    """Return estimates plus support/numerical gates, never a physical acceptance.

    Equal-prior weighted Bennett first connects target normalizers. Their
    source corrections give a common mixture measure for endpoint free energies
    and straight-path component integrals. Thus component closure is a
    derivative identity on the *same* finite-sample measure.
    """
    m, v, c = [np.asarray(x, dtype=float) for x in (u_source, u_target, components)]
    raw_counts = np.asarray(n_source)
    if raw_counts.shape != (2,) or not np.all(np.isfinite(raw_counts)):
        raise ValueError("n_source must contain two positive integer counts")
    counts = raw_counts.astype(int)
    if np.any(counts <= 0) or not np.array_equal(counts, raw_counts):
        raise ValueError("n_source must contain two positive integer counts")
    n = int(counts.sum())
    if m.shape != (2, n) or v.shape != (2, n) or c.ndim != 2 or c.shape[1] != n or not len(c):
        raise ValueError("expected u_source/u_target [2,N], components [C,N]")
    if not all(np.all(np.isfinite(x)) for x in (m, v, c)):
        raise ValueError("non-finite energy; retain and investigate the originating frames")
    coupling = v[0] - v[1]
    error = float(np.max(np.abs(c.sum(axis=0) - coupling)))
    if not np.allclose(c.sum(axis=0), coupling, atol=1e-7, rtol=1e-10):
        raise ValueError(f"per-frame coupling accounting does not close: max error {error}")
    if order < 2 or int(order) != order:
        raise ValueError("quadrature order must be an integer >= 2")
    if min_ess <= 0 or not 0 < min_fraction <= 1 or not 0 < max_weight <= 1:
        raise ValueError("invalid weight thresholds")
    if not 0 < min_overlap <= 1 or consistency_tol <= 0 or quadrature_tol <= 0:
        raise ValueError("invalid overlap/numerical thresholds")
    blocks = (slice(0, counts[0]), slice(counts[0], n))
    weights, corrections = [], []
    for s, block in enumerate(blocks):
        z = -(v[s, block] - m[s, block])
        weights.append(normalized(z))
        corrections.append(-logsumexp(z) + np.log(counts[s]))
    forward, reverse = -coupling[blocks[0]], coupling[blocks[1]]

    def equation(delta):
        return (np.dot(weights[0], sigmoid(delta - forward))
                - np.dot(weights[1], sigmoid(-delta - reverse)))

    lo = float(min(forward.min(), -reverse.max()) - 50.0)
    hi = float(max(forward.max(), -reverse.min()) + 50.0)
    if not np.isfinite(lo + hi) or not equation(lo) < 0 < equation(hi):
        raise ValueError("could not bracket target free-energy difference")
    for _ in range(256):
        mid = lo + (hi - lo) / 2
        value = equation(mid)
        if value == 0:
            lo = hi = mid
            break
        if value < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * max(1.0, abs(mid)):
            break
    delta = (lo + hi) / 2
    overlap = float(2 * np.dot(weights[0], sigmoid(delta - forward)))
    f_source = np.array([0.0, delta]) - corrections
    logden = logsumexp(np.log(counts)[:, None] + f_source[:, None] - m, axis=0)

    def at(t):
        logw = -((1 - t) * v[1] + t * v[0]) - logden
        return normalized(logw), -logsumexp(logw)

    wa, fa = at(1.0)
    wb, fb = at(0.0)
    endpoint_delta = fb - fa

    def integrate(nq):
        nodes, qw = np.polynomial.legendre.leggauss(nq)
        values = np.zeros(len(c))
        stats = []
        for t, q in zip((nodes + 1) / 2, qw / 2):
            w, _ = at(t)
            values -= q * (c @ w)
            stats.append(weight_stats(w))
        return values, stats

    coarse, _ = integrate(int(order))
    fine, path_stats = integrate(2 * int(order))
    stats = [weight_stats(w) for w in (*weights, wa, wb)] + path_stats
    min_effective = min(s["ess"] for s in stats)
    min_frac = min(s["ess_fraction"] for s in stats)
    max_w = max(s["max_weight"] for s in stats)
    reasons = []
    if min_effective < min_ess or min_frac < min_fraction or max_w > max_weight:
        reasons.append("importance_weights")
    if overlap < min_overlap:
        reasons.append("endpoint_overlap")
    consistency = abs(endpoint_delta - delta)
    quad_error = float(np.max(np.abs(fine - coarse)))
    closure = abs(float(fine.sum()) - endpoint_delta)
    numerical = []
    if consistency > consistency_tol:
        numerical.append("nbb_mixture_disagreement")
    if quad_error > quadrature_tol or closure > quadrature_tol:
        numerical.append("quadrature_or_closure")
    return {
        "status": "insufficient_support" if reasons else ("numerical_failure" if numerical else "diagnostics_passed"),
        "development_prototype": True,
        "physical_validation": "not_assessed",
        "uncertainty": None,
        "units": "reduced",
        "delta_f_nbb": float(delta),
        "delta_f": float(endpoint_delta),
        "component_delta_f": fine.tolist(),
        "source_target_corrections": [float(x) for x in corrections],
        "source_target_weight_stats": [weight_stats(w) for w in weights],
        "bennett_overlap": overlap,
        "minimum_ess": min_effective,
        "minimum_ess_fraction": min_frac,
        "maximum_weight": max_w,
        "nbb_mixture_difference": float(consistency),
        "quadrature_difference": quad_error,
        "closure_error": float(closure),
        "frame_accounting_error": error,
        "quadrature_orders": [int(order), 2 * int(order)],
        "support_failures": reasons,
        "numerical_failures": numerical,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as z:
        if str(z["units"].item()) != "reduced" or not bool(z["decorrelated"].item()):
            parser.error("input must declare units='reduced' and decorrelated=True")
        temperature = float(z["temperature_K"].item())
        if not np.isfinite(temperature) or temperature <= 0:
            parser.error("temperature_K must be finite and positive")
        names = z["component_names"].tolist()
        if len(names) != len(z["components"]) or not all(isinstance(x, str) for x in names):
            parser.error("component_names must name every component")
        result = analyze(z["u_source"], z["u_target"], z["components"], z["n_source"])
    kt = 0.008314462618 * temperature
    result.update(temperature_K=temperature, component_names=names, input=str(args.input.resolve()),
                  delta_g_kj_mol=result["delta_f"] * kt,
                  component_delta_g_kj_mol=(np.array(result["component_delta_f"]) * kt).tolist())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['status']}: {result['delta_g_kj_mol']:+.6f} kJ/mol -> {args.output}")
    return 0 if result["status"] == "diagnostics_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
