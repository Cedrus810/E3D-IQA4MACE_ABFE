"""Verify the groupwise design on independently enumerable partition functions.

Two groups, three source MM states, and three discrete configurations.
Integer sample multiplicities exactly integrate each different source density.
This verifies offline equations, not molecular sampling or model accuracy.
"""

import json
from pathlib import Path

import numpy as np


def lse(x, axis=None):
    maximum = np.max(x, axis=axis, keepdims=True)
    out = maximum + np.log(np.exp(x - maximum).sum(axis=axis, keepdims=True))
    return float(out.item()) if axis is None else np.squeeze(out, axis=axis)


def normalize(logw):
    return np.exp(logw - lse(logw))


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def solve_bennett(v, m, blocks, correct=True):
    corrections, weights = [], []
    for s, b in enumerate(blocks):
        logw = -(v[s, b] - m[s, b]) if correct else np.zeros(len(v[s, b]))
        corrections.append(-lse(logw) + np.log(len(logw)))
        weights.append(normalize(logw))
    steps = []
    overlaps = []
    for s in range(len(blocks) - 1):
        forward = (v[s + 1] - v[s])[blocks[s]]
        reverse = (v[s] - v[s + 1])[blocks[s + 1]]

        def equation(delta):
            return (weights[s] @ sigmoid(delta - forward)
                    - weights[s + 1] @ sigmoid(-delta - reverse))

        low = min(forward.min(), -reverse.max()) - 50
        high = max(forward.max(), -reverse.min()) + 50
        for _ in range(160):
            mid = (low + high) / 2
            if equation(mid) < 0:
                low = mid
            else:
                high = mid
        delta = (low + high) / 2
        steps.append(delta)
        overlaps.append(float(2 * (weights[s] @ sigmoid(delta - forward))))
    return np.array(steps), np.array(corrections), overlaps


def infer(source, target, atomic, labels, n_per_source=1000):
    states = len(source)
    blocks = [slice(s * n_per_source, (s + 1) * n_per_source) for s in range(states)]
    pilot, corrections, overlaps = solve_bennett(target, source, blocks)
    target_f = np.r_[0.0, pilot.cumsum()]
    source_f = target_f - corrections
    logden = lse(np.log(n_per_source) + source_f[:, None] - source, axis=0)
    common_f = np.array([-lse(-v - logden) for v in target])
    parts = np.zeros(len(atomic))
    nodes, qw = np.polynomial.legendre.leggauss(64)
    for group in range(states - 1):
        members = labels == group
        for t, q in zip((nodes + 1) / 2, qw / 2):
            u = (1 - t) * target[group + 1] + t * target[group]
            w = normalize(-u - logden)
            parts[members] -= q * (atomic[members] @ w)
    return common_f, parts, pilot, overlaps


def main():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]])
    source_unique = -np.log(probabilities)
    atomic_unique = np.array([[0.7, -0.3, 0.1], [-0.2, 0.8, 0.3], [0.1, -0.4, 0.9]])
    labels = np.array([0, 0, 1])
    target_unique = np.array([source_unique[-1] + atomic_unique[labels >= s].sum(axis=0)
                              for s in range(3)])
    exact_f = np.array([-lse(v * -1) for v in target_unique])
    index = np.concatenate([np.repeat(np.arange(3), (p * 1000).astype(int)) for p in probabilities])
    source, target, atomic = source_unique[:, index], target_unique[:, index], atomic_unique[:, index]
    f, parts, pilot, overlaps = infer(source, target, atomic, labels)
    np.testing.assert_allclose(f - f[0], exact_f - exact_f[0], atol=1e-11)
    np.testing.assert_allclose(pilot, np.diff(exact_f), atol=1e-11)
    np.testing.assert_allclose([parts[labels == s].sum() for s in range(2)], np.diff(exact_f), atol=1e-11)
    np.testing.assert_allclose(parts.sum(), f[-1] - f[0], atol=1e-11)
    # Independent target Boltzmann integrals on unique configurations.
    direct = np.zeros(3)
    nodes, weights = np.polynomial.legendre.leggauss(100)
    for group in range(2):
        for t, q in zip((nodes + 1) / 2, weights / 2):
            p = normalize(-((1 - t) * target_unique[group + 1] + t * target_unique[group]))
            direct[labels == group] -= q * (atomic_unique[labels == group] @ p)
    np.testing.assert_allclose(parts, direct, atol=1e-11)
    shifted_f, shifted_parts, _, _ = infer(source + np.array([[5.0], [-7.0], [20.0]]) + 1000,
                                          target + 1000, atomic, labels)
    np.testing.assert_allclose(shifted_f - shifted_f[0], f - f[0], atol=1e-10)
    np.testing.assert_allclose(shifted_parts, parts, atol=1e-10)
    blocks = [slice(s * 1000, (s + 1) * 1000) for s in range(3)]
    uncorrected, _, _ = solve_bennett(target, source, blocks, correct=False)
    wrong_error = float(np.max(np.abs(uncorrected - np.diff(exact_f))))
    assert wrong_error > 0.01, wrong_error
    result = {
        'kind': 'exact finite-state mathematical validation; not molecular data',
        'sampling_states': 3, 'groups': 2, 'atoms': 3,
        'checks_passed': ['all_state_partition_functions', 'weighted_adjacent_differences',
                          'group_closure', 'total_closure', 'independent_atomic_integrals',
                          'energy_offset_invariance', 'wrong_ensemble_negative_control'],
        'exact_state_free_energies_relative_to_S0': (exact_f - exact_f[0]).tolist(),
        'estimated_state_free_energies_relative_to_S0': (f - f[0]).tolist(),
        'group_delta_f': np.diff(f).tolist(), 'atom_delta_f': parts.tolist(),
        'total_delta_f': float(f[-1] - f[0]),
        'total_closure_error': float(abs(parts.sum() - (f[-1] - f[0]))),
        'uncorrected_maximum_step_error': wrong_error,
        'adjacent_overlap': overlaps,
        'physical_validation': 'not_assessed',
    }
    Path(__file__).with_name('math_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
