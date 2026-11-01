"""Independent finite-state partition functions test offline endpoint inference."""

import unittest

import numpy as np

from estimate import analyze


def fixture():
    # Integer multiplicities exactly integrate two different source distributions.
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    index = np.concatenate([np.repeat(np.arange(3), (p * 1000).astype(int)) for p in probabilities])
    source = -np.log(probabilities)
    target = np.array([[0.2, 0.7, 2.4], [1.5, 0.3, 0.8]])
    atomic = np.array([[0.4, -0.3, 0.1], [-0.1, 0.5, 0.2]])
    residual = target[0] - target[1] - atomic.sum(axis=0)
    components = np.vstack([atomic, residual])
    exact = -np.log(np.exp(-target[1]).sum() / np.exp(-target[0]).sum())
    return source[:, index], target[:, index], components[:, index], exact


class EndpointTests(unittest.TestCase):
    def test_different_sources_recover_exact_partition_ratio_and_components(self):
        m, v, c, exact = fixture()
        result = analyze(m, v, c, [1000, 1000])
        self.assertEqual(result["status"], "diagnostics_passed")
        self.assertAlmostEqual(result["delta_f_nbb"], exact, places=10)
        self.assertAlmostEqual(result["delta_f"], exact, places=10)
        self.assertAlmostEqual(sum(result["component_delta_f"]), exact, places=10)
        # Direct target Boltzmann integration on the three unique states,
        # independent of source normalization or Bennett weights.
        _, unique = np.unique(v, axis=1, return_index=True)
        vv, cc = v[:, unique], c[:, unique]
        nodes, weights = np.polynomial.legendre.leggauss(96)
        direct = np.zeros(len(c))
        for t, w in zip((nodes + 1) / 2, weights / 2):
            p = np.exp(-((1 - t) * vv[1] + t * vv[0]))
            direct -= w * (cc @ (p / p.sum()))
        np.testing.assert_allclose(result["component_delta_f"], direct, atol=1e-10)

    def test_omitting_source_correction_gives_wrong_answer(self):
        m, v, c, exact = fixture()
        wrong = analyze(v, v, c, [1000, 1000])
        self.assertGreater(abs(wrong["delta_f_nbb"] - exact), 0.1)

    def test_energy_zero_and_source_offset_invariance(self):
        m, v, c, _ = fixture()
        baseline = analyze(m, v, c, [1000, 1000])
        shifted = analyze(m + np.array([[20.0], [-13.0]]) + 1e4, v + 1e4, c, [1000, 1000])
        np.testing.assert_allclose(shifted["component_delta_f"], baseline["component_delta_f"], atol=1e-9)
        self.assertAlmostEqual(shifted["delta_f"], baseline["delta_f"], places=9)

    def test_unequal_source_counts(self):
        m, v, c, exact = fixture()
        idx = np.r_[np.arange(1000), np.tile(np.arange(1000, 2000), 2)]
        result = analyze(m[:, idx], v[:, idx], c[:, idx], [1000, 2000])
        self.assertAlmostEqual(result["delta_f"], exact, places=10)

    def test_disconnected_endpoints_are_rejected_even_with_algebraic_closure(self):
        v = np.array([[0.0] * 100 + [1e6] * 100, [1e6] * 100 + [0.0] * 100])
        result = analyze(v, v, (v[0] - v[1])[None, :], [100, 100])
        self.assertEqual(result["status"], "insufficient_support")
        self.assertIn("endpoint_overlap", result["support_failures"])

    def test_bad_accounting_and_nonfinite_input_are_rejected(self):
        m, v, c, _ = fixture()
        with self.assertRaisesRegex(ValueError, "does not close"):
            analyze(m, v, c[:-1], [1000, 1000])
        m[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            analyze(m, v, c, [1000, 1000])


if __name__ == "__main__":
    unittest.main()
