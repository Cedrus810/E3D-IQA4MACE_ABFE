"""Compare with exact state partition functions, not with a model target."""

import unittest

import numpy as np

from step_fep import KB, estimate_step, exponential_free_energy


class PhysicalStepTests(unittest.TestCase):
    def setUp(self):
        self.temperature = 300.0
        self.kt = KB * self.temperature
        self.p_before = np.array([0.5, 0.3, 0.2])
        self.p_after = np.array([0.2, 0.4, 0.4])
        self.offset = 2.3
        before = -self.kt * np.log(self.p_before)
        after = -self.kt * np.log(self.p_after) + self.offset
        self.work_unique = after - before
        self.before_index = np.repeat(np.arange(3), (1000 * self.p_before).astype(int))
        self.after_index = np.repeat(np.arange(3), (1000 * self.p_after).astype(int))
        self.work = self.work_unique[self.before_index]
        self.reverse = -self.work_unique[self.after_index]
        # Deliberately imperfect learned coupling split; residual must correct it.
        d_unique = np.array([[-3.0, 0.5], [-0.2, -0.3], [1.4, -2.0]])
        self.d = d_unique[self.before_index]

    def test_actual_partition_difference_forward_and_reverse(self):
        result = estimate_step(self.work, self.temperature, work_reverse_kj=self.reverse)
        self.assertAlmostEqual(result["delta_g_kj_mol"], self.offset, places=12)
        self.assertAlmostEqual(result["delta_g_reverse_exp_kj_mol"], self.offset, places=12)
        self.assertGreater(abs(self.work.mean() - self.offset), 0.1)

    def test_atom_and_residual_integrals_close_without_rescaling(self):
        result = estimate_step(self.work, self.temperature, self.d)
        self.assertTrue(result["numerical_integration_success"])
        self.assertAlmostEqual(sum(result["atom_delta_g_kj_mol"]) + result["residual_delta_g_kj_mol"], self.offset, places=10)
        self.assertLess(abs(result["closure_error_kj_mol"]), 1e-10)
        self.assertGreater(abs(result["residual_delta_g_kj_mol"]), 0.01)
        # Independent direct sum over the three configurations, Gauss-Legendre.
        d_unique = self.d[np.r_[0, 500, 800]]
        atom_reference = np.zeros(2)
        nodes, weights = np.polynomial.legendre.leggauss(128)
        for t, q in zip((nodes + 1) / 2, weights / 2):
            probability = self.p_before * np.exp(-t * self.work_unique / self.kt)
            probability /= probability.sum()
            atom_reference += q * (probability @ -d_unique)
        np.testing.assert_allclose(result["atom_delta_g_kj_mol"], atom_reference, atol=1e-10)

    def test_changing_ml_scores_cannot_change_physical_total(self):
        first = estimate_step(self.work, self.temperature, self.d)
        changed = estimate_step(self.work, self.temperature, self.d * 20 + 50)
        self.assertEqual(first["delta_g_kj_mol"], changed["delta_g_kj_mol"])
        self.assertAlmostEqual(changed["component_sum_kj_mol"], self.offset, places=9)
        self.assertGreater(abs(changed["residual_delta_g_kj_mol"] - first["residual_delta_g_kj_mol"]), 1)

    def test_constant_work_and_large_offsets(self):
        result = estimate_step(np.full(10, 5.0), self.temperature, np.tile([-3.0, -4.0], (10, 1)))
        np.testing.assert_allclose(result["atom_delta_g_kj_mol"], [3.0, 4.0], atol=1e-12)
        self.assertAlmostEqual(result["residual_delta_g_kj_mol"], -2.0, places=12)
        self.assertAlmostEqual(exponential_free_energy(self.work + 10000, self.temperature), self.offset + 10000, places=10)

    def test_reverse_disagreement_is_reported_not_averaged(self):
        result = estimate_step(self.work, self.temperature, work_reverse_kj=np.full(5, 10000.0))
        self.assertAlmostEqual(result["delta_g_kj_mol"], self.offset, places=12)
        self.assertAlmostEqual(result["delta_g_reverse_exp_kj_mol"], -10000.0, places=10)
        self.assertGreater(result["forward_minus_reverse_kj_mol"], 10000)

    def test_bad_inputs_do_not_silently_drop_frames(self):
        for work in ([], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                exponential_free_energy(work, self.temperature)
        with self.assertRaises(ValueError):
            estimate_step(self.work, self.temperature, self.d[:-1])
        with self.assertRaises(ValueError):
            exponential_free_energy(self.work, 0)


if __name__ == "__main__":
    unittest.main()
