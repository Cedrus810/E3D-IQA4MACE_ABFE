import unittest
import numpy as np
from fidelity import learned_atoms, nearest_water, response, rotate


class DiagnosticTests(unittest.TestCase):
    def test_water_selection_uses_imaged_geometry(self):
        # Water 10 is near the ligand after imaging; choosing from unwrapped
        # coordinates would prefer water 20. The selector consumes carved data.
        p = np.array([[0., 0., 0.], [.1, 0., 0.], [.1, .05, 0.], [.1, 0., .05],
                      [1., 0., 0.], [1., .05, 0.], [1., 0., .05]])
        np.testing.assert_array_equal(nearest_water(p, np.array([0, 10, 11, 12, 20, 21, 22]), 1), [10, 11, 12])

    def test_constant_bias_and_wrong_response_are_distinguished(self):
        reference = np.array([-3., -1., 2., 4.])
        true_energy = reference + 100
        shifted_head = true_energy - 20
        good = response(shifted_head - (100 - 20), reference)
        self.assertEqual(good["rmse_error"], 0)
        self.assertEqual(good["slope_through_origin"], 1)
        bad = response(np.zeros(4), reference)
        self.assertEqual(bad["slope_through_origin"], 0)
        self.assertGreater(bad["rmse_error"], 2)

    def test_directed_pair_reduction_excludes_internal_edges(self):
        frag = np.array([0, 0, 1])
        edges = np.array([[0, 2, 1, 2, 0, 1], [2, 0, 2, 1, 1, 0]])
        D = np.array([[1., 2.], [1., 2.], [2., 3.], [2., 3.], [100., 100.], [100., 100.]])
        np.testing.assert_array_equal(learned_atoms(D, frag, edges, 2), [3., 5.])
        wrong_frag = np.array([0, 1, 1])
        self.assertFalse(np.array_equal(learned_atoms(D, wrong_frag, edges, 2), [3., 5.]))

    def test_rigid_probe_preserves_distances_and_inverts(self):
        x = np.array([[0., 0., 0.], [.1, .2, .3], [.4, .1, .2], [2., 2., 2.]])
        ids = np.array([0, 1, 2])
        center = x[ids].mean(0)
        y = rotate(x, ids, center, 1, .3)
        np.testing.assert_allclose(np.linalg.norm(y[0]-y[1]), np.linalg.norm(x[0]-x[1]), atol=1e-15)
        np.testing.assert_allclose(rotate(y, ids, center, 1, -.3), x, atol=1e-15)
        np.testing.assert_array_equal(y[3], x[3])


if __name__ == "__main__":
    unittest.main()
