"""Negative controls for loss of atomic identity despite unchanged scalar energies."""

import unittest
import numpy as np
from audit import components


class AtomicIdentityTests(unittest.TestCase):
    def setUp(self):
        self.parts = {"C_ML": np.array([3., 6.]), "C_ML_learned": np.array([2., 4.]),
                      "C_ML_coul": np.array([1., 2.]), "C_MM": np.array([[0.5, 1.], [1., 2.]]),
                      "dU_atom": np.array([1.5, 3.]), "omega_water_resid": np.arange(50)}

    def test_swapped_atomic_rows_fail_even_with_correct_scalar_total(self):
        self.assertEqual(components(self.parts, 4.5)["dU"], 4.5)
        self.parts["dU_atom"] = self.parts["dU_atom"][::-1]
        self.assertEqual(self.parts["dU_atom"].sum(), 4.5)
        with self.assertRaisesRegex(ValueError, "does not close"):
            components(self.parts, 4.5)

    def test_double_counted_coulomb_fails(self):
        self.parts["C_ML"] += self.parts["C_ML_coul"]
        self.parts["dU_atom"] += self.parts["C_ML_coul"]
        with self.assertRaisesRegex(ValueError, "does not close"):
            components(self.parts, self.parts["dU_atom"].sum())

    def test_duplicate_water_identity_fails_with_unchanged_energies(self):
        self.parts["omega_water_resid"][-1] = 0
        with self.assertRaisesRegex(ValueError, "unique water"):
            components(self.parts, 4.5)


if __name__ == "__main__":
    unittest.main()
