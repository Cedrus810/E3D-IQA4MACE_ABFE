"""CPU checks for the K2 query and identity (no model weights, no MD). python -m unittest test_k2_energy -v"""

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

import k2_energy as k2
from eta_records import check_contract


def fake_backbone(*class_names):
    mods = [(f"m{i}", type(n, (), {"offset": 0.1})()) for i, n in enumerate(class_names)]
    return types.SimpleNamespace(named_modules=lambda: iter([("", object())] + mods))


STENCIL = ("RealSpaceFiniteDiffereneEnergy", "RealSpaceFiniteDifferenceElectrostaticFeatures")


class FakeLeg:
    def __init__(self, n_lig=3, q=(0.4, -0.2, -0.2), dummy_j=None):
        self.lig = list(range(n_lig))
        self.q = np.r_[np.asarray(q, float), np.zeros(9)]
        self.top = (None, None, None, None, np.r_[np.zeros(n_lig, int), np.repeat([1, 2, 3], 3)])
        self.K = 2
        self._gidx = [[0], [1, 2]]

    def dummy(self, j):
        d = np.zeros(len(self.lig), bool)
        for g in self._gidx[:j]:
            d[g] = True
        return d

    def carve(self, x, n):
        return x * 10, np.ones(len(x), int), np.r_[np.zeros(3, int), np.ones(len(x) - 3, int)], np.arange(len(x))


class TestCompose(unittest.TestCase):
    def test_closure_and_keys(self):
        rng = np.random.default_rng(3)
        ch, lr = rng.normal(size=4), rng.normal(size=4)
        mm = rng.normal(size=(4, 2))
        du, comp = k2.compose(ch, lr, ch - lr, mm, -7.25, np.zeros(4, bool), np.arange(10), [5, 6])
        self.assertAlmostEqual(comp["d_atom"].sum() + float(comp["q"]), du, places=12)
        self.assertAlmostEqual(du, -7.25 - mm.sum(), places=12)
        self.assertAlmostEqual(float(comp["q"]), -7.25 - ch.sum(), places=12)
        self.assertEqual(set(comp), set(k2.K2_DETAILS))
        self.assertNotIn("dU_atom", comp)        # the legacy key's sum was the old scalar; must not be reused

    def test_head_change_moves_only_attribution(self):
        mm = np.ones((3, 2))
        du1, c1 = k2.compose(np.array([1., 2, 3]), np.zeros(3), np.zeros(3), mm, -4., np.zeros(3, bool), [0], [0])
        du2, c2 = k2.compose(np.array([-5., 0, 9]), np.zeros(3), np.zeros(3), mm, -4., np.zeros(3, bool), [0], [0])
        self.assertEqual(du1, du2)
        self.assertNotEqual(float(c1["q"]), float(c2["q"]))

    def test_dummy_rows_and_nonfinite(self):
        dm = np.array([True, False])
        with self.assertRaises(ValueError):
            k2.compose(np.array([1., 0]), np.zeros(2), np.zeros(2), np.zeros((2, 2)), 0., dm, [0], [0])
        with self.assertRaises(FloatingPointError):
            k2.compose(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros((2, 2)), np.nan, np.zeros(2, bool), [0], [0])


class TestEndpoints(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(k2.endpoint_kind(np.zeros(3, bool)), "S0")
        self.assertEqual(k2.endpoint_kind(np.ones(3, bool)), "SK")
        with self.assertRaises(NotImplementedError):
            k2.endpoint_kind(np.array([True, False, False]))

    def _energy(self, leg, j):
        fake_model = types.SimpleNamespace(backbone=fake_backbone(*STENCIL))
        fake_model.to = lambda dtype: fake_model
        mod = types.SimpleNamespace(load_model=lambda **kw: fake_model)
        with patch.dict(sys.modules, {"model_eval": mod}), patch("k2_energy.os.path.getmtime", return_value=0.0):
            return k2.ModelEnergyK2(leg, j, head="/h.pt", backbone="/b.model", device="cpu")

    def test_all_dummy_is_zero_without_model_call(self):
        me = self._energy(FakeLeg(), 2)
        x = np.random.default_rng(0).normal(size=(12, 3))
        self.assertEqual(me(x), 0.0)
        self.assertEqual(me.calls, 0)
        self.assertEqual(float(me.last_components["q"]), 0.0)
        self.assertTrue(np.all(me.last_components["d_atom"] == 0))

    def test_partial_dummy_and_charged_ligand_refused(self):
        with self.assertRaises(NotImplementedError):
            self._energy(FakeLeg(), 1)
        with self.assertRaises(NotImplementedError):
            self._energy(FakeLeg(q=(1.0, 0.0, 0.0)), 0)

    def test_requires_explicit_paths_and_float64(self):
        with self.assertRaises(ValueError):
            k2.ModelEnergyK2(FakeLeg(), 0, head=None, backbone="/b")
        with self.assertRaises(ValueError):
            k2.ModelEnergyK2(FakeLeg(), 0, head="/h", backbone="/b", dtype_name="float32")


class TestIdentity(unittest.TestCase):
    def test_backend_labels(self):
        self.assertEqual(k2.backend_identity(fake_backbone(*STENCIL))["realspace_backend"], "graph_longrange_stencil")
        self.assertEqual(k2.backend_identity(fake_backbone("AnalyticEnergy", "AnalyticFeatures"))["realspace_backend"],
                         "analytic_l01_v1")
        with self.assertRaises(ValueError):
            k2.backend_identity(fake_backbone(STENCIL[0], "AnalyticFeatures"))
        with self.assertRaises(ValueError):
            k2.backend_identity(fake_backbone("Linear"))

    def test_contract_refuses_mixing_both_ways(self):
        old = {"E_construction": "local_E v1", "backbone": "/b", "head": "/h", "dtype": "float64"}
        new = {"E_construction": k2.TARGET, "attribution_definition": k2.ATTRIBUTION, "backbone": "/b", "head": "/h",
               "realspace_backend": "graph_longrange_stencil", "dtype": "float64", "head_mtime": 1.0}
        with self.assertRaises(ValueError):
            check_contract(old, new)              # legacy chain cannot be continued as K2
        with self.assertRaises(ValueError):
            check_contract(new, old)              # K2 chain cannot be continued as legacy
        with self.assertRaises(ValueError):
            check_contract(new, {**new, "realspace_backend": "analytic_l01_v1"})
        check_contract(new, {**new, "head_mtime": 2.0})   # mtime is provenance only


if __name__ == "__main__":
    unittest.main()
