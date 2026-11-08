"""Input and temperature integration, including wrong-temperature negative controls."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import openmm

ROOT = Path(__file__).resolve().parents[2]
ETA = ROOT / "experiments/2026-10-25_eta_hmc"
sys.path[:0] = [str(ETA), str(ROOT / "experiments/2026-10-26_mm_bridge")]
import eta_hmc as eh
import bridge_mm as bm
from eta_records import run_recorded
from eta_exchange import read_ladder, run_ladder
from test_atomic_records import AnalyticEnergy, contract, synthetic_record, toy_context


class InputTests(unittest.TestCase):
    def test_acetamide_mapping_units_and_default_ibs_temperature(self):
        xml, lig, groups, xyz, box = eh.gm.leg("acetamide")
        self.assertEqual(len(lig), 9)
        self.assertEqual(sorted(i for g in groups for i in g), lig)
        self.assertEqual(xyz.shape, (2634, 3))
        source = np.load(xml.parent / "em_acetamide.npz", allow_pickle=False)
        np.testing.assert_allclose(xyz * 10, source["positions"], rtol=0, atol=1e-13)
        np.testing.assert_allclose(np.diag(box) * 10, source["box"], rtol=0, atol=1e-13)
        self.assertEqual(eh.gm.temperature("acetamide"), 298.15)
        self.assertEqual(eh.gm.temperature("CyclophilinD_ligand-8"), 300.0)
        self.assertEqual(eh.proposal_integrator(2, 2).getGlobalVariableByName("kT"), eh.KT)
        self.assertEqual(eh.proposal_integrator(2, 2, 298.15).getGlobalVariableByName("kT"), eh.gm.R * 298.15)

    def test_model_topology_uses_the_same_source(self):
        from model_eval import topology
        top = topology("acetamide")
        self.assertEqual(len(top[0]), 9)
        self.assertEqual(top[3][:9].tolist(), [6, 6, 7, 8, 1, 1, 1, 1, 1])
        self.assertEqual(len(top[-1]), 2634)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                eh.gm.thermal_energy(value)


class TemperatureTests(unittest.TestCase):
    def test_endpoint_rescore_rejects_wrong_temperature_before_model(self):
        from model_eval import load_model
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "runs/acetamide"
            source.mkdir(parents=True)
            np.savez(source / "rep1_S0.npz", T_K=300.)
            fake_leg = SimpleNamespace(lig=list(range(9)), K=2, temperature_K=298.15)
            args = SimpleNamespace(system="acetamide", n_water=50, max_frames=None)
            with mock.patch.object(eh.local_E, "GW", root), mock.patch.object(eh.local_E, "HERE", root / "output"), \
                 mock.patch.object(eh.local_E, "Leg", return_value=fake_leg), \
                 mock.patch("model_eval.load_model", wraps=load_model) as loader:
                with self.assertRaisesRegex(ValueError, "temperature"):
                    eh.local_E.evaluate(args)
                loader.assert_not_called()

    def test_298K_recorded_ladder_uses_one_temperature(self):
        pairs = [toy_context() for _ in range(3)]
        ctxs, igs = map(list, zip(*pairs))
        rt = eh.gm.thermal_energy(298.15)
        for ig in igs:
            ig.setGlobalVariableByName("kT", rt)
        target = {**contract(), "T_K": 298.15}
        energy = AnalyticEnergy()
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            path = Path(td) / "ladder.npz"
            meta = run_ladder(ctxs, igs, energy, [np.random.default_rng(i) for i in range(3)],
                              np.random.default_rng(99), [np.zeros((5, 3)) for _ in range(3)],
                              [0., .5, 1.], 4, path, target, eh.hmc, rt, checkpoint_every=2, coords_every=1)
            checked, data = read_ladder(path)
        self.assertTrue(meta["complete"])
        self.assertEqual(checked["T_K"], 298.15)
        self.assertEqual(energy.calls, 15)
        self.assertEqual(data["dU_atom"].shape, (4, 3, 2))

    def test_record_and_exchange_reject_mixed_temperature_before_queries(self):
        c, ig = toy_context()
        energy = AnalyticEnergy()
        bad = {**contract(), "T_K": 298.15}
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "integrator temperature"):
                run_recorded(c, ig, energy, np.random.default_rng(0), np.zeros((5, 3)), 1, 4,
                             Path(td) / "chain.npz", bad, eh.hmc)
            rt = eh.gm.thermal_energy(298.15)
            with self.assertRaisesRegex(ValueError, "integrator temperature"):
                run_ladder([c, c], [ig, ig], energy, [None, None], None, [None, None], [0, 1], 4,
                           Path(td) / "ladder.npz", bad, eh.hmc, rt)
            with self.assertRaisesRegex(ValueError, "exchange temperature"):
                run_ladder([c, c], [ig, ig], energy, [None, None], None, [None, None], [0, 1], 4,
                           Path(td) / "ladder.npz", bad, eh.hmc, eh.KT)
        self.assertEqual(energy.calls, 0)

    def test_hmc_and_mm_acceptance_use_integrator_temperature(self):
        # Fixed reversible proposal work, with u strictly between the 298.15 K and 300 K criteria.
        class Integrator:
            def __init__(self):
                self.values = {"kT": eh.gm.thermal_energy(298.15), "Enew": 1., "Eold": 0., "Vnew": 1., "Vold": 0.}

            def getGlobalVariableByName(self, name):
                return self.values[name]

            def setGlobalVariableByName(self, name, value):
                self.values[name] = value

            def step(self, n):
                pass

        x = np.zeros((1, 3))
        ctx = mock.Mock()
        ig = Integrator()
        u = np.exp(-0.5 * (1 / ig.values["kT"] + 1 / eh.KT))
        rng = mock.Mock(random=lambda: u)
        with mock.patch.object(eh, "positions", return_value=x):
            out = eh.hmc(ctx, ig, x, 0, 1, lambda _: 0., rng, initial={"x": x, "e": 0.})
            self.assertFalse(out["acc"][0])
            self.assertTrue(np.log(u) < -1 / eh.KT, "the old 300 K criterion would accept")
            _, _, _, acc = eh.mm_updates(ctx, ig, x, 1, rng)
            self.assertFalse(acc[0])

    def test_scalar_analysis_converts_bar_with_the_recorded_temperature(self):
        from atomic_report import report
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for eta in (0., 1.):
                path = Path(td) / f"eta{eta}.npz"
                synthetic_record(path, eta, 1, int(eta) + 3)
                z = np.load(path, allow_pickle=False)
                data = {k: z[k] for k in z.files}
                meta = json.loads(str(data["meta"]))
                meta["T_K"] = 298.15
                data["meta"] = json.dumps(meta)
                np.savez(path, **data)
                paths.append(path)
            args = SimpleNamespace(discard=0, system="acetamide", state=0, expect_etas="0,1", expect_starts="1")
            with mock.patch("pymbar.other_estimators.bar", return_value={"Delta_f": 2., "dDelta_f": .1}), contextlib.redirect_stdout(io.StringIO()):
                result = eh.analyze_target(args, paths)
                atomic = report(paths, [0., 1.], [1], discard=0., bootstrap=0)
            self.assertAlmostEqual(result["BAR_sum"], 2 * eh.gm.thermal_energy(298.15))
            self.assertAlmostEqual(atomic["a_BAR"], 2 * eh.gm.thermal_energy(298.15))

    def test_bridge_analysis_rejects_window_temperature_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text(json.dumps({"T_K": 298.15, "groups": [1],
                "windows": [{"label": "S0"}, {"label": "S1"}]}))
            for i in range(2):
                np.savez(root / f"w{i:03d}_S{i}.npz", meta=json.dumps({"T_K": 300., "kT_kJmol": eh.KT}))
            with mock.patch.object(bm, "run_dir", return_value=root), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    bm.analyze(SimpleNamespace(system="acetamide", rep=1, tag=""))


if __name__ == "__main__":
    unittest.main()
