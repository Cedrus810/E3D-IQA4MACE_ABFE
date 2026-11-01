"""Panel initialization contracts and Reference replay; no GPU or molecular sampling."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import eta_hmc as eh
from compare_ladders import start_ligands
from eta_exchange import (DETAILS, check_panel_contract, load_initial_panel, panel_checked_kernel,
                          read_ladder, run_ladder)
from test_atomic_records import AnalyticEnergy, contract, toy_context


def toy_panel(path):
    x = np.zeros((5, 3))
    x[0, 0], x[1, 0] = .15, -.1
    energy = AnalyticEnergy()
    expected = energy(x)
    meta = {"schema": "e3d-deep-shallow-panel-1", "contract": contract(),
            "frames": [{"source": "toy", "sweep": 1000, "walker": 0, "rep": 1}]}
    np.savez(path, meta=json.dumps(meta), xyz_nm=x[None], expected_dU=[expected],
             **{k: np.asarray(energy.last_components[k])[None] for k in DETAILS})
    return load_initial_panel(path, 0)


def run(out, x, info, sweeps=12, extend_from=None):
    pairs = [toy_context() for _ in range(3)]
    for w, (_, ig) in enumerate(pairs):
        ig.setRandomNumberSeed(42 + w)
    energy = AnalyticEnergy()
    try:
        m = run_ladder([p[0] for p in pairs], [p[1] for p in pairs], energy,
                       [np.random.default_rng(19 + w) for w in range(3)], np.random.default_rng(99),
                       [x.copy() for _ in pairs], [0., .5, 1.], sweeps, out,
                       {**contract(), "panel_initialization": info}, panel_checked_kernel(info), eh.KT,
                       checkpoint_every=4, coords_every=2, extend_from=extend_from)
        return m, energy.calls
    finally:
        pairs.clear()


class TestPanelInitialization(unittest.TestCase):
    def test_source_contract_and_analysis_without_external_panel(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "panel.npz"
            x, c, info = toy_panel(p)
            check_panel_contract(c, contract())
            with self.assertRaisesRegex(ValueError, "box_nm"):
                check_panel_contract(c, {**contract(), "box_nm": np.eye(3).tolist()})
            with self.assertRaisesRegex(ValueError, "index"):
                load_initial_panel(p, -1)
            p.unlink()
            m = {**contract(), "etas": [0., .5, 1.], "panel_initialization": info}
            for lig in start_ligands(m):
                np.testing.assert_array_equal(lig, x[:2])

    def test_initial_query_budget_bad_cache_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            x, _, info = toy_panel(td / "panel.npz")
            _, calls = run(td / "full.npz", x, info)
            self.assertEqual(calls, 3 * 13)  # initialization check adds no model queries
            run(td / "short.npz", x, info, sweeps=8)
            _, calls = run(td / "continued.npz", x, info, extend_from=td / "short.npz")
            self.assertEqual(calls, 3 + 3 * 4)  # rechecks plus extra sweeps
            _, full = read_ladder(td / "full.npz")
            _, continued = read_ladder(td / "continued.npz")
            for key in ("dU", "dU_atom", "acc", "dH", "perm", "xyz", "prop_xyz"):
                np.testing.assert_array_equal(full[key], continued[key], err_msg=key)
            self.assertEqual(run(td / "full.npz", x, info)[1], 0)
            # A changed solvent coordinate must not silently reuse a completed result.
            changed = json.loads(json.dumps(info))
            changed["positions_nm"][4][0] = .01
            with self.assertRaisesRegex(ValueError, "panel_initialization"):
                run(td / "full.npz", x, changed)
            for key in ("expected_dU", "expected_components"):
                bad = json.loads(json.dumps(info))
                if key == "expected_dU":
                    bad[key] += .01
                else:
                    bad[key]["omega_water_resid"][0] += 1
                with self.assertRaisesRegex(ValueError, "panel initial state"):
                    run(td / f"bad_{key}.npz", x, bad)
                self.assertFalse((td / f"bad_{key}.npz").exists())


if __name__ == "__main__":
    unittest.main()
