"""CPU checks for K2 chain records, restart and report (toy OpenMM system, analytic K2 target; no model).
python -m unittest test_k2_records -v"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import openmm

import eta_hmc as eh
import eta_records
import k2_energy as k2
import k2_records as kr
from k2_report import report


class AnalyticK2:
    """Two ligand atoms; C_bb != sum C_H so q is a nonzero, configuration-dependent remainder."""
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        ch = np.array([400 * x[0, 0] ** 2, -10 * x[1, 0]])
        mm = np.array([[0.3 * x[0, 1], 0.1], [0.0, -0.2 * x[1, 1]]])
        cbb = ch.sum() + 5.0 * x[0, 2] - 1.5
        du, comp = k2.compose(ch, 0.75 * ch, 0.25 * ch, mm, cbb, np.zeros(2, bool), np.arange(5), [17])
        self.last_components = comp
        return du


def toy_context():
    system = openmm.System()
    f = openmm.CustomExternalForce("0.5*50*(x*x+y*y+z*z)")
    for i in range(5):
        system.addParticle(12.0)
        f.addParticle(i, [])
    system.addForce(f)
    ig = eh.proposal_integrator(4, 10.0)
    ig.setRandomNumberSeed(42)
    c = openmm.Context(system, ig, openmm.Platform.getPlatformByName("Reference"))
    c.setPeriodicBoxVectors(openmm.Vec3(3, 0, 0), openmm.Vec3(0, 3, 0), openmm.Vec3(0, 0, 3))
    return c, ig


def contract(eta=1.0, start="a"):
    return {"system": "analytic_harmonic", "state": 0, "leg": "solvent", "T_K": 300.0, "eta": eta, "start_label": start,
            "L": 4, "dt_fs": 10.0, "E_construction": kr.TARGET, "attribution_definition": k2.ATTRIBUTION,
            "dtype": "float64", "backbone": "analytic", "head": "analytic", "realspace_backend": "none",
            "realspace_modules": [], "total_charge": 0, "omega_rule": "toy", "n_particles": 5, "n_water": 1,
            "ligand_system_indices": [0, 1], "groups": [[0], [1]], "dummy_mask": [False, False],
            "box_nm": (np.eye(3) * 3).tolist(), "system_xml": "toy", "softcore_alpha": 0.5,
            "system_definition": "five-particle harmonic k=50 m=12", "context_parameters": {},
            "energy_units": "kJ/mol", "xyz_units": "nm"}


def run_toy(out, kernel=eh.hmc, eta=1.0, start="a", seed=19, cycles=40, device=None):
    c, ig = toy_context()
    energy = AnalyticK2()
    x0 = np.zeros((5, 3))
    x0[0, 0], x0[1, 0], x0[0, 2] = 0.15, -0.1, 0.05
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            meta = kr.run_recorded(c, ig, energy, np.random.default_rng(seed), x0, eta, cycles, out,
                                   contract(eta, start), kernel, checkpoint_every=7, coords_every=3,
                                   device=device)
        return meta, energy.calls
    finally:
        del c, ig


class TestK2Records(unittest.TestCase):
    def test_accounting_rejections_and_validation(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "k2.npz"
            meta, calls = run_toy(p)
            m, z = kr.read_record(p)
            self.assertTrue(meta["complete"])
            self.assertEqual(calls, 41)
            self.assertTrue(z["acc"].any() and (~z["acc"]).any())
            np.testing.assert_allclose(z["d_atom"].sum(1) + z["q"], z["dU"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(z["C_bb"] - z["C_MM"].sum((1, 2)), z["dU"], atol=1e-12, rtol=0)
            self.assertGreater(np.ptp(z["q"]), 1e-3)          # remainder really varies with configuration
            for i in np.flatnonzero(~z["acc"]):
                if i:
                    self.assertEqual(z["q"][i], z["q"][i - 1])
                    np.testing.assert_array_equal(z["d_atom"][i], z["d_atom"][i - 1])
            # acceptance used dU_K2: proposal dU = previous state + dE
            np.testing.assert_allclose((z["prop_d_atom"].sum(1) + z["prop_q"])[1:], z["dU"][:-1] + z["dE"][1:],
                                       atol=1e-10, rtol=0)
            for key, msg in (("q", "q disagrees"), ("d_atom", "d_atom disagrees"), ("C_bb", "q disagrees")):
                bad = {**z, key: z[key].copy()}
                bad[key][0, ...] += 1
                with self.assertRaisesRegex(ValueError, msg):
                    kr.validate_record(bad)

    def test_checkpoint_replays_uncommitted_chunk(self):
        with tempfile.TemporaryDirectory() as td:
            ref, cut = Path(td) / "ref.npz", Path(td) / "cut.npz"
            run_toy(ref)
            chunks = 0

            def fail_inside_second(*args, **kw):
                nonlocal chunks
                chunks += 1
                if chunks == 2:
                    args = list(args)
                    args[4] = 3
                    eh.hmc(*args, **kw)
                    raise RuntimeError("simulated process interruption")
                return eh.hmc(*args, **kw)

            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run_toy(cut, fail_inside_second)
            progress = cut.with_name("cut.progress.npz")
            self.assertEqual(kr.read_record(progress)[0]["completed_cycles"], 7)
            run_toy(cut)
            _, a = kr.read_record(ref)
            _, b = kr.read_record(cut)
            for k in ("acc", "dH", "dE", "dU", "V", "d_atom", "q", "C_bb", "C_H", "lig_xyz", "xyz", "prop_q"):
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)
            self.assertEqual(run_toy(cut)[1], 0)

    def test_schemas_refuse_each_other(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "k2.npz"
            run_toy(p)
            with np.load(p) as z:
                data = {k: z[k] for k in z.files}
            with self.assertRaises((ValueError, KeyError)):
                eta_records.validate_record(data)             # legacy tools cannot read K2 records
            legacy = dict(data)
            m = json.loads(str(legacy["meta"]))
            m["storage_schema"], m["E_construction"] = eta_records.SCHEMA, "local_E v1"
            legacy["meta"] = json.dumps(m)
            with self.assertRaisesRegex(ValueError, "not a k2_1"):
                kr.validate_record(legacy)
            before = p.read_bytes()
            with self.assertRaisesRegex(ValueError, "contract differs"):
                c, ig = toy_context()
                with contextlib.redirect_stdout(io.StringIO()):
                    kr.run_recorded(c, ig, AnalyticK2(), np.random.default_rng(1), np.zeros((5, 3)), 1.0, 40, p,
                                    {**contract(), "realspace_backend": "analytic_l01_v1"}, eh.hmc, 7, 3)
            self.assertEqual(p.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "require E_construction"):
                kr.run_recorded(None, None, None, None, None, 1.0, 1, p, {**contract(), "E_construction": "local_E v1"},
                                eh.hmc)


class TestK2Device(unittest.TestCase):
    def test_resume_only_on_writing_gpu_model(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.npz"
            n = 0

            def fail(*args, **kw):
                nonlocal n
                n += 1
                if n == 2:
                    raise RuntimeError("stop")
                return eh.hmc(*args, **kw)

            with self.assertRaises(RuntimeError):
                run_toy(p, fail, device="gpuA")
            progress = p.with_name("d.progress.npz")
            before = progress.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "same GPU model"):
                run_toy(p, device="gpuB")
            self.assertEqual(progress.read_bytes(), before)
            run_toy(p, device="gpuA")
            self.assertEqual(run_toy(p, device="gpuB")[1], 0)      # a completed record is reusable anywhere


class TestK2Report(unittest.TestCase):
    def test_atoms_plus_remainder_close_and_remainder_not_apportioned(self):
        with tempfile.TemporaryDirectory() as td:
            files, means = [], {}
            for e in (0.0, 0.5, 1.0):
                rows = []
                for s in ("a", "b"):
                    p = Path(td) / f"e{e}_{s}.npz"
                    run_toy(p, eta=e, start=s, seed=11 + int(10 * e) + ord(s), cycles=60)
                    _, z = kr.read_record(p)
                    rows.append(np.c_[z["d_atom"][12:], z["q"][12:]])
                    files.append(p)
                means[e] = np.concatenate(rows).mean(axis=0)
            r = report(files, [0, 0.5, 1], ["a", "b"], discard=0.2, bootstrap=8, block_length=5, seed=3)
            expected = 0.25 * means[0] + 0.5 * means[0.5] + 0.25 * means[1]
            np.testing.assert_allclose(r["A_H_atom_TI"], expected[:2], atol=1e-12, rtol=0)
            self.assertAlmostEqual(r["R_TI"], expected[2], places=12)
            self.assertAlmostEqual(r["atom_plus_R_minus_scalar_TI"], 0.0, places=10)
            self.assertAlmostEqual(r["eps_eta_BAR_minus_atom_plus_R_TI"], r["a_BAR"] - expected.sum(), places=10)
            self.assertIn("R_quantiles", r["block_bootstrap"])
            self.assertFalse(report(files[:-1], [0, 0.5, 1], ["a", "b"])["complete"])


if __name__ == "__main__":
    unittest.main()
