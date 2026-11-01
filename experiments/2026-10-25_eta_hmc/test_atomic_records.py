"""CPU tests of accepted-state recording, checkpoint replay and atomic reports.

Uses a five-particle harmonic Reference Context and a cheap analytic energy,
plus synthetic Gaussian records. No molecular trajectories, GPU or model.
"""

import contextlib
import io
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import openmm

import eta_hmc as eh
from eta_records import SCHEMA, atomic_save, read_record, run_recorded, validate_record
from atomic_report import report


class AnalyticEnergy:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        atom = np.array([400 * x[0, 0] ** 2, -10 * x[1, 0]])
        self.last_components = {"dU_atom": atom, "C_ML": atom.copy(), "C_MM": np.zeros((2, 2)),
                                "C_ML_learned": 0.75 * atom, "C_ML_coul": 0.25 * atom,
                                "cluster_system_indices": np.arange(5), "omega_water_resid": np.array([17])}
        return float(atom.sum())


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


def contract(eta=1.0, start=1):
    return {"system": "analytic_harmonic", "state": 0, "leg": "solvent", "T_K": 300.0,
            "eta": eta, "start_rep": start, "L": 4, "dt_fs": 10.0,
            "E_construction": "test-only harmonic plus analytic correction", "dtype": "float64",
            "backbone": "analytic", "head": "analytic", "n_particles": 5, "n_water": 1,
            "ligand_system_indices": [0, 1], "groups": [[0], [1]], "dummy_mask": [False, False],
            "box_nm": (np.eye(3) * 3).tolist(), "system_xml": "analytic-system", "softcore_alpha": 0.5,
            "system_definition": "five-particle harmonic k=50 m=12", "context_parameters": {},
            "energy_units": "kJ/mol", "xyz_units": "nm"}


def run_toy(out, kernel=eh.hmc):
    c, ig = toy_context()
    energy = AnalyticEnergy()
    x0 = np.zeros((5, 3))
    x0[0, 0], x0[1, 0] = 0.15, -0.1
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            meta = run_recorded(c, ig, energy, np.random.default_rng(19), x0, 1.0, 40, out,
                                contract(), kernel, checkpoint_every=7, coords_every=3)
        return meta, energy.calls
    finally:
        del c, ig


def synthetic_record(path, eta, start, seed):
    rng = np.random.default_rng(seed)
    n = 120
    x = rng.normal(0, np.sqrt(eh.KT), n) - 0.2 * eta
    atom = np.c_[x, 1 - 0.8 * x]
    m = {**contract(eta, start), "storage_schema": SCHEMA, "cycles": n, "completed_cycles": n,
         "complete": True, "seconds": 1.0, "checkpoint_every": 7, "coords_every": 3}
    pos = np.zeros((1, 5, 3))
    z = {"meta": json.dumps(m), "dU_atom": atom, "C_ML": atom.copy(), "C_MM": np.zeros((n, 2, 2)),
         "dU": atom.sum(axis=1), "V": np.zeros(n), "acc": np.ones(n, bool),
         "dH": np.zeros(n), "dE": np.zeros(n), "cycle": np.arange(1, n + 1),
         "cluster_system_indices": np.tile(np.arange(5), (n, 1)), "omega_water_resid": np.full((n, 1), 17),
         "lig_xyz": np.zeros((n, 2, 3)), "xyz": pos, "xyz_cycles": np.array([n]),
         "last_positions": pos[0], "box": np.eye(3) * 3,
         "checkpoint": np.array([1], np.uint8), "rng_state": json.dumps(rng.bit_generator.state),
         "C_ML_learned": atom.copy(), "C_ML_coul": np.zeros_like(atom)}
    z.update({"prop_" + k: z[k] for k in ("C_ML", "C_ML_learned", "C_ML_coul", "C_MM", "dU_atom",
                                          "cluster_system_indices", "omega_water_resid", "V", "lig_xyz", "xyz")})
    atomic_save(path, z)
    return atom


class TestRecordedChain(unittest.TestCase):
    def test_model_energy_atom_units_and_dummy(self):
        class Leg:
            top = [None, None, None, None, np.array([0, 0, 17, 17, 17])]

            def carve(self, xyz, nwater):
                return xyz, np.ones(5), np.array([0, 0, 1, 1, 1]), np.arange(5)

            def c_mm(self, pos, indices, dummy):
                return np.array([[0.0, 0.0], [2.0, 3.0]])

        me = object.__new__(eh.ModelEnergy)
        me.leg, me.dm, me.model, me.dtype = Leg(), np.array([True, False]), None, None
        me.calls, me.seconds = 0, 0.0
        split = (np.array([[np.nan, 0.2]]), np.array([[np.nan, 0.15]]), np.array([[np.nan, 0.05]]))
        with patch.object(eh, "couplings_dt", return_value=split):
            value = me(np.zeros((5, 3)))
        self.assertAlmostEqual(value, 0.2 * eh.EV - 5.0)
        np.testing.assert_allclose(me.last_components["dU_atom"], [0, value], atol=1e-12, rtol=0)
        np.testing.assert_allclose(me.last_components["C_ML_learned"], [0, 0.15 * eh.EV], atol=1e-12, rtol=0)
        np.testing.assert_allclose(me.last_components["C_ML_coul"], [0, 0.05 * eh.EV], atol=1e-12, rtol=0)
        self.assertEqual(me.calls, 1)

    def test_rejected_states_and_thinning(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "chain.npz"
            meta, calls = run_toy(p)
            m, z = read_record(p)
            self.assertTrue(meta["complete"])
            self.assertEqual(calls, 41)  # initial + one query per proposal; recording adds none
            self.assertTrue(z["acc"].any())
            self.assertTrue((~z["acc"]).any())
            np.testing.assert_array_equal(z["xyz_cycles"], np.r_[np.arange(3, 40, 3), 40])
            expected = np.c_[400 * z["lig_xyz"][:, 0, 0] ** 2, -10 * z["lig_xyz"][:, 1, 0]]
            np.testing.assert_allclose(z["dU_atom"], expected, atol=1e-12, rtol=0)
            # Every proposal is kept for diagnosis: its energy is the previous state plus dE.
            np.testing.assert_allclose(z["prop_dU_atom"][1:].sum(1), z["dU"][:-1] + z["dE"][1:], atol=1e-10, rtol=0)
            prop_expected = np.c_[400 * z["prop_lig_xyz"][:, 0, 0] ** 2, -10 * z["prop_lig_xyz"][:, 1, 0]]
            np.testing.assert_allclose(z["prop_dU_atom"], prop_expected, atol=1e-12, rtol=0)
            rej = ~z["acc"]
            self.assertFalse(np.array_equal(z["prop_lig_xyz"][rej], z["lig_xyz"][rej]))
            np.testing.assert_array_equal(z["prop_xyz"][:, :2], z["prop_lig_xyz"][z["xyz_cycles"] - 1])
            for i in np.flatnonzero(~z["acc"]):
                if i:
                    np.testing.assert_array_equal(z["dU_atom"][i], z["dU_atom"][i - 1])
                    np.testing.assert_array_equal(z["lig_xyz"][i], z["lig_xyz"][i - 1])
            # Finite but inconsistent atomic records must fail before being written.
            bad = {**z, "dU_atom": z["dU_atom"].copy()}
            bad["dU_atom"][0, 0] += 1
            with self.assertRaisesRegex(ValueError, "atomic difference"):
                validate_record(bad)

    def test_checkpoint_replays_uncommitted_chunk(self):
        with tempfile.TemporaryDirectory() as td:
            ref, interrupted = Path(td) / "reference.npz", Path(td) / "interrupted.npz"
            run_toy(ref)
            chunks = 0

            def fail_inside_second(*args, **kw):
                nonlocal chunks
                chunks += 1
                if chunks == 2:
                    args = list(args)
                    args[4] = 3  # consume proposals AND both RNG streams, then lose this chunk
                    eh.hmc(*args, **kw)
                    raise RuntimeError("simulated process interruption")
                return eh.hmc(*args, **kw)

            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run_toy(interrupted, fail_inside_second)
            self.assertFalse(interrupted.exists())
            progress = interrupted.with_name("interrupted.progress.npz")
            self.assertEqual(read_record(progress)[0]["completed_cycles"], 7)
            run_toy(interrupted)
            self.assertFalse(progress.exists())
            _, a = read_record(ref)
            _, b = read_record(interrupted)
            for k in ("acc", "dH", "dE", "dU", "V", "dU_atom", "C_ML", "lig_xyz", "xyz", "xyz_cycles",
                      "prop_V", "prop_dU_atom", "prop_C_ML_learned", "prop_lig_xyz", "prop_xyz"):
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)
            _, calls = run_toy(interrupted)
            self.assertEqual(calls, 0)  # completed record is reused without any energy query
            # Crash after the complete progress commit but before final rename.
            interrupted.rename(progress)
            _, calls = run_toy(interrupted)
            self.assertEqual(calls, 0)
            self.assertTrue(interrupted.exists())
            self.assertFalse(progress.exists())

    def test_contract_and_history_preservation(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "legacy.npz"
            np.savez(p, meta=json.dumps({"legacy": True}), dU=[1.0])
            before = p.read_bytes()
            with self.assertRaisesRegex(ValueError, "legacy chain"):
                run_toy(p)
            self.assertEqual(p.read_bytes(), before)
            p = Path(td) / "new.npz"
            run_toy(p)
            _, data = read_record(p)
            m = json.loads(str(data["meta"]))
            m["head"] = "another-target"
            data["meta"] = json.dumps(m)
            atomic_save(p, data)
            before = p.read_bytes()
            with self.assertRaisesRegex(ValueError, "contract differs"):
                run_toy(p)
            self.assertEqual(p.read_bytes(), before)


class TestAtomicReport(unittest.TestCase):
    def test_means_residual_covariance_and_missing_grid(self):
        with tempfile.TemporaryDirectory() as td:
            files, means = [], {}
            for e in (0.0, 0.5, 1.0):
                arrays = []
                for s in (1, 2):
                    p = Path(td) / f"e{e}_s{s}.npz"
                    arrays.append(synthetic_record(p, e, s, 93 + int(e * 100) + s))
                    files.append(p)
                means[e] = np.concatenate(arrays).mean(axis=0)
            r = report(files, [0, 0.5, 1], [1, 2], discard=0, bootstrap=12, block_length=8, seed=71)
            expected = 0.25 * means[0] + 0.5 * means[0.5] + 0.25 * means[1]
            np.testing.assert_allclose(r["A_atom_TI"], expected, atol=1e-12, rtol=0)
            self.assertAlmostEqual(r["r_eta_BAR_minus_atomic_TI"], r["a_BAR"] - sum(expected), places=12)
            self.assertGreater(abs(r["r_eta_BAR_minus_atomic_TI"]), 1e-7)
            q = np.array(r["block_bootstrap"]["A_atom_quantiles_2p5_50_97p5"])
            np.testing.assert_allclose(q[:, 1], 1 - 0.8 * q[::-1, 0], atol=1e-12, rtol=0)
            missing = report(files[:-1], [0, 0.5, 1], [1, 2], discard=0)
            self.assertFalse(missing["complete"])
            self.assertNotIn("A_atom_TI", missing)
            meta, z = read_record(files[0])
            meta["head"] = "other-head"
            z["meta"] = json.dumps(meta)
            atomic_save(files[0], z)
            with self.assertRaisesRegex(ValueError, "incompatible targets"):
                report(files, [0, 0.5, 1], [1, 2], discard=0)

    def test_scheduler_logs_preserve_history(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "logs").mkdir()
            old = d / "logs/step3_failed.txt"
            old.write_text("historical failure record\n")
            fake = d / "fake_python"
            fake.write_text("#!/usr/bin/python3\nimport os,sys\n"
                            "a=sys.argv\n"
                            "if 'analyze' in a: print('fake analysis'); sys.exit(0)\n"
                            "sys.exit(1 if os.environ.get('FAKE_FAIL') == '1' and "
                            "a[a.index('--eta')+1] == '0.3' else 0)\n")
            fake.chmod(0o755)
            source = Path(__file__).with_name("run_step3.sh").read_text()
            source = source.replace("cd /home/ruigengji/E3D/experiments/2026-10-25_eta_hmc", "cd " + shlex.quote(td))
            source = source.replace("P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python", "P=" + shlex.quote(str(fake)))
            script = d / "run.sh"
            script.write_text(source)
            env = {**os.environ, "TAG": "_failure_test", "NW": "2", "FAKE_FAIL": "1"}
            failed = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1, failed.stderr)
            self.assertEqual(len((d / "logs/step3_failure_test_failed.txt").read_text().splitlines()), 4)
            self.assertIn("INCOMPLETE", (d / "logs/step3_failure_test_analyze.log").read_text())
            env.update(TAG="_success_test", FAKE_FAIL="0")
            passed = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn("ALL DONE", (d / "logs/step3_success_test_analyze.log").read_text())
            self.assertEqual(old.read_text(), "historical failure record\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
