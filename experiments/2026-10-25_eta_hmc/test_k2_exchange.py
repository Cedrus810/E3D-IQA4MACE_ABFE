"""CPU checks for the K2 exchange ladder (k2_exchange.py) and its report. Toy Reference Contexts, analytic K2 target.
python -m unittest test_k2_exchange -v"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import eta_exchange
import eta_hmc as eh
import k2_exchange as kx
from k2_report import report_ladders
from test_k2_records import AnalyticK2, contract, toy_context

ETAS = [0.0, 0.5, 1.0]


def run_toy_ladder(out, kernel=eh.hmc, exchange=True, sweeps=20, device=None, label="a", seed=19, m_up=1):
    K = len(ETAS)
    ctxs, igs = zip(*[toy_context() for _ in range(K)])
    for w, ig in enumerate(igs):
        ig.setRandomNumberSeed(42 + w + seed)
    energy = AnalyticK2()
    x0s = []
    for w in range(K):
        x0 = np.zeros((5, 3))
        x0[0, 0], x0[1, 0], x0[0, 2] = 0.05 * (w + 1), -0.1 * w, 0.02 * w
        x0s.append(x0)
    rngs = [np.random.default_rng(seed + w) for w in range(K)]
    c = {k: v for k, v in contract().items() if k not in ("eta",)}
    c["start_label"] = label
    if m_up > 1:
        kernel = kx.eta0_mm_kernel(kernel, m_up)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            meta = kx.run_ladder(list(ctxs), list(igs), energy, rngs, np.random.default_rng(99 + seed), x0s, ETAS,
                                 sweeps, out, c, kernel, eh.KT, exchange=exchange, checkpoint_every=7, coords_every=3,
                                 device=device, eta0_mm_updates=m_up)
        return meta, energy.calls
    finally:
        del ctxs, igs


class TestK2Ladder(unittest.TestCase):
    KEYS = ("acc", "dH", "dE", "V", "dU", "d_atom", "q", "C_bb", "perm", "ex_acc", "ex_logR",
            "label_dU", "label_d_atom", "label_q", "lig_xyz", "xyz")

    def test_accounting_calls_replay_and_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            ref, cut = Path(td) / "ref.npz", Path(td) / "cut.npz"
            meta, calls = run_toy_ladder(ref)
            self.assertEqual(calls, 3 * 21)
            m, z = kx.read_ladder(ref)
            self.assertTrue(z["ex_acc"].any() and (z["ex_attempt"] & ~z["ex_acc"]).any())
            np.testing.assert_allclose(z["d_atom"].sum(-1) + z["q"], z["dU"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(z["label_d_atom"].sum(-1) + z["label_q"], z["label_dU"], atol=1e-12, rtol=0)
            n = 0

            def fail(*args, **kw):
                nonlocal n
                n += 1
                o = eh.hmc(*args, **kw)
                if n == 7 * 3 + 5:
                    raise RuntimeError("simulated interruption")
                return o

            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run_toy_ladder(cut, fail)
            self.assertEqual(kx.read_ladder(cut.with_name("cut.progress.npz"))[0]["completed_sweeps"], 7)
            run_toy_ladder(cut)
            _, a = kx.read_ladder(ref)
            _, b = kx.read_ladder(cut)
            for k in self.KEYS:
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)
            self.assertEqual(run_toy_ladder(cut)[1], 0)

    def test_device_change_continues_from_coordinates(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dev.npz"
            run_toy_ladder(p, sweeps=20, device="gpuA")
            q = Path(td) / "devcut.npz"
            n = 0

            def fail(*args, **kw):
                nonlocal n
                n += 1
                o = eh.hmc(*args, **kw)
                if n == 7 * 3 + 2:
                    raise RuntimeError("stop")
                return o

            with self.assertRaises(RuntimeError):
                run_toy_ladder(q, fail, device="gpuA")
            run_toy_ladder(q, device="gpuB")
            m, z = kx.read_ladder(q)
            self.assertEqual(m["continuations"], [{"after_sweep": 7, "mode": "positions",
                                                   "checkpoint_device": "gpuA", "device": "gpuB"}])
            self.assertEqual(m["checkpoint_device"], "gpuB")

    def test_tamper_and_schema_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.npz"
            run_toy_ladder(p)
            _, z = kx.read_ladder(p)
            for key, msg in (("label_q", "label view"), ("q", "q disagrees"), ("C_bb", "q disagrees")):
                bad = {**z, key: z[key].copy()}
                bad[key][0, 0] += 1
                with self.assertRaisesRegex(ValueError, msg):
                    kx.validate_ladder(bad)
            s = int(np.flatnonzero(z["ex_acc"].any(1))[0])
            bad = {**z, "ex_acc": z["ex_acc"].copy()}
            bad["ex_acc"][s] = False
            with self.assertRaisesRegex(ValueError, "exchange decision|permutation"):
                kx.validate_ladder(bad)
            with self.assertRaises((ValueError, KeyError)):
                eta_exchange.validate_ladder(z)                     # legacy reader refuses K2 ladders
            m = json.loads(str(z["meta"]))
            m["storage_schema"], m["E_construction"] = eta_exchange.SCHEMA, "local_E v1"
            with self.assertRaisesRegex(ValueError, "not a k2_exchange"):
                kx.validate_ladder({**z, "meta": json.dumps(m)})
            m = json.loads(str(z["meta"]))
            m["realspace_backend"] = "analytic_l01_v1"
            kx.atomic_save(p, {**z, "meta": json.dumps(m)})
            before = p.read_bytes()
            with self.assertRaisesRegex(ValueError, "contract differs"):
                run_toy_ladder(p)
            self.assertEqual(p.read_bytes(), before)

    def test_eta0_mm_updates_kernel(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.npz"
            meta, calls = run_toy_ladder(p, m_up=3, sweeps=14)
            self.assertEqual(calls, 3 * 15)
            m, z = kx.read_ladder(p)
            self.assertEqual(m["eta0_mm_updates"], 3)
            e0 = z["walker_eta_local"] == 0.0
            self.assertTrue(np.isnan(z["dH"][e0]).all() and np.isfinite(z["dH"][~e0]).all())


class TestK2LadderReport(unittest.TestCase):
    def test_label_view_integration(self):
        with tempfile.TemporaryDirectory() as td:
            files = []
            for i, lab in enumerate(("a", "b")):
                p = Path(td) / f"{lab}.npz"
                run_toy_ladder(p, sweeps=40, label=lab, seed=5 + 10 * i)
                files.append(p)
            r = report_ladders(files, discard=0.25, bootstrap=6, block_length=4, seed=2)
            self.assertTrue(r["complete"])
            means = []
            for k in range(3):
                rows = []
                for p in files:
                    _, z = kx.read_ladder(p)
                    rows.append(np.c_[z["label_d_atom"][10:, k], z["label_q"][10:, k]])
                means.append(np.concatenate(rows).mean(0))
            expected = 0.25 * means[0] + 0.5 * means[1] + 0.25 * means[2]
            np.testing.assert_allclose(r["A_H_atom_TI"], expected[:2], atol=1e-12, rtol=0)
            self.assertAlmostEqual(r["R_TI"], expected[2], places=12)
            self.assertAlmostEqual(r["atom_plus_R_minus_scalar_TI"], 0.0, places=10)
            self.assertEqual(set(r["exchange_acceptance"]), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
