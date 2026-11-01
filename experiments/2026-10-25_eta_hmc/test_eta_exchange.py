"""CPU tests of eta-label exchange (eta_exchange.py). No molecule, GPU or model.

1. exchange_step: forced accept / reject, labels only, non-overlapping pairs.
2. Finite-state joint target: per-label histograms against exact Boltzmann distributions, with a flipped-sign
   negative control that must fail the same comparison.
3. Recorded ladder on five-particle Reference Contexts: model-call count, interrupted-chunk replay identical to an
   uninterrupted run, completed-ladder reuse, matched no-exchange control, tampered records rejected.
"""

import contextlib
import io
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

import eta_hmc as eh
from eta_exchange import atomic_save, exchange_step, read_ladder, run_ladder, validate_ladder
from test_atomic_records import AnalyticEnergy, contract, toy_context


class FixedRng:
    def __init__(self, u):
        self.u = u

    def random(self):
        return self.u


class TestExchangeStep(unittest.TestCase):
    def test_forced_accept_reject_and_schedule(self):
        etas, e = [0.0, 0.5, 1.0, 1.0], np.array([-10.0, 0.0, -5.0, 7.0])
        # phase 0: pairs (0,1), (2,3); phase 1: pair (1,2) only
        p, att, lr, lu, ac = exchange_step([0, 1, 2, 3], e, etas, 0, FixedRng(1e-300), 1.0)
        self.assertEqual(att.tolist(), [True, False, True])
        self.assertAlmostEqual(lr[0], (0.0 - 0.5) * (-10.0 - 0.0))
        self.assertEqual(lr[2], 0.0)                        # equal etas: always accepted
        self.assertEqual(p.tolist(), [1, 0, 3, 2])
        p, att, lr, lu, ac = exchange_step([0, 1, 2, 3], e, etas, 1, FixedRng(1 - 1e-16), 1.0)
        self.assertEqual(att.tolist(), [False, True, False])
        self.assertAlmostEqual(lr[1], (0.5 - 1.0) * (0.0 - (-5.0)))   # walker 1 (dU 0) at 0.5, walker 2 (dU -5) at 1.0
        self.assertFalse(ac[1])
        self.assertEqual(p.tolist(), [0, 1, 2, 3])          # rejection: nothing moves
        p, att, *_ = exchange_step([2, 0, 1, 3], e, etas, 0, FixedRng(0.5), 1.0, enabled=False)
        self.assertEqual(p.tolist(), [2, 0, 1, 3])
        self.assertFalse(att.any())


class TestReadbackRounding(unittest.TestCase):
    """CUDA mixed precision may round a restore/readback; the cached chain state must not absorb it."""

    def test_no_accumulation_and_wrong_state_refused(self):
        c, ig = toy_context()
        energy = AnalyticEnergy()
        rng, noise = np.random.default_rng(3), np.random.default_rng(4)
        true_positions = eh.positions

        def noisy(ctx):
            return true_positions(ctx) + noise.uniform(-1e-9, 1e-9, (5, 3))

        x0 = np.zeros((5, 3))
        x0[0, 0], x0[1, 0] = 0.15, -0.1
        state, xs, acc = None, [], []
        with unittest.mock.patch.object(eh, "positions", noisy):
            for _ in range(150):                            # one cycle per call, as in the ladder; eta 20 -> many rejections
                o = eh.hmc(c, ig, x0, 20.0, 1, energy, rng, record=lambda x: np.array(x, copy=True),
                           capture=lambda: {k: np.asarray(v).copy() for k, v in energy.last_components.items()},
                           initial=state)
                state = o["final_state"]
                xs.append(o["rec"][0])
                acc.append(bool(o["acc"][0]))
        acc = np.array(acc)
        self.assertGreater((~acc).sum(), 50)
        for i in np.flatnonzero(~acc[1:]) + 1:              # a rejected cycle repeats the previous state bit for bit
            np.testing.assert_array_equal(xs[i], xs[i - 1])
        e_check = AnalyticEnergy()(state["x"])
        self.assertEqual(e_check, state["e"])               # cache and its coordinates still agree exactly
        bad = {**state, "x": state["x"] + 1e-2}
        with self.assertRaisesRegex(ValueError, "checkpoint positions differ"):
            eh.hmc(c, ig, x0, 20.0, 1, energy, rng, initial=bad)


def finite_state_run(sign, sweeps=200000, seed=5):
    V = np.array([0.0, 1.0, 3.0, 3.0, 1.0, 0.0])
    dU = np.array([2.0, 1.0, 0.0, -1.0, -2.0, -4.0])
    etas, rt, K, M = [0.0, 0.5, 1.0], 1.0, 3, 6
    rng = np.random.default_rng(seed)
    x = np.array([0, 2, 5])                                 # walker states
    perm = np.arange(K)
    counts = np.zeros((20, K, M))
    for s in range(sweeps):
        inv = np.empty(K, int)
        inv[perm] = np.arange(K)
        for w in range(K):                                  # slow local move: +-1 neighbour Metropolis
            eta = etas[inv[w]]
            y = x[w] + (1 if rng.random() < 0.5 else -1)
            if 0 <= y < M and np.log(rng.random()) < -((V[y] + eta * dU[y]) - (V[x[w]] + eta * dU[x[w]])) / rt:
                x[w] = y
        perm = exchange_step(perm, sign * dU[x], etas, s % 2, rng, rt)[0]
        np.add.at(counts[s * 20 // sweeps], (np.arange(K), x[perm]), 1)
    exact = np.exp(-(V[None] + np.array(etas)[:, None] * dU[None]) / rt)
    exact /= exact.sum(1, keepdims=True)
    batch = counts / counts.sum(-1, keepdims=True)          # 20 batch means per label
    est, se = batch.mean(0), batch.std(0, ddof=1) / np.sqrt(len(batch))
    return np.abs(est - exact) / np.maximum(se, 1e-12), est, exact


class TestJointTarget(unittest.TestCase):
    def test_label_marginals_and_negative_control(self):
        z, est, exact = finite_state_run(+1)
        self.assertLess(z.max(), 5.0, f"correct exchange deviates: z={z.round(1)}")
        z_bad, *_ = finite_state_run(-1)
        self.assertGreater(z_bad.max(), 10.0, "flipped-sign exchange should be detectably wrong")


def run_toy_ladder(out, kernel=eh.hmc, exchange=True, sweeps=20, extend_from=None, device=None):
    K = 3
    ctxs, igs = zip(*[toy_context() for _ in range(K)])
    for w, ig in enumerate(igs):
        ig.setRandomNumberSeed(42 + w)
    energy = AnalyticEnergy()
    x0s = []
    for w in range(K):
        x0 = np.zeros((5, 3))
        x0[0, 0], x0[1, 0] = 0.05 * (w + 1), -0.1 * w
        x0s.append(x0)
    rngs = [np.random.default_rng(19 + w) for w in range(K)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            meta = run_ladder(list(ctxs), list(igs), energy, rngs, np.random.default_rng(99), x0s, [0.0, 0.5, 1.0],
                              sweeps, out, contract(), kernel, eh.KT, exchange=exchange, checkpoint_every=7, coords_every=3,
                              extend_from=extend_from, device=device)
        return meta, energy.calls
    finally:
        del ctxs, igs


class TestRecordedLadder(unittest.TestCase):
    KEYS = ("acc", "dH", "dE", "V", "dU", "prop_V", "dU_atom", "C_ML_learned", "perm", "ex_attempt", "ex_acc",
            "ex_logR", "label_dU", "lig_xyz", "prop_lig_xyz", "xyz", "prop_xyz", "xyz_sweeps", "walker_eta_local")

    def test_calls_records_replay_and_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            ref, cut = Path(td) / "ref.npz", Path(td) / "cut.npz"
            meta, calls = run_toy_ladder(ref)
            self.assertEqual(calls, 3 * 21)                 # one per walker at start + one per local proposal
            self.assertEqual(meta["model_calls_ladder"], 63)
            m, z = read_ladder(ref)
            self.assertTrue(z["ex_acc"].any() and (z["ex_attempt"] & ~z["ex_acc"]).any())
            self.assertTrue(z["acc"].any() and (~z["acc"]).any())
            # labels moved, coordinates followed walkers: walker w's ligand never jumps on an exchange-only sweep
            moved = (z["perm"][1:] != z["perm"][:-1]).any(1)
            self.assertTrue(moved.any())
            still = ~z["acc"][1:]
            self.assertTrue(np.array_equal(z["lig_xyz"][1:][still], z["lig_xyz"][:-1][still]))
            calls_n = [0]

            def fail_in_second_chunk(*args, **kw):
                calls_n[0] += 1
                o = eh.hmc(*args, **kw)
                if calls_n[0] == 7 * 3 + 5:                 # consumed proposals and RNG inside chunk 2
                    raise RuntimeError("simulated interruption")
                return o

            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run_toy_ladder(cut, fail_in_second_chunk)
            progress = cut.with_name("cut.progress.npz")
            self.assertEqual(read_ladder(progress)[0]["completed_sweeps"], 7)
            run_toy_ladder(cut)
            self.assertFalse(progress.exists())
            _, a = read_ladder(ref)
            _, b = read_ladder(cut)
            for k in self.KEYS:
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)
            _, calls = run_toy_ladder(cut)
            self.assertEqual(calls, 0)                      # completed ladder reused without model calls

    def test_extend_completed_ladder(self):
        with tempfile.TemporaryDirectory() as td:
            full, short, ext = Path(td) / "full.npz", Path(td) / "short.npz", Path(td) / "ext.npz"
            run_toy_ladder(full, sweeps=20)
            run_toy_ladder(short, sweeps=14)
            before = short.read_bytes()
            _, calls = run_toy_ladder(ext, sweeps=20, extend_from=short)
            self.assertEqual(calls, 3 + 3 * 6)               # one restart check per walker + 6 sweeps
            self.assertEqual(short.read_bytes(), before)    # source ladder untouched
            m, b = read_ladder(ext)
            self.assertEqual(m["extended_from"], {"file": "short.npz", "sweeps": 14})
            _, a = read_ladder(full)
            for k in self.KEYS:
                if k in ("xyz", "prop_xyz", "xyz_sweeps"):
                    continue
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)   # identical to an uninterrupted 20-sweep run
            # snapshots: the source's final sweep (14) is kept in addition to the regular ones
            np.testing.assert_array_equal(b["xyz_sweeps"], np.union1d(a["xyz_sweeps"], [14]))
            common = np.isin(b["xyz_sweeps"], a["xyz_sweeps"])
            np.testing.assert_array_equal(b["xyz"][common], a["xyz"])
            np.testing.assert_array_equal(b["prop_xyz"][common], a["prop_xyz"])
            with self.assertRaisesRegex(ValueError, "fewer sweeps"):
                run_toy_ladder(Path(td) / "bad.npz", sweeps=14, extend_from=short)

    def test_device_change_continues_from_coordinates(self):
        with tempfile.TemporaryDirectory() as td:
            full, short = Path(td) / "full.npz", Path(td) / "short.npz"
            run_toy_ladder(full, sweeps=20, device="gpuA")
            run_toy_ladder(short, sweeps=14, device="gpuA")
            same = Path(td) / "same.npz"
            run_toy_ladder(same, sweeps=20, extend_from=short, device="gpuA")
            _, a = read_ladder(full)
            m, b = read_ladder(same)
            self.assertEqual(m["continuations"], [])
            np.testing.assert_array_equal(a["dU"], b["dU"])            # same device: exact checkpoint replay
            other = Path(td) / "other.npz"
            _, calls = run_toy_ladder(other, sweeps=20, extend_from=short, device="gpuB")
            self.assertEqual(calls, 3 + 3 * 6)
            m, c = read_ladder(other)                                  # validates the whole record
            self.assertEqual(m["checkpoint_device"], "gpuB")
            self.assertEqual(m["continuations"], [{"after_sweep": 14, "mode": "positions",
                                                   "checkpoint_device": "gpuA", "device": "gpuB"}])
            for k in ("dU", "perm", "lig_xyz", "acc"):
                np.testing.assert_array_equal(c[k][:14], a[k][:14], err_msg=k)
            # the first proposal after the coordinate restart starts from the cached state, not a corrupted one
            np.testing.assert_allclose(c["lig_xyz"][14][~c["acc"][14]], c["lig_xyz"][13][~c["acc"][14]], atol=1e-12, rtol=0)
            self.assertTrue(np.all(np.abs(c["dH"][14:]) < 50))

    def test_no_exchange_control_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ctl.npz"
            meta, calls = run_toy_ladder(p, exchange=False)
            self.assertEqual(calls, 63)
            _, z = read_ladder(p)
            self.assertFalse(z["ex_attempt"].any())
            self.assertTrue((z["perm"] == np.arange(3)).all())
            p2 = Path(td) / "ex.npz"
            run_toy_ladder(p2)
            _, z = read_ladder(p2)
            s = int(np.flatnonzero(z["ex_acc"].any(1))[0])
            bad = {**z, "ex_acc": z["ex_acc"].copy()}
            bad["ex_acc"][s] = False                         # claims rejection but permutation swapped
            with self.assertRaisesRegex(ValueError, "exchange decision|permutation"):
                validate_ladder(bad)
            bad = {**z, "label_dU": z["label_dU"][:, ::-1].copy()}
            with self.assertRaisesRegex(ValueError, "label view"):
                validate_ladder(bad)
            m = json.loads(str(z["meta"]))
            m["etas"] = [0.0, 0.25, 1.0]
            bad = {**z, "meta": json.dumps(m)}
            with self.assertRaisesRegex(ValueError, "log R"):
                validate_ladder(bad)
            m = json.loads(str(z["meta"]))
            m["head"] = "another-target"
            atomic_save(p2, {**z, "meta": json.dumps(m)})
            before = p2.read_bytes()
            with self.assertRaisesRegex(ValueError, "contract differs"):
                run_toy_ladder(p2)
            self.assertEqual(p2.read_bytes(), before)       # a different target is refused, file untouched


if __name__ == "__main__":
    unittest.main(verbosity=2)
