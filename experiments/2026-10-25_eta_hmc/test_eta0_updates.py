"""CPU tests of the eta=0 multi-update kernel (ETA0_MM_UPDATES_DECISION_2026-11-01.md "实现后的必要检查").
No molecule, GPU or model: five-particle harmonic Reference Contexts (test_atomic_records.toy_context).

1. Substeps: m attempts, each rejection restores the last accepted state (accept-then-reject and all-reject cases);
   no model query inside; one query at the end whose cache belongs to the final state; m = 1 is the original hmc.
2. Target: m substeps keep pi_0 ~ exp(-V/kT) (harmonic, <x^2> = kT/k per coordinate); flipping the sign of the
   Metropolis test (negative control) must fail the same comparison.
3. Recorded ladder with m = 5: model calls K (n + 1), substep records only on the walker that held eta=0, replay of an
   interrupted chunk identical to an uninterrupted run, completed-ladder reuse, m changes refused as a contract change,
   tampered substep records rejected.
"""

import contextlib
import io
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

import openmm

import eta_hmc as eh
from eta_exchange import eta0_mm_kernel, read_ladder, run_ladder, validate_ladder
from test_atomic_records import AnalyticEnergy, contract, toy_context


def harmonic_context(dt_fs, L=4):
    """test_atomic_records.toy_context (five particles, k=50, m=12) with a chosen proposal step."""
    system = openmm.System()
    f = openmm.CustomExternalForce("0.5*50*(x*x+y*y+z*z)")
    for i in range(5):
        system.addParticle(12.0)
        f.addParticle(i, [])
    system.addForce(f)
    ig = eh.proposal_integrator(L, dt_fs)
    ig.setRandomNumberSeed(42)
    return openmm.Context(system, ig, openmm.Platform.getPlatformByName("Reference")), ig


class SeqRng:
    """random() returns the next value; 1e-300 accepts any finite work, 1 - 1e-16 rejects any dH > ~0."""

    def __init__(self, u):
        self.u = list(u)

    def random(self):
        return self.u.pop(0)


def x_start():
    x0 = np.zeros((5, 3))
    x0[0, 0], x0[1, 0] = 0.15, -0.1
    return x0


def capture_of(energy):
    return lambda: {k: np.asarray(v).copy() for k, v in energy.last_components.items()}


class TestSubsteps(unittest.TestCase):
    def test_rollback_single_query_and_cache(self):
        c, ig = harmonic_context(600.0)                         # dH ~ kT: both outcomes occur
        energy = AnalyticEnergy()
        kern = eta0_mm_kernel(eh.hmc, 25)
        init = {"x": x_start(), "e": energy(x_start()), "details": capture_of(energy)()}
        c.setPositions(x_start())
        u = np.random.default_rng(3).random(25)
        reads = []
        true_positions = eh.positions

        def spy(ctx):
            reads.append(true_positions(ctx))
            return reads[-1]

        with unittest.mock.patch.object(eh, "positions", spy):
            o = kern(c, ig, None, 0.0, 1, energy, SeqRng(u), record=lambda x: np.array(x, copy=True),
                     capture=capture_of(energy), initial=init)
        acc, dH = o["mm_sub_acc"], o["mm_sub_dH"]
        self.assertEqual(energy.calls, 2)                       # initial cache + one final query, none per substep
        self.assertEqual(len(acc), 25)
        np.testing.assert_array_equal(acc, np.log(u) < -dH / eh.KT)                 # each substep: its own Metropolis
        self.assertTrue((acc[:-1] & ~acc[1:]).any(), "need an accept followed by a reject")
        self.assertEqual(len(reads), 1 + acc.sum())             # position check + one readback per accepted substep
        self.assertTrue(np.isnan(o["dH"][0]) and o["acc"][0])
        np.testing.assert_array_equal(o["final_state"]["x"], reads[-1])            # last accepted state
        np.testing.assert_array_equal(true_positions(c), o["final_state"]["x"])    # later rejections rolled back
        np.testing.assert_array_equal(o["rec"][0], o["final_state"]["x"])
        self.assertEqual(o["e"][0], AnalyticEnergy()(o["final_state"]["x"]))      # cache belongs to the final state
        xf = o["final_state"]["x"]
        np.testing.assert_array_equal(o["details"]["dU_atom"][0], [400 * xf[0, 0] ** 2, -10 * xf[1, 0]])

    def test_all_rejected_keeps_state(self):
        c, ig = harmonic_context(1500.0)                        # unstable leapfrog: dH >> kT, every substep rejected
        energy = AnalyticEnergy()
        init = {"x": x_start(), "e": energy(x_start()), "details": capture_of(energy)()}
        c.setPositions(x_start())
        o = eta0_mm_kernel(eh.hmc, 25)(c, ig, None, 0.0, 1, energy, SeqRng([0.5] * 25),
                                       record=lambda x: np.array(x, copy=True), capture=capture_of(energy), initial=init)
        self.assertEqual(len(o["mm_sub_acc"]), 25)
        self.assertTrue((o["mm_sub_dH"] > 10 * eh.KT).all())
        self.assertFalse(o["mm_sub_acc"].any() or o["acc"][0])
        np.testing.assert_array_equal(o["final_state"]["x"], x_start())
        self.assertEqual(o["dE"][0], 0.0)

    def test_other_labels_and_m1_use_base(self):
        calls = []

        def base(*a, **k):
            calls.append(a[3])
            return "base"
        for m, eta in ((5, 0.5), (1, 0.0)):
            self.assertEqual(eta0_mm_kernel(base, m)(None, None, None, eta, 1, None, None), "base")
        self.assertEqual(calls, [0.5, 0.0])


def x2_mean(sign, m=5, n=3000, seed=11):
    c, ig = harmonic_context(600.0)
    ig.setRandomNumberSeed(7)
    rng = np.random.default_rng(seed)
    c.setPositions(np.zeros((5, 3)))
    x = eh.positions(c)
    kt = eh.KT
    samples = []
    with unittest.mock.patch.object(eh, "KT", sign * kt):     # sign -1: Metropolis test on +dH (wrong target)
        for _ in range(n):
            x, *_ = eh.mm_updates(c, ig, x, m, rng)
            samples.append((x ** 2).mean())
    s = np.array(samples[n // 10:])
    batch = s[: len(s) // 20 * 20].reshape(20, -1).mean(1)
    return batch.mean(), batch.std(ddof=1) / np.sqrt(20), kt / 50.0


class TestTarget(unittest.TestCase):
    def test_harmonic_pi0_and_negative_control(self):
        est, se, exact = x2_mean(+1)
        self.assertLess(abs(est - exact) / se, 5.0, f"<x^2> {est:.5f} +- {se:.5f} vs kT/k {exact:.5f}")
        est_b, se_b, _ = x2_mean(-1)
        self.assertGreater(abs(est_b - exact) / se_b, 10.0, f"flipped test should be wrong: {est_b:.5f} +- {se_b:.5f}")


def run_toy(out, m=5, kernel_wrap=None, sweeps=20):
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
    kernel = eta0_mm_kernel(eh.hmc, m)
    if kernel_wrap is not None:
        kernel = kernel_wrap(kernel)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            meta = run_ladder(list(ctxs), list(igs), energy, [np.random.default_rng(19 + w) for w in range(K)],
                              np.random.default_rng(99), x0s, [0.0, 0.5, 1.0], sweeps, out, contract(), kernel, eh.KT,
                              checkpoint_every=7, coords_every=3, eta0_mm_updates=m)
        return meta, energy.calls
    finally:
        del ctxs, igs


class TestLadder(unittest.TestCase):
    KEYS = ("acc", "dH", "dE", "V", "dU", "prop_V", "dU_atom", "perm", "ex_acc", "ex_logR", "label_dU", "lig_xyz",
            "xyz", "walker_eta_local", "mm_sub_dH", "mm_sub_acc")

    def test_calls_records_replay_reuse_contract(self):
        with tempfile.TemporaryDirectory() as td:
            ref, cut = Path(td) / "ref.npz", Path(td) / "cut.npz"
            meta, calls = run_toy(ref)
            self.assertEqual(calls, 3 * 21)
            self.assertEqual(meta["eta0_mm_updates"], 5)
            m, z = read_ladder(ref)
            eta0 = z["walker_eta_local"] == 0.0
            self.assertEqual(eta0.sum(), 20)                       # exactly one walker per sweep
            self.assertTrue(np.isnan(z["dH"][eta0]).all() and np.isfinite(z["dH"][~eta0]).all())
            self.assertEqual(z["mm_sub_acc"].shape, (20, 3, 5))
            self.assertTrue(z["mm_sub_acc"][eta0].any())          # rejection paths: TestSubsteps
            self.assertTrue(z["perm"][1:].tolist() != z["perm"][:-1].tolist())   # eta=0 changed hands
            n = [0]

            def fail_in_second_chunk(k):
                def wrapped(*a, **kw):
                    n[0] += 1
                    o = k(*a, **kw)
                    if n[0] == 7 * 3 + 5:
                        raise RuntimeError("simulated interruption")
                    return o
                return wrapped

            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run_toy(cut, kernel_wrap=fail_in_second_chunk)
            run_toy(cut)
            _, b = read_ladder(cut)
            for k in self.KEYS:
                np.testing.assert_array_equal(z[k], b[k], err_msg=k)
            _, calls = run_toy(cut)
            self.assertEqual(calls, 0)
            with self.assertRaisesRegex(ValueError, "contract differs"):
                run_toy(cut, m=25)
            bad = dict(b)
            bad["mm_sub_acc"] = b["mm_sub_acc"].copy()
            w = np.argwhere(b["walker_eta_local"] == 0.0)[0]
            bad["mm_sub_acc"][w[0], w[1]] = False                  # recorded state change without any accepted substep
            bad["mm_sub_acc"][w[0], w[1], 0] = not b["acc"][w[0], w[1]]
            with self.assertRaisesRegex(ValueError, "state change disagrees"):
                validate_ladder(bad)

    def test_m1_ladder_contract_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            meta, calls = run_toy(Path(td) / "m1.npz", m=1)
            self.assertEqual(calls, 3 * 21)
            self.assertNotIn("eta0_mm_updates", meta)
            _, z = read_ladder(Path(td) / "m1.npz")
            self.assertNotIn("mm_sub_dH", z)
            self.assertTrue(np.isfinite(z["dH"]).all())


if __name__ == "__main__":
    unittest.main()
