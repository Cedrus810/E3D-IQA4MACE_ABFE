"""CPU tests of the two-stage local geometric MC (local_geom_mc.py; LOCAL_GEOMETRIC_MC_REVIEW_2026-11-02.md §5 and the
correctness gate). No molecule, GPU or model.

1. Two-stage step on a finite ring (k = 1 and k = 4 MTM), driven by the real two_stage_step: histograms against exact
   exp(-(V + eta dU)). Negative controls from the same records must fail: stage 2 left out; stage 2 using the full
   dU difference exp(-dU_total) instead of exp(-eta d(dU)); MTM with the forward sum only; MTM accepting the selected
   candidate with exp(-dV); MTM reference set without x.
2. Configuration-dependent selection set (an Omega analogue: the 2 of 3 particles nearest a site), mixed with a global
   move: correct with the reverse-reachability rejection, detectably wrong without it; an inadmissible proposal
   costs no energy and no model query.
3. k = 1 is the single-candidate step; rejections keep the input state and its components; one model query at most.
4. Geometry: torsion and water moves invert exactly, keep water geometry, have Jacobian determinant 1; draws symmetric.
"""

import unittest

import numpy as np

from local_geom_mc import gram, logsumexp, torsion_move, two_stage_step, water_move

M = 12
VA = np.array([0.0, 1.8, 0.4, 2.5, 1.1, 0.2, 3.0, 1.4, 0.6, 2.2, 0.9, 1.6])
DA = np.array([1.5, -2.0, 0.8, -1.2, 2.4, -0.5, -2.2, 1.0, -1.6, 0.3, 2.0, -0.9])
ETA, RT = 1.0, 1.0


def ring_draw(rng):
    return (-2, -1, 1, 2)[int(rng.integers(4))]


def ring_apply(x, s):
    return (x + s) % M


def exact():
    p = np.exp(-(VA + ETA * DA) / RT)
    return p / p.sum()


def ring_chain(k, variant="correct", n=100000, seed=4):
    rng, rng2 = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    st = {"x": 0, "V": VA[0], "dU": DA[0], "details": {"id": 0}}
    counts = np.zeros((20, M))
    for s in range(n):
        new, r = two_stage_step(st, k, ring_draw, ring_apply, lambda x: VA[x], lambda y: (DA[y], {"id": y}),
                                ETA, RT, rng)
        if variant == "correct":
            st = new
        else:
            y, x = r["y"], st["x"]
            if variant == "no_stage2":
                acc = r["stage1"]
            else:
                if variant == "full_dU_stage2":
                    s1 = r["stage1"]
                    a2 = -((VA[y] - VA[x]) + ETA * (DA[y] - DA[x])) / RT
                else:
                    a1 = {"forward_only": lambda: logsumexp(r["lw_y"]),
                          "plain_dV": lambda: -(VA[y] - VA[x]) / RT,
                          "no_x_in_ref": lambda: logsumexp(r["lw_y"]) - logsumexp(r["lw_x"][:-1])}[variant]()
                    s1 = r["log_u1"] < a1
                    a2 = -ETA * (DA[y] - DA[x]) / RT
                u2 = r["log_u2"] if r["log_u2"] is not None else float(np.log(rng2.random()))
                acc = s1 and u2 < a2
            if acc:
                st = {"x": y, "V": VA[y], "dU": DA[y], "details": {"id": y}}
        counts[s * 20 // n, st["x"]] += 1
    batch = counts / counts.sum(1, keepdims=True)
    est, se = batch.mean(0), batch.std(0, ddof=1) / np.sqrt(20)
    return float((np.abs(est - exact()) / np.maximum(se, 1e-12)).max())


class TestTwoStage(unittest.TestCase):
    def test_single(self):
        self.assertLess(ring_chain(1), 5.0)
        for bad in ("no_stage2", "full_dU_stage2"):
            self.assertGreater(ring_chain(1, bad), 10.0, bad)

    def test_mtm(self):
        self.assertLess(ring_chain(4), 5.0)
        for bad in ("no_stage2", "forward_only", "plain_dV", "no_x_in_ref"):
            self.assertGreater(ring_chain(4, bad), 10.0, bad)

    def test_k1_rejection_and_cache(self):
        rng = np.random.default_rng(9)
        calls = [0]

        def dU(y):
            calls[0] += 1
            return DA[y], {"id": y}
        st = {"x": 3, "V": VA[3], "dU": DA[3], "details": {"id": 3}}
        seen = {"s1_reject": 0, "s2_reject": 0, "acc": 0}
        for _ in range(400):
            c0 = calls[0]
            new, r = two_stage_step(st, 1, ring_draw, ring_apply, lambda x: VA[x], dU, ETA, RT, rng)
            self.assertAlmostEqual(r["log_a1"], -(VA[r["y"]] - VA[st["x"]]) / RT, places=12)
            self.assertEqual(calls[0] - c0, r["n_model"])
            self.assertLessEqual(r["n_model"], 1)
            if not r["stage1"]:
                seen["s1_reject"] += 1
                self.assertIs(new, st)
                self.assertEqual(r["n_model"], 0)
            elif not r["acc"]:
                seen["s2_reject"] += 1
                self.assertIs(new, st)                       # the queried candidate's components are not taken
                self.assertEqual(new["details"]["id"], st["x"])
            else:
                seen["acc"] += 1
                self.assertEqual(new["details"]["id"], new["x"])
            st = new
        self.assertTrue(all(v > 0 for v in seen.values()), seen)


L = 8
PA = np.array([0.0, 0.9, 1.7, 0.4, 2.1, 0.6, 1.3, 0.2])
PB = np.array([-1.5, 1.0, -0.4, 2.0, -1.8, 0.7, 1.4, -0.9])


def omega(pos):
    d = [min(p, L - p) for p in pos]
    return sorted(sorted(range(3), key=lambda j: (d[j], j))[:2])


def Vp(pos):
    return float(sum(PA[p] for p in pos))


def dUp(pos):
    return float(sum(PB[pos[j]] for j in omega(pos))), {"omega": omega(pos)}


def omega_chain(variant, n=150000, seed=6):
    rng = np.random.default_rng(seed)
    pos = (0, 3, 5)
    st = {"x": pos, "V": Vp(pos), "dU": dUp(pos)[0], "details": dUp(pos)[1]}
    states = [(a, b, c) for a in range(L) for b in range(L) for c in range(L)]
    index = {s: i for i, s in enumerate(states)}
    counts = np.zeros((20, len(states)))
    for s in range(n):
        x = st["x"]
        if rng.random() < 0.5:                                   # Omega move
            j = omega(x)[int(rng.integers(2))]
            adm = (lambda a, y, j=j: j in omega(y)) if variant == "correct" else None
        else:                                                    # global move: any particle
            j, adm = int(rng.integers(3)), None
        draw = lambda r: 1 if r.random() < 0.5 else -1          # noqa: E731

        def apply(x, d, j=j):
            y = list(x)
            y[j] = (y[j] + d) % L
            return tuple(y)
        st, _ = two_stage_step(st, 1, draw, apply, Vp, dUp, ETA, RT, rng, adm)
        counts[s * 20 // n, index[st["x"]]] += 1
    w = np.array([np.exp(-(Vp(p) + ETA * dUp(p)[0]) / RT) for p in states])
    w /= w.sum()
    batch = counts / counts.sum(1, keepdims=True)
    est, se = batch.mean(0), batch.std(0, ddof=1) / np.sqrt(20)
    keep = w > 2e-3
    return float((np.abs(est - w) / np.maximum(se, 1e-12))[keep].max())


class TestSelection(unittest.TestCase):
    def test_reverse_reachability(self):
        self.assertLess(omega_chain("correct"), 5.0)
        self.assertGreater(omega_chain("ignore_leaving"), 10.0)

    def test_inadmissible_costs_nothing(self):
        calls = {"V": 0, "dU": 0}

        def V(x):
            calls["V"] += 1
            return 0.0

        def dU(y):
            calls["dU"] += 1
            return 0.0, {}
        st = {"x": (1, 2, 3), "V": 0.0, "dU": 0.0, "details": {}}
        new, r = two_stage_step(st, 1, lambda r: 1, lambda x, d: x, V, dU, ETA, RT, np.random.default_rng(0),
                                lambda x, y: False)
        self.assertIs(new, st)
        self.assertEqual((r["n_V"], r["n_model"], calls["V"], calls["dU"]), (0, 0, 0, 0))
        self.assertFalse(r["admissible"])


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12)
        self.x = self.rng.normal(0, 0.3, (40, 3))
        self.w = [36, 37, 38]
        self.x[37] = self.x[36] + [0.09572, 0, 0]
        self.x[38] = self.x[36] + [-0.024, 0.0927, 0]

    def jac_det(self, f, x):
        v0, h = x.ravel(), 1e-6
        J = np.array([(f(v0 + h * e) - f(v0 - h * e)) / (2 * h) for e in np.eye(v0.size)]).T
        return np.linalg.det(J)

    def test_inverse_lengths_jacobian(self):
        for name in ("T1", "T5"):
            draw, apply = torsion_move(name)
            th = draw(self.rng)
            y = apply(self.x, th)
            np.testing.assert_allclose(apply(y, -th), self.x, atol=1e-12)
            self.assertAlmostEqual(self.jac_det(lambda v: apply(v.reshape(-1, 3), th).ravel(), self.x[:35]), 1.0, places=5)
        cons = [(36, 37), (36, 38), (37, 38)]
        for kind in ("trans", "rot"):
            draw, apply = water_move(self.w, kind)
            m = draw(self.rng)
            y = apply(self.x, m)
            inv = -m if kind == "trans" else (m[0], -m[1])
            np.testing.assert_allclose(apply(y, inv), self.x, atol=1e-12)
            np.testing.assert_allclose(gram(y, cons, np.ones(40) * 2), gram(self.x, cons, np.ones(40) * 2), atol=1e-12)
            if kind == "rot":
                np.testing.assert_allclose(y[36], self.x[36], atol=0)          # about O
            sub = self.x[[36, 37, 38]]

            def f(v, m=m):
                z = self.x.copy()
                z[[36, 37, 38]] = v.reshape(3, 3)
                return apply(z, m)[[36, 37, 38]].ravel()
            self.assertAlmostEqual(self.jac_det(f, sub), 1.0, places=5)

    def test_draw_symmetry(self):
        dt, _ = torsion_move("T1")
        th = np.array([dt(self.rng) for _ in range(20000)])
        self.assertLess(abs(th.mean()), 0.01)
        self.assertLessEqual(np.abs(th).max(), np.radians(20.0))
        drt, _ = water_move(self.w, "rot")
        ms = [drt(self.rng) for _ in range(20000)]
        ax = np.array([m[0] for m in ms])
        ph = np.array([m[1] for m in ms])
        self.assertLess(np.abs(ax.mean(0)).max(), 0.02)
        self.assertLess(abs(ph.mean()), 0.005)
        self.assertLessEqual(np.abs(ph).max(), np.radians(15.0))
        dtr, _ = water_move(self.w, "trans")
        d = np.array([dtr(self.rng) for _ in range(20000)])
        self.assertLessEqual(np.abs(d).max(), 0.01)
        self.assertLess(np.abs(d.mean(0)).max(), 2e-4)


if __name__ == "__main__":
    unittest.main()
