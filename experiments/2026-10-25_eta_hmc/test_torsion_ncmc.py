"""CPU tests of the T1/T5 path proposal (torsion_ncmc.py; TORSION_PATH_MC_DECISION_2026-11-02.md "实现后的必要检查").
No molecule, GPU or model.

1. Finite-state chain driven by the real path_attempt (Metropolis kernels at every switch point, a symmetric cyclic
   "torsion" move at the weakest point, eta * dU at the ends): label histograms against exact exp(-(V + eta dU)),
   with and without the move. Negative controls built from the same attempts must fail: work without the switch terms,
   without the move term, or the end-point energy difference used as work.
2. Limits: no switch and no propagation = symmetric Metropolis on U; no move and no propagation = zero work.
3. Geometry: rotation inverse for single and both orders, the two orders give the same result (nested sides rotated about
   current bond axes), bond lengths on the moving side
   and the axis are kept, Jacobian determinant 1 (finite differences), angle / order draw symmetric.
"""

import unittest

import numpy as np

from torsion_ncmc import apply_spec, draw_spec, inverse_spec, path_attempt, rotate, SIDES, TORSIONS

A = np.array([0.0, 1.5, 0.3, 2.0, 0.8, 1.2])
B = np.array([0.0, 0.7, 1.4, 0.2])
C = np.array([[1.2, -0.8, 0.5, 2.0], [-1.0, 0.4, 1.8, -0.6], [0.9, 1.5, -1.3, 0.2],
              [-0.4, 2.2, 0.7, -1.5], [1.6, -0.2, -0.9, 1.1], [0.3, 0.8, 1.4, -1.2]])
DU = np.array([[0.5, -1.0, 1.5, 0.0], [-0.8, 1.2, 0.3, -1.6], [1.0, 0.2, -0.5, 1.8],
               [-1.4, 0.6, 1.1, -0.3], [0.7, -1.2, 0.0, 0.9], [1.3, -0.4, 0.8, -1.0]])
POINTS = [1.0, 0.6, 0.3, 0.0]                               # coupling scale; S0 first, weakest last
ETA, RT = 1.0, 1.0


def V(lam, x):
    t, e = x
    return A[t] + B[e] + lam * C[t, e]


def U(x):
    return V(1.0, x) + ETA * DU[x]


def make_kernel(rng):
    def propagate(lam, x):                                  # Metropolis on V_lam: t +-1 (cyclic) or e +-1 (bounded)
        t, e = x
        if rng.random() < 0.5:
            y = ((t + (1 if rng.random() < 0.5 else -1)) % 6, e)
        else:
            e2 = e + (1 if rng.random() < 0.5 else -1)
            if not 0 <= e2 < 4:
                return x
            y = (t, e2)
        return y if np.log(rng.random()) < -(V(lam, y) - V(lam, x)) / RT else x
    return propagate


def make_move(rng, on=True):
    def move(x):
        if not on:
            return x, None
        d = (-2, -1, 1, 2)[int(rng.integers(4))]            # symmetric, bijective, unit Jacobian
        return ((x[0] + d) % 6, x[1]), d
    return move


def chain(variant, move_on=True, n=60000, seed=3):
    rng = np.random.default_rng(seed)
    prop, mv = make_kernel(rng), make_move(rng, move_on)
    x = (0, 0)
    counts = np.zeros((20, 6, 4))
    for s in range(n):
        r = path_attempt(x, POINTS, prop, V, mv, lambda y: DU[y], DU[x], ETA, RT, rng,
                         label=lambda a, b: "switch")
        y = r["y"]
        if variant == "correct":
            acc = r["acc"]
        else:
            sw = r["W_total"] - r["W"]["rot"]
            w = {"no_switch": r["W"]["rot"], "no_rot": sw, "endpoint": V(1.0, y) - V(1.0, x)}[variant]
            acc = r["logu"] < -(w + ETA * (DU[y] - DU[x])) / RT
        if acc:
            x = y
        counts[s * 20 // n][x] += 1
    exact = np.exp(-np.array([[U((t, e)) for e in range(4)] for t in range(6)]) / RT)
    exact /= exact.sum()
    batch = counts / counts.sum((1, 2), keepdims=True)
    est, se = batch.mean(0), batch.std(0, ddof=1) / np.sqrt(20)
    return (np.abs(est - exact) / np.maximum(se, 1e-12)).max()


class TestFiniteState(unittest.TestCase):
    def test_with_move(self):
        self.assertLess(chain("correct"), 5.0)
        for bad in ("no_switch", "no_rot", "endpoint"):
            self.assertGreater(chain(bad), 10.0, f"negative control {bad} should fail")

    def test_without_move(self):
        self.assertLess(chain("correct", move_on=False), 5.0)
        for bad in ("no_switch", "endpoint"):
            self.assertGreater(chain(bad, move_on=False), 10.0, f"negative control {bad} (no move) should fail")


class TestLimits(unittest.TestCase):
    def test_symmetric_metropolis_and_zero_work(self):
        rng = np.random.default_rng(5)
        for _ in range(50):
            x = (int(rng.integers(6)), int(rng.integers(4)))
            r = path_attempt(x, [1.0], lambda lam, y: y, V, make_move(rng), lambda y: DU[y], DU[x], ETA, RT, rng)
            self.assertAlmostEqual(r["logR"], -(U(r["y"]) - U(x)) / RT, places=12)
            r0 = path_attempt(x, POINTS, lambda lam, y: y, V, make_move(rng, False), lambda y: DU[y], DU[x], ETA, RT, rng)
            self.assertAlmostEqual(r0["W_total"], 0.0, places=12)
            self.assertEqual(r0["y"], x)


class TestGeometry(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.x = rng.normal(0, 0.3, (35, 3))
        self.rng = rng

    def test_inverse_order_and_lengths(self):
        for names in (["T1"], ["T5"], ["T1", "T5"]):
            for _ in range(20):
                sp = draw_spec(names, self.rng)
                y = apply_spec(self.x, sp)
                np.testing.assert_allclose(apply_spec(y, inverse_spec(sp)), self.x, atol=1e-12)
                for n, _th in sp:
                    side = SIDES[n]
                    a, b = TORSIONS[n][1], TORSIONS[n][2]
                    np.testing.assert_allclose(y[[a, b]], apply_spec(self.x, [s for s in sp if s[0] != n])[[a, b]]
                                               if len(sp) > 1 else self.x[[a, b]], atol=1e-12)
                    d0 = np.linalg.norm(self.x[side][:, None] - self.x[side][None], axis=-1)
                    d1 = np.linalg.norm(y[side][:, None] - y[side][None], axis=-1)
                    if n == "T1":                               # T5's side lies inside T1's: whole T1 side rigid only for T1 alone
                        if len(sp) == 1:
                            np.testing.assert_allclose(d1, d0, atol=1e-12)
                    else:
                        np.testing.assert_allclose(d1, d0, atol=1e-12)
                    np.testing.assert_allclose(np.linalg.norm(y[b] - y[a]), np.linalg.norm(self.x[b] - self.x[a]), atol=1e-12)
        # rotations about the current bond axes of nested sides (T5 side inside T1 side) change the two dihedrals
        # independently, so both orders give the same result; the order draw and reversed inverse stay harmless
        y12 = apply_spec(self.x, [("T1", 1.0), ("T5", 2.0)])
        y21 = apply_spec(self.x, [("T5", 2.0), ("T1", 1.0)])
        np.testing.assert_allclose(y12, y21, atol=1e-12)

    def test_jacobian_and_draws(self):
        x = self.x[:30].copy()
        x[[3, 6, 14, 16, 2]] += 0                                   # indices used by T1 / T5 exist in the 30-atom toy
        f = lambda v: apply_spec(v.reshape(-1, 3), [("T1", 0.7), ("T5", -1.9)]).ravel()
        v0, h = x.ravel(), 1e-6
        J = np.array([(f(v0 + h * e) - f(v0 - h * e)) / (2 * h) for e in np.eye(v0.size)]).T
        self.assertAlmostEqual(np.linalg.det(J), 1.0, places=5)
        sp = [draw_spec(["T1", "T5"], self.rng) for _ in range(20000)]
        first_t1 = np.mean([s[0][0] == "T1" for s in sp])
        th = np.array([dict(s)["T1"] for s in sp])
        self.assertLess(abs(first_t1 - 0.5), 0.02)
        self.assertLess(abs(th.mean()), 0.05)
        self.assertLess(abs(np.mean(np.abs(th)) - np.pi / 2), 0.05)


if __name__ == "__main__":
    unittest.main()
