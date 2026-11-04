"""CPU tests of the torsion + water collective move (collective_mc.py; TORSION_WATER_COLLECTIVE_MC_REVIEW_2026-11-02.md §5).
No molecule, GPU or model. Gates 1 and 3 on the real frames: `collective_mc.py check`.

2. Full toy system (axis anchors, side atoms and waters in the core, the transition zone and outside): inverse, invariants,
   numerical Jacobian determinant of the whole map at three step sizes.
3. Periodic images: a water shifted by box vectors gets the same displacement; a support larger than half the box raises.
4. Known equilibrium with torsion-water coupling (side atom on a circle about the axis, one rigid water; U depends on the
   torsion psi and the water O only, so psi is exp(-B cos psi) and (Delta = phi_O - psi, rho, z) follow from a grid):
   base local kernels (1/6 + 1/6) mixed with the collective two-stage step (2/3) reproduce the marginals; negative controls must fail:
   the model stage left out, and a radially scaling (non volume preserving) variant used without its Jacobian.
5. Faults and bookkeeping through two_stage_step: non-finite MM energy raises; a rejection keeps the input state; at most
   one model query; no redraw until acceptance.
"""

import unittest

import numpy as np

from collective_mc import NotApplicable, collective_apply, geometry, rodrigues
from local_geom_mc import two_stage_step

MASS = np.array([12.0, 12.0, 12.0, 12.0] + [16.0, 1.0, 1.0] * 3)


def toy_system(rng):
    """atoms 0 = a (origin), 1 = b (on z), 2-3 side, then 3 waters (O, H, H): core, transition, outside."""
    x = np.zeros((13, 3))
    x[1] = [0, 0, 0.15]
    x[2] = [0.25, 0.05, 0.2]
    x[3] = [0.20, -0.15, 0.30]
    wd = np.array([[0, 0, 0], [0.09572, 0, 0], [-0.024, 0.0927, 0]])
    rL = max(np.hypot(*x[2][:2]), np.hypot(*x[3][:2]))
    for k, (r, z) in enumerate(((0.3, 0.25), (rL + 0.30 + 0.15, 0.25), (rL + 0.30 + 0.45, 0.25))):
        phi = rng.uniform(0, 2 * np.pi)
        R = rodrigues(wd, np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8]), rng.uniform(0, 6))
        x[4 + 3 * k: 7 + 3 * k] = np.array([r * np.cos(phi), r * np.sin(phi), z]) + R
    return x


WAT = np.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]])
SIDE = np.array([2, 3])
BOX = np.array([5.0, 5.0, 5.0])


class TestToyGeometry(unittest.TestCase):
    def test_inverse_invariants_jacobian(self):
        rng = np.random.default_rng(2)
        x = toy_system(rng)
        th = 0.31
        y, g = collective_apply(x, th, 0, 1, SIDE, WAT, MASS, BOX, with_info=True)
        self.assertTrue(g["w"][0] == 1.0 and 0 < g["w"][1] < 1 and g["w"][2] == 0.0, g["w"])
        np.testing.assert_allclose(collective_apply(y, -th, 0, 1, SIDE, WAT, MASS, BOX), x, atol=1e-14)
        g2 = geometry(y, 0, 1, SIDE, WAT, MASS, BOX)
        np.testing.assert_allclose([g2["rL"], g2["zmin"], g2["zmax"]], [g["rL"], g["zmin"], g["zmax"]], atol=1e-14)
        np.testing.assert_allclose(g2["w"], g["w"], atol=1e-14)
        v0 = x.ravel()
        for h in (1e-4, 1e-5, 1e-6):
            J = np.array([(collective_apply((v0 + h * e).reshape(-1, 3), th, 0, 1, SIDE, WAT, MASS, BOX)
                           - collective_apply((v0 - h * e).reshape(-1, 3), th, 0, 1, SIDE, WAT, MASS, BOX)).ravel() / (2 * h)
                          for e in np.eye(v0.size)]).T
            self.assertAlmostEqual(np.linalg.det(J), 1.0, places=6, msg=f"step {h}")
        for k in range(3):                                         # every water stays rigid
            d0 = x[WAT[k]][:, None] - x[WAT[k]][None]
            d1 = y[WAT[k]][:, None] - y[WAT[k]][None]
            np.testing.assert_allclose(np.linalg.norm(d1, axis=-1), np.linalg.norm(d0, axis=-1), atol=1e-14)

    def test_periodic(self):
        rng = np.random.default_rng(4)
        x = toy_system(rng)
        y = collective_apply(x, 0.4, 0, 1, SIDE, WAT, MASS, BOX)
        xs = x.copy()
        for k, s in enumerate(([5.0, 0, 0], [0, -5.0, 5.0], [-5.0, 5.0, 0])):
            xs[WAT[k]] += s
        ys = collective_apply(xs, 0.4, 0, 1, SIDE, WAT, MASS, BOX)
        np.testing.assert_allclose(ys - xs, y - x, atol=1e-12)
        with self.assertRaises(NotApplicable):
            collective_apply(x, 0.4, 0, 1, SIDE, WAT, MASS, np.array([1.5, 1.5, 1.5]))


# ---------------------------------------------------------------- known equilibrium

R_S, Z_S = 0.3, 0.2                        # side atom on a circle of radius R_S at height Z_S
A_REP, SIG, B_T, C_REL, D_DU, K_CONF, R_CONF = 2.0, 0.25, 1.2, 2.5, -2.0, 40.0, 0.6
C0 = np.array([0.0, 0.0, 0.2])
W1 = np.array([[1, 2, 3]])
M1 = np.array([12.0, 12.0, 12.0, 16.0, 1.0, 1.0])
WD = np.array([[0, 0, 0], [0.09572, 0, 0], [-0.024, 0.0927, 0]])


def parts(x):
    s, o = x[0], x[3]
    psi = np.arctan2(s[1], s[0])
    g = np.exp(-np.sum((s - o) ** 2) / SIG ** 2)
    conf = K_CONF * max(0.0, np.linalg.norm(o - C0) - R_CONF) ** 2
    V = B_T * np.cos(psi) + A_REP * g + conf
    dU = C_REL * np.cos(np.arctan2(o[1], o[0]) - psi) + D_DU * g
    return V, dU


def full_x(state):
    """state: (psi, O, R) -> coordinates [side S, axis b, axis a, O, H, H] laid out for collective_apply."""
    psi, o, Rw = state
    x = np.zeros((6, 3))
    x[0] = [R_S * np.cos(psi), R_S * np.sin(psi), Z_S]
    x[1] = [0, 0, 0.15]
    x[2] = [0, 0, 0]
    x[3:6] = o + Rw
    return x


def unpack(x):
    return (float(np.arctan2(x[0, 1], x[0, 0])), x[3].copy(), x[3:6] - x[3])


def exact_marginals(nb=12, nr=10):
    """psi: exp(-B cos psi); (Delta, rho, z) on a grid with rho weight."""
    ps = (np.arange(720) + 0.5) * 2 * np.pi / 720
    p_psi = np.exp(-B_T * np.cos(ps))
    D = (np.arange(240) + 0.5) * 2 * np.pi / 240
    rmax = R_CONF + 0.6
    rho = (np.arange(160) + 0.5) * rmax / 160
    zz = 0.2 + (np.arange(200) + 0.5) * 2 * rmax / 200 - rmax
    DD, RR, ZZ = np.meshgrid(D, rho, zz, indexing="ij")
    # side atom at psi = 0 without loss of generality: O at angle Delta
    sx, sy, sz = R_S, 0.0, Z_S
    ox, oy = RR * np.cos(DD), RR * np.sin(DD)
    g = np.exp(-((sx - ox) ** 2 + (sy - oy) ** 2 + (sz - ZZ) ** 2) / SIG ** 2)
    conf = K_CONF * np.maximum(0.0, np.sqrt(RR ** 2 + (ZZ - 0.2) ** 2) - R_CONF) ** 2
    w = np.exp(-(A_REP * g + conf + C_REL * np.cos(DD) + D_DU * g)) * RR
    pD = w.sum((1, 2))
    pR = w.sum((0, 2))
    to_bins = lambda p, n: p.reshape(n, -1).sum(1) / p.sum()      # noqa: E731
    return to_bins(p_psi, nb), to_bins(pD, nb), to_bins(pR, nr), rmax


def eq_chain(variant="correct", n=240000, seed=8):
    rng = np.random.default_rng(seed)
    nb, nr = 12, 10
    _, _, _, rmax = exact_marginals()
    x = full_x((0.5, np.array([0.1, 0.4, 0.3]), WD))
    V0, U0 = parts(x)
    st = {"x": x, "V": V0, "dU": U0, "details": None}
    H = np.zeros((20, 3, max(nb, nr)))

    def apply_c(y, th):
        return collective_apply(y, th, 2, 1, np.array([0]), W1 + 2, M1, BOX, h=0.2, ell=0.3)

    def apply_bad(y, th):                                   # rotation plus radial scaling of the water: Jacobian != 1
        z = apply_c(y, th)
        o = z[3]
        rad = np.array([o[0], o[1], 0.0])
        z[3:6] += rad * (np.exp(0.5 * th) - 1)
        return z
    for s in range(n):
        u = rng.random()
        if u < 1 / 6:                                        # base: torsion only, full target Metropolis
            th = rng.uniform(-0.8, 0.8)
            y = st["x"].copy()
            y[0] = rodrigues(y[0], np.array([0, 0, 1.0]), th)
            Vy, Uy = parts(y)
            if np.log(rng.random()) < -((Vy + Uy) - (st["V"] + st["dU"])):
                st = {"x": y, "V": Vy, "dU": Uy, "details": None}
        elif u < 1 / 3:                                      # base: rigid water translation
            y = st["x"].copy()
            y[3:6] += rng.uniform(-0.12, 0.12, 3)
            Vy, Uy = parts(y)
            if np.log(rng.random()) < -((Vy + Uy) - (st["V"] + st["dU"])):
                st = {"x": y, "V": Vy, "dU": Uy, "details": None}
        else:                                                # collective two-stage step
            draw = lambda r: float(r.uniform(-0.6, 0.6))   # noqa: E731
            app = apply_bad if variant == "radial_no_jacobian" else apply_c
            new, r = two_stage_step(st, 1, draw, app, lambda y: parts(y)[0], lambda y: (parts(y)[1], None), 1.0, 1.0, rng)
            if variant == "no_stage2":
                if r["stage1"]:
                    y = r["y"]
                    st = {"x": y, "V": parts(y)[0], "dU": parts(y)[1], "details": None}
            else:
                st = new
        psi, o, _ = unpack(st["x"])
        b = int(s * 20 // n)
        H[b, 0, int((psi % (2 * np.pi)) / (2 * np.pi) * nb) % nb] += 1
        H[b, 1, int(((np.arctan2(o[1], o[0]) - psi) % (2 * np.pi)) / (2 * np.pi) * nb) % nb] += 1
        H[b, 2, min(int(np.hypot(o[0], o[1]) / rmax * nr), nr - 1)] += 1
    ex = exact_marginals(nb, nr)
    zmax = 0.0
    for k, e in enumerate(ex[:3]):
        m = len(e)
        bt = H[:, k, :m] / H[:, k, :m].sum(1, keepdims=True)
        est, se = bt.mean(0), bt.std(0, ddof=1) / np.sqrt(20)
        keep = e > 5e-3
        zmax = max(zmax, float((np.abs(est - e) / np.maximum(se, 1e-12))[keep].max()))
    return zmax


class TestEquilibrium(unittest.TestCase):
    def test_target_and_negative_controls(self):
        self.assertLess(eq_chain("correct"), 5.0)
        self.assertGreater(eq_chain("no_stage2"), 10.0)
        self.assertGreater(eq_chain("radial_no_jacobian"), 10.0)


class TestFaults(unittest.TestCase):
    def test_nonfinite_rejection_counts(self):
        rng = np.random.default_rng(1)
        x = toy_system(rng)
        st = {"x": x, "V": 0.0, "dU": 0.0, "details": {"tag": "x"}}
        app = lambda y, th: collective_apply(y, th, 0, 1, SIDE, WAT, MASS, BOX)   # noqa: E731
        with self.assertRaises(FloatingPointError):
            two_stage_step(st, 1, lambda r: 0.2, app, lambda y: np.nan, lambda y: (0.0, {}), 1.0, 1.0, rng)
        calls = [0]

        def dU(y):
            calls[0] += 1
            return 1e6, {"tag": "y"}                       # stage 2 always rejects
        new, r = two_stage_step(st, 1, lambda r: 0.2, app, lambda y: -1e6, dU, 1.0, 1.0, rng)
        self.assertTrue(r["stage1"] and not r["acc"])
        self.assertIs(new, st)
        self.assertEqual((calls[0], r["n_model"]), (1, 1))
        new, r = two_stage_step(st, 1, lambda r: 0.2, app, lambda y: 1e6, dU, 1.0, 1.0, rng)
        self.assertTrue(not r["stage1"] and r["n_model"] == 0 and new is st and calls[0] == 1)


if __name__ == "__main__":
    unittest.main()
