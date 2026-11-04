"""CPU tests of the T1 window ladder (torsion_window.py; TORSION_WINDOW_PLAN_DRAFT_2026-11-04.md §6). No molecule or model.

1. Finite periodic toy (torsion t on 36 points, solvent s on 5 points; V, dU couple them), driven by the real pure
   functions: biased two-stage moves, the HMC-type acceptance with dW, and the exchange schedule / criterion.
   Every label's histogram matches exp(-(V + eta dU + W_b)). Negative controls must fail: bias counted in both stages,
   exchange without the bias terms, exchange with the sign flipped; a mixed-eta label set checks the eta term.
2. MBAR from all window samples back to b = 0 recovers the unbiased t marginal; naive pooling must not.
3. (eta x b) estimator toy (exact samples from each state): MBAR a0 = F(1, 0) - F(0, 0) and the eta-trapezoid of the
   reweighted atomic means at (eta, b = 0) match the exact values, with sum_i A_i = TI of dU; plugging biased-window
   means into the eta integral must not.
"""

import unittest

import numpy as np

from torsion_window import bias, biased_step, exchange, hmc_log_accept, swap_log_accept

NT, NS, RT = 36, 5, 1.0
TT = 2 * np.pi * np.arange(NT) / NT
rng0 = np.random.default_rng(0)
A_T = 1.5 * np.cos(TT) + 0.8 * np.cos(2 * TT + 0.4)
B_S = np.array([0.0, 0.6, 1.2, 0.4, 1.0])
C_TS = 0.9 * np.cos(TT[:, None] - 0.7 * np.arange(NS)[None])
D_TS = 1.2 * np.sin(TT[:, None] + 0.5 * np.arange(NS)[None]) - 0.4 * np.arange(NS)[None]


def V(x):
    return A_T[x[0]] + B_S[x[1]] + C_TS[x[0], x[1]]


def dU(x):
    return D_TS[x[0], x[1]]


def exact(label):
    eta = label[0]
    u = A_T[:, None] + B_S[None] + C_TS + eta * D_TS + bias(TT, label)[:, None]
    p = np.exp(-(u - u.min()) / RT)
    return p / p.sum()


WINDOWS = [(1.0, 0.0, 0.0)] + [(1.0, 3.0, c) for c in np.radians([-90, -40, 10, 60, 110, 160])]
MIXED = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]


def ladder(labels, variant="correct", sweeps=40000, seed=3):
    rng, rx = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    K = len(labels)
    states = [{"x": (0, 0), "V": V((0, 0)), "dU": dU((0, 0)), "details": None, "t1": TT[0]} for _ in range(K)]
    perm = list(range(K))
    H = np.zeros((20, K, NT, NS))
    t1_of = lambda y: TT[y[0]]                                         # noqa: E731
    for s in range(sweeps):
        inv = np.empty(K, int)
        inv[perm] = np.arange(K)
        for w in range(K):
            lab, st = labels[inv[w]], states[w]
            # HMC-type joint move: symmetric proposal, dH = dV, full acceptance incl. eta d(dU) and dW
            y = ((st["x"][0] + rng.choice((-1, 1))) % NT, st["x"][1] + rng.choice((-1, 1)))
            if 0 <= y[1] < NS:
                lr = hmc_log_accept(V(y) - st["V"], lab[0], dU(y), st["dU"], bias(TT[y[0]], lab), bias(st["t1"], lab), RT)
                if np.log(rng.random()) < lr:
                    st = {"x": y, "V": V(y), "dU": dU(y), "details": None, "t1": TT[y[0]]}
            # torsion and solvent single moves, biased two-stage
            tdraw = lambda r: int(r.choice((-2, -1, 1, 2)))               # noqa: E731
            tapp = lambda x, d: ((x[0] + d) % NT, x[1])                  # noqa: E731
            sdraw = lambda r: int(r.choice((-1, 1)))                      # noqa: E731
            sapp = lambda x, d: (x[0], x[1] + d)                          # noqa: E731
            sadm = lambda x, y: 0 <= y[1] < NS                            # noqa: E731
            if variant == "double_bias":
                for draw, app, adm in ((tdraw, tapp, None), (sdraw, sapp, sadm)):
                    lab_eff = lab
                    dUb = lambda y, lab_eff=lab_eff: (dU(y) + bias(TT[y[0]], lab_eff) / max(lab_eff[0], 1e-12), None)  # noqa: E731
                    s2 = {**st, "dU": st["dU"] + bias(st["t1"], lab) / max(lab[0], 1e-12)}
                    new, r = biased_step(s2, 1, draw, app, V, dUb, t1_of, lab, RT, rng, adm)
                    if new is not s2:
                        st = {"x": new["x"], "V": new["V"], "dU": dU(new["x"]), "details": None, "t1": new["t1"]}
            else:
                st, _ = biased_step(st, 1, tdraw, tapp, V, lambda y: (dU(y), None), t1_of, lab, RT, rng)
                st, _ = biased_step(st, 1, sdraw, sapp, V, lambda y: (dU(y), None), t1_of, lab, RT, rng, sadm)
            states[w] = st
        caches = [(st["dU"], st["t1"]) for st in states]
        if variant == "swap_no_bias":
            perm, _ = exchange(perm, caches, [(l[0], 0.0, 0.0) for l in labels], s, rx, RT)
        elif variant == "swap_sign":
            import torsion_window as tw
            orig = tw.swap_log_accept
            tw.swap_log_accept = lambda la, lb, cx, cy, rt: -orig(la, lb, cx, cy, rt)
            try:
                perm, _ = exchange(perm, caches, labels, s, rx, RT)
            finally:
                tw.swap_log_accept = orig
        else:
            perm, _ = exchange(perm, caches, labels, s, rx, RT)
        for k, w in enumerate(perm):
            H[s * 20 // sweeps, k][states[w]["x"]] += 1
    zmax = 0.0
    for k, lab in enumerate(labels):
        b = H[:, k] / H[:, k].sum((1, 2), keepdims=True)
        est, se = b.mean(0), b.std(0, ddof=1) / np.sqrt(20)
        e = exact(lab)
        keep = e > 2e-3
        zmax = max(zmax, float((np.abs(est - e) / np.maximum(se, 1e-12))[keep].max()))
    return zmax, H


class TestLadder(unittest.TestCase):
    def test_windows(self):
        z, _ = ladder(WINDOWS)
        self.assertLess(z, 5.0)
        for bad in ("double_bias", "swap_no_bias", "swap_sign"):
            self.assertGreater(ladder(WINDOWS, bad, sweeps=20000)[0], 10.0, bad)

    def test_mixed_eta(self):
        self.assertLess(ladder(MIXED, sweeps=30000)[0], 5.0)
        self.assertGreater(ladder(MIXED, "swap_sign", sweeps=20000)[0], 10.0)

    def test_swap_formula(self):
        la, lb = (1.0, 3.0, 0.2), (0.5, 2.0, -1.0)
        cx, cy = (0.7, 1.1), (-0.4, -2.0)
        U = lambda l, c: l[0] * c[0] + bias(c[1], l)                       # noqa: E731
        want = -((U(la, cy) + U(lb, cx)) - (U(la, cx) + U(lb, cy))) / RT
        self.assertAlmostEqual(swap_log_accept(la, lb, cx, cy, RT), want, places=12)


class TestUnbias(unittest.TestCase):
    def test_mbar_back_to_b0(self):
        from pymbar import MBAR
        rng = np.random.default_rng(5)
        K = len(WINDOWS)
        # exact independent samples from each window (isolates the estimator)
        n = 4000
        samples = []
        for lab in WINDOWS:
            p = exact(lab).ravel()
            samples.append(rng.choice(NT * NS, size=n, p=p))
        idx = np.concatenate(samples)
        t = idx // NS
        u = np.array([bias(TT[t], lab) / RT for lab in WINDOWS])           # V, dU common to all labels: they cancel
        m = MBAR(u, np.full(K, n))
        w = m.weights()[:, 0]
        p_mbar = np.bincount(t, weights=w, minlength=NT)
        p_true = exact(WINDOWS[0]).sum(1)
        p_naive = np.bincount(t, minlength=NT) / len(t)
        self.assertLess(np.abs(p_mbar - p_true).max(), 0.01)
        self.assertGreater(np.abs(p_naive - p_true).max(), 0.03)


class TestEtaBEstimator(unittest.TestCase):
    def test_a0_and_atoms(self):
        from pymbar import MBAR
        rng = np.random.default_rng(7)
        etas = [0.0, 0.5, 1.0]
        wins = [(0.0, 0.0), (3.0, np.radians(-60)), (3.0, np.radians(80))]
        du = np.stack([0.7 * D_TS, 0.3 * D_TS + 0.5 * np.cos(TT)[:, None]])     # two "atoms": sum = D_TS + ...
        dUfull = du.sum(0)
        states = [(e, k, c) for e in etas for (k, c) in wins]
        u0 = A_T[:, None] + B_S[None] + C_TS

        def energy(st):
            return u0 + st[0] * dUfull + bias(TT, st)[:, None]

        def probs(st):
            p = np.exp(-(energy(st) - energy(st).min()) / RT)
            return p / p.sum()
        n = 6000
        idx = np.concatenate([rng.choice(NT * NS, size=n, p=probs(st).ravel()) for st in states])
        t, s = idx // NS, idx % NS
        ukn = np.array([energy(st)[t, s] / RT for st in states])
        m = MBAR(ukn, np.full(len(states), n))
        f = m.compute_free_energy_differences()["Delta_f"][0]
        i0 = [states.index((e, 0.0, 0.0)) for e in etas]
        a0 = RT * (f[i0[2]] - f[i0[0]])
        Z = lambda st: np.exp(-(energy(st) - u0.min()) / RT).sum()          # noqa: E731
        a0_true = -RT * np.log(Z(states[i0[2]]) / Z(states[i0[0]]))
        self.assertLess(abs(a0 - a0_true), 0.05)
        trap = lambda y: float(np.sum(0.5 * (np.asarray(y[1:]) + np.asarray(y[:-1])) * np.diff(etas)))   # noqa: E731
        A_est, A_true, A_bad = [], [], []
        for i in range(2):
            mu = m.compute_expectations(du[i][t, s])["mu"]
            A_est.append(trap([mu[j] for j in i0]))
            A_true.append(trap([(probs(states[j]) * du[i]).sum() for j in i0]))
            ib = [states.index((e,) + wins[1]) for e in etas]                # biased-window means used directly
            A_bad.append(trap([(probs(states[j]) * du[i]).sum() for j in ib]))
        np.testing.assert_allclose(A_est, A_true, atol=0.05)
        mu_full = m.compute_expectations(dUfull[t, s])["mu"]
        self.assertAlmostEqual(sum(A_est), trap([mu_full[j] for j in i0]), places=10)
        err = np.abs(np.array(A_est) - np.array(A_true)).max()               # the estimator's own error is the yardstick
        self.assertGreater(np.abs(np.array(A_bad) - np.array(A_true)).max(), 10 * err)


if __name__ == "__main__":
    unittest.main()
