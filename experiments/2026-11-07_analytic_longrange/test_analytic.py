import math
import unittest

import torch
from graph_longrange.realspace_electrostatics import RealSpaceFiniteDiffereneEnergy, RealSpaceFiniteDifferenceElectrostaticFeatures
from analytic_gto import AnalyticEnergy, AnalyticFeatures


class AnalyticTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        self.p = torch.tensor([[0., 0., 0.], [2., .7, -.3], [-.6, 1.8, .5]])
        self.s = torch.tensor([[.3, -.1, .05, .2], [-.3, .03, .15, -.1], [.1, .04, -.05, .08]])
        self.batch = torch.zeros(3, dtype=torch.long)
        self.R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])

    def tearDown(self):
        torch.set_default_dtype(self.previous)

    def rotated_sources(self):
        p = self.s[:, [3, 1, 2]] @ self.R.T
        return torch.cat([self.s[:, :1], p[:, [1, 2, 0]]], dim=1)

    def test_monopole_matches_original_and_dipole_stencil_converges(self):
        original = RealSpaceFiniteDiffereneEnergy(0, 1.)
        analytic = AnalyticEnergy(original)
        torch.testing.assert_close(analytic(self.s[:, :1], self.p, self.batch), original(self.s[:, :1], self.p, self.batch), rtol=1e-13, atol=1e-13)
        exact = AnalyticEnergy(RealSpaceFiniteDiffereneEnergy(1, 1.))(self.s, self.p, self.batch)
        errors = [float((RealSpaceFiniteDiffereneEnergy(1, 1., offset=h)(self.s, self.p, self.batch) - exact).abs()) for h in (.02, .01, .005)]
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[1], errors[0])

    def test_energy_rotates_translates_and_has_correct_position_gradient(self):
        energy = AnalyticEnergy(RealSpaceFiniteDiffereneEnergy(1, 1.))
        value = energy(self.s, self.p, self.batch)
        torch.testing.assert_close(energy(self.rotated_sources(), self.p @ self.R.T, self.batch), value, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(energy(self.s, self.p + 17., self.batch), value, rtol=1e-13, atol=1e-13)
        p = self.p.clone().requires_grad_(True)
        grad, = torch.autograd.grad(energy(self.s, p, self.batch).sum(), p)
        delta = torch.zeros_like(p); delta[1, 0] = 1e-5
        fd = (energy(self.s, p + delta, self.batch) - energy(self.s, p - delta, self.batch)) / 2e-5
        self.assertAlmostEqual(float(grad[1, 0]), float(fd), places=8)

    def test_field_stencil_limit_and_vector_covariance(self):
        def template(h):
            return RealSpaceFiniteDifferenceElectrostaticFeatures(1, 1., 1, [1., 1.3], offset=h)
        analytic = AnalyticFeatures(template(.1))
        exact = analytic(self.s[:, None], self.p, self.batch)[0]
        errors = [float((template(h)(self.s[:, None], self.p, self.batch)[0] - exact).norm()) for h in (.1, .05, .025)]
        self.assertLess(errors[2], errors[1]); self.assertLess(errors[1], errors[0])
        rotated = analytic(self.rotated_sources()[:, None], self.p @ self.R.T, self.batch)[0]
        torch.testing.assert_close(rotated[:, :2], exact[:, :2], rtol=1e-12, atol=1e-12)
        vec = exact[:, 2:].reshape(3, 2, 3)[..., [2, 0, 1]]
        expected = (vec @ self.R.T)[..., [1, 2, 0]]
        torch.testing.assert_close(rotated[:, 2:].reshape(3, 2, 3), expected, rtol=1e-12, atol=1e-12)

    def test_cross_energy_euler_split_and_separate_graphs(self):
        energy = AnalyticEnergy(RealSpaceFiniteDiffereneEnergy(1, 1.))
        s = self.s.clone().requires_grad_(True)
        mask = torch.tensor([[1.], [0.], [0.]])
        cross = energy(s, self.p, self.batch) - energy(s*mask, self.p, self.batch) - energy(s*(1-mask), self.p, self.batch)
        grad, = torch.autograd.grad(cross.sum(), s)
        atom = (s*grad).sum(-1)
        torch.testing.assert_close(atom[:1].sum(), cross[0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(atom[1:].sum(), cross[0], rtol=1e-12, atol=1e-12)
        separate = energy(s, self.p, torch.arange(3))
        torch.testing.assert_close(separate, torch.zeros(3), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
