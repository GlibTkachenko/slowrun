"""Tests of randomized recurrent-credit recovery (Priority D)."""

import unittest

import torch

import credit


def _two_pass_gradients(scale: float | None):
    """Gradient of a two-pass shared-weight recurrence for one credit mode.

    None truncates the first pass (the record); a number differentiates through
    it with the path gradient scaled by that number.
    """
    torch.manual_seed(0)
    w = torch.randn(4, 4, requires_grad=True)
    x = torch.randn(3, 4)
    if scale is None:
        with torch.no_grad():
            h1 = torch.tanh(x @ w)
    else:
        h1 = credit.scale_backward(torch.tanh(x @ w), scale)
    h2 = torch.tanh(h1 @ w + x)
    h2.square().sum().backward()
    return w.grad.clone(), h1.detach()


class CreditTest(unittest.TestCase):

    def test_scale_backward_is_identity_forward(self):
        x = torch.randn(5, requires_grad=True)
        y = credit.scale_backward(x, 4.0)
        self.assertTrue(torch.equal(y, x))
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.full_like(x, 4.0)))

    def test_compensated_estimator_is_unbiased(self):
        g_direct, h_trunc = _two_pass_gradients(None)
        g_full, h_full = _two_pass_gradients(1.0)
        self.assertTrue(torch.equal(h_trunc, h_full))
        g_path = g_full - g_direct
        for p in (0.25, 0.5):
            g_on, _ = _two_pass_gradients(credit.path_scale('random_comp', p))
            expectation = p * g_on + (1 - p) * g_direct
            torch.testing.assert_close(expectation, g_full, rtol=1e-5, atol=1e-6)
            # Cov_Z = (1 - p)/p * g_path g_path^T on the diagonal.
            variance = p * (g_on - g_full).square() + (1 - p) * (g_direct - g_full).square()
            torch.testing.assert_close(variance, (1 - p) / p * g_path.square(), rtol=1e-4, atol=1e-7)

    def test_uncompensated_estimator_is_biased_toward_truncation(self):
        g_direct, _ = _two_pass_gradients(None)
        g_full, _ = _two_pass_gradients(1.0)
        expectation = 0.25 * g_full + 0.75 * g_direct
        self.assertGreater(float((expectation - g_full).norm()), 1e-4)

    def test_compensation_stays_unbiased_with_three_passes(self):
        def gradient(scale, rule):
            torch.manual_seed(0)
            w = torch.randn(4, 4, requires_grad=True)
            x = torch.randn(3, 4)
            h = x
            for i in range(3):
                if scale is None and i < 2:
                    with torch.no_grad():
                        h = torch.tanh(h @ w + x)
                else:
                    h = torch.tanh(h @ w + x)
                    if scale is not None and rule(i, 3):
                        h = credit.scale_backward(h, scale)
            h.square().sum().backward()
            return w.grad.clone()

        p = 0.25
        truncated = gradient(None, credit.is_credit_boundary)
        full = gradient(1.0, credit.is_credit_boundary)
        compensated = gradient(1.0 / p, credit.is_credit_boundary)
        torch.testing.assert_close(p * compensated + (1 - p) * truncated, full, rtol=1e-5, atol=1e-6)
        # Scaling every boundary multiplies the longest path by 1/p twice and is biased.
        every_boundary = gradient(1.0 / p, lambda i, passes: i < passes - 1)
        self.assertGreater(float((p * every_boundary + (1 - p) * truncated - full).norm()), 1e-3)

    def test_path_scale(self):
        self.assertEqual(credit.path_scale('random_comp', 0.25), 4.0)
        self.assertEqual(credit.path_scale('random', 0.25), 1.0)
        with self.assertRaises(ValueError):
            credit.path_scale('random_comp', 0.0)
        with self.assertRaises(ValueError):
            credit.path_scale('sometimes', 0.5)

    def test_sampler_is_synchronized_and_only_draws_on_multi_pass_steps(self):
        a = credit.CreditSampler(mode='random_comp', p=0.25, seed=7)
        b = credit.CreditSampler(mode='random_comp', p=0.25, seed=7)
        draws_a = [a.draw(2) for _ in range(4000)]
        self.assertEqual(draws_a, [b.draw(2) for _ in range(4000)])
        self.assertAlmostEqual(sum(draws_a) / len(draws_a), 0.25, delta=0.03)
        self.assertFalse(a.draw(1))
        self.assertEqual(a.multi_pass_steps, 4000)
        self.assertTrue(all(credit.CreditSampler(mode='full', p=0.1, seed=0).draw(2) for _ in range(5)))
        self.assertFalse(credit.CreditSampler(mode='trunc', p=0.5, seed=0).draw(2))


if __name__ == '__main__':
    unittest.main()
