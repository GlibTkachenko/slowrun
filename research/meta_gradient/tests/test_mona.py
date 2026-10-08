"""Tests that the single-buffer MONA correction equals Algorithm 1 of the paper."""

import unittest

import torch

import mona


def _algorithm_1(gradients, beta, alpha):
    """Yields corrected gradients with the paper's previous-gradient and difference-EMA buffers."""
    prev = torch.zeros_like(gradients[0])
    avg = torch.zeros_like(gradients[0])
    for g in gradients:
        avg = beta * avg + (1 - beta) * (g - prev)
        prev = g.clone()
        yield g + alpha * avg


class MonaTest(unittest.TestCase):

    def test_single_buffer_equals_algorithm_1(self):
        torch.manual_seed(0)
        gradients = [torch.randn(3, 5, dtype=torch.float64) for _ in range(200)]
        beta = 0.98
        for alpha in (mona.default_alpha(beta), -5.0, 0.7):
            ema = torch.zeros(3, 5, dtype=torch.float64)
            for g, expected in zip(gradients, _algorithm_1(gradients, beta, alpha)):
                got = g.clone()
                mona.mona_correct_(got, ema, torch.tensor(beta, dtype=torch.float64),
                                  torch.tensor(alpha, dtype=torch.float64))
                torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-12)

    def test_default_alpha_mixes_gradient_with_preceding_ema(self):
        torch.manual_seed(1)
        beta = 0.9
        ema = torch.zeros(4, dtype=torch.float64)
        preceding = torch.zeros(4, dtype=torch.float64)
        for _ in range(50):
            g = torch.randn(4, dtype=torch.float64)
            got = g.clone()
            mona.mona_correct_(got, ema, torch.tensor(beta, dtype=torch.float64),
                              torch.tensor(mona.default_alpha(beta), dtype=torch.float64))
            torch.testing.assert_close(got, 0.5 * (g + preceding), rtol=1e-10, atol=1e-12)
            preceding = beta * preceding + (1 - beta) * g


if __name__ == '__main__':
    unittest.main()
