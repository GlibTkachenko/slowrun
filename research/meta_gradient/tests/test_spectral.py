"""Tests of the spectral response maps and the polar linearization."""

import math
import unittest

import torch

import spectral


def _polar(m: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(m.double(), full_matrices=False)
    return u @ vh


class SpectralResponseTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_eigh_path_matches_float64_reference(self):
        for shape in ((3, 48, 20), (3, 20, 48), (2, 32, 32)):
            x = torch.randn(*shape, dtype=torch.float64)
            for exponent, eps in ((0.5, 0.05), (2 / 3, 0.1), (1.0, 0.3)):
                with self.subTest(shape=shape, exponent=exponent, eps=eps):
                    got = spectral.spectral_response(x, exponent=exponent, eps=eps)
                    want = spectral.reference_response(x, exponent=exponent, eps=eps)
                    self.assertLess(spectral.relative_error(got, want).max().item(), 1e-4)

    def test_float64_path_resolves_small_eps(self):
        # Spectra down to 1e-4 with eps = 1e-3 exceed float32 eigendecomposition accuracy.
        values = torch.exp(torch.empty(2, 64).uniform_(math.log(1e-4), 0.0))
        x = spectral.synthetic_matrices(values, rows=128, cols=64)
        want = spectral.reference_response(x, exponent=0.5, eps=1e-3)
        got = spectral.spectral_response(x, exponent=0.5, eps=1e-3, dtype=torch.float64)
        self.assertEqual(got.dtype, torch.float64)
        self.assertLess(spectral.relative_error(got, want).max().item(), 1e-8)

    def test_outputs_have_polar_frobenius_norm(self):
        x = torch.randn(4, 40, 24)
        for fn in (lambda t: spectral.spectral_response(t, exponent=2 / 3, eps=0.1),
                   lambda t: spectral.augmented_polar(t, eps=0.1)):
            norms = fn(x).norm(dim=(-2, -1))
            torch.testing.assert_close(norms, torch.full_like(norms, math.sqrt(24)), rtol=1e-4, atol=0)

    def test_maps_are_scale_invariant(self):
        x = torch.randn(2, 30, 18, dtype=torch.float64)
        a = spectral.spectral_response(x, exponent=2 / 3, eps=0.1)
        b = spectral.spectral_response(1e-3 * x, exponent=2 / 3, eps=0.1)
        self.assertLess(spectral.relative_error(a, b).max().item(), 1e-4)

    def test_zero_input_maps_to_zero(self):
        x = torch.zeros(2, 16, 8)
        for y in (spectral.spectral_response(x, exponent=2 / 3, eps=0.1),
                  spectral.augmented_polar(x, eps=0.1)):
            self.assertTrue(torch.isfinite(y).all())
            self.assertEqual(float(y.abs().max()), 0.0)

    def test_nonpositive_eps_is_rejected(self):
        with self.assertRaises(ValueError):
            spectral.spectral_response(torch.randn(8, 4), exponent=0.5, eps=0.0)

    def test_augmented_polar_has_polar_express_accuracy(self):
        values = torch.exp(torch.empty(3, 64).uniform_(math.log(1e-2), 0.0))
        x = spectral.synthetic_matrices(values, rows=160, cols=64)
        got = spectral.augmented_polar(x, eps=0.1)
        want = spectral.reference_response(x, exponent=0.5, eps=0.1)
        self.assertLess(spectral.relative_error(got, want).max().item(), 0.15)

    def test_polar_express_matches_record_code(self):
        # Verbatim from the record's muon_step_fused.
        coeffs = spectral.POLAR_EXPRESS_COEFFS
        for shape in ((24, 40), (40, 24)):
            g = torch.randn(3, *shape)
            X = g.bfloat16()
            X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
            if g.size(-2) > g.size(-1):
                for a, b, c in coeffs[:5]:
                    A = X.mT @ X
                    X = a * X + X @ (b * A + c * (A @ A))
            else:
                for a, b, c in coeffs[:5]:
                    A = X @ X.mT
                    X = a * X + (b * A + c * (A @ A)) @ X
            self.assertTrue(torch.equal(spectral.polar_express(g), X))

    def test_alternative_maps_match_the_polar_express_norm_per_matrix(self):
        values = torch.exp(torch.empty(3, 24).uniform_(math.log(1e-3), 0.0))
        x = spectral.synthetic_matrices(values, rows=40, cols=24).float()
        reference = spectral.polar_express(x).float().norm(dim=(-2, -1))
        for method, exponent in (('eigh', 2 / 3), ('eigh', 0.5), ('augmented_polar', 0.5)):
            y = spectral.apply_map(x, method=method, exponent=exponent, eps=0.1)
            torch.testing.assert_close(y.norm(dim=(-2, -1)), reference, rtol=1e-5, atol=0)
        ideal = spectral.apply_map(x, method='eigh', exponent=2 / 3, eps=0.1, norm='ideal')
        torch.testing.assert_close(ideal.norm(dim=(-2, -1)), torch.full((3,), math.sqrt(24)),
                                   rtol=1e-5, atol=0)

    def test_apply_map_dispatch(self):
        x = torch.randn(2, 12, 8)
        self.assertEqual(spectral.apply_map(x, method='polar_express').dtype, torch.bfloat16)
        with self.assertRaises(ValueError):
            spectral.apply_map(x, method='augmented_polar', exponent=0.7, eps=0.1)
        with self.assertRaises(ValueError):
            spectral.apply_map(x, method='nope')


class PolarLinearizationTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1)

    def test_derivative_matches_finite_differences(self):
        for shape in ((30, 20), (20, 30), (16, 16)):
            m = torch.randn(*shape, dtype=torch.float64)
            e = torch.randn(*shape, dtype=torch.float64)
            h = 1e-6
            numeric = (_polar(m + h * e) - _polar(m - h * e)) / (2 * h)
            with self.subTest(shape=shape):
                error = (spectral.polar_derivative(m, e) - numeric).norm() / numeric.norm()
                self.assertLess(float(error), 1e-6)

    def test_energy_fractions_partition_the_perturbation(self):
        m, e = torch.randn(30, 20), torch.randn(30, 20)
        lin = spectral.linearize_polar(m, e)
        total = lin.symmetric_fraction + lin.antisymmetric_fraction + lin.complement_fraction
        self.assertAlmostEqual(total, 1.0, places=9)
        self.assertAlmostEqual(sum(lin.band_energy), 1.0, places=9)

    def test_symmetric_perturbation_is_discarded(self):
        m = torch.randn(12, 12, dtype=torch.float64)
        u, _, vh = torch.linalg.svd(m)
        s = torch.randn(12, 12, dtype=torch.float64)
        e = u @ (s + s.mT) @ vh
        lin = spectral.linearize_polar(m, e)
        self.assertAlmostEqual(lin.symmetric_fraction, 1.0, places=9)
        self.assertLess(float(spectral.polar_derivative(m, e).norm()), 1e-9)

    def test_gain_matches_derivative_norm(self):
        m, e = torch.randn(25, 15, dtype=torch.float64), torch.randn(25, 15, dtype=torch.float64)
        lin = spectral.linearize_polar(m, e)
        expected = spectral.polar_derivative(m, e).norm() * torch.linalg.svdvals(m)[0] / e.norm()
        self.assertAlmostEqual(lin.gain, float(expected), places=7)

    def test_band_cosines(self):
        m = torch.randn(20, 12)
        e1, e2 = torch.randn(20, 12), torch.randn(20, 12)
        same = spectral.band_cosines(spectral.linearize_polar(m, e1), spectral.linearize_polar(m, e1))
        for value in same:
            self.assertAlmostEqual(value, 1.0, places=9)
        flipped = spectral.band_cosines(spectral.linearize_polar(m, e1), spectral.linearize_polar(m, -e1))
        self.assertAlmostEqual(flipped[-1], -1.0, places=9)
        other = spectral.band_cosines(spectral.linearize_polar(m, e1), spectral.linearize_polar(m, e2))
        self.assertLess(abs(other[-1]), 0.9)


if __name__ == '__main__':
    unittest.main()
