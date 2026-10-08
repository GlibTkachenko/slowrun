"""Spectral response maps for Muon-style orthogonalized updates.

The record optimizers send every momentum matrix through Polar Express, a fixed
polynomial approximation of the polar factor U V^T. The update-map experiments
(Priority C) study the wider family

    T_c(X) = (X X^T + eps^2 I)^(-c) X,

whose singular-value response is s / (s^2 + eps^2)^c. At c = 1/2 the map is a
regularized polar factor; above 1/2 it emphasizes smaller resolved directions.
This module provides the production maps, a float64 reference, the Fréchet
derivative of the polar factor and the perturbation decomposition used by the
meta-gradient diagnostic.

Throughout, `eps` is relative to the RMS singular value ||X||_F / sqrt(min(m, n)),
which keeps every map invariant to the scale of its input.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from typing import NamedTuple

import torch

# The record's Polar Express coefficients, one (a, b, c) triple per iteration.
POLAR_EXPRESS_COEFFS: tuple[tuple[float, float, float], ...] = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)

# Matrix maps selectable from the training scripts.
SPECTRAL_METHODS = ('polar_express', 'eigh', 'augmented_polar')

# How alternative maps set the size of their output. 'polar_express' matches, per
# matrix, the Frobenius norm the record's map would have produced on the same input,
# so only the direction of the update changes; 'ideal' uses sqrt(min(m, n)).
NORM_MATCHING = ('polar_express', 'ideal')


def polar_express(x: torch.Tensor, *, steps: int = 5) -> torch.Tensor:
    """Returns the record's Polar Express approximation of the polar factor.

    Mirrors the record optimizers exactly: bfloat16 arithmetic, Frobenius
    normalization with a 1.02 safety factor, then quintic iterations.

    Args:
        x: A matrix, or a batch of matrices in the last two dimensions.
        steps: Number of polynomial iterations to apply.

    Returns:
        A bfloat16 tensor with the shape of `x`.
    """
    y = x.bfloat16()
    y = y / (y.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if y.size(-2) > y.size(-1):
        for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
            gram = y.mT @ y
            y = a * y + y @ (b * gram + c * (gram @ gram))
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
            gram = y @ y.mT
            y = a * y + (b * gram + c * (gram @ gram)) @ y
    return y


def _absolute_eps(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Returns `eps` times the RMS singular value of each matrix, shaped (..., 1, 1)."""
    rank = min(x.size(-2), x.size(-1))
    return eps * x.norm(dim=(-2, -1), keepdim=True) / math.sqrt(rank)


def _match_polar_rms(y: torch.Tensor) -> torch.Tensor:
    """Rescales each matrix to the Frobenius norm sqrt(min(m, n)) of an exact polar factor."""
    rank = min(y.size(-2), y.size(-1))
    return y * (math.sqrt(rank) / y.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30))


def response_curve(s: torch.Tensor, *, exponent: float, eps: float) -> torch.Tensor:
    """Returns the scalar response s / (s^2 + eps^2)^c for absolute `eps`.

    Args:
        s: Singular values.
        exponent: The power c.
        eps: Absolute regularization.

    Returns:
        The response of T_c at each singular value.
    """
    return s * (s.square() + eps**2).pow(-exponent)


def spectral_response(
    x: torch.Tensor,
    *,
    exponent: float,
    eps: float,
    match_rms: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Applies T_c(X) = (X X^T + eps^2 I)^(-c) X exactly to the last two dimensions.

    The map is computed from an eigendecomposition of the smaller Gram matrix.
    Squaring the singular values limits float32 to directions above roughly 3e-4
    of the largest singular value, which is why `eps` must be positive: directions
    below the regularization scale are damped rather than resolved. Use float64
    for eps well below 1e-2.

    Args:
        x: A matrix, or a batch of matrices in the last two dimensions.
        exponent: The power c. 0.5 gives a regularized polar factor.
        eps: Regularization relative to the RMS singular value of each matrix.
        match_rms: If set, rescales every output to the Frobenius norm of an exact
            polar factor so that changing the response does not change the step size.
        dtype: Precision of the eigendecomposition and the result.

    Returns:
        A tensor of `dtype` with the shape of `x`.

    Raises:
        ValueError: `eps` is not positive.
    """
    if eps <= 0:
        raise ValueError(f'eps must be positive for the eigendecomposition path, got {eps}.')
    xf = x.to(dtype)
    tall = xf.size(-2) > xf.size(-1)
    gram = xf.mT @ xf if tall else xf @ xf.mT
    evals, evecs = torch.linalg.eigh(gram)
    eps_sq = _absolute_eps(xf, eps).square().squeeze(-1)
    weights = (evals.clamp_min(0) + eps_sq).clamp_min(1e-30).pow(-exponent)
    gram_filter = (evecs * weights.unsqueeze(-2)) @ evecs.mT
    y = xf @ gram_filter if tall else gram_filter @ xf
    return _match_polar_rms(y) if match_rms else y


def augmented_polar(
    x: torch.Tensor, *, eps: float, steps: int = 5, match_rms: bool = True
) -> torch.Tensor:
    """Returns the regularized polar factor (c = 1/2) using only Polar Express.

    For a tall X the polar factor of the stacked matrix N = [X; eps I] has top
    block X (X^T X + eps^2 I)^(-1/2) = T_{1/2}(X). Every singular value of N is at
    least eps, so the record's iteration converges as well as it does on X, at
    roughly (m + n) / m times the cost.

    Args:
        x: A matrix, or a batch of matrices in the last two dimensions.
        eps: Regularization relative to the RMS singular value of each matrix.
        steps: Number of Polar Express iterations.
        match_rms: If set, rescales every output to the Frobenius norm of an exact
            polar factor.

    Returns:
        A float32 tensor with the shape of `x`.
    """
    transpose = x.size(-2) < x.size(-1)
    tall = (x.mT if transpose else x).float()
    rows, cols = tall.shape[-2:]
    eye = torch.eye(cols, device=tall.device, dtype=tall.dtype).expand(*tall.shape[:-2], cols, cols)
    stacked = torch.cat([tall, eye * _absolute_eps(tall, eps)], dim=-2)
    y = polar_express(stacked, steps=steps)[..., :rows, :].float()
    y = y.mT if transpose else y
    return _match_polar_rms(y) if match_rms else y


def reference_response(
    x: torch.Tensor, *, exponent: float, eps: float, match_rms: bool = True
) -> torch.Tensor:
    """Returns T_c(X) from a float64 singular value decomposition.

    This is the accuracy reference for `spectral_response`, `augmented_polar` and
    `polar_express`; it is too slow for training.

    Args:
        x: A matrix, or a batch of matrices in the last two dimensions.
        exponent: The power c.
        eps: Regularization relative to the RMS singular value of each matrix.
        match_rms: If set, rescales every output to the Frobenius norm of an exact
            polar factor.

    Returns:
        A float64 tensor with the shape of `x`.
    """
    x64 = x.double()
    u, s, vh = torch.linalg.svd(x64, full_matrices=False)
    eps_abs = _absolute_eps(x64, eps).squeeze(-1)
    response = s * (s.square() + eps_abs.square()).pow(-exponent)
    y = (u * response.unsqueeze(-2)) @ vh
    return _match_polar_rms(y) if match_rms else y


def apply_map(
    x: torch.Tensor,
    *,
    method: str,
    exponent: float = 0.5,
    eps: float = 0.0,
    dtype: torch.dtype = torch.float32,
    norm: str = 'polar_express',
) -> torch.Tensor:
    """Applies one of the selectable update maps to a batch of preprocessed momenta.

    Args:
        x: Batch of matrices.
        method: One of SPECTRAL_METHODS.
        exponent: Power c for 'eigh'.
        eps: Relative regularization for 'eigh' and 'augmented_polar'.
        dtype: Precision of the 'eigh' path.
        norm: One of NORM_MATCHING, for the alternative maps.

    Returns:
        The mapped batch. 'polar_express' returns bfloat16, the others float32 or
        `dtype`.

    Raises:
        ValueError: Unknown method or norm, or an exponent other than 1/2 for
            'augmented_polar'.
    """
    if method == 'polar_express':
        return polar_express(x)
    if norm not in NORM_MATCHING:
        raise ValueError(f'Unknown norm matching {norm!r}; expected one of {NORM_MATCHING}.')
    if method == 'eigh':
        y = spectral_response(x, exponent=exponent, eps=eps, dtype=dtype, match_rms=False)
    elif method == 'augmented_polar':
        if abs(exponent - 0.5) > 1e-12:
            raise ValueError('augmented_polar only implements exponent 0.5.')
        y = augmented_polar(x, eps=eps, match_rms=False)
    else:
        raise ValueError(f'Unknown spectral method {method!r}; expected one of {SPECTRAL_METHODS}.')
    if norm == 'ideal':
        return _match_polar_rms(y)
    target = polar_express(x).float().norm(dim=(-2, -1), keepdim=True)
    return y * (target / y.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)).to(y.dtype)


# =============================================================================
# Linearized polar map, used by the meta-gradient diagnostic.
# =============================================================================


def _band_index(count: int, bands: int, device: torch.device) -> torch.Tensor:
    """Assigns singular indices (sorted descending) to `bands` contiguous groups."""
    return torch.div(torch.arange(count, device=device) * bands, count, rounding_mode='floor')


class PolarLinearization(NamedTuple):
    """First-order action of the ideal polar map on one perturbation.

    Attributes:
        rotation: The n x n matrix Ω = (Ê - Ê^T) / (s_i + s_j) in the singular basis.
        complement: The m x n matrix (I - U U^T) E V diag(1/s) for tall inputs.
        symmetric_fraction: Share of ||E||^2 the polar map discards to first order.
        antisymmetric_fraction: Share of ||E||^2 that rotates singular vectors.
        complement_fraction: Share of ||E||^2 outside the column space.
        gain: ||D Q[E]||_F / ||E||_F with the input scaled to unit spectral norm.
        band_energy: Share of ||D Q[E]||^2 per singular band, largest band first.
    """

    rotation: torch.Tensor
    complement: torch.Tensor
    symmetric_fraction: float
    antisymmetric_fraction: float
    complement_fraction: float
    gain: float
    band_energy: tuple[float, ...]


def linearize_polar(
    m: torch.Tensor, e: torch.Tensor, *, bands: int = 3, rcond: float = 1e-6
) -> PolarLinearization:
    """Decomposes a perturbation by its first-order effect on the polar factor.

    For m = U diag(s) V^T with full column rank the Fréchet derivative is

        D Q[E] = U Ω V^T + (I - U U^T) E V diag(1/s) V^T,
        Ω_ij = (Ê_ij - Ê_ji) / (s_i + s_j),   Ê = U^T E V.

    The symmetric part of Ê, including every singular-value change, is
    discarded; the antisymmetric part and the complement survive, amplified near
    small singular values. Wide matrices use Q(m^T) = Q(m)^T. Computation is in
    float64 on a single matrix.

    Args:
        m: The matrix whose polar factor is perturbed.
        e: The perturbation, with the shape of `m`.
        bands: Number of contiguous singular-value bands for `band_energy`.
        rcond: Singular values below rcond * s_max are clamped before inversion.

    Returns:
        The linearization of the polar map at `m` along `e`.

    Raises:
        ValueError: The inputs are not matrices of equal shape.
    """
    if m.ndim != 2 or m.shape != e.shape:
        raise ValueError(f'Expected two matrices of equal shape, got {m.shape} and {e.shape}.')
    if m.size(0) < m.size(1):
        m, e = m.mT, e.mT
    u, s, vh = torch.linalg.svd(m.double(), full_matrices=False)
    # Work at unit spectral norm so that `gain` is comparable across matrices.
    scale = s[0].clamp_min(1e-300)
    s = (s / scale).clamp_min(rcond)
    e64 = e.double() / scale
    v = vh.mT
    e_hat = u.mT @ e64 @ v
    antisym = 0.5 * (e_hat - e_hat.mT)
    rotation = 2 * antisym / (s[:, None] + s[None, :])
    outside = e64 - u @ (u.mT @ e64)
    complement = (outside @ v) / s[None, :]

    total = e64.square().sum().clamp_min(1e-300)
    symmetric_energy = (e_hat - antisym).square().sum()
    band = _band_index(s.numel(), bands, s.device)
    pair_band = torch.maximum(band[:, None], band[None, :])
    derivative_energy = rotation.square().sum() + complement.square().sum()
    band_energy = []
    for b in range(bands):
        energy = rotation.square()[pair_band == b].sum() + complement.square()[:, band == b].sum()
        band_energy.append(float(energy / derivative_energy.clamp_min(1e-300)))
    return PolarLinearization(
        rotation=rotation,
        complement=complement,
        symmetric_fraction=float(symmetric_energy / total),
        antisymmetric_fraction=float(antisym.square().sum() / total),
        complement_fraction=float(outside.square().sum() / total),
        gain=float(derivative_energy.sqrt() / total.sqrt()),
        band_energy=tuple(band_energy),
    )


def polar_derivative(m: torch.Tensor, e: torch.Tensor, *, rcond: float = 1e-6) -> torch.Tensor:
    """Returns the Fréchet derivative D Q[E] of the polar factor at m.

    Args:
        m: The matrix whose polar factor is differentiated.
        e: The direction of differentiation.
        rcond: Singular values below rcond * s_max are clamped before inversion.

    Returns:
        A float64 matrix with the shape of `m`.
    """
    wide = m.size(0) < m.size(1)
    mt, et = (m.mT, e.mT) if wide else (m, e)
    m64 = mt.double()
    u, s, vh = torch.linalg.svd(m64, full_matrices=False)
    s = s.clamp_min(rcond * s[0])
    e_hat = u.mT @ et.double() @ vh.mT
    rotation = (e_hat - e_hat.mT) / (s[:, None] + s[None, :])
    outside = et.double() - u @ (u.mT @ et.double())
    derivative = u @ rotation @ vh + ((outside @ vh.mT) / s[None, :]) @ vh
    return derivative.mT if wide else derivative


def band_cosines(
    first: PolarLinearization, second: PolarLinearization, *, bands: int = 3
) -> tuple[float, ...]:
    """Returns per-band cosine similarities of two linearizations at the same matrix.

    Repeatable corrections give large cosines across independent batch pairs;
    noise gives cosines near zero. Bands follow `linearize_polar`.

    Args:
        first: Linearization along one perturbation.
        second: Linearization along an independent perturbation.
        bands: Number of bands; must match the linearizations.

    Returns:
        One cosine per band, largest singular values first, then the overall cosine.
    """
    count = first.rotation.size(0)
    band = _band_index(count, bands, first.rotation.device)
    pair_band = torch.maximum(band[:, None], band[None, :])

    def _cosine(mask_rot: torch.Tensor, mask_col: torch.Tensor) -> float:
        r1, r2 = first.rotation * mask_rot, second.rotation * mask_rot
        c1, c2 = first.complement * mask_col, second.complement * mask_col
        dot = (r1 * r2).sum() + (c1 * c2).sum()
        norm = ((r1.square().sum() + c1.square().sum()) * (r2.square().sum() + c2.square().sum())).sqrt()
        return float(dot / norm.clamp_min(1e-300))

    result = [_cosine(pair_band == b, (band == b)[None, :]) for b in range(bands)]
    ones = torch.ones_like(first.rotation, dtype=torch.bool)
    result.append(_cosine(ones, torch.ones(1, count, dtype=torch.bool, device=ones.device)))
    return tuple(result)


# =============================================================================
# Accuracy and cost measurements for the production paths.
# =============================================================================


def synthetic_matrices(
    singular_values: torch.Tensor, *, rows: int, cols: int, seed: int = 0
) -> torch.Tensor:
    """Returns float64 matrices with prescribed singular values and random bases.

    Args:
        singular_values: (batch, min(rows, cols)) singular values.
        rows: Number of rows of each matrix.
        cols: Number of columns of each matrix.
        seed: Seed for the random orthogonal factors.

    Returns:
        A (batch, rows, cols) float64 tensor.
    """
    gen = torch.Generator().manual_seed(seed)
    batch, rank = singular_values.shape
    left, _ = torch.linalg.qr(torch.randn(batch, rows, rank, generator=gen, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(batch, cols, rank, generator=gen, dtype=torch.float64))
    return (left * singular_values.double().unsqueeze(-2)) @ right.mT


def relative_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """Returns ||approx - exact||_F / ||exact||_F for each matrix in a batch."""
    diff = (approx.double() - exact.double()).norm(dim=(-2, -1))
    return diff / exact.double().norm(dim=(-2, -1)).clamp_min(1e-300)


def _benchmark(fn, x: torch.Tensor, repeats: int) -> float:
    """Returns the median wall time of fn(x) in milliseconds."""
    fn(x)
    times = []
    for _ in range(repeats):
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
    return sorted(times)[len(times) // 2]


def run_benchmark(
    shapes: Sequence[tuple[int, int, int]],
    *,
    exponent: float,
    eps: float,
    device: str,
    repeats: int = 5,
    spectra: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    """Measures error and time of the production maps against the float64 reference.

    Args:
        shapes: (batch, rows, cols) triples; the record shapes per rank are the
            defaults of this module's command line.
        exponent: Power c for the 'eigh' path; the reference uses the same value.
        eps: Relative regularization for every path.
        device: Device on which the production maps run.
        repeats: Timing repetitions.
        spectra: Optional recorded (count, rank) singular values to impose on the
            test matrices instead of log-uniform ones.

    Returns:
        One row per shape with relative errors and median times.
    """
    rows_out = []
    for batch, rows, cols in shapes:
        rank = min(rows, cols)
        if spectra is not None and spectra.size(-1) >= rank:
            values = spectra[:batch, :rank]
            batch = values.size(0)
        else:
            gen = torch.Generator().manual_seed(rows * 31 + cols)
            values = torch.exp(torch.empty(batch, rank).uniform_(math.log(1e-3), 0.0, generator=gen))
        x64 = synthetic_matrices(values, rows=rows, cols=cols)
        x = x64.float().to(device)
        exact_reg = reference_response(x64, exponent=exponent, eps=eps)
        exact_half = reference_response(x64, exponent=0.5, eps=eps)
        exact_polar = reference_response(x64, exponent=0.5, eps=1e-9)
        pe = polar_express(x)
        row = {
            'batch': batch, 'rows': rows, 'cols': cols,
            'err_polar_express_vs_polar': relative_error(_match_polar_rms(pe.float()).cpu(), exact_polar).mean().item(),
            'err_eigh': relative_error(spectral_response(x, exponent=exponent, eps=eps).cpu(), exact_reg).mean().item(),
            'err_augmented_polar': relative_error(augmented_polar(x, eps=eps).cpu(), exact_half).mean().item(),
            'ms_polar_express': _benchmark(polar_express, x, repeats),
            'ms_eigh': _benchmark(lambda t: spectral_response(t, exponent=exponent, eps=eps), x, repeats),
            'ms_augmented_polar': _benchmark(lambda t: augmented_polar(t, eps=eps), x, repeats),
        }
        rows_out.append(row)
    return rows_out


# Matrices owned by one of eight ranks in each record (batch, rows, cols), rounded up.
TINY_SHAPES = ((8, 1024, 1024), (6, 2816, 1024), (2, 1024, 2816))
HOUR_SHAPES = ((16, 1792, 1792), (8, 4864, 1792), (4, 1792, 4864))


def main() -> None:
    """Prints accuracy and timing of the update maps for the record shapes."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--track', choices=('tiny', 'hour'), default='tiny')
    parser.add_argument('--exponent', type=float, default=2 / 3)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--spectra', default=None,
                        help='Optional .pt file of recorded singular values from a diagnostic run.')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    spectra = None
    if args.spectra:
        recorded = torch.load(args.spectra, weights_only=True)
        spectra = torch.stack([v for v in recorded.values()]) if isinstance(recorded, dict) else recorded
    shapes = TINY_SHAPES if args.track == 'tiny' else HOUR_SHAPES
    rows = run_benchmark(shapes, exponent=args.exponent, eps=args.eps, device=args.device,
                         repeats=args.repeats, spectra=spectra)
    header = list(rows[0])
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for row in rows:
        print('| ' + ' | '.join(f'{row[k]:.4g}' if isinstance(row[k], float) else str(row[k])
                                for k in header) + ' |')


if __name__ == '__main__':
    main()
