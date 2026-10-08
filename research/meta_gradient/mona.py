"""MONA's gradient-difference correction with a single state buffer.

Algorithm 1 of MONA (Li et al., 2026) keeps the previous gradient and an EMA of
gradient differences:

    A_t = beta A_{t-1} + (1 - beta)(G_t - G_{t-1}),   G~_t = G_t + alpha A_t,

with A_0 = G_0 = 0. By induction A_t = (1 - beta)(G_t - m_{t-1}), where
m_t = beta m_{t-1} + (1 - beta) G_t is a plain gradient EMA with m_0 = 0. So

    G~_t = G_t + alpha (1 - beta)(G_t - m_{t-1}),

which needs only m. With the paper's default alpha = -1/(2(1 - beta)) this is
(G_t + m_{t-1}) / 2: a 50/50 mix of the current gradient and the EMA of the
preceding ones. Only the correction is applied; the record's Nesterov momentum
and NorMuon tail stay in place, so this is MONA's correction on the record
optimizer, not a line-by-line copy of the paper's optimizer.
"""

from __future__ import annotations

import torch


def default_alpha(beta: float) -> float:
    """Returns the paper's coupling alpha = -1/(2(1 - beta))."""
    return -1.0 / (2.0 * (1.0 - beta))


def mona_correct_(
    grads: torch.Tensor, ema: torch.Tensor, beta_t: torch.Tensor, alpha_t: torch.Tensor
) -> None:
    """Applies the correction to `grads` and advances the gradient EMA, in place.

    Args:
        grads: Gradients G_t; replaced by G~_t.
        ema: The EMA m_{t-1}; replaced by m_t (computed from the raw G_t).
        beta_t: Scalar tensor beta.
        alpha_t: Scalar tensor alpha.
    """
    beta = beta_t.to(grads.dtype)
    alpha = alpha_t.to(grads.dtype)
    correction = (grads - ema) * (alpha * (1 - beta))
    ema.lerp_(grads, 1 - beta)
    grads.add_(correction)
