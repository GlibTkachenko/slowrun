"""Randomized recovery of the recurrent credit that the tiny record truncates.

In its multi-pass phase the tiny record differentiates only the last recurrent
pass, so the gradient path through the earlier passes, g_path, is dropped.
Priority D draws Z ~ Bernoulli(p) once per optimizer step and, when Z = 1,
differentiates through every pass with the path gradient scaled by 1/p:

    g_hat = g_direct + (Z / p) g_path,   E_Z[g_hat] = g_direct + g_path.

The scale is applied once, to the state entering the last pass: every earlier
path crosses that boundary exactly once. Scaling every boundary would multiply
longer paths by 1/p repeatedly and bias the estimator beyond two passes.

The guarantee is for the raw gradient at fixed weights. Combined with the MG
step it does not carry over: the same draw changes the adaptation gradient, so
the query half is evaluated at a Z-dependent point. D+MG arms are interaction
experiments; the unbiased construction is tested by the D arms without MG.
The scaling is implemented by an identity whose backward pass multiplies the
incoming gradient, so the forward computation is unchanged.
"""

from __future__ import annotations

import random

import torch

# Training modes for the first recurrent pass.
CREDIT_MODES = ('trunc', 'full', 'random', 'random_comp')


def scale_backward(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Returns x unchanged in the forward pass and scales its gradient by `scale`.

    The forward value is bitwise identical to `x` because x - x.detach() is
    exactly zero.

    Args:
        x: Any tensor that requires gradients.
        scale: Multiplier applied to the gradient flowing into `x`.

    Returns:
        A tensor equal to `x`.
    """
    if scale == 1.0:
        return x
    detached = x.detach()
    return detached + (x - detached) * scale


def is_credit_boundary(iteration: int, passes: int) -> bool:
    """Returns whether the state leaving pass `iteration` gets the 1/p path scale.

    Only the state entering the last pass is scaled, so each truncated path is
    compensated exactly once whatever the number of passes.

    Args:
        iteration: Zero-based index of the pass that produced the state.
        passes: Number of recurrent passes in this forward call.

    Returns:
        True for the boundary between the last two passes.
    """
    return iteration == passes - 2


def path_scale(mode: str, p: float) -> float:
    """Returns the gradient multiplier of the first-pass path on Z = 1 steps.

    Args:
        mode: One of CREDIT_MODES.
        p: Probability of a full-credit step for the random modes.

    Returns:
        1/p for compensated random credit and 1 otherwise.

    Raises:
        ValueError: Unknown mode, or p outside (0, 1] for a random mode.
    """
    if mode not in CREDIT_MODES:
        raise ValueError(f'Unknown credit mode {mode!r}; expected one of {CREDIT_MODES}.')
    if mode in ('random', 'random_comp') and not 0.0 < p <= 1.0:
        raise ValueError(f'Credit probability must lie in (0, 1], got {p}.')
    return 1.0 / p if mode == 'random_comp' else 1.0


class CreditSampler:
    """Draws the per-step credit decision, identically on every rank.

    The draws come from a private generator seeded only by the run seed, so all
    ranks agree without communication and the data and dropout streams are
    untouched. This keeps every rank in the same compiled graph on every step.
    """

    def __init__(self, *, mode: str, p: float, seed: int):
        """Initializes the sampler.

        Args:
            mode: One of CREDIT_MODES.
            p: Probability of a full-credit step for the random modes.
            seed: Run seed; the generator is offset from it to stay independent of
                other seeded streams.
        """
        self.mode = mode
        self.p = p
        self.scale = path_scale(mode, p)
        self._rng = random.Random(seed * 1_000_003 + 17)
        self.full_steps = 0
        self.multi_pass_steps = 0

    def draw(self, num_passes: int) -> bool:
        """Returns whether this step differentiates through every recurrent pass.

        Args:
            num_passes: Active number of recurrent passes for the step.

        Returns:
            True for a full-credit step.
        """
        if num_passes < 2 or self.mode == 'trunc':
            return False
        self.multi_pass_steps += 1
        full = self.mode == 'full' or self._rng.random() < self.p
        self.full_steps += int(full)
        return full

    def summary(self) -> dict[str, float | str]:
        """Returns counters for the result file."""
        return {
            'mode': self.mode,
            'p': self.p,
            'scale': self.scale,
            'full_steps': self.full_steps,
            'multi_pass_steps': self.multi_pass_steps,
        }
