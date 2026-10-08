"""Temporary meta-gradient displacements of the plastic matrices.

The record meta-gradient (MG) step splits each data-parallel rank's batch into
an adaptation half A and a query half B. After the A backward pass the plastic
matrices move along the normalized A gradient, the B gradient is accumulated at
the moved point, and the move is undone before the optimizer sees the summed
gradient. This module implements that step for any number of virtual ranks per
process, the per-half objective weights of Priority A and the activation-metric
displacement geometries of Priority B. `VirtualRankState` keeps the state that
physical ranks hold separately (RNG at the start of a step, activation moments)
so that emulated ranks reproduce them.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.distributed as dist

# Multipliers of the auxiliary objective in the (A, B) halves (Priority A).
AUX_SPLITS: dict[str, tuple[float, float]] = {
    'shared': (1.0, 1.0),
    'adapt': (2.0, 0.0),
    'query': (0.0, 2.0),
}

# Displacement rules for the MLP output projections (Priority B).
CPROJ_MODES = ('global', 'euclid_layer', 'act_radius', 'act_metric')

CALIBRATION_FILE = 'mg_calibration.json'


def aux_multipliers(mode: str) -> tuple[float, float]:
    """Returns the auxiliary-loss multipliers of the adaptation and query halves.

    'shared' is the record (lambda in both halves). 'adapt' doubles the auxiliary
    weight in A and removes it from B, which keeps the expected unperturbed
    objective unchanged; 'query' is the reverse ordering control.

    Args:
        mode: One of AUX_SPLITS.

    Returns:
        A pair (m_A, m_B) of multipliers of the record auxiliary weight.

    Raises:
        ValueError: Unknown mode.
    """
    if mode not in AUX_SPLITS:
        raise ValueError(f'Unknown aux split {mode!r}; expected one of {sorted(AUX_SPLITS)}.')
    return AUX_SPLITS[mode]


def split_halves(batches: Sequence[tuple]) -> tuple[list[tuple], list[tuple]]:
    """Splits one virtual rank's microbatches into adaptation and query halves.

    A single microbatch is split along its sequences (tiny record); several
    microbatches are split into the first and second half of the list (one-hour
    record).

    Args:
        batches: The (x, y, *rest) microbatches of one virtual rank, in order.

    Returns:
        The adaptation microbatches and the query microbatches.

    Raises:
        ValueError: An odd number of microbatches above one, or a single
            microbatch with fewer than two sequences.
    """
    if len(batches) == 1:
        x, y, *rest = batches[0]
        if x.size(0) < 2:
            raise ValueError('A single microbatch needs at least two sequences to split.')
        half = x.size(0) // 2
        return [(x[:half], y[:half], *rest)], [(x[half:], y[half:], *rest)]
    if len(batches) % 2:
        raise ValueError(f'Cannot split {len(batches)} microbatches into equal halves.')
    middle = len(batches) // 2
    return list(batches[:middle]), list(batches[middle:])


def select_plastic(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    muon_groups: Sequence[Sequence[torch.nn.Parameter]],
    *,
    rule: str,
    rank: int,
    world_size: int,
) -> tuple[list[str], list[torch.nn.Parameter], list[bool]]:
    """Selects the plastic matrices exactly like the record scripts.

    Args:
        named_parameters: The model's named parameters, in registration order.
        muon_groups: Parameter lists of the optimizer's Muon groups.
        rule: 'matrix_all' for every Muon matrix or 'mlp_all' for MLP matrices.
        rank: This process's rank.
        world_size: Number of processes.

    Returns:
        Names, parameters and ownership flags of the plastic matrices. A rank owns
        a matrix when its Muon shard updates it.

    Raises:
        ValueError: Unknown rule.
    """
    if rule not in ('matrix_all', 'mlp_all'):
        raise ValueError(f'Unknown plastic rule {rule!r}.')
    owned_ids, muon_ids = set(), set()
    for params in muon_groups:
        chunk = (len(params) + world_size - 1) // world_size
        owned_ids.update(id(p) for p in params[rank * chunk:(rank + 1) * chunk])
        muon_ids.update(id(p) for p in params)
    names, params, owned = [], [], []
    for name, p in named_parameters:
        if id(p) in muon_ids and (rule == 'matrix_all' or '.mlp.' in name):
            names.append(name)
            params.append(p)
            owned.append(id(p) in owned_ids)
    return names, params, owned


# =============================================================================
# Input second moments of the MLP output projections.
# =============================================================================


def update_input_moment_(
    moment: torch.Tensor, count: torch.Tensor, h: torch.Tensor, *, stride: int, decay: float
) -> None:
    """Folds a strided subsample of activations into a diagonal second-moment EMA.

    Runs inside the compiled forward pass. The statistics are detached, so they
    never enter autograd.

    Args:
        moment: (features,) float32 EMA of E[h_j^2], updated in place.
        count: Scalar float32 number of updates, incremented in place.
        h: (batch, time, features) activations entering the projection.
        stride: Keep every `stride`-th position along time.
        decay: EMA decay per update.
    """
    sample = h.detach()[:, ::stride].float()
    moment.lerp_(sample.square().mean(dim=(0, 1)), 1.0 - decay)
    count.add_(1.0)


def corrected_moment(moment: torch.Tensor, count: torch.Tensor, decay: float) -> torch.Tensor:
    """Returns the bias-corrected EMA moment (zero before the first update)."""
    return moment / (1.0 - decay**count).clamp_min(1e-12)


class VirtualRankState:
    """State that data-parallel ranks keep separately, for ranks emulated in one process.

    The records seed every physical rank identically, so all ranks start each step
    from the same RNG state and draw the same dropout masks, and each rank keeps
    its own activation moments. `enter(v)` restores the step's starting RNG state
    and loads virtual rank v's moments; `exit(v)` stores the moments back. With one
    virtual rank per process every method is a no-op, so the record path is
    untouched.
    """

    def __init__(self, buffers: Sequence[torch.Tensor], local_vranks: int):
        """Initializes per-rank copies of the buffers.

        Args:
            buffers: Mutable buffers updated by forward passes (activation moments).
            local_vranks: Virtual ranks emulated by this process.
        """
        self.buffers = list(buffers)
        self.local_vranks = local_vranks
        self.active = local_vranks > 1
        self._rng: tuple[torch.Tensor, torch.Tensor | None] | None = None
        self._stored: list[list[torch.Tensor]] = []
        self.reset()

    def reset(self) -> None:
        """Copies the current buffers into every virtual rank's slot."""
        self._stored = (
            [[b.detach().clone() for b in self.buffers] for _ in range(self.local_vranks)]
            if self.active else []
        )

    def tensors(self) -> list[torch.Tensor]:
        """Returns the stored per-rank copies (for exact snapshots)."""
        return [t for slot in self._stored for t in slot]

    def begin_step(self) -> None:
        """Records the RNG state every virtual rank starts the step from."""
        if self.active:
            cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            self._rng = (torch.get_rng_state(), cuda)

    @torch.no_grad()
    def enter(self, vrank: int) -> None:
        """Switches the process to virtual rank `vrank`."""
        if not self.active:
            return
        if self._rng is None:
            raise RuntimeError('enter() called before begin_step().')
        torch.set_rng_state(self._rng[0])
        if self._rng[1] is not None:
            torch.cuda.set_rng_state(self._rng[1])
        for buffer, stored in zip(self.buffers, self._stored[vrank]):
            buffer.copy_(stored)

    @torch.no_grad()
    def exit(self, vrank: int) -> None:
        """Stores virtual rank `vrank`'s buffers after its microbatches."""
        if not self.active:
            return
        for buffer, stored in zip(self.buffers, self._stored[vrank]):
            stored.copy_(buffer)


# =============================================================================
# The displacement.
# =============================================================================


@dataclasses.dataclass(frozen=True)
class LookaheadConfig:
    """Hyperparameters of the temporary displacement.

    Attributes:
        step_norm: Global Frobenius norm eta of the record displacement.
        schedule: 'const', or 'lr' to scale every radius by the LR multiplier.
        cproj: Displacement rule for MLP output projections; one of CPROJ_MODES.
        act_radius: Functional radius rho for the activation modes, measured with
            the undamped metric: tr(D C D^T) = rho^2 in both modes.
        act_damping: Damping added to C before inversion in 'act_metric', relative to
            the mean of C. Large values recover the raw-gradient direction.
        act_decay: Decay of the activation moment EMA (needed for bias correction).
        layer_radii: Per-layer Frobenius radii for 'euclid_layer'.
    """

    step_norm: float = 0.5
    schedule: str = 'const'
    cproj: str = 'global'
    act_radius: float = 0.0
    act_damping: float = 0.1
    act_decay: float = 0.9
    layer_radii: tuple[float, ...] = ()

    def __post_init__(self):
        if self.schedule not in ('const', 'lr'):
            raise ValueError(f'Unknown MG schedule {self.schedule!r}.')
        if self.cproj not in CPROJ_MODES:
            raise ValueError(f'Unknown c_proj mode {self.cproj!r}; expected one of {CPROJ_MODES}.')
        if self.cproj in ('act_radius', 'act_metric') and self.act_radius <= 0:
            raise ValueError(f'{self.cproj} needs a positive act_radius.')


class _CprojMove(NamedTuple):
    """What is needed to undo one c_proj displacement exactly as it was applied."""

    scale: torch.Tensor
    inverse_metric: torch.Tensor | None


class Lookahead:
    """Applies and undoes the MG displacement for a set of plastic matrices.

    Usage per virtual rank, mirroring the record scripts:

        lookahead.stash()                # set aside earlier virtual ranks' grads
        <backward on the A half>
        lookahead.displace(lr_multiplier)
        <backward on the B half>
        lookahead.restore()              # undo the move, merge all gradients

    With one virtual rank per process this is numerically identical to the record:
    the A gradients are not cloned but detached from `.grad`, so the B pass writes
    fresh gradients and the merge performs the same single addition autograd would.
    """

    def __init__(
        self,
        names: Sequence[str],
        params: Sequence[torch.nn.Parameter],
        *,
        config: LookaheadConfig,
        moments: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        """Initializes the displacement.

        Args:
            names: Names of the plastic matrices.
            params: The plastic matrices.
            config: Displacement hyperparameters.
            moments: Map from c_proj weight name to its (moment, count) buffers. Needed
                by the activation modes; when given in 'global' mode, only feeds the
                calibration statistics.

        Raises:
            ValueError: Missing moments or radii for the requested mode.
        """
        self.config = config
        self.names = list(names)
        self.params = list(params)
        self.moments = moments or {}
        # Matrices with input moments are tracked for calibration in every mode.
        self._tracked = [i for i, n in enumerate(self.names) if n in self.moments]
        if config.cproj == 'global':
            self._cproj = []
        elif config.cproj == 'euclid_layer':
            self._cproj = [i for i, n in enumerate(self.names) if n.endswith('mlp.c_proj.weight')]
            if len(config.layer_radii) != len(self._cproj):
                raise ValueError(
                    f'euclid_layer needs {len(self._cproj)} layer radii, '
                    f'got {len(config.layer_radii)}.'
                )
        else:
            if not self._tracked:
                raise ValueError('Activation modes need input moments of the c_proj matrices.')
            self._cproj = list(self._tracked)
        managed = set(self._cproj)
        self._others = [i for i in range(len(self.params)) if i not in managed]
        self._stash: list[torch.Tensor | None] | None = None
        self._grads: list[torch.Tensor] | None = None
        self._alpha = 0.0
        self._moves: list[_CprojMove] = []
        self._steps = 0
        self._sum_grad_norm = 0.0
        self._sum_eta = 0.0
        device = self.params[0].device if self.params else torch.device('cpu')
        self._sum_act = torch.zeros(len(self._tracked), device=device)
        self._sum_frob = torch.zeros(len(self._tracked), device=device)

    @property
    def cproj_names(self) -> list[str]:
        """Names of the matrices whose functional displacement is tracked."""
        return [self.names[i] for i in self._tracked]

    def stash(self) -> None:
        """Sets aside gradients accumulated by earlier virtual ranks on this process."""
        if any(p.grad is not None for p in self.params):
            self._stash = [p.grad for p in self.params]
            for p in self.params:
                p.grad = None
        else:
            self._stash = None

    def _metric(self, name: str) -> torch.Tensor:
        """Returns the corrected input moment of one c_proj matrix."""
        moment, count = self.moments[name]
        return corrected_moment(moment, count, self.config.act_decay)

    @torch.no_grad()
    def displace(self, lr_multiplier: float, *, record: bool = True) -> None:
        """Moves the plastic matrices along their current (adaptation) gradients.

        Args:
            lr_multiplier: The base LR multiplier of this step, used by the 'lr'
                schedule.
            record: Whether the move enters the logged statistics. Diagnostics
                pass False so that they leave no trace in the run's metrics.
        """
        cfg = self.config
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))).clamp_min(1e-12)
        schedule = lr_multiplier if cfg.schedule == 'lr' else 1.0
        eta = cfg.step_norm * schedule
        norm_value = float(norm)
        self._alpha = -eta / norm_value
        if self._others:
            torch._foreach_add_(
                [self.params[i] for i in self._others], [grads[i] for i in self._others],
                alpha=self._alpha,
            )
        self._moves = []
        for slot, i in enumerate(self._cproj):
            move = self._cproj_move(slot, i, grads[i], schedule)
            self._moves.append(move)
            self.params[i].add_(self._apply_move(grads[i], move))
        if record:
            self._track(grads, schedule)
            self._steps += 1
            self._sum_grad_norm += norm_value
            self._sum_eta += eta
        self._grads = grads
        for p in self.params:
            p.grad = None

    def abandon(self) -> None:
        """Forgets the pending displacement without touching parameters or gradients.

        Used by diagnostics, which restore parameters from an exact snapshot.
        """
        self._grads, self._stash, self._moves = None, None, []

    def _cproj_move(self, slot: int, index: int, grad: torch.Tensor, schedule: float) -> _CprojMove:
        """Computes the scale (and metric) of one c_proj displacement."""
        cfg = self.config
        g = grad.float()
        if cfg.cproj == 'euclid_layer':
            radius = cfg.layer_radii[slot] * schedule
            return _CprojMove(scale=-radius / g.norm().clamp_min(1e-12), inverse_metric=None)
        # D = -rho * G W / sqrt(tr(G W C W G^T)), with W = I (radius only) or the damped
        # inverse metric (metric direction); either way tr(D C D^T) = rho^2 exactly.
        c = self._metric(self.names[index])
        inverse = None
        if cfg.cproj == 'act_metric':
            inverse = 1.0 / (c + cfg.act_damping * c.mean()).clamp_min(1e-30)
            size = (g.square() * inverse.square() * c).sum().sqrt()
        else:
            size = (g.square() * c).sum().sqrt()
        return _CprojMove(scale=-cfg.act_radius * schedule / size.clamp_min(1e-12),
                          inverse_metric=inverse)

    @staticmethod
    def _apply_move(grad: torch.Tensor, move: _CprojMove) -> torch.Tensor:
        """Returns the displacement D of one c_proj matrix (deterministic, re-computable)."""
        direction = grad if move.inverse_metric is None else grad * move.inverse_metric
        return direction * move.scale.to(grad.dtype)

    def _track(self, grads: Sequence[torch.Tensor], schedule: float) -> None:
        """Accumulates functional and Frobenius displacement of the tracked matrices."""
        if not self._tracked:
            return
        managed = {i: m for i, m in zip(self._cproj, self._moves)}
        act, frob = [], []
        for i in self._tracked:
            if i in managed:
                d = self._apply_move(grads[i], managed[i]).float()
            else:
                d = grads[i].float() * self._alpha
            act.append((d.square() * self._metric(self.names[i])).sum().sqrt())
            frob.append(d.norm())
        scale = 1.0 / max(schedule, 1e-12)
        self._sum_act += torch.stack(act) * scale
        self._sum_frob += torch.stack(frob) * scale

    @torch.no_grad()
    def restore(self) -> None:
        """Undoes the displacement and merges adaptation, query and stashed gradients."""
        grads = self._grads
        if grads is None:
            raise RuntimeError('restore() called without a preceding displace().')
        if self._others:
            torch._foreach_add_(
                [self.params[i] for i in self._others], [grads[i] for i in self._others],
                alpha=-self._alpha,
            )
        for i, move in zip(self._cproj, self._moves):
            self.params[i].sub_(self._apply_move(grads[i], move))
        for i, p in enumerate(self.params):
            merged = grads[i] if p.grad is None else grads[i].add_(p.grad)
            if self._stash is not None and self._stash[i] is not None:
                merged = merged.add_(self._stash[i])
            p.grad = merged
        self._grads, self._stash, self._moves = None, None, []

    def _sums(self, reduce: bool) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:
        """Returns (steps, grad-norm sum, eta sum, act sums, frob sums), optionally over all ranks."""
        packed = torch.cat([
            torch.tensor([self._steps, self._sum_grad_norm, self._sum_eta], dtype=torch.float64,
                         device=self._sum_act.device),
            self._sum_act.double(),
            self._sum_frob.double(),
        ])
        if reduce and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(packed)
        count = len(self._tracked)
        steps, grad_norm, eta = packed[:3].tolist()
        return steps, grad_norm, eta, packed[3:3 + count], packed[3 + count:]

    def statistics(self, *, reduce: bool = False) -> dict[str, float]:
        """Returns running means of the logged quantities (synchronizes the device).

        Args:
            reduce: Average over all ranks. Every rank must call it then.
        """
        steps, grad_norm, eta, act, frob = self._sums(reduce)
        if steps == 0:
            return {}
        stats = {'mg/grad_norm': grad_norm / steps, 'mg/eta': eta / steps}
        if self._tracked:
            stats['mg/cproj_act_disp'] = float(act.mean()) / steps
            stats['mg/cproj_frob_disp'] = float(frob.mean()) / steps
        return stats

    def calibration(self, *, reduce: bool = True) -> dict:
        """Returns per-layer mean displacements over all ranks, normalized by the schedule.

        The 'act_disp_mean' entry calibrates the functional radius of the activation
        arms; 'frob_disp' provides the matched radii of the 'euclid_layer' control.

        Args:
            reduce: Average over all ranks. Every rank must call it then.
        """
        steps, _, _, act, frob = self._sums(reduce)
        if steps == 0 or not self._tracked:
            return {}
        act_means = (act / steps).tolist()
        return {
            'layers': self.cproj_names,
            'act_disp': act_means,
            'frob_disp': (frob / steps).tolist(),
            'act_disp_mean': sum(act_means) / len(act_means),
            'steps': int(steps),
            'mode': self.config.cproj,
        }


def load_calibration(path: str) -> dict:
    """Reads a calibration file written by a run with input moments enabled.

    Args:
        path: Path of an mg_calibration.json file.

    Returns:
        The calibration dictionary.

    Raises:
        ValueError: The file holds no calibration data.
    """
    with open(path) as f:
        data = json.load(f)
    if not data.get('act_disp'):
        raise ValueError(f'{path} holds no calibration data.')
    return data


def functional_radius(calibration: dict, multiplier: float) -> float:
    """Returns rho = multiplier * the baseline's mean functional c_proj displacement."""
    radius = calibration['act_disp_mean'] * multiplier
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f'Calibration produced an invalid radius {radius}.')
    return radius


def matched_radii(calibration: dict, names: Sequence[str]) -> tuple[float, ...]:
    """Returns per-layer Frobenius radii from a calibration, ordered like `names`.

    Args:
        calibration: Calibration of the arm whose displacement sizes are matched.
        names: c_proj matrix names of the current run, in plastic order.

    Returns:
        One radius per name.

    Raises:
        ValueError: A layer of the current run is missing from the calibration.
    """
    by_name = dict(zip(calibration['layers'], calibration['frob_disp']))
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f'Calibration lacks radii for {missing[:3]}...')
    return tuple(float(by_name[n]) for n in names)
