"""Exact training-state snapshots and the meta-gradient survival diagnostic.

Priority C asks which components of the meta-gradient (MG) correction survive
Muon's update map, and whether they improve clean next-token prediction. At a
few chosen steps `mg_survival`:

1. computes, for every virtual rank of the process, the adaptation gradient g_A
   and the query gradient at the original weights g_B(θ) and at the displaced
   weights g_B(θ - ηd) with identical data and dropout masks, so that
   E = g_B(θ - ηd) - g_B(θ) isolates the displacement and the mixes below are
   exactly the gradients of the training step;
2. runs the real optimizer from the same state on the baseline mix G0 = g_A + g_B(θ)
   and on the MG mix G1 = g_A + g_B(θ - ηd), measuring how the update changes and
   how a fixed training probe responds to each virtual update;
3. decomposes the induced change of the Muon input of a few representative
   matrices in their singular basis, and compares it with the change induced by
   an independent batch pair;
4. restores parameters, optimizer state, RNG streams and mutable buffers exactly,
   so a diagnostic run follows the same trajectory as a plain run.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist

import spectral
from lookahead import Lookahead

Batches = Sequence[tuple]


@dataclasses.dataclass
class Snapshot:
    """A restorable copy of everything a training step mutates.

    Attributes:
        params: Copies of the parameters, in the order given to `take_snapshot`.
        state: Optimizer state per parameter index; tensors are copied.
        buffers: Copies of extra mutable buffers (e.g. activation moments).
        cpu_rng: CPU generator state.
        cuda_rng: Generator state of the current CUDA device, if any.
    """

    params: list[torch.Tensor]
    state: dict[int, dict[str, Any]]
    buffers: list[torch.Tensor]
    cpu_rng: torch.Tensor
    cuda_rng: torch.Tensor | None


def _copy_value(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device, copy=True)
    return copy.deepcopy(value)


def take_snapshot(
    params: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    buffers: Sequence[torch.Tensor] = (),
    *,
    device: torch.device | str = 'cpu',
) -> Snapshot:
    """Copies parameters, optimizer state, mutable buffers and RNG states.

    Args:
        params: Every parameter the optimizer may touch.
        optimizer: The optimizer whose `state` is copied.
        buffers: Additional tensors mutated by forward passes.
        device: Where the copies live; the host keeps device memory free.

    Returns:
        The snapshot.
    """
    index = {id(p): i for i, p in enumerate(params)}
    state = {
        index[id(p)]: {k: _copy_value(v, device) for k, v in s.items()}
        for p, s in optimizer.state.items()
        if id(p) in index
    }
    return Snapshot(
        params=[p.detach().to(device, copy=True) for p in params],
        state=state,
        buffers=[b.detach().to(device, copy=True) for b in buffers],
        cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    )


@torch.no_grad()
def restore_snapshot(
    snapshot: Snapshot,
    params: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    buffers: Sequence[torch.Tensor] = (),
) -> None:
    """Restores a snapshot in place, keeping tensor identities where possible.

    State entries created after the snapshot (for example by a first optimizer
    step) are removed.

    Args:
        snapshot: The snapshot to restore.
        params: The parameters given to `take_snapshot`, in the same order.
        optimizer: The optimizer given to `take_snapshot`.
        buffers: The buffers given to `take_snapshot`, in the same order.
    """
    for p, saved in zip(params, snapshot.params):
        p.copy_(saved)
    for i, p in enumerate(params):
        saved = snapshot.state.get(i)
        if saved is None:
            optimizer.state.pop(p, None)
            continue
        live = optimizer.state[p]
        for key in [k for k in live if k not in saved]:
            del live[key]
        for key, value in saved.items():
            current = live.get(key)
            if torch.is_tensor(value) and torch.is_tensor(current) and current.shape == value.shape:
                current.copy_(value)
            else:
                live[key] = _copy_value(value, p.device if torch.is_tensor(value) else 'cpu')
    for b, saved in zip(buffers, snapshot.buffers):
        b.copy_(saved)
    torch.set_rng_state(snapshot.cpu_rng)
    if snapshot.cuda_rng is not None:
        torch.cuda.set_rng_state(snapshot.cuda_rng)


def snapshot_digest(params: Sequence[torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Returns a cheap order-sensitive checksum of parameters and optimizer state.

    Used by tests and smoke runs to assert that a diagnostic left no trace.
    """
    total = 0.0
    for i, p in enumerate(params):
        total += (i + 1) * float(p.detach().double().sum())
        for value in optimizer.state.get(p, {}).values():
            if torch.is_tensor(value) and value.is_floating_point():
                total += (i + 1) * 1e-3 * float(value.detach().double().sum())
    return total


# =============================================================================
# The diagnostic.
# =============================================================================


def parameter_role(name: str) -> str:
    """Maps a parameter name to a coarse role such as 'attn.c_q' or 'mlp.c_proj'."""
    match = re.search(r'(attn|mlp)\.(\w+)\.weight$', name)
    if match:
        prefix = 'mtp.' if name.startswith('mtp_') else ''
        return f'{prefix}{match.group(1)}.{match.group(2)}'
    for role in ('ve_projs', 'lm_head', 'wte', 'ngram', 'mtp_proj'):
        if role in name:
            return role
    return 'scalars'


def _rng_state() -> tuple[torch.Tensor, torch.Tensor | None]:
    return torch.get_rng_state(), torch.cuda.get_rng_state() if torch.cuda.is_available() else None


def _set_rng(state: tuple[torch.Tensor, torch.Tensor | None]) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1])


def _grads(params: Sequence[torch.Tensor]) -> list[torch.Tensor | None]:
    return [None if p.grad is None else p.grad.detach().clone() for p in params]


def _add(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    if a is None:
        return b
    return a if b is None else a + b


def _global_mean(t: torch.Tensor) -> torch.Tensor:
    """Returns the cross-rank mean of a tensor (a copy)."""
    t = t.clone()
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t


class _Accumulator:
    """Per-role sums of squared norms and inner products."""

    def __init__(self):
        self.sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def add(self, role: str, **values: float) -> None:
        for key, value in values.items():
            self.sums[role][key] += value
            self.sums['all'][key] += value


@dataclasses.dataclass
class SurvivalInputs:
    """Track-specific callables used by `mg_survival`.

    Attributes:
        run_half: Runs forward and backward on a list of microbatches, labelled 'A'
            or 'B', with exactly the scaling and objective of training.
        probe_loss: Returns the cross-rank mean clean loss on the training probe.
        muon_input: Returns the Muon input M = (1 - μ²) G + μ² buf for a matrix owned
            by this rank given its globally averaged gradient G, else None.
        preprocess: The map applied before orthogonalization (MuonEq-R or identity).
        zero_grad: Clears all gradients.
    """

    run_half: Callable[[Batches, str], None]
    probe_loss: Callable[[], float]
    muon_input: Callable[[str, torch.Tensor], torch.Tensor | None]
    preprocess: Callable[[torch.Tensor], torch.Tensor]
    zero_grad: Callable[[], None]


def _query_gradients(
    params: Sequence[torch.Tensor],
    lookahead: Lookahead,
    inputs: SurvivalInputs,
    pair: tuple[Batches, Batches],
    lr_multiplier: float,
    moment_buffers: Sequence[torch.Tensor],
) -> tuple[list, list, list]:
    """Returns g_A, g_B at θ and g_B at the displaced point for one batch pair."""
    adapt, query = pair
    inputs.zero_grad()
    inputs.run_half(adapt, 'A')
    grad_a = _grads(params)
    # Training displaces right after A, so the extra unadapted B pass must not leak
    # into the activation moments that the displacement reads.
    after_adapt = [b.detach().clone() for b in moment_buffers]
    rng = _rng_state()
    inputs.zero_grad()
    inputs.run_half(query, 'B')
    grad_b0 = _grads(params)
    with torch.no_grad():
        for buffer, saved in zip(moment_buffers, after_adapt):
            buffer.copy_(saved)
    inputs.zero_grad()
    for p, g in zip(params, grad_a):
        p.grad = None if g is None else g.clone()
    lookahead.displace(lr_multiplier, record=False)
    lookahead.abandon()
    inputs.zero_grad()
    _set_rng(rng)
    inputs.run_half(query, 'B')
    grad_b1 = _grads(params)
    inputs.zero_grad()
    return grad_a, grad_b0, grad_b1


def mg_survival(
    *,
    named_params: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    lookahead: Lookahead,
    lr_multiplier: float,
    inputs: SurvivalInputs,
    step_pairs: Sequence[tuple[Batches, Batches]],
    independent_pairs: Sequence[tuple[Batches, Batches]],
    representative: Sequence[str],
    buffers: Sequence[torch.Tensor] = (),
    moment_buffers: Sequence[torch.Tensor] = (),
    enter_vrank: Callable[[int], None] | None = None,
    snapshot_device: torch.device | str = 'cpu',
    seed: int = 0,
    bands: int = 3,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Measures how the MG correction propagates through the optimizer.

    Leaves parameters, optimizer state, buffers and RNG streams exactly as found.
    Every rank must call this at the same step.

    Args:
        named_params: All trainable parameters with names.
        optimizer: The live optimizer, already configured for this step.
        lookahead: The MG displacement of this run.
        lr_multiplier: LR multiplier of this step (for scheduled MG radii).
        inputs: Track-specific callables.
        step_pairs: The (A, B) microbatches of this step, one pair per virtual rank
            of this process; their mean gradient is the training step's gradient.
        independent_pairs: Other (A, B) pairs from the probe pool, one per virtual
            rank, for repeatability.
        representative: Names of matrices analysed in their singular basis.
        buffers: Every mutable buffer to restore, including per-rank copies.
        moment_buffers: The live activation-moment buffers read by the displacement.
        enter_vrank: Switches the process to a virtual rank (state and RNG), if any.
        snapshot_device: Device for the state snapshot.
        seed: Seed of the RNG streams used for the independent pairs.
        bands: Number of singular-value bands.

    Returns:
        Scalar metrics (identical on every rank) and, for matrices this rank owns,
        the singular values of their Muon inputs.
    """
    names = [n for n, _ in named_params]
    params = [p for _, p in named_params]
    snap = take_snapshot(params, optimizer, buffers, device=snapshot_device)
    metrics: dict[str, float] = {}
    keep = set(representative)

    # Pass 1: the step's own batch pairs, averaged over this process's virtual ranks
    # like the training step. Gradient-space statistics use cross-rank means; the
    # virtual updates use local gradients, as the optimizer averages.
    local_g0, local_g1 = None, None
    for vrank, pair in enumerate(step_pairs):
        if enter_vrank is not None:
            enter_vrank(vrank)
        grad_a, grad_b0, grad_b1 = _query_gradients(
            params, lookahead, inputs, pair, lr_multiplier, moment_buffers)
        restore_snapshot(snap, params, optimizer, buffers)
        g0 = [_add(a, b) for a, b in zip(grad_a, grad_b0)]
        g1 = [_add(a, b) for a, b in zip(grad_a, grad_b1)]
        del grad_a, grad_b0, grad_b1
        local_g0 = g0 if local_g0 is None else [_add(x, y) for x, y in zip(local_g0, g0)]
        local_g1 = g1 if local_g1 is None else [_add(x, y) for x, y in zip(local_g1, g1)]
    local_g0 = [None if g is None else g / len(step_pairs) for g in local_g0]
    local_g1 = [None if g is None else g / len(step_pairs) for g in local_g1]
    grad_acc = _Accumulator()
    global_g0, global_e = {}, {}
    for name, g0, g1 in zip(names, local_g0, local_g1):
        if g0 is None or g1 is None:
            continue
        mean_g0 = _global_mean(g0.float())
        mean_e = _global_mean((g1 - g0).float())
        grad_acc.add(
            parameter_role(name),
            g0_sq=float(mean_g0.square().sum()),
            e_sq=float(mean_e.square().sum()),
            e_dot_g0=float((mean_e * mean_g0).sum()),
        )
        if name in keep:
            global_g0[name], global_e[name] = mean_g0, mean_e

    # Pass 2: virtual optimizer updates from the same state.
    probe_ref = inputs.probe_loss()
    probe_after, first_delta = [], None
    upd_acc = _Accumulator()
    for local in (local_g0, local_g1):
        restore_snapshot(snap, params, optimizer, buffers)
        inputs.zero_grad()
        for p, g in zip(params, local):
            p.grad = g
        optimizer.step()
        inputs.zero_grad()
        probe_after.append(inputs.probe_loss())
        with torch.no_grad():
            if first_delta is None:
                first_delta = [(p.detach() - s.to(p.device)).to(snapshot_device)
                               for p, s in zip(params, snap.params)]
            else:
                for name, p, s, d0 in zip(names, params, snap.params, first_delta):
                    d1 = p.detach().float() - s.to(p.device).float()
                    d0 = d0.to(p.device).float()
                    upd_acc.add(
                        parameter_role(name),
                        d0_sq=float(d0.square().sum()),
                        diff_sq=float((d1 - d0).square().sum()),
                        d0_dot_d1=float((d0 * d1).sum()),
                        d1_sq=float(d1.square().sum()),
                    )
    restore_snapshot(snap, params, optimizer, buffers)
    del local_g0, local_g1, first_delta

    metrics['diag/probe_ref'] = probe_ref
    metrics['diag/probe_gain_base'] = probe_ref - probe_after[0]
    metrics['diag/probe_gain_mg'] = probe_after[0] - probe_after[1]
    for role, s in grad_acc.sums.items():
        g0_norm, e_norm = math.sqrt(s['g0_sq']), math.sqrt(s['e_sq'])
        grad_rel = e_norm / max(g0_norm, 1e-30)
        metrics[f'diag/grad_rel/{role}'] = grad_rel
        metrics[f'diag/grad_cos/{role}'] = s['e_dot_g0'] / max(e_norm * g0_norm, 1e-30)
        u = upd_acc.sums.get(role)
        if u and u['d0_sq'] > 0:
            upd_rel = math.sqrt(u['diff_sq'] / u['d0_sq'])
            metrics[f'diag/upd_rel/{role}'] = upd_rel
            metrics[f'diag/upd_cos/{role}'] = u['d0_dot_d1'] / max(
                math.sqrt(u['d0_sq'] * u['d1_sq']), 1e-30)
            metrics[f'diag/survival/{role}'] = upd_rel / max(grad_rel, 1e-30)

    # Pass 3: independent pairs, for the repeatability of the representative
    # matrices' corrections. Fresh RNG streams avoid sharing dropout masks.
    local_e: dict[str, torch.Tensor] = {}
    for vrank, pair in enumerate(independent_pairs):
        if enter_vrank is not None:
            enter_vrank(vrank)
        torch.manual_seed(seed + vrank)
        _, grad_b0, grad_b1 = _query_gradients(
            params, lookahead, inputs, pair, lr_multiplier, moment_buffers)
        restore_snapshot(snap, params, optimizer, buffers)
        for name, b0, b1 in zip(names, grad_b0, grad_b1):
            if name in keep and b0 is not None and b1 is not None:
                e = (b1 - b0).float() / len(independent_pairs)
                local_e[name] = e if name not in local_e else local_e[name] + e
        del grad_b0, grad_b1
    independent_e = {name: _global_mean(e) for name, e in local_e.items()}

    # Pass 4: singular-basis analysis on the owning rank.
    local_metrics, spectra = {}, {}
    for name in representative:
        if name not in global_g0:
            continue
        base = inputs.muon_input(name, global_g0[name])
        if base is None:
            continue
        x0 = inputs.preprocess(base)
        x1 = inputs.preprocess(inputs.muon_input(name, global_g0[name] + global_e[name]))
        lin = spectral.linearize_polar(x0, x1 - x0, bands=bands)
        actual = spectral.polar_express(x1).double() - spectral.polar_express(x0).double()
        ideal = spectral.polar_derivative(x0, x1 - x0)
        tag = f'diag/spec/{name}'
        local_metrics[f'{tag}/symmetric'] = lin.symmetric_fraction
        local_metrics[f'{tag}/antisymmetric'] = lin.antisymmetric_fraction
        local_metrics[f'{tag}/complement'] = lin.complement_fraction
        local_metrics[f'{tag}/gain'] = lin.gain
        for b, share in enumerate(lin.band_energy):
            local_metrics[f'{tag}/band{b}'] = share
        local_metrics[f'{tag}/pe_vs_linear_cos'] = float(
            (actual * ideal).sum() / (actual.norm() * ideal.norm()).clamp_min(1e-300))
        local_metrics[f'{tag}/pe_vs_linear_ratio'] = float(actual.norm() / ideal.norm().clamp_min(1e-300))
        if name in independent_e:
            x1_other = inputs.preprocess(inputs.muon_input(name, global_g0[name] + independent_e[name]))
            other = spectral.linearize_polar(x0, x1_other - x0, bands=bands)
            cosines = spectral.band_cosines(lin, other, bands=bands)
            for b, value in enumerate(cosines[:-1]):
                local_metrics[f'{tag}/repeat_band{b}'] = value
            local_metrics[f'{tag}/repeat_all'] = cosines[-1]
        spectra[name] = torch.linalg.svdvals(x0.double()).float().cpu()
    restore_snapshot(snap, params, optimizer, buffers)

    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered: list[dict[str, float]] = [{} for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_metrics)
        for part in gathered:
            metrics.update(part)
    else:
        metrics.update(local_metrics)
    return metrics, spectra
