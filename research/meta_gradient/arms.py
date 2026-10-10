"""Named experiment arms and studies for the tiny and one-hour tracks.

An arm is a set of command-line flags on top of the record defaults of a track's
trainer. Flags may reference a file produced by another arm's run through the
placeholder {ref:ARM:FILE}; the runner substitutes the path of that file in the
reference-seed run of ARM and schedules ARM first. A study is an ordered list
of arms that answers one research question.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Sequence

TRACKS = ('tiny', 'hour')
TRAINERS = {'tiny': 'tiny_train.py', 'hour': 'hour_train.py'}

_REF = re.compile(r'\{ref:([^:}]+):([^}]+)\}')
_CALIBRATION = 'mg_calibration.json'


@dataclasses.dataclass(frozen=True)
class Arm:
    """One configuration of a trainer.

    Attributes:
        name: Short identifier, unique within a track.
        track: 'tiny' or 'hour'.
        flags: Command-line flags added to the record defaults.
        priority: Research direction of the arm ('base', 'A'...'E', 'AB').
        description: What the arm changes and why it is there.
    """

    name: str
    track: str
    flags: tuple[str, ...]
    priority: str
    description: str

    def references(self) -> list[tuple[str, str]]:
        """Returns the (arm, file) pairs this arm's flags depend on."""
        return [m.groups() for flag in self.flags for m in _REF.finditer(flag)]

    def resolve(self, locate: Callable[[str, str], str]) -> list[str]:
        """Returns the flags with every reference replaced by `locate(arm, file)`."""
        return [_REF.sub(lambda m: locate(m.group(1), m.group(2)), flag) for flag in self.flags]


def _arms_for(track: str) -> list[Arm]:
    """Builds every arm of one track."""

    def arm(name: str, priority: str, description: str, *flags: str) -> Arm:
        return Arm(name=name, track=track, flags=tuple(flags), priority=priority,
                   description=description)

    nomg_flags = ('--mg-every', '0') if track == 'tiny' else (
        '--mg-every', '0', '--dropout', '0.1', '--stoch-depth', '0.05')
    strata = ('--eval-strata',) if track == 'tiny' else ()
    calibration = f'{{ref:B-calib:{_CALIBRATION}}}'
    arms = [
        arm('base', 'base', 'The record, unchanged (A0, C-polar-express, D-trunc).', *strata),
        arm('nomg', 'base', 'The previous record without the meta-gradient step.', *nomg_flags, *strata),
        arm('nomg-same', 'base', 'No meta-gradient step, record regularization (clean MG x MONA 2x2).',
            '--mg-every', '0', *strata),
        # Priority A: auxiliary adaptation, clean query.
        arm('A1', 'A', 'Auxiliary loss only in the adaptation half, at twice the weight.',
            '--aux-split', 'adapt'),
        arm('A2', 'A', 'Ordering control: auxiliary loss only in the query half.',
            '--aux-split', 'query'),
        arm('A3', 'A', 'A1 objective with zero displacement (variance control).',
            '--aux-split', 'adapt', '--mg-step-norm', '0'),
        arm('A4', 'A', 'Record objective with zero displacement (split control).',
            '--mg-step-norm', '0'),
        # Priority B: functional radius of the temporary step on c_proj.
        arm('B-calib', 'B', 'Record with c_proj input moments logged; calibrates rho. '
            'Training is identical to base.', '--mg-act-stats'),
    ]
    for mult in ('0.5', '1', '2'):
        arms.append(arm(f'B-rad-x{mult}', 'B',
                        f'Raw-gradient direction, activation radius {mult} x calibrated.',
                        '--mg-act-stats', '--mg-cproj', 'act_radius', '--mg-act-calibration',
                        calibration, '--mg-act-radius-mult', mult))
        arms.append(arm(f'B-met-x{mult}', 'B',
                        f'Metric direction G C^-1, activation radius {mult} x calibrated.',
                        '--mg-act-stats', '--mg-cproj', 'act_metric', '--mg-act-calibration',
                        calibration, '--mg-act-radius-mult', mult))
    arms += [
        arm('B-euc-rad', 'B', 'Euclidean per-layer radii matched to B-rad-x1 displacements.',
            '--mg-act-stats', '--mg-cproj', 'euclid_layer', '--mg-cproj-radii',
            f'{{ref:B-rad-x1:{_CALIBRATION}}}'),
        arm('B-euc-met', 'B', 'Euclidean per-layer radii matched to B-met-x1 displacements.',
            '--mg-act-stats', '--mg-cproj', 'euclid_layer', '--mg-cproj-radii',
            f'{{ref:B-met-x1:{_CALIBRATION}}}'),
        arm('B-met-x1-damp', 'B', 'B-met-x1 with damping 100: must recover the Euclidean direction.',
            '--mg-act-stats', '--mg-cproj', 'act_metric', '--mg-act-calibration', calibration,
            '--mg-act-radius-mult', '1', '--mg-act-damping', '100'),
        # Priority C: survival diagnostic, update maps, adjacent baselines.
        arm('C-diag', 'C', 'Record trajectory with the MG survival diagnostic at fixed steps.',
            '--diag-steps', '300,900,1500,1900,2100,2500,2800' if track == 'tiny'
            else '150,500,900,1300,1600,1900'),
        arm('C-exact', 'C', 'Near-exact polar factor (float64 eigh, regularized below 1e-3 RMS), '
            'norm-matched to Polar Express.',
            '--spectral', 'eigh', '--spectral-c', '0.5', '--spectral-eps', '0.001', '--spectral-fp64'),
        arm('C-reg', 'C', 'Regularized polar response s/sqrt(s^2+eps^2), eps = 0.1 RMS, norm-matched.',
            '--spectral', 'eigh', '--spectral-c', '0.5', '--spectral-eps', '0.1'),
        arm('C-reg-aug', 'C', 'Approximate C-reg: Polar Express on [X; eps I] (fast path, PE accuracy).',
            '--spectral', 'augmented_polar', '--spectral-eps', '0.1'),
        arm('C-inv', 'C', 'Inverse-power response c = 2/3 (Freon-like), eps = 0.1 RMS, norm-matched.',
            '--spectral', 'eigh', '--spectral-c', '0.6667', '--spectral-eps', '0.1'),
        arm('C-2phase', 'C', 'c = 1/2 until 70% of training, then 2/3.',
            '--spectral', 'eigh', '--spectral-c', '0.5', '--spectral-eps', '0.1',
            '--spectral-c-late', '0.6667', '--spectral-switch-frac', '0.7'),
        arm('mona', 'C', "MONA's gradient correction on the record optimizer, without MG (comparator).",
            '--mona-beta', '0.98', '--mg-every', '0'),
        arm('mona-mg', 'C', "MONA's gradient correction stacked on the record MG step.", '--mona-beta', '0.98'),
        # Combination of the first two branches (2x2 with base, A1 and B-met-x1).
        arm('AB-met', 'AB', 'A1 objective with the B-met-x1 displacement.',
            '--aux-split', 'adapt', '--mg-act-stats', '--mg-cproj', 'act_metric',
            '--mg-act-calibration', calibration, '--mg-act-radius-mult', '1'),
        arm('AB-rad', 'AB', 'A1 objective with the B-rad-x1 displacement.',
            '--aux-split', 'adapt', '--mg-act-stats', '--mg-cproj', 'act_radius',
            '--mg-act-calibration', calibration, '--mg-act-radius-mult', '1'),
    ]
    # Same small LR budget for the baseline and the inverse-power map.
    base_lr = 0.04
    for factor in ('0.8', '1.25'):
        lr = f'{base_lr * float(factor):.4g}'
        arms.append(arm(f'base-lr{factor}', 'C', f'Record with matrix LR x{factor}.',
                        '--matrix-lr', lr))
        arms.append(arm(f'C-inv-lr{factor}', 'C', f'C-inv with matrix LR x{factor}.',
                        '--spectral', 'eigh', '--spectral-c', '0.6667', '--spectral-eps', '0.1',
                        '--matrix-lr', lr))
    if track == 'tiny':
        arms += [
            arm('eqr', 'C', 'MuonEq-R row normalization (absent from the tiny record).',
                '--muon-eq-r'),
            # Follow-ups of C-reg-aug: its regularization, MuonEq-R on top, and the map without MG.
            arm('C-reg-aug-e0.05', 'C', 'C-reg-aug with eps = 0.05 RMS.',
                '--spectral', 'augmented_polar', '--spectral-eps', '0.05'),
            arm('C-reg-aug-e0.2', 'C', 'C-reg-aug with eps = 0.2 RMS.',
                '--spectral', 'augmented_polar', '--spectral-eps', '0.2'),
            arm('C-reg-aug-eqr', 'C', 'C-reg-aug after MuonEq-R row normalization.',
                '--spectral', 'augmented_polar', '--spectral-eps', '0.1', '--muon-eq-r'),
            arm('C-reg-aug-nomg', 'C', 'C-reg-aug without the MG step (pairs with nomg-same).',
                '--spectral', 'augmented_polar', '--spectral-eps', '0.1', '--mg-every', '0'),
            # Priority D: recurrent credit. Without MG the compensated estimator is an unbiased
            # raw gradient at fixed weights (compare with nomg); with MG the draw also moves the
            # adaptation point, so the MG arms are D x MG interaction experiments (compare with base).
            arm('D-full-nomg', 'D', 'Always differentiate through both passes, no MG.',
                '--credit', 'full', '--mg-every', '0'),
            arm('D-r25-nomg', 'D', 'Full credit on 25% of steps, uncompensated, no MG.',
                '--credit', 'random', '--credit-p', '0.25', '--mg-every', '0'),
            arm('D-r25c-nomg', 'D', 'Full credit on 25% of steps, path gradient x4 (unbiased), no MG.',
                '--credit', 'random_comp', '--credit-p', '0.25', '--mg-every', '0'),
            arm('D-r50c-nomg', 'D', 'Full credit on 50% of steps, path gradient x2 (unbiased), no MG.',
                '--credit', 'random_comp', '--credit-p', '0.5', '--mg-every', '0'),
            arm('D-full', 'D', 'D x MG: always differentiate through both passes, with MG.',
                '--credit', 'full'),
            arm('D-r25', 'D', 'D x MG: full credit on 25% of steps, uncompensated, with MG.',
                '--credit', 'random', '--credit-p', '0.25'),
            arm('D-r25c', 'D', 'D x MG: full credit on 25% of steps, path gradient x4, with MG.',
                '--credit', 'random_comp', '--credit-p', '0.25'),
            arm('D-r50c', 'D', 'D x MG: full credit on 50% of steps, path gradient x2, with MG.',
                '--credit', 'random_comp', '--credit-p', '0.5'),
            # Priority E: conditional n-gram memory.
            arm('E-uni', 'E', 'Capacity control: gated unigram memory, 4 heads (same slots and width).',
                '--ngram', 'unigram', '--ngram-heads', '4', '--eval-strata'),
            arm('E-hash', 'E', 'Hashed 2/3-gram memory, no gate, no shrinkage.',
                '--ngram', 'hashed', '--ngram-gate', 'none', '--eval-strata'),
            arm('E-gate', 'E', 'Hashed 2/3-gram memory with the contextual gate.',
                '--ngram', 'hashed', '--eval-strata'),
            arm('E-shrink', 'E', 'Gated memory with exact-count shrinkage c/(c+8).',
                '--ngram', 'hashed', '--ngram-kappa', '8', '--eval-strata'),
            arm('E-shrink-nomg', 'E', 'E-shrink without the MG step (pairs with nomg).',
                '--ngram', 'hashed', '--ngram-kappa', '8', '--eval-strata', *nomg_flags),
            arm('E-dense', 'E', 'Parameter-matched dense control: an MLP adapter at the same layer.',
                '--ngram', 'dense', '--eval-strata'),
        ]
    return arms


ARMS: dict[str, dict[str, Arm]] = {
    track: {a.name: a for a in _arms_for(track)} for track in TRACKS
}

STUDIES: dict[str, tuple[str, tuple[str, ...]]] = {
    'tiny-reproduce': ('tiny', ('base',)),
    'tiny-A': ('tiny', ('base', 'A1', 'A2', 'A3', 'A4')),
    'tiny-B': ('tiny', ('B-rad-x1', 'B-met-x1', 'B-euc-rad', 'B-euc-met', 'B-met-x1-damp')),
    'tiny-B-radius': ('tiny', ('B-rad-x0.5', 'B-rad-x2', 'B-met-x0.5', 'B-met-x2')),
    'tiny-C-diag': ('tiny', ('C-diag',)),
    'tiny-C': ('tiny', ('base', 'C-exact', 'C-reg', 'C-reg-aug', 'C-inv', 'C-2phase', 'eqr',
                        'nomg-same', 'mona', 'mona-mg')),
    'tiny-C-lr': ('tiny', ('base-lr0.8', 'base-lr1.25', 'C-inv-lr0.8', 'C-inv-lr1.25')),
    'tiny-C-reg-aug': ('tiny', ('C-reg-aug-e0.05', 'C-reg-aug-e0.2', 'C-reg-aug-eqr', 'nomg-same',
                                'C-reg-aug-nomg')),
    'tiny-D': ('tiny', ('nomg', 'D-full-nomg', 'D-r25-nomg', 'D-r25c-nomg', 'D-r50c-nomg')),
    'tiny-DxMG': ('tiny', ('base', 'D-full', 'D-r25', 'D-r25c', 'D-r50c')),
    'tiny-E': ('tiny', ('base', 'nomg', 'E-uni', 'E-hash', 'E-gate', 'E-shrink', 'E-shrink-nomg',
                        'E-dense')),
    'tiny-AB': ('tiny', ('base', 'A1', 'B-met-x1', 'AB-met')),
    'hour-reproduce': ('hour', ('base',)),
    'hour-A': ('hour', ('base', 'A1', 'A2', 'A3', 'A4')),
    'hour-B': ('hour', ('B-rad-x1', 'B-met-x1', 'B-euc-rad', 'B-euc-met')),
    'hour-C-diag': ('hour', ('C-diag',)),
    'hour-C': ('hour', ('base', 'nomg-same', 'mona', 'mona-mg')),
    'hour-AB': ('hour', ('base', 'A1', 'B-met-x1', 'AB-met')),
}


def get_arm(track: str, name: str) -> Arm:
    """Returns a registered arm.

    Raises:
        KeyError: Unknown track or arm, with the valid names in the message.
    """
    if track not in ARMS:
        raise KeyError(f'Unknown track {track!r}; expected one of {TRACKS}.')
    if name not in ARMS[track]:
        raise KeyError(f'Unknown {track} arm {name!r}; known: {", ".join(sorted(ARMS[track]))}.')
    return ARMS[track][name]


def requested_names(track: str, names: Sequence[str]) -> list[str]:
    """Expands study names (and 'all') into arm names, keeping order.

    Args:
        track: The track of every arm.
        names: Arm names, study names or 'all'.

    Returns:
        The arm names, possibly with repeats.

    Raises:
        KeyError: A study of another track.
    """
    requested = []
    for name in names:
        if name in STUDIES:
            study_track, members = STUDIES[name]
            if study_track != track:
                raise KeyError(f'Study {name!r} belongs to track {study_track!r}.')
            requested.extend(members)
        elif name == 'all':
            requested.extend(ARMS[track])
        else:
            requested.append(name)
    return requested


def expand(track: str, names: Sequence[str]) -> list[Arm]:
    """Returns the arms named (or study-named) in `names`, dependencies first.

    Args:
        track: The track of every arm.
        names: Arm names and study names, in the desired order.

    Returns:
        Unique arms in an order where every referenced arm precedes its users.
    """
    requested = requested_names(track, names)
    ordered: list[Arm] = []

    def visit(name: str, chain: tuple[str, ...]) -> None:
        if name in chain:
            raise ValueError(f'Circular arm references: {" -> ".join(chain + (name,))}.')
        arm = get_arm(track, name)
        for ref, _ in arm.references():
            visit(ref, chain + (name,))
        if arm not in ordered:
            ordered.append(arm)

    for name in requested:
        visit(name, ())
    return ordered


def describe(track: str) -> str:
    """Returns a markdown table of a track's arms."""
    lines = ['| Arm | Priority | Flags | Description |', '|---|---|---|---|']
    for arm in ARMS[track].values():
        flags = ' '.join(arm.flags) or '(none)'
        lines.append(f'| {arm.name} | {arm.priority} | `{flags}` | {arm.description} |')
    return '\n'.join(lines)


if __name__ == '__main__':
    for _track in TRACKS:
        print(f'## {_track}\n')
        print(describe(_track))
        print()
