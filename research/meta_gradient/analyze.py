"""Summarizes experiment runs with paired, seed-matched statistics.

Reads runs/<prefix>/<track>/<arm>/seed<s>/result.json and prints markdown tables:
per-arm means, paired differences against a reference arm with Student-t
intervals, optional 2x2 interaction contrasts, held-out strata and the
meta-gradient survival diagnostic.

Examples, from the repository root:

    python research/meta_gradient/analyze.py --track tiny
    python research/meta_gradient/analyze.py --track tiny --interaction base,A1,B-met-x1,AB-met
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Sequence
from typing import NamedTuple

# Two-sided 95% Student-t quantiles by degrees of freedom.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
    10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
}

# Training-time caps of the tracks, in minutes.
TIME_CAPS = {'tiny': 15.0, 'hour': 60.0}

# Decision thresholds (val-loss improvement): a useful component and a substantial lead.
USEFUL_GAIN = 0.003
SUBSTANTIAL_GAIN = 0.01

# Flags that define the records' schedule, batch and data; any change makes a run
# a research result rather than a leaderboard-eligible one.
RECORD_SETUP = {
    'tiny': {'num_epochs': 15, 'total_batch_size': 524288, 'device_batch_size': 32, 'seq_len': 2048,
             'eval_tokens': 10_000_000, 'max_train_steps': 3040, 'input_bin': None,
             'input_val_bin': None, 'virtual_ranks': 8, 'diag_steps': ''},
    'hour': {'num_epochs': 11, 'total_batch_size': 524288, 'device_batch_size': 4, 'seq_len': 2048,
             'eval_tokens': 10_000_000, 'input_bin': None, 'input_val_bin': None,
             'virtual_ranks': 8, 'diag_steps': ''},
}


def eligibility_problems(result: dict, track: str) -> list[str]:
    """Returns why a run cannot count toward the leaderboard (empty if it can).

    A run must use eight H100 processes, the record schedule, batch and data, no
    diagnostic work, and finish training within the cap. Diagnostic runs are
    research-only because their probe work is excluded from the timer.

    Args:
        result: A parsed result.json.
        track: 'tiny' or 'hour'.

    Returns:
        Reasons, empty for an eligible run.
    """
    problems = []
    if result.get('world_size') != 8:
        problems.append(f'{result.get("world_size")} processes')
    if 'H100' not in str(result.get('env', {}).get('gpu', '')):
        problems.append(f'device {result.get("env", {}).get("gpu")}')
    args = result.get('args', {})
    for key, value in RECORD_SETUP[track].items():
        if args.get(key, value) != value:
            problems.append(f'{key}={args.get(key)!r}')
    minutes = result.get('total_training_time_s', math.inf) / 60
    if minutes > TIME_CAPS[track]:
        problems.append(f'{minutes:.1f} min > {TIME_CAPS[track]:.0f} min cap')
    return problems


def _configuration(result: dict) -> tuple:
    """Returns the hardware configuration a result was produced on."""
    return result.get('world_size'), result.get('virtual_ranks'), result.get('env', {}).get('gpu')


def t_quantile(df: int) -> float:
    """Returns the two-sided 95% Student-t quantile for `df` degrees of freedom."""
    if df < 1:
        return math.nan
    if df in _T975:
        return _T975[df]
    lower = max(k for k in _T975 if k < df)
    upper = min((k for k in _T975 if k > df), default=None)
    if upper is None:
        return 1.96 + (_T975[30] - 1.96) * 30 / df
    return _T975[lower] + (_T975[upper] - _T975[lower]) * (df - lower) / (upper - lower)


class Estimate(NamedTuple):
    """Mean of paired differences with a 95% Student-t interval.

    Attributes:
        mean: Mean difference.
        sd: Sample standard deviation (nan for one pair).
        half_width: Half-width of the 95% interval (nan for one pair).
        values: The individual differences.
    """

    mean: float
    sd: float
    half_width: float
    values: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.values)

    def interval(self) -> tuple[float, float]:
        return self.mean - self.half_width, self.mean + self.half_width


def paired_estimate(values: Sequence[float]) -> Estimate:
    """Returns the mean, sd and 95% t-interval of paired differences.

    Args:
        values: One difference per seed.

    Returns:
        The estimate; sd and interval are nan with fewer than two values.

    Raises:
        ValueError: No values.
    """
    if not values:
        raise ValueError('No paired values.')
    mean = statistics.fmean(values)
    if len(values) < 2:
        return Estimate(mean, math.nan, math.nan, tuple(values))
    sd = statistics.stdev(values)
    return Estimate(mean, sd, t_quantile(len(values) - 1) * sd / math.sqrt(len(values)), tuple(values))


def interaction(results: dict[str, dict[int, float]], base: str, a: str, b: str, ab: str) -> Estimate:
    """Returns the 2x2 interaction I = L_AB - L_A - L_B + L_base over shared seeds.

    Negative values mean the combination beats the sum of its parts.

    Args:
        results: arm -> seed -> metric.
        base: Reference arm.
        a: First single-change arm.
        b: Second single-change arm.
        ab: Combined arm.

    Returns:
        The estimate over seeds present in all four arms.

    Raises:
        ValueError: No seed is shared by all four arms.
    """
    seeds = set(results[base]) & set(results[a]) & set(results[b]) & set(results[ab])
    if not seeds:
        raise ValueError('No seed is shared by all four arms.')
    return paired_estimate([
        results[ab][s] - results[a][s] - results[b][s] + results[base][s] for s in sorted(seeds)
    ])


def load_results(root: str, track: str) -> dict[str, dict[int, dict]]:
    """Returns completed results as arm -> seed -> result dictionary."""
    found: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(root, track, '*', 'seed*', 'result.json'))):
        try:
            with open(path) as f:
                result = json.load(f)
        except (OSError, ValueError):
            continue
        if result.get('status') != 'complete':
            continue
        found[result.get('arm', path.split(os.sep)[-3])][int(result['seed'])] = result
    return found


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '–'
    return f'{value:.{digits}f}'


def _signed(value: float, digits: int = 4) -> str:
    return '–' if math.isnan(value) else f'{value:+.{digits}f}'


def _verdict(estimate: Estimate) -> str:
    """Labels a paired difference (variant minus reference; negative is better)."""
    if estimate.n < 2:
        return 'one seed'
    low, high = estimate.interval()
    if high < 0:
        gain = -estimate.mean
        return 'substantial' if gain >= SUBSTANTIAL_GAIN else ('useful' if gain >= USEFUL_GAIN else 'small win')
    if low > 0:
        return 'worse'
    return 'inconclusive'


def arm_table(results: dict[str, dict[int, dict]], *, track: str, metric: str, reference: str) -> str:
    """Returns the per-arm markdown table with paired differences.

    Loss verdicts are research results. "Eligible" counts the runs that satisfy every
    leaderboard condition, including the time cap per run (not on average).
    """
    cap = TIME_CAPS[track]
    lines = [
        f'| Arm | Seeds | {metric} (mean ± sd) | Δ vs {reference} (paired) | 95% CI | Verdict | '
        f'Train min (max) | Over {cap:.0f} min | Eligible | Peak GiB |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    notes = []
    ref = {s: r[metric] for s, r in results.get(reference, {}).items() if r.get(metric) is not None}
    for arm in sorted(results, key=lambda a: (a != reference, a)):
        runs = results[arm]
        values = [r[metric] for r in runs.values() if r.get(metric) is not None]
        if not values:
            continue
        sd = statistics.stdev(values) if len(values) > 1 else math.nan
        minutes = statistics.fmean(r.get('total_training_time_s', math.nan) for r in runs.values()) / 60
        memory = statistics.fmean(r.get('peak_memory_mib', math.nan) for r in runs.values()) / 1024
        shared = sorted(set(ref) & {s for s, r in runs.items() if r.get(metric) is not None})
        delta, interval, verdict = '–', '–', '–'
        if arm != reference and shared:
            est = paired_estimate([runs[s][metric] - ref[s] for s in shared])
            low, high = est.interval()
            delta = f'{_signed(est.mean)} (n={est.n})'
            interval = '–' if math.isnan(low) else f'[{low:+.4f}, {high:+.4f}]'
            verdict = _verdict(est)
        run_minutes = [r.get('total_training_time_s', math.nan) / 60 for r in runs.values()]
        over = sum(m > cap for m in run_minutes)
        eligible = sum(not eligibility_problems(r, track) for r in runs.values())
        lines.append(
            f'| {arm} | {len(values)} | {_fmt(statistics.fmean(values))} ± {_fmt(sd)} | {delta} | '
            f'{interval} | {verdict} | {_fmt(minutes, 1)} ({_fmt(max(run_minutes), 1)}) | '
            f'{over}/{len(run_minutes)} | {eligible}/{len(runs)} | {_fmt(memory, 1)} |'
        )
        configurations = {_configuration(r) for r in runs.values()}
        if len(configurations) > 1:
            notes.append(f'**{arm}** mixes hardware configurations {sorted(configurations, key=str)}; '
                         'compare it only within one configuration.')
        if eligible < len(runs):
            reasons = sorted({p for r in runs.values() for p in eligibility_problems(r, track)})
            notes.append(f'{arm}: not eligible because of ' + '; '.join(reasons[:4]))
    return '\n'.join(lines + ([''] + [f'- {n}' for n in notes] if notes else []))


def component_table(results: dict[str, dict[int, dict]], track: str) -> str:
    """Returns the aggregation components and the train/held-out gap per arm."""
    if track == 'tiny':
        columns = [('single ckpt', 'epochs'), ('EMA', 'ema_val_loss'), ('ckpt avg', 'ckpt_avg_val_loss')]
    else:
        columns = [('single ckpt', 'epochs'), ('logit avg equal', 'logit_avg_equal_loss'),
                   ('logit avg recency', 'logit_avg_weighted_loss')]
    header = ' | '.join(c for c, _ in columns)
    lines = [f'| Arm | {header} | train probe | gap (val − probe) |',
             '|---|' + '---|' * (len(columns) + 2)]
    for arm in sorted(results):
        cells = []
        for _, key in columns:
            if key == 'epochs':
                vals = [r['epochs'][-1]['val_loss'] for r in results[arm].values() if r.get('epochs')]
            else:
                vals = [r[key] for r in results[arm].values() if r.get(key) is not None]
            cells.append(_fmt(statistics.fmean(vals)) if vals else '–')
        probes = [r['epochs'][-1]['train_probe_loss'] for r in results[arm].values() if r.get('epochs')]
        gaps = [r['epochs'][-1]['val_loss'] - r['epochs'][-1]['train_probe_loss']
                for r in results[arm].values() if r.get('epochs')]
        lines.append(f'| {arm} | ' + ' | '.join(cells) +
                     f' | {_fmt(statistics.fmean(probes)) if probes else "–"}'
                     f' | {_fmt(statistics.fmean(gaps)) if gaps else "–"} |')
    return '\n'.join(lines)


def _coefficient_of_variation(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else math.nan


def displacement_table(results: dict[str, dict[int, dict]]) -> str | None:
    """Returns the per-layer c_proj displacement profile of runs with input moments.

    The hypothesis is that activation-normalized steps reduce layerwise outliers of
    the functional displacement; the coefficient of variation across layers shows it.
    """
    rows = []
    for arm in sorted(results):
        calibrations = [r['mg']['calibration'] for r in results[arm].values()
                        if r.get('mg', {}).get('calibration')]
        if not calibrations:
            continue
        act = statistics.fmean(c['act_disp_mean'] for c in calibrations)
        act_cv = statistics.fmean(_coefficient_of_variation(c['act_disp']) for c in calibrations)
        frob = statistics.fmean(statistics.fmean(c['frob_disp']) for c in calibrations)
        frob_cv = statistics.fmean(_coefficient_of_variation(c['frob_disp']) for c in calibrations)
        rows.append(f'| {arm} | {calibrations[0]["mode"]} | {_fmt(act)} | {_fmt(act_cv, 3)} | '
                    f'{_fmt(frob)} | {_fmt(frob_cv, 3)} |')
    if not rows:
        return None
    return '\n'.join(['| Arm | c_proj mode | functional disp. | CV across layers | Frobenius disp. | '
                      'CV across layers |', '|---|---|---|---|---|---|'] + rows)


def timing_table(results: dict[str, dict[int, dict]]) -> str | None:
    """Returns mean step time per training phase, in milliseconds."""
    phases = sorted({p for runs in results.values() for r in runs.values()
                     for p in r.get('step_time_by_phase_s', {})})
    if not phases:
        return None
    lines = ['| Arm | ' + ' | '.join(phases) + ' |', '|---|' + '---|' * len(phases)]
    for arm in sorted(results):
        cells = []
        for phase in phases:
            values = [r['step_time_by_phase_s'][phase] for r in results[arm].values()
                      if phase in r.get('step_time_by_phase_s', {})]
            cells.append(_fmt(1000 * statistics.fmean(values), 1) if values else '–')
        lines.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def strata_table(results: dict[str, dict[int, dict]]) -> str | None:
    """Returns held-out loss by training support of the causal suffix, if recorded."""
    rows = []
    for arm in sorted(results):
        runs = [r for r in results[arm].values() if r.get('strata')]
        if not runs:
            continue
        cells = []
        for name in ('unseen', 'rare', 'common', 'no_context'):
            losses = [r['strata'][f'strata/{name}/loss'] for r in runs]
            share = statistics.fmean(r['strata'][f'strata/{name}/share'] for r in runs)
            cells.append(f'{_fmt(statistics.fmean(losses))} ({100 * share:.1f}%)')
        rows.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
    if not rows:
        return None
    return '\n'.join(['| Arm | unseen | rare (1–9) | common (≥10) | no context |', '|---|---|---|---|---|'] + rows)


def diag_table(root: str, track: str) -> str | None:
    """Summarizes diag.json files: survival and probe gains by role, averaged over steps."""
    paths = sorted(glob.glob(os.path.join(root, track, '*', 'seed*', 'diag.json')))
    if not paths:
        return None
    sums: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        with open(path) as f:
            for record in json.load(f):
                for key, value in record.items():
                    if key.startswith('diag/') and isinstance(value, (int, float)):
                        sums[key].append(float(value))
    roles = sorted({k.split('/')[2] for k in sums if k.startswith('diag/survival/')})
    lines = ['| Role | ‖E‖/‖G0‖ (grad) | cos(E, G0) | ‖ΔU‖/‖U0‖ (update) | survival ratio |',
             '|---|---|---|---|---|']
    for role in roles:
        cells = [statistics.fmean(sums.get(f'diag/{name}/{role}', [math.nan]))
                 for name in ('grad_rel', 'grad_cos', 'upd_rel', 'survival')]
        lines.append(f'| {role} | ' + ' | '.join(_fmt(c) for c in cells) + ' |')
    gain_base = sums.get('diag/probe_gain_base', [])
    gain_mg = sums.get('diag/probe_gain_mg', [])
    if gain_base:
        lines.append('')
        lines.append(f'Probe loss gain of one baseline step: {statistics.fmean(gain_base):+.3e}; '
                     f'extra gain from the MG correction: {statistics.fmean(gain_mg):+.3e} '
                     f'(positive = helps; {len(gain_mg)} diagnostic steps).')
    spectral_keys = sorted({k.rsplit('/', 1)[0] for k in sums if k.startswith('diag/spec/')})
    if spectral_keys:
        lines += ['', '| Matrix | discarded (sym) | rotation (antisym) | complement | gain | '
                      'repeatability top/mid/bottom | PE vs linear cos |',
                  '|---|---|---|---|---|---|---|']
        for key in spectral_keys:
            def mean(name: str) -> float:
                return statistics.fmean(sums.get(f'{key}/{name}', [math.nan]))
            repeat = '/'.join(_fmt(mean(f'repeat_band{b}'), 2) for b in range(3))
            lines.append(f'| {key.removeprefix("diag/spec/")} | {_fmt(mean("symmetric"), 3)} | '
                         f'{_fmt(mean("antisymmetric"), 3)} | {_fmt(mean("complement"), 3)} | '
                         f'{_fmt(mean("gain"), 2)} | {repeat} | {_fmt(mean("pe_vs_linear_cos"), 3)} |')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default='runs/lab', help='runs/<prefix> directory')
    parser.add_argument('--track', choices=tuple(TIME_CAPS), required=True)
    parser.add_argument('--metric', default='best_val_loss',
                        help='Result field to compare (best_val_loss is the leaderboard metric)')
    parser.add_argument('--reference', default='base')
    parser.add_argument('--interaction', action='append', default=[],
                        help='base,A,B,AB arm names for a 2x2 interaction contrast (repeatable)')
    parser.add_argument('--output', default=None, help='Also write the report to this markdown file')
    args = parser.parse_args()

    results = load_results(args.root, args.track)
    if not results:
        raise SystemExit(f'No completed runs under {os.path.join(args.root, args.track)}.')
    metric_results = {arm: {s: r[args.metric] for s, r in runs.items() if r.get(args.metric) is not None}
                      for arm, runs in results.items()}
    sections = [f'# {args.track} results ({args.metric}; lower is better)', '',
                arm_table(results, track=args.track, metric=args.metric, reference=args.reference), '',
                '## Aggregation components and generalization gap', '',
                component_table(results, args.track)]
    for spec in args.interaction:
        names = spec.split(',')
        if len(names) != 4:
            raise SystemExit(f'--interaction needs four arms, got {spec!r}.')
        est = interaction(metric_results, *names)
        low, high = est.interval()
        sections += ['', f'## Interaction {names[3]} vs {names[1]} + {names[2]} over {names[0]}', '',
                     f'I = L_AB − L_A − L_B + L_base = {est.mean:+.5f} (n={est.n}, sd={_fmt(est.sd, 5)}, '
                     f'95% CI {"–" if math.isnan(low) else f"[{low:+.5f}, {high:+.5f}]"}). '
                     'Negative means the combination beats the sum of the single changes.']
    timing = timing_table(results)
    if timing:
        sections += ['', '## Step time by phase (ms; compilation and warm-up excluded)', '', timing]
    displacement = displacement_table(results)
    if displacement:
        sections += ['', '## c_proj displacement profile (per-layer means, schedule-normalized)', '',
                     displacement]
    strata = strata_table(results)
    if strata:
        sections += ['', '## Held-out loss by training support of the causal suffix', '', strata]
    diag = diag_table(args.root, args.track)
    if diag:
        sections += ['', '## Meta-gradient survival diagnostic', '', diag]
    report = '\n'.join(sections)
    print(report)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report + '\n')


if __name__ == '__main__':
    main()
