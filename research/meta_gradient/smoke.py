"""End-to-end smoke tests of the experiment arms.

Every arm runs through the complete schedule of its track (recurrence switch or
layer replay, SWA cycles, MTP annealing, checkpoint averaging) on a dataset
small enough that a run takes minutes, then the results are checked: finite
final and aggregate losses, every epoch completed, and evidence that each
requested feature actually ran.

    # Laptop: CPU, synthetic data, a small model, every arm.
    python research/meta_gradient/smoke.py --device cpu --track both

    # GPU node, before full runs: real kernels and model, reduced data.
    python research/meta_gradient/smoke.py --device cuda --track tiny --processes 8
    python research/meta_gradient/smoke.py --device cuda --track hour --processes 8

Results go to runs/smoke-<device><processes>/ by default, and run identities
(code, flags, data, device, process count) prevent a GPU check from reusing CPU
results. On CPU the smoke test also checks that the survival diagnostic and the
calibration logging leave no trace: their runs must end with exactly the
parameters and losses of base.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

import torch

import run_queue
from loader import write_synthetic_dataset

DATA_DIR = os.path.join(run_queue.RUNS_DIR, 'smoke-data')

# Arms that together exercise every code path, for the (paid) GPU smoke test.
GPU_ARMS = {
    'tiny': ('base', 'A1', 'B-calib', 'B-met-x1', 'B-euc-met', 'C-diag', 'C-inv', 'C-reg-aug',
             'mona-mg', 'eqr', 'D-r25c-nomg', 'D-r25c', 'E-shrink'),
    'hour': ('base', 'A1', 'B-calib', 'B-met-x1', 'B-euc-met', 'C-diag', 'C-reg-aug', 'mona-mg'),
}
# Runs that must reproduce base exactly on CPU (they only add logging or restored probes).
NO_TRACE_ARMS = ('C-diag', 'B-calib')

_CPU_MODEL = ['--n_layer', '4', '--n_embd', '128', '--n_head', '2', '--seq-len', '64',
              '--eval-tokens', '1024', '--param-digest']
CPU_FLAGS = {
    'tiny': _CPU_MODEL + ['--device-batch-size', '4', '--total-batch-size', '2048',
                          '--num-epochs', '3', '--wd-phase1-epoch', '1', '--wd-phase2-epoch', '2',
                          '--swa-last-epochs', '2', '--update-ema-every', '2', '--max-train-steps', '0',
                          '--ngram-table-size', '4099'],
    'hour': _CPU_MODEL + ['--device-batch-size', '2', '--total-batch-size', '2048',
                          '--num-epochs', '3', '--dupe-start-epoch', '2', '--dupe-layers-start', '2',
                          '--dupe-layers-end', '3', '--dupe-loops', '1', '--swa-last-epochs', '1',
                          '--logit-avg', '2'],
}
CUDA_FLAGS = {
    'tiny': ['--eval-tokens', '2097152', '--max-train-steps', '0'],
    'hour': ['--eval-tokens', '2097152'],
}
# Diagnostic steps that fall inside the short smoke schedules (arms with --diag-steps only).
DIAG_STEPS = {('cpu', 'tiny'): '5,27', ('cpu', 'hour'): '5,28',
              ('cuda', 'tiny'): '10,50', ('cuda', 'hour'): '10,40'}


def _data_flags(train: str, val: str) -> list[str]:
    return ['--input_bin', train, '--input_val_bin', val]


def synthetic_data() -> tuple[str, str]:
    """Writes (once) a synthetic train/validation pair sharing one Markov chain."""
    train, val = os.path.join(DATA_DIR, 'train.pt'), os.path.join(DATA_DIR, 'val.pt')
    if not os.path.exists(train):
        write_synthetic_dataset(train, num_tokens=24_000, seed=1)
        write_synthetic_dataset(val, num_tokens=6_000, seed=2)
    return train, val


def reduced_real_data(tokens: int) -> tuple[str, str]:
    """Cuts the first documents of the real training set into a small file."""
    source = os.path.join(run_queue.REPO_DIR, 'fineweb_data', 'fineweb_train.pt')
    target = os.path.join(DATA_DIR, f'fineweb_train_{tokens}.pt')
    if not os.path.exists(target):
        data = torch.load(source, weights_only=True)
        starts = data['doc_starts']
        later = starts[starts >= tokens]
        cut = int(later[0]) if len(later) else data['tokens'].numel()
        os.makedirs(os.path.dirname(target), exist_ok=True)
        torch.save({**data, 'tokens': data['tokens'][:cut].clone(), 'doc_starts': starts[starts < cut].clone()},
                   target)
    return target, os.path.join(run_queue.REPO_DIR, 'fineweb_data', 'fineweb_val.pt')


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def result_problems(result: dict) -> list[str]:
    """Returns what is wrong with one completed result (empty if it passes).

    Checks finite final, best and aggregate losses, a finite trajectory, every
    epoch completed, and evidence that each requested feature ran.

    Args:
        result: A parsed result.json.

    Returns:
        Human-readable problems.
    """
    problems = []
    args, features = result.get('args', {}), result.get('features', {})
    for key in ('val_loss', 'best_val_loss', 'final_train_loss'):
        if not _finite(result.get(key)):
            problems.append(f'{key} is {result.get(key)!r}')
    aggregates = (('ema_val_loss', 'ckpt_avg_val_loss') if result.get('track') == 'tiny'
                  else ('logit_avg_equal_loss', 'logit_avg_weighted_loss'))
    for key in aggregates:
        if not _finite(result.get(key)):
            problems.append(f'{key} is {result.get(key)!r}')
    epochs = result.get('epochs', [])
    if len(epochs) != args.get('num_epochs'):
        problems.append(f'{len(epochs)} of {args.get("num_epochs")} epochs completed')
    if not all(_finite(e.get('val_loss')) and _finite(e.get('train_probe_loss')) for e in epochs):
        problems.append('non-finite epoch loss')

    def require(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    if args.get('mg_every', 0) > 0:
        require(features.get('mg_steps', 0) > 0, 'MG never ran')
    if args.get('aux_split', 'shared') != 'shared':
        require(features.get('aux_split_steps', 0) > 0, 'aux split never ran')
    requested_diag = [int(s) for s in str(args.get('diag_steps', '')).split(',') if s.strip()]
    requested_diag = [s for s in requested_diag if s < result.get('steps', 0)]
    require(features.get('diag_steps_run', 0) == len(requested_diag),
            f'{features.get("diag_steps_run", 0)} of {len(requested_diag)} diagnostic steps ran')
    if args.get('spectral', 'polar_express') != 'polar_express':
        require(features.get('spectral_calls', 0) > 0, 'spectral map never ran')
    if args.get('mona_beta', 0) > 0:
        require(features.get('mona_calls', 0) > 0, 'MONA never ran')
    if args.get('mg_act_stats') or args.get('mg_cproj') in ('act_radius', 'act_metric'):
        require(bool(result.get('mg', {}).get('calibration')), 'no c_proj displacement statistics')
    if result.get('track') == 'tiny':
        if args.get('credit', 'trunc') != 'trunc':
            require(features.get('multi_pass_steps', 0) > 0, 'no multi-pass step for recurrent credit')
        if args.get('credit') == 'full':
            require(features.get('credit_full_steps') == features.get('multi_pass_steps'),
                    'full credit skipped multi-pass steps')
        if args.get('ngram') in ('unigram', 'hashed'):
            require(features.get('ngram_params', 0) > 0, 'n-gram memory missing')
        if args.get('ngram') == 'dense':
            require(features.get('dense_params', 0) > 0, 'dense control missing')
    return problems


def trace_problems(result: dict, base: dict) -> list[str]:
    """Returns differences between a no-trace run and base (CPU runs are deterministic)."""
    problems = []
    if result.get('final_param_digest') != base.get('final_param_digest'):
        problems.append('final parameters differ from base')
    for key in ('final_train_loss', 'val_loss', 'best_val_loss', 'ema_val_loss', 'ckpt_avg_val_loss',
                'logit_avg_equal_loss', 'logit_avg_weighted_loss'):
        if result.get(key) != base.get(key):
            problems.append(f'{key} {result.get(key)!r} != base {base.get(key)!r}')
    trajectory = [(e['val_loss'], e['train_probe_loss']) for e in result.get('epochs', [])]
    if trajectory != [(e['val_loss'], e['train_probe_loss']) for e in base.get('epochs', [])]:
        problems.append('epoch trajectory differs from base')
    return problems


def check(specs, outcome: run_queue.QueueOutcome, *, device: str) -> list[str]:
    """Returns human-readable problems found in the smoke results."""
    problems, results = [], {}
    for spec in specs:
        if outcome.status.get(spec.run_name) not in ('complete', 'skipped'):
            problems.append(f'{spec.run_name}: {outcome.status.get(spec.run_name)} '
                            f'(see {spec.run_dir}/launcher.log)')
            continue
        with open(spec.result_path) as f:
            result = json.load(f)
        results[(spec.track, spec.arm, spec.seed)] = result
        problems += [f'{spec.run_name}: {p}' for p in result_problems(result)]
    if device == 'cpu':
        for (track, arm, seed), result in results.items():
            base = results.get((track, 'base', seed))
            if arm in NO_TRACE_ARMS and base is not None:
                problems += [f'{track} {arm}: {p}' for p in trace_problems(result, base)]
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--track', choices=('tiny', 'hour', 'both'), default='both')
    parser.add_argument('--arms', default=None,
                        help='Comma-separated arms; default: all on CPU, the "gpu" subset on CUDA')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--processes', type=int, default=1,
                        help='Processes per run (CPU: >1 tests sharding with gloo)')
    parser.add_argument('--parallel', type=int, default=None,
                        help='Concurrent runs (default: CPU 4, CUDA as many slots as GPUs allow)')
    parser.add_argument('--gpus', default=None, help='CUDA: comma-separated GPU ids (default: all)')
    parser.add_argument('--real-tokens', type=int, default=2_500_000,
                        help='CUDA: training tokens kept from fineweb_train.pt')
    parser.add_argument('--prefix', default=None,
                        help='Runs go to runs/<prefix>/<track>/... (default: smoke-<device><processes>)')
    parser.add_argument('--fresh', action='store_true', help='Delete previous runs under the prefix first')
    args = parser.parse_args()

    prefix = args.prefix or f'smoke-{args.device}{args.processes}'
    if args.fresh:
        shutil.rmtree(os.path.join(run_queue.RUNS_DIR, prefix), ignore_errors=True)
    tracks = ('tiny', 'hour') if args.track == 'both' else (args.track,)
    env = {'WANDB_MODE': 'disabled'}
    if args.device == 'cpu':
        train, val = synthetic_data()
        env['TORCH_COMPILE_DISABLE'] = '1'
        # Bind gloo to loopback: hostname lookup fails on some laptops (e.g. after network changes).
        env['GLOO_SOCKET_IFNAME'] = 'lo0' if sys.platform == 'darwin' else 'lo'
        groups = [[f'cpu{i}' for i in range(args.processes)] for _ in range(args.parallel or 4)]
        launcher = 'env'
    else:
        train, val = reduced_real_data(args.real_tokens)
        gpus = args.gpus.split(',') if args.gpus else run_queue.visible_gpus()
        groups = [gpus[i:i + args.processes] for i in range(0, len(gpus), args.processes)]
        groups = [g for g in groups if len(g) == args.processes]
        groups = groups[:args.parallel] if args.parallel else groups
        launcher = 'env'
    if not groups:
        raise SystemExit('No complete slot of devices for the requested --processes.')
    environment = run_queue.Environment(device=run_queue.describe_device(groups[0]),
                                        processes=args.processes)
    env['OMP_NUM_THREADS'] = str(max(1, (os.cpu_count() or 4) // len(groups) // args.processes))

    all_specs, problems = [], []
    for track in tracks:
        names = args.arms.split(',') if args.arms else (
            ['all'] if args.device == 'cpu' else list(GPU_ARMS[track]))
        if names == ['gpu']:
            names = list(GPU_ARMS[track])
        flags = (CPU_FLAGS if args.device == 'cpu' else CUDA_FLAGS)[track] + _data_flags(train, val)
        specs = run_queue.plan_runs(track=track, names=names, seeds=[args.seed], prefix=prefix,
                                    extra_flags=flags)
        for spec in specs:
            if '--diag-steps' in spec.flags:
                spec.flags += ['--diag-steps', DIAG_STEPS[(args.device, track)]]
        print(f'[smoke] {track}: {len(specs)} runs on {len(groups)} slot(s) ({environment.device}) '
              f'under runs/{prefix}')
        try:
            outcome = run_queue.run_queue(specs, slots=groups, environment=environment, launcher=launcher,
                                          wandb_group='smoke', env_overrides=env, poll_seconds=1.0)
        except run_queue.QueueConflict as error:
            print(f'[smoke] {error}')
            sys.exit(run_queue.EXIT_CONFLICT)
        all_specs += specs
        problems += check(specs, outcome, device=args.device)
    print()
    for spec in all_specs:
        try:
            with open(spec.result_path) as f:
                result = json.load(f)
            print(f'  {spec.track:4s} {spec.arm:14s} best_val_loss {result["best_val_loss"]:.5f}  '
                  f'final {result["val_loss"]:.5f}  train {result["total_training_time_s"] / 60:.2f} min')
        except (OSError, ValueError, KeyError, TypeError):
            print(f'  {spec.track:4s} {spec.arm:14s} FAILED')
    if problems:
        print('\n[smoke] PROBLEMS:\n  ' + '\n  '.join(problems))
        sys.exit(1)
    print(f'\n[smoke] all {len(all_specs)} runs passed')


if __name__ == '__main__':
    main()
