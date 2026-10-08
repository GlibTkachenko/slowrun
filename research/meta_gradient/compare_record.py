"""Compares the first steps of a record trainer and its research fork on one stack.

Runs the original script (tiny/train.py or train.py) and the fork with default
flags and the same seed, one after the other on the same GPUs, stops each after
a number of steps and compares the per-step training losses and step times both
print. The fork emulates exactly as many ranks as the record runs with (one per
GPU), so both see the same batches and dropout masks. The tiny record only takes
its meta-gradient step without gradient accumulation, i.e. on 8 GPUs; on fewer it
silently trains without MG and logs only the last micro-batch loss, so the tiny
comparison needs 8 GPUs. With default flags the fork must reproduce the record; differences
beyond GPU nondeterminism (non-deterministic attention backward kernels) point to
a porting error. Run it once per track and stack before funding experiments.

    python research/meta_gradient/compare_record.py --track tiny --steps 30
    python research/meta_gradient/compare_record.py --track hour --steps 20
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import signal
import statistics
import subprocess
import sys

import run_queue

_STEP = re.compile(r'^step (\d+) .*\| loss: ([0-9.]+) \| dt: ([0-9.]+)ms')
RECORD_SCRIPTS = {'tiny': 'tiny/train.py', 'hour': 'train.py'}


def run_steps(command: list[str], *, ranks: int, steps: int, log_path: str,
              env: dict[str, str]) -> dict[int, tuple]:
    """Runs one process per rank of a trainer until rank 0 reports `steps` steps, then stops them.

    Processes are started directly with the rank variables torchrun would set, so a
    failing rank's own traceback reaches the log.

    Args:
        command: The trainer's command line (interpreter, script, flags).
        ranks: Number of processes, one per GPU.
        steps: Number of optimizer steps to observe.
        log_path: Where the trainers' output is copied.
        env: Environment of the run.

    Returns:
        Map from step to (smoothed training loss, step time in ms).

    Raises:
        RuntimeError: The trainer exited before reaching `steps`; the message ends with
            the last lines of its log.
    """
    seen: dict[int, tuple] = {}
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    port = str(run_queue.free_port())
    processes = []
    with open(log_path, 'w', buffering=1) as log:
        for rank in range(ranks):
            rank_env = dict(env, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(ranks),
                            LOCAL_WORLD_SIZE=str(ranks), MASTER_ADDR='127.0.0.1', MASTER_PORT=port,
                            PYTHONUNBUFFERED='1')
            processes.append(subprocess.Popen(
                command, cwd=run_queue.REPO_DIR, env=rank_env, text=True, start_new_session=True,
                stdout=subprocess.PIPE if rank == 0 else log, stderr=subprocess.STDOUT))
        for line in processes[0].stdout:
            log.write(line)
            match = _STEP.match(line.strip())
            if match:
                seen[int(match.group(1))] = (float(match.group(2)), float(match.group(3)))
                if len(seen) >= steps:
                    break
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    if len(seen) < steps:
        with open(log_path) as f:
            tail = ''.join(f.readlines()[-25:])
        raise RuntimeError(f'{command[1]} stopped after {len(seen)} steps; last lines of {log_path}:\n{tail}')
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--track', choices=tuple(RECORD_SCRIPTS), required=True)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', default=None, help='Comma-separated GPU ids (default: all visible)')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='Largest acceptable |difference| of the smoothed training loss')
    args = parser.parse_args()

    gpus = args.gpus.split(',') if args.gpus else run_queue.visible_gpus()
    if not gpus:
        raise SystemExit('No GPUs found.')
    if args.track == 'tiny' and len(gpus) < 8:
        raise SystemExit('The tiny record skips its meta-gradient step under gradient accumulation '
                         '(fewer than 8 GPUs), so it would train a different algorithm than the fork. '
                         'Run this comparison on 8 GPUs.')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    env = run_queue.single_node_env(dict(os.environ, WANDB_MODE='disabled', CUDA_VISIBLE_DEVICES=','.join(gpus)))
    fork = os.path.join('research', 'meta_gradient', f'{args.track}_train.py')
    traces = {}
    for label, script, extra in (('record', RECORD_SCRIPTS[args.track], []),
                                 ('fork', fork, ['--cleanup-checkpoints', '--virtual-ranks', str(len(gpus))])):
        run_name = f'compare/{args.track}_{stamp}/{label}'
        print(f'[compare] {label}: {script} for {args.steps} steps', flush=True)
        traces[label] = run_steps([sys.executable, script, '--run-name', run_name, '--seed', str(args.seed)] + extra,
                                  ranks=len(gpus), steps=args.steps, env=env,
                                  log_path=os.path.join(run_queue.RUNS_DIR, run_name, 'compare.log'))

    record, fork_trace = traces['record'], traces['fork']
    shared = sorted(set(record) & set(fork_trace))
    print('\n| step | record loss | fork loss | Δ | record ms | fork ms |\n|---|---|---|---|---|---|')
    for step in shared:
        (a, ta), (b, tb) = record[step], fork_trace[step]
        print(f'| {step} | {a:.6f} | {b:.6f} | {b - a:+.2e} | {ta:.0f} | {tb:.0f} |')
    worst = max(abs(fork_trace[s][0] - record[s][0]) for s in shared)
    timed = [s for s in shared if s > 3]
    ratio = statistics.fmean(fork_trace[s][1] for s in timed) / statistics.fmean(record[s][1] for s in timed)
    first_split = next((s for s in shared if abs(fork_trace[s][0] - record[s][0]) > 1e-5), None)
    print(f'\nmax |Δ loss| = {worst:.2e} (tolerance {args.tolerance:g}); '
          f'first step with |Δ| > 1e-5: {first_split}; fork/record step time = {ratio:.3f}')
    if worst > args.tolerance:
        print('[compare] FAIL: the fork does not reproduce the record on this stack.')
        sys.exit(1)
    print('[compare] PASS')


if __name__ == '__main__':
    main()
