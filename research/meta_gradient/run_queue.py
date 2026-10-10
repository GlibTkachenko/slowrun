"""Runs experiment arms over seeds on the GPUs of one machine.

Every run is one arm at one seed, launched with torchrun on a slot of GPUs.

Each run directory records the run's identity in identity.json: its resolved
flags, fingerprints of the training code, the data and any calibration inputs,
the device and the process count. A run is reused only when its result is
complete and its identity matches the one requested now. A complete run with a
different or missing identity is a conflict: the queue refuses to start unless
`--on-conflict replace` moves it aside. An interrupted run restarts from the
beginning; its partial directory is kept as <run>.attempt-N.

The exit status is 0 only if every run completed and every requested sync to
Cloud Storage succeeded (1: failed or blocked runs, 2: conflicts, 3: sync).

Examples, from the repository root:

    # Print the plan of a study without running anything.
    python research/meta_gradient/run_queue.py --track tiny --arms tiny-A --seeds 0,1 --dry-run

    # Record-faithful runs: one arm at a time on all eight GPUs.
    python research/meta_gradient/run_queue.py --track tiny --arms tiny-A --seeds 0,1 --gpus-per-run 8

    # Screening throughput: eight runs at once, one GPU each. The trainer emulates
    # the record's eight data-parallel ranks; keep these under their own prefix.
    python research/meta_gradient/run_queue.py --track tiny --arms tiny-A --seeds 0,1 --gpus-per-run 1 \\
        --prefix screen-1gpu
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import NoReturn

import arms as arms_lib

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(LAB_DIR, '..', '..'))
# The trainers write to runs/<run name> relative to the repository root.
RUNS_DIR = os.path.join(REPO_DIR, 'runs')

# Modules imported by the trainers; together with the trainer they define the code identity.
TRAINING_MODULES = ('credit.py', 'loader.py', 'lookahead.py', 'mona.py', 'ngram.py', 'probe.py',
                    'spectral.py')
DEFAULT_DATA = {'--input_bin': 'fineweb_data/fineweb_train.pt',
                '--input_val_bin': 'fineweb_data/fineweb_val.pt'}
IDENTITY_FILE = 'identity.json'

EXIT_RUNS_FAILED = 1
EXIT_CONFLICT = 2
EXIT_SYNC_FAILED = 3
EXIT_MISSING_DATA = 4


@dataclasses.dataclass(frozen=True)
class Environment:
    """Where runs execute; part of every run's identity.

    Attributes:
        device: 'cpu', or the GPU model name(s) of a slot.
        processes: Processes (GPUs) per run.
    """

    device: str
    processes: int


_HASHES: dict[tuple[str, int, int], str] = {}


def file_sha256(path: str) -> str | None:
    """Returns the SHA-256 of a file (cached by size and modification time), or None."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASHES:
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                digest.update(chunk)
        _HASHES[key] = digest.hexdigest()
    return _HASHES[key]


def code_fingerprint(track: str) -> str:
    """Returns a SHA-256 over the trainer of `track` and every module it imports."""
    paths = [os.path.join(LAB_DIR, arms_lib.TRAINERS[track])]
    paths += [os.path.join(LAB_DIR, name) for name in TRAINING_MODULES]
    if track == 'tiny':
        paths.append(os.path.join(REPO_DIR, 'tiny', 'cuda_kernels.py'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(os.path.relpath(path, REPO_DIR).encode())
        with open(path, 'rb') as f:
            digest.update(f.read())
    return digest.hexdigest()


def _portable(token: str) -> str:
    """Replaces this checkout's absolute path in a flag, so identities move between machines."""
    return token.replace(REPO_DIR + os.sep, '')


def _data_paths(flags: Sequence[str]) -> dict[str, str]:
    """Returns the data files a trainer will read, by flag."""
    paths = dict(DEFAULT_DATA)
    for i, flag in enumerate(flags[:-1]):
        if flag in paths:
            paths[flag] = flags[i + 1]
    return {flag: p if os.path.isabs(p) else os.path.join(REPO_DIR, p) for flag, p in paths.items()}


def missing_data(specs: Sequence['RunSpec']) -> list[str]:
    """Returns the data files that the planned runs read but that do not exist, sorted."""
    paths = {path for spec in specs for path in _data_paths(spec.flags).values()}
    return sorted(_portable(path) for path in paths if not os.path.isfile(path))


@dataclasses.dataclass
class RunSpec:
    """One planned run.

    Attributes:
        track: 'tiny' or 'hour'.
        arm: Arm name.
        seed: Model seed.
        run_name: Name under runs/ given to the trainer.
        flags: Trainer flags, references resolved.
        needs: Files that must exist before the run can start.
    """

    track: str
    arm: str
    seed: int
    run_name: str
    flags: list[str]
    needs: list[str]

    @property
    def run_dir(self) -> str:
        return os.path.join(RUNS_DIR, self.run_name)

    @property
    def result_path(self) -> str:
        return os.path.join(self.run_dir, 'result.json')

    def identity(self, environment: Environment) -> dict | None:
        """Returns what the run's result depends on, or None while an input file is missing."""
        needs = {}
        for path in self.needs:
            digest = file_sha256(path)
            if digest is None:
                return None
            needs[_portable(path)] = digest
        data = {flag: {'path': _portable(path), 'sha256': file_sha256(path)}
                for flag, path in _data_paths(self.flags).items()}
        return {
            'track': self.track,
            'arm': self.arm,
            'seed': self.seed,
            'flags': [_portable(f) for f in self.flags],
            'code': code_fingerprint(self.track),
            'data': data,
            'needs': needs,
            'device': environment.device,
            'processes': environment.processes,
        }

    def stored_identity(self) -> dict | None:
        """Returns the identity recorded when the run was launched, if any."""
        try:
            with open(os.path.join(self.run_dir, IDENTITY_FILE)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def result_status(self) -> str | None:
        """Returns the status field of result.json, if any."""
        try:
            with open(self.result_path) as f:
                return json.load(f).get('status')
        except (OSError, ValueError):
            return None

    def state(self, environment: Environment) -> tuple[str, str]:
        """Returns ('complete' | 'conflict' | 'pending', reason)."""
        if self.result_status() != 'complete':
            return 'pending', ''
        stored = self.stored_identity()
        if stored is None:
            return 'conflict', 'complete result without an identity record'
        current = self.identity(environment)
        if current is None:
            return 'conflict', 'a calibration input is missing, so the result cannot be verified'
        differing = sorted(k for k in current if stored.get(k) != current[k])
        if differing:
            return 'conflict', f'identity differs in {", ".join(differing)}'
        return 'complete', ''


class QueueConflict(Exception):
    """Complete runs exist whose identity differs from the requested runs."""

    def __init__(self, conflicts: Sequence[tuple[RunSpec, str]]):
        lines = [f'  {spec.run_name}: {reason}' for spec, reason in conflicts]
        super().__init__(
            'Refusing to reuse or overwrite complete runs with a different identity:\n'
            + '\n'.join(lines)
            + '\nUse a new --prefix for a new configuration, or --on-conflict replace to move '
              'these runs aside and rerun them.')
        self.conflicts = list(conflicts)


def run_name_for(prefix: str, track: str, arm: str, seed: int) -> str:
    """Returns the run name (relative to runs/) of an arm at a seed."""
    return f'{prefix}/{track}/{arm}/seed{seed}'


def plan_runs(
    *,
    track: str,
    names: Sequence[str],
    seeds: Sequence[int],
    prefix: str,
    ref_seed: int | None = None,
    extra_flags: Sequence[str] = (),
    shard: tuple[int, int] = (0, 1),
) -> list[RunSpec]:
    """Plans the runs of some arms over some seeds.

    Requested arms run at this shard's seeds. Every arm whose output another arm
    references also runs at the reference seed, on every shard that needs it, so a
    shard is self-contained; fetch finished producers first to avoid repeats.

    Args:
        track: The track.
        names: Arm names and study names.
        seeds: Seeds of the requested arms.
        prefix: Run-name prefix under runs/.
        ref_seed: Seed whose runs provide referenced files; defaults to the first seed.
        extra_flags: Flags appended to every run (override arm flags).
        shard: (index, count): keep seeds whose position modulo count equals index.

    Returns:
        The runs, ordered so that every producer precedes its consumers.
    """
    ref_seed = seeds[0] if ref_seed is None else ref_seed
    requested = set(arms_lib.requested_names(track, names))
    index, count = shard
    my_seeds = [s for i, s in enumerate(seeds) if i % count == index]
    ordered = arms_lib.expand(track, names)
    referenced = {ref for arm in ordered for ref, _ in arm.references()}
    specs = []
    for arm in ordered:
        needs: list[str] = []

        def locate(ref_arm: str, filename: str) -> str:
            path = os.path.join(RUNS_DIR, run_name_for(prefix, track, ref_arm, ref_seed), filename)
            needs.append(path)
            return path

        flags = arm.resolve(locate)
        run_seeds = list(my_seeds) if arm.name in requested else []
        if arm.name in referenced and ref_seed not in run_seeds:
            run_seeds.insert(0, ref_seed)
        for seed in run_seeds:
            specs.append(RunSpec(
                track=track,
                arm=arm.name,
                seed=seed,
                run_name=run_name_for(prefix, track, arm.name, seed),
                flags=list(flags) + list(extra_flags),
                needs=list(needs),
            ))
    return specs


def trainer_command(
    spec: RunSpec, *, gpus_per_run: int, launcher: str, wandb_group: str, python: str = sys.executable
) -> list[str]:
    """Returns the command line of one run.

    Args:
        spec: The run.
        gpus_per_run: Processes per run.
        launcher: 'torchrun', or 'env' to start processes directly (see `launch`).
        wandb_group: Weights & Biases group.
        python: Interpreter for the 'env' launcher.

    Returns:
        argv of the trainer (without the launcher prefix for 'env').
    """
    script = os.path.join(LAB_DIR, arms_lib.TRAINERS[spec.track])
    args = [script, '--run-name', spec.run_name, '--seed', str(spec.seed), '--arm', spec.arm,
            '--wandb_group', wandb_group, '--cleanup-checkpoints']
    if spec.track == 'tiny':
        args.append('--no-save-model')
    args += spec.flags
    if launcher == 'torchrun':
        return [python, '-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={gpus_per_run}'] + args
    return [python] + args


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def single_node_env(env: dict[str, str]) -> dict[str, str]:
    """Returns `env` without the multi-node NCCL presets of Google's GPU images.

    Deep Learning VM images preset NCCL for cluster networking (NCCL_NET=gIB and a
    network plugin on LD_LIBRARY_PATH whose checker rejects other settings). One
    machine needs no network plugin, and on a3-highgpu VMs the presets make every
    run exit at NCCL start-up without a message. NCCL_DEBUG settings are kept.
    """
    clean = {k: v for k, v in env.items() if not k.startswith('NCCL_') or k.startswith('NCCL_DEBUG')}
    paths = [p for p in clean.pop('LD_LIBRARY_PATH', '').split(':') if p and '/gib/' not in p + '/']
    if paths:
        clean['LD_LIBRARY_PATH'] = ':'.join(paths)
    clean['NCCL_NET_PLUGIN'] = 'none'
    return clean


class Launched:
    """Processes of one running spec."""

    def __init__(self, spec: RunSpec, processes: list[subprocess.Popen], log, slot: int):
        self.spec = spec
        self.processes = processes
        self.log = log
        self.slot = slot
        self.started = time.time()

    def poll(self) -> int | None:
        """Returns the exit code once every process has exited, else None."""
        codes = [p.poll() for p in self.processes]
        if any(c is None for c in codes):
            if any(c not in (None, 0) for c in codes):
                for p in self.processes:
                    if p.poll() is None:
                        p.terminate()
            return None
        self.log.close()
        return next((c for c in codes if c != 0), 0)


def _set_aside(run_dir: str) -> None:
    """Moves an earlier attempt of a run out of the way, keeping it for inspection."""
    attempt = 1
    while os.path.exists(f'{run_dir}.attempt-{attempt}'):
        attempt += 1
    shutil.move(run_dir, f'{run_dir}.attempt-{attempt}')


def launch(
    spec: RunSpec,
    *,
    gpus: Sequence[str],
    slot: int,
    launcher: str,
    wandb_group: str,
    environment: Environment,
    env_overrides: dict[str, str] | None = None,
) -> Launched:
    """Starts one run on the given GPUs, recording its identity first.

    The 'env' launcher (the default) sets RANK, LOCAL_RANK, WORLD_SIZE and MASTER_*
    itself and starts one Python process per rank, so each rank's errors reach the
    run's log; 'torchrun' is kept for hosts that need its rendezvous.

    Args:
        spec: The run.
        gpus: Device ids for CUDA_VISIBLE_DEVICES; on CPU, one entry per process.
        slot: Slot index (for logging).
        launcher: 'torchrun' or 'env'.
        wandb_group: Weights & Biases group.
        environment: Device and process count recorded in the identity.
        env_overrides: Extra environment variables.

    Returns:
        A handle to poll.
    """
    if os.path.exists(spec.run_dir):
        _set_aside(spec.run_dir)
    os.makedirs(spec.run_dir)
    with open(os.path.join(spec.run_dir, IDENTITY_FILE), 'w') as f:
        json.dump(spec.identity(environment), f, indent=2)
    log = open(os.path.join(spec.run_dir, 'launcher.log'), 'a', buffering=1)
    env = single_node_env(dict(os.environ))
    env.update(env_overrides or {})
    if all(g.isdigit() for g in gpus):
        env['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
    command = trainer_command(spec, gpus_per_run=len(gpus), launcher=launcher, wandb_group=wandb_group)
    log.write(f'# {datetime.datetime.now().isoformat()} slot {slot} gpus {",".join(gpus)}\n')
    log.write('# ' + shlex.join(command) + '\n')
    if launcher == 'torchrun':
        processes = [subprocess.Popen(command, cwd=REPO_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)]
    else:
        port = str(free_port())
        processes = []
        for rank in range(len(gpus)):
            rank_env = dict(env, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(len(gpus)),
                            MASTER_ADDR='127.0.0.1', MASTER_PORT=port)
            processes.append(subprocess.Popen(command, cwd=REPO_DIR, env=rank_env, stdout=log,
                                              stderr=subprocess.STDOUT))
    return Launched(spec, processes, log, slot)


@dataclasses.dataclass
class QueueOutcome:
    """Final state of a queue.

    Attributes:
        status: Per run name: 'complete', 'skipped' (reused), 'failed' or 'blocked'.
        sync_ok: False if any requested sync to Cloud Storage failed.
    """

    status: dict[str, str]
    sync_ok: bool = True

    @property
    def ok(self) -> bool:
        return self.sync_ok and all(v in ('complete', 'skipped') for v in self.status.values())

    def counts(self) -> dict[str, int]:
        return {k: sum(v == k for v in self.status.values())
                for k in ('complete', 'skipped', 'failed', 'blocked')}


def run_queue(
    specs: Sequence[RunSpec],
    *,
    slots: Sequence[Sequence[str]],
    environment: Environment,
    launcher: str = 'env',
    wandb_group: str = 'meta_gradient',
    env_overrides: dict[str, str] | None = None,
    retries: int = 0,
    sync: str | None = None,
    state_path: str | None = None,
    poll_seconds: float = 5.0,
    on_conflict: str = 'error',
) -> QueueOutcome:
    """Executes runs on GPU slots until none is left or ready.

    Args:
        specs: Planned runs, producers first.
        slots: GPU id lists; one run occupies one slot.
        environment: Device and process count of every slot.
        launcher: 'torchrun' or 'env'.
        wandb_group: Weights & Biases group.
        env_overrides: Extra environment variables for every run.
        retries: Extra attempts for a failed run.
        sync: Optional destination for `gcloud storage rsync` after every run.
        state_path: Optional JSON-lines file recording every finished attempt.
        poll_seconds: Polling interval.
        on_conflict: 'error' to refuse conflicting complete runs, 'replace' to rerun them.

    Returns:
        The outcome.

    Raises:
        QueueConflict: Conflicting complete runs exist and `on_conflict` is 'error'.
    """
    status: dict[str, str] = {}
    conflicts = []
    for spec in specs:
        state, reason = spec.state(environment)
        if state == 'complete':
            status[spec.run_name] = 'skipped'
        elif state == 'conflict':
            conflicts.append((spec, reason))
    if conflicts and on_conflict == 'error':
        raise QueueConflict(conflicts)
    outcome = QueueOutcome(status)
    pending = [s for s in specs if s.run_name not in status]
    attempts = {s.run_name: 0 for s in pending}
    running: list[Launched] = []
    free = list(range(len(slots)))
    failed_dirs: set[str] = set()
    while pending or running:
        for handle in list(running):
            code = handle.poll()
            if code is None:
                continue
            running.remove(handle)
            free.append(handle.slot)
            spec = handle.spec
            ok = code == 0 and spec.state(environment)[0] == 'complete'
            minutes = (time.time() - handle.started) / 60
            attempts[spec.run_name] += 1
            if ok:
                status[spec.run_name] = 'complete'
            elif attempts[spec.run_name] <= retries:
                pending.insert(0, spec)
            else:
                status[spec.run_name] = 'failed'
                failed_dirs.add(spec.run_dir)
            print(f'[queue] {spec.run_name}: {"complete" if ok else f"exit {code}"} '
                  f'after {minutes:.1f} min', flush=True)
            if state_path:
                with open(state_path, 'a') as f:
                    f.write(json.dumps({'run': spec.run_name, 'ok': ok, 'exit': code,
                                        'minutes': minutes, 'time': time.time()}) + '\n')
            if sync and not sync_runs(sync):
                outcome.sync_ok = False
        for spec in list(pending):
            if any(os.path.dirname(n) in failed_dirs for n in spec.needs):
                pending.remove(spec)
                status[spec.run_name] = 'blocked'
                print(f'[queue] {spec.run_name}: blocked by a failed dependency', flush=True)
        while free and pending:
            ready = next((s for s in pending if all(os.path.exists(n) for n in s.needs)), None)
            if ready is None:
                break
            pending.remove(ready)
            slot = free.pop(0)
            print(f'[queue] start {ready.run_name} on slot {slot} ({",".join(slots[slot])})', flush=True)
            running.append(launch(ready, gpus=slots[slot], slot=slot, launcher=launcher,
                                  wandb_group=wandb_group, environment=environment,
                                  env_overrides=env_overrides))
        if pending and not running and free:
            for spec in pending:
                status[spec.run_name] = 'blocked'
                print(f'[queue] {spec.run_name}: blocked, missing {spec.needs}', flush=True)
            break
        if running:
            time.sleep(poll_seconds)
    return outcome


def sync_runs(destination: str) -> bool:
    """Mirrors runs/ to a Cloud Storage prefix; returns whether it succeeded."""
    if shutil.which('gcloud') is None:
        print('[queue] sync failed: gcloud not found', flush=True)
        return False
    command = ['gcloud', 'storage', 'rsync', '--recursive', '--quiet', RUNS_DIR, destination]
    result = subprocess.run(command, cwd=REPO_DIR, capture_output=True, text=True)
    if result.returncode:
        print(f'[queue] sync failed: {result.stderr.strip()[:300]}', flush=True)
        return False
    return True


def check_sync(destination: str, queue_dir: str) -> bool:
    """Verifies before any GPU work that this machine can write to the destination."""
    if shutil.which('gcloud') is None:
        print('[queue] --sync needs the gcloud CLI', flush=True)
        return False
    os.makedirs(queue_dir, exist_ok=True)
    marker = os.path.join(queue_dir, '.sync_check')
    with open(marker, 'w') as f:
        f.write(datetime.datetime.now().isoformat())
    target = destination.rstrip('/') + '/.meta_gradient_sync_check'
    result = subprocess.run(['gcloud', 'storage', 'cp', marker, target], capture_output=True, text=True)
    if result.returncode:
        print(f'[queue] cannot write to {destination}: {result.stderr.strip()[:300]}', flush=True)
        return False
    return True


def write_manifest(path: str, *, argv: Sequence[str], specs: Sequence[RunSpec],
                   environment: Environment) -> None:
    """Records code, software, hardware and data identity of a queue invocation."""

    def capture(*command: str) -> str | None:
        try:
            return subprocess.run(command, cwd=REPO_DIR, capture_output=True, text=True,
                                  timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    tracks = sorted({s.track for s in specs})
    manifest = {
        'time': datetime.datetime.now().isoformat(),
        'argv': list(argv),
        'environment': dataclasses.asdict(environment),
        'code': {track: code_fingerprint(track) for track in tracks},
        'git_commit': capture('git', 'rev-parse', 'HEAD'),
        'git_dirty': bool(capture('git', 'status', '--porcelain')),
        'python': sys.version,
        'pip_freeze': capture(sys.executable, '-m', 'pip', 'freeze'),
        'nvidia_smi': capture('nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                              '--format=csv,noheader'),
        'data_sha256': {name: file_sha256(os.path.join(REPO_DIR, 'fineweb_data', name))
                        for name in ('fineweb_train.pt', 'fineweb_val.pt')},
        'runs': [dataclasses.asdict(s) for s in specs],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)


def visible_gpus() -> list[str]:
    """Returns the indices of the GPUs nvidia-smi reports."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def describe_device(gpus: Sequence[str]) -> str:
    """Returns 'cpu' for CPU slots, else the GPU model name(s) of the given indices."""
    if not all(g.isdigit() for g in gpus):
        return 'cpu'
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader', '-i', ','.join(gpus)],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return 'cuda'
    names = sorted({line.strip() for line in out.splitlines() if line.strip()})
    return ' + '.join(names) or 'cuda'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--track', choices=arms_lib.TRACKS, required=True)
    parser.add_argument('--arms', required=True,
                        help='Comma-separated arm or study names (see arms.py), or "all"')
    parser.add_argument('--seeds', default='0', help='Comma-separated model seeds')
    parser.add_argument('--ref-seed', type=int, default=None,
                        help='Seed whose runs provide calibration files (default: first seed)')
    parser.add_argument('--prefix', default='lab', help='Runs go to runs/<prefix>/<track>/<arm>/seed<s>')
    parser.add_argument('--gpus', default=None, help='Comma-separated GPU ids (default: all visible)')
    parser.add_argument('--gpus-per-run', type=int, default=8)
    parser.add_argument('--launcher', choices=('env', 'torchrun'), default='env')
    parser.add_argument('--shard', default='0/1', help='i/n: run every n-th seed starting at i')
    parser.add_argument('--extra', default='', help='Flags appended to every run, e.g. "--max-train-steps 40"')
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--on-conflict', choices=('error', 'replace'), default='error',
                        help='What to do with complete runs whose identity differs (default: refuse)')
    parser.add_argument('--wandb-group', default=None)
    parser.add_argument('--sync', default=None, help='gs:// prefix to mirror runs/ after every run')
    parser.add_argument('--shutdown-when-done', action='store_true',
                        help='Power the VM off when the queue finishes (stops GPU billing)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    shard = tuple(int(v) for v in args.shard.split('/'))
    specs = plan_runs(track=args.track, names=args.arms.split(','), seeds=seeds, prefix=args.prefix,
                      ref_seed=args.ref_seed, extra_flags=shlex.split(args.extra), shard=shard)
    gpus = args.gpus.split(',') if args.gpus else visible_gpus()
    if not gpus:
        raise SystemExit('No GPUs found; pass --gpus explicitly.')
    if len(gpus) % args.gpus_per_run:
        raise SystemExit(f'{len(gpus)} GPUs cannot be split into slots of {args.gpus_per_run}.')
    slots = [gpus[i:i + args.gpus_per_run] for i in range(0, len(gpus), args.gpus_per_run)]
    environment = Environment(device=describe_device(slots[0]), processes=args.gpus_per_run)
    group = args.wandb_group or f'{args.prefix}-{args.track}'

    print(f'[queue] {len(specs)} runs on {len(slots)} slot(s) of {args.gpus_per_run} '
          f'({environment.device})')
    my_seeds = {s for i, s in enumerate(seeds) if i % shard[1] == shard[0]}
    for spec in specs:
        state, reason = spec.state(environment)
        label = {'complete': 'done', 'pending': 'todo', 'conflict': 'CONFLICT'}[state]
        note = '' if spec.seed in my_seeds else '  (calibration producer outside this shard)'
        print(f'  [{label}] {spec.run_name}  {" ".join(spec.flags)}{note}'
              + (f'\n         {reason}' if reason else ''))
    if args.dry_run:
        for spec in specs[:3]:
            print('  $ ' + shlex.join(trainer_command(spec, gpus_per_run=args.gpus_per_run,
                                                      launcher=args.launcher, wandb_group=group)))
        return

    queue_dir = os.path.join(RUNS_DIR, args.prefix)
    os.makedirs(queue_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    def finish(exit_code: int, summary: dict, *, mirror: bool) -> NoReturn:
        """Records the outcome, mirrors it if the bucket works, powers off if asked, and exits."""
        with open(os.path.join(queue_dir, f'queue_summary_{args.track}_{stamp}.json'), 'w') as f:
            json.dump(dict(summary, exit_code=exit_code), f, indent=2)
        if args.sync and mirror:
            sync_runs(args.sync)
        if args.shutdown_when_done:
            # Stop GPU billing whatever happened; the summary stays on the boot disk.
            subprocess.run(['sudo', 'shutdown', '-h', 'now'])
        sys.exit(exit_code)

    missing = missing_data(specs)
    if missing:
        print(f'[queue] missing data: {", ".join(missing)}', flush=True)
        finish(EXIT_MISSING_DATA, {'error': 'missing data', 'missing': missing}, mirror=False)
    if args.sync and not check_sync(args.sync, queue_dir):
        finish(EXIT_SYNC_FAILED, {'error': f'cannot write to {args.sync}'}, mirror=False)
    write_manifest(os.path.join(queue_dir, f'manifest_{args.track}_{stamp}.json'), argv=sys.argv,
                   specs=specs, environment=environment)
    threads = max(1, (os.cpu_count() or 8) // len(slots) // max(args.gpus_per_run, 1))
    try:
        outcome = run_queue(specs, slots=slots, environment=environment, launcher=args.launcher,
                            wandb_group=group, env_overrides={'OMP_NUM_THREADS': str(threads)},
                            retries=args.retries, sync=args.sync, on_conflict=args.on_conflict,
                            state_path=os.path.join(queue_dir, 'queue_state.jsonl'))
    except QueueConflict as error:
        print(f'[queue] {error}', flush=True)
        finish(EXIT_CONFLICT, {'error': str(error)}, mirror=True)
    if args.sync and not sync_runs(args.sync):
        outcome.sync_ok = False
    print(f'[queue] finished: {outcome.counts()}' + ('' if outcome.sync_ok else '; SYNC FAILED'))
    if outcome.ok:
        exit_code = 0
    elif any(v in ('failed', 'blocked') for v in outcome.status.values()):
        exit_code = EXIT_RUNS_FAILED
    else:
        exit_code = EXIT_SYNC_FAILED
    finish(exit_code, {'status': outcome.status, 'counts': outcome.counts(), 'sync_ok': outcome.sync_ok},
           mirror=outcome.sync_ok)


if __name__ == '__main__':
    main()
