"""Tests of run identities, conflicts, shard planning, smoke checks and eligibility."""

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import analyze
import run_queue
import smoke

CPU = run_queue.Environment(device='cpu', processes=1)


class IdentityTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved_runs_dir = run_queue.RUNS_DIR
        run_queue.RUNS_DIR = os.path.join(self.tmp.name, 'runs')
        self.data = os.path.join(self.tmp.name, 'train.pt')
        with open(self.data, 'wb') as f:
            f.write(b'tokens')
        self.flags = ['--input_bin', self.data, '--input_val_bin', self.data]

    def tearDown(self):
        run_queue.RUNS_DIR = self.saved_runs_dir
        self.tmp.cleanup()

    def _spec(self, names=('A1',), extra=()):
        return run_queue.plan_runs(track='tiny', names=list(names), seeds=[0], prefix='t',
                                   extra_flags=self.flags + list(extra))[-1]

    def _complete(self, spec, identity):
        os.makedirs(spec.run_dir, exist_ok=True)
        with open(spec.result_path, 'w') as f:
            json.dump({'status': 'complete'}, f)
        if identity is not None:
            with open(os.path.join(spec.run_dir, run_queue.IDENTITY_FILE), 'w') as f:
                json.dump(identity, f)

    def test_completion_requires_a_matching_identity(self):
        spec = self._spec()
        self.assertEqual(spec.state(CPU)[0], 'pending')
        self._complete(spec, None)
        self.assertEqual(spec.state(CPU), ('conflict', 'complete result without an identity record'))
        self._complete(spec, spec.identity(CPU))
        self.assertEqual(spec.state(CPU)[0], 'complete')

    def test_device_process_count_flags_and_data_are_part_of_the_identity(self):
        spec = self._spec()
        self._complete(spec, spec.identity(CPU))
        state, reason = spec.state(run_queue.Environment(device='NVIDIA H100 80GB HBM3', processes=8))
        self.assertEqual(state, 'conflict')
        self.assertIn('device', reason)
        self.assertIn('processes', reason)
        changed = self._spec(extra=['--max-train-steps', '40'])
        self.assertEqual(changed.state(CPU), ('conflict', 'identity differs in flags'))
        with open(self.data, 'wb') as f:
            f.write(b'other tokens')
        self.assertEqual(spec.state(CPU), ('conflict', 'identity differs in data'))

    def test_identity_is_portable_between_checkouts(self):
        spec = self._spec(extra=[os.path.join(run_queue.REPO_DIR, 'fineweb_data', 'x.pt')])
        self.assertNotIn(run_queue.REPO_DIR, json.dumps(spec.identity(CPU)))

    def test_queue_refuses_conflicting_complete_runs(self):
        spec = self._spec()
        self._complete(spec, None)
        with self.assertRaises(run_queue.QueueConflict):
            run_queue.run_queue([spec], slots=[['cpu0']], environment=CPU, launcher='env')

    def test_cli_exits_with_the_conflict_code_and_records_a_summary(self):
        self._complete(self._spec(), None)
        argv = ['run_queue.py', '--track', 'tiny', '--arms', 'A1', '--prefix', 't', '--gpus', 'cpu0',
                '--gpus-per-run', '1', '--launcher', 'env', '--extra', ' '.join(self.flags)]
        with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                run_queue.main()
        self.assertEqual(raised.exception.code, run_queue.EXIT_CONFLICT)
        queue_dir = os.path.join(run_queue.RUNS_DIR, 't')
        [summary] = [name for name in os.listdir(queue_dir) if name.startswith('queue_summary_')]
        with open(os.path.join(queue_dir, summary)) as f:
            self.assertEqual(json.load(f)['exit_code'], run_queue.EXIT_CONFLICT)

    def test_cli_refuses_to_start_without_the_data(self):
        os.remove(self.data)
        argv = ['run_queue.py', '--track', 'tiny', '--arms', 'A1', '--prefix', 't', '--gpus', 'cpu0',
                '--gpus-per-run', '1', '--launcher', 'env', '--extra', ' '.join(self.flags)]
        with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()) as out:
            with mock.patch.object(run_queue, 'run_queue') as launch:
                with self.assertRaises(SystemExit) as raised:
                    run_queue.main()
        self.assertEqual(raised.exception.code, run_queue.EXIT_MISSING_DATA)
        launch.assert_not_called()
        self.assertIn('missing data', out.getvalue())

    def test_interrupted_attempts_are_kept_aside(self):
        spec = self._spec()
        os.makedirs(spec.run_dir)
        run_queue._set_aside(spec.run_dir)
        self.assertTrue(os.path.isdir(spec.run_dir + '.attempt-1'))
        self.assertFalse(os.path.exists(spec.run_dir))

    def test_outcome_fails_on_failed_runs_and_failed_sync(self):
        self.assertTrue(run_queue.QueueOutcome({'a': 'complete', 'b': 'skipped'}).ok)
        self.assertFalse(run_queue.QueueOutcome({'a': 'blocked'}).ok)
        self.assertFalse(run_queue.QueueOutcome({'a': 'complete'}, sync_ok=False).ok)


class LaunchEnvTest(unittest.TestCase):

    def test_google_multi_node_nccl_presets_are_removed(self):
        env = run_queue.single_node_env({
            'NCCL_NET': 'gIB', 'NCCL_IB_TC': '52', 'NCCL_P2P_NET_CHUNKSIZE': '131072', 'NCCL_DEBUG': 'WARN',
            'LD_LIBRARY_PATH': '/usr/local/gib/lib64:/opt/lib', 'PATH': '/usr/bin'})
        self.assertEqual(env, {'NCCL_DEBUG': 'WARN', 'LD_LIBRARY_PATH': '/opt/lib', 'PATH': '/usr/bin',
                               'NCCL_NET_PLUGIN': 'none'})
        self.assertNotIn('LD_LIBRARY_PATH', run_queue.single_node_env({'LD_LIBRARY_PATH': '/usr/local/gib/lib64:'}))


class ShardPlanTest(unittest.TestCase):

    def test_every_shard_schedules_the_producers_it_depends_on(self):
        specs = run_queue.plan_runs(track='tiny', names=['B-rad-x1', 'B-euc-rad'], seeds=[0, 1],
                                    prefix='t', shard=(1, 2))
        planned = [(s.arm, s.seed) for s in specs]
        self.assertEqual(planned, [('B-calib', 0), ('B-rad-x1', 0), ('B-rad-x1', 1), ('B-euc-rad', 1)])
        index = {(s.arm, s.seed): i for i, s in enumerate(specs)}
        consumer = specs[index[('B-euc-rad', 1)]]
        self.assertTrue(consumer.needs[0].endswith(os.path.join('B-rad-x1', 'seed0', 'mg_calibration.json')))
        self.assertLess(index[('B-rad-x1', 0)], index[('B-euc-rad', 1)])


def _result(**overrides):
    result = {
        'status': 'complete', 'track': 'tiny', 'steps': 30, 'val_loss': 3.3, 'best_val_loss': 3.29,
        'final_train_loss': 3.1, 'ema_val_loss': 3.31, 'ckpt_avg_val_loss': 3.29,
        'epochs': [{'val_loss': 3.5, 'train_probe_loss': 3.4}] * 3,
        'args': {'num_epochs': 3, 'mg_every': 1, 'aux_split': 'shared', 'diag_steps': '', 'spectral':
                 'polar_express', 'mona_beta': 0.0, 'credit': 'trunc', 'ngram': 'off'},
        'features': {'mg_steps': 30, 'diag_steps_run': 0},
        'mg': {},
    }
    result.update(overrides)
    return result


class SmokeCheckTest(unittest.TestCase):

    def test_a_clean_result_passes(self):
        self.assertEqual(smoke.result_problems(_result()), [])

    def test_non_finite_final_loss_fails_despite_a_finite_best(self):
        problems = smoke.result_problems(_result(val_loss=float('nan')))
        self.assertTrue(any('val_loss' in p for p in problems))

    def test_missing_aggregate_and_incomplete_schedule_fail(self):
        problems = smoke.result_problems(_result(ckpt_avg_val_loss=None, epochs=[{'val_loss': 3.5,
                                                                                  'train_probe_loss': 3.4}]))
        self.assertTrue(any('ckpt_avg_val_loss' in p for p in problems))
        self.assertTrue(any('epochs completed' in p for p in problems))

    def test_requested_features_must_leave_evidence(self):
        args = dict(_result()['args'], aux_split='adapt', diag_steps='5,27', spectral='eigh', mona_beta=0.98)
        problems = smoke.result_problems(_result(args=args, features={'mg_steps': 0, 'diag_steps_run': 1}))
        for expected in ('MG never ran', 'aux split never ran', '1 of 2 diagnostic steps ran',
                         'spectral map never ran', 'MONA never ran'):
            self.assertIn(expected, problems)

    def test_no_trace_check_compares_parameters_and_trajectory(self):
        base = _result(final_param_digest='a')
        self.assertEqual(smoke.trace_problems(_result(final_param_digest='a'), base), [])
        problems = smoke.trace_problems(_result(final_param_digest='b', epochs=[
            {'val_loss': 3.6, 'train_probe_loss': 3.4}] * 3), base)
        self.assertIn('final parameters differ from base', problems)
        self.assertIn('epoch trajectory differs from base', problems)


class EligibilityTest(unittest.TestCase):

    def _run(self, minutes, **overrides):
        run = {'world_size': 8, 'virtual_ranks': 8, 'env': {'gpu': 'NVIDIA H100 80GB HBM3'}, 'args': {},
               'total_training_time_s': minutes * 60, 'best_val_loss': 3.29, 'peak_memory_mib': 1024}
        run.update(overrides)
        return run

    def test_cap_is_checked_per_run(self):
        runs = {'base': {0: self._run(14.0), 1: self._run(16.0)}}
        table = analyze.arm_table(runs, track='tiny', metric='best_val_loss', reference='base')
        self.assertIn('| 1/2 | 1/2 |', table)
        self.assertEqual(analyze.eligibility_problems(runs['base'][0], 'tiny'), [])
        self.assertTrue(analyze.eligibility_problems(runs['base'][1], 'tiny')[0].startswith('16.0 min'))

    def test_hardware_schedule_and_diagnostics_are_checked(self):
        problems = analyze.eligibility_problems(
            self._run(10.0, world_size=1, env={'gpu': 'cpu'}, args={'diag_steps': '5', 'num_epochs': 3}),
            'tiny')
        self.assertEqual(len(problems), 4)

    def test_mixed_configurations_are_flagged(self):
        runs = {'base': {0: self._run(14.0), 1: self._run(14.0, world_size=1)}}
        table = analyze.arm_table(runs, track='tiny', metric='best_val_loss', reference='base')
        self.assertIn('mixes hardware configurations', table)


if __name__ == '__main__':
    unittest.main()
