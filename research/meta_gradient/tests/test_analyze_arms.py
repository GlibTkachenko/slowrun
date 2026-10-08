"""Tests of the paired statistics and of the arm registry."""

import math
import os
import re
import unittest

import analyze
import arms
import run_queue

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StatisticsTest(unittest.TestCase):

    def test_t_quantiles(self):
        self.assertEqual(analyze.t_quantile(3), 3.182)
        self.assertTrue(2.086 > analyze.t_quantile(22) > 2.060)
        self.assertTrue(math.isnan(analyze.t_quantile(0)))

    def test_paired_estimate_matches_a_worked_example(self):
        # Four pairs with sd 0.002 give a half-width of 3.182 * 0.002 / 2 = 0.003182.
        values = [-0.002 + d for d in (-0.0016330, 0.0016330, -0.0016330, 0.0016330)]
        est = analyze.paired_estimate(values)
        self.assertAlmostEqual(est.mean, -0.002, places=9)
        self.assertAlmostEqual(est.sd, 0.0018856, places=6)
        self.assertAlmostEqual(est.half_width, 3.182 * est.sd / 2, places=12)

    def test_interaction(self):
        results = {'base': {0: 3.30, 1: 3.31}, 'A': {0: 3.29, 1: 3.30},
                   'B': {0: 3.295, 1: 3.305}, 'AB': {0: 3.28, 1: 3.29}}
        est = analyze.interaction(results, 'base', 'A', 'B', 'AB')
        self.assertAlmostEqual(est.mean, -0.005, places=9)
        with self.assertRaises(ValueError):
            analyze.interaction({'base': {0: 1.0}, 'A': {1: 1.0}, 'B': {0: 1.0}, 'AB': {0: 1.0}},
                                'base', 'A', 'B', 'AB')

    def test_verdicts(self):
        self.assertEqual(analyze._verdict(analyze.paired_estimate([-0.012, -0.011, -0.013])), 'substantial')
        self.assertEqual(analyze._verdict(analyze.paired_estimate([-0.004, -0.0041, -0.0039])), 'useful')
        self.assertEqual(analyze._verdict(analyze.paired_estimate([0.004, 0.0041, 0.0039])), 'worse')
        self.assertEqual(analyze._verdict(analyze.paired_estimate([-0.01, 0.01])), 'inconclusive')


class ArmRegistryTest(unittest.TestCase):

    def test_every_flag_exists_in_its_trainer(self):
        for track, trainer in arms.TRAINERS.items():
            with open(os.path.join(LAB_DIR, trainer)) as f:
                source = f.read()
            for arm in arms.ARMS[track].values():
                for flag in arm.flags:
                    if flag.startswith('--'):
                        with self.subTest(track=track, arm=arm.name, flag=flag):
                            self.assertRegex(source, re.escape(f'"{flag}"'))

    def test_references_point_to_known_arms_and_resolve(self):
        for track in arms.TRACKS:
            for arm in arms.ARMS[track].values():
                for ref, filename in arm.references():
                    self.assertIn(ref, arms.ARMS[track])
                    self.assertEqual(filename, 'mg_calibration.json')
                resolved = ' '.join(arm.resolve(lambda a, f: f'/x/{a}/{f}'))
                self.assertNotIn('{ref:', resolved)

    def test_expand_orders_producers_first(self):
        ordered = [a.name for a in arms.expand('tiny', ['AB-met', 'B-euc-met'])]
        self.assertLess(ordered.index('B-calib'), ordered.index('AB-met'))
        self.assertLess(ordered.index('B-met-x1'), ordered.index('B-euc-met'))
        for study, (track, members) in arms.STUDIES.items():
            for name in members:
                arms.get_arm(track, name)

    def test_plan_runs_dependencies_only_at_reference_seed(self):
        specs = run_queue.plan_runs(track='tiny', names=['B-euc-met'], seeds=[3, 4], prefix='t')
        by_arm = {}
        for spec in specs:
            by_arm.setdefault(spec.arm, []).append(spec.seed)
        self.assertEqual(by_arm, {'B-calib': [3], 'B-met-x1': [3], 'B-euc-met': [3, 4]})
        consumer = [s for s in specs if s.arm == 'B-euc-met'][1]
        self.assertTrue(consumer.needs[0].endswith(os.path.join('t', 'tiny', 'B-met-x1', 'seed3',
                                                                'mg_calibration.json')))

    def test_plan_runs_shards_by_seed(self):
        shard0 = run_queue.plan_runs(track='tiny', names=['A1'], seeds=[0, 1, 2], prefix='t', shard=(0, 2))
        shard1 = run_queue.plan_runs(track='tiny', names=['A1'], seeds=[0, 1, 2], prefix='t', shard=(1, 2))
        self.assertEqual([s.seed for s in shard0], [0, 2])
        self.assertEqual([s.seed for s in shard1], [1])


if __name__ == '__main__':
    unittest.main()
