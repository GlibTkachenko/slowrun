"""Tests of the MG displacement, its record equivalence and the activation metric."""

import unittest

import torch
import torch.nn as nn

import lookahead


def _toy_model(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 10, bias=False), nn.Tanh(), nn.Linear(10, 10, bias=False),
                         nn.Tanh(), nn.Linear(10, 3, bias=False))


def _loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x).square().mean()


def _record_mg_step(model, plastic, xa, xb, step_norm):
    """The record's split step, transcribed from tiny/train.py (one rank owns everything)."""
    model.zero_grad(set_to_none=True)
    (_loss(model, xa) * 0.5).backward()
    with torch.no_grad():
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in plastic]
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))).clamp_min(1e-12)
        alpha = -step_norm / float(norm)
        kept = [g.clone() for g in grads]
        torch._foreach_add_(plastic, grads, alpha=alpha)
    (_loss(model, xb) * 0.5).backward()
    with torch.no_grad():
        torch._foreach_add_(plastic, kept, alpha=-alpha)


class RecordEquivalenceTest(unittest.TestCase):

    def test_single_virtual_rank_matches_record_bitwise(self):
        xa, xb = torch.randn(8, 6), torch.randn(8, 6)
        record = _toy_model(0)
        plastic = [record[0].weight, record[2].weight]
        _record_mg_step(record, plastic, xa, xb, 0.5)

        model = _toy_model(0)
        names = ['0.weight', '2.weight']
        look = lookahead.Lookahead(names, [model[0].weight, model[2].weight],
                                   config=lookahead.LookaheadConfig(step_norm=0.5))
        model.zero_grad(set_to_none=True)
        look.stash()
        (_loss(model, xa) * 0.5).backward()
        look.displace(1.0)
        (_loss(model, xb) * 0.5).backward()
        look.restore()
        for p, q in zip(record.parameters(), model.parameters()):
            self.assertTrue(torch.equal(p, q))
            self.assertTrue(torch.equal(p.grad, q.grad))

    def test_virtual_ranks_are_isolated(self):
        batches = [(torch.randn(8, 6), torch.randn(8, 6)) for _ in range(2)]
        # Reference: each virtual rank in a fresh process, gradients summed afterwards.
        separate = []
        for xa, xb in batches:
            model = _toy_model(0)
            look = lookahead.Lookahead(['0.weight', '2.weight'], [model[0].weight, model[2].weight],
                                       config=lookahead.LookaheadConfig(step_norm=0.5))
            model.zero_grad(set_to_none=True)
            look.stash()
            (_loss(model, xa) * 0.25).backward()
            look.displace(1.0)
            (_loss(model, xb) * 0.25).backward()
            look.restore()
            separate.append([p.grad.clone() for p in model.parameters()])
        model = _toy_model(0)
        look = lookahead.Lookahead(['0.weight', '2.weight'], [model[0].weight, model[2].weight],
                                   config=lookahead.LookaheadConfig(step_norm=0.5))
        model.zero_grad(set_to_none=True)
        for xa, xb in batches:
            look.stash()
            (_loss(model, xa) * 0.25).backward()
            look.displace(1.0)
            (_loss(model, xb) * 0.25).backward()
            look.restore()
        for i, p in enumerate(model.parameters()):
            torch.testing.assert_close(p.grad, separate[0][i] + separate[1][i], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(model[0].weight, _toy_model(0)[0].weight, rtol=0, atol=1e-6)

    def test_zero_step_norm_leaves_parameters_exact(self):
        model = _toy_model(0)
        before = [p.detach().clone() for p in model.parameters()]
        look = lookahead.Lookahead(['0.weight'], [model[0].weight],
                                   config=lookahead.LookaheadConfig(step_norm=0.0))
        look.stash()
        _loss(model, torch.randn(4, 6)).backward()
        look.displace(1.0)
        _loss(model, torch.randn(4, 6)).backward()
        look.restore()
        for p, q in zip(before, model.parameters()):
            self.assertTrue(torch.equal(p, q))


class ActivationMetricTest(unittest.TestCase):

    def _setup(self, mode, damping=0.1, radius=0.3, layer_radii=()):
        torch.manual_seed(3)
        weight = nn.Parameter(torch.randn(5, 7))
        weight.grad = torch.randn(5, 7)
        moment = torch.rand(7) + 0.05
        count = torch.tensor(1e6)  # bias correction ~ 1
        config = lookahead.LookaheadConfig(step_norm=0.5, cproj=mode, act_radius=radius,
                                           act_damping=damping, layer_radii=layer_radii)
        look = lookahead.Lookahead(['h.0.mlp.c_proj.weight'], [weight], config=config,
                                   moments={'h.0.mlp.c_proj.weight': (moment, count)})
        return weight, moment, look

    def _displacement(self, mode, **kwargs):
        weight, moment, look = self._setup(mode, **kwargs)
        grad, before = weight.grad.clone(), weight.detach().clone()
        look.displace(1.0)
        return weight.detach() - before, grad, moment

    def test_functional_radius_is_exact_in_both_modes(self):
        for mode in ('act_radius', 'act_metric'):
            d, _, c = self._displacement(mode)
            with self.subTest(mode=mode):
                self.assertAlmostEqual(float((d.square() * c).sum().sqrt()), 0.3, places=5)

    def test_radius_mode_keeps_gradient_direction(self):
        d, g, _ = self._displacement('act_radius')
        cosine = torch.nn.functional.cosine_similarity(d.flatten(), -g.flatten(), dim=0)
        self.assertAlmostEqual(float(cosine), 1.0, places=5)

    def test_undamped_metric_direction_is_g_c_inverse(self):
        d, g, c = self._displacement('act_metric', damping=0.0)
        expected = -(g / c)
        cosine = torch.nn.functional.cosine_similarity(d.flatten(), expected.flatten(), dim=0)
        self.assertAlmostEqual(float(cosine), 1.0, places=5)

    def test_strong_damping_recovers_radius_mode(self):
        damped, _, _ = self._displacement('act_metric', damping=1e7)
        radius, _, _ = self._displacement('act_radius')
        torch.testing.assert_close(damped, radius, rtol=1e-4, atol=1e-7)

    def test_euclid_layer_radius(self):
        weight = nn.Parameter(torch.randn(5, 7))
        weight.grad = torch.randn(5, 7)
        before = weight.detach().clone()
        config = lookahead.LookaheadConfig(cproj='euclid_layer', layer_radii=(0.25,))
        look = lookahead.Lookahead(['h.0.mlp.c_proj.weight'], [weight], config=config)
        look.displace(1.0)
        self.assertAlmostEqual(float((weight.detach() - before).norm()), 0.25, places=5)
        look.restore()
        torch.testing.assert_close(weight.detach(), before, rtol=0, atol=1e-6)

    def test_calibration_reports_schedule_normalized_means(self):
        weight, moment, look = self._setup('act_radius')
        look.displace(0.5)  # 'const' schedule: the factor is ignored
        look.abandon()
        calibration = look.calibration()
        self.assertAlmostEqual(calibration['act_disp_mean'], 0.3, places=5)
        self.assertEqual(calibration['layers'], ['h.0.mlp.c_proj.weight'])

    def test_record_flag_does_not_track(self):
        weight, _, look = self._setup('act_radius')
        look.displace(1.0, record=False)
        self.assertEqual(look.statistics(), {})

    def test_input_moment_update(self):
        moment, count = torch.zeros(4), torch.zeros(())
        h = torch.randn(2, 32, 4)
        for _ in range(50):
            lookahead.update_input_moment_(moment, count, h, stride=4, decay=0.9)
        expected = h[:, ::4].square().mean(dim=(0, 1))
        torch.testing.assert_close(lookahead.corrected_moment(moment, count, 0.9), expected)


class VirtualRankStateTest(unittest.TestCase):

    def _moves(self, activations_per_rank, *, emulate: bool):
        """Returns each rank's act_metric displacement after two steps of its own data."""
        name = 'h.0.mlp.c_proj.weight'
        config = lookahead.LookaheadConfig(cproj='act_metric', act_radius=0.3, act_damping=0.0)
        moves = []
        if emulate:
            moment, count = torch.zeros(2), torch.zeros(())
            state = lookahead.VirtualRankState([moment, count], len(activations_per_rank))
            for _ in range(2):
                state.begin_step()
                moves = []
                for v, activation in enumerate(activations_per_rank):
                    state.enter(v)
                    lookahead.update_input_moment_(moment, count, activation, stride=1, decay=0.9)
                    weight = nn.Parameter(torch.zeros(2, 2))
                    weight.grad = torch.ones(2, 2)
                    lookahead.Lookahead([name], [weight], config=config,
                                        moments={name: (moment, count)}).displace(1.0)
                    moves.append(weight.detach().clone())
                    state.exit(v)
            return moves
        for activation in activations_per_rank:
            moment, count = torch.zeros(2), torch.zeros(())
            for _ in range(2):
                lookahead.update_input_moment_(moment, count, activation, stride=1, decay=0.9)
                weight = nn.Parameter(torch.zeros(2, 2))
                weight.grad = torch.ones(2, 2)
                lookahead.Lookahead([name], [weight], config=config,
                                    moments={name: (moment, count)}).displace(1.0)
            moves.append(weight.detach().clone())
        return moves

    def test_emulated_ranks_keep_separate_moments(self):
        activations = [torch.tensor([1.0, 10.0]).view(1, 1, 2), torch.tensor([10.0, 1.0]).view(1, 1, 2)]
        for emulated, isolated in zip(self._moves(activations, emulate=True),
                                      self._moves(activations, emulate=False)):
            torch.testing.assert_close(emulated, isolated)

    def test_every_virtual_rank_starts_from_the_same_rng_state(self):
        state = lookahead.VirtualRankState([], 3)
        state.begin_step()
        draws = []
        for v in range(3):
            state.enter(v)
            draws.append(torch.rand(4))
        self.assertTrue(all(torch.equal(d, draws[0]) for d in draws))
        single = lookahead.VirtualRankState([], 1)
        single.begin_step()
        before = torch.get_rng_state()
        single.enter(0)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))


class HelpersTest(unittest.TestCase):

    def test_aux_multipliers_preserve_expected_objective(self):
        for mode in lookahead.AUX_SPLITS:
            a, b = lookahead.aux_multipliers(mode)
            self.assertEqual((a + b) / 2, 1.0)
        with self.assertRaises(ValueError):
            lookahead.aux_multipliers('both')

    def test_split_halves(self):
        x, y = torch.arange(8).view(4, 2), torch.arange(8).view(4, 2)
        adapt, query = lookahead.split_halves([(x, y, 1)])
        self.assertTrue(torch.equal(adapt[0][0], x[:2]))
        self.assertTrue(torch.equal(query[0][0], x[2:]))
        adapt, query = lookahead.split_halves([(x, y, 1)] * 4)
        self.assertEqual((len(adapt), len(query)), (2, 2))
        with self.assertRaises(ValueError):
            lookahead.split_halves([(x, y, 1)] * 3)

    def test_select_plastic_ownership_matches_record(self):
        model = nn.ModuleDict({'attn': nn.Linear(4, 4), 'mlp': nn.Linear(4, 4), 'mlp2': nn.Linear(4, 4)})
        named = [(f'h.{k}.{n}', p) for k, m in model.items() for n, p in m.named_parameters()]
        named = [(n.replace('h.mlp2', 'h.x.mlp'), p) for n, p in named]
        muon = [[p for _, p in named]]
        names, params, owned = lookahead.select_plastic(named, muon, rule='mlp_all', rank=1, world_size=2)
        self.assertTrue(all('.mlp' in n for n in names))
        chunk = (len(muon[0]) + 1) // 2
        owned_ids = {id(p) for p in muon[0][chunk:]}
        self.assertEqual(owned, [id(p) in owned_ids for p in params])


if __name__ == '__main__':
    unittest.main()
