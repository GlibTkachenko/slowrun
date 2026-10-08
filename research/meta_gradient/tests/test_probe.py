"""Tests of exact snapshots and of the survival diagnostic's no-trace guarantee."""

import math
import unittest

import torch
import torch.nn as nn

import lookahead
import probe


def _model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8, bias=False), nn.Dropout(0.2), nn.Tanh(),
                         nn.Linear(8, 8, bias=False), nn.Tanh(), nn.Linear(8, 2, bias=False))


class SnapshotTest(unittest.TestCase):

    def test_roundtrip_restores_parameters_state_buffers_and_rng(self):
        model = _model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        params = list(model.parameters())
        buffer = torch.zeros(3)
        model(torch.randn(4, 6)).sum().backward()
        optimizer.step()
        digest = probe.snapshot_digest(params, optimizer)
        snap = probe.take_snapshot(params, optimizer, [buffer])
        draw = torch.rand(3)
        for _ in range(3):
            model.zero_grad()
            model(torch.randn(4, 6)).sum().backward()
            optimizer.step()
        buffer.add_(1.0)
        probe.restore_snapshot(snap, params, optimizer, [buffer])
        self.assertEqual(probe.snapshot_digest(params, optimizer), digest)
        self.assertTrue(torch.equal(torch.rand(3), draw))
        self.assertEqual(float(buffer.abs().sum()), 0.0)

    def test_restore_removes_state_created_after_snapshot(self):
        model = _model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        params = list(model.parameters())
        snap = probe.take_snapshot(params, optimizer)
        model(torch.randn(4, 6)).sum().backward()
        optimizer.step()
        probe.restore_snapshot(snap, params, optimizer)
        self.assertEqual(len(optimizer.state), 0)

    def test_parameter_role(self):
        self.assertEqual(probe.parameter_role('transformer.h.3.mlp.c_proj.weight'), 'mlp.c_proj')
        self.assertEqual(probe.parameter_role('mtp_block.attn.c_q.weight'), 'mtp.attn.c_q')
        self.assertEqual(probe.parameter_role('ve_projs.4.weight'), 've_projs')
        self.assertEqual(probe.parameter_role('resid_lambdas'), 'scalars')


class SurvivalTest(unittest.TestCase):

    def _setup(self, *, dropout: bool = True):
        torch.manual_seed(0)
        model = _model() if dropout else nn.Sequential(
            nn.Linear(6, 8, bias=False), nn.Tanh(), nn.Linear(8, 8, bias=False), nn.Tanh(),
            nn.Linear(8, 2, bias=False))
        model.train()
        names = [n for n, _ in model.named_parameters()]
        params = list(model.parameters())
        optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.0 if not dropout else 0.9)
        target = torch.randn(8, 2)

        def run_half(batches, half):
            for (x,) in batches:
                ((model(x) - target).square().mean() / 2).backward()

        def probe_loss():
            with torch.no_grad():
                model.eval()
                value = float((model(torch.ones(8, 6)) - target).square().mean())
                model.train()
                return value

        inputs = probe.SurvivalInputs(
            run_half=run_half,
            probe_loss=probe_loss,
            muon_input=lambda name, grad: grad.float(),
            preprocess=lambda m: m,
            zero_grad=lambda: model.zero_grad(set_to_none=True),
        )
        return model, names, params, optimizer, inputs, run_half, probe_loss

    def test_diagnostic_leaves_no_trace_and_reports_finite_metrics(self):
        model, names, params, optimizer, inputs, _, _ = self._setup()
        model(torch.randn(4, 6)).square().mean().backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        look = lookahead.Lookahead(names[:2], params[:2], config=lookahead.LookaheadConfig(step_norm=0.5))
        pairs = [([(torch.randn(8, 6),)], [(torch.randn(8, 6),)]) for _ in range(2)]
        others = [([(torch.randn(8, 6),)], [(torch.randn(8, 6),)]) for _ in range(2)]
        digest = probe.snapshot_digest(params, optimizer)
        rng = torch.get_rng_state()
        metrics, spectra = probe.mg_survival(
            named_params=list(model.named_parameters()), optimizer=optimizer, lookahead=look,
            lr_multiplier=1.0, inputs=inputs, step_pairs=pairs, independent_pairs=others,
            representative=[names[1]])
        self.assertEqual(probe.snapshot_digest(params, optimizer), digest)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(all(p.grad is None for p in params))
        self.assertEqual(look.statistics(), {})
        for key in ('diag/probe_gain_base', 'diag/probe_gain_mg', 'diag/grad_rel/all', 'diag/survival/all'):
            self.assertTrue(math.isfinite(metrics[key]), key)
        self.assertIn(f'diag/spec/{names[1]}/repeat_all', metrics)
        self.assertEqual(set(spectra), {names[1]})

    def test_virtual_update_uses_the_mean_gradient_of_all_virtual_ranks(self):
        model, names, params, optimizer, inputs, run_half, probe_loss = self._setup(dropout=False)
        look = lookahead.Lookahead(names[:2], params[:2], config=lookahead.LookaheadConfig(step_norm=0.0))
        pairs = [([(torch.randn(8, 6),)], [(torch.randn(8, 6),)]) for _ in range(3)]
        # Reference: zero displacement, so G0 is the mean over pairs of g_A + g_B.
        model.zero_grad(set_to_none=True)
        for adapt, query in pairs:
            run_half(adapt, 'A')
            run_half(query, 'B')
        mean_grads = [p.grad / len(pairs) for p in params]
        before = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p, g in zip(params, mean_grads):
                p.sub_(0.1 * g)
        expected = probe_loss()
        with torch.no_grad():
            for p, b in zip(params, before):
                p.copy_(b)
        model.zero_grad(set_to_none=True)
        metrics, _ = probe.mg_survival(
            named_params=list(model.named_parameters()), optimizer=optimizer, lookahead=look,
            lr_multiplier=1.0, inputs=inputs, step_pairs=pairs, independent_pairs=pairs[:1],
            representative=[])
        self.assertAlmostEqual(metrics['diag/probe_ref'] - metrics['diag/probe_gain_base'], expected, places=6)

    def test_displacement_reads_moments_from_after_the_adaptation_half(self):
        weight = nn.Parameter(torch.ones(2, 2))
        moment, count = torch.ones(2), torch.tensor(1e6)
        seen = []

        class Recording(lookahead.Lookahead):
            def _metric(self, name):
                seen.append(moment.clone())
                return super()._metric(name)

        name = 'h.0.mlp.c_proj.weight'
        look = Recording([name], [weight], moments={name: (moment, count)},
                         config=lookahead.LookaheadConfig(cproj='act_radius', act_radius=0.1))

        def run_half(batches, half):
            moment.mul_(2.0 if half == 'A' else 3.0)  # what a forward pass does to the EMA
            (weight.sum() * batches[0][0]).backward()

        inputs = probe.SurvivalInputs(
            run_half=run_half, probe_loss=lambda: 0.0, muon_input=lambda name, grad: None,
            preprocess=lambda m: m, zero_grad=lambda: setattr(weight, 'grad', None))
        optimizer = torch.optim.SGD([weight], lr=0.1)
        pair = ([(torch.tensor(1.0),)], [(torch.tensor(1.0),)])
        probe.mg_survival(named_params=[(name, weight)], optimizer=optimizer, lookahead=look,
                          lr_multiplier=1.0, inputs=inputs, step_pairs=[pair], independent_pairs=[pair],
                          representative=[], buffers=[moment], moment_buffers=[moment])
        self.assertTrue(seen)
        self.assertTrue(all(torch.equal(m, torch.full((2,), 2.0)) for m in seen))
        self.assertTrue(torch.equal(moment, torch.ones(2)))


if __name__ == '__main__':
    unittest.main()
