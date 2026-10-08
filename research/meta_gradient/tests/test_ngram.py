"""Tests of the n-gram memory: causality, exact counts, hashing and strata."""

import collections
import unittest

import torch

import ngram

BOS = ngram.BOS_ID


def _stream(num_docs: int = 40, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    docs = []
    for _ in range(num_docs):
        length = int(torch.randint(2, 30, (1,), generator=gen))
        docs.append(torch.cat([torch.tensor([BOS]), torch.randint(0, 12, (length,), generator=gen)]))
    return torch.cat(docs)


def _bruteforce_counts(tokens: list[int], order: int) -> collections.Counter:
    counts = collections.Counter()
    for t in range(order - 1, len(tokens)):
        gram = tuple(tokens[t - order + 1:t + 1])
        if BOS not in gram[1:]:
            counts[gram] += 1
    return counts


class SuffixTest(unittest.TestCase):

    def test_validity_respects_documents_and_sequence_start(self):
        idx = torch.tensor([[BOS, 5, 6, BOS, 7, 8]])
        self.assertEqual(ngram.suffix_validity(idx, 1).tolist(), [[True] * 6])
        self.assertEqual(ngram.suffix_validity(idx, 2).tolist(), [[False, True, True, False, True, True]])
        self.assertEqual(ngram.suffix_validity(idx, 3).tolist(), [[False, False, True, False, False, True]])

    def test_keys_are_exact(self):
        idx = torch.tensor([[1, 2, 3, 50256]])
        keys = ngram.suffix_keys(idx, 3)
        self.assertEqual(int(keys[0, 2]), 3 + 2 * 50257 + 1 * 50257**2)
        self.assertEqual(int(keys[0, 3]), 50256 + 3 * 50257 + 2 * 50257**2)

    def test_counts_match_bruteforce(self):
        tokens = _stream()
        counts = ngram.SuffixCounts.build(tokens, orders=(1, 2, 3), device='cpu')
        found, valid = counts.lookup(tokens.view(1, -1))
        listed = tokens.tolist()
        for i, order in enumerate((1, 2, 3)):
            brute = _bruteforce_counts(listed, order)
            for t in range(len(listed)):
                expected = brute[tuple(listed[t - order + 1:t + 1])] if valid[0, t, i] else 0
                self.assertEqual(int(found[0, t, i]), expected)

    def test_support_and_strata(self):
        tokens = _stream()
        counts = ngram.SuffixCounts.build(tokens, orders=(2, 3), device='cpu')
        idx = tokens[:50].view(1, -1)
        raw, valid = counts.lookup(idx)
        support = counts.support(idx, kappa=4.0)
        torch.testing.assert_close(support, raw.float() / (raw.float() + 4.0))
        self.assertTrue(torch.equal(counts.support(idx, kappa=0.0), valid.float()))
        self.assertTrue(torch.equal(ngram.validity_support(idx, (2, 3)), valid.float()))
        strata = counts.strata(idx)
        self.assertEqual(int(strata[0, 0]), ngram.STRATA.index('no_context'))
        unseen = counts.strata(torch.tensor([[BOS, 100, 101, 102]]))
        self.assertEqual(int(unseen[0, 3]), ngram.STRATA.index('unseen'))


class HasherTest(unittest.TestCase):

    def test_rows_are_deterministic_and_in_range(self):
        hasher = ngram.SuffixHasher(orders=(2, 3), heads=2, table_size=101, seed=3)
        idx = torch.randint(0, 50257, (3, 40))
        rows = hasher(idx)
        self.assertTrue(torch.equal(rows, ngram.SuffixHasher(orders=(2, 3), heads=2, table_size=101, seed=3)(idx)))
        self.assertEqual(rows.shape, (3, 40, 4))
        for slot in range(4):
            lo, hi = hasher.offsets[slot], hasher.offsets[slot] + hasher.sizes[slot]
            self.assertTrue(((rows[..., slot] >= lo) & (rows[..., slot] < hi)).all())

    def test_collision_statistics(self):
        tokens = _stream(num_docs=200)
        counts = ngram.SuffixCounts.build(tokens, orders=(2, 3), device='cpu')
        hasher = ngram.SuffixHasher(orders=(2, 3), heads=1, table_size=17)
        stats = hasher.collision_statistics(counts)
        for key, value in stats.items():
            if key.endswith('purity') or key.endswith('used'):
                self.assertTrue(0.0 <= value <= 1.0, key)


class MemoryTest(unittest.TestCase):

    def _memory(self, gate: str = 'context') -> ngram.NgramMemory:
        torch.manual_seed(0)
        config = ngram.MemoryConfig(mode='hashed', orders=(2, 3), heads=2, head_dim=4,
                                    table_size=97, gate=gate)
        memory = ngram.NgramMemory(16, config)
        memory.init_weights()
        return memory

    def test_identity_at_initialization(self):
        memory = self._memory()
        h = torch.randn(2, 10, 16)
        idx = torch.randint(0, 50, (2, 10))
        support = torch.ones(2, 10, 2)
        self.assertTrue(torch.equal(memory(h, idx, support), h))

    def test_memory_is_causal(self):
        for gate in ('context', 'none'):
            memory = self._memory(gate)
            torch.nn.init.normal_(memory.w_v.weight)
            h = torch.randn(1, 12, 16)
            idx = torch.randint(0, 50, (1, 12))
            changed = idx.clone()
            changed[0, 7:] = torch.randint(50, 100, (5,))
            support = torch.ones(1, 12, 2)
            out_a = memory(h, idx, support)
            out_b = memory(h, changed, support)
            with self.subTest(gate=gate):
                self.assertTrue(torch.equal(out_a[:, :7], out_b[:, :7]))
                self.assertFalse(torch.equal(out_a[:, 7:], out_b[:, 7:]))

    def test_zero_support_disables_reads(self):
        memory = self._memory('none')
        torch.nn.init.normal_(memory.w_v.weight)
        h = torch.randn(1, 6, 16)
        out = memory(h, torch.randint(0, 50, (1, 6)), torch.zeros(1, 6, 2))
        self.assertTrue(torch.equal(out, h))

    def test_table_rows_are_padded_for_sharding(self):
        self.assertEqual(self._memory().table.weight.shape[0] % 64, 0)

    def test_dense_control_is_identity_and_parameter_matched(self):
        config = ngram.MemoryConfig(mode='dense', orders=(2, 3), heads=2, head_dim=8, table_size=4099)
        hidden = ngram.matched_dense_hidden(64, config)
        adapter = ngram.DenseAdapter(64, hidden)
        adapter.init_weights()
        h = torch.randn(2, 5, 64)
        self.assertTrue(torch.equal(adapter(h), h))
        hashed = ngram.NgramMemory(64, ngram.MemoryConfig(mode='hashed', orders=(2, 3), heads=2,
                                                          head_dim=8, table_size=4099))
        ratio = adapter.parameter_count() / hashed.parameter_count()
        self.assertTrue(0.9 < ratio < 1.1, ratio)
        self.assertFalse(config.uses_lookup)

    def test_accumulate_strata(self):
        sums = torch.zeros(len(ngram.STRATA), dtype=torch.float64)
        tokens = torch.zeros(len(ngram.STRATA), dtype=torch.int64)
        ngram.accumulate_strata(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0, 0, 2]), sums, tokens)
        self.assertEqual(sums.tolist(), [3.0, 0.0, 3.0, 0.0])
        self.assertEqual(tokens.tolist(), [2, 0, 1, 0])


if __name__ == '__main__':
    unittest.main()
