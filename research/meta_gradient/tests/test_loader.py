"""Tests that the virtual-rank loader reproduces the record's per-rank batches."""

import os
import tempfile
import unittest

import numpy as np
import torch

import loader


class RecordDataLoader:
    """The record's DataLoader (tiny/train.py), with rank and world size as arguments."""

    def __init__(self, filepath, B, T, *, rank, world_size, doc_shuffle=False):
        data = torch.load(filepath, weights_only=True)
        all_tokens = data["tokens"].long()
        raw_doc_starts = data["doc_starts"].long()
        doc_ends = torch.cat([raw_doc_starts[1:], torch.tensor([all_tokens.numel()])])
        self.doc_tokens = [all_tokens[s:e] for s, e in zip(raw_doc_starts.tolist(), doc_ends.tolist())]
        self.default_shuffle_seed = data["seq_shuffle_seed"]
        self.rank, self.world_size = rank, world_size
        self.B, self.T, self.seq_size = B, T, T + 1
        self.doc_shuffle = doc_shuffle
        self.epoch = 1
        self._build_batches()

    def _build_batches(self):
        tokens = torch.cat(self.doc_tokens)
        num_seqs = len(tokens) // self.seq_size
        all_seqs = tokens[:num_seqs * self.seq_size].view(num_seqs, self.seq_size)
        if self.doc_shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + 1000)
            all_seqs = all_seqs[torch.randperm(num_seqs, generator=g)]
        else:
            perm = np.random.RandomState(self.default_shuffle_seed).permutation(num_seqs)
            all_seqs = all_seqs[torch.from_numpy(perm)]
        seqs_per_step = self.B * self.world_size
        num_steps = len(all_seqs) // seqs_per_step
        usable = num_steps * seqs_per_step
        all_seqs = all_seqs[:usable].view(num_steps, self.world_size, self.B, self.seq_size)
        self.rank_data = all_seqs[:, self.rank].contiguous()
        self.num_steps = num_steps
        self.total_tokens = usable * self.T
        self.pos = 0

    def _next_epoch(self):
        self.epoch += 1
        if self.doc_shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            perm = torch.randperm(len(self.doc_tokens), generator=g)
            self.doc_tokens = [self.doc_tokens[i] for i in perm.tolist()]
            self._build_batches()
        else:
            self.pos = 0
            g = torch.Generator()
            g.manual_seed(self.epoch)
            self.rank_data = self.rank_data[torch.randperm(self.num_steps, generator=g)]

    def __next__(self):
        if self.pos >= self.num_steps:
            self._next_epoch()
        batch = self.rank_data[self.pos]
        self.pos += 1
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), self.epoch


class VirtualRankLoaderTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, 'train.pt')
        loader.write_synthetic_dataset(cls.path, num_tokens=6000, seed=4, doc_lengths=(5, 60))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _compare(self, *, world_size, micro, doc_shuffle, steps=40, batch=2, seq_len=8, ranks=4):
        records = [RecordDataLoader(self.path, batch, seq_len, rank=r, world_size=ranks,
                                    doc_shuffle=doc_shuffle) for r in range(ranks)]
        emulated = [loader.VirtualRankLoader(self.path, batch, seq_len, device='cpu', doc_shuffle=doc_shuffle,
                                             rank=p, world_size=world_size, virtual_ranks=ranks,
                                             micro_per_vrank=micro) for p in range(world_size)]
        self.assertEqual(emulated[0].total_tokens, records[0].total_tokens)
        local = ranks // world_size
        for _ in range(steps):
            expected = [[next(records[r]) for _ in range(micro)] for r in range(ranks)]
            for p, emu in enumerate(emulated):
                got = emu.next_step()
                for v in range(local):
                    for m in range(micro):
                        x, y, epoch = got[v * micro + m]
                        rx, ry, repoch = expected[p * local + v][m]
                        self.assertTrue(torch.equal(x, rx) and torch.equal(y, ry))
                        self.assertEqual(epoch, repoch)

    def test_one_process_emulates_all_ranks(self):
        for doc_shuffle in (True, False):
            with self.subTest(doc_shuffle=doc_shuffle):
                self._compare(world_size=1, micro=1, doc_shuffle=doc_shuffle)

    def test_two_processes_split_the_ranks(self):
        self._compare(world_size=2, micro=1, doc_shuffle=True)

    def test_microbatches_with_epoch_boundaries_inside_steps(self):
        # Three microbatches per rank do not divide the steps of an epoch.
        self._compare(world_size=1, micro=3, doc_shuffle=True)

    def test_identity_when_every_rank_is_a_process(self):
        self._compare(world_size=4, micro=2, doc_shuffle=True)

    def test_peek_epoch(self):
        emu = loader.VirtualRankLoader(self.path, 2, 8, device='cpu', doc_shuffle=True, rank=0,
                                       world_size=1, virtual_ranks=4, micro_per_vrank=1)
        for _ in range(200):
            expected = emu.peek_epoch()
            self.assertEqual(emu.next_step()[0][2], expected)

    def test_data_seed_changes_order_only_when_set(self):
        a = loader.VirtualRankLoader(self.path, 2, 8, device='cpu', doc_shuffle=True, rank=0,
                                     world_size=1, virtual_ranks=4, micro_per_vrank=1)
        b = loader.VirtualRankLoader(self.path, 2, 8, device='cpu', doc_shuffle=True, rank=0,
                                     world_size=1, virtual_ranks=4, micro_per_vrank=1, data_seed=3)
        self.assertFalse(torch.equal(a.next_step()[0][0], b.next_step()[0][0]))

    def test_invalid_virtual_ranks(self):
        with self.assertRaises(ValueError):
            loader.VirtualRankLoader(self.path, 2, 8, device='cpu', doc_shuffle=True, rank=0,
                                     world_size=3, virtual_ranks=4, micro_per_vrank=1)

    def test_probe_is_independent_of_process_count(self):
        whole = loader.probe_sequences(self.path, seq_len=8, per_vrank=3, virtual_ranks=4,
                                       first_vrank=0, local_vranks=4, device='cpu')[0]
        parts = torch.cat([loader.probe_sequences(self.path, seq_len=8, per_vrank=3, virtual_ranks=4,
                                                  first_vrank=2 * p, local_vranks=2, device='cpu')[0]
                           for p in range(2)])
        self.assertTrue(torch.equal(whole, parts))


if __name__ == '__main__':
    unittest.main()
