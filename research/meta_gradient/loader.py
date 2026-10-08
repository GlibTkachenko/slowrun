"""Data loading that lets fewer processes reproduce the record's 8-rank batches.

The record loaders lay each epoch out as (steps, ranks, batch, sequence) and give
rank r the column r. `VirtualRankLoader` keeps that layout for a configurable
number of *virtual* ranks and gives each process a contiguous block of them, so
one GPU can run the exact per-rank meta-gradient algorithm of an 8-GPU run.
With as many virtual ranks as processes it is identical to the record loader.

The module also builds the fixed training probe set and synthetic datasets for
smoke tests.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch

BOS_ID = 50256


def _print0(message: str) -> None:
    if int(os.environ.get('RANK', 0)) == 0:
        print(message)


def load_documents(filepath: str) -> tuple[list[torch.Tensor], int]:
    """Reads a prepared token file into per-document tensors.

    Args:
        filepath: A file written by prepare_data.py (or `write_synthetic_dataset`).

    Returns:
        The documents and the stored default sequence-shuffle seed.

    Raises:
        ValueError: The file's BOS token differs from BOS_ID.
    """
    data = torch.load(filepath, weights_only=True)
    tokens = data['tokens'].long()
    starts = data['doc_starts'].long()
    if int(data['bos_id']) != BOS_ID:
        raise ValueError(f'data bos_id {int(data["bos_id"])} != expected {BOS_ID}')
    ends = torch.cat([starts[1:], torch.tensor([tokens.numel()])])
    docs = [tokens[s:e] for s, e in zip(starts.tolist(), ends.tolist())]
    return docs, int(data['seq_shuffle_seed'])


class VirtualRankLoader:
    """Record-equivalent training loader over virtual data-parallel ranks.

    Every optimizer step consumes `micro_per_vrank` record loader steps. The
    microbatches are emitted grouped by virtual rank, i.e. all microbatches of
    the first local virtual rank, then the second, and so on, exactly as the
    record would have fed them to the corresponding physical ranks. Epoch
    boundaries may fall inside a step, as in the one-hour record.
    """

    def __init__(
        self,
        filepath: str,
        batch_size: int,
        seq_len: int,
        *,
        device: torch.device | str,
        doc_shuffle: bool,
        rank: int,
        world_size: int,
        virtual_ranks: int,
        micro_per_vrank: int,
        data_seed: int | None = None,
    ):
        """Initializes the loader and builds the first epoch.

        Args:
            filepath: Prepared training token file.
            batch_size: Sequences per microbatch.
            seq_len: Tokens per sequence (targets add one).
            device: Device for the yielded tensors.
            doc_shuffle: Reshuffle documents every epoch (record default).
            rank: This process's rank.
            world_size: Number of processes.
            virtual_ranks: Number of emulated data-parallel ranks; a multiple of
                `world_size`.
            micro_per_vrank: Microbatches each virtual rank processes per step.
            data_seed: Optional offset of every shuffle seed. None reproduces the
                record's data order.

        Raises:
            ValueError: `virtual_ranks` is not a positive multiple of `world_size`.
        """
        if virtual_ranks < world_size or virtual_ranks % world_size:
            raise ValueError(
                f'virtual_ranks={virtual_ranks} must be a positive multiple of '
                f'world_size={world_size}.'
            )
        self.doc_tokens, self.default_shuffle_seed = load_documents(filepath)
        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seq_size = seq_len + 1
        self.doc_shuffle = doc_shuffle
        self.virtual_ranks = virtual_ranks
        self.local_vranks = virtual_ranks // world_size
        self.first_vrank = rank * self.local_vranks
        self.micro_per_vrank = micro_per_vrank
        self.data_seed = data_seed
        self.epoch = 1
        self._build_batches()

    def _seed(self, base: int) -> int:
        """Offsets a record shuffle seed by the optional data seed."""
        return base if self.data_seed is None else base + 7919 * (self.data_seed + 1)

    def _build_batches(self) -> None:
        tokens = torch.cat(self.doc_tokens)
        num_seqs = len(tokens) // self.seq_size
        all_seqs = tokens[: num_seqs * self.seq_size].view(num_seqs, self.seq_size)
        if self.doc_shuffle:
            g = torch.Generator()
            g.manual_seed(self._seed(self.epoch + 1000))
            all_seqs = all_seqs[torch.randperm(num_seqs, generator=g)]
        else:
            perm = np.random.RandomState(self.default_shuffle_seed).permutation(num_seqs)
            all_seqs = all_seqs[torch.from_numpy(perm)]
        seqs_per_step = self.batch_size * self.virtual_ranks
        num_steps = len(all_seqs) // seqs_per_step
        usable = num_steps * seqs_per_step
        layout = all_seqs[:usable].view(num_steps, self.virtual_ranks, self.batch_size, self.seq_size)
        block = slice(self.first_vrank, self.first_vrank + self.local_vranks)
        self.rank_data = layout[:, block].contiguous()
        self.num_steps = num_steps
        self.total_tokens = usable * self.seq_len
        self.pos = 0

    def _next_epoch(self) -> None:
        self.epoch += 1
        _print0(f'Starting epoch {self.epoch}')
        if self.doc_shuffle:
            g = torch.Generator()
            g.manual_seed(self._seed(self.epoch))
            perm = torch.randperm(len(self.doc_tokens), generator=g)
            self.doc_tokens = [self.doc_tokens[i] for i in perm.tolist()]
            self._build_batches()
        else:
            self.pos = 0
            g = torch.Generator()
            g.manual_seed(self._seed(self.epoch))
            self.rank_data = self.rank_data[torch.randperm(self.num_steps, generator=g)]

    def _next_loader_step(self) -> tuple[torch.Tensor, int]:
        """Returns the next record loader step for all local virtual ranks."""
        if self.pos >= self.num_steps:
            self._next_epoch()
        batch = self.rank_data[self.pos]
        self.pos += 1
        return batch, self.epoch

    def peek_epoch(self) -> int:
        """Returns the epoch of the next microbatch without consuming it."""
        return self.epoch if self.pos < self.num_steps else self.epoch + 1

    def next_step(self) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        """Returns all microbatches of one optimizer step, grouped by virtual rank.

        Returns:
            local_vranks * micro_per_vrank tuples (inputs, targets, epoch).
        """
        chunks = [self._next_loader_step() for _ in range(self.micro_per_vrank)]
        out = []
        for v in range(self.local_vranks):
            for batch, epoch in chunks:
                seqs = batch[v].to(self.device, non_blocking=True)
                out.append((seqs[:, :-1].contiguous(), seqs[:, 1:].contiguous(), epoch))
        return out


def probe_sequences(
    filepath: str,
    *,
    seq_len: int,
    per_vrank: int,
    virtual_ranks: int,
    first_vrank: int,
    local_vranks: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns a fixed sample of training sequences for diagnostics.

    The probe is defined by position in the stored default permutation (its last
    sequences), not by any model output, so it is identical across arms, seeds
    and process counts. It is part of the training set, so its loss is a
    training-loss probe.

    Args:
        filepath: Prepared training token file.
        seq_len: Tokens per sequence.
        per_vrank: Sequences per virtual rank.
        virtual_ranks: Number of virtual ranks.
        first_vrank: First virtual rank of this process.
        local_vranks: Virtual ranks of this process.
        device: Device of the returned tensors.

    Returns:
        (local_vranks * per_vrank, seq_len) inputs and targets, grouped by
        virtual rank.

    Raises:
        ValueError: The dataset is too small for the requested probe.
    """
    docs, seed = load_documents(filepath)
    tokens = torch.cat(docs)
    seq_size = seq_len + 1
    num_seqs = len(tokens) // seq_size
    total = per_vrank * virtual_ranks
    if total > num_seqs:
        raise ValueError(f'Probe needs {total} sequences but the dataset has {num_seqs}.')
    seqs = tokens[: num_seqs * seq_size].view(num_seqs, seq_size)
    perm = torch.from_numpy(np.random.RandomState(seed).permutation(num_seqs))
    picked = perm[num_seqs - total:].view(virtual_ranks, per_vrank)
    picked = picked[first_vrank:first_vrank + local_vranks].reshape(-1)
    batch = seqs[picked].to(device)
    return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


def write_synthetic_dataset(
    path: str,
    *,
    num_tokens: int,
    seed: int,
    chain_seed: int = 0,
    vocab: int = 50257,
    doc_lengths: Sequence[int] = (40, 400),
) -> None:
    """Writes a small random dataset in the prepare_data.py format.

    Tokens follow a random order-1 Markov chain over a 512-token subset of the
    vocabulary, so models can learn something and n-grams repeat. Train and
    validation files share the chain through `chain_seed` and differ in `seed`.

    Args:
        path: Output file.
        num_tokens: Total tokens including BOS markers.
        seed: Seed of the sampled documents (and the stored shuffle seed).
        chain_seed: Seed of the alphabet and transition matrix.
        vocab: Vocabulary size.
        doc_lengths: Inclusive range of document lengths.
    """
    chain = np.random.default_rng(chain_seed)
    alphabet = chain.choice(vocab - 1, size=512, replace=False)
    transition = chain.dirichlet(np.full(512, 0.05), size=512)
    rng = np.random.default_rng(seed)
    tokens, starts = [], []
    while len(tokens) < num_tokens:
        starts.append(len(tokens))
        length = int(rng.integers(doc_lengths[0], doc_lengths[1] + 1))
        state = int(rng.integers(512))
        doc = [BOS_ID]
        for _ in range(length - 1):
            state = int(rng.choice(512, p=transition[state]))
            doc.append(int(alphabet[state]))
        tokens.extend(doc)
    tokens = tokens[:num_tokens]
    data = {
        'tokens': torch.tensor(tokens, dtype=torch.int64).to(torch.uint16 if vocab < 65536 else torch.int64),
        'doc_starts': torch.tensor([s for s in starts if s < num_tokens], dtype=torch.int64),
        'bos_id': BOS_ID,
        'seq_shuffle_seed': seed,
        'seq_size': 2049,
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(data, path)
