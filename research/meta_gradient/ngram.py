"""Causal hashed n-gram memory with count-based shrinkage (Priority E).

The memory follows Engram's lookup-and-gate design at a deliberately small
scale. For every position t it looks up embeddings of the suffix n-grams
(x_{t-n+1}, ..., x_t) of a few short orders under several multiplicative-XOR
hash heads, concatenates them and adds a gated projection to the hidden state:

    e_t = concat_{n,k} s_{t,n} E_{n,k}[phi_{n,k}(x_{t-n+1..t})],
    h'_t = h_t + alpha_t W_V e_t,
    alpha_t = sigmoid(<rms(h_t), rms(W_K e_t)> / sqrt(d) + b).

The limited-data adaptation is the support factor s_{t,n} = c/(c+kappa),
where c is the *exact* training count of the n-gram (not of its hash bucket),
so collisions cannot create evidence for rare suffixes. Only prefix tokens enter
the keys, and suffixes never cross a document boundary (a BOS token).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import time
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

BOS_ID = 50256
VOCAB_SIZE = 50257
MAX_COUNT = 65535
MEMORY_MODES = ('off', 'unigram', 'hashed', 'dense')
GATE_MODES = ('context', 'none')

# Held-out strata by the training support of the longest valid suffix.
STRATA = ('unseen', 'rare', 'common', 'no_context')
RARE_MAX = 9

_MASK31 = (1 << 31) - 1


def _shift_right(idx: torch.Tensor, offset: int) -> torch.Tensor:
    """Returns idx delayed by `offset` positions along time, zero padded."""
    if offset == 0:
        return idx
    return torch.cat([idx.new_zeros(idx.size(0), offset), idx[:, :-offset]], dim=1)


def suffix_validity(idx: torch.Tensor, order: int, *, bos_id: int = BOS_ID) -> torch.Tensor:
    """Returns where the suffix n-gram ending at each position lies in one document.

    The n-gram ending at t is valid when all its tokens are inside the sequence
    and no BOS token occurs after its first token; a BOS at its first position
    marks the start of the document that contains it.

    Args:
        idx: (batch, time) token ids.
        order: Length n of the suffix.
        bos_id: Document-start token.

    Returns:
        A (batch, time) boolean tensor.
    """
    valid = torch.ones_like(idx, dtype=torch.bool)
    for back in range(order - 1):
        valid &= _shift_right(idx, back) != bos_id
    if order > 1:
        valid[:, : order - 1] = False
    return valid


def validity_support(
    idx: torch.Tensor, orders: Sequence[int], *, bos_id: int = BOS_ID
) -> torch.Tensor:
    """Returns the support of arms without shrinkage: one for valid suffixes, else zero.

    Equal to `SuffixCounts.support(idx, kappa=0)` without building or querying counts.

    Args:
        idx: (batch, time) token ids.
        orders: Suffix lengths.
        bos_id: Document-start token.

    Returns:
        A (batch, time, orders) float32 tensor.
    """
    return torch.stack([suffix_validity(idx, n, bos_id=bos_id) for n in orders], dim=-1).float()


def suffix_keys(idx: torch.Tensor, order: int, *, vocab: int = VOCAB_SIZE) -> torch.Tensor:
    """Returns exact int64 keys sum_i x_{t-i} vocab^i of the suffix n-grams.

    Args:
        idx: (batch, time) token ids below `vocab`.
        order: Length n of the suffix, at most 3 for int64 keys.
        vocab: Vocabulary size.

    Returns:
        A (batch, time) int64 tensor; entries at invalid positions are arbitrary.

    Raises:
        ValueError: The order is too long for exact int64 keys.
    """
    if vocab**order >= 2**62:
        raise ValueError(f'Order {order} keys overflow int64 for vocabulary {vocab}.')
    idx = idx.long()
    key = idx.clone()
    for back in range(1, order):
        key = key + _shift_right(idx, back) * vocab**back
    return key


@dataclasses.dataclass
class SuffixCounts:
    """Exact training counts of short suffixes, kept as sorted keys for lookups.

    Attributes:
        orders: Suffix lengths.
        keys: Per order, sorted unique int64 keys.
        counts: Per order, int32 counts clamped to MAX_COUNT.
        vocab: Vocabulary size of the keys.
        bos_id: Document-start token.
    """

    orders: tuple[int, ...]
    keys: list[torch.Tensor]
    counts: list[torch.Tensor]
    vocab: int = VOCAB_SIZE
    bos_id: int = BOS_ID

    @classmethod
    def build(
        cls,
        tokens: torch.Tensor,
        *,
        orders: Sequence[int],
        device: torch.device | str,
        vocab: int = VOCAB_SIZE,
        bos_id: int = BOS_ID,
    ) -> SuffixCounts:
        """Counts every in-document suffix of the training token stream.

        Args:
            tokens: 1-D training token stream, documents starting with BOS.
            orders: Suffix lengths to count.
            device: Device for counting and lookups.
            vocab: Vocabulary size.
            bos_id: Document-start token.

        Returns:
            The counts.
        """
        stream = tokens.to(device).long().view(1, -1)
        keys, counts = [], []
        for order in orders:
            valid = suffix_validity(stream, order, bos_id=bos_id)
            unique, count = torch.unique(
                suffix_keys(stream, order, vocab=vocab)[valid], sorted=True, return_counts=True
            )
            keys.append(unique)
            counts.append(count.clamp_max(MAX_COUNT).int())
            del valid
        return cls(orders=tuple(orders), keys=keys, counts=counts, vocab=vocab, bos_id=bos_id)

    def lookup(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns training counts and validity of the suffixes at every position.

        Args:
            idx: (batch, time) token ids.

        Returns:
            (batch, time, orders) int32 counts, zero where unseen or invalid, and the
            matching boolean validity.
        """
        all_counts, all_valid = [], []
        for order, keys, counts in zip(self.orders, self.keys, self.counts):
            valid = suffix_validity(idx, order, bos_id=self.bos_id)
            all_valid.append(valid)
            if keys.numel() == 0:
                all_counts.append(torch.zeros_like(idx, dtype=torch.int32))
                continue
            key = suffix_keys(idx, order, vocab=self.vocab)
            pos = torch.searchsorted(keys, key).clamp_max(keys.numel() - 1)
            found = (keys[pos] == key) & valid
            all_counts.append(torch.where(found, counts[pos], torch.zeros_like(counts[pos])))
        return torch.stack(all_counts, dim=-1), torch.stack(all_valid, dim=-1)

    def support(self, idx: torch.Tensor, *, kappa: float) -> torch.Tensor:
        """Returns per-order shrinkage factors c/(c+kappa), zero on invalid suffixes.

        Args:
            idx: (batch, time) token ids.
            kappa: Prior strength. Zero disables shrinkage (factor one when valid).

        Returns:
            A (batch, time, orders) float32 tensor.
        """
        if kappa <= 0:
            return validity_support(idx, self.orders, bos_id=self.bos_id)
        counts, _ = self.lookup(idx)
        counts = counts.float()
        return counts / (counts + kappa)

    def strata(self, idx: torch.Tensor) -> torch.Tensor:
        """Returns the support stratum of every position (see STRATA).

        The longest valid suffix decides: unseen in training, rare (1..RARE_MAX
        occurrences), common, or no valid suffix at all.

        Args:
            idx: (batch, time) token ids.

        Returns:
            A (batch, time) int64 tensor of indices into STRATA.
        """
        counts, valid = self.lookup(idx)
        chosen = torch.zeros_like(counts[..., 0])
        has_any = torch.zeros_like(valid[..., 0])
        for i in range(len(self.orders)):
            chosen = torch.where(valid[..., i], counts[..., i], chosen)
            has_any |= valid[..., i]
        stratum = torch.full_like(chosen, 2, dtype=torch.long)
        stratum[chosen <= RARE_MAX] = 1
        stratum[chosen == 0] = 0
        stratum[~has_any] = 3
        return stratum

    def nbytes(self) -> int:
        """Returns the device memory held by the tables."""
        return sum(t.numel() * t.element_size() for t in self.keys + self.counts)


# =============================================================================
# Hashing.
# =============================================================================


def _primes_from(start: int, count: int) -> list[int]:
    """Returns the `count` smallest primes not below `start`."""
    primes, candidate = [], max(start, 2)
    while len(primes) < count:
        if all(candidate % d for d in range(2, int(math.isqrt(candidate)) + 1)):
            primes.append(candidate)
        candidate += 1
    return primes


class SuffixHasher:
    """Multi-head multiplicative-XOR hashing of suffix n-grams into table rows.

    Every (order, head) slot owns its own prime-sized block of rows in one shared
    table, so a single gather serves all lookups.
    """

    def __init__(self, *, orders: Sequence[int], heads: int, table_size: int, seed: int = 0):
        """Initializes the hasher.

        Args:
            orders: Suffix lengths, e.g. (2, 3); (1,) for the unigram control.
            heads: Hash functions per order.
            table_size: Approximate rows per slot; rounded up to distinct primes.
            seed: Seed of the hash multipliers.
        """
        self.orders = tuple(orders)
        self.heads = heads
        self.slots = len(self.orders) * heads
        self.sizes = _primes_from(table_size, self.slots)
        self.offsets = [sum(self.sizes[:i]) for i in range(self.slots)]
        self.rows = sum(self.sizes)
        rng = random.Random(seed)
        max_order = max(self.orders)
        # Odd multipliers below 2^31 keep every product inside int64.
        self.multipliers = [
            [rng.randrange(1 << 20, 1 << 31) | 1 for _ in range(max_order + 1)]
            for _ in range(self.slots)
        ]
        self.slot_order = [i for i, _ in enumerate(self.orders) for _ in range(heads)]

    def bucket(self, idx: torch.Tensor, slot: int) -> torch.Tensor:
        """Returns the bucket (within its slot) of every suffix for one slot.

        Args:
            idx: (batch, time) token ids.
            slot: Index into the (order, head) slots.

        Returns:
            A (batch, time) int64 tensor in [0, sizes[slot]).
        """
        order = self.orders[self.slot_order[slot]]
        mult = self.multipliers[slot]
        idx = idx.long()
        h = torch.zeros_like(idx)
        for back in reversed(range(order)):
            h = ((h * mult[0]) ^ (_shift_right(idx, back) * mult[back + 1])) & _MASK31
        return h % self.sizes[slot]

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        """Returns (batch, time, slots) rows of the shared table."""
        return torch.stack(
            [self.bucket(idx, s) + self.offsets[s] for s in range(self.slots)], dim=-1
        )

    def collision_statistics(self, counts: SuffixCounts) -> dict[str, float]:
        """Summarizes how distinct training suffixes share hash buckets.

        Args:
            counts: Exact counts whose orders include this hasher's orders.

        Returns:
            Per slot: used-bucket fraction, mean and max distinct suffixes per used
            bucket, and count-weighted purity (share of a bucket's occurrences
            belonging to its most frequent suffix).
        """
        stats = {}
        for slot in range(self.slots):
            order = self.orders[self.slot_order[slot]]
            position = counts.orders.index(order)
            keys, freq = counts.keys[position], counts.counts[position].long()
            tokens = torch.stack(
                [(keys // counts.vocab**back) % counts.vocab for back in reversed(range(order))],
                dim=-1,
            )
            bucket = self.bucket(tokens, slot)[:, -1]
            size = self.sizes[slot]
            distinct = torch.bincount(bucket, minlength=size)
            total = torch.zeros(size, dtype=torch.long, device=keys.device).scatter_add_(0, bucket, freq)
            top = torch.zeros(size, dtype=torch.long, device=keys.device).scatter_reduce_(
                0, bucket, freq, reduce='amax'
            )
            used = distinct > 0
            tag = f'ngram/slot{slot}_order{order}'
            stats[f'{tag}_used'] = float(used.float().mean())
            stats[f'{tag}_distinct_mean'] = float(distinct[used].float().mean())
            stats[f'{tag}_distinct_max'] = float(distinct.max())
            stats[f'{tag}_purity'] = float(top.sum() / total.sum().clamp_min(1))
        return stats


# =============================================================================
# The memory module.
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MemoryConfig:
    """Hyperparameters of the n-gram memory.

    Attributes:
        mode: One of MEMORY_MODES.
        orders: Suffix lengths for 'hashed'; 'unigram' always uses (1,).
        heads: Hash heads per order.
        head_dim: Embedding width per slot.
        table_size: Approximate rows per slot.
        layer: Index of the block whose input receives the memory.
        gate: One of GATE_MODES.
        gate_bias: Initial gate bias; negative values start conservatively.
        kappa: Shrinkage prior strength; zero disables shrinkage.
        seed: Hash seed.
    """

    mode: str = 'off'
    orders: tuple[int, ...] = (2, 3)
    heads: int = 2
    head_dim: int = 32
    table_size: int = 196613
    layer: int = 2
    gate: str = 'context'
    gate_bias: float = 0.0
    kappa: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if self.mode not in MEMORY_MODES:
            raise ValueError(f'Unknown memory mode {self.mode!r}; expected one of {MEMORY_MODES}.')
        if self.gate not in GATE_MODES:
            raise ValueError(f'Unknown gate {self.gate!r}; expected one of {GATE_MODES}.')

    @property
    def active_orders(self) -> tuple[int, ...]:
        """Suffix lengths actually looked up."""
        return (1,) if self.mode == 'unigram' else self.orders

    @property
    def uses_lookup(self) -> bool:
        """Whether the mode reads hashed tables (and so needs suffix counts)."""
        return self.mode in ('unigram', 'hashed')


class NgramMemory(nn.Module):
    """Gated hashed n-gram memory added to the hidden state of one block."""

    def __init__(self, n_embd: int, config: MemoryConfig):
        """Initializes the memory.

        Args:
            n_embd: Width of the hidden state.
            config: Memory hyperparameters; `mode` must not be 'off'.
        """
        super().__init__()
        self.config = config
        self.hasher = SuffixHasher(
            orders=config.active_orders, heads=config.heads, table_size=config.table_size,
            seed=config.seed,
        )
        # Pad rows so that ZeRO-style sharding splits the table evenly.
        rows = (self.hasher.rows + 63) // 64 * 64
        d_mem = self.hasher.slots * config.head_dim
        self.d_mem = d_mem
        self.table = nn.Embedding(rows, config.head_dim)
        self.w_v = nn.Linear(d_mem, n_embd, bias=False)
        if config.gate == 'context':
            self.w_k = nn.Linear(d_mem, n_embd, bias=False)
            self.gate_bias = nn.Parameter(torch.zeros(1))
        else:
            self.w_k = None
            self.gate_bias = None
        self.register_buffer(
            'slot_order', torch.tensor(self.hasher.slot_order, dtype=torch.long), persistent=False
        )

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initializes so that the memory starts as an exact identity map."""
        nn.init.normal_(self.table.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.w_v.weight)
        if self.w_k is not None:
            bound = math.sqrt(3.0 / self.d_mem)
            nn.init.uniform_(self.w_k.weight, -bound, bound)
            self.gate_bias.fill_(self.config.gate_bias)
        self.slot_order.copy_(torch.tensor(self.hasher.slot_order, dtype=torch.long))

    def forward(self, h: torch.Tensor, idx: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Returns the hidden state with the gated memory read added.

        Args:
            h: (batch, time, n_embd) hidden state entering the block.
            idx: (batch, time) token ids of the same positions.
            support: (batch, time, orders) shrinkage factors, zero for invalid suffixes.

        Returns:
            The updated hidden state.
        """
        e = self.table(self.hasher(idx))
        weight = support.index_select(-1, self.slot_order).unsqueeze(-1)
        e = (e * weight.to(e.dtype)).flatten(-2)
        v = self.w_v(e)
        if self.w_k is None:
            return h + v
        k = self.w_k(e)
        score = (F.rms_norm(h, (h.size(-1),)) * F.rms_norm(k, (k.size(-1),))).sum(-1, keepdim=True)
        alpha = torch.sigmoid(score / math.sqrt(h.size(-1)) + self.gate_bias.to(score.dtype))
        return h + alpha * v

    def parameter_count(self) -> int:
        """Returns the number of added parameters."""
        return sum(p.numel() for p in self.parameters())


class DenseAdapter(nn.Module):
    """Parameter-matched dense control for the n-gram memory (Priority E).

    A residual SiLU MLP at the memory's layer with as many parameters as the
    hashed memory. If it helps as much as the memory, the gain is capacity, not
    conditional lookup.
    """

    def __init__(self, n_embd: int, hidden: int):
        """Initializes the adapter.

        Args:
            n_embd: Width of the hidden state.
            hidden: Width of the adapter's hidden layer.
        """
        super().__init__()
        self.w_in = nn.Linear(n_embd, hidden, bias=False)
        self.w_out = nn.Linear(hidden, n_embd, bias=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initializes so that the adapter starts as an exact identity map."""
        bound = math.sqrt(3.0 / self.w_in.in_features)
        nn.init.uniform_(self.w_in.weight, -bound, bound)
        nn.init.zeros_(self.w_out.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns the hidden state plus the adapter's residual update."""
        return h + self.w_out(F.silu(self.w_in(F.rms_norm(h, (h.size(-1),)))))

    def parameter_count(self) -> int:
        """Returns the number of added parameters."""
        return sum(p.numel() for p in self.parameters())


def matched_dense_hidden(n_embd: int, config: MemoryConfig) -> int:
    """Returns the adapter width that matches the hashed memory's parameter count.

    Args:
        n_embd: Width of the hidden state.
        config: Memory hyperparameters; table sizes and widths of the hashed variant.

    Returns:
        A multiple of 256.
    """
    hashed = dataclasses.replace(config, mode='hashed')
    with torch.device('meta'):
        count = NgramMemory(n_embd, hashed).parameter_count()
    return max(256, round(count / (2 * n_embd) / 256) * 256)


def accumulate_strata(
    losses: torch.Tensor, strata: torch.Tensor, sums: torch.Tensor, tokens: torch.Tensor
) -> None:
    """Adds per-token losses into per-stratum sums in place.

    Args:
        losses: Per-token losses, any shape.
        strata: Stratum index per token, same shape.
        sums: (len(STRATA),) float64 loss sums.
        tokens: (len(STRATA),) int64 token counts.
    """
    flat = strata.reshape(-1)
    sums.scatter_add_(0, flat, losses.reshape(-1).double())
    tokens.scatter_add_(0, flat, torch.ones_like(flat))


def main() -> None:
    """Measures the full-data cost of exact suffix counts (Priority E preprocessing)."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--data', default='fineweb_data/fineweb_train.pt')
    parser.add_argument('--orders', default='2,3')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--table-size', type=int, default=196613)
    args = parser.parse_args()
    orders = tuple(int(n) for n in args.orders.split(','))
    tokens = torch.load(args.data, weights_only=True)['tokens'].long()
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    counts = SuffixCounts.build(tokens, orders=orders, device=args.device)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    print(f'{tokens.numel():,} tokens, orders {orders}: built in {seconds:.2f}s')
    for order, keys in zip(counts.orders, counts.keys):
        print(f'  order {order}: {keys.numel():,} distinct suffixes')
    print(f'  tables: {counts.nbytes() / 2**20:.0f} MiB')
    if args.device.startswith('cuda'):
        print(f'  peak device memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    hasher = SuffixHasher(orders=orders, heads=args.heads, table_size=args.table_size)
    for key, value in hasher.collision_statistics(counts).items():
        print(f'  {key}: {value:.3f}')


if __name__ == '__main__':
    main()
