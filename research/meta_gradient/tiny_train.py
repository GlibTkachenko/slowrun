"""
Research fork of the tiny-track record (commit 45bf3fd, 3.295 val loss in 14.2 min).

With default flags this script runs the record exactly. Every experiment is a
flag, and arms.py names the combinations:

  * --virtual-ranks: run the exact 8-rank meta-gradient (MG) algorithm on 1, 2 or 4 GPUs.
  * A  --aux-split: move the MTP auxiliary objective into the adaptation half only.
  * B  --mg-cproj: activation-metric radius or direction of the temporary step on c_proj.
  * C  --spectral / --muon-eq-r / --mona-beta: update-map variants;
       --diag-steps: the diagnostic of which MG corrections survive Muon.
  * D  --credit: randomized, compensated recovery of the truncated recurrent credit.
  * E  --ngram: causal hashed n-gram memory with count shrinkage; --eval-strata.

Usage:
    torchrun --standalone --nproc_per_node=8 research/meta_gradient/tiny_train.py [flags]
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import math
import time
import hashlib
import json
import argparse
import sys
import shutil
import platform
import subprocess
from types import SimpleNamespace
from dataclasses import dataclass, field
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import wandb
import numpy as np

import tiktoken

_LAB_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.abspath(os.path.join(_LAB_DIR, "..", ".."))
sys.path.insert(1, os.path.join(_REPO_DIR, "tiny"))  # the record's cuda_kernels.py

import credit as credit_lib
import lookahead as lookahead_lib
import mona as mona_lib
import ngram as ngram_lib
import probe as probe_lib
import spectral as spectral_lib
from loader import VirtualRankLoader, probe_sequences

# Bind this rank to its GPU *before* importing cuda_kernels: that module compiles the
# fused CE CUDA kernel at import time, and torch.cuda._compile_kernel binds the
# resulting function to whatever CUDA context is current. Without this, every rank
# compiles against cuda:0, and non-zero ranks later hit "CUDA error: invalid resource
# handle" when launching the kernel on their own device.
if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# Custom CUDA/Triton kernels (fused fp8 softcapped CE with MTP). Imported here, after the
# device bind above, because the CE kernel is compiled against the current CUDA context.
# Without CUDA (CPU smoke tests) a pure-PyTorch twin of the loss is used instead.
if torch.cuda.is_available():
    from cuda_kernels import CE_KERNEL_VOCAB_SIZE, FusedSoftcappedCrossEntropy
else:
    CE_KERNEL_VOCAB_SIZE, FusedSoftcappedCrossEntropy = 50304, None

_script_start = time.time()

# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Train GPT model")
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--num-epochs", type=int, default=15)
parser.add_argument("--patience", type=int, default=-1)
parser.add_argument("--run-name", type=str, default=None,
                    help="Run name under runs/ (default: random 6-char string)")
parser.add_argument("--scalar-lr", type=float, default=0.25)
parser.add_argument("--matrix-lr", type=float, default=0.04)
parser.add_argument("--embedding-lr", type=float, default=0.15)
parser.add_argument("--unembedding-lr", type=float, default=0.001)
parser.add_argument("--weight-decay", type=float, default=0.8)
# WD follows a 3-phase schedule: hold → decay → ramp
#   [0, wd-phase1-epoch]:          hold at --weight-decay
#   [wd-phase1-epoch, wd-phase2-epoch]: decay to --wd-mid
#   [wd-phase2-epoch, num-epochs]:      ramp up to --wd-end
parser.add_argument("--wd-phase1-epoch", type=int, default=2)
parser.add_argument("--wd-phase2-epoch", type=int, default=8)
parser.add_argument("--wd-mid", type=float, default=0.1)
parser.add_argument("--wd-end", type=float, default=0.93)
parser.add_argument("--warmup-ratio", type=float, default=0.0)
parser.add_argument("--warmdown-ratio", type=float, default=0.4)
parser.add_argument("--total-batch-size", type=int, default=524288)
parser.add_argument("--save-result", type=str, default="")
parser.add_argument("--n_layer", type=int, default=16)
parser.add_argument(
    "--num-iterations",
    type=int,
    default=2,
    help="Maximum recurrent iterations through the network",
)
parser.add_argument(
    "--iteration-schedule",
    type=str,
    default="late-transition",
    choices=["constant", "late-transition"],
    help="Schedule recurrent iterations over training",
)
parser.add_argument(
    "--min-iterations",
    type=int,
    default=1,
    help="Initial recurrent iterations for non-constant schedules",
)
parser.add_argument(
    "--iteration-transition-ratio",
    type=float,
    default=0.3,
    help=(
        "Fraction of training spent at --num-iterations for late-transition "
        "schedules; kept separate from LR warmdown"
    ),
)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--n_embd", type=int, default=1024)
parser.add_argument("--lr_multiplier", type=float, default=0.8)
parser.add_argument("--input_bin", type=str, default=None)
parser.add_argument("--input_val_bin", type=str, default=None)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--update-ema-every", type=int, default=10)
parser.add_argument("--ema-decay-per-epoch", type=float, default=0.15)
parser.add_argument("--swa-last-epochs", type=int, default=4,
                    help="SWA: cosine-cycle LR in last N epochs for checkpoint diversity (0=off)")
parser.add_argument("--no-doc-shuffle", action="store_true",
                    help="Disable per-epoch document reshuffling (still shuffles batch order)")
parser.add_argument("--max-train-steps", type=int, default=3040,
                    help="Stop after this many optimizer steps. Use 0 to train for all epochs.")
parser.add_argument("--xsa-mode", choices=("off", "first6", "all"), default="all",
                    help="Exclusive self-attention schedule.")
parser.add_argument("--mtp-predict", type=int, default=3,
                    help="MTP: number of future tokens predicted from each position (1 = plain next-token).")
parser.add_argument("--mtp-anneal-frac", type=float, default=0.66,
                    help="Fraction of training over which the extra MTP heads decay to zero weight.")
# --- every-step meta-gradient step (default for this entry; --mg-every 0 = entry 14) ---
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mg-every", type=int, default=1, help="meta-gradient step every N steps (1 = every step, the record; 0 = off = entry 14)")
parser.add_argument("--mg-step-norm", type=float, default=0.5, help="L2 norm of the temporary step on the plastic matrices")
parser.add_argument("--mg-step-schedule", type=str, default="const", choices=["const", "lr"])
parser.add_argument("--mg-plastic", type=str, default="mlp_all", choices=["matrix_all", "mlp_all"])
# --- research fork (research/meta_gradient); every default reproduces the record ---
lab = parser.add_argument_group("research fork")
lab.add_argument("--arm", type=str, default="base", help="Arm label written to result.json")
lab.add_argument("--virtual-ranks", type=int, default=8,
                 help="Data-parallel ranks emulated by the MG split (record: 8 GPUs)")
lab.add_argument("--data-seed", type=int, default=None,
                 help="Offset every data shuffle (default: the record's data order)")
lab.add_argument("--seq-len", type=int, default=2048)
lab.add_argument("--eval-tokens", type=int, default=10_000_000)
lab.add_argument("--log-every", type=int, default=25, help="Steps between research-metric logs")
lab.add_argument("--cleanup-checkpoints", action="store_true",
                 help="Delete epoch checkpoints after the final averaging")
lab.add_argument("--no-save-model", action="store_true")
# A: objective separation
lab.add_argument("--aux-split", choices=sorted(lookahead_lib.AUX_SPLITS), default="shared",
                 help="MTP extra-offset multipliers: shared (1,1), adapt (2,0), query (0,2)")
# B: activation-metric temporary step on the MLP output projections
lab.add_argument("--mg-act-stats", action="store_true",
                 help="Track c_proj input moments and log functional displacements")
lab.add_argument("--mg-cproj", choices=lookahead_lib.CPROJ_MODES, default="global")
lab.add_argument("--mg-act-radius", type=float, default=0.0,
                 help="Functional radius rho; 0 = calibration x multiplier")
lab.add_argument("--mg-act-calibration", type=str, default=None,
                 help="mg_calibration.json of a baseline run with --mg-act-stats")
lab.add_argument("--mg-act-radius-mult", type=float, default=1.0)
lab.add_argument("--mg-act-damping", type=float, default=0.1)
lab.add_argument("--mg-act-decay", type=float, default=0.9)
lab.add_argument("--mg-act-stride", type=int, default=16)
lab.add_argument("--mg-cproj-radii", type=str, default=None,
                 help="mg_calibration.json whose frob_disp sets euclid_layer radii")
# C: update maps and the survival diagnostic
lab.add_argument("--muon-eq-r", action="store_true", help="MuonEq-R row normalization (PR #77)")
lab.add_argument("--spectral", choices=spectral_lib.SPECTRAL_METHODS, default="polar_express")
lab.add_argument("--spectral-c", type=float, default=0.5)
lab.add_argument("--spectral-eps", type=float, default=0.1,
                 help="Regularization relative to the RMS singular value")
lab.add_argument("--spectral-c-late", type=float, default=None,
                 help="Exponent after --spectral-switch-frac (two-phase schedule)")
lab.add_argument("--spectral-switch-frac", type=float, default=0.7)
lab.add_argument("--spectral-norm", choices=spectral_lib.NORM_MATCHING, default="polar_express",
                 help="Update norm of alternative maps: Polar Express's on the same input, or sqrt(min(m, n))")
lab.add_argument("--spectral-fp64", action="store_true",
                 help="Eigendecompose in float64 (needed for --spectral-eps well below 1e-2)")
lab.add_argument("--mona-beta", type=float, default=0.0,
                 help="MONA gradient-difference EMA decay (0 = off)")
lab.add_argument("--mona-alpha", type=float, default=None, help="Default -1/(2(1-beta))")
lab.add_argument("--diag-steps", type=str, default="",
                 help="Comma-separated steps for the MG survival diagnostic")
lab.add_argument("--diag-params", type=str, default="",
                 help="Comma-separated matrices for the singular-basis analysis")
# D: recurrent credit
lab.add_argument("--credit", choices=credit_lib.CREDIT_MODES, default="trunc")
lab.add_argument("--credit-p", type=float, default=0.25)
# E: conditional n-gram memory
lab.add_argument("--ngram", choices=ngram_lib.MEMORY_MODES, default="off")
lab.add_argument("--ngram-orders", type=str, default="2,3")
lab.add_argument("--ngram-heads", type=int, default=2)
lab.add_argument("--ngram-head-dim", type=int, default=32)
lab.add_argument("--ngram-table-size", type=int, default=196613)
lab.add_argument("--ngram-layer", type=int, default=2)
lab.add_argument("--ngram-gate", choices=ngram_lib.GATE_MODES, default="context")
lab.add_argument("--ngram-gate-bias", type=float, default=0.0)
lab.add_argument("--ngram-kappa", type=float, default=0.0)
lab.add_argument("--ngram-lr", type=float, default=None, help="Default: the embedding LR")
lab.add_argument("--param-digest", action="store_true",
                 help="Record a SHA-256 of the final parameters (for exact-equality checks)")
lab.add_argument("--eval-strata", action="store_true",
                 help="Final held-out loss by training support of the causal suffix")
args = parser.parse_args()

# Resolve output path
if args.output_json and not args.save_result:
    args.save_result = args.output_json

# =============================================================================
# Hyperparameters
# =============================================================================

# RLM
NUM_ITERATIONS = args.num_iterations

# Architecture
N_EMBD = args.n_embd if args.n_embd is not None else 768
N_HEAD = args.n_head if args.n_head is not None else 6
HEAD_DIM = N_EMBD // N_HEAD
if NUM_ITERATIONS < 1:
    raise ValueError("--num-iterations must be >= 1")
ITERATION_TRANSITION_RATIO = min(max(args.iteration_transition_ratio, 0.0), 1.0)
if args.iteration_schedule != "constant":
    if args.min_iterations < 1:
        raise ValueError("--min-iterations must be >= 1")
    if args.min_iterations > NUM_ITERATIONS:
        raise ValueError("--min-iterations must be <= --num-iterations")
DEPTH = args.n_layer
MAX_SEQ_LEN = args.seq_len
WINDOW_PATTERN = "SSSL"
TOTAL_BATCH_SIZE = args.total_batch_size
EVAL_TOKENS = args.eval_tokens
DATA_DIR = "fineweb_data"
BOS_ID = 50256  # <|endoftext|>
RUNS_DIR = "runs"

# Base optimizer hyperparameters
BASE_MATRIX_LR = args.matrix_lr
BASE_SCALAR_LR = args.scalar_lr
BASE_EMBEDDING_LR = args.embedding_lr
BASE_UNEMBEDDING_LR = args.unembedding_lr

# Apply LR multiplier if provided (scales all LRs uniformly)
_lr_mult = args.lr_multiplier if args.lr_multiplier is not None else 1.0
MATRIX_LR = BASE_MATRIX_LR * _lr_mult
UNEMBEDDING_LR = BASE_UNEMBEDDING_LR * _lr_mult
EMBEDDING_LR = BASE_EMBEDDING_LR * _lr_mult
SCALAR_LR = BASE_SCALAR_LR * _lr_mult

WEIGHT_DECAY = args.weight_decay
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = args.warmup_ratio
WARMDOWN_RATIO = args.warmdown_ratio
FINAL_LR_FRAC = 0.0
TRAIN_BACKPROP_ITERATIONS = 1
if TRAIN_BACKPROP_ITERATIONS < 1:
    raise ValueError("TRAIN_BACKPROP_ITERATIONS must be >= 1")

# Multi-token prediction (MTP) ------------------------------------------------
# fp8 scales for the fused softcapped cross-entropy kernel.
MTP_X_S = 100 / 448
MTP_W_S = 1.6 / 448
MTP_GRAD_S = 0.75 / 448

# Per-offset loss weights: predict targets[t], targets[t+1], ... with these weights.
# Extra offsets anneal to zero over training so it ends as plain next-token prediction.
MTP_START_WEIGHTS = [1.0, 0.5, 0.25, 0.125]

# Research fork -----------------------------------------------------------------
# Input moments of c_proj are needed by the activation modes and by calibration runs.
ACT_STATS = args.mg_act_stats or args.mg_cproj in ("act_radius", "act_metric")
NGRAM_LR = args.ngram_lr if args.ngram_lr is not None else EMBEDDING_LR
MEMORY_CONFIG = ngram_lib.MemoryConfig(
    mode=args.ngram,
    orders=tuple(int(n) for n in args.ngram_orders.split(",")),
    heads=args.ngram_heads,
    head_dim=args.ngram_head_dim,
    table_size=args.ngram_table_size,
    layer=args.ngram_layer,
    gate=args.ngram_gate,
    gate_bias=args.ngram_gate_bias,
    kappa=args.ngram_kappa,
    seed=args.seed,
)
CREDIT_SCALE = credit_lib.path_scale(args.credit, args.credit_p)
MONA = None
if args.mona_beta > 0:
    MONA = (args.mona_beta,
            args.mona_alpha if args.mona_alpha is not None else mona_lib.default_alpha(args.mona_beta))
if args.spectral == "augmented_polar" and (args.spectral_c != 0.5 or args.spectral_c_late is not None):
    raise ValueError("--spectral augmented_polar implements only exponent 0.5")

# =============================================================================
# Utilities
# =============================================================================

def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return True, int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return False, 0, 0, 1

def print0(s="", **kwargs):
    if int(os.environ.get('RANK', 0)) == 0:
        print(s, **kwargs)

@dataclass(frozen=True)
class IterationScheduleStage:
    start_frac: float
    end_frac: float
    iterations: int


@dataclass(frozen=True)
class IterationSchedule:
    stages: tuple
    avg_iterations: float


def build_iteration_schedule(
    schedule_name,
    min_iterations,
    max_iterations,
    n_layer,
    transition_ratio,
):
    if max_iterations < 1:
        raise ValueError("--num-iterations must be >= 1")
    if n_layer <= 0:
        raise ValueError("--n_layer must be > 0")

    if schedule_name == "constant":
        avg_iterations = float(max_iterations)
        return IterationSchedule(
            stages=(IterationScheduleStage(0.0, 1.0, max_iterations),),
            avg_iterations=avg_iterations,
        )

    if min_iterations < 1:
        raise ValueError("--min-iterations must be >= 1")
    if min_iterations > max_iterations:
        raise ValueError("--min-iterations must be <= --num-iterations")

    transition_duration = min(max(transition_ratio, 0.0), 1.0)

    if min_iterations == max_iterations or transition_duration >= 1.0 - 1e-9:
        stages = (IterationScheduleStage(0.0, 1.0, max_iterations),)
    elif transition_duration <= 1e-9:
        stages = (IterationScheduleStage(0.0, 1.0, min_iterations),)
    else:
        transition_start = 1.0 - transition_duration
        stages = (
            IterationScheduleStage(0.0, transition_start, min_iterations),
            IterationScheduleStage(transition_start, 1.0, max_iterations),
        )
    avg_iterations = sum(
        (stage.end_frac - stage.start_frac) * stage.iterations
        for stage in stages
    )
    return IterationSchedule(
        stages=stages,
        avg_iterations=avg_iterations,
    )


def get_scheduled_iterations(schedule, step, total_steps):
    if total_steps <= 0:
        return schedule.stages[-1].iterations
    frac = min(max(step / total_steps, 0.0), 1.0)
    for stage in schedule.stages:
        if frac < stage.end_frac or stage is schedule.stages[-1]:
            return stage.iterations
    return schedule.stages[-1].iterations


def format_iteration_schedule(schedule):
    return ", ".join(
        f"{stage.start_frac:.3f}-{stage.end_frac:.3f}: {stage.iterations}x"
        for stage in schedule.stages
    )


def iteration_schedule_counts(schedule):
    return tuple(dict.fromkeys(stage.iterations for stage in schedule.stages))


def get_expected_scheduled_iterations(schedule, step, total_steps):
    return float(get_scheduled_iterations(schedule, step, total_steps))


ITERATION_SCHEDULE = build_iteration_schedule(
    args.iteration_schedule,
    args.min_iterations,
    NUM_ITERATIONS,
    DEPTH,
    ITERATION_TRANSITION_RATIO,
)

class DummyWandb:
    def __init__(self): self.summary = {}
    def log(self, *a, **kw): pass
    def finish(self): pass

class TeeStream:
    """Save terminal output to file."""
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")
    def write(self, data):
        for stream in self.streams: stream.write(data)
        return len(data)
    def flush(self):
        for stream in self.streams: stream.flush()
    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)
    def fileno(self):
        return self.streams[0].fileno()

def resolve_run_dir(run_name):
    if run_name:
        actual_run_name = run_name
    else:
        actual_run_name = time.strftime('%Y%m%d_%H%M%S')
    return actual_run_name, os.path.join(RUNS_DIR, actual_run_name)

# =============================================================================
# Flash Attention (FA3 on Hopper, SDPA fallback elsewhere)
# =============================================================================
def _load_fa3():
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('kernels-community/flash-attn3', version=1)
    except ImportError:
        print0("Warning: kernels package not found. Install with: pip install -U kernels")
        return None
    except Exception as e:
        print0(f"Warning: Failed to load FA3 kernel: {e}")
        return None

_fa3 = _load_fa3()

def _sdpa_attention(q, k, v, window_size, enable_gqa):
    Tq, Tk = q.size(2), k.size(2)
    window = window_size[0]
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
    if Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k, v = k[:, :, start:, :], v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Flash Attention for training. q,k,v: (B, T, H, D)."""
    if _fa3 is not None:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)

flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)

# =============================================================================
# GPT Model
# =============================================================================

@dataclass
class GPTConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 32768
    n_layer: int = DEPTH
    n_head: int = N_HEAD
    n_kv_head: int = N_HEAD
    n_embd: int = N_EMBD
    window_pattern: str = WINDOW_PATTERN
    dropout: float = 0.05
    device_batch_size: int = 32
    xsa_mode: str = "all"
    xsa_eps: float = 1e-4
    num_iterations: int = NUM_ITERATIONS
    act_stats: bool = False
    act_stride: int = 16
    act_decay: float = 0.9
    credit_scale: float = 1.0
    memory: ngram_lib.MemoryConfig = field(default_factory=ngram_lib.MemoryConfig)

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    """Value Embedding on alternating layers, last layer always included."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        # Per-head attention gate: enables context-based attention no-op
        self.attn_gate_channels = 12
        self.attn_gate = nn.Linear(self.attn_gate_channels, self.n_head, bias=False)
        # Determine if this is a long-window layer for partial key offset
        pattern = config.window_pattern.upper()
        char = pattern[layer_idx % len(pattern)]
        self.use_key_offset = (char == 'L') or (layer_idx == config.n_layer - 1)
        self.xsa_eps = config.xsa_eps

    def forward(self, x, ve, cos_sin, window_size, xsa_alpha=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        # Value residual (ResFormer)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        # Partial key offset: shift stationary dims forward by 1 on long-window layers
        if self.use_key_offset and T > 1:
            k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:].clone()
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        if xsa_alpha is not None:
            v_ref = v
            if self.n_kv_head != self.n_head:
                v_ref = v_ref.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
            alpha = torch.tanh(xsa_alpha).type_as(y).view(1, 1, self.n_head, 1)
            v_hat = v_ref / v_ref.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(self.xsa_eps)
            xsa_coeff = ((y * v_hat).sum(dim=-1, keepdim=True)) * alpha
            y = torch.addcmul(y, v_hat, xsa_coeff, value=-1.0)
        # Per-head attention gate (sparse gated attention, zero-init → sigmoid(0)=0.5 at start)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate_channels])).unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 256 * ((8 * config.n_embd // 3 + 255) // 256)
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        # Diagonal second moment of the c_proj inputs for the activation-metric MG step.
        self.act_stats = config.act_stats
        self.act_stride, self.act_decay = config.act_stride, config.act_decay
        if self.act_stats:
            self.register_buffer("act_moment", torch.zeros(hidden), persistent=False)
            self.register_buffer("act_count", torch.zeros(()), persistent=False)

    def forward(self, x):
        h = F.silu(self.c_gate(x)) * self.c_fc(x)
        if self.act_stats and self.training:
            lookahead_lib.update_input_moment_(
                self.act_moment, self.act_count, h, stride=self.act_stride, decay=self.act_decay)
        return self.c_proj(h)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, xsa_alpha=None):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, xsa_alpha=xsa_alpha)
        x = x + self.mlp(norm(x))
        return x


class CastedLinearT(nn.Module):
    """Linear layer with weight stored as (in_features, out_features) — matches the layout
    expected by FusedSoftcappedCrossEntropy without any transpose at call time."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.bfloat16))

    def forward(self, x):
        return x @ self.weight.type_as(x)


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab}")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = CastedLinearT(config.n_embd, padded_vocab)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.ve_projs = nn.ModuleDict({str(i): nn.Linear(config.n_embd, kv_dim, bias=False) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # U-Net skip connections: encoder layer i → decoder layer (n_layer - 1 - i)
        self.encoder_layers = config.n_layer // 2
        self.skip_weights = nn.Parameter(torch.ones(self.encoder_layers))
        self.xsa_alphas = nn.Parameter(torch.zeros(config.n_layer, config.n_head))
        self.ngram = (ngram_lib.NgramMemory(config.n_embd, config.memory)
                      if config.memory.uses_lookup else None)
        # Parameter-matched dense control at the same layer (Priority E).
        self.dense_adapter = (ngram_lib.DenseAdapter(
            config.n_embd, ngram_lib.matched_dense_hidden(config.n_embd, config.memory))
            if config.memory.mode == "dense" else None)
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        self.resid_lambdas.fill_(1.1)
        self.x0_lambdas.fill_(0.1)
        self.xsa_alphas.zero_()
        for proj in self.ve_projs.values():
            torch.nn.init.uniform_(proj.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.attn_gate.weight)
        self.skip_weights.fill_(1.0)
        if self.ngram is not None:
            self.ngram.init_weights()
        if self.dense_adapter is not None:
            self.dense_adapter.init_weights()
        for block in self.transformer.h:
            if block.mlp.act_stats:
                block.mlp.act_moment.zero_()
                block.mlp.act_count.zero_()
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        # Half-truncated RoPE: only rotate half the dims, leave the rest stationary
        half = head_dim // 4  # number of frequency pairs for the rotated half
        inv_freq = 1.0 / (base ** (torch.arange(0, half * 2, 2, dtype=torch.float32, device=device) / (half * 2)))
        # Pad with zeros for the stationary half
        inv_freq = torch.cat([inv_freq, torch.zeros(head_dim // 2 - half, dtype=torch.float32, device=device)])
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)  # final layer always full context
        return sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def _xsa_enabled(self, layer_idx):
        if self.config.xsa_mode == "off":
            return False
        if self.config.xsa_mode == "first6":
            return layer_idx < min(6, self.config.n_layer)
        if self.config.xsa_mode == "all":
            return True
        raise ValueError(f"unknown xsa_mode: {self.config.xsa_mode}")

    def _avg_causal_attended_keys(self, window, seq_len):
        if window < 0 or window >= seq_len - 1:
            return (seq_len + 1) / 2
        max_keys = min(window + 1, seq_len)
        return max_keys - max_keys * (max_keys - 1) / (2 * seq_len)

    def estimate_flops(self, num_iterations=None):
        active_num_iterations = (
            self.config.num_iterations if num_iterations is None else num_iterations
        )
        if not 1 <= active_num_iterations <= self.config.num_iterations + 2:
            raise ValueError(
                f"num_iterations must be in [1, {self.config.num_iterations + 2}], "
                f"got {active_num_iterations}"
            )
        nparams = sum(p.numel() for p in self.parameters())
        shared_recurrent_params = (
            sum(p.numel() for p in self.transformer.h.parameters())
            + sum(p.numel() for p in self.ve_projs.parameters())
        )
        # Exclude non-matmul params: embedding lookup + elementwise scalars
        nparams_exclude = (self.transformer.wte.weight.numel()
                          + self.resid_lambdas.numel()
                          + self.x0_lambdas.numel()
                          + self.skip_weights.numel()
                          + self.xsa_alphas.numel()
                          + (self.ngram.table.weight.numel() if self.ngram is not None else 0))
        extra_nonshared_params = (
            nparams - nparams_exclude - shared_recurrent_params
        )
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Exact causal sliding-window attention FLOPs: 12 * h * q * E[keys attended per query]
        attn_flops = active_num_iterations * sum(12 * h * q * self._avg_causal_attended_keys(w[0], t) for w in self.window_sizes)
        effective_params = (
            extra_nonshared_params
            + active_num_iterations * shared_recurrent_params
        )
        return 6 * effective_params + attn_flops

    def setup_optimizer(self):
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate attn_gate params (small, Adam-optimized) from matrix params (Muon)
        attn_gate_params = [block.attn.attn_gate.weight for block in self.transformer.h]
        attn_gate_ids = {id(p) for p in attn_gate_params}
        all_h_params = list(self.transformer.h.parameters()) + list(self.ve_projs.parameters())
        matrix_params = [
            p
            for p in all_h_params
            if id(p) not in attn_gate_ids
        ]
        embed_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        skip_params = [self.skip_weights]
        xsa_params = [self.xsa_alphas] if self.config.xsa_mode != "off" else []

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=UNEMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
            dict(kind='adamw', params=embed_params, lr=EMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
            dict(kind='adamw', params=resid_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=SCALAR_LR, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=skip_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=attn_gate_params, lr=SCALAR_LR, betas=(0.9, 0.99), eps=1e-10, weight_decay=0.0),
        ]
        if xsa_params:
            param_groups.append(dict(kind='adamw', params=xsa_params, lr=SCALAR_LR,
                                     betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if self.ngram is not None:
            # Engram trains its tables with Adam and no weight decay.
            param_groups.append(dict(kind='adamw', params=[self.ngram.table.weight], lr=NGRAM_LR,
                                     betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0))
            if self.ngram.gate_bias is not None:
                param_groups.append(dict(kind='adamw', params=[self.ngram.gate_bias], lr=SCALAR_LR,
                                         betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0))
            matrix_params = matrix_params + [self.ngram.w_v.weight] + (
                [self.ngram.w_k.weight] if self.ngram.w_k is not None else [])
        if self.dense_adapter is not None:
            matrix_params = matrix_params + [self.dense_adapter.w_in.weight, self.dense_adapter.w_out.weight]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=group_params, lr=MATRIX_LR,
                                     momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=WEIGHT_DECAY,
                                     eq_r=args.muon_eq_r, mona=MONA, spectral=None))

        optimizer = DistMuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _run_network_once(self, x, x0, cos_sin, memory_inputs=None):
        skip_connections = []
        for i, block in enumerate(self.transformer.h):
            if i >= self.encoder_layers and skip_connections:
                skip = skip_connections.pop()
                x = x + self.skip_weights[i - self.encoder_layers] * skip

            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if i == self.config.memory.layer:
                if memory_inputs is not None:
                    x = self.ngram(x, *memory_inputs)
                elif self.dense_adapter is not None:
                    x = self.dense_adapter(x)
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            xsa_alpha = self.xsa_alphas[i] if self._xsa_enabled(i) else None
            x = block(x, ve, cos_sin, self.window_sizes[i], xsa_alpha=xsa_alpha)
            if i < self.encoder_layers:
                skip_connections.append(x)

        return x

    def _mtp_loss(self, x, targets, mtp_weights):
        """Fused softcapped cross-entropy with multi-token prediction.

        a SINGLE lm_head produces one set of logits per position,
        scored against targets[t], targets[t+1], ... targets[t+k-1] with weights
        `mtp_weights` (no separate prediction heads). The CUDA kernel computes the
        logits internally via an fp8 matmul, so the lm_head weight is passed transposed
        to (n_embd, vocab) -- the layout the kernel expects.
        """
        x_flat = x.reshape(-1, x.size(-1)).contiguous()
        if FusedSoftcappedCrossEntropy is None:
            return mtp_loss_reference(x_flat, targets.reshape(-1), mtp_weights, self.lm_head.weight)
        assert self.lm_head.weight.size(1) == CE_KERNEL_VOCAB_SIZE, (
            f"lm_head vocab {self.lm_head.weight.size(1)} != fused kernel VOCAB_SIZE {CE_KERNEL_VOCAB_SIZE}")
        return FusedSoftcappedCrossEntropy.apply(
            x_flat, targets.reshape(-1), mtp_weights, self.lm_head.weight,
            MTP_X_S, MTP_W_S, MTP_GRAD_S, 1.0)

    def forward(
        self,
        idx,
        targets=None,
        loss_reduction='mean',
        num_iterations=None,
        mtp_weights=None,
        full_credit=False,
        ngram_support=None,
    ):
        """Runs the recurrent model.

        Research arguments (defaults reproduce the record):
            full_credit: Differentiate through every recurrent pass (Priority D); the
                gradient entering earlier passes is scaled by config.credit_scale.
            ngram_support: (B, T, orders) shrinkage factors of the n-gram memory.
        """
        B, T = idx.size()
        active_num_iterations = (
            self.config.num_iterations if num_iterations is None else num_iterations
        )
        if not 1 <= active_num_iterations <= self.config.num_iterations + 2:
            raise ValueError(
                f"num_iterations must be in [1, {self.config.num_iterations + 2}], "
                f"got {active_num_iterations}"
            )

        if (self.ngram is None) != (ngram_support is None):
            raise ValueError("ngram_support must be given exactly when the n-gram memory is on")
        memory_inputs = None if self.ngram is None else (idx, ngram_support)

        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x_emb = x  # original embedded input, used as the recurrent re-injection anchor
        grad_start_iteration = 0
        if self.training and not full_credit:
            backprop_iterations = min(TRAIN_BACKPROP_ITERATIONS, active_num_iterations)
            grad_start_iteration = active_num_iterations - backprop_iterations
        for iteration in range(active_num_iterations):
            # Re-inject the original token embedding (not the drifting recurrent
            # state) as the per-layer x0 anchor + value-embedding source each pass.
            if self.training and iteration < grad_start_iteration:
                with torch.no_grad():
                    x = self._run_network_once(x, x_emb, cos_sin, memory_inputs)
                    x = norm(x)
                x = x.detach()
            else:
                x = self._run_network_once(x, x_emb, cos_sin, memory_inputs)
                x = norm(x)
                if (full_credit and self.training
                        and credit_lib.is_credit_boundary(iteration, active_num_iterations)):
                    # g_hat = g_direct + (Z/p) g_path: every truncated path crosses this boundary once.
                    x = credit_lib.scale_backward(x, self.config.credit_scale)
        x = norm(x)
        if targets is not None:
            # MTP training path: single lm_head, shifted targets, fused fp8 softcapped CE.
            # Eval (mtp_weights=None / model.eval()) keeps the plain bf16 path below so
            # validation loss/bpb stay on the true next-token objective.
            if self.training and mtp_weights is not None:
                return self._mtp_loss(x, targets, mtp_weights)
            logits = self.lm_head(x)[..., :self.config.vocab_size].float()
            logits = 15 * torch.tanh(logits / 15)  # softcap
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)  # softcap
        return logits

def mtp_loss_reference(
    x: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, lm_head_weight: torch.Tensor,
    cap: float = 15.0,
) -> torch.Tensor:
    """Returns the fused kernel's per-row MTP loss without fp8, for CPU smoke tests.

    Row r scores one softcapped logit vector against targets[r + k] with weight
    weights[k], ignoring offsets past the end of the flattened batch.

    Args:
        x: (rows, n_embd) final hidden states.
        targets: (rows,) next-token targets, -1 for ignored positions.
        weights: (n_predict,) offset weights.
        lm_head_weight: (n_embd, vocab) unembedding.
        cap: Softcap of the logits.

    Returns:
        (rows,) float32 weighted losses.
    """
    logits = (x @ lm_head_weight.type_as(x)).float()
    logits = cap * torch.tanh(logits / cap)
    lse = torch.logsumexp(logits, dim=-1)
    loss = torch.zeros_like(lse)
    for k in range(weights.numel()):
        shifted = torch.full_like(targets, -1)
        shifted[: targets.numel() - k] = targets[k:]
        picked = logits.gather(1, shifted.clamp_min(0)[:, None]).squeeze(1)
        loss = loss + weights[k] * torch.where(shifted >= 0, lse - picked, torch.zeros_like(lse))
    return loss


# =============================================================================
# Optimizer: MuonAdamW (Muon for matrices, AdamW for embeddings/scalars)
# =============================================================================

# Polar Express coefficients for orthogonalization
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    p.add_(exp_avg / ((exp_avg_sq / bias2).sqrt() + eps_t), alpha=-(lr_t / bias1))

def _normuon_tail(g, stacked_params, second_momentum_buffer, lr_t, wd_t, beta2_t, active, red_dim):
    """Variance reduction, cautious weight decay and the update, shared by both Muon steps."""
    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_update = (1 - beta2) * active.to(second_momentum_buffer.dtype)
    second_momentum_buffer.mul_(1 - second_momentum_update).add_(
        v_mean.to(dtype=second_momentum_buffer.dtype) * second_momentum_update
    )
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_((lr * g + lr * wd * stacked_params * mask) * active)


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, active_mask, ns_steps, red_dim, eq_r=False):
    momentum = momentum_t.to(stacked_grads.dtype)
    active = active_mask.to(stacked_grads.dtype)
    momentum_update = (1 - momentum) * active
    momentum_buffer.mul_(1 - momentum_update).add_(stacked_grads * momentum_update)
    g = stacked_grads.lerp(momentum_buffer, momentum) * active
    if eq_r:
        # MuonEq-R row normalization, as in the one-hour record (absent from the tiny record).
        g = g / g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7).to(g.dtype)
    # Polar Express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X
    _normuon_tail(g, stacked_params, second_momentum_buffer, lr_t, wd_t, beta2_t, active, red_dim)


def muon_step_spectral(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                       momentum_t, lr_t, wd_t, beta2_t, active_mask, red_dim, eq_r, method,
                       exponent, eps, dtype, norm):
    """The record Muon step with Polar Express replaced by a spectral response (Priority C).

    Runs eagerly: the eigendecomposition dominates its cost. By default the output is
    rescaled per matrix to the norm Polar Express would have produced on the same
    input and cast to bfloat16 like it, so only the direction of the update changes.
    """
    momentum = momentum_t.to(stacked_grads.dtype)
    active = active_mask.to(stacked_grads.dtype)
    momentum_update = (1 - momentum) * active
    momentum_buffer.mul_(1 - momentum_update).add_(stacked_grads * momentum_update)
    g = stacked_grads.lerp(momentum_buffer, momentum) * active
    if eq_r:
        g = g / g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7).to(g.dtype)
    g = spectral_lib.apply_map(g, method=method, exponent=exponent, eps=eps, dtype=dtype,
                               norm=norm).bfloat16()
    _normuon_tail(g, stacked_params, second_momentum_buffer, lr_t, wd_t, beta2_t, active, red_dim)


# MONA's correction with one EMA buffer (research/meta_gradient/mona.py has the derivation).
mona_correct_ = torch.compile(mona_lib.mona_correct_, dynamic=False, fullgraph=True)


def _collective(collective, *args, **kwargs):
    """Launches a collective and returns its future.

    NCCL runs it asynchronously exactly as the record does. Gloo (CPU smoke tests) lacks
    futures for some collectives, so it runs synchronously behind a completed future.
    """
    if dist.get_backend() == "nccl":
        return collective(*args, async_op=True, **kwargs).get_future()
    collective(*args, **kwargs)
    future = torch.futures.Future()
    future.set_result(None)
    return future


class DistMuonAdamW(torch.optim.Optimizer):
    """Distributed MuonAdamW with ZeRO-2 style sharding."""
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0)
        self._adamw_lr_t = torch.tensor(0.0)
        self._adamw_beta1_t = torch.tensor(0.0)
        self._adamw_beta2_t = torch.tensor(0.0)
        self._adamw_eps_t = torch.tensor(0.0)
        self._adamw_wd_t = torch.tensor(0.0)
        self._muon_momentum_t = torch.tensor(0.0)
        self._muon_lr_t = torch.tensor(0.0)
        self._muon_wd_t = torch.tensor(0.0)
        self._muon_beta2_t = torch.tensor(0.0)
        self._mona_beta_t = torch.tensor(0.0)
        self._mona_alpha_t = torch.tensor(0.0)
        # Evidence that research options actually ran (checked by smoke tests).
        self.calls = {"spectral": 0, "mona": 0}

    def _reduce_adamw(self, group, world_size):
        infos = {}
        for p in group['params']:
            grad = p.grad
            if grad is None:
                continue
            if p.numel() < 1024 or grad.shape[0] % world_size != 0:
                future = _collective(dist.all_reduce, grad, op=dist.ReduceOp.AVG)
                infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = _collective(dist.reduce_scatter_tensor, grad_slice, grad, op=dist.ReduceOp.AVG)
                infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=infos)

    def _reduce_muon(self, group, world_size):
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        stacked_grads = torch.empty(padded, *shape, dtype=dtype, device=device)
        active_mask = torch.zeros(
            (padded,) + (1,) * len(shape), dtype=torch.bool, device=device
        )
        for i, p in enumerate(params):
            if p.grad is None:
                stacked_grads[i].zero_()
            else:
                stacked_grads[i].copy_(p.grad)
                active_mask[i].fill_(True)
        if len(params) < padded:
            stacked_grads[len(params):].zero_()
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = _collective(dist.reduce_scatter_tensor, grad_chunk, stacked_grads, op=dist.ReduceOp.AVG)
        return dict(
            future=future,
            grad_chunk=grad_chunk,
            stacked_grads=stacked_grads,
            active_mask=active_mask,
            chunk_size=chunk_size,
        )

    def _compute_adamw(self, group, info, gather_list, rank, world_size):
        for p, pinfo in info['param_infos'].items():
            pinfo['future'].wait()
            state = self.state[p]
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p_slice, pinfo['grad_slice'], state['exp_avg'], state['exp_avg_sq'],
                           self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                           self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
            if not pinfo['is_small']:
                future = _collective(dist.all_gather_into_tensor, p, p_slice)
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group, info, gather_list, rank):
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            s = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(s, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        updated = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        if num_owned > 0:
            owned = torch.stack([params[start_idx + i] for i in range(num_owned)])
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            grads = info['grad_chunk'][:num_owned]
            if group.get("mona") is not None:
                if "mona_ema" not in state:
                    state["mona_ema"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
                self._mona_beta_t.fill_(group["mona"][0])
                self._mona_alpha_t.fill_(group["mona"][1])
                mona_correct_(grads, state["mona_ema"][:num_owned], self._mona_beta_t, self._mona_alpha_t)
                self.calls["mona"] += 1
            if group.get("spectral") is None:
                muon_step_fused(grads, owned,
                              state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                              self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                              info["active_mask"][start_idx:start_idx + num_owned],
                              group["ns_steps"], red_dim, group.get("eq_r", False))
            else:
                self.calls["spectral"] += 1
                muon_step_spectral(grads, owned,
                                   state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                                   self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                                   info["active_mask"][start_idx:start_idx + num_owned],
                                   red_dim, group.get("eq_r", False), *group["spectral"])
            updated[:num_owned].copy_(owned)
        if num_owned < chunk_size:
            updated[num_owned:].zero_()
        stacked_params = info["stacked_grads"]
        future = _collective(dist.all_gather_into_tensor, stacked_params, updated)
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    @torch.no_grad()
    def step(self):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        reduce_infos = []
        for group in self.param_groups:
            if group['kind'] == 'adamw': reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon': reduce_infos.append(self._reduce_muon(group, world_size))
        gather_list = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw': self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon': self._compute_muon(group, info, gather_list, rank)
        for info in gather_list:
            info["future"].wait()
            if info.get("params") is not None:
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))
# =============================================================================
# Dataloader: BOS-aligned best-fit packing
# =============================================================================

class DataLoader:
    """Loads flat tokens + chunks into batches.

    doc_shuffle=False: applies the stored default sequence permutation (bitwise match
    with the old chunked pipeline), shuffles batch order each epoch.
    doc_shuffle=True: reshuffles documents each epoch, re-chunks, re-shuffles sequences.
    """

    def __init__(self, filepath, B, T, device="cuda", *, doc_shuffle=False):
        data = torch.load(filepath, weights_only=True)
        all_tokens = data["tokens"].long()
        raw_doc_starts = data["doc_starts"].long()
        bos_id = int(data["bos_id"])
        assert bos_id == BOS_ID, f"data bos_id {bos_id} != expected {BOS_ID}"

        doc_ends = torch.cat([raw_doc_starts[1:], torch.tensor([all_tokens.numel()])])
        self.doc_tokens = [all_tokens[s:e] for s, e in zip(raw_doc_starts.tolist(), doc_ends.tolist())]
        self.default_shuffle_seed = data["seq_shuffle_seed"]

        _, rank, _, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.B = B
        self.T = T
        self.seq_size = T + 1
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
        else:   # Use dataset-stored permutation seed for backwards compatibility.
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

    def __iter__(self):
        return self

    def _next_epoch(self):
        self.epoch += 1
        print0(f"Starting epoch {self.epoch}")
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
        batch = self.rank_data[self.pos].to(self.device, non_blocking=True)
        self.pos += 1
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), self.epoch

# =============================================================================
# Loss evaluation
# =============================================================================

@torch.no_grad()
def evaluate_bpb(
    model,
    batches,
    steps,
    token_bytes,
    num_iterations=None,
    extra_kwargs=None,
):
    """Compute bits per byte and mean cross-entropy loss on a set of batches.

    `extra_kwargs(x)` returns additional model arguments (the n-gram support).
    """
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    model_kwargs = {"num_iterations": num_iterations}
    for _ in range(steps):
        x, y, _ = next(batch_iter)
        extra = extra_kwargs(x) if extra_kwargs is not None else {}
        loss2d = model(x, y, loss_reduction='none', **model_kwargs, **extra).view(-1)
        y = y.view(-1)
        mask = y != -1
        total_loss += loss2d[mask].sum()
        total_tokens += mask.sum()
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens = total_loss.item(), total_tokens.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return bpb, loss


def precompile_iteration_stages(model, x, y, mtp_weights, train_iteration_counts, eval_iteration_counts=None,
                                extra_kwargs=None, full_credit_counts=()):
    train_iteration_counts = tuple(dict.fromkeys(train_iteration_counts))
    if eval_iteration_counts is None:
        eval_iteration_counts = train_iteration_counts
    else:
        eval_iteration_counts = tuple(dict.fromkeys(eval_iteration_counts))
    if not train_iteration_counts and not eval_iteration_counts:
        return
    print0(
        "Precompiling recurrent iteration stages: "
        f"train={train_iteration_counts}, eval={eval_iteration_counts}"
    )
    was_training = model.training
    # Train precompile passes mirror the real training call: fused fp8 MTP path, so the
    # warmed-up graph for each iteration count matches step 1 and avoids a recompile.
    extra = extra_kwargs(x) if extra_kwargs is not None else {}
    for count in train_iteration_counts:
        # Randomized-credit arms also need the fully differentiated graph (Priority D).
        modes = ({}, {"full_credit": True}) if count in full_credit_counts else ({},)
        for mode in modes:
            print0(f"  precompile train {count}x {mode}", flush=True)
            model.train()
            model.zero_grad(set_to_none=True)
            with autocast_ctx:
                loss = model(x, y, num_iterations=count, mtp_weights=mtp_weights, **mode, **extra).mean()
            loss.backward()
            model.zero_grad(set_to_none=True)

    # Eval precompile passes stay MTP-free (plain bf16 next-token), matching real eval.
    for count in eval_iteration_counts:
        print0(f"  precompile eval {count}x", flush=True)
        model.eval()
        with torch.no_grad():
            with autocast_ctx:
                model(x, y, loss_reduction='none', num_iterations=count, **extra)
        synchronize()
    print0("  precompile done", flush=True)
    model.train(was_training)
    model.zero_grad(set_to_none=True)


@torch.no_grad()
def evaluate_strata(model, batches, steps, counts, num_iterations=None, extra_kwargs=None):
    """Returns held-out loss and token share per training-support stratum of the causal suffix."""
    sums = torch.zeros(len(ngram_lib.STRATA), dtype=torch.float64, device=model.get_device())
    tokens = torch.zeros(len(ngram_lib.STRATA), dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, _ = next(batch_iter)
        extra = extra_kwargs(x) if extra_kwargs is not None else {}
        losses = model(x, y, loss_reduction='none', num_iterations=num_iterations, **extra).view(x.shape)
        ngram_lib.accumulate_strata(losses, counts.strata(x), sums, tokens)
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
    total = max(int(tokens.sum()), 1)
    result = {}
    for i, name in enumerate(ngram_lib.STRATA):
        count = int(tokens[i])
        result[f"strata/{name}/loss"] = float(sums[i] / count) if count else float("nan")
        result[f"strata/{name}/share"] = count / total
    return result


def run_environment():
    """Returns software and hardware identifiers for result.json."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_DIR, capture_output=True,
                                text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "git_commit": commit,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "fa3": _fa3 is not None,
        "hostname": platform.node(),
    }


# =============================================================================
# Training
# =============================================================================

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
master_process = ddp_rank == 0
torch.manual_seed(args.seed)

if ddp and torch.cuda.is_available():
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(args.seed)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
elif ddp:
    # CPU smoke tests: the sharded optimizer still needs a process group.
    device = torch.device("cpu")
    dist.init_process_group(backend="gloo")
    dist.barrier()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_type = device.type
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# GPU info for MFU
gpu_peak_flops = float('inf')
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "h100" in gpu_name: gpu_peak_flops = 989e12
    elif "a100" in gpu_name: gpu_peak_flops = 312e12
    elif "4090" in gpu_name: gpu_peak_flops = 165.2e12

# FA3 status
if _fa3 is not None:
    print0("Using Flash Attention 3 (Hopper GPU detected)")
else:
    print0("Using PyTorch SDPA fallback (no FA3)")

# Run / logging paths
run_name, run_dir = resolve_run_dir(args.run_name)
if dist.is_initialized():
    shared = [run_name]
    dist.broadcast_object_list(shared, src=0)
    run_name = shared[0]
    run_dir = os.path.join(RUNS_DIR, run_name)
checkpoints_dir = os.path.join(run_dir, "checkpoints")
artifact_model_path = os.path.join(run_dir, "model.pt")
terminal_log_path = os.path.join(run_dir, "terminal.log")
stdout_orig = sys.stdout
stderr_orig = sys.stderr
artifacts_log_f = None
result_path = os.path.join(run_dir, "result.json")
os.makedirs(run_dir, exist_ok=True)
if master_process:
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "wandb"), exist_ok=True)
    shutil.copy2(__file__, os.path.join(run_dir, "train.py"))
    os.makedirs(os.path.join(run_dir, "lab"), exist_ok=True)
    for _module in sorted(os.listdir(_LAB_DIR)):
        if _module.endswith(".py"):
            shutil.copy2(os.path.join(_LAB_DIR, _module), os.path.join(run_dir, "lab", _module))
if dist.is_initialized():
    dist.barrier()
artifacts_log_f = open(terminal_log_path, "a", encoding="utf-8", buffering=1)
sys.stdout = TeeStream(sys.stdout, artifacts_log_f)
sys.stderr = TeeStream(sys.stderr, artifacts_log_f)

# wandb
_wandb_kwargs = {"project": "slowrun", "name": run_name}
if args.wandb_group:
    _wandb_kwargs["group"] = args.wandb_group
_wandb_kwargs["dir"] = os.path.join(run_dir, "wandb")
wandb_run = DummyWandb() if not master_process else wandb.init(**_wandb_kwargs)
if master_process:
    # Log the repository's code, not virtualenvs, data or the copies inside runs/.
    wandb_run.log_code(".", exclude_fn=lambda path, root: any(
        f"{os.sep}{part}{os.sep}" in f"{os.sep}{os.path.relpath(path, root)}"
        for part in (".venv", "venv", "runs", "fineweb_data", "wandb", ".git", "node_modules")))

# Print hyperparameters
print0(f"--- Hyperparameters ---")
print0(f"  n_layer={DEPTH}, n_embd={N_EMBD}, n_head={N_HEAD}, head_dim={HEAD_DIM}")
print0(f"  seq_len={MAX_SEQ_LEN}, window_pattern={WINDOW_PATTERN}")
print0(f"  total_batch_size={TOTAL_BATCH_SIZE}, device_batch_size={args.device_batch_size}")
print0(f"  matrix_lr={MATRIX_LR}, scalar_lr={SCALAR_LR}, embedding_lr={EMBEDDING_LR}, unembedding_lr={UNEMBEDDING_LR}")
print0(f"  weight_decay={WEIGHT_DECAY}, adam_betas={ADAM_BETAS}")
print0(f"  wd_schedule=hold {args.weight_decay} -> mid {args.wd_mid} -> end {args.wd_end}")
print0(
    f"  warmup_ratio={WARMUP_RATIO}, warmdown_ratio={WARMDOWN_RATIO}, "
    f"final_lr_frac={FINAL_LR_FRAC}"
)
print0(f"  num_epochs={args.num_epochs}, patience={args.patience}")
print0(f"  max_num_iterations={NUM_ITERATIONS}")
print0(f"  train_backprop_iterations={TRAIN_BACKPROP_ITERATIONS}")
print0(
    f"  iteration_schedule={args.iteration_schedule} "
    f"({format_iteration_schedule(ITERATION_SCHEDULE)}), "
    f"transition_ratio={ITERATION_TRANSITION_RATIO:.3f}, "
    f"avg_layer_passes={DEPTH * ITERATION_SCHEDULE.avg_iterations:.3f}"
)
print0(f"  dropout={args.dropout}")
print0(f"  doc_shuffle={not args.no_doc_shuffle}")
print0(f"  max_train_steps={args.max_train_steps}, xsa_mode={args.xsa_mode}")
print0(f"  mtp_predict={args.mtp_predict}, mtp_anneal_frac={args.mtp_anneal_frac}")
print0(f"  run={run_name}")
print0(f"  run_dir={run_dir}")
print0(f"-----------------------")

# Load GPT-2 tokenizer and compute token_bytes for BPB evaluation
encoder = tiktoken.get_encoding("gpt2")
vocab_size = encoder.n_vocab  # 50257
print0(f"Vocab size: {vocab_size:,}")

eot_id = encoder._special_tokens['<|endoftext|>']
token_bytes_list = []
for i in range(vocab_size):
    if i == eot_id:
        token_bytes_list.append(0)
    else:
        token_bytes_list.append(len(encoder.decode_single_token_bytes(i)))
token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device=device)

# Build model
config = GPTConfig(vocab_size=vocab_size, dropout=args.dropout, device_batch_size=args.device_batch_size,
                   xsa_mode=args.xsa_mode, num_iterations=NUM_ITERATIONS,
                   act_stats=ACT_STATS, act_stride=args.mg_act_stride, act_decay=args.mg_act_decay,
                   credit_scale=CREDIT_SCALE, memory=MEMORY_CONFIG)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

# The fused fp8 MTP kernel has a compile-time-fixed vocab; the padded model vocab must match.
assert model.lm_head.weight.size(1) == CE_KERNEL_VOCAB_SIZE, (
    f"padded vocab {model.lm_head.weight.size(1)} != fused kernel VOCAB_SIZE {CE_KERNEL_VOCAB_SIZE}; "
    "the MTP CUDA kernel requires these to match")

param_counts = sum(p.numel() for p in model.parameters())
transformer_params = sum(p.numel() for p in model.transformer.h.parameters())
ve_params = sum(p.numel() for p in model.ve_projs.parameters())
lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
other_params = param_counts - transformer_params - ve_params - lm_head_params
max_flops_per_token = model.estimate_flops(NUM_ITERATIONS)
scheduled_iteration_counts = iteration_schedule_counts(ITERATION_SCHEDULE)
schedule_flops_per_token = {
    iteration_count: model.estimate_flops(iteration_count)
    for iteration_count in scheduled_iteration_counts
}
avg_flops_per_token = sum(
    (stage.end_frac - stage.start_frac)
    * schedule_flops_per_token[stage.iterations]
    for stage in ITERATION_SCHEDULE.stages
)
print0(f"Parameters: {param_counts:,} (transformer: {transformer_params:,}, value_embeds: {ve_params:,}, lm_head: {lm_head_params:,}, other: {other_params:,})")
if model.ngram is not None:
    print0(f"N-gram memory: {model.ngram.parameter_count():,} parameters, orders {MEMORY_CONFIG.active_orders}, "
           f"{model.ngram.hasher.slots} slots of ~{MEMORY_CONFIG.table_size:,} rows, layer {MEMORY_CONFIG.layer}")
if model.dense_adapter is not None:
    print0(f"Dense control adapter: {model.dense_adapter.parameter_count():,} parameters, "
           f"hidden {model.dense_adapter.w_in.out_features}, layer {MEMORY_CONFIG.layer}")
print0(f"FLOPs per token at max iterations: {max_flops_per_token:e}")
print0(f"Average scheduled FLOPs per token: {avg_flops_per_token:e}")

# Compile
orig_model = model
model = torch.compile(model, dynamic=False)

# Optimizer
optimizer = model.setup_optimizer()

# ---- every-step meta-gradient step (research fork: research/meta_gradient/lookahead.py) ----
MG = args.mg_every > 0
muon_groups = [g["params"] for g in optimizer.param_groups if g.get("kind") == "muon"]
mg_names, mg_params, mg_owned = lookahead_lib.select_plastic(
    list(orig_model.named_parameters()), muon_groups, rule=args.mg_plastic,
    rank=ddp_rank, world_size=ddp_world_size)
act_moments = {
    f"transformer.h.{i}.mlp.c_proj.weight": (block.mlp.act_moment, block.mlp.act_count)
    for i, block in enumerate(orig_model.transformer.h) if block.mlp.act_stats
}
act_buffers = [t for pair in act_moments.values() for t in pair]
lookahead = None
if MG:
    act_radius = args.mg_act_radius
    if args.mg_cproj in ("act_radius", "act_metric") and act_radius <= 0:
        if not args.mg_act_calibration:
            raise ValueError(f"--mg-cproj {args.mg_cproj} needs --mg-act-radius or --mg-act-calibration")
        act_radius = lookahead_lib.functional_radius(
            lookahead_lib.load_calibration(args.mg_act_calibration), args.mg_act_radius_mult)
    layer_radii = ()
    if args.mg_cproj == "euclid_layer":
        if not args.mg_cproj_radii:
            raise ValueError("--mg-cproj euclid_layer needs --mg-cproj-radii")
        layer_radii = lookahead_lib.matched_radii(
            lookahead_lib.load_calibration(args.mg_cproj_radii),
            [n for n in mg_names if n.endswith("mlp.c_proj.weight")])
    lookahead = lookahead_lib.Lookahead(
        mg_names, mg_params,
        config=lookahead_lib.LookaheadConfig(
            step_norm=args.mg_step_norm, schedule=args.mg_step_schedule, cproj=args.mg_cproj,
            act_radius=act_radius, act_damping=args.mg_act_damping, act_decay=args.mg_act_decay,
            layer_radii=layer_radii),
        moments=act_moments)
    print0(f"[mg] every {args.mg_every} steps, step norm {args.mg_step_norm} ({args.mg_step_schedule}), "
           f"{len(mg_params)} plastic matrices, {sum(p.numel() for p in mg_params):,} parameters; "
           f"rank {ddp_rank} owns {sum(mg_owned)} of them; c_proj {args.mg_cproj}"
           + (f" (rho={act_radius:.4g})" if act_radius > 0 else "")
           + f"; aux split {args.aux_split}")
elif args.aux_split != "shared" or args.mg_cproj != "global":
    raise ValueError("--aux-split and --mg-cproj need the MG split (--mg-every >= 1)")

# Dataloaders. The training loader emulates --virtual-ranks data-parallel ranks, so the MG
# split sees exactly the record's per-rank batches on 1, 2, 4 or 8 GPUs.
_train_path = args.input_bin if args.input_bin else os.path.join(DATA_DIR, "fineweb_train.pt")
_val_path = args.input_val_bin if args.input_val_bin else os.path.join(DATA_DIR, "fineweb_val.pt")
tokens_per_fwdbwd = args.device_batch_size * MAX_SEQ_LEN * ddp_world_size
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
VIRTUAL_RANKS = args.virtual_ranks
if VIRTUAL_RANKS % ddp_world_size or grad_accum_steps % max(VIRTUAL_RANKS // ddp_world_size, 1):
    raise ValueError(f"--virtual-ranks {VIRTUAL_RANKS} must be a multiple of the {ddp_world_size} processes "
                     f"and divide the {grad_accum_steps} accumulated microbatches per process")
LOCAL_VRANKS = VIRTUAL_RANKS // ddp_world_size
MICRO_PER_VRANK = grad_accum_steps // LOCAL_VRANKS
if MG and MICRO_PER_VRANK > 1 and MICRO_PER_VRANK % 2:
    raise ValueError(f"the MG split needs an even number of microbatches per virtual rank, got {MICRO_PER_VRANK}")
train_loader = VirtualRankLoader(
    _train_path, args.device_batch_size, MAX_SEQ_LEN, device=device, doc_shuffle=not args.no_doc_shuffle,
    rank=ddp_rank, world_size=ddp_world_size, virtual_ranks=VIRTUAL_RANKS,
    micro_per_vrank=MICRO_PER_VRANK, data_seed=args.data_seed)
build_val_loader = lambda: DataLoader(_val_path, args.device_batch_size, MAX_SEQ_LEN, device=device)
TOKENS_PER_EPOCH = train_loader.total_tokens
step_batches = train_loader.next_step()
x, y, current_epoch = step_batches[0]

# Research state ---------------------------------------------------------------
# Exact training counts of short suffixes feed the n-gram memory's shrinkage. Building them is
# part of the method, so its time is charged to training. Arms without shrinkage only need
# suffix validity and build no counts.
memory_counts, ngram_build_time, ngram_stats = None, 0.0, {}
if orig_model.ngram is not None and args.ngram_kappa > 0:
    synchronize()
    _t = time.time()
    memory_counts = ngram_lib.SuffixCounts.build(
        torch.cat(train_loader.doc_tokens), orders=MEMORY_CONFIG.active_orders, device=device)
    synchronize()
    ngram_build_time = time.time() - _t
    ngram_stats = orig_model.ngram.hasher.collision_statistics(memory_counts)
    print0(f"[ngram] counted suffixes in {ngram_build_time:.2f}s "
           f"({memory_counts.nbytes() / 2**20:.0f} MiB of tables); kappa={args.ngram_kappa}")


def memory_kwargs(xs):
    """Returns the n-gram support argument of a model call, if the memory is on."""
    if orig_model.ngram is None:
        return {}
    if memory_counts is None:
        return {"ngram_support": ngram_lib.validity_support(xs, MEMORY_CONFIG.active_orders)}
    return {"ngram_support": memory_counts.support(xs, kappa=args.ngram_kappa)}


# The fixed training probe: per virtual rank, the first half measures training loss and the
# second half forms an independent (A, B) pair for the survival diagnostic.
PROBE_PAIR = MICRO_PER_VRANK * args.device_batch_size
probe_x, probe_y = probe_sequences(
    _train_path, seq_len=MAX_SEQ_LEN, per_vrank=2 * PROBE_PAIR, virtual_ranks=VIRTUAL_RANKS,
    first_vrank=ddp_rank * LOCAL_VRANKS, local_vranks=LOCAL_VRANKS, device=device)
probe_x = probe_x.view(LOCAL_VRANKS, 2 * PROBE_PAIR, -1)
probe_y = probe_y.view(LOCAL_VRANKS, 2 * PROBE_PAIR, -1)
probe_loss_batches = [
    (probe_x[v, i:i + args.device_batch_size], probe_y[v, i:i + args.device_batch_size])
    for v in range(LOCAL_VRANKS) for i in range(0, PROBE_PAIR, args.device_batch_size)
]
probe_pair_batches = [
    [(probe_x[v, PROBE_PAIR + i:PROBE_PAIR + i + args.device_batch_size],
      probe_y[v, PROBE_PAIR + i:PROBE_PAIR + i + args.device_batch_size], 0)
     for i in range(0, PROBE_PAIR, args.device_batch_size)]
    for v in range(LOCAL_VRANKS)
]


def train_probe_loss(num_iterations):
    """Returns the clean next-token loss on the fixed training probe (all ranks)."""
    was_training = model.training
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad(), autocast_ctx:
        for xs, ys in probe_loss_batches:
            losses = model(xs, ys, loss_reduction='none', num_iterations=num_iterations, **memory_kwargs(xs))
            totals[0] += losses.double().sum()
            totals[1] += losses.numel()
    if dist.is_initialized():
        dist.all_reduce(totals)
    model.train(was_training)
    return float(totals[0] / totals[1])


credit_sampler = credit_lib.CreditSampler(mode=args.credit, p=args.credit_p, seed=args.seed)
aux_mult_a, aux_mult_b = lookahead_lib.aux_multipliers(args.aux_split)


def half_mtp_weights(weights, multiplier):
    """Scales the auxiliary (extra-offset) MTP weights; offset 0 stays at 1."""
    if multiplier == 1.0:
        return weights
    scaled = weights.clone()
    scaled[1:] *= multiplier
    return scaled


def model_call_kwargs(xs, full_credit):
    """Extra arguments of a training forward call; empty for the record."""
    extra = memory_kwargs(xs)
    if full_credit:
        extra["full_credit"] = True
    return extra


diag_steps = sorted({int(s) for s in args.diag_steps.split(",") if s.strip()})
if diag_steps and lookahead is None:
    raise ValueError("--diag-steps needs the MG split (--mg-every >= 1)")
diag_params = [n for n in args.diag_params.split(",") if n] or [
    f"transformer.h.{layer}.{role}.weight"
    for layer in sorted({0, DEPTH // 2, DEPTH - 1}) for role in ("attn.c_q", "mlp.c_fc", "mlp.c_proj")
]
named_parameters = dict(orig_model.named_parameters())
muon_slots = {}
for _group in optimizer.param_groups:
    if _group.get("kind") == "muon":
        _chunk = (len(_group["params"]) + ddp_world_size - 1) // ddp_world_size
        for _i, _p in enumerate(_group["params"]):
            muon_slots[id(_p)] = (_group, _i // _chunk, _i % _chunk)


def diag_muon_input(name, grad):
    """Returns the Muon input (1 - mu^2) G + mu^2 buf of an owned matrix, else None."""
    group, owner, slot = muon_slots[id(named_parameters[name])]
    if owner != ddp_rank:
        return None
    buf = optimizer.state.get(group["params"][0], {}).get("momentum_buffer")
    mu = group["momentum"]
    base = torch.zeros_like(grad, dtype=torch.float32) if buf is None else buf[slot].float()
    return (1 - mu**2) * grad.float() + mu**2 * base


def diag_preprocess(m):
    """The map applied before orthogonalization (MuonEq-R when enabled)."""
    if not args.muon_eq_r:
        return m
    return m / m.norm(dim=-1, keepdim=True).clamp_min(1e-7)


def survival_inputs(weights_a, weights_b, num_iterations):
    """Binds the diagnostic to this step's objective and recurrence."""

    def run_half(batches, half):
        weights = weights_a if half == "A" else weights_b
        for xs, ys, *_ in batches:
            with autocast_ctx:
                loss = model(xs, ys, num_iterations=num_iterations, mtp_weights=weights,
                             **memory_kwargs(xs)).mean()
            (loss / (2 * len(batches))).backward()

    return probe_lib.SurvivalInputs(
        run_half=run_half,
        probe_loss=lambda: train_probe_loss(num_iterations),
        muon_input=diag_muon_input,
        preprocess=diag_preprocess,
        zero_grad=lambda: model.zero_grad(set_to_none=True),
    )


# Training config
num_train_steps = round(TOKENS_PER_EPOCH * args.num_epochs / TOTAL_BATCH_SIZE)  # estimate for LR schedule
target_iterations = args.max_train_steps if args.max_train_steps > 0 else num_train_steps
steps_per_epoch = num_train_steps / args.num_epochs
# Convert epoch boundaries to steps (must happen after num_train_steps is known)
wd_phase1_end_step = round(args.wd_phase1_epoch / args.num_epochs * num_train_steps)
wd_phase2_end_step = round(args.wd_phase2_epoch / args.num_epochs * num_train_steps)
print0(f"Batch size: {TOTAL_BATCH_SIZE:,} tokens, grad accum: {grad_accum_steps} steps")
print0(f"Virtual ranks: {VIRTUAL_RANKS} ({LOCAL_VRANKS} per process, {MICRO_PER_VRANK} microbatch(es) each)")
print0(f"Training for {args.num_epochs} epoch(s) (~{num_train_steps} steps estimated)")
if args.max_train_steps > 0:
    print0(f"Stopping after max_train_steps={args.max_train_steps:,}")
print0(f"Recurrent iteration schedule: {format_iteration_schedule(ITERATION_SCHEDULE)}")
for stage in ITERATION_SCHEDULE.stages:
    print0(
        f"  {stage.start_frac:.3f}-{stage.end_frac:.3f} "
        f"(~steps {round(stage.start_frac * num_train_steps)}-"
        f"{round(stage.end_frac * num_train_steps)}): {stage.iterations} iteration(s)"
    )
print0(f"Eval set: {EVAL_TOKENS:,} tokens")


def get_lr_multiplier(it):
    warmup = round(WARMUP_RATIO * num_train_steps)
    warmdown = round(WARMDOWN_RATIO * num_train_steps)
    if warmup > 0 and it < warmup: return (it + 1) / warmup
    elif warmdown <= 0 or it <= num_train_steps - warmdown: return 1.0
    else:
        progress = (num_train_steps - it) / warmdown
        return progress + (1 - progress) * FINAL_LR_FRAC

def get_muon_momentum(it):
    return (1 - min(it / 300, 1)) * 0.85 + min(it / 300, 1) * 0.95

def get_mtp_weights(it):
    """Per-offset MTP loss weights at optimizer step `it`.

    Offset 0 (the real next token) is always weight 1.0. Extra offsets start at
    MTP_START_WEIGHTS and decay linearly to zero over staggered sub-windows of
    [0, mtp_anneal_frac]; the furthest-ahead offset dies first, so training ends as
    plain next-token prediction. Length stays fixed (= mtp_predict) so the fused
    kernel's grid and the compiled graph never need to change shape."""
    n = max(1, min(args.mtp_predict, len(MTP_START_WEIGHTS)))
    w = [1.0]
    if n > 1:
        f = min(1.0, it / max(1, num_train_steps))
        A = args.mtp_anneal_frac
        for k in range(1, n):
            seg_start = (n - 1 - k) / (n - 1) * A
            seg_end = (n - k) / (n - 1) * A
            if f <= seg_start:
                frac = 1.0
            elif f >= seg_end:
                frac = 0.0
            else:
                frac = 1.0 - (f - seg_start) / (seg_end - seg_start)
            w.append(MTP_START_WEIGHTS[k] * frac)
    return torch.tensor(w, device=device, dtype=torch.float32)

# Training loop
step = 0
min_val_bpb = float("inf")
min_val_loss = float("inf")
epochs_without_improvement = 0
smooth_train_loss = 0
total_training_time = 0
eval_steps = EVAL_TOKENS // (args.device_batch_size * MAX_SEQ_LEN * ddp_world_size)
param_ema_beta = args.ema_decay_per_epoch ** (args.update_ema_every / steps_per_epoch) if args.update_ema_every > 0 else 0
ema_params = [torch.zeros_like(p) for p in model.parameters()] if args.update_ema_every > 0 else None

wall_clock_start = time.time()
_swa_start_step = (num_train_steps - args.swa_last_epochs * steps_per_epoch) if args.swa_last_epochs > 0 else -1
late_ckpt_paths = []
def get_eval_iterations(active_iterations):
    return active_iterations

# Research bookkeeping (written to result.json).
epoch_records = []
diag_records, diag_time = [], 0.0
# Actual update size of the representative matrices, sampled every --log-every steps, and
# mean step time per training phase.
update_probe = [(n, named_parameters[n]) for n in diag_params if n in named_parameters]
update_sums, update_count, phase_times = {}, 0, {}
diag_dir = os.path.join(run_dir, "diag")
if diag_steps and master_process:
    os.makedirs(diag_dir, exist_ok=True)
FULL_CREDIT_COUNTS = tuple(c for c in scheduled_iteration_counts if c > 1) if args.credit != "trunc" else ()


active_num_iterations = get_scheduled_iterations(
    ITERATION_SCHEDULE, step, num_train_steps
)
active_expected_num_iterations = get_expected_scheduled_iterations(
    ITERATION_SCHEDULE, step, num_train_steps
)
precompile_eval_iteration_counts = tuple(
    dict.fromkeys(get_eval_iterations(count) for count in scheduled_iteration_counts)
)
precompile_iteration_stages(
    model,
    x,
    y,
    get_mtp_weights(step),
    scheduled_iteration_counts,
    precompile_eval_iteration_counts,
    extra_kwargs=memory_kwargs,
    # With MG on every step each training pass is a half batch, so the fully differentiated
    # graph is only warmed at that shape below; warming it on the full batch needs twice the
    # activation memory of any real step and runs out of memory on one H100.
    full_credit_counts=() if MG and args.mg_every == 1 else FULL_CREDIT_COUNTS,
)
if MG:
    # the step runs two half-batch passes per step; warm that shape for every recurrence stage
    # too, otherwise the 1x->2x transition recompiles inside the timed region
    _h = x.size(0) // 2
    precompile_iteration_stages(model, x[:_h], y[:_h], get_mtp_weights(step), scheduled_iteration_counts, (),
                                extra_kwargs=memory_kwargs, full_credit_counts=FULL_CREDIT_COUNTS)
# Precompile passes must not leak into the activation statistics.
for _buffer in act_buffers:
    _buffer.zero_()
# Emulated ranks keep separate activation moments and start each step from the same RNG
# state, like the identically seeded physical ranks of the record (no-op on 8 GPUs).
vrank_state = lookahead_lib.VirtualRankState(act_buffers, LOCAL_VRANKS)
mg_step_count = 0

# Initial val evaluation
model.eval()
val_loader = build_val_loader()
eval_num_iterations = get_eval_iterations(active_num_iterations)
with autocast_ctx:
    val_bpb, val_loss = evaluate_bpb(
        model, val_loader, eval_steps, token_bytes, num_iterations=eval_num_iterations,
        extra_kwargs=memory_kwargs,
    )
print0(
    f"Step {step:05d} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f} "
    f"| eval iters: {eval_num_iterations} | train iters: {active_num_iterations}"
)
wandb_run.log({
    "step": step,
    "val/bpb": val_bpb,
    "val/loss": val_loss,
    "val/num_iterations": eval_num_iterations,
    "val/train_num_iterations": active_num_iterations,
    "val/train_expected_num_iterations": active_expected_num_iterations,
    "val/train_expected_effective_layers": DEPTH * active_expected_num_iterations,
    "val/train_actual_effective_layers": DEPTH * active_num_iterations,
})
if eval_num_iterations == NUM_ITERATIONS:
    min_val_bpb = val_bpb
    min_val_loss = val_loss
model.train()

while current_epoch <= args.num_epochs:
    next_num_iterations = get_scheduled_iterations(
        ITERATION_SCHEDULE, step, num_train_steps
    )
    next_expected_num_iterations = get_expected_scheduled_iterations(
        ITERATION_SCHEDULE, step, num_train_steps
    )
    if next_num_iterations != active_num_iterations:
        active_num_iterations = next_num_iterations
        active_expected_num_iterations = next_expected_num_iterations
        print0(
            f"\n=== Recurrent iterations -> {active_num_iterations} "
            f"at step {step} ({100 * step / num_train_steps:.2f}%, "
            f"epoch {current_epoch}) ==="
        )

    # Schedules. They are applied before the forward pass, which never reads them, so that the
    # survival diagnostic can run the optimizer exactly as this step will.
    lrm = get_lr_multiplier(step)
    # SWA: cosine-cycle LR in final epochs for diverse checkpoints to average
    if _swa_start_step >= 0 and step >= _swa_start_step:
        cycle_pos = (step - _swa_start_step) % steps_per_epoch
        swa_base = max(lrm, 0.05)
        lrm = 0.05 + (swa_base - 0.05) * (1 + math.cos(math.pi * cycle_pos / steps_per_epoch)) / 2
    # WD schedule:
    #   [0, wd_phase1_end_step]:              hold at weight_decay
    #   [wd_phase1_end_step, wd_phase2_end_step]: decay to wd_mid
    #   [wd_phase2_end_step, num_train_steps]:    ramp up to wd_end
    wd = np.interp(step,
        [0, wd_phase1_end_step, wd_phase2_end_step, num_train_steps],
        [args.weight_decay, args.weight_decay, args.wd_mid, args.wd_end])
    # Convert to a scale factor;
    # groups with weight_decay=0.0 (scalar params) correctly stay at zero.
    wd_scale = wd / args.weight_decay if args.weight_decay > 0 else 0.0
    spectral_setting = None
    if args.spectral != "polar_express":
        exponent = args.spectral_c
        if args.spectral_c_late is not None and step >= args.spectral_switch_frac * num_train_steps:
            exponent = args.spectral_c_late
        spectral_setting = (args.spectral, exponent, args.spectral_eps,
                            torch.float64 if args.spectral_fp64 else torch.float32, args.spectral_norm)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if "initial_wd" not in group:
            group["initial_wd"] = group.get("weight_decay", 0.0)
        group["weight_decay"] = group["initial_wd"] * wd_scale
        if group['kind'] == 'muon':
            group["momentum"] = get_muon_momentum(step)
            group["spectral"] = spectral_setting

    mtp_w = get_mtp_weights(step)  # same weights across the grad-accum micro-steps
    mtp_w_a, mtp_w_b = half_mtp_weights(mtp_w, aux_mult_a), half_mtp_weights(mtp_w, aux_mult_b)
    mg_step = MG and step % args.mg_every == 0
    vrank_state.begin_step()

    # Survival diagnostic (Priority C). It restores everything it touches, so the run continues
    # exactly as without it; its time is reported separately and excluded from training time.
    if mg_step and step in diag_steps:
        synchronize()
        _t = time.time()
        diag_metrics, diag_spectra = probe_lib.mg_survival(
            named_params=list(orig_model.named_parameters()),
            optimizer=optimizer,
            lookahead=lookahead,
            lr_multiplier=get_lr_multiplier(step),
            inputs=survival_inputs(mtp_w_a, mtp_w_b, active_num_iterations),
            step_pairs=[lookahead_lib.split_halves(step_batches[v * MICRO_PER_VRANK:(v + 1) * MICRO_PER_VRANK])
                        for v in range(LOCAL_VRANKS)],
            independent_pairs=[lookahead_lib.split_halves(pair) for pair in probe_pair_batches],
            representative=diag_params,
            buffers=act_buffers + vrank_state.tensors(),
            moment_buffers=act_buffers,
            enter_vrank=vrank_state.enter,
            seed=args.seed * 7919 + step,
        )
        synchronize()
        diag_time += time.time() - _t
        diag_records.append({"step": step, **diag_metrics})
        wandb_run.log({"step": step, **diag_metrics})
        if diag_spectra:
            torch.save(diag_spectra, os.path.join(diag_dir, f"spectra_step{step:05d}_rank{ddp_rank}.pt"))
        print0(f"[diag] step {step}: probe gain of the baseline step {diag_metrics['diag/probe_gain_base']:.3e}, "
               f"extra gain from MG {diag_metrics['diag/probe_gain_mg']:.3e} ({time.time() - _t:.1f}s)")

    # Training step
    synchronize()
    t0 = time.time()
    full_credit = credit_sampler.draw(active_num_iterations)
    step_losses = []
    if mg_step:
        # Split pairing per virtual rank: the first half adapts, the second half is evaluated at the
        # adapted point; the optimizer receives the ordinary full-batch mean gradient.
        mg_step_count += 1
        for v in range(LOCAL_VRANKS):
            adapt, query = lookahead_lib.split_halves(
                step_batches[v * MICRO_PER_VRANK:(v + 1) * MICRO_PER_VRANK])
            scale = 1.0 / (LOCAL_VRANKS * (len(adapt) + len(query)))
            vrank_state.enter(v)
            lookahead.stash()
            for xs, ys, _ in adapt:
                with autocast_ctx:
                    loss = model(xs, ys, num_iterations=active_num_iterations, mtp_weights=mtp_w_a,
                                 **model_call_kwargs(xs, full_credit)).mean()
                (loss * scale).backward()
                step_losses.append(loss.detach())
            lookahead.displace(get_lr_multiplier(step))
            for xs, ys, _ in query:
                with autocast_ctx:
                    loss = model(xs, ys, num_iterations=active_num_iterations, mtp_weights=mtp_w_b,
                                 **model_call_kwargs(xs, full_credit)).mean()
                (loss * scale).backward()
                step_losses.append(loss.detach())
            lookahead.restore()
            vrank_state.exit(v)
    else:
        for v in range(LOCAL_VRANKS):
            vrank_state.enter(v)
            for xs, ys, _ in step_batches[v * MICRO_PER_VRANK:(v + 1) * MICRO_PER_VRANK]:
                with autocast_ctx:
                    # MTP path returns per-token weighted loss (B*T,); mean() reduces it as
                    # before. During the MTP phase this loss is the weighted multi-offset sum,
                    # so the logged value runs higher than the naive run until the extra
                    # offsets anneal to zero.
                    loss = model(xs, ys, num_iterations=active_num_iterations, mtp_weights=mtp_w,
                                 **model_call_kwargs(xs, full_credit)).mean()
                step_losses.append(loss.detach())
                (loss / grad_accum_steps).backward()
            vrank_state.exit(v)
    train_loss = torch.stack(step_losses).mean()
    step_batches = train_loader.next_step()
    epoch = step_batches[0][2]

    # Update optimizer
    measure_update = step % args.log_every == 0
    if measure_update:
        update_before = [p.detach().clone() for _, p in update_probe]
    optimizer.step()
    model.zero_grad(set_to_none=True)
    if ema_params is not None and step % args.update_ema_every == 0:
        torch._foreach_lerp_(ema_params, list(model.parameters()), 1 - param_ema_beta)
    train_loss_f = train_loss.item()
    synchronize()
    dt = time.time() - t0
    if step > 2:
        phase = f"passes{active_num_iterations}" + ("_swa" if 0 <= _swa_start_step <= step else "")
        phase_times.setdefault(phase, [0.0, 0])
        phase_times[phase][0] += dt
        phase_times[phase][1] += 1
    update_log = {}
    if measure_update:
        for (name, p), before in zip(update_probe, update_before):
            rms = float((p.detach().float() - before.float()).square().mean().sqrt())
            update_log[f"opt/update_rms/{name}"] = rms
            update_sums[name] = update_sums.get(name, 0.0) + rms
        update_count += 1
        del update_before

    step += 1

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_train_loss / (1 - ema_beta**step)
    pct = 100 * step / target_iterations
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    active_flops_per_token = schedule_flops_per_token[active_num_iterations]
    mfu = 100 * active_flops_per_token * TOTAL_BATCH_SIZE / dt / (gpu_peak_flops * ddp_world_size)
    if step > 3:
        total_training_time += dt
    steps_done = step - 3
    eta_str = f" | eta: {(target_iterations - step) * total_training_time / steps_done / 60:.1f}m" if steps_done > 0 else ""
    train_backprop_iterations = min(TRAIN_BACKPROP_ITERATIONS, active_num_iterations)
    print0(f"step {step:05d} ({pct:.2f}%) | loss: {debiased:.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f}% | iters: {active_num_iterations}{eta_str}")
    wandb_run.log({
        "step": step,
        "train/loss": debiased,
        "train/lr_multiplier": lrm,
        "train/mfu": mfu,
        "train/num_iterations": active_num_iterations,
        "train/backprop_iterations": train_backprop_iterations,
    })
    if step % args.log_every == 0:
        research_log = lookahead.statistics() if lookahead is not None else {}
        if args.credit != "trunc":
            research_log["credit/full_frac"] = credit_sampler.full_steps / max(credit_sampler.multi_pass_steps, 1)
        if research_log:
            wandb_run.log({"step": step, **research_log})
    if update_log:
        wandb_run.log({"step": step, **update_log})

    # Synchronize epoch across ranks (different ranks may exhaust data at different steps)
    if ddp:
        epoch_tensor = torch.tensor([epoch], dtype=torch.long, device=device)
        dist.all_reduce(epoch_tensor, op=dist.ReduceOp.MAX)
        epoch = epoch_tensor.item()

    # Epoch boundary: evaluate when the dataloader advances to a new epoch
    if epoch != current_epoch:
        model.eval()
        val_loader = build_val_loader()
        eval_num_iterations = get_eval_iterations(active_num_iterations)
        with autocast_ctx:
            val_bpb, val_loss = evaluate_bpb(
                model, val_loader, eval_steps, token_bytes, num_iterations=eval_num_iterations,
                extra_kwargs=memory_kwargs,
            )
        probe_loss = train_probe_loss(eval_num_iterations)
        epoch_records.append({"epoch": current_epoch, "step": step, "val_loss": val_loss,
                              "val_bpb": val_bpb, "train_probe_loss": probe_loss,
                              "training_time_s": total_training_time})
        print0(
            f"Step {step:05d} | Epoch {current_epoch} | Val BPB: {val_bpb:.6f} "
            f"| Val Loss: {val_loss:.6f} | eval iters: {eval_num_iterations} "
            f"| train iters: {active_num_iterations} | train probe: {probe_loss:.6f}"
        )
        wandb_run.log({"step": step, "train_probe/loss": probe_loss,
                       "gap/val_minus_train_probe": val_loss - probe_loss})
        wandb_run.log({
            "step": step,
            "epoch": current_epoch,
            "val/bpb": val_bpb,
            "val/loss": val_loss,
            "val/num_iterations": eval_num_iterations,
            "val/train_num_iterations": active_num_iterations,
            "val/train_expected_num_iterations": active_expected_num_iterations,
            "val/train_expected_effective_layers": DEPTH * active_expected_num_iterations,
            "val/train_actual_effective_layers": DEPTH * active_num_iterations,
        })
        # Save checkpoint for weight averaging
        ckpt_path = os.path.join(checkpoints_dir, f"epoch_{current_epoch:03d}.pt")
        if master_process:
            torch.save({n: p.data.float().cpu() for n, p in orig_model.named_parameters()}, ckpt_path)
        late_ckpt_paths.append(ckpt_path)
        if len(late_ckpt_paths) > args.swa_last_epochs:
            old = late_ckpt_paths.pop(0)
            if master_process and os.path.exists(old): os.remove(old)
        # Early stopping / best-val tracking only compare max-iteration evals.
        if eval_num_iterations == NUM_ITERATIONS:
            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
                min_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if args.patience >= 0 and epochs_without_improvement >= args.patience:
                    print0(f"Early stopping: no improvement for {args.patience} epoch(s)")
                    break

        model.train()
        current_epoch = epoch

    if args.max_train_steps > 0 and step >= args.max_train_steps:
        print0(f"Reached max_train_steps={args.max_train_steps}")
        break

    # GC management
    if step == 1:
        gc.collect(); gc.freeze(); gc.disable()

# Final EMA evaluation
ema_loss = avg_loss = None
if ema_params is not None:
    ema_updates = step // args.update_ema_every
    if ema_updates > 0:
        correction = 1.0 / (1.0 - param_ema_beta ** ema_updates)
        model.eval()
        with torch.no_grad():
            for p, ema in zip(model.parameters(), ema_params):
                p.copy_(ema * correction)
        val_loader = build_val_loader()
        with autocast_ctx:
            ema_bpb, ema_loss = evaluate_bpb(
                model, val_loader, eval_steps, token_bytes, num_iterations=NUM_ITERATIONS,
                extra_kwargs=memory_kwargs,
            )
        print0(f"EMA Val BPB: {ema_bpb:.6f} | EMA Val Loss: {ema_loss:.6f} | iters: {NUM_ITERATIONS}")
        wandb_run.log({
            "step": step,
            "val/ema_bpb": ema_bpb,
            "val/ema_loss": ema_loss,
            "val/num_iterations": NUM_ITERATIONS,
        })
        val_bpb = ema_bpb
        val_loss = ema_loss
        if ema_bpb < min_val_bpb:
            min_val_bpb = ema_bpb
            min_val_loss = ema_loss

# Checkpoint weight averaging: average ALL saved late-epoch checkpoints, recency-weighted.
# (Window sweep 2026-06-05 on the wd-end-1.0 config swept exclude-last {0,1,2} × window
#  {2,3,4} × {recency,uniform}: averaging all swa_last_epochs ckpts recency-weighted beat
#  excluding the final epoch by 0.0019 same-checkpoint — recency weighting already dilutes
#  the final epoch's over-shrink, so no exclusion is needed. Bigger window > smaller.)
if len(late_ckpt_paths) >= 2:
    if ddp: dist.barrier()
    n = len(late_ckpt_paths)
    raw_w = list(range(1, n + 1))
    weights = [w / sum(raw_w) for w in raw_w]
    if master_process:
        ckpts = [torch.load(p, map_location="cpu", weights_only=True) for p in late_ckpt_paths]
        merged = {name: sum(weights[i] * ckpts[i][name].float() for i in range(n)) for name in ckpts[0]}
        with torch.no_grad():
            for name, p in orig_model.named_parameters():
                if name in merged: p.copy_(merged[name].to(p.device, p.dtype))
    if ddp:
        dist.barrier()
        for p in orig_model.parameters(): dist.broadcast(p.data, src=0)
    model.eval()
    with autocast_ctx:
        avg_bpb, avg_loss = evaluate_bpb(model, build_val_loader(), eval_steps, token_bytes, num_iterations=NUM_ITERATIONS,
                                         extra_kwargs=memory_kwargs)
    print0(f"Ckpt avg Val BPB: {avg_bpb:.6f} | Val Loss: {avg_loss:.6f} | iters: {NUM_ITERATIONS}")
    wandb_run.log({"ckpt_avg/bpb": avg_bpb, "ckpt_avg/loss": avg_loss, "ckpt_avg/num_iterations": NUM_ITERATIONS})
    if avg_loss < min_val_loss:
        min_val_loss, min_val_bpb = avg_loss, avg_bpb

# Research diagnostics on the final (averaged) weights; not part of training time.
final_probe_loss = train_probe_loss(NUM_ITERATIONS)
print0(f"Final train-probe loss: {final_probe_loss:.6f}")
strata = {}
if args.eval_strata:
    strata_counts = memory_counts
    if strata_counts is None or strata_counts.orders != (2, 3):
        strata_counts = ngram_lib.SuffixCounts.build(
            torch.cat(train_loader.doc_tokens), orders=(2, 3), device=device)
    model.eval()
    with autocast_ctx:
        strata = evaluate_strata(model, build_val_loader(), eval_steps, strata_counts,
                                 num_iterations=NUM_ITERATIONS, extra_kwargs=memory_kwargs)
    print0("Held-out loss by suffix support: " + ", ".join(
        f"{name} {strata[f'strata/{name}/loss']:.4f} ({100 * strata[f'strata/{name}/share']:.1f}%)"
        for name in ngram_lib.STRATA))
    wandb_run.log(strata)
mg_calibration = lookahead.calibration() if lookahead is not None else {}
mg_stats = lookahead.statistics(reduce=True) if lookahead is not None else {}
if orig_model.ngram is not None and not ngram_stats:
    # Arms without shrinkage built no counts during training; collect collision statistics
    # now, outside the charged time.
    ngram_stats = orig_model.ngram.hasher.collision_statistics(ngram_lib.SuffixCounts.build(
        torch.cat(train_loader.doc_tokens), orders=MEMORY_CONFIG.active_orders, device=device))
param_digest = None
if args.param_digest:
    _digest = hashlib.sha256()
    for _name, _p in orig_model.named_parameters():
        _digest.update(_name.encode())
        _digest.update(_p.detach().float().cpu().numpy().tobytes())
    param_digest = _digest.hexdigest()
if master_process and mg_calibration:
    with open(os.path.join(run_dir, lookahead_lib.CALIBRATION_FILE), "w") as f:
        json.dump(mg_calibration, f, indent=2)
if master_process and diag_records:
    with open(os.path.join(run_dir, "diag.json"), "w") as f:
        json.dump(diag_records, f, indent=2)
if args.cleanup_checkpoints and master_process:
    shutil.rmtree(checkpoints_dir, ignore_errors=True)

# Summary
wall_clock_time = time.time() - wall_clock_start
print0(f"Wall clock time: {wall_clock_time/60:.2f}m")
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
final_train_loss = smooth_train_loss / (1 - 0.9**step) if step > 0 else float('inf')
print0(f"Final train loss: {final_train_loss:.6f}")
print0(f"Min val BPB: {min_val_bpb:.6f}")
print0(f"Min val Loss: {min_val_loss:.6f}")
wandb_run.summary["final_train_loss"] = final_train_loss
wandb_run.summary["best_val_loss"] = min_val_loss

if master_process:
    result = {
        "matrix_lr": args.matrix_lr,
        "weight_decay": args.weight_decay,
        "wd_mid": args.wd_mid,
        "wd_end": args.wd_end,
        "dropout": args.dropout,
        "num_epochs": args.num_epochs,
        "max_train_steps": args.max_train_steps,
        "xsa_mode": args.xsa_mode,
        "mtp_predict": args.mtp_predict,
        "mtp_anneal_frac": args.mtp_anneal_frac,
        "effective_train_tokens": step * TOTAL_BATCH_SIZE,
        "steps": step,
        "max_num_iterations": NUM_ITERATIONS,
        "train_backprop_iterations": TRAIN_BACKPROP_ITERATIONS,
        "iteration_schedule": args.iteration_schedule,
        "iteration_transition_ratio": ITERATION_TRANSITION_RATIO,
        "avg_effective_layers": DEPTH * ITERATION_SCHEDULE.avg_iterations,
        "val_loss": val_loss,
        "best_val_loss": min_val_loss,
        "wandb_url": getattr(wandb_run, "url", None),
        # Research fork fields (research/meta_gradient/analyze.py reads these).
        "status": "complete",
        "track": "tiny",
        "arm": args.arm,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "argv": sys.argv[1:],
        "args": vars(args),
        "world_size": ddp_world_size,
        "virtual_ranks": VIRTUAL_RANKS,
        "micro_per_vrank": MICRO_PER_VRANK,
        "total_training_time_s": total_training_time + ngram_build_time,
        "ngram_build_time_s": ngram_build_time,
        "wall_clock_s": wall_clock_time,
        "diag_time_s": diag_time,
        "step_time_by_phase_s": {k: v[0] / v[1] for k, v in phase_times.items()},
        "update_rms": {k: v / max(update_count, 1) for k, v in update_sums.items()},
        "peak_memory_mib": get_max_memory() / 1024 / 1024,
        "ema_val_loss": ema_loss,
        "ckpt_avg_val_loss": avg_loss,
        "train_probe_loss": final_probe_loss,
        "epochs": epoch_records,
        "strata": strata,
        "mg": {**mg_stats, "calibration": mg_calibration},
        "final_train_loss": final_train_loss,
        "epochs_completed": len(epoch_records),
        "num_train_steps": num_train_steps,
        "final_param_digest": param_digest,
        "features": {
            "mg_steps": mg_step_count,
            "aux_split_steps": mg_step_count if args.aux_split != "shared" else 0,
            "diag_steps_run": len(diag_records),
            "spectral_calls": optimizer.calls["spectral"],
            "mona_calls": optimizer.calls["mona"],
            "credit_full_steps": credit_sampler.full_steps,
            "multi_pass_steps": credit_sampler.multi_pass_steps,
            "ngram_params": orig_model.ngram.parameter_count() if orig_model.ngram is not None else 0,
            "dense_params": (orig_model.dense_adapter.parameter_count()
                             if orig_model.dense_adapter is not None else 0),
            "act_stats": bool(act_buffers),
        },
        "credit": credit_sampler.summary(),
        "ngram": {"params": orig_model.ngram.parameter_count() if orig_model.ngram is not None else 0,
                  "dense_params": (orig_model.dense_adapter.parameter_count()
                                   if orig_model.dense_adapter is not None else 0),
                  **ngram_stats},
        "env": run_environment(),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print0(f"Result saved to {result_path}")

# Save final model
if master_process and not args.no_save_model:
    print0(f"Saving model to {artifact_model_path}")
    torch.save({n: p.data.float().cpu() for n, p in orig_model.named_parameters()}, artifact_model_path)

print0(f"Min val BPB: {min_val_bpb:.6f} | Min val Loss: {min_val_loss:.6f}")
total_wall_time = time.time() - _script_start
print0(f"Total wall time: {total_wall_time:.2f}s ({total_wall_time/60:.2f}m)")

wandb_run.finish()
if dist.is_initialized():
    dist.destroy_process_group()
if artifacts_log_f is not None:
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = stdout_orig
    sys.stderr = stderr_orig
    artifacts_log_f.close()