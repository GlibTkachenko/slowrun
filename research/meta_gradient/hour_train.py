"""
Research fork of the one-hour record (commit 45bf3fd, 3.183 val loss in 59.3 min).

With default flags this script runs the record exactly. Every experiment that
applies to this track is a flag, and arms.py names the combinations:

  * --virtual-ranks: run the exact 8-rank meta-gradient (MG) algorithm on fewer GPUs.
  * A  --aux-split: auxiliary MTP block at weight 0.6 in the adaptation half only.
  * B  --mg-cproj: activation-metric radius or direction of the temporary step on c_proj.
  * C  --spectral / --mona-beta: update-map variants;
       --diag-steps: the diagnostic of which MG corrections survive Muon.

Usage:
    torchrun --standalone --nproc_per_node=8 research/meta_gradient/hour_train.py [flags]
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import math
import time
import json
import hashlib
import sys
import shutil
import argparse
import platform
import subprocess
from types import SimpleNamespace
from functools import partial
from dataclasses import dataclass
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
import wandb
import tiktoken

_LAB_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.abspath(os.path.join(_LAB_DIR, "..", ".."))

import lookahead as lookahead_lib
import mona as mona_lib
import probe as probe_lib
import spectral as spectral_lib
from loader import VirtualRankLoader, probe_sequences

_script_start = time.time()

# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Train GPT model")
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--num-epochs", type=int, default=11)
parser.add_argument("--patience", type=int, default=-1)
parser.add_argument("--run-name", type=str, default=None,
                    help="Run name under runs/ (default: random 6-char string)")
parser.add_argument("--scalar-lr", type=float, default=0.1)
parser.add_argument("--matrix-lr", type=float, default=0.04)
parser.add_argument("--weight-decay", type=float, default=1.3)
parser.add_argument("--total-batch-size", type=int, default=524288)
parser.add_argument("--save-result", type=str, default="")
parser.add_argument("--n_layer", type=int, default=30)
parser.add_argument("--n_head", type=int, default=14)
parser.add_argument("--n_embd", type=int, default=1792)
parser.add_argument("--lr_multiplier", type=float, default=0.25)
parser.add_argument("--input_bin", type=str, default=None)
parser.add_argument("--input_val_bin", type=str, default=None)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--dropout", type=float, default=0.05)   # 0.1 in entry 20
parser.add_argument("--dupe-start-epoch", type=int, default=7,
                    help="Epoch to enable layer duplication")
parser.add_argument("--dupe-layers-start", type=int, default=15,
                    help="First decoder layer to duplicate (inclusive)")
parser.add_argument("--dupe-layers-end", type=int, default=21,
                    help="Last decoder layer to duplicate (exclusive)")
parser.add_argument("--dupe-loops", type=int, default=2,
                    help="Number of extra replay passes through dupe layers")
parser.add_argument("--warmdown-ratio", type=float, default=None,
                    help="Override warmdown ratio (default 0.2)")
parser.add_argument("--logit-cap", type=float, default=10.0,
                    help="Logit soft-capping value (0=disabled)")
parser.add_argument("--logit-avg", type=int, default=3,
                    help="Number of late checkpoints for logit (probability) averaging (0=disabled)")
parser.add_argument("--logit-avg-dir", type=str, default=None,
                    help="Directory to save/load epoch checkpoints for logit averaging (default: <run_dir>/logit_avg)")
parser.add_argument("--logit-avg-mode", type=str, default="both",
                    choices=["equal", "weighted", "both"],
                    help="Weight scheme: equal, linear recency weighted, or compare both")
parser.add_argument("--eval-logit-avg", action="store_true",
                    help="Skip training and only run logit-avg eval on saved checkpoints")
parser.add_argument("--swa-last-epochs", type=int, default=3,
                    help="SWA: cosine-cycle LR in last N epochs for checkpoint diversity (0=off)")
parser.add_argument("--stoch-depth", type=float, default=0.0,   # 0.05 in entry 20
                    help="Stochastic depth max drop rate (linear schedule, 0=off)")
parser.add_argument("--mtp-weight", type=float, default=0.3,
                    help="Multi-token prediction weight (0=off)")
parser.add_argument("--iha", action="store_true", default=True,
                    help="Enable Interleaved Head Attention (cross-head Q/K/V mixing)")
parser.add_argument("--no-iha", action="store_false", dest="iha",
                    help="Disable IHA cross-head mixing")
parser.add_argument("--iha-lr", type=float, default=0.02,
                    help="LR for IHA mixing matrices")
parser.add_argument("--no-doc-shuffle", action="store_true",
                    help="Disable per-epoch document reshuffling (still shuffles batch order)")
# --- every-step meta-gradient step (default for this entry; --mg-every 0 --dropout 0.1 --stoch-depth 0.05 = entry 20) ---
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mg-every", type=int, default=1, help="meta-gradient step every N steps (1 = every step, the record; 0 = off = entry 20)")
parser.add_argument("--mg-step-norm", type=float, default=0.5, help="L2 norm of the temporary step on the plastic matrices")
parser.add_argument("--mg-step-schedule", type=str, default="lr", choices=["const", "lr"],
                    help="lr: scale the step norm by the warmdown learning-rate multiplier")
parser.add_argument("--mg-plastic", type=str, default="matrix_all", choices=["matrix_all", "mlp_all"])
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
                 help="Delete the logit-averaging checkpoints after the final evaluation")
# A: objective separation
lab.add_argument("--aux-split", choices=sorted(lookahead_lib.AUX_SPLITS), default="shared",
                 help="MTP block weight multipliers: shared (1,1), adapt (2,0), query (0,2)")
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
lab.add_argument("--param-digest", action="store_true",
                 help="Record a SHA-256 of the final parameters (for exact-equality checks)")
lab.add_argument("--diag-params", type=str, default="",
                 help="Comma-separated matrices for the singular-basis analysis")
args = parser.parse_args()

# Resolve output path
if args.output_json and not args.save_result:
    args.save_result = args.output_json

# =============================================================================
# Hardwired d12 (GPT-2 small) hyperparameters
# =============================================================================

# Architecture (defaults = d12 GPT-2 small)
DEPTH = args.n_layer if args.n_layer is not None else 12
N_EMBD = args.n_embd if args.n_embd is not None else 768
N_HEAD = args.n_head if args.n_head is not None else 6
HEAD_DIM = N_EMBD // N_HEAD
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
BASE_EMBEDDING_LR = 0.15
BASE_UNEMBEDDING_LR = 0.002

# Apply LR multiplier if provided (scales all LRs uniformly)
_lr_mult = args.lr_multiplier if args.lr_multiplier is not None else 1.0
MATRIX_LR = BASE_MATRIX_LR * _lr_mult
UNEMBEDDING_LR = BASE_UNEMBEDDING_LR * _lr_mult
EMBEDDING_LR = BASE_EMBEDDING_LR * _lr_mult
SCALAR_LR = BASE_SCALAR_LR * _lr_mult

WEIGHT_DECAY = args.weight_decay
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = args.warmdown_ratio if args.warmdown_ratio is not None else 0.2
FINAL_LR_FRAC = 0.0
WARMDOWN_POWER = 0.5      # sqrt-shaped warmdown (stays ~41% higher at midpoint than linear)
WD_PRE_HOLD_FRAC = 0.40   # hold at base WD for first 40% of training, then decay to LOW by SWA start
WD_SWA_LOW_FACTOR = 0.65  # WD at start of each SWA epoch (LR is high → less regularization)
WD_SWA_HIGH_FACTOR = 1.50 # WD at end of each SWA epoch (LR has decayed → more regularization)
LOGIT_CAP = args.logit_cap

# Research fork -----------------------------------------------------------------
ACT_STATS = args.mg_act_stats or args.mg_cproj in ("act_radius", "act_metric")
MONA = None
if args.mona_beta > 0:
    MONA = (args.mona_beta,
            args.mona_alpha if args.mona_alpha is not None else mona_lib.default_alpha(args.mona_beta))
if args.spectral == "augmented_polar" and (args.spectral_c != 0.5 or args.spectral_c_late is not None):
    raise ValueError("--spectral augmented_polar implements only exponent 0.5")
# Auxiliary MTP branch per half: 'base' (record weight), 'double' or 'off' (branch skipped).
_MTP_MODES = {1.0: "base", 2.0: "double", 0.0: "off"}
MTP_MODE_A, MTP_MODE_B = (_MTP_MODES[m] for m in lookahead_lib.aux_multipliers(args.aux_split))

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
        return run_name, os.path.join(RUNS_DIR, run_name)
    name = time.strftime('%Y%m%d_%H%M%S')
    return name, os.path.join(RUNS_DIR, name)

# =============================================================================
def load_state_dict_into_model(model, state_dict):
    """Load a state dict into model, handling dtype conversion."""
    for name, p in model.named_parameters():
        if name in state_dict:
            p.data.copy_(state_dict[name].to(p.device, dtype=p.dtype))

# =============================================================================
# Flash Attention (FA3 on Hopper)
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
    row_idx = (Tk - Tq) + torch.arange(Tq, device=q.device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=q.device).unsqueeze(0)
    mask = col_idx <= row_idx
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Flash Attention for training (FA3; SDPA fallback for smoke tests). q,k,v: (B, T, H, D)."""
    if _fa3 is not None:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    return _sdpa_attention(q, k, v, window_size, q.size(1) != k.size(1)).transpose(1, 2)

flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)

# =============================================================================
# GPT Model
# =============================================================================

@dataclass
class GPTConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 50257
    n_layer: int = DEPTH
    n_head: int = N_HEAD
    n_kv_head: int = N_HEAD
    n_embd: int = N_EMBD
    window_pattern: str = WINDOW_PATTERN
    dropout: float = 0.0
    stoch_depth: float = 0.05
    use_iha: bool = False
    iha_mix_v: bool = True
    act_stats: bool = False
    act_stride: int = 16
    act_decay: float = 0.9

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
        # Attention gate: per-head gating to enable context-based no-op
        self.attn_gate_channels = 12
        self.attn_gate = nn.Linear(self.attn_gate_channels, self.n_head, bias=False)
        # IHA: cross-head mixing matrices (Interleaved Head Attention)
        # Mixing is fused into projection weights at forward time.
        # Cost: [H,H]@[H,d*C] matmul is negligible vs the [B*T,C]@[C,H*d] projection.
        self.use_iha = config.use_iha
        if self.use_iha:
            self.q_mix = nn.Parameter(torch.zeros(self.n_head, self.n_head))
            self.k_mix = nn.Parameter(torch.zeros(self.n_kv_head, self.n_kv_head))
            self.iha_mix_v = config.iha_mix_v
            if self.iha_mix_v:
                self.v_mix = nn.Parameter(torch.zeros(self.n_kv_head, self.n_kv_head))

    def _fuse_mix(self, weight, mix, H):
        """Fuse mixing matrix into projection weight: W_fused[h] = sum_m mix[h,m]*W[m]."""
        d = self.head_dim
        return (mix @ weight.view(H, d, -1).flatten(1)).view_as(weight)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        if self.use_iha:
            # Fuse mixing into weights then project — grad flows through mix params
            q = F.linear(x, self._fuse_mix(self.c_q.weight, self.q_mix, self.n_head))
            q = q.view(B, T, self.n_head, self.head_dim)
            k = F.linear(x, self._fuse_mix(self.c_k.weight, self.k_mix, self.n_kv_head))
            k = k.view(B, T, self.n_kv_head, self.head_dim)
            if self.iha_mix_v:
                v = F.linear(x, self._fuse_mix(self.c_v.weight, self.v_mix, self.n_kv_head))
                v = v.view(B, T, self.n_kv_head, self.head_dim)
            else:
                v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        else:
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
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # Attention gate: per-head sigmoid gate
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
        self.resid_dropout = nn.Dropout(config.dropout)
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
        return self.resid_dropout(self.c_proj(h))

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        # Stochastic depth: linear schedule from 0 at layer 0 to stoch_depth at last layer
        self.drop_prob = config.stoch_depth * (layer_idx / max(config.n_layer - 1, 1))

    def forward(self, x, ve, cos_sin, window_size):
        # Stochastic depth: blend with identity when dropped (compile-friendly, no graph break)
        if self.training and self.drop_prob > 0:
            keep = (torch.rand((), device=x.device) >= self.drop_prob).to(x.dtype)
            x_in = x
            x = x + self.attn(norm(x), ve, cos_sin, window_size)
            x = x + self.mlp(norm(x))
            x = x_in + keep * (x - x_in)
        else:
            x = x + self.attn(norm(x), ve, cos_sin, window_size)
            x = x + self.mlp(norm(x))
        return x


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
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.ve_projs = nn.ModuleDict({str(i): nn.Linear(config.n_embd, kv_dim, bias=False) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # U-Net skip connections: encoder layer i → decoder layer (n_layer - 1 - i)
        self.encoder_layers = config.n_layer // 2
        self.skip_weights = nn.Parameter(torch.ones(self.encoder_layers))
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim, base=10000)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._dupe_layers = None  # (start, end) or None
        self.mtp_weight = args.mtp_weight
        if self.mtp_weight > 0:
            self.mtp_proj = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)
            self.mtp_block = Block(config, config.n_layer)

    def set_dupe_layers(self, start, end, loops=2):
        assert start >= self.encoder_layers, "dupe layers must be decoder-only"
        assert end <= self.config.n_layer
        self._dupe_layers = (start, end)
        self._dupe_loops = loops
        print0(f"Dupe layers {start}-{end-1}: {loops} extra replays ({loops+1} total passes)")

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        normal_std = self.config.n_embd ** -0.5
        all_blocks = list(self.transformer.h)
        if self.mtp_weight > 0:
            all_blocks.append(self.mtp_block)
            torch.nn.init.uniform_(self.mtp_proj.weight, -s, s)
        for block in all_blocks:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=normal_std)
            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=normal_std)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.attn_gate.weight)
            # IHA: initialize mixing matrices to identity (baseline-equivalent)
            if block.attn.use_iha:
                torch.nn.init.eye_(block.attn.q_mix)
                torch.nn.init.eye_(block.attn.k_mix)
                if block.attn.iha_mix_v:
                    torch.nn.init.eye_(block.attn.v_mix)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for proj in self.ve_projs.values():
            torch.nn.init.uniform_(proj.weight, -s, s)
        self.skip_weights.fill_(1.0)
        for block in all_blocks:
            if block.mlp.act_stats:
                block.mlp.act_moment.zero_()
                block.mlp.act_count.zero_()
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim, base=10000)
        self.cos = cos
        self.sin = sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
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

    def _avg_causal_attended_keys(self, window, seq_len):
        if window < 0 or window >= seq_len - 1:
            return (seq_len + 1) / 2
        max_keys = min(window + 1, seq_len)
        return max_keys - max_keys * (max_keys - 1) / (2 * seq_len)

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embedding lookup + elementwise scalars
        nparams_exclude = (self.transformer.wte.weight.numel()
                          + self.resid_lambdas.numel()
                          + self.x0_lambdas.numel()
                          + self.skip_weights.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Exact causal sliding-window attention FLOPs: 12 * h * q * E[keys attended per query]
        attn_flops = sum(12 * h * q * self._avg_causal_attended_keys(w[0], t) for w in self.window_sizes)
        return 6 * (nparams - nparams_exclude) + attn_flops

    def setup_optimizer(self):
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate IHA mixing params (small H×H matrices) from large matrix params
        iha_params = []
        iha_param_ids = set()
        all_blocks = list(self.transformer.h)
        if self.mtp_weight > 0:
            all_blocks_for_iha = all_blocks + [self.mtp_block]
        else:
            all_blocks_for_iha = all_blocks
        for block in all_blocks_for_iha:
            if block.attn.use_iha:
                iha_params.append(block.attn.q_mix)
                iha_params.append(block.attn.k_mix)
                iha_param_ids.add(id(block.attn.q_mix))
                iha_param_ids.add(id(block.attn.k_mix))
                if block.attn.iha_mix_v:
                    iha_params.append(block.attn.v_mix)
                    iha_param_ids.add(id(block.attn.v_mix))
        all_h_params = list(self.transformer.h.parameters())
        matrix_params = [p for p in all_h_params if id(p) not in iha_param_ids] + list(self.ve_projs.parameters())
        if self.mtp_weight > 0:
            mtp_params = [p for p in list(self.mtp_block.parameters()) + list(self.mtp_proj.parameters()) if id(p) not in iha_param_ids]
            matrix_params += mtp_params
        ve_params = []
        embed_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        skip_params = [self.skip_weights]

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=UNEMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
            dict(kind='adamw', params=embed_params, lr=EMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
            dict(kind='adamw', params=ve_params, lr=EMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
            dict(kind='adamw', params=resid_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=SCALAR_LR, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=skip_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
        ]
        # IHA mixing matrices: use AdamW with dedicated or scalar-like LR
        if iha_params:
            iha_lr = args.iha_lr if args.iha_lr is not None else SCALAR_LR
            param_groups.append(dict(kind='adamw', params=iha_params, lr=iha_lr, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0))
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=group_params, lr=MATRIX_LR,
                                     momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=WEIGHT_DECAY,
                                     mona=MONA, spectral=None))

        optimizer = DistMuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _run_decoder_layers(self, x, x0, encoder_outputs, start, end, T):
        """Run decoder layers [start, end), with U-Net skip connections."""
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for i in range(start, end):
            # Encoder layer j connects to decoder layer (n_layer - 1 - j)
            j = self.config.n_layer - 1 - i
            if 0 <= j < self.encoder_layers:
                x = x + self.skip_weights[i - self.encoder_layers] * encoder_outputs[j]
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = self.transformer.h[i](x, ve, cos_sin, self.window_sizes[i])
        return x

    def forward(self, idx, targets=None, loss_reduction='mean', mtp_mode='base'):
        """Runs the model; `mtp_mode` ('base', 'double' or 'off') scales or skips the MTP block."""
        B, T = idx.size()
        x = norm(self.transformer.wte(idx))
        x0 = x
        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        # Encoder half: run layers and collect outputs for skip connections
        encoder_outputs = []
        for i in range(self.encoder_layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = self.transformer.h[i](x, ve, cos_sin, self.window_sizes[i])
            encoder_outputs.append(x)

        # Decoder half
        dupe = self._dupe_layers
        if dupe is None:
            x = self._run_decoder_layers(x, x0, encoder_outputs,
                                        self.encoder_layers, self.config.n_layer, T)
        else:
            # First pass: encoder boundary through end of dupe range
            x = self._run_decoder_layers(x, x0, encoder_outputs,
                                        self.encoder_layers, dupe[1], T)
            # Extra replays through dupe range
            for _ in range(self._dupe_loops):
                x = self._run_decoder_layers(x, x0, encoder_outputs,
                                            dupe[0], dupe[1], T)
            # Remaining decoder layers
            x = self._run_decoder_layers(x, x0, encoder_outputs,
                                        dupe[1], self.config.n_layer, T)

        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = LOGIT_CAP * torch.tanh(logits / LOGIT_CAP) if LOGIT_CAP > 0 else logits
        if targets is None:
            return logits
        lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                  ignore_index=-1, reduction=loss_reduction)
        if loss_reduction != 'mean':
            return lm_loss
        if self.mtp_weight <= 0 or mtp_mode == 'off':
            return lm_loss, {'lm_loss': lm_loss}
        mtp_emb = norm(self.transformer.wte(targets[:, :-1].clamp(min=0)))
        combined = self.mtp_proj(torch.cat([x[:, :-1], mtp_emb], dim=-1))
        mT = combined.size(1)
        mtp_out = norm(self.mtp_block(combined, None, (self.cos[:, :mT], self.sin[:, :mT]), (-1, -1)))
        mtp_logits = self.lm_head(mtp_out)[..., :self.config.vocab_size].float()
        if LOGIT_CAP > 0:
            mtp_logits = LOGIT_CAP * torch.tanh(mtp_logits / LOGIT_CAP)
        mtp_loss = F.cross_entropy(mtp_logits.view(-1, mtp_logits.size(-1)),
                                   targets[:, 1:].reshape(-1), ignore_index=-1)
        mtp_weight = self.mtp_weight * (2.0 if mtp_mode == 'double' else 1.0)
        loss = lm_loss + mtp_weight * mtp_loss
        return loss, {'lm_loss': lm_loss, 'mtp_loss': mtp_loss}

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

def _normuon_tail(g, stacked_params, second_momentum_buffer, lr_t, wd_t, beta2_t, red_dim):
    """Variance reduction, cautious weight decay and the update, shared by both Muon steps."""
    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # MuonEq-R row normalization
    g /= g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7).to(g.dtype)
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
    _normuon_tail(g, stacked_params, second_momentum_buffer, lr_t, wd_t, beta2_t, red_dim)

def muon_step_spectral(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                       momentum_t, lr_t, wd_t, beta2_t, red_dim, method, exponent, eps, dtype, norm):
    """The record Muon step with Polar Express replaced by a spectral response (Priority C).

    Runs eagerly: the eigendecomposition dominates its cost. By default the output is
    rescaled per matrix to the norm Polar Express would have produced on the same
    input and cast to bfloat16 like it, so only the direction of the update changes.
    """
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    g /= g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7).to(g.dtype)
    g = spectral_lib.apply_map(g, method=method, exponent=exponent, eps=eps, dtype=dtype,
                               norm=norm).bfloat16()
    _normuon_tail(g, stacked_params, second_momentum_buffer, lr_t, wd_t, beta2_t, red_dim)

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
            if p.numel() < 1024:
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
        stacked_grads[:len(params)].copy_(torch.stack([p.grad for p in params]))
        if len(params) < padded:
            stacked_grads[len(params):].zero_()
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = _collective(dist.reduce_scatter_tensor, grad_chunk, stacked_grads, op=dist.ReduceOp.AVG)
        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group, info, gather_list, rank, world_size):
        for p in group['params']:
            pinfo = info['param_infos'][p]
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
                              group["ns_steps"], red_dim)
            else:
                self.calls["spectral"] += 1
                muon_step_spectral(grads, owned,
                                   state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                                   self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                                   red_dim, *group["spectral"])
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
    """Loads flat tokens , chunks into batches.

    doc_shuffle=False: applies the stored default sequence permutation (bitwise match
    with the old chunked pipeline), shuffles batch order each epoch.
    doc_shuffle=True: reshuffles documents each epoch, re-chunks, re-shuffles sequences.

    Always yields (x, y, epoch).
    """

    def __init__(self, filepath, B, T, device="cuda", doc_shuffle=False):
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
def evaluate_bpb(model, batches, steps, token_bytes):
    """Compute bits per byte and mean cross-entropy loss on a set of batches."""
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, _ = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none').view(-1)
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


@torch.no_grad()
def evaluate_bpb_logit_avg(eval_model, ckpt_paths, weights, steps):
    """Evaluate using probability averaging across checkpoints (proper ensemble).

    Loads each checkpoint from disk once, runs all val batches for it, then
    moves to the next — one CPU->GPU weight transfer per checkpoint, not per batch.
    Accumulates running scalar totals instead of per-token tensors.
    """
    dev = orig_model.get_device()
    V   = orig_model.config.vocab_size

    # Pre-fetch all val batches to CPU (token ids, tiny ~10 MB)
    val_loader = build_val_loader()
    all_x, all_y = [], []
    for _ in range(steps):
        x, y, _ = next(val_loader)
        all_x.append(x.cpu())
        all_y.append(y.cpu())

    BT = all_y[0].numel()

    # Per-batch accumulated weighted target probs, kept on GPU
    # Shape: (steps, BT) — only target-token probs, not full vocab
    batch_target_probs = torch.zeros(steps, BT, dtype=torch.float32, device=dev)

    # Checkpoint-outer, batch-inner: each checkpoint loaded exactly once
    for path, w in zip(ckpt_paths, weights):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        load_state_dict_into_model(orig_model, ckpt)
        del ckpt
        for i, (x, y) in enumerate(zip(all_x, all_y)):
            y_flat = y.view(-1).to(dev)
            with autocast_ctx:
                logits = eval_model(x.to(dev))
            probs = torch.softmax(logits.view(BT, V).float(), dim=-1)
            tgt   = probs[torch.arange(BT, device=dev), y_flat.clamp_min(0)]
            batch_target_probs[i].add_(tgt, alpha=w)

    # Compute metrics from accumulated target probs using running totals
    total_nats   = torch.tensor(0.0, dtype=torch.float64, device=dev)
    total_bytes  = torch.tensor(0, dtype=torch.int64, device=dev)
    total_loss   = torch.tensor(0.0, dtype=torch.float64, device=dev)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=dev)

    for i, y in enumerate(all_y):
        y_flat = y.view(-1).to(dev)
        mask = y_flat != -1
        log_probs = batch_target_probs[i].clamp_min(1e-40).log()
        num_bytes_batch = token_bytes[y_flat.clamp_min(0)]

        total_nats   += (log_probs.neg() * (num_bytes_batch > 0)).sum().double()
        total_bytes  += num_bytes_batch.sum()
        total_loss   += log_probs[mask].neg().sum().double()
        total_tokens += mask.sum()

    del batch_target_probs

    if dist.is_initialized():
        dist.all_reduce(total_nats,   op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes,  op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss,   op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    bpb  = total_nats.item()  / (math.log(2) * total_bytes.item())  if total_bytes.item()  > 0 else float('inf')
    loss = total_loss.item()  / total_tokens.item()                  if total_tokens.item() > 0 else float('inf')
    return bpb, loss

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
elif device_type == "cuda":
    raise RuntimeError("Flash Attention 3 is required but not available. A Hopper (sm90) GPU is needed.")
else:
    print0("Warning: CPU smoke test with the PyTorch SDPA fallback (no FA3)")

# Run / logging paths
run_name, run_dir = resolve_run_dir(args.run_name)
if dist.is_initialized():
    shared = [run_name]
    dist.broadcast_object_list(shared, src=0)
    run_name = shared[0]
    run_dir = os.path.join(RUNS_DIR, run_name)
checkpoints_dir = os.path.join(run_dir, "checkpoints")
if args.logit_avg_dir is None:
    args.logit_avg_dir = os.path.join(run_dir, "logit_avg")
terminal_log_path = os.path.join(run_dir, "terminal.log")
stdout_orig = sys.stdout
stderr_orig = sys.stderr
artifacts_log_f = None
result_path = os.path.join(run_dir, "result.json")
if master_process:
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "wandb"), exist_ok=True)
    shutil.copy2(__file__, os.path.join(run_dir, "train.py"))
    os.makedirs(os.path.join(run_dir, "lab"), exist_ok=True)
    for _module in sorted(os.listdir(_LAB_DIR)):
        if _module.endswith(".py"):
            shutil.copy2(os.path.join(_LAB_DIR, _module), os.path.join(run_dir, "lab", _module))
    artifacts_log_f = open(terminal_log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, artifacts_log_f)
    sys.stderr = TeeStream(sys.stderr, artifacts_log_f)

# wandb
_wandb_kwargs = {"project": "slowrun", "name": run_name, "dir": os.path.join(run_dir, "wandb")}
if args.wandb_group:
    _wandb_kwargs["group"] = args.wandb_group
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
print0(f"  stoch_depth={args.stoch_depth}")
print0(f"  total_batch_size={TOTAL_BATCH_SIZE}, device_batch_size={args.device_batch_size}")
print0(f"  matrix_lr={MATRIX_LR}, scalar_lr={SCALAR_LR}, embedding_lr={EMBEDDING_LR}, unembedding_lr={UNEMBEDDING_LR}")
print0(f"  weight_decay={WEIGHT_DECAY}, adam_betas={ADAM_BETAS}")
print0(f"  warmup_ratio={WARMUP_RATIO}, warmdown_ratio={WARMDOWN_RATIO}, final_lr_frac={FINAL_LR_FRAC}")
print0(f"  num_epochs={args.num_epochs}, patience={args.patience}")
print0(f"  dropout={args.dropout}, doc_shuffle={not args.no_doc_shuffle}")
print0(f"  run={run_name}")
print0(f"  run_dir={run_dir}")
if args.iha:
    print0(f"  iha=True, iha_lr={args.iha_lr}")
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
config = GPTConfig(vocab_size=vocab_size, dropout=args.dropout,
                   stoch_depth=args.stoch_depth,
                   use_iha=args.iha, iha_mix_v=args.iha,
                   act_stats=ACT_STATS, act_stride=args.mg_act_stride, act_decay=args.mg_act_decay)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = sum(p.numel() for p in model.parameters())
transformer_params = sum(p.numel() for p in model.transformer.h.parameters())
ve_params = sum(p.numel() for p in model.ve_projs.parameters())
lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
other_params = param_counts - transformer_params - ve_params - lm_head_params
num_flops_per_token = model.estimate_flops()
print0(f"Parameters: {param_counts:,} (transformer: {transformer_params:,}, value_embeds: {ve_params:,}, lm_head: {lm_head_params:,}, other: {other_params:,})")
print0(f"FLOPs per token: {num_flops_per_token:e}")

# Compile
orig_model = model
model = torch.compile(model, dynamic=False)

# Optimizer
optimizer = model.setup_optimizer()

# ---- every-step meta-gradient step (research fork: research/meta_gradient/lookahead.py) ----
MG = args.mg_every > 0
muon_groups = [g["params"] for g in optimizer.param_groups if g["kind"] == "muon"]
mg_names, mg_params, mg_owned = lookahead_lib.select_plastic(
    list(orig_model.named_parameters()), muon_groups, rule=args.mg_plastic,
    rank=ddp_rank, world_size=ddp_world_size)
_mlp_blocks = [(f"transformer.h.{i}", block) for i, block in enumerate(orig_model.transformer.h)]
if orig_model.mtp_weight > 0:
    _mlp_blocks.append(("mtp_block", orig_model.mtp_block))
act_moments = {
    f"{prefix}.mlp.c_proj.weight": (block.mlp.act_moment, block.mlp.act_count)
    for prefix, block in _mlp_blocks if block.mlp.act_stats
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
           + f"; MTP block per half {MTP_MODE_A}/{MTP_MODE_B}")
elif args.aux_split != "shared" or args.mg_cproj != "global":
    raise ValueError("--aux-split and --mg-cproj need the MG split (--mg-every >= 1)")

# Dataloaders. The training loader emulates --virtual-ranks data-parallel ranks, so the MG
# split sees exactly the record's per-rank microbatches on fewer GPUs.
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


def train_probe_loss():
    """Returns the clean next-token loss on the fixed training probe (all ranks)."""
    was_training = model.training
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad(), autocast_ctx:
        for xs, ys in probe_loss_batches:
            losses = model(xs, ys, loss_reduction='none')
            totals[0] += losses.double().sum()
            totals[1] += losses.numel()
    if dist.is_initialized():
        dist.all_reduce(totals)
    model.train(was_training)
    return float(totals[0] / totals[1])


def mtp_kwargs(mode):
    """Model arguments selecting the MTP branch of one half; empty for the record."""
    return {} if mode == "base" else {"mtp_mode": mode}


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
    if _group["kind"] == "muon":
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
    """MuonEq-R row normalization, applied by the record before orthogonalization."""
    return m / m.norm(dim=-1, keepdim=True).clamp_min(1e-7)


def survival_inputs():
    """Binds the diagnostic to the objective of each half."""

    def run_half(batches, half):
        mode = MTP_MODE_A if half == "A" else MTP_MODE_B
        for xs, ys, *_ in batches:
            with autocast_ctx:
                loss, _ = model(xs, ys, **mtp_kwargs(mode))
            (loss / (2 * len(batches))).backward()

    return probe_lib.SurvivalInputs(
        run_half=run_half,
        probe_loss=train_probe_loss,
        muon_input=diag_muon_input,
        preprocess=diag_preprocess,
        zero_grad=lambda: model.zero_grad(set_to_none=True),
    )


# Training config
num_iterations = round(TOKENS_PER_EPOCH * args.num_epochs / TOTAL_BATCH_SIZE)  # estimate for LR schedule
print0(f"Batch size: {TOTAL_BATCH_SIZE:,} tokens, grad accum: {grad_accum_steps} steps")
print0(f"Virtual ranks: {VIRTUAL_RANKS} ({LOCAL_VRANKS} per process, {MICRO_PER_VRANK} microbatch(es) each)")
print0(f"Training for {args.num_epochs} epoch(s) (~{num_iterations} steps estimated)")
print0(f"Eval set: {EVAL_TOKENS:,} tokens")

# Schedulers
def get_lr_multiplier(it):
    warmup = round(WARMUP_RATIO * num_iterations)
    warmdown = round(WARMDOWN_RATIO * num_iterations)
    if it < warmup: return (it + 1) / warmup
    elif it <= num_iterations - warmdown: return 1.0
    else:
        progress = (num_iterations - it) / warmdown
        shaped = progress ** WARMDOWN_POWER  # concave (stays higher longer) when POWER < 1
        return shaped + (1 - shaped) * FINAL_LR_FRAC

def get_muon_momentum(it):
    return (1 - min(it / 300, 1)) * 0.85 + min(it / 300, 1) * 0.95

steps_per_epoch = num_iterations / args.num_epochs
_swa_start_step = (num_iterations - args.swa_last_epochs * steps_per_epoch) if args.swa_last_epochs > 0 else -1

def get_wd_multiplier(it):
    """Anti-phase WD: hold at 1.0 pre-SWA, decay to LOW by SWA start, then sawtooth LOW→HIGH per SWA epoch (anti-phase with the LR cosine cycle)."""
    if _swa_start_step >= 0 and it >= _swa_start_step:
        cycle_pos = (it - _swa_start_step) % steps_per_epoch
        frac = cycle_pos / steps_per_epoch
        return WD_SWA_LOW_FACTOR + (WD_SWA_HIGH_FACTOR - WD_SWA_LOW_FACTOR) * frac
    t = it / num_iterations
    if t < WD_PRE_HOLD_FRAC:
        return 1.0
    swa_start_frac = _swa_start_step / num_iterations if _swa_start_step > 0 else 1.0
    decay_frac = (t - WD_PRE_HOLD_FRAC) / max(swa_start_frac - WD_PRE_HOLD_FRAC, 1e-6)
    decay_frac = min(max(decay_frac, 0.0), 1.0)
    return 1.0 - (1.0 - WD_SWA_LOW_FACTOR) * decay_frac

# Training loop
step = 0
min_val_bpb = float("inf")
min_val_loss = float("inf")
epochs_without_improvement = 0
smooth_train_loss = 0
total_training_time = 0
timed_steps = 0
timing_start_step = 4  # skip first compile + 3 warmup steps
eval_steps = EVAL_TOKENS // (args.device_batch_size * MAX_SEQ_LEN * ddp_world_size)
dupe_active = False

# Research bookkeeping (written to result.json).
# Emulated ranks keep separate activation moments and start each step from the same RNG
# state, like the identically seeded physical ranks of the record (no-op on 8 GPUs).
vrank_state = lookahead_lib.VirtualRankState(act_buffers, LOCAL_VRANKS)
mg_step_count = 0
epoch_records = []
diag_records, diag_time = [], 0.0
# Actual update size of the representative matrices, sampled every --log-every steps, and
# mean step time per training phase.
update_probe = [(n, named_parameters[n]) for n in diag_params if n in named_parameters]
update_sums, update_count, phase_times = {}, 0, {}
diag_dir = os.path.join(run_dir, "diag")
eq_loss = wt_loss = None

late_checkpoint_paths = []  # paths to saved epoch checkpoints for logit averaging
logit_avg_count = args.logit_avg
if logit_avg_count > 0 and master_process:
    os.makedirs(args.logit_avg_dir, exist_ok=True)
if logit_avg_count > 0:
    print0(f"Logit averaging: saving last {logit_avg_count} epoch checkpoints to {args.logit_avg_dir}/")

if args.eval_logit_avg:
    print0("--eval-logit-avg set: skipping training, loading checkpoints from disk.")
else:
    # Initial val evaluation
    model.eval()
    val_loader = build_val_loader()
    with autocast_ctx:
        val_bpb, val_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    print0(f"Step {step:05d} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
    wandb_run.log({"step": step, "val/bpb": val_bpb, "val/loss": val_loss})
    min_val_bpb = val_bpb
    min_val_loss = val_loss
    model.train()

while not args.eval_logit_avg and current_epoch <= args.num_epochs:
    if not dupe_active and current_epoch >= args.dupe_start_epoch:
        print0(f"\n=== Enabling dupe-layers at epoch {current_epoch} ===")
        orig_model.set_dupe_layers(args.dupe_layers_start, args.dupe_layers_end, args.dupe_loops)
        model = torch.compile(orig_model, dynamic=False)
        # model = orig_model # replace compile with this line for eager mode
        dupe_active = True
        timing_start_step = step + 4  # skip dupe recompile + 3 warmup steps
        gc.enable(); gc.collect()

    # Schedules. They are applied before the forward pass, which never reads them, so that the
    # survival diagnostic can run the optimizer exactly as this step will.
    lrm = get_lr_multiplier(step)
    # SWA: cosine-cycle LR in final epochs for diverse checkpoints to average
    if _swa_start_step >= 0 and step >= _swa_start_step:
        cycle_pos = (step - _swa_start_step) % steps_per_epoch
        swa_base = max(lrm, 0.05)
        lrm = 0.05 + (swa_base - 0.05) * (1 + math.cos(math.pi * cycle_pos / steps_per_epoch)) / 2
    # WD schedule: pre-SWA decay from base to 0.65×base, then anti-phase sawtooth
    # during SWA's N epochs (0.65→1.50×base per epoch, anti-phase with LR cosine cycle).
    wdm = get_wd_multiplier(step)
    spectral_setting = None
    if args.spectral != "polar_express":
        exponent = args.spectral_c
        if args.spectral_c_late is not None and step >= args.spectral_switch_frac * num_iterations:
            exponent = args.spectral_c_late
        spectral_setting = (args.spectral, exponent, args.spectral_eps,
                            torch.float64 if args.spectral_fp64 else torch.float32, args.spectral_norm)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if "initial_wd" not in group:
            group["initial_wd"] = group.get("weight_decay", 0.0)
        group["weight_decay"] = group["initial_wd"] * wdm
        if group['kind'] == 'muon':
            group["momentum"] = get_muon_momentum(step)
            group["spectral"] = spectral_setting
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
            inputs=survival_inputs(),
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
            os.makedirs(diag_dir, exist_ok=True)
            torch.save(diag_spectra, os.path.join(diag_dir, f"spectra_step{step:05d}_rank{ddp_rank}.pt"))
        print0(f"[diag] step {step}: probe gain of the baseline step {diag_metrics['diag/probe_gain_base']:.3e}, "
               f"extra gain from MG {diag_metrics['diag/probe_gain_mg']:.3e} ({time.time() - _t:.1f}s)")

    # Training step
    synchronize()
    t0 = time.time()
    if mg_step:
        # Split pairing per virtual rank: the first half of its micro-batches adapts, the second
        # half is evaluated at the adapted point; the optimizer receives the full-batch mean gradient.
        mg_step_count += 1
        for v in range(LOCAL_VRANKS):
            adapt, query = lookahead_lib.split_halves(
                step_batches[v * MICRO_PER_VRANK:(v + 1) * MICRO_PER_VRANK])
            vrank_state.enter(v)
            lookahead.stash()
            for xs, ys, _ in adapt:
                with autocast_ctx:
                    loss, metrics = model(xs, ys, **mtp_kwargs(MTP_MODE_A))
                train_loss = loss.detach()
                (loss / grad_accum_steps).backward()
            lookahead.displace(get_lr_multiplier(step))
            for xs, ys, _ in query:
                with autocast_ctx:
                    loss, metrics = model(xs, ys, **mtp_kwargs(MTP_MODE_B))
                train_loss = loss.detach()
                (loss / grad_accum_steps).backward()
            lookahead.restore()
            vrank_state.exit(v)
    else:
        for v in range(LOCAL_VRANKS):
            vrank_state.enter(v)
            for xs, ys, _ in step_batches[v * MICRO_PER_VRANK:(v + 1) * MICRO_PER_VRANK]:
                with autocast_ctx:
                    loss, metrics = model(xs, ys)
                train_loss = loss.detach()
                (loss / grad_accum_steps).backward()
            vrank_state.exit(v)
    step_batches = train_loader.next_step()
    epoch = step_batches[0][2]

    # Update optimizer
    measure_update = step % args.log_every == 0
    if measure_update:
        update_before = [p.detach().clone() for _, p in update_probe]
    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    synchronize()
    dt = time.time() - t0
    if step + 1 >= timing_start_step:
        phase = ("replay" if dupe_active else "plain") + ("_swa" if 0 <= _swa_start_step <= step else "")
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
    pct = 100 * step / num_iterations
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / (gpu_peak_flops * ddp_world_size)
    if step >= timing_start_step:
        total_training_time += dt
        timed_steps += 1
    eta_str = f" | eta: {(num_iterations - step) * total_training_time / timed_steps / 60:.1f}m" if timed_steps > 0 else ""
    dupe_str = " [DUPE]" if dupe_active else ""
    print0(f"step {step:05d} ({pct:.2f}%) | loss: {debiased:.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f}%{dupe_str}{eta_str}")
    wandb_run.log({"step": step, "train/loss": debiased, "train/mfu": mfu,
                   **{f"train/{k}": v.item() for k, v in metrics.items()}})
    if update_log or (step % args.log_every == 0 and lookahead is not None):
        wandb_run.log({"step": step, **update_log,
                       **(lookahead.statistics() if lookahead is not None else {})})

    # Synchronize epoch across ranks (different ranks may exhaust data at different steps)
    if ddp:
        epoch_tensor = torch.tensor([epoch], dtype=torch.long, device=device)
        dist.all_reduce(epoch_tensor, op=dist.ReduceOp.MAX)
        epoch = epoch_tensor.item()

    # Epoch boundary: evaluate when the dataloader advances to a new epoch
    if epoch != current_epoch:
        model.eval()
        val_loader = build_val_loader()
        with autocast_ctx:
            val_bpb, val_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        probe_loss = train_probe_loss()
        epoch_records.append({"epoch": current_epoch, "step": step, "val_loss": val_loss,
                              "val_bpb": val_bpb, "train_probe_loss": probe_loss,
                              "training_time_s": total_training_time})
        print0(f"Step {step:05d} | Epoch {current_epoch} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f} "
               f"| train probe: {probe_loss:.6f}")
        wandb_run.log({"step": step, "epoch": current_epoch, "val/bpb": val_bpb, "val/loss": val_loss,
                       "train_probe/loss": probe_loss, "gap/val_minus_train_probe": val_loss - probe_loss})
        # Early stopping
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
            min_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience >= 0 and epochs_without_improvement >= args.patience:
                print0(f"Early stopping: no improvement for {args.patience} epoch(s)")
                break
        # Save checkpoint to disk for logit averaging
        if logit_avg_count > 0:
            ckpt_path = os.path.join(args.logit_avg_dir, f"epoch_{current_epoch:03d}.pt")
            if master_process:
                ckpt = {name: p.data.float().cpu() for name, p in orig_model.named_parameters()}
                torch.save(ckpt, ckpt_path)
                del ckpt
            late_checkpoint_paths.append(ckpt_path)
            if len(late_checkpoint_paths) > logit_avg_count:
                old = late_checkpoint_paths.pop(0)
                if master_process and os.path.exists(old):
                    os.remove(old)
            print0(f"  Saved checkpoint {ckpt_path} ({len(late_checkpoint_paths)}/{logit_avg_count})")

        model.train()
        # Update num_iterations estimate now that we know real steps per epoch
        # steps_per_epoch = step // current_epoch
        # num_iterations = steps_per_epoch * args.num_epochs
        # print0(f"Epoch {current_epoch} took {steps_per_epoch} steps. Updated estimate: {num_iterations} total steps.")
        current_epoch = epoch

    # GC management
    if step == 1:
        gc.collect(); gc.freeze(); gc.disable()

# =============================================================================
# Post-training: evaluate checkpoint averages
# =============================================================================

# Evaluate logit (probability) average
if logit_avg_count > 0:
    # In eval-only mode, discover checkpoints from disk; otherwise use what was saved during training
    if args.eval_logit_avg:
        import glob as _glob
        all_disk = sorted(_glob.glob(os.path.join(args.logit_avg_dir, "epoch_*.pt")))
        ckpt_paths_for_logit = all_disk[-logit_avg_count:]
    else:
        ckpt_paths_for_logit = late_checkpoint_paths

    if len(ckpt_paths_for_logit) >= 2:
        n = len(ckpt_paths_for_logit)
        print0(f"\n--- Evaluating logit avg ({n} checkpoints: {[os.path.basename(p) for p in ckpt_paths_for_logit]}) ---")

        la_model = torch.compile(orig_model, dynamic=False)
        la_model.eval()

        def _run_mode(label, weights):
            print0(f"  [{label}] weights: {[f'{w:.3f}' for w in weights]}")
            bpb, loss = evaluate_bpb_logit_avg(la_model, ckpt_paths_for_logit, weights, eval_steps)
            print0(f"  [{label}] Val BPB: {bpb:.6f} | Val Loss: {loss:.6f}")
            wandb_run.log({f"logit_avg_{label}/bpb": bpb, f"logit_avg_{label}/loss": loss})
            return bpb, loss

        equal_w    = [1.0 / n] * n
        raw_w      = list(range(1, n + 1))
        weighted_w = [w / sum(raw_w) for w in raw_w]

        if args.logit_avg_mode in ("equal", "both"):
            eq_bpb, eq_loss = _run_mode("equal", equal_w)
            if eq_loss < min_val_loss:
                min_val_loss, min_val_bpb = eq_loss, eq_bpb
                print0(f"  ** New best! (logit avg equal weights)")

        if args.logit_avg_mode in ("weighted", "both"):
            wt_bpb, wt_loss = _run_mode("weighted", weighted_w)
            if wt_loss < min_val_loss:
                min_val_loss, min_val_bpb = wt_loss, wt_bpb
                print0(f"  ** New best! (logit avg recency weights)")


# Research diagnostics on the final weights; not part of training time.
final_probe_loss = train_probe_loss() if not args.eval_logit_avg else None
mg_calibration = lookahead.calibration() if lookahead is not None else {}
mg_stats = lookahead.statistics(reduce=True) if lookahead is not None else {}
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
    shutil.rmtree(args.logit_avg_dir, ignore_errors=True)
    shutil.rmtree(checkpoints_dir, ignore_errors=True)


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


# Summary
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
final_train_loss = smooth_train_loss / (1 - 0.9**step) if step > 0 else float('inf')
print0(f"Final train loss: {final_train_loss:.6f}")
print0(f"Min val BPB: {min_val_bpb:.6f}")
print0(f"Min val Loss: {min_val_loss:.6f}")
wandb_run.summary["final_train_loss"] = final_train_loss
wandb_run.summary["best_val_loss"] = min_val_loss

_result_out = args.save_result or result_path
if master_process:
    result = {
        "matrix_lr": args.matrix_lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "val_loss": val_loss,
        "best_val_loss": min_val_loss,
        "wandb_url": getattr(wandb_run, "url", None),
        # Research fork fields (research/meta_gradient/analyze.py reads these).
        "status": "complete",
        "track": "hour",
        "arm": args.arm,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "argv": sys.argv[1:],
        "args": vars(args),
        "world_size": ddp_world_size,
        "virtual_ranks": VIRTUAL_RANKS,
        "micro_per_vrank": MICRO_PER_VRANK,
        "steps": step,
        "total_training_time_s": total_training_time,
        "timed_steps": timed_steps,
        "wall_clock_s": time.time() - _script_start,
        "diag_time_s": diag_time,
        "step_time_by_phase_s": {k: v[0] / v[1] for k, v in phase_times.items()},
        "update_rms": {k: v / max(update_count, 1) for k, v in update_sums.items()},
        "peak_memory_mib": get_max_memory() / 1024 / 1024,
        "logit_avg_equal_loss": eq_loss,
        "logit_avg_weighted_loss": wt_loss,
        "train_probe_loss": final_probe_loss,
        "epochs": epoch_records,
        "mg": {**mg_stats, "calibration": mg_calibration},
        "final_train_loss": final_train_loss,
        "epochs_completed": len(epoch_records),
        "num_train_steps": num_iterations,
        "final_param_digest": param_digest,
        "features": {
            "mg_steps": mg_step_count,
            "aux_split_steps": mg_step_count if args.aux_split != "shared" else 0,
            "diag_steps_run": len(diag_records),
            "spectral_calls": optimizer.calls["spectral"],
            "mona_calls": optimizer.calls["mona"],
            "act_stats": bool(act_buffers),
        },
        "env": run_environment(),
    }
    with open(_result_out, "w") as f:
        json.dump(result, f, indent=2)
    print0(f"Result saved to {_result_out}")

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
