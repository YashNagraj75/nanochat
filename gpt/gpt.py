import torch
import sys, os
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import partial


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt.common import get_base_dir
from gpt.flash_attention import flash_attn
from gpt.optim import MuonAdamW, DistMuonAdamW


@dataclass
class GPTConfig:
    sequence_lens: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # No of query heads
    n_kv_head: int = 6  # No of kv heads (GQA)
    n_embd: int = 768


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.dim() == 4
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    x_rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rotated * sin
