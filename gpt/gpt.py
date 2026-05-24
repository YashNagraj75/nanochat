import torch
import sys, os
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import partial


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt.common import get_base_dir
from gpt.flash_attention import flash_attn, flash_attn_func, flash_attn_kv_func
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


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embed = config.n_embd
        self.head_dim = self.n_embed // self.n_head
        assert self.n_embed % self.n_head == 0, (
            "Embedding dimension must be divisible by number of heads"
        )
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0, (
            "Number of KV heads must be less than or equal to number of query heads and divide it evenly"
        )
        self.q_proj = Linear(self.n_embed, self.n_embed, bias=False)
        self.v_proj = Linear(self.n_embed, self.n_kv_head * self.head_dim, bias=False)
        self.k_proj = Linear(self.n_embed, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = Linear(self.n_embed, self.n_embed, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = (
            Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

        def forward(self, x, ve, cos, sin, window_size=None, kv_cache=None):
            B, T, C = x.shape

            # Get the query, value and key projections according to GQA
            q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
            v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
            k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)

            # Now add value_embeddings to alternating layerr
            if ve is not None:
                ve = ve.view(B, T, self.n_kv_head, self.n_head)
                gate = 3 * torch.sigmoid(
                    self.ve_gate(ve[..., : self.ve_gate_channels])
                )  # Scaling by factor of 3
                v = v + gate.unsqueeze(-1) * ve

            # Add the rotary embeddings
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q = norm(q)
            k = norm(k)
            q = q * 1.2
            k = k * 1.2

            # Now attention with and without kv_cache
            if kv_cache is None:
                # While training have no kv_cache and causal attention with optional window
                y = flash_attn_func(q, k, v, causal=True, window_size=window_size)

            else:  # While inference have kv_cache
                k_cache, v_cache = kv_cache.get_layer_cache(layer_idx)
                y = flash_attn_kv_func(
                    q,
                    k_cache,
                    v_cache,
                    k,
                    v,
                    causal=True,
                    window_size=window_size,
                    cache_seqlens=kv_cache.cache_seqlens,
                )

                if self.layer_idx == kv_cache.n_layers - 1:
                    kv_cache.advance(T)

            y = y.contiguous().view(B, T, -1)
            y = self.o_proj(y)
            return y
