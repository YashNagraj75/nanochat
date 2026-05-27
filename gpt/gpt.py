import torch
import sys, os
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import partial


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt.common import COMPUTE_DTYPE, get_base_dir
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
    """
       Causal Self-Attention with Grouped Query Attention (GQA) and optional Value Embeddings.

       DATA FLOW
       ---------
       Input x: (B, T, n_embed)
           │
           ├─ q_proj ──→ Q: (B, T, n_head,    head_dim)   # all query heads, full width
           ├─ k_proj ──→ K: (B, T, n_kv_head, head_dim)   # fewer KV heads (GQA)
           └─ v_proj ──→ V: (B, T, n_kv_head, head_dim)
                   │
                   [optional: add gated raw token embedding to V — see Value Embeddings below]
                   │
                   [apply RoPE to Q and K, then normalize + scale]
                   │
    //               flash_attn(Q, K, V)  →  y: (B, T, n_head, head_dim)
                   │
                   reshape → (B, T, n_embed)
                   │
           o_proj ──→ output: (B, T, n_embed)   ← added back to the residual stream by the caller

       WHY head_dim = n_embed // n_head (NOT n_kv_head)
       -------------------------------------------------
       head_dim is the size of the vector each individual head works with.
       That split is always over n_head (the number of query heads), so:
           head_dim = n_embed // n_head   e.g. 768 // 12 = 64

       In GQA, KV heads are fewer (e.g. 4 instead of 12) but they still use the
       same head_dim = 64. The "grouping" means each KV head is shared by
       (n_head // n_kv_head) query heads. Using n_kv_head here would give the
       wrong (larger) head_dim and break all projection shapes.

       VALUE EMBEDDINGS (VE) — every other layer starting from the last
       ----------------------------------------------------------------
       Problem: in deep transformers, a token's identity gets washed out as
       the residual stream is rewritten layer by layer.

       Solution: on alternating layers (controlled by has_ve), inject the raw
       token embedding (from the embedding table, before any transformer processing)
       directly into the value vectors via a learned gate:

           gate = 3 * sigmoid( W_gate( x[..., :12] ) )   # shape (B, T, n_kv_head)
           v    = v + gate.unsqueeze(-1) * ve

       - gate range (0, 3): the 3× ceiling lets ve *dominate* v if the model learns to,
         not just blend with it.
       - Only the first 12 dims of x are used for the gate (cheap — avoids a full
         n_embed projection just to compute a scalar per head).
       - Result: v carries both "what the model thinks this position means" and
         "what this token literally is", weighted by the gate.

       TRAINING vs INFERENCE
       ---------------------
       Training (kv_cache=None): standard causal flash attention, optional sliding window.
       Inference (kv_cache provided): uses a KV cache; the last layer advances the
       cache pointer by T after each forward pass.
    """

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


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.fc2 = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x).square()
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print(
                f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency"
            )
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
            }
        )
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(
            torch.ones(config.n_layer)
        )  # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(
            torch.zeros(config.n_layer)
        )  # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(padded_vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = (
            config.sequence_len * 10
        )  # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.register_buffer(
            "cos", cos, persistent=False
        )  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad():os.wait
    def init_weights(self):
        """
        Inits weight
            1. wte (embedding): normal, std= 1.0
            2. lm_head: normal, std= 0.001
            for all the blocks:
              1. q_proj, k_proj, v_proj: uniform, std= 1/sqrt(n_embed)
              2. o_proj: zeros
              3. mlp_fc1: uniform, std= 1/sqrt(n_embed)
              4. mlp_fc2: zeros
        """

        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)


        # Now for the transformer blocks
        n_embed = self.config.n_embed
        a = 3**0.5 * n_embed ** -0.5 
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.q_proj.weight, -a,a)
            torch.nn.init.uniform_(block.attn.k_proj.weight, -a,a)
            torch.nn.init.uniform_(block.attn.v_proj.weight, -a,a)
            torch.nn.init.uniform_(block.mlp.fc1.weight, -a,a)  # Ablation instead of 0.4 times lets just do same as the attn heads
            torch.nn.init.zeros_(block.attn.o_proj.weight)
            torch.nn.init.zeros_(block.mlp.fc2.weight)

        # Per layer initializations
        layers = self.config.n_layer
        for layer in range(layers):
            self.resid_lambdas.data[layer] = 1.15 - (0.1 * layer / max(layers -1, 1))
            self.x0_lambdas.data[layer] = 0.20 - (0.15 * layer / max(layers -1, 1)) # initial layers get more embeddings blending

        # Smear and blackout gates 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.1, 0.15)

        # Now value embeddings same as v_proj
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -a,a)

        # Now init the value_gates at a higher value so that they start slightly higher than neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0,0.2)


        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)























