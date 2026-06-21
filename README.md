# nanochat

> **Work in Progress** — This project is under active development and is not yet complete. The pretraining loop, DDP training, and chat interface are still being built out.

A from-scratch GPT pretraining stack written in Python, targeting efficiency on Hopper-class (H100/H200) GPUs. Inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), but rebuilt with modern techniques: Grouped Query Attention, Flash Attention 3, a custom Muon+AdamW optimizer, BPE tokenization via Rust, and a KV-cached inference engine with tool-use support.

## Architecture

### Model (`gpt/gpt.py`)

A decoder-only transformer with several non-standard design choices:

| Component | Design |
|---|---|
| Normalization | RMSNorm (`F.rms_norm`) — no affine parameters, applied pre-block |
| Attention | Grouped Query Attention (GQA) with `n_head` query heads and `n_kv_head` KV heads |
| Position encoding | Rotary embeddings (RoPE) applied to Q and K after projection |
| Activation | Squared ReLU (`relu(x)²`) in MLP |
| Attention backend | Flash Attention 3 on Hopper GPUs, falls back to `F.scaled_dot_product_attention` |
| Logit capping | `softcap * tanh(logits / softcap)` (cap=15) for numerical stability |
| Sliding window | Per-layer pattern string (e.g. `"SLSL"`) — `S` = quarter-context, `L` = full context |

**Default config:** 12 layers, 12 query heads, 6 KV heads, 768 embedding dim, 2048 sequence length, 32768 vocab size.

**Non-standard features:**

- **Value Embeddings (ResFormer-style):** On alternating layers (starting from the last), the raw token embedding is injected directly into the value vectors via a learned gate (range 0–3). Prevents token identity from being washed out in deep networks.
- **Smear Gate:** Mixes the previous token's embedding into each token before attention runs — cheap bigram-level context without extra parameters.
- **Backout Lambda:** Subtracts a fraction of the mid-layer residual from the final residual before the LM head, removing low-level features so the LM head sees a cleaner, more abstract representation.
- **Residual Lambdas:** Per-layer learned scalars on the residual stream, initialized as a linear decay from 1.15 (early layers) to 1.05 (late layers), preventing vanishing gradients in deep networks.
- **X0 Lambdas:** Per-layer learned blend of the original token embeddings back into the residual stream — early layers get more, later layers less.

### Attention (`gpt/flash_attention.py`)

Two dispatch paths:

- **FA3** (`varunneal/flash-attention-3` kernel): used on Hopper (SM 9.x) with bfloat16. Supports sliding window and KV-cache with `flash_attn_with_kvcache`.
- **SDPA** (`F.scaled_dot_product_attention`): fallback for all other hardware. Handles prefill, decoding, and chunk inference with explicit masks.

Force a backend with `_override_imp = "sdpa"` or `"fa3"` for testing.

### Optimizer (`gpt/optim.py`)

`MuonAdamW` (single GPU) / `DistMuonAdamW` (multi-GPU) — a combined optimizer that routes parameters differently by type:

- **2D matrix params** (attention projections, MLP weights) → **Muon**: momentum + Polar Express orthogonalization + NorMuon variance reduction + cautious update, fused into a single `torch.compile` kernel.
- **Embeddings, scalars, LM head** → **AdamW**: fused kernel with 0-D CPU tensors to prevent recompilation on LR changes.

`DistMuonAdamW` uses ZeRO-2-style sharding for AdamW and a 3-phase async comms pattern (reduce → compute + launch gather → wait) for Muon. All param groups must explicitly specify `kind: "adamw"` or `kind: "muon"`, and all params in a Muon group must share the same shape.

### Tokenizer (`gpt/tokenizer.py`)

`RustBPE_Tokenizer` — a two-phase tokenizer:

- **Training:** `rustbpe` (Rust BPE implementation) for fast vocabulary construction.
- **Inference:** `tiktoken` for efficient encoding/decoding.

Vocabulary: 32768 tokens = 9 special tokens + 256 byte tokens + BPE merges. Special tokens include chat template markers (`<|python_start|>`, `<|python_end|>`, `<|output_start|>`, `<|output_end|>`, `<|assistant_end|>`) and support multi-turn conversation rendering with loss masking via `render_conversation` and `render_for_completion`.

### Dataset (`gpt/dataset.py`)

Downloads `karpathy/climbmix-400b-shuffle` parquet shards (up to shard 06542) from HuggingFace to `~/.cache/nanochat/base_data_climbmix/`. The last shard is always the validation split.

### Dataloader (`gpt/dataloader.py`)

Streams text from parquet files, handles DDP rank-based shard skipping, and supports approximate resume from a checkpoint state dict.

### Inference Engine (`gpt/engine.py`)

`Engine` wraps the model and tokenizer for KV-cached inference:

- **Prefill:** single forward pass over the prompt with batch size 1, populates the KV cache.
- **Decode:** clones the prefill KV cache for `num_samples` parallel generations; each row is an independent sequence with its own `RowState`.
- **Tool use:** native `<python>...</python>` block detection — expressions are evaluated with a sandboxed calculator and the result is injected back as forced tokens via `<|output_start|>...<|output_end|>`.

### Checkpoint Manager (`gpt/checkpoint_manager.py`)

Handles saving and loading model checkpoints with training metadata.

## Tech Stack

| Component | Library |
|---|---|
| Deep learning | PyTorch 2.11 |
| CUDA runtime | CUDA 13 (blas, cudnn, curand, cusolver, nccl, nvjitlink) |
| Attention kernel | Flash Attention 3 (`varunneal/flash-attention-3`) |
| Tokenization | `rustbpe` (training), `tiktoken` (inference), `tokenizers` |
| Dataset format | Apache Parquet via `pyarrow` |
| Model hub | `huggingface-hub` |
| CLI | Typer, Rich |
| Package manager | uv |
| Python | 3.14 |

## Project Structure

```
nanochat/
├── gpt/
│   ├── gpt.py               # GPT model, GPTConfig, attention, MLP, weight init
│   ├── engine.py            # KV-cached inference engine, KVCache, tool use
│   ├── flash_attention.py   # FA3 / SDPA dispatch layer
│   ├── optim.py             # MuonAdamW + DistMuonAdamW optimizer
│   ├── tokenizer.py         # RustBPE_Tokenizer (train + inference)
│   ├── dataloader.py        # Parquet streaming dataloader with DDP sharding
│   ├── dataset.py           # HuggingFace dataset downloader
│   ├── checkpoint_manager.py# Checkpoint save/load utilities
│   └── common.py            # COMPUTE_DTYPE, get_base_dir, dist helpers
├── scripts/
│   └── train_tok.py         # BPE tokenizer training script
├── main.py                  # CLI entry point (stub)
├── ARCHITECTURE_NOTES.md    # In-depth notes on design decisions
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Installation

### Prerequisites

- Python 3.14
- CUDA-capable GPU (Hopper/H100 recommended for FA3; any CUDA GPU otherwise)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
git clone https://github.com/YashNagraj75/nanochat.git
cd nanochat

# With uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

## Usage

### 1. Download the dataset

```bash
python gpt/dataset.py -n <num_train_shards> -w <num_workers>
```

Downloads `karpathy/climbmix-400b-shuffle` parquet shards to `~/.cache/nanochat/base_data_climbmix/`.

### 2. Train the tokenizer

```bash
python scripts/train_tok.py --vocab-size 32768 --max-chars 2000000000 --doc-cap 10000
```

Trains a BPE tokenizer on the downloaded dataset. Output is saved to `~/.cache/nanochat/`.

### 3. Override compute dtype (optional)

```bash
export NANOCHAT_COMPUTE_DTYPE=float32  # default: bfloat16 on SM >= 8.0
```

## Development Status

This project is a **work in progress and is not yet complete**. What is implemented:

- GPT model with GQA, RoPE, value embeddings, smear gate, backout lambda, residual/x0 lambdas
- Flash Attention 3 + SDPA fallback dispatch
- Muon + AdamW combined optimizer with single-GPU and DDP variants
- BPE tokenizer (Rust training + tiktoken inference)
- Parquet dataloader with DDP sharding and checkpoint resume
- KV-cached inference engine with multi-sample parallel generation and tool use
- Checkpoint manager

What is still under construction:

- Pretraining loop (training script)
- DDP / multi-GPU training launch
- Evaluation harness
- Chat interface and CLI
