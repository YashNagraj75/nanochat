# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
# or with uv
uv sync

# Train the BPE tokenizer (reads dataset from ~/.cache/nanochat/base_data_climbmix/)
python scripts/train_tok.py --vocab-size 32768 --max-chars 2000000000 --doc-cap 10000

# Download dataset shards from HuggingFace (karpathy/climbmix-400b-shuffle)
python gpt/dataset.py -n <num_train_shards> -w <num_workers>
```

## Architecture

This project is a from-scratch GPT pretraining stack (inspired by karpathy/nanochat) targeting efficiency on Hopper-class GPUs.

### Model (`gpt/gpt.py`)
Defines `GPTConfig` (sequence_lens, vocab_size, n_layer, n_head, n_kv_head for GQA, n_embd) and the GPT model. Uses RMSNorm (`F.rms_norm`), a dtype-agnostic `Linear` subclass, and Rotary embeddings. The `has_ve` helper controls value-embedding placement (every other layer, starting from the last). The model is still under construction.

### Attention (`gpt/flash_attention.py`)
Two public functions exposed via `flash_attn` SimpleNamespace:
- `flash_attn_func(q, k, v, causal, window_size)` — for pretraining (no KV cache), tensors are `(B, T, H, D)`
- `flash_attn_kv_func(q, k_cache, v_cache, k, v, causal, window_size, cache_seqlens)` — for inference with KV cache

Both dispatch to Flash Attention 3 (loaded from `varunneal/flash-attention-3` kernel) when running on Hopper (SM 9.x) with bfloat16, and fall back to `F.scaled_dot_product_attention` otherwise. The SDPA fallback handles three cases: prefill (`Tq == Tk`, uses `is_causal=True`), decoding (`Tq == 1`), and chunk inference (explicit mask). Set `_override_imp = "sdpa"` or `"fa3"` to force a specific backend for testing.

### Optimizer (`gpt/optim.py`)
`MuonAdamW` (single GPU) and `DistMuonAdamW` (multi-GPU) combined optimizers:
- 2D matrix params → **Muon** (momentum + Polar Express orthogonalization + NorMuon variance reduction + cautious update), implemented as a single fused `torch.compile` kernel
- Embeddings, scalars, biases → **AdamW** (fused kernel with 0-D CPU tensors to prevent recompilation on hyperparameter changes)

`DistMuonAdamW` uses a 3-phase async comms pattern (reduce → compute+launch gather → wait gathers) with ZeRO-2-style sharding for AdamW and chunk-based sharding for Muon.

Param groups must explicitly set `kind: "adamw"` or `kind: "muon"`. All params in a Muon group must have the same shape.

### Tokenizer (`gpt/tokenizer.py`)
`RustBPE_Tokenizer` wraps `rustbpe` (fast BPE training in Rust) for training, and `tiktoken` for inference. Vocabulary is 32768 by default (9 special tokens + 256 byte tokens + BPE merges). Special tokens support chat templates via `render_conversation` (returns token ids + loss mask) and `render_for_completion` (for RL). System prompt content is prepended to the first user message.

### Data (`gpt/dataset.py`)
Downloads `karpathy/climbmix-400b-shuffle` parquet shards (up to shard_06542) from HuggingFace to `~/.cache/nanochat/base_data_climbmix/`. The last shard is always the validation split. `parquet_iter_batched` supports `start`/`step` for DDP rank-based shard skipping.

### Utilities (`gpt/common.py`)
- `get_base_dir()` → `~/.cache/nanochat/` (all training artifacts go here)
- `save_training_metadata()` → saves JSON, `.pt` tensor, and markdown simultaneously
- `COMPUTE_DTYPE` / `COMPUTE_DTYPE_REASON` — auto-detected from CUDA capability (SM ≥ 8.0 → bfloat16, else float32); overrideable via `NANOCHAT_COMPUTE_DTYPE` env var
