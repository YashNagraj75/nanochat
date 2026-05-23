# Graph Report - .  (2026-05-22)

## Corpus Check
- Corpus is ~5,675 words - fits in a single context window. You may not need a graph.

## Summary
- 77 nodes · 98 edges · 12 communities (9 shown, 3 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Tokenizer|Tokenizer]]
- [[_COMMUNITY_Model Architecture|Model Architecture]]
- [[_COMMUNITY_Flash Attention|Flash Attention]]
- [[_COMMUNITY_Distributed Optimizer|Distributed Optimizer]]
- [[_COMMUNITY_AdamW Implementation|AdamW Implementation]]
- [[_COMMUNITY_Muon Implementation|Muon Implementation]]
- [[_COMMUNITY_Tokenizer Loading|Tokenizer Loading]]
- [[_COMMUNITY_Common Utilities|Common Utilities]]
- [[_COMMUNITY_Optimizer Coordination|Optimizer Coordination]]
- [[_COMMUNITY_Dataset Loading|Dataset Loading]]

## God Nodes (most connected - your core abstractions)
1. `RustBPE_Tokenizer` - 12 edges
2. `DistMuonAdamW` - 10 edges
3. `step()` - 8 edges
4. `MuonAdamW` - 7 edges
5. `encode_special()` - 5 edges
6. `_sdpa_attention_cal()` - 4 edges
7. `adamw_step_fused()` - 4 edges
8. `muon_step_fused()` - 4 edges
9. `Linear` - 4 edges
10. `flash_attn_func()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `GPTConfig` --uses--> `DistMuonAdamW`  [INFERRED]
  gpt.py → optim.py
- `Linear` --uses--> `DistMuonAdamW`  [INFERRED]
  gpt.py → optim.py
- `GPTConfig` --uses--> `MuonAdamW`  [INFERRED]
  gpt.py → optim.py
- `Linear` --uses--> `MuonAdamW`  [INFERRED]
  gpt.py → optim.py

## Communities (12 total, 3 thin omitted)

### Community 0 - "Tokenizer"
Cohesion: 0.19
Nodes (5): encode_special(), Tokenize the converstion which is called as doc or document.         Returns ids, Small helper function useful in debugging: visualize the tokenization of render_, Used during Reinforcement Learning. In that setting, we want to         render t, RustBPE_Tokenizer

### Community 1 - "Model Architecture"
Cohesion: 0.20
Nodes (4): GPTConfig, Linear, MuonAdamW, Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU vers

### Community 2 - "Flash Attention"
Cohesion: 0.24
Nodes (8): flash_attn_func(), flash_attn_kv_func(), _load_flash_attention(), Flash attention function without KV cache (for pretraining)      Args:         q, Helper function to load FS3, will have another to fallback to SDPA     on non-Ho, Flash attention function with KV cache for inference      Args:         q: Query, SDPA attention with sliding window support.     q,k,v are (B,H, T,D), _sdpa_attention_cal()

### Community 3 - "Distributed Optimizer"
Cohesion: 0.33
Nodes (4): DistMuonAdamW, Combined distributed optimizer: Muon for 2D matrix params, AdamW for others., Launch async reduce ops for AdamW group. Returns info dict with per-param infos., Launch async reduce op for Muon group. Returns info dict.

### Community 4 - "AdamW Implementation"
Cohesion: 0.33
Nodes (4): adamw_step_fused(), A nice and efficient mixed AdamW/Muon Combined Optimizer. Usually the embeddings, AdamW update for each param in the group individually.         Lazy init the sta, Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_up

### Community 5 - "Muon Implementation"
Cohesion: 0.33
Nodes (4): muon_step_fused(), Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_upd, Muon update for all params in the group (stacked for efficiency).         Lazy i, Wait for reduce, compute Muon updates, launch gather.

### Community 8 - "Optimizer Coordination"
Cohesion: 0.40
Nodes (3): Wait for reduce, compute AdamW updates, launch gathers for large params., Wait for all gathers and copy Muon params back., step()

## Knowledge Gaps
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DistMuonAdamW` connect `Distributed Optimizer` to `Optimizer Coordination`, `Model Architecture`, `AdamW Implementation`, `Muon Implementation`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `RustBPE_Tokenizer` connect `Tokenizer` to `Tokenizer Loading`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `MuonAdamW` connect `Model Architecture` to `AdamW Implementation`, `Muon Implementation`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `DistMuonAdamW` (e.g. with `GPTConfig` and `Linear`) actually correct?**
  _`DistMuonAdamW` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `MuonAdamW` (e.g. with `GPTConfig` and `Linear`) actually correct?**
  _`MuonAdamW` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Tokenize the converstion which is called as doc or document.         Returns ids`, `Small helper function useful in debugging: visualize the tokenization of render_`, `Used during Reinforcement Learning. In that setting, we want to         render t` to the rest of the system?**
  _20 weakly-connected nodes found - possible documentation gaps or missing edges._