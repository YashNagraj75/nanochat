# nanochat Architecture Notes

Answers to questions that came up while building the GPT from scratch.

---

## 1. Rotary Position Embeddings (RoPE)

**What it does:** Encodes position by rotating pairs of dimensions in Q and K vectors, so attention scores naturally reflect relative distance between tokens.

**The math:** For a pair `(x1, x2)` at position `m`:
```
new_x1 = x1·cos - x2·sin
new_x2 = x2·cos + x1·sin
```
Compactly: `x * cos + rotate_half(x) * sin`, where `rotate_half(x) = [-x2, x1]`.

**The code:**
```python
def apply_rotary_emb(x, cos, sin):   # x: (B, T, H, D)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    x_rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rotated * sin
```

**Precomputing cos/sin (done once, registered as buffer):**
```python
freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
t = torch.arange(seq_len)
freqs = torch.outer(t, freqs)                          # (T, D//2)
cos = torch.cos(freqs).repeat_interleave(2, dim=-1)    # (T, D)
sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
# reshape to (1, T, 1, D) for broadcasting over (B, T, H, D)
```

---

## 2. Why `head_dim = n_embd // n_head`, not `n_embd // n_kv_head`

`head_dim` is the size of the vector **each individual head** works with. That split is always over `n_head` (the number of query heads):

```
n_embd = n_head × head_dim
→ head_dim = n_embd // n_head     e.g. 768 // 12 = 64
```

In **GQA (Grouped Query Attention)**, KV heads are fewer (e.g. 4) but still use the same `head_dim = 64`. Each KV head is shared by `n_head // n_kv_head = 3` query heads.

Using `n_kv_head` would give a *larger* head_dim and break all projection shapes.

```
Q: (B, T, 12, 64)   ← n_head    query heads
K: (B, T,  4, 64)   ← n_kv_head key heads, same head_dim
V: (B, T,  4, 64)
```

---

## 3. Full Attention Data Flow

```
Input x: (B, T, n_embd)
    │
    ├─ q_proj → Q: (B, T, n_head,    head_dim)
    ├─ k_proj → K: (B, T, n_kv_head, head_dim)
    └─ v_proj → V: (B, T, n_kv_head, head_dim)
            │
            [optional: inject raw token embedding into V via gate — see §4]
            │
            apply RoPE to Q and K, then normalize and scale
            │
            flash_attn(Q, K, V) → y: (B, T, n_head, head_dim)
            │
            reshape → (B, T, n_embd)
            │
    o_proj → output: (B, T, n_embd)    ← caller adds this to the residual stream
```

Two separate linear layers:
- `q/k/v_proj` — compress input down into per-head vectors
- `o_proj` — project attention output back up to `n_embd` for the residual stream

---

## 4. Value Embeddings (VE)

**Problem:** In deep transformers, a token's original identity gets washed out as the residual stream is rewritten layer by layer.

**Solution:** On alternating layers, inject the **raw token embedding** (straight from the embedding table, before any transformer processing) directly into the value vectors via a learned gate:

```python
gate = 3 * sigmoid(W_gate(x[..., :12]))   # (B, T, n_kv_head), range (0, 3)
v    = v + gate.unsqueeze(-1) * ve
```

- **Gate range (0, 3):** the 3× ceiling lets `ve` *dominate* `v` if the model learns to — not just blend, but fully override.
- **Only first 12 dims of x** used for the gate — cheap, avoids a full 768-dim projection just to compute a scalar per head.
- **Every other layer** (`has_ve`): a compromise between effectiveness and compute cost.

Result: `v` carries both *"what this position means in context"* (from `v_proj`) and *"what this token literally is"* (from `ve`), weighted by the gate.

---

## 5. Smear Gate — `Linear(24, 1)`

**What it does:** Mixes the **previous token's embedding** into the current token — cheap bigram-level information before attention even runs.

```python
gate   = sigmoid(W_smear(x[..., :24]))   # (B, T, 1) scalar per token
x_prev = F.pad(x[:, :-1], (0, 0, 1, 0)) # shift x right by 1 position
x      = x + gate * smear_lambda.exp() * x_prev
```

**Why `Linear(24, 1)`:**
- Input `24` — uses only the first 24 channels (cheap; the gate doesn't need full context)
- Output `1` — a single scalar: *"how much of the previous token to blend in?"*

**`smear_lambda` initialized to 0:** model starts with no smearing; opts in gradually as training proceeds. Same pattern as `backout_lambda`.

---

## 6. Backout Lambda

**What it does:** Subtracts a fraction of a **mid-layer residual** from the final residual, just before the LM head.

```python
x_final = x_final - backout_lambda * x_mid_layer
```

**Why:** By the final layer, the residual stream mixes low-level features (raw token identity, local syntax) with high-level features (abstract semantics, next-token predictions). The LM head has to "look past" the low-level noise.

Backout removes the low-level residue so the LM head sees a cleaner, more abstract representation.

**Analogy:** Like contrast enhancement in image processing — subtract the "blurry low-frequency layer" to make the sharp high-frequency detail stand out.

**`backout_lambda = 0.2`** (not zero): authors already know it helps, so it starts at a working value rather than opting in from zero.

---

## 7. Weight Initialization — Uniform vs Normal, and the √3 Multiplier

**The `√3` multiplier** exists to make `Uniform` achieve the same standard deviation as `Normal`:

| Distribution | std |
|---|---|
| `N(0, σ²)` | σ |
| `U(-a, a)` | `a / √3` |

To match: `a / √3 = σ  →  a = √3 · σ`

```python
σ = n_embd ** -0.5           # target std
a = 3**0.5 * n_embd**-0.5   # uniform bound = √3 · σ
torch.nn.init.uniform_(w, -a, a)
```

**Why use Uniform for projections instead of Normal:**
- Normal can produce rare large outliers (3σ, 4σ). With bfloat16, these cause **numerical instability** in attention scores early in training.
- Uniform hard-caps at `±a`, eliminating outliers while keeping the same variance.
- `wte` and `lm_head` use Normal because: `wte` is looked up (not chained through multiplications), and `lm_head` uses std=0.001 (tails are negligible).

**Projection outputs initialized to zero:**
```
attn.o_proj → zeros
mlp.c_proj  → zeros
```
Means each block contributes nothing at init — the model starts as an identity function and gradually "turns on" each block. Stable training from step 1.

---

## 8. Residual Lambdas — the 1.15 → 1.05 Schedule

Instead of `x = x + block(x)`, the model uses:
```python
x = resid_lambda[i] * x + block(x)
```

Initialized as a linear decay:
```python
resid_lambdas[i] = 1.15 - (0.10 * i / (n_layer - 1))
# Layer 0 (first): 1.15
# Layer N (last):  1.05
```

**Why λ > 1.0:** Amplifies the existing residual stream before adding the block's contribution. Keeps signal strength through deep networks and prevents vanishing gradients.

**Why decay from 1.15 → 1.05:**
- Early layers transform raw embeddings (low information → needs more amplification)
- Later layers refine rich representations (less amplification needed)

**Why these specific numbers:** The *shape* (linear decay, slightly above 1.0) is theory-motivated. The exact values (1.15, 0.10 range) are empirically tuned — tried values in this ballpark and these gave the best training loss.

---

## 9. Attention FLOP Count — why `12 * n_embd * q * effective_seq_len`

The `12` is the product of three independent factors:

### Factor 1: Basic matmul cost = 2
Any matrix multiply `(m, k) @ (k, n)` costs `2 * m * k * n` FLOPs.
The `2` is one multiply + one add per output element.

For `Q @ K^T` on one head:
```
Q: (T_q, head_dim)  @  K^T: (head_dim, T_k)
FLOPs = 2 * T_q * head_dim * T_k
```

### Factor 2: Two matmuls in attention = ×2
Both have the same shape cost:
```
1. Q @ K^T     → scores:  (T_q, T_k)
2. scores @ V  → output:  (T_q, head_dim)
```
Forward-only total = `4 * head_dim * T_q * T_k` per head.

### Factor 3: Training = ×3
For each matmul `C = A @ B`, the backward pass needs:
```
Forward:    C  = A @ B        (compute output)
Backward:   dA = dC @ B^T    (grad w.r.t. A)
Backward:   dB = A^T @ dC    (grad w.r.t. B)
```
Three passes, each the same cost → **3× the forward**.

### Putting it together
```
2  (multiply-add)
× 2  (two matmuls: Q@K^T and scores@V)
× 3  (training: forward + 2 backward passes)
= 12

→ 12 * head_dim * T_q * T_k  per head
→ 12 * n_embd  * T_q * T_k  all heads  (n_head × head_dim = n_embd)
```

### Why "effective" seq len?
In causal attention, token `t` attends only to positions `0..t`. On average each query sees `T/2` keys, so `effective_seq_len ≈ T/2` for a full causal sequence.