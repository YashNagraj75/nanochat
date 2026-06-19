from logging import BufferingFormatter
from pickle import BINFLOAT
import torch
from torch.cuda import temperature
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt.common import compute_init, autodetect_device_type
from gpt.checkpoint_manager import load_model


@contextmanager  # Calculator tool helpers
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"{formula}: timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        return None


def use_calculator(expr):
    """
    Evaluate a python expression safely
    Support both Math expressions and string operations like .count()
    """
    expr = expr.replace(",", "")

    if all([x in "0123456789+-*/(). " for x in expr]):
        if "**" in expr:
            return None
        return eval_with_timeout(expr)

    # Support string operations like "hello world".count("o")
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ._()\"'"
    )
    if not all(c in allowed_chars for c in expr):
        return None

        # Disallow dangerous patterns
    dangerous_patterns = [
        "__",
        "import",
        "exec",
        "eval",
        "compile",
        "open",
        "file",
        "input",
        "raw_input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    ]
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if ".count(" not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)


class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(
        self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype
    ) -> None:
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.n_layers = num_layers

        # Initialise the kv_cache, before hand (n_layers, B,T,H,D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim)

        # Current seq length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous tokens normalized embedding for smear gate
        self.prev_embedding = None

    def reset(self):
        "Reset the cache"
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        "Get the position of the current position as its the same Batch -> this is during decoding"
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        "Return layer values from the cache"
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        "Advance the cache by num_tokens"
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel
        """
        assert self.get_pos() == 0, "Cannot prefill non-empty KV cache"
        assert (
            self.n_layers == other.n_layers
            and self.n_heads == other.n_heads
            and self.head_dim == other.head_dim
        )
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(
                self.batch_size, -1, -1
            ).clone()


@torch.inference_mode()
def sample_next_token(logits, rng, temp=1.0, top_k=None):
    "Sample single next token given logits of shape (B, vocab_size). Returns (B,1)"
    assert temp >= 0.0, "temperature must be non-negative"
    if temp == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temp
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)  # Sampling one token
    else:
        logits = logits / temp
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


class RowState:
    """Per-row state for one sequence in a generation batch (B, T) matrix.

    Each batch row is an independent sequence being generated in parallel.
    This object tracks everything that differs between rows so the decode
    loop can handle each one independently:

      current_tokens    – full token sequence so far (prompt + generated)
      forced_tokens     – FIFO queue of tokens to inject without sampling;
                          used to feed back python eval results or template tokens
      in_python_block   – True while collecting tokens inside a <python>…</python>
                          block; tokens are buffered rather than emitted
      python_expr_tokens– tokens accumulated inside the current python block,
                          decoded and eval()'d when the closing tag is detected
      completed         – True once this row has hit EOS or max_tokens;
                          the decode loop skips it while other rows continue
    """

    def __init__(self, current_tokens=None) -> None:
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False


class Engine:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        tokens,
        max_tokens=None,
        num_samples=1,
        temperature=1.0,
        top_k=None,
        seed=42,
    ):
        """
        Same as generate but then does a single prefill and then clones the KV cache
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), (
            "expecting a list of ints for tokens"
        )
        device = self.model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        get_special = lambda s: self.tokenizer.encode_special(s)
