import torch
import torch.nn.functional as F
from kernels import get_kernel
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt.common import _DTYPE_MAP, COMPUTE_DTYPE


def _load_flash_attention():
    """
    Helper function to load FS3, will have another to fallback to SDPA
    on non-Hopper architecture but for now its None
    """
    if not torch.cuda.is_available():
        return None

    try:
        arch = torch.cuda.get_device_capability()
        if arch[0] != 9:
            return None
        return get_kernel("varunneal/flash-attention-3").flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention()
HAS_FA3 = _fa3 is not None

_override_imp = None  # Override for testing, fa3, sdpa, None


def _use_fa3():
    if HAS_FA3:
        if COMPUTE_DTYPE == "bfloat16":
            return True
    if _override_imp == "sdpa":
        return False
    if _override_imp == "fa3":
        assert HAS_FA3, "FA3 is not available, cannot override to FA3"
        return True


USE_FA3 = _use_fa3()
print(f"Using Flash Attention 3: {USE_FA3}")


def _sdpa_attention_cal(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q,k,v are (B,H, T,D)
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    if (window < 0 or window >= Tq) and Tq == Tk:
        """
        This is the prefill phase as to when the model uses all the input query and key values to 
        build the initial KV cache, so suppose a user gives a para of 500 words the model attends to 
        all the 500 words at the same time to build the initial KV cache, so the is_causal flag is set to True
        as the output tokens during the decoding phase might cheat and predict the future. enable_gqa is just 
        a feature
        """
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=enable_gqa
        )

    if Tq == 1:
        """
        This is the decoding phase where the model produces the output tokens during any stage of training.
        So here the model produces one token at a time, so it compares the query vector to all the previous key 
        vectors in order to calculate attn, so the is_causal flag is set to False here as a model is predicting 
        at this stage and not  learning
        """
        if window >= 0 and window < Tk:
            # Sliding window's window calculation
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=False, enable_gqa=enable_gqa
            )

    # Need explicit masking for sliding window/chunk inference
    device = q.device
    """
    So for chunk/window inference we need an explicit mask matrix as (Tq != Tk) so the cache position will not 
    match with the new tokens so the is_causal mask is off because it will apply the mask incorrectly so
    we need to create an explicit mask matrix here to make sure the model attends to the correct tokens in the cache
     and not the future tokens which will be the case if we use is_causal mask
    """
    q_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(0)
    """
    So here we are calculating the row_idx of the qin coming query vector so if Tk = 100 and incominh query is of length Tq =5 then the position of the queries will be 95, 96, 97,98, 99 so (100 - 5) + [0,1,2,3,4] gives you that
    """
    k_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = (
        k_idx <= q_idx
    )  # this is a check cus query has to attend to key tokens behind it so the position of k will be lesser than q

    if window >= 0 and window < Tk:
        mask = mask & ((q_idx - k_idx) <= window)

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=~mask, enable_gqa=enable_gqa
    )


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash attention function without KV cache (for pretraining)

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback for non-Hopper GPU's
    q = q.transpose(1, 2)  # (B, H, T, D)
    k = k.transpose(1, 2)  # (B, H, T, D)
    v = v.transpose(1, 2)  # (B, H, T, D)
    enable_gqa = q.size(1) != k.size(1)  # GQA if num_heads in q and k are different
    return _sdpa_attention_cal(q, k, v, window_size, enable_gqa=enable_gqa).transpose(
        1, 2
    )  # (B, T, H, D)


def flash_attn_kv_func(
    q, k_cache, v_cache, k, v, causal=False, window_size=(-1, -1), cache_seqlens=None
):
    """
    Flash attention function with KV cache for inference

    Args:
        q: Query tensor of shape (B, T_new, H, D)
        k_cache, v_cache: Cached key and value tensors of shape (B, T_max, H_kv, D)
        k, v: Current key and value tensors of shape (B, 1, H, D)
        cache_seqlens: Current position in the cache (B,)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=causal,
            window_size=window_size,
        )

    # SDPA fallback : manually manage the KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()
    if k is not None and v is not None:
        k_cache[:, pos : pos + T_new, :, :] = k
        v_cache[:, pos : pos + T_new, :, :] = v

    # Now retreive k and v from the current_pos + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Now tranpose to SDPA layout (B,T,H,D) -> (B,H,T,D)
    q_sdpa = q.tranpose(1, 2)
    k_sdpa = k_full.tranpose(1, 2)
    v_sdpa = v_full.tranpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention_cal(
        q_sdpa, k_sdpa, v_sdpa, window_size=window_size, enable_gqa=enable_gqa
    )
    return y_sdpa.transpose(1, 2)


from types import SimpleNamespace

flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func, flash_attn_kv_func=flash_attn_kv_func
)
