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


def _sdpa_attention_cal(q, k, v, window_size=None, mask=None):
    return None
