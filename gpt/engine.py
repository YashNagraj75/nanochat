import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt.common import compute_init, autodetect_device_type
from gpt.checkpoint_manager import s
