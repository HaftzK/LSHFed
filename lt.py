# lt.py
from typing import Callable, Dict, Any, Tuple
import time
import torch

class LocalTrainer:
    def __init__(self, node_id: int, is_dn: bool = True):
        self.node_id = node_id
        self.is_dn = is_dn

    # pass trained tensor to compute_fn
    # add privacy noise before passing
    @torch.no_grad()
    def run_local_round(self, compute_fn: Callable[[], torch.Tensor]) -> Tuple[torch.Tensor, float]:
        t0 = time.time()
        update = compute_fn()
        elapsed = time.time() - t0
        if not isinstance(update, torch.Tensor):
            raise TypeError("compute_fn must return torch.Tensor")
        return update, elapsed
