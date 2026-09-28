"""Private random streams; NumPy is required and PyTorch is optional.

Import PyTorch and initialize any required CUDA context before constructing a
PyTorch policy adapter. This module never imports Torch or initializes CUDA.
A first import or CUDA initialization inside inference is rejected.
"""
from __future__ import annotations

from contextlib import contextmanager
import random
import sys

import numpy as np


def random_state():
    torch = sys.modules.get("torch")
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state() if torch is not None else None,
            "cuda": torch.cuda.get_rng_state_all()
            if torch is not None and torch.cuda.is_initialized() else []}


def restore_random_state(state):
    torch = sys.modules.get("torch")
    if state.get("cuda") and (torch is None or not torch.cuda.is_initialized()):
        raise ValueError("CUDA RNG restoration requires an already initialized context")
    if state.get("torch") is not None and torch is None:
        raise ValueError("Import PyTorch before restoring a PyTorch policy stream")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch = sys.modules.get("torch")
    if torch is not None:
        # torch.manual_seed would also queue CUDA initialization.
        torch.default_generator.manual_seed(seed)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(seed)


class _TorchImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch":
            raise RuntimeError("Import PyTorch before constructing PolicyAdapter; lazy framework imports inside policy inference are unsupported")
        return None


@contextmanager
def policy_import_guard():
    """Reject first-time runtime loading before it can alter private streams.

PyTorch CUDA operations go through its lazy initializer. Temporarily rejecting
that entry point keeps CPU inference from creating an untracked GPU RNG stream.
Policy adapters are serial local inference contexts, not thread-safe sandboxes.
"""
    guard = _TorchImportGuard() if "torch" not in sys.modules else None
    if guard is not None:
        sys.meta_path.insert(0, guard)
    torch = sys.modules.get("torch")
    cuda = torch.cuda if torch is not None else None
    initializer = None
    if cuda is not None and not cuda.is_initialized():
        initializer = getattr(cuda, "_lazy_init", None)
        if initializer is not None:
            def reject_cuda_initialization(*args, **kwargs):
                raise RuntimeError("Initialize CUDA before constructing PolicyAdapter; lazy CUDA initialization inside a policy is unsupported")
            cuda._lazy_init = reject_cuda_initialization
    try:
        yield
    finally:
        if initializer is not None:
            cuda._lazy_init = initializer
        if guard is not None:
            sys.meta_path.remove(guard)
