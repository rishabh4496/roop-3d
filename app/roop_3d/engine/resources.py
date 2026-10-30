"""Process-local inference/build fence and explicit ownership of GPU resources."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import gc
import sys
import threading
from typing import Any, Iterator
import weakref

from roop_3d.core.errors import DomainError

_fence = threading.RLock()
_owners: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def register(owner: object, release_method: str) -> None:
    with _fence:
        _owners[owner] = release_method


def guarded(function):
    @wraps(function)
    def invoke(*args, **kwargs):
        with _fence:
            return function(*args, **kwargs)
    return invoke


@contextmanager
def build_fence() -> Iterator[None]:
    with _fence:
        yield


def evict_for_build() -> dict[str, Any]:
    """Close all owned sessions, collect, empty any loaded torch CUDA allocator.

    Unmanaged ORT aliases or live torch tensors halt compilation, never disappear
    from accounting by pretending empty_cache can free a live allocation.
    """
    with _fence:
        owners = list(_owners.items())
        for owner, method in owners:
            getattr(owner, method)()
        del owners
        gc.collect()
        torch = sys.modules.get("torch")
        torch_state = "NOT_LOADED"
        if torch is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch_state = "CALLED"
                if torch.cuda.memory_allocated() != 0:
                    raise DomainError("BUILD_LIVE_TORCH_ALLOCATIONS", "Live CUDA tensors remain after session eviction", 409)
            else:
                torch_state = "CUDA_UNAVAILABLE"
        # The application never imports torch just to create a new CUDA context.
        # No loaded torch module means no process-local torch CUDA allocator.
        import onnxruntime as ort
        remaining = sum(isinstance(obj, ort.InferenceSession) and getattr(obj, "_sess", None) is not None
                        for obj in gc.get_objects())
        if remaining:
            raise DomainError("BUILD_LIVE_ORT_SESSIONS", "An ORT session is still referenced after eviction; compilation refused", 409)
        return {"registered_owners_released": len(_owners), "gc_collected": True,
                "torch_cuda_empty_cache": torch_state, "remaining_ort_sessions": remaining}


def register_torch_module(module: object) -> object:
    """Optional adapter hook: caller must retain this small owner until release."""
    class TorchOwner:
        def __init__(self, value):
            self.module = value
        def release(self):
            self.module.to("cpu")
    owner = TorchOwner(module)
    register(owner, "release")
    return owner
