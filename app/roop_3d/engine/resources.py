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

class SharedExclusiveFence:
    """Re-entrant shared/exclusive lock: many inferences at once, builds and releases alone.

    The previous single ``RLock`` serialised every ONNX call in the process, so a detector, the swapper and the
    restorer could never run at the same time even on separate threads or CUDA streams. Inference now takes the
    *shared* side; model construction, release, ``evict_for_build`` and TensorRT builds still take the *exclusive*
    side and therefore still wait for every in-flight inference (and block new ones) exactly as before.

    Re-entrancy matches ``RLock``: a thread may nest shared in shared, anything inside exclusive, and exclusive in
    exclusive. A thread that holds the shared side and needs the exclusive side (a lazy model build inside a frame
    inference) is *upgraded*: its shared holds are parked, it waits for the other readers to drain, runs exclusively,
    and gets its shared holds back on release. Nothing ever waits while holding the shared side, so two upgrading
    threads cannot deadlock each other.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._readers: dict[int, int] = {}      # thread id -> shared depth
        self._writer: int | None = None
        self._writer_depth = 0
        self._parked: dict[int, int] = {}       # writer thread id -> shared depth parked during an upgrade
        self._writers_waiting = 0

    # -- shared side -------------------------------------------------------------------------------------
    def acquire_shared(self) -> None:
        me = threading.get_ident()
        with self._condition:
            if self._writer == me or me in self._readers:   # re-entrant: never wait on ourselves
                self._readers[me] = self._readers.get(me, 0) + 1
                return
            while self._writer is not None or self._writers_waiting:
                self._condition.wait()
            self._readers[me] = 1

    def release_shared(self) -> None:
        me = threading.get_ident()
        with self._condition:
            depth = self._readers[me] - 1
            if depth:
                self._readers[me] = depth
            else:
                del self._readers[me]
                self._condition.notify_all()

    # -- exclusive side ----------------------------------------------------------------------------------
    def acquire_exclusive(self) -> None:
        me = threading.get_ident()
        with self._condition:
            if self._writer == me:
                self._writer_depth += 1
                return
            parked = self._readers.pop(me, 0)               # upgrade: park our own shared holds
            self._writers_waiting += 1
            try:
                while self._writer is not None or self._readers:
                    self._condition.wait()
            finally:
                self._writers_waiting -= 1
            self._writer, self._writer_depth = me, 1
            if parked:
                self._parked[me] = parked

    def release_exclusive(self) -> None:
        me = threading.get_ident()
        with self._condition:
            self._writer_depth -= 1
            if self._writer_depth:
                return
            self._writer = None
            parked = self._parked.pop(me, 0)
            if parked:
                self._readers[me] = self._readers.get(me, 0) + parked
            self._condition.notify_all()

    @contextmanager
    def shared(self) -> Iterator[None]:
        self.acquire_shared()
        try:
            yield
        finally:
            self.release_shared()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        self.acquire_exclusive()
        try:
            yield
        finally:
            self.release_exclusive()

    # An exclusive holder is also entitled to nest shared inference (a build that runs a canary): see acquire_shared.
    __enter__ = acquire_exclusive

    def __exit__(self, *_exc) -> None:
        self.release_exclusive()

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {"shared_holders": len(self._readers), "exclusive": int(self._writer is not None),
                    "writers_waiting": self._writers_waiting}


_fence = SharedExclusiveFence()
_owners: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_instance_locks_guard = threading.Lock()


def register(owner: object, release_method: str) -> None:
    with _fence:
        _owners[owner] = release_method


def guarded(function):
    """Exclusive: construction, release and eviction. Waits for all in-flight inference."""
    @wraps(function)
    def invoke(*args, **kwargs):
        with _fence:
            return function(*args, **kwargs)
    return invoke


def inference(function):
    """Shared: per-frame inference. Concurrent with other models, serialised only against builds/evictions.

    A model instance owns mutable I/O-binding state, so calls on the *same* instance still run one at a time
    (the instance lock is created lazily on first use); different instances proceed in parallel.
    """
    @wraps(function)
    def invoke(self, *args, **kwargs):
        lock = self.__dict__.get("_inference_lock")
        if lock is None:
            with _instance_locks_guard:
                lock = self.__dict__.setdefault("_inference_lock", threading.RLock())
        # Instance lock first: a thread waiting for a busy instance must not hold the shared side, or a
        # concurrent exclusive upgrade (a lazy build inside the lock holder) could never drain the readers.
        with lock, _fence.shared():
            return function(self, *args, **kwargs)
    return invoke


def fence_snapshot() -> dict[str, int]:
    return _fence.snapshot()


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
