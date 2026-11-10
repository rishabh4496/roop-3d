"""Bounded hand-off queues and a CUDA pinned-host frame pool.

Two things matter for keeping a GPU busy: producers must never run unboundedly ahead (VRAM / RAM exhaustion), and a
consumer must never wait on a lock held for longer than a pointer swap.  ``BoundedQueue`` therefore holds its mutex
only for O(1) ``deque`` operations; all waiting happens on condition variables, never while holding the data.  (A
literally lock-free MPMC queue is not expressible in CPython: the GIL already makes ``deque.append/popleft``
atomic, and the mutex here is uncontended in the single-producer/single-consumer layout the pipeline uses.)

``PinnedFramePool`` hands out page-locked host buffers.  ``torch.empty(..., pin_memory=True)`` is backed by
``cudaHostAlloc`` through PyTorch's caching host allocator, which is what makes ``tensor.to("cuda", non_blocking=True)``
a true asynchronous DMA that overlaps with kernels on other streams.  Without CUDA the pool degrades to ordinary
numpy buffers and says so in ``backend``.
"""
from __future__ import annotations

from collections import deque
import threading
import time
from typing import Any, Generic, TypeVar

import numpy as np

T = TypeVar("T")

DEFAULT_QUEUE_SIZE = 8   # bounded: 8 slots of 4K BGR is ~200 MB of pinned host memory, never unbounded


class PipelineCancelled(Exception):
    """Raised inside a worker when the pipeline was cancelled or another stage failed."""


class _Closed:
    """End-of-stream marker."""
    __slots__ = ()

    def __repr__(self) -> str:
        return "<END>"


END = _Closed()


class BoundedQueue(Generic[T]):
    """Single mutex, O(1) critical sections, back-pressure with cancellation, and the numbers a HUD needs."""

    def __init__(self, name: str, maxsize: int = DEFAULT_QUEUE_SIZE, *, cancel: threading.Event | None = None):
        if not 1 <= maxsize <= 64:
            raise ValueError("Queue size must be within 1..64")
        self.name, self.maxsize = name, maxsize
        self._items: deque[T] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._cancel = cancel or threading.Event()
        self._closed = False
        # telemetry (written under the lock, read racily by the HUD: single-word reads are fine)
        self.high_water = 0
        self.puts = self.gets = 0
        self.blocked_put_s = 0.0     # producer waited because the consumer was slower  -> downstream is the bottleneck
        self.starved_get_s = 0.0     # consumer waited because the producer was slower   -> upstream is the bottleneck

    def __len__(self) -> int:
        return len(self._items)

    @property
    def depth(self) -> int:
        return len(self._items)

    def resize(self, maxsize: int) -> None:
        if not 1 <= maxsize <= 64:
            raise ValueError("Queue size must be within 1..64")
        with self._lock:
            self.maxsize = maxsize
            self._not_full.notify_all()

    def wake_all(self) -> None:
        with self._lock:
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def put(self, item: T) -> None:
        began = None
        with self._lock:
            while len(self._items) >= self.maxsize and not self._cancel.is_set():
                if began is None:
                    began = time.perf_counter()
                self._not_full.wait(0.1)
            if began is not None:
                self.blocked_put_s += time.perf_counter() - began
            if self._cancel.is_set():
                raise PipelineCancelled()
            if self._closed:
                raise RuntimeError(f"Queue {self.name} is closed")
            self._items.append(item)
            self.puts += 1
            if len(self._items) > self.high_water:
                self.high_water = len(self._items)
            self._not_empty.notify()

    def get(self) -> T | _Closed:
        began = None
        with self._lock:
            while not self._items and not self._closed and not self._cancel.is_set():
                if began is None:
                    began = time.perf_counter()
                self._not_empty.wait(0.1)
            if began is not None:
                self.starved_get_s += time.perf_counter() - began
            if self._cancel.is_set():
                raise PipelineCancelled()
            if not self._items:
                return END
            item = self._items.popleft()
            self.gets += 1
            self._not_full.notify()
            return item

    def close(self) -> None:
        """No more items will be produced; consumers drain what is queued and then see ``END``."""
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()

    def stats(self) -> dict[str, Any]:
        return {"name": self.name, "depth": self.depth, "capacity": self.maxsize, "high_water": self.high_water,
                "items": self.puts, "blocked_put_s": round(self.blocked_put_s, 4),
                "starved_get_s": round(self.starved_get_s, 4)}


class PinnedFramePool:
    """A fixed set of reusable page-locked ``uint8`` frame buffers, handed out with back-pressure.

    ``acquire()`` blocks when every buffer is in flight (that is the back-pressure that bounds host memory);
    ``release(buffer)`` returns it.  The numpy view of a buffer shares memory with the pinned tensor.
    """

    def __init__(self, shape: tuple[int, ...], count: int = DEFAULT_QUEUE_SIZE + 2, *,
                 cancel: threading.Event | None = None, pin: bool = True):
        if not 1 <= count <= 64 or min(shape) < 1:
            raise ValueError("Invalid pinned pool geometry")
        self.shape, self.count = tuple(shape), count
        self._cancel = cancel or threading.Event()
        self._free: deque[int] = deque(range(count))
        self._condition = threading.Condition(threading.Lock())
        self.backend = "numpy"
        self._tensors: list[Any] = []
        self._arrays: list[np.ndarray] = []
        if pin:
            try:
                import torch
                if torch.cuda.is_available():
                    for _ in range(count):
                        tensor = torch.empty(self.shape, dtype=torch.uint8, pin_memory=True)
                        self._tensors.append(tensor)
                        self._arrays.append(tensor.numpy())
                    self.backend = "cuda_pinned"
            except (ImportError, RuntimeError, OSError):
                self._tensors, self._arrays = [], []
        if not self._arrays:
            self._arrays = [np.empty(self.shape, np.uint8) for _ in range(count)]
        self.waits = 0
        self.wait_s = 0.0          # time producers spent blocked here: back-pressure, not work

    @property
    def pinned(self) -> bool:
        return self.backend == "cuda_pinned"

    @property
    def in_flight(self) -> int:
        return self.count - len(self._free)

    @property
    def nbytes(self) -> int:
        return self.count * int(np.prod(self.shape))

    def acquire(self) -> tuple[int, np.ndarray]:
        with self._condition:
            began = None
            while not self._free:
                if self._cancel.is_set():
                    raise PipelineCancelled()
                if began is None:
                    began = time.perf_counter()
                self._condition.wait(0.1)
            if began is not None:
                self.waits += 1
                self.wait_s += time.perf_counter() - began
            if self._cancel.is_set():
                raise PipelineCancelled()
            slot = self._free.popleft()
        return slot, self._arrays[slot]

    def tensor(self, slot: int):
        """The pinned ``torch.Tensor`` behind a slot (``None`` when the pool is not pinned)."""
        return self._tensors[slot] if self._tensors else None

    def release(self, slot: int) -> None:
        # Slots are reused strictly first-in-first-out: a released buffer goes to the back of the line, so it is not
        # overwritten until ``count - 1`` other frames have been staged. Code downstream that keeps a reference to the
        # previous frame (optical-flow / scene-cut trackers) therefore never sees it change under it.
        with self._condition:
            if slot in self._free:
                raise ValueError("Pinned slot released twice")
            self._free.append(slot)
            self._condition.notify()

    def wake_all(self) -> None:
        with self._condition:
            self._condition.notify_all()


class CudaFrameLease:
    """One ring slot from asynchronous upload through its final GPU consumer."""

    def __init__(self, ring: "CudaFrameRing", slot: int):
        self.ring, self.slot = ring, slot
        self.tensor = None
        self.ready_event = None
        self._owner = None
        self._released = False

    def stage_host(self, frame: np.ndarray):
        """Copy a decoded HWC uint8 frame from the slot's pinned page into CUDA."""
        if self._released or frame.dtype != np.uint8 or tuple(frame.shape) != self.ring.shape:
            raise ValueError("Host frame does not match the CUDA ring's HWC uint8 contract")
        slot = self.ring._slots[self.slot]
        np.copyto(slot["host"].numpy(), frame)
        torch = self.ring.torch
        with torch.cuda.stream(self.ring.stream):
            slot["device"].copy_(slot["host"], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.ring.stream)
        self.tensor, self.ready_event = slot["device"], event
        return self.tensor

    def stage_cuda(self, frame, *, producer_stream=None, copy: bool = True):
        """Stage a decoded CUDA surface; ``copy=False`` borrows its DLPack tensor in-place."""
        torch = self.ring.torch
        if (self._released or not torch.is_tensor(frame) or not frame.is_cuda
                or frame.dtype != torch.uint8 or tuple(frame.shape) != self.ring.shape
                or frame.device != self.ring.device or not frame.is_contiguous()):
            raise ValueError("CUDA frame does not match the ring's contiguous HWC uint8 device contract")
        producer_stream = producer_stream or torch.cuda.current_stream(self.ring.device)
        if copy:
            with torch.cuda.stream(self.ring.stream):
                self.ring.stream.wait_stream(producer_stream)
                slot = self.ring._slots[self.slot]
                slot["device"].copy_(frame, non_blocking=True)
                event = torch.cuda.Event()
                event.record(self.ring.stream)
            self.tensor = slot["device"]
            self._owner = frame
        else:
            event = torch.cuda.Event()
            event.record(producer_stream)
            self.tensor = frame
            self._owner = frame
        self.ready_event = event
        return self.tensor

    def wait_on(self, stream) -> None:
        if self.ready_event is not None:
            stream.wait_event(self.ready_event)

    def keepalive(self, *owners) -> None:
        self._owner = (self._owner, *owners)

    def release_after(self, event) -> None:
        """Return the slot after a downstream CUDA event has finished using its pixels."""
        if self._released:
            raise ValueError("CUDA frame lease released twice")
        self._released = True
        self.ring._release(self.slot, event, self._owner)
        self.tensor = self._owner = None

    def release(self, stream=None) -> None:
        if self._released:
            raise ValueError("CUDA frame lease released twice")
        torch = self.ring.torch
        event = torch.cuda.Event()
        event.record(stream or torch.cuda.current_stream(self.ring.device))
        self.release_after(event)


class CudaFrameRing:
    """Bounded pinned-host/CUDA ring with stream-ordered upload and reuse.

    Native CUDA decoder surfaces can be borrowed with ``copy=False``; software
    decoders stage through pinned host slots and enqueue one asynchronous H2D
    transfer. The existing NumPy pipeline remains the no-CUDA fallback.
    """

    def __init__(self, shape: tuple[int, ...], count: int = 4, *, device: int = 0,
                 cancel: threading.Event | None = None):
        if len(shape) != 3 or shape[2] != 3 or min(shape) < 1 or not 2 <= count <= 32:
            raise ValueError("CUDA frame ring expects bounded HWC RGB frames and 2..32 slots")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA frame ring requires an available CUDA device")
        self.torch, self.device = torch, torch.device("cuda", device)
        self.shape, self.count = tuple(shape), count
        self._cancel = cancel or threading.Event()
        self._condition = threading.Condition(threading.Lock())
        self._free: deque[int] = deque(range(count))
        self._leased: set[int] = set()
        self._retired: dict[int, Any] = {}
        self.waits = 0
        self.wait_s = 0.0
        self.stream = torch.cuda.Stream(device=self.device, priority=0)
        self._slots = []
        try:
            with torch.cuda.device(self.device):
                for _ in range(count):
                    host = torch.empty(self.shape, dtype=torch.uint8, pin_memory=True)
                    device_tensor = torch.empty(self.shape, dtype=torch.uint8, device=self.device)
                    self._slots.append({"host": host, "device": device_tensor})
        except BaseException:
            self._slots.clear()
            raise
        self.backend = "cuda_pinned_ring"

    @property
    def pinned(self) -> bool:
        return True

    @property
    def in_flight(self) -> int:
        with self._condition:
            self._reclaim_ready()
            return len(self._leased) + len(self._retired)

    @property
    def nbytes(self) -> int:
        return self.count * int(np.prod(self.shape)) * 2

    def _reclaim_ready(self) -> None:
        for slot, (event, owner) in list(self._retired.items()):
            if event.query():
                del self._retired[slot]
                self._free.append(slot)
        if self._retired:
            self._condition.notify_all()

    def acquire(self) -> CudaFrameLease:
        with self._condition:
            began = None
            while True:
                self._reclaim_ready()
                if self._cancel.is_set():
                    raise PipelineCancelled()
                if self._free:
                    if began is not None:
                        self.waits += 1
                        self.wait_s += time.perf_counter() - began
                    slot = self._free.popleft()
                    self._leased.add(slot)
                    return CudaFrameLease(self, slot)
                if began is None:
                    began = time.perf_counter()
                self._condition.wait(0.005)

    def _release(self, slot: int, event, owner=None) -> None:
        with self._condition:
            if slot not in self._leased:
                raise ValueError("CUDA ring slot is not leased")
            self._leased.remove(slot)
            self._retired[slot] = (event, owner)
            self._reclaim_ready()
            self._condition.notify_all()

    def wake_all(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._leased:
                raise RuntimeError("Cannot close the CUDA frame ring while slots are leased")
            events = tuple(event for event, _owner in self._retired.values())
        for event in events:
            event.synchronize()
        self.stream.synchronize()
        with self._condition:
            self._retired.clear()
            self._free = deque(range(self.count))
            self._slots.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
