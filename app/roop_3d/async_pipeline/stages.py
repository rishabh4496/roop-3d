"""Asynchronous multi-stage worker graph with per-stage CUDA streams and event hand-off.

    source -> [stage 1] -> [stage 2] -> ... -> [stage N] -> results()
              thread+stream  thread+stream      thread+stream

Every stage owns one thread and, optionally, one non-blocking ``torch.cuda.Stream``.  Stages are joined by
``BoundedQueue`` objects, so frame ``N+2`` can be in stage 1 while ``N+1`` is in stage 2 and ``N`` is in stage 3.

Hand-off between CUDA streams never uses ``cudaDeviceSynchronize``: the producing stage records a ``cudaEvent`` on its
stream after the last enqueued operation and the consuming stage's stream does ``cudaStreamWaitEvent`` on it, so the
*host* thread moves on immediately and the GPU orders the dependent work.  A stage without a stream (a CPU/IO stage)
synchronises on just that one event, not the device.  Tensors that cross streams are ``record_stream``-ed so the
caching allocator cannot recycle them while the second stream still reads them.

Ordering: one worker per stage and FIFO queues mean frames leave in the order they entered, which the temporal
trackers downstream of detection rely on.
"""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
import itertools
import logging
import threading
import time
from typing import Any, Callable, Iterable, Iterator
import weakref

from roop_3d.core.transfer_ledger import TRANSFERS
from roop_3d.async_pipeline.buffers import BoundedQueue, DEFAULT_QUEUE_SIZE, END, PipelineCancelled


logger = logging.getLogger(__name__)


def _torch():
    try:
        import torch
    except ImportError:  # pragma: no cover - the app always ships torch
        return None
    return torch if torch.cuda.is_available() else None


@dataclass
class Item:
    """One frame travelling through the graph."""
    index: int
    payload: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
    event: Any = None                                    # CUDA event recorded on the producing stream
    entered: float = field(default_factory=time.perf_counter)
    wall_ms: dict[str, float] = field(default_factory=dict)
    cleanups: list[Callable[[], None]] = field(default_factory=list)
    _release_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def release(self) -> None:
        """Run (once) the callbacks that return pooled buffers; the consumer of the final stage calls this."""
        with self._release_lock:
            callbacks, self.cleanups = self.cleanups, []
        first_error = None
        for callback in callbacks:
            try:
                callback()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def adopt(payload: Any, stream) -> None:
    """``record_stream`` every CUDA tensor in ``payload`` for ``stream`` (allocator safety across streams)."""
    torch = _torch()
    if torch is None or stream is None:
        return
    stack = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                value.record_stream(stream)
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, dict):
            stack.extend(value.values())


class StageStats:
    """Thread-safe accumulators for one stage."""

    def __init__(self) -> None:
        self.items = 0
        self.busy_s = 0.0
        self.cuda_ms = 0.0
        self.cuda_items = 0
        self.intervals: deque[tuple[float, float]] = deque(maxlen=8192)   # host wall (start, end)
        self.gpu_windows: deque[tuple[float, float]] = deque(maxlen=8192)  # ms on the GPU timeline
        self.lanes: deque[tuple[int, float, float]] = deque(maxlen=512)    # (frame index, host start, host end): the flamegraph lanes
        self.last_ms = 0.0


class Stage:
    def __init__(self, name: str, function: Callable[[Item], Any], *, stream: bool = False, priority: int = 0):
        self.name, self.function, self.wants_stream, self.priority = name, function, stream, priority
        self.stream = None
        self.stats = StageStats()
        self.queue_in: BoundedQueue | None = None
        self.queue_out: BoundedQueue | None = None
        self._pending: deque[tuple[Any, Any]] = deque()
        self._drain_lock = threading.Lock()
        self.thread: threading.Thread | None = None

    # -- CUDA timing, resolved without ever blocking the host ------------------------------------------------
    def _drain_cuda_timings(self, base) -> None:
        # The stage thread and the telemetry endpoint both call this; whoever is second simply skips.
        if not self._drain_lock.acquire(blocking=False):
            return
        try:
            while self._pending:
                start, end = self._pending[0]
                if not end.query():
                    return
                self._pending.popleft()
                self.stats.cuda_ms += start.elapsed_time(end)
                self.stats.cuda_items += 1
                if base is not None:
                    self.stats.gpu_windows.append((base.elapsed_time(start), base.elapsed_time(end)))
        finally:
            self._drain_lock.release()


class AsyncPipeline:
    """Build with ``add_stage`` then iterate ``results()``; use as a context manager so threads always stop."""

    _ids = itertools.count(1)
    registry: "weakref.WeakValueDictionary[str, AsyncPipeline]" = weakref.WeakValueDictionary()

    def __init__(self, name: str, source: Iterable[Any] | None, *, queue_size: int = DEFAULT_QUEUE_SIZE,
                 source_name: str = "decode", device: int = 0, priority_streams: bool = False):
        self.name = name
        self.pipeline_id = f"{name}-{next(self._ids)}"
        self._source, self.source_name = source, source_name
        self.queue_size, self.device, self.priority_streams = queue_size, device, priority_streams
        self.cancel_event = threading.Event()
        self._stages: list[Stage] = []
        self._queues: list[BoundedQueue] = []
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._pending_cleanup: dict[int, Item] = {}
        self._threads: list[threading.Thread] = []
        self._started = False
        self._begun = self._finished = 0.0
        self._base_event = None
        self.source_stats = StageStats()
        self.frames_out = 0
        self.pools: list[Any] = []

    # -- construction ---------------------------------------------------------------------------------------
    def add_stage(self, name: str, function: Callable[[Item], Any], *, stream: bool = False) -> "AsyncPipeline":
        if self._started:
            raise RuntimeError("Stages must be added before the pipeline starts")
        if any(stage.name == name for stage in self._stages) or name == self.source_name:
            raise ValueError(f"Duplicate stage name: {name}")
        self._stages.append(Stage(name, function, stream=stream))
        return self

    def set_source(self, source: Iterable[Any]) -> "AsyncPipeline":
        """Late-bind the source (it may need the pipeline's cancel event, e.g. for a pinned-buffer pool)."""
        if self._started:
            raise RuntimeError("The source must be set before the pipeline starts")
        self._source = source
        return self

    def add_pool(self, pool: Any) -> None:
        """Register a ``PinnedFramePool`` so its occupancy is reported and its waiters wake on cancel."""
        self.pools.append(pool)

    def set_queue_capacity(self, capacity: int) -> None:
        """Adapt every in-flight queue to current VRAM pressure without dropping queued frames."""
        if not self._started:
            raise RuntimeError("Queue capacity can only change on a running pipeline")
        for queue in self._queues:
            queue.resize(capacity)

    # -- failure / cancellation -----------------------------------------------------------------------------
    def fail(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None and not isinstance(error, PipelineCancelled):
                self._error = error
                logger.error("Pipeline %s stage failed: %s: %s", self.pipeline_id, type(error).__name__, error,
                             exc_info=(type(error), error, error.__traceback__))
        self.cancel()

    def cancel(self) -> None:
        self.cancel_event.set()
        for queue in self._queues:
            queue.wake_all()
        for pool in self.pools:
            pool.wake_all()

    def _track_cleanup(self, item: Item) -> None:
        """Keep cleanup-bearing items reachable until consumed or the pipeline has fully stopped."""
        if not item.cleanups:
            return
        key = id(item)
        with self._cleanup_lock:
            self._pending_cleanup[key] = item

        def untrack() -> None:
            with self._cleanup_lock:
                self._pending_cleanup.pop(key, None)

        item.cleanups.append(untrack)

    def _release_pending(self) -> None:
        with self._cleanup_lock:
            pending, self._pending_cleanup = list(self._pending_cleanup.values()), {}
        first_error = None
        for item in pending:
            try:
                item.release()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    # -- workers --------------------------------------------------------------------------------------------
    def _run_source(self, out: BoundedQueue) -> None:
        iterator = None
        try:
            iterator = iter(self._source)
            index = 0
            while not self.cancel_event.is_set():
                began = time.perf_counter()                      # decode time = the time spent producing the next frame
                parked = sum(pool.wait_s for pool in self.pools)
                produced = next(iterator, END)
                if produced is END:
                    break
                ended = time.perf_counter()
                # Time blocked on a full buffer pool is back-pressure from downstream, not decode work.
                ended -= min(ended - began, sum(pool.wait_s for pool in self.pools) - parked)
                if isinstance(produced, Item):
                    item = produced
                else:
                    item = Item(index=index, payload=produced)
                self._track_cleanup(item)
                item.entered = began
                item.wall_ms[self.source_name] = (ended - began) * 1000
                self.source_stats.items += 1
                self.source_stats.busy_s += ended - began
                self.source_stats.intervals.append((began, ended))
                self.source_stats.lanes.append((item.index, began, ended))
                index += 1
                out.put(item)
            out.close()
        except BaseException as error:
            self.fail(error)
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _run_stage(self, stage: Stage) -> None:
        torch = _torch()
        stream = stage.stream
        try:
            if torch is not None:
                torch.cuda.set_device(self.device)
            while True:
                item = stage.queue_in.get()
                if item is END:
                    stage.queue_out.close()
                    return
                began = time.perf_counter()
                if item.event is not None and torch is not None:
                    if stream is not None:
                        stream.wait_event(item.event)      # GPU-side dependency; the host thread does not block
                    else:
                        item.event.synchronize()           # host stage: wait for this one event only
                    item.event = None
                if stream is not None:
                    adopt(item.payload, stream)
                start = end = None
                context = torch.cuda.stream(stream) if stream is not None else nullcontext()
                with context:
                    if stream is not None:
                        start = torch.cuda.Event(enable_timing=True)
                        start.record(stream)
                    with TRANSFERS.scope(stage.name):       # attribute ORT host<->device copies to this stage
                        item.payload = stage.function(item)
                    if stream is not None:
                        end = torch.cuda.Event(enable_timing=True)
                        end.record(stream)
                        item.event = end
                finished = time.perf_counter()
                item.wall_ms[stage.name] = (finished - began) * 1000
                stage.stats.items += 1
                stage.stats.busy_s += finished - began
                stage.stats.last_ms = item.wall_ms[stage.name]
                stage.stats.intervals.append((began, finished))
                stage.stats.lanes.append((item.index, began, finished))
                if start is not None:
                    stage._pending.append((start, end))
                    stage._drain_cuda_timings(self._base_event)
                stage.queue_out.put(item)
        except BaseException as error:
            self.fail(error)

    # -- lifecycle ------------------------------------------------------------------------------------------
    def start(self) -> "AsyncPipeline":
        if self._started:
            return self
        self._started = True
        torch = _torch()
        if torch is not None:
            torch.cuda.set_device(self.device)
            self._base_event = torch.cuda.Event(enable_timing=True)
            self._base_event.record()
            self._base_event.synchronize()
            for number, stage in enumerate(self._stages):
                if stage.wants_stream:
                    stage.stream = torch.cuda.Stream(device=self.device, priority=-1 if self.priority_streams else 0)
        names = [self.source_name, *(stage.name for stage in self._stages)]
        self._queues = [BoundedQueue(f"{names[i]}->{names[i + 1] if i + 1 < len(names) else 'results'}",
                                     self.queue_size, cancel=self.cancel_event) for i in range(len(names))]
        for number, stage in enumerate(self._stages):
            stage.queue_in, stage.queue_out = self._queues[number], self._queues[number + 1]
        self._begun = time.perf_counter()
        self._threads = [threading.Thread(target=self._run_source, args=(self._queues[0],),
                                          name=f"r3d-{self.source_name}", daemon=True)]
        self._threads += [threading.Thread(target=self._run_stage, args=(stage,), name=f"r3d-{stage.name}", daemon=True)
                          for stage in self._stages]
        AsyncPipeline.registry[self.pipeline_id] = self
        for thread in self._threads:
            thread.start()
        return self

    def results(self) -> Iterator[Item]:
        """Frames leaving the last stage, in order.  Raises the first worker error, if any."""
        self.start()
        out = self._queues[-1]
        try:
            while True:
                try:
                    item = out.get()
                except PipelineCancelled:
                    break
                if item is END:
                    break
                self.frames_out += 1
                yield item
        finally:
            self._finished = time.perf_counter()
        if self._error is not None:
            raise self._error

    def join(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        torch = _torch()
        if torch is not None:
            for stage in self._stages:
                if stage.stream is not None and not alive:
                    stage.stream.synchronize()
                    stage._drain_cuda_timings(self._base_event)
        if alive:
            raise RuntimeError(f"Pipeline workers did not stop: {', '.join(alive)}")
        # A failed or abandoned frame may still own a pinned slot. Reclaim only after every worker and CUDA stream
        # has stopped using it. Items released by the normal consumer have already removed themselves from this set.
        self._release_pending()

    def __enter__(self) -> "AsyncPipeline":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.cancel()
        self.join()
        if self._error is not None and _exc[0] is None:
            raise self._error

    # -- telemetry ------------------------------------------------------------------------------------------
    @staticmethod
    def _overlap(intervals: list[tuple[float, float]]) -> dict[str, float]:
        """Fraction of the covered timeline during which >=2 / >=3 workers were simultaneously active."""
        events = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals])
        if not events:
            return {"ge1": 0.0, "ge2": 0.0, "ge3": 0.0, "mean_active": 0.0}
        active, last = 0, events[0][0]
        spans = {1: 0.0, 2: 0.0, 3: 0.0}
        weighted = 0.0
        for moment, change in events:
            duration = moment - last
            for k in spans:
                if active >= k:
                    spans[k] += duration
            weighted += active * duration
            active += change
            last = moment
        total = events[-1][0] - events[0][0]
        if total <= 0:
            return {"ge1": 0.0, "ge2": 0.0, "ge3": 0.0, "mean_active": 0.0}
        return {"ge1": round(spans[1] / total, 4), "ge2": round(spans[2] / total, 4),
                "ge3": round(spans[3] / total, 4), "mean_active": round(weighted / total, 3)}

    def _lanes(self, *, limit: int) -> dict[str, Any]:
        """Last ``limit`` frames per stage as ms offsets from the earliest start in the window (host wall clock, measured)."""
        groups = [(self.source_name, self.source_stats), *((stage.name, stage.stats) for stage in self._stages)]
        rows = {name: list(stats.lanes)[-limit:] for name, stats in groups}
        starts = [row[0][1] for row in rows.values() if row]
        if not starts:
            return {"window_ms": 0.0, "stages": {}}
        origin = min(starts)
        stages = {name: [[index, round((a - origin) * 1000, 3), round((b - origin) * 1000, 3)] for index, a, b in row] for name, row in rows.items()}
        ends = [row[-1][2] for row in rows.values() if row]
        return {"window_ms": round((max(ends) - origin) * 1000, 3), "stages": stages}

    def snapshot(self, *, lane_frames: int = 48) -> dict[str, Any]:
        now = time.perf_counter()
        began = self._begun or now
        wall = max(1e-9, (self._finished or now) - began)
        stages = []
        for number, stage in enumerate([None, *self._stages]):
            stats = self.source_stats if stage is None else stage.stats
            name = self.source_name if stage is None else stage.name
            queue_out = self._queues[number] if self._queues else None
            if stage is not None:
                stage._drain_cuda_timings(self._base_event)
            entry = {"name": name, "items": stats.items,
                     "busy_pct": round(100 * stats.busy_s / wall, 1),
                     "mean_ms": round(1000 * stats.busy_s / stats.items, 3) if stats.items else None,
                     "last_ms": round(stats.last_ms, 3),
                     "cuda_stream": stage is not None and stage.stream is not None,
                     "cuda_mean_ms": round(stats.cuda_ms / stats.cuda_items, 3) if stats.cuda_items else None,
                     "queue_out": queue_out.stats() if queue_out is not None else None}
            if stage is not None and stage.queue_in is not None:
                entry["starved_pct"] = round(100 * stage.queue_in.starved_get_s / wall, 1)
            if queue_out is not None:
                entry["blocked_pct"] = round(100 * queue_out.blocked_put_s / wall, 1)
            stages.append(entry)
        intervals = [i for s in [self.source_stats, *(st.stats for st in self._stages)] for i in list(s.intervals)]
        gpu = [w for st in self._stages for w in list(st.stats.gpu_windows)]
        bottleneck = max(stages, key=lambda s: s["busy_pct"], default=None)
        lanes = self._lanes(limit=lane_frames)
        return {"id": self.pipeline_id, "name": self.name, "running": bool(self._started and not self._finished),
                "frames_out": self.frames_out, "wall_s": round(wall, 3), "fps": round(self.frames_out / wall, 2),
                "stages": stages, "bottleneck": bottleneck["name"] if bottleneck else None,
                "overlap_wall": self._overlap(intervals),
                "overlap_gpu_streams": self._overlap(gpu) if gpu else None,
                "gpu_stream_note": None if gpu else "No CUDA stream windows were recorded: torch.cuda streams need PyTorch, which this environment does not use, so only host-thread overlap is measured.",
                "lanes": lanes,
                "transfers": TRANSFERS.snapshot(frames=self.frames_out or None),
                "pools": [{"backend": p.backend, "count": p.count, "in_flight": p.in_flight,
                           "bytes": p.nbytes, "waits": p.waits, "blocked_s": round(p.wait_s, 3)} for p in self.pools]}


def snapshot_all() -> list[dict[str, Any]]:
    return [pipeline.snapshot() for pipeline in list(AsyncPipeline.registry.values())]
