"""Defect #3: the asynchronous multi-stream pipeline, the shared/exclusive inference fence and hardware telemetry."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from roop_3d.async_pipeline import AsyncPipeline, BoundedQueue, END, Item, PinnedFramePool, PipelineCancelled
from roop_3d.async_pipeline.hwmon import HardwareMonitor, Nvml
from roop_3d.engine.resources import SharedExclusiveFence, inference

try:
    import torch
    CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    CUDA = False

needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA device required")


# ------------------------------------------------------------------------------------------- bounded queue
def test_queue_blocks_producer_at_capacity_and_reports_depth():
    queue = BoundedQueue("q", 2)
    queue.put(1)
    queue.put(2)
    done = threading.Event()

    def producer():
        queue.put(3)
        done.set()

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    assert not done.wait(0.25), "a full queue must hold the producer (bounded memory)"
    assert queue.depth == 2 and queue.high_water == 2
    assert queue.get() == 1
    assert done.wait(2)
    assert [queue.get(), queue.get()] == [2, 3]
    assert queue.stats()["blocked_put_s"] > 0.15      # back-pressure is measured, so the HUD can name the bottleneck


def test_queue_close_drains_then_ends_and_cancel_raises():
    queue = BoundedQueue("q", 4)
    queue.put("a")
    queue.close()
    assert queue.get() == "a"
    assert queue.get() is END
    cancel = threading.Event()
    other = BoundedQueue("c", 1, cancel=cancel)
    cancel.set()
    with pytest.raises(PipelineCancelled):
        other.get()


def test_queue_size_is_bounded():
    with pytest.raises(ValueError):
        BoundedQueue("q", 0)
    with pytest.raises(ValueError):
        BoundedQueue("q", 65)


# ------------------------------------------------------------------------------------------- pinned pool
def test_pinned_pool_back_pressure_and_double_release():
    pool = PinnedFramePool((4, 4, 3), 2)
    a, _ = pool.acquire()
    b, _ = pool.acquire()
    assert pool.in_flight == 2
    got = []
    thread = threading.Thread(target=lambda: got.append(pool.acquire()[0]), daemon=True)
    thread.start()
    time.sleep(0.2)
    assert not got, "an exhausted pool must block, not allocate"
    pool.release(a)
    thread.join(2)
    assert got == [a]
    with pytest.raises(ValueError):
        pool.release(b), pool.release(b)


@needs_cuda
def test_pinned_pool_is_page_locked_and_shares_memory():
    pool = PinnedFramePool((8, 8, 3), 2)
    slot, array = pool.acquire()
    assert pool.backend == "cuda_pinned" and pool.tensor(slot).is_pinned()
    array[:] = 7
    assert int(pool.tensor(slot)[0, 0, 0]) == 7        # the numpy view IS the pinned tensor
    device = pool.tensor(slot).to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    assert int(device[0, 0, 0]) == 7


# ------------------------------------------------------------------------------------------- pipeline
def _sleeper(milliseconds):
    def work(item):
        time.sleep(milliseconds / 1000)
        return item.payload + 1
    return work


def test_stages_overlap_and_preserve_order():
    frames, stage_ms = 40, 20
    pipeline = AsyncPipeline("t", range(frames), queue_size=4)
    pipeline.add_stage("a", _sleeper(stage_ms)).add_stage("b", _sleeper(stage_ms)).add_stage("c", _sleeper(stage_ms))
    began = time.perf_counter()
    with pipeline:
        out = [(item.index, item.payload) for item in pipeline.results()]
    wall = time.perf_counter() - began
    assert [i for i, _ in out] == list(range(frames)), "frames must leave in the order they entered"
    assert [p for _, p in out] == [i + 3 for i in range(frames)]
    serial = frames * 3 * stage_ms / 1000
    assert wall < 0.6 * serial, f"{wall:.2f}s is not meaningfully faster than serial {serial:.2f}s"
    snap = pipeline.snapshot()
    assert snap["overlap_wall"]["ge3"] > 0.6, snap["overlap_wall"]
    assert [s["name"] for s in snap["stages"]] == ["decode", "a", "b", "c"]
    assert all(s["queue_out"]["high_water"] <= 4 for s in snap["stages"]), "queues must stay bounded"


def test_slowest_stage_is_named_the_bottleneck():
    pipeline = AsyncPipeline("t", range(20), queue_size=2)
    pipeline.add_stage("fast", _sleeper(2)).add_stage("slow", _sleeper(25)).add_stage("fast2", _sleeper(2))
    with pipeline:
        list(pipeline.results())
    snap = pipeline.snapshot()
    assert snap["bottleneck"] == "slow"
    by_name = {s["name"]: s for s in snap["stages"]}
    assert by_name["slow"]["busy_pct"] > 70
    assert by_name["fast"]["blocked_pct"] > 30      # the producer in front of the slow stage waits on back-pressure
    assert by_name["fast2"]["starved_pct"] > 30     # the consumer behind it waits for work


def test_worker_exception_propagates_and_all_threads_stop():
    def explode(item):
        if item.index == 5:
            raise RuntimeError("boom")
        return item.payload

    pipeline = AsyncPipeline("t", range(1000), queue_size=2)
    pipeline.add_stage("a", _sleeper(1)).add_stage("b", explode).add_stage("c", _sleeper(1))
    with pytest.raises(RuntimeError, match="boom"):
        with pipeline:
            for _ in pipeline.results():
                pass
    assert not any(thread.is_alive() for thread in pipeline._threads)


def test_consumer_abandoning_the_stream_cancels_cleanly():
    pipeline = AsyncPipeline("t", range(10_000), queue_size=2)
    pipeline.add_stage("a", _sleeper(1))
    with pipeline:
        for item in pipeline.results():
            if item.index == 3:
                break
    assert not any(thread.is_alive() for thread in pipeline._threads)


def test_source_exception_is_reported():
    def source():
        yield 1
        raise ValueError("decoder died")

    pipeline = AsyncPipeline("t", source())
    pipeline.add_stage("a", lambda item: item.payload)
    with pytest.raises(ValueError, match="decoder died"):
        with pipeline:
            list(pipeline.results())


def test_source_iterator_initialization_error_propagates_without_hanging():
    class InvalidSource:
        def __iter__(self):
            raise ValueError("source could not start")

    pipeline = AsyncPipeline("invalid-source", InvalidSource())
    pipeline.add_stage("pass", lambda item: item.payload)
    with pytest.raises(ValueError, match="source could not start"):
        with pipeline:
            list(pipeline.results())
    assert not any(thread.is_alive() for thread in pipeline._threads)


def test_cleanups_run_once():
    freed = []
    items = []
    for number in range(3):
        item = Item(index=number)
        item.cleanups.append(lambda n=number: freed.append(n))
        items.append(item)
    pipeline = AsyncPipeline("t", items)
    pipeline.add_stage("a", lambda item: item.payload)
    with pipeline:
        for item in pipeline.results():
            item.release()
            item.release()
    assert freed == [0, 1, 2]


def test_worker_failure_releases_all_pipeline_owned_items():
    freed = []
    items = []
    for number in range(12):
        item = Item(index=number)
        item.cleanups.append(lambda n=number: freed.append(n))
        items.append(item)

    def explode(item):
        if item.index == 2:
            time.sleep(0.05)  # allow the source to transfer the bounded batch before cancellation
            raise RuntimeError("stage failed")
        return item.payload

    pipeline = AsyncPipeline("cleanup-failure", items, queue_size=16)
    pipeline.add_stage("explode", explode)
    with pytest.raises(RuntimeError, match="stage failed"):
        with pipeline:
            list(pipeline.results())
    assert sorted(freed) == list(range(12))
    assert len(freed) == len(set(freed)), "cleanup callbacks must run exactly once"


def test_consumer_abandonment_releases_unconsumed_items():
    freed = []

    def source():
        for number in range(100):
            item = Item(index=number)
            item.cleanups.append(lambda n=number: freed.append(n))
            yield item

    pipeline = AsyncPipeline("cleanup-abandoned", source(), queue_size=2)
    pipeline.add_stage("pass", lambda item: item.payload)
    with pipeline:
        for item in pipeline.results():
            assert item.index == 0
            break
    assert freed, "the abandoned result must be reclaimed during pipeline shutdown"
    assert len(freed) == len(set(freed)), "cleanup callbacks must run exactly once"


@needs_cuda
def test_cuda_streams_hand_off_with_events_not_device_sync():
    torch.cuda.init()
    size = 256

    def stage_a(item):
        tensor = item.payload.to("cuda", non_blocking=True)
        return tensor * 2

    def stage_b(item):
        return (item.payload @ item.payload.T) / size

    def stage_c(item):
        return item.payload + 1   # stays on device; ordering is carried by the event

    def host(item):
        return float(item.payload.sum().cpu())

    source = [torch.full((size, size), float(n), pin_memory=True) for n in range(12)]
    pipeline = AsyncPipeline("cuda", source, queue_size=4)
    pipeline.add_stage("a", stage_a, stream=True).add_stage("b", stage_b, stream=True)
    pipeline.add_stage("c", stage_c, stream=True).add_stage("host", host)
    with pipeline:
        out = [item.payload for item in pipeline.results()]
    for n, got in enumerate(out):
        a = torch.full((size, size), float(n), device="cuda") * 2
        want = float(((a @ a.T) / size + 1).sum().cpu())
        assert got == pytest.approx(want, rel=1e-4)
    snap = pipeline.snapshot()
    streams = [s for s in snap["stages"] if s["cuda_stream"]]
    assert len(streams) == 3 and all(s["cuda_mean_ms"] is not None for s in streams), snap["stages"]
    assert len({id(stage.stream) for stage in pipeline._stages if stage.stream}) == 3, "each stage owns its own stream"


# ------------------------------------------------------------------------------------------- fence
def test_fence_shared_holders_run_concurrently():
    fence = SharedExclusiveFence()
    inside, release = [], threading.Event()
    barrier = threading.Barrier(3, timeout=3)

    def reader():
        with fence.shared():
            barrier.wait()          # only passes if all three hold the fence at the same moment
            inside.append(1)
            release.wait(2)

    threads = [threading.Thread(target=reader, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    assert len(inside) == 3 and fence.snapshot()["shared_holders"] == 3
    release.set()
    for t in threads:
        t.join(2)


def test_fence_exclusive_waits_for_readers_and_blocks_new_ones():
    fence = SharedExclusiveFence()
    order = []
    reader_in, reader_go = threading.Event(), threading.Event()

    def reader():
        with fence.shared():
            reader_in.set()
            reader_go.wait(3)
            order.append("reader-done")

    def writer():
        with fence.exclusive():
            order.append("writer")

    def late_reader():
        with fence.shared():
            order.append("late-reader")

    r = threading.Thread(target=reader, daemon=True)
    r.start()
    reader_in.wait(2)
    w = threading.Thread(target=writer, daemon=True)
    w.start()
    time.sleep(0.15)
    late = threading.Thread(target=late_reader, daemon=True)
    late.start()
    time.sleep(0.15)
    assert order == [], "neither the writer nor a new reader may enter while a reader is in flight"
    reader_go.set()
    for t in (r, w, late):
        t.join(3)
    assert order == ["reader-done", "writer", "late-reader"]


def test_fence_is_reentrant_like_an_rlock():
    fence = SharedExclusiveFence()
    with fence:
        with fence:
            with fence.shared():
                with fence.shared():
                    pass
    with fence.shared():
        with fence.shared():
            pass
    assert fence.snapshot() == {"shared_holders": 0, "exclusive": 0, "writers_waiting": 0}


def test_fence_shared_to_exclusive_upgrade_restores_shared_and_never_deadlocks():
    fence = SharedExclusiveFence()
    results, errors = [], []

    def worker(name):
        try:
            with fence.shared():
                time.sleep(0.05)
                with fence:                     # a lazy model build inside a frame inference
                    results.append(name)
                with fence.shared():            # shared holds are back after the upgrade
                    pass
        except Exception as error:               # pragma: no cover
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(n,), daemon=True) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert not any(t.is_alive() for t in threads), "two threads upgrading at once deadlocked"
    assert sorted(results) == [0, 1, 2, 3] and not errors
    assert fence.snapshot() == {"shared_holders": 0, "exclusive": 0, "writers_waiting": 0}


def test_inference_decorator_runs_different_models_in_parallel_but_one_model_serially():
    class Model:
        active = 0
        peak = 0

        @inference
        def run(self):
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
            time.sleep(0.1)
            type(self).active -= 1

    class A(Model):
        active = peak = 0

    class B(Model):
        active = peak = 0

    a, b = A(), B()
    pairs = [threading.Thread(target=a.run), threading.Thread(target=a.run), threading.Thread(target=b.run)]
    began = time.perf_counter()
    for t in pairs:
        t.start()
    for t in pairs:
        t.join(3)
    assert A.peak == 1, "one instance must never run two inferences at once (its I/O binding is mutable state)"
    assert time.perf_counter() - began < 0.28, "the second model must have overlapped the first"


# ------------------------------------------------------------------------------------------- telemetry
def test_hardware_monitor_reports_measurements_or_none():
    monitor = HardwareMonitor(interval=0.05)
    sample = monitor.sample_once()
    for key in ("gpu_util_pct", "cpu_util_pct", "pcie_tx_mb_s", "pcie_rx_mb_s", "encoder_util_pct", "vram_used_mib"):
        assert key in sample
    if sample["gpu_util_pct"] is not None:
        assert 0 <= sample["gpu_util_pct"] <= 100
    monitor.acquire()
    time.sleep(0.4)
    snapshot = monitor.snapshot()
    monitor.release()
    assert snapshot["backend"] in ("nvml", "nvidia-smi")
    assert snapshot["window"]["cpu_util_pct"] is None or 0 <= snapshot["window"]["cpu_util_pct"]["mean"] <= 100


@pytest.mark.skipif(not Nvml().available, reason="NVML unavailable")
def test_nvml_exposes_pcie_and_engine_counters():
    sample = Nvml().sample()
    assert sample["vram_total_mib"] and sample["vram_total_mib"] > 0
    assert sample["pcie_rx_mb_s"] is not None and sample["encoder_util_pct"] is not None
