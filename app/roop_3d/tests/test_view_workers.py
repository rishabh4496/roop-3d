"""Per-view correspondence jobs run side by side but behave exactly like the sequential loop."""
import threading
import time

import numpy as np
import pytest

from roop_3d.reconstruction.dense_correspondence import ReferenceSurfaceRasterizer, ViewWorkers, correspondence_field
from roop_3d.reconstruction.neural_face import view_worker_count


class RecordingRasterizer:
    def __init__(self):
        self.released = []
        self.lock = threading.Lock()

    def release(self):
        with self.lock:
            self.released.append(threading.get_ident())


def test_results_come_back_in_job_order_even_when_later_jobs_finish_first():
    workers = ViewWorkers(4, RecordingRasterizer())
    try:
        jobs = [(lambda i=i: (time.sleep(0.05 * (4 - i)), i)[1]) for i in range(4)]
        assert workers.run(jobs) == [0, 1, 2, 3]
    finally:
        workers.close()


def test_jobs_actually_overlap():
    workers = ViewWorkers(4, RecordingRasterizer())
    barrier = threading.Barrier(4, timeout=5)
    try:
        # Four jobs that can only finish if all four are running at the same moment.
        assert workers.run([lambda: barrier.wait() >= 0 for _ in range(4)]) == [True] * 4
    finally:
        workers.close()


def test_one_worker_or_one_job_runs_inline_on_the_calling_thread():
    for count, jobs in ((1, 3), (4, 1)):
        workers = ViewWorkers(count, RecordingRasterizer())
        seen = workers.run([threading.get_ident for _ in range(jobs)])
        assert seen == [threading.get_ident()] * jobs
        assert workers._pool is None


def test_first_failure_in_view_order_is_raised_after_every_job_finished():
    finished = []
    workers = ViewWorkers(4, RecordingRasterizer())

    def slow_ok():
        time.sleep(0.15)
        finished.append("ok")

    def boom(name):
        def job():
            raise ValueError(name)
        return job
    try:
        with pytest.raises(ValueError, match="first"):
            workers.run([slow_ok, boom("first"), boom("second"), slow_ok])
        assert finished == ["ok", "ok"]          # nothing is still running once run() has raised
    finally:
        workers.close()


def test_close_releases_one_context_per_worker_thread_and_the_pool_is_reusable():
    rasterizer = RecordingRasterizer()
    workers = ViewWorkers(3, rasterizer)
    barrier = threading.Barrier(3, timeout=5)
    threads = set(workers.run([lambda: (barrier.wait(), threading.get_ident())[1] for _ in range(3)]))
    assert len(threads) == 3 and threading.get_ident() not in threads
    workers.close()
    assert set(rasterizer.released) == threads and len(rasterizer.released) == 3
    workers.close()                                 # idempotent
    assert workers.run([lambda: 1, lambda: 2]) == [1, 2]      # a released engine can be used again
    workers.close()


def test_parallel_fields_are_bit_identical_to_sequential():
    rng = np.random.default_rng(3)
    grid = 14
    u = np.linspace(-.05, .05, grid)
    x, y = np.meshgrid(u, u)
    vertices = np.column_stack((x.ravel(), y.ravel(), .6 + 2 * (x.ravel() ** 2 + y.ravel() ** 2))).astype(np.float32)
    index = np.arange(grid * grid).reshape(grid, grid)
    a, b, c, d = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel()
    faces = np.concatenate((np.column_stack((a, c, b)), np.column_stack((b, c, d)))).astype(np.int32)
    target_k = np.array([[900, 0, 24], [0, 900, 24], [0, 0, 1]], np.float32)
    source_k = np.array([[1000, 0, 32], [0, 1000, 32], [0, 0, 1]], np.float32)
    depth = np.full((64, 64), .6, np.float32)
    rasterizer = ReferenceSurfaceRasterizer()
    shifts = [rng.normal(0, .002, 3).astype(np.float32) for _ in range(4)]

    def job(shift):
        return lambda: correspondence_field(rasterizer, faces, vertices, source_k, vertices + shift, target_k,
                                            (48, 48), depth, (64, 64))
    sequential = [job(s)() for s in shifts]
    workers = ViewWorkers(4, rasterizer)
    try:
        parallel = workers.run([job(s) for s in shifts])
    finally:
        workers.close()
    for one, other in zip(sequential, parallel):
        assert np.array_equal(one.source_xy, other.source_xy) and np.array_equal(one.confidence, other.confidence)


def test_worker_count_follows_the_profile_and_the_environment(monkeypatch):
    monkeypatch.delenv("ROOP3D_VIEW_WORKERS", raising=False)
    assert view_worker_count("12gb") == 4 and view_worker_count("6gb") == 2
    monkeypatch.setenv("ROOP3D_VIEW_WORKERS", "1")
    assert view_worker_count("12gb") == 1
    monkeypatch.setenv("ROOP3D_VIEW_WORKERS", "junk")
    assert view_worker_count("12gb") == 4
    monkeypatch.setenv("ROOP3D_VIEW_WORKERS", "99")
    assert view_worker_count("12gb") == 8
