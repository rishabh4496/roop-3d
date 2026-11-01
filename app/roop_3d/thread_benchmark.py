"""Explicit, bounded local thread/pool benchmark; never runs at API startup."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import threading
import time

import numpy as np


class ThreadBenchmark:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._cancel = threading.Event()
        self._state = {"state": "IDLE", "results": [], "recommendation": None}

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._state)

    def start(self, profile="quick"):
        if profile not in ("quick", "full"):
            raise ValueError("Unknown thread benchmark profile")
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("A thread benchmark is already running")
            self._state = {"state": "RUNNING", "profile": profile, "results": [], "recommendation": None,
                           "scope": "synthetic_affine_iobinding_host_transfer", "since": time.time()}
            self._cancel.clear()
            self._thread = threading.Thread(target=self._run, args=(profile,), name="roop3d-thread-benchmark", daemon=True)
            self._thread.start()
        return self.snapshot()

    def _run(self, profile):
        try:
            from roop_3d.models.native_libraries import register_native_libraries
            from roop_3d.render_policy import CudaWorkerContext
            from roop_3d.vectorized_pipeline import affine_graph, OrtValueRunner, normalized_matrix
            import onnxruntime as ort
            register_native_libraries()
            # Prove an actual CUDA call before recommending GPU concurrency.
            from roop_3d.core.hardware import probe_hardware
            from roop_3d.core.settings import Settings
            hardware = probe_hardware(Settings())
            gpu = bool(hardware.gpus) and "CUDAExecutionProvider" in ort.get_available_providers()
            provider = "CUDAExecutionProvider" if gpu else "CPUExecutionProvider"
            device = "cuda" if gpu else "cpu"
            image = np.linspace(0, 1, 1 * 3 * 256 * 256, dtype=np.float32).reshape(1, 3, 256, 256)
            matrix = normalized_matrix(np.array([[1, 0, 40], [0, 1, 40]], np.float32), 256, 256)[None]
            graph = affine_graph(256, 256, 1).SerializeToString()
            counts = (1, 2, 4) if profile == "quick" else (1, 2, 4, 8)
            pools = (1, 2) if profile == "quick" else (1, 2, 4)
            iterations = 12 if profile == "quick" else 48
            results = []
            for threads in counts:
                for pool in pools:
                    if self._cancel.is_set():
                        with self._lock:
                            self._state["state"] = "CANCELLED"
                        return
                    local = threading.local()
                    def work(_):
                        if not hasattr(local, "runner"):
                            local.context = CudaWorkerContext(enabled=gpu)
                            with local.context.activate():
                                options = ort.SessionOptions()
                                options.intra_op_num_threads = 1
                                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1" if gpu else "0")
                                session = ort.InferenceSession(graph, sess_options=options, providers=[provider])
                                session.disable_fallback()
                                if session.get_providers()[0] != provider:
                                    raise RuntimeError("Benchmark provider failed activation")
                                local.runner = OrtValueRunner(session, {"output": (1, 3, 128, 128)}, device=device)
                        with local.context.activate():
                            value = local.runner.run({"image": image, "matrix": matrix})["output"].numpy()
                            if not np.isfinite(value).all():
                                raise RuntimeError("Benchmark output is non-finite")
                    with ThreadPoolExecutor(max_workers=threads) as executor:
                        # Session creation and warmup are excluded from the measurement.
                        barrier = threading.Barrier(threads)
                        def warm():
                            work(0)
                            barrier.wait(timeout=30)
                        warmups = [executor.submit(warm) for _ in range(threads)]
                        for future in warmups:
                            future.result()
                        start = time.perf_counter()
                        for first in range(0, iterations, threads * pool):
                            tasks = [executor.submit(work, i) for i in range(first, min(first + threads * pool, iterations))]
                            for task in tasks:
                                task.result()
                        elapsed = time.perf_counter() - start
                    result = {"execution_threads": threads, "pool_size": pool, "samples": iterations,
                              "seconds": elapsed, "samples_per_second": iterations / elapsed, "provider": provider}
                    results.append(result)
                    with self._lock:
                        self._state["results"] = copy.deepcopy(results)
            best = max(results, key=lambda value: value["samples_per_second"])
            with self._lock:
                self._state.update(state="DONE", recommendation=best, provider=provider,
                    note="Recommendation is for the measured synthetic affine/transfer workload, not model or video-pipeline FPS.",
                    completed_at=time.time())
        except Exception as error:
            with self._lock:
                self._state.update(state="FAILED", error=type(error).__name__, completed_at=time.time())

    def close(self):
        self._cancel.set()
        thread = self._thread
        if thread:
            thread.join(timeout=60)
