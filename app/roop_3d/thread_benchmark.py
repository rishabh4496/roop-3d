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
        # The benchmark measured a synthetic affine crop-warp graph (the removed 2D crop-paste kernel). Until a true-3D render
        # workload exists there is nothing honest to measure, so it reports that instead of a fabricated recommendation.
        with self._lock:
            self._state.update(state="UNAVAILABLE", completed_at=time.time(),
                               note="3D renderer not wired: the synthetic affine crop workload this benchmark used was removed.")

    def close(self):
        self._cancel.set()
        thread = self._thread
        if thread:
            thread.join(timeout=60)
