"""Explicitly requested strict TRT inference; constructor never loads a model."""
from __future__ import annotations

from pathlib import Path
import copy
import threading

from roop_3d.models.dynamic_shapes import assert_finite
from roop_3d.trt_session_builder import BatchProfile, TensorRTSessionBuilder
from roop_3d.engine.resources import guarded


class OptimizedTRTEngine:
    def __init__(self, model: str | Path, *, builder: TensorRTSessionBuilder | None = None,
                 profile: BatchProfile | None = None, shapes=None):
        self.model, self.builder = Path(model), builder or TensorRTSessionBuilder()
        self.profile, self.shapes = profile, shapes
        self._session = None
        self._build = None
        self._lock = threading.RLock()

    @guarded
    def prepare(self):
        with self._lock:
            if self._session is None:
                self._session = self.builder.build(self.model, profile=self.profile, shapes=self.shapes)
                self._build = copy.deepcopy(self.builder.last_build)
            return self._session

    @guarded
    def run(self, feeds):
        with self._lock:
            session = self.prepare()
            # Reject out-of-profile requests instead of silently triggering another build.
            batches = self._build["checked_batches"]
            for name, value in feeds.items():
                if name not in ("w", "fidelity_weight") and value.ndim and value.shape[0] not in batches:
                    raise ValueError("Batch must be one of the executed canary batches")
                limits = self._build["profiles"]
                if name not in limits["min"] or len(value.shape) != len(limits["min"][name]) or any(
                    not lo <= n <= hi for lo, n, hi in zip(limits["min"][name], value.shape, limits["max"][name])):
                    raise ValueError(f"Input {name} is outside the compiled shape profile")
            outputs = session.run(None, feeds)
            assert_finite(outputs)
            return outputs

    @guarded
    def close(self):
        with self._lock:
            self._session = None
            self._build = None
