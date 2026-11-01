"""Measured per-frame stage timings for the live bottleneck flamegraph.

Only stages that were actually timed are reported; an unmeasured stage is absent, never zero or a default.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import threading
import time
from typing import Iterator, Mapping

# Canonical order shown in the flamegraph.
STAGES: tuple[str, ...] = ("decode", "detection", "pose_tps", "inswapper", "masking", "restorer", "encode")
STAGE_LABELS: dict[str, str] = {
    "decode": "Decode", "detection": "Detection", "pose_tps": "3D Pose / TPS", "inswapper": "Inswapper",
    "masking": "Masking", "restorer": "Restorer", "encode": "Encode",
}


def validate_timings(value: Mapping[str, float]) -> dict[str, float]:
    """Reject unknown stages and non-finite/negative durations (raises ValueError)."""
    if len(value) > len(STAGES):
        raise ValueError("Too many stage timings")
    result: dict[str, float] = {}
    for key, ms in value.items():
        if key not in STAGES:
            raise ValueError(f"Unknown pipeline stage: {key}")
        if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms < 0:
            raise ValueError("Stage timings must be finite non-negative milliseconds")
        result[key] = float(ms)
    return result


class StageTimings:
    """Thread-safe accumulator: `add` from any worker, `take` once per reported frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def add(self, stage: str, milliseconds: float, count: int = 1) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        if not math.isfinite(milliseconds) or milliseconds < 0 or count < 1:
            return
        with self._lock:
            self._totals[stage] = self._totals.get(stage, 0.0) + milliseconds
            self._counts[stage] = self._counts.get(stage, 0) + count

    @contextmanager
    def measure(self, stage: str, count: int = 1) -> Iterator[None]:
        began = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, (time.perf_counter() - began) * 1000, count)

    def take(self) -> dict[str, float]:
        """Mean milliseconds per measured item since the previous `take`; resets the window."""
        with self._lock:
            result = {stage: round(self._totals[stage] / self._counts[stage], 3) for stage in STAGES if stage in self._totals}
            self._totals.clear()
            self._counts.clear()
        return result
