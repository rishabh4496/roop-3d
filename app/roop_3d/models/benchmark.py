"""Synchronous session timing including output transfer; bounded raw samples."""
from __future__ import annotations

from dataclasses import dataclass
import time
from collections.abc import Callable
import numpy as np


@dataclass(frozen=True)
class BenchmarkResult:
    warmups: int
    iterations: int
    p50_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float
    samples_ms: tuple[float, ...]


def benchmark(call: Callable[[], object], *, warmups: int = 2, iterations: int = 10) -> BenchmarkResult:
    if not 0 <= warmups <= 100 or not 2 <= iterations <= 1000:
        raise ValueError("Benchmark requires 2..1000 measured calls and 0..100 warmups")
    for _ in range(warmups):
        call()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns() - started) / 1000000)
        del result
    return summarize(samples, warmups=warmups)


def summarize(samples: list[float], *, warmups: int = 0) -> BenchmarkResult:
    values = np.asarray(samples, np.float64)
    if values.ndim != 1 or not 2 <= len(values) <= 1000 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Timing samples must be bounded, finite and nonnegative")
    return BenchmarkResult(warmups, len(values), float(np.percentile(values, 50, method="linear")),
                           float(np.percentile(values, 95, method="linear")), float(values.min()),
                           float(values.max()), tuple(float(x) for x in values))
