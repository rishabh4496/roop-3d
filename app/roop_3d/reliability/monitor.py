"""Always-on, bounded resource monitor: peaks, growth verdicts, stage timings and a sanitized warning buffer.

Memory use of the monitor itself is fixed: a ring of ``capacity`` samples, per-stage running aggregates, and the last
``MAX_JOBS`` per-job peaks. Nothing here stores media, paths or biometric data.
"""
from __future__ import annotations

from collections import deque
import logging
import threading
import time
from typing import Any

from roop_3d.reliability import probe
from roop_3d.reliability.growth import DEFAULT_PROFILE, BoundedProfile, GrowthReport, analyse
from roop_3d.reliability.ledger import LEDGER, ResourceLedger

MAX_JOBS = 50


class WarningBuffer(logging.Handler):
    """Keeps the most recent WARNING+ records of the application, scrubbed with the shared log sanitizer."""

    def __init__(self, capacity: int = 200):
        super().__init__(level=logging.WARNING)
        self.records: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._counts: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from roop_3d.security.log_sanitizer import sanitize_log_message
            message = sanitize_log_message(record.getMessage())[:300]
            self.records.append({"time": round(record.created, 3), "level": record.levelname,
                                 "logger": record.name, "message": message,
                                 "exception": record.exc_info[0].__name__ if record.exc_info else None})
            self._counts[record.levelname] = self._counts.get(record.levelname, 0) + 1
        except Exception:  # a logging handler must never raise into the code that logged
            pass

    def counts(self) -> dict[str, int]:
        return dict(self._counts)


class ResourceMonitor:
    def __init__(self, *, interval: float = 5.0, capacity: int = 720, ledger: ResourceLedger = LEDGER,
                 profile: BoundedProfile = DEFAULT_PROFILE, with_gpu: bool = True):
        self.interval, self.ledger, self.profile, self.with_gpu = interval, ledger, profile, with_gpu
        self.samples: deque[probe.ResourceSample] = deque(maxlen=capacity)
        self.started = time.monotonic()
        self.peaks: dict[str, float] = {}
        self.job_peaks: dict[str, dict[str, float]] = {}
        self._active_jobs: set[str] = set()
        self.stage_stats: dict[str, dict[str, float]] = {}
        self.job_stages: dict[str, dict[str, dict[str, float]]] = {}
        self.warnings = WarningBuffer()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        logging.getLogger().addHandler(self.warnings)
        self._thread = threading.Thread(target=self._run, name="roop3d-resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.interval + 5)
        self._thread = None
        logging.getLogger().removeHandler(self.warnings)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    # ---- sampling
    def sample(self, *, with_gpu: bool | None = None) -> probe.ResourceSample:
        value = probe.take_sample(self.started, gauges=self.ledger.counts(),
                                  with_gpu=self.with_gpu if with_gpu is None else with_gpu)
        with self._lock:
            self.samples.append(value)
            metrics = value.metrics()
            for key in ("rss_mib", "private_mib", "gpu_used_mib", "gpu_process_mib", "handles", "os_threads", "children"):
                if key in metrics:
                    self.peaks[key] = max(self.peaks.get(key, 0.0), metrics[key])
            if value.memory.peak_rss_mib is not None:
                self.peaks["os_peak_rss_mib"] = max(self.peaks.get("os_peak_rss_mib", 0.0), value.memory.peak_rss_mib)
            for job_id in self._active_jobs:
                peak = self.job_peaks.setdefault(job_id, {})
                for key in ("rss_mib", "gpu_used_mib", "gpu_process_mib"):
                    if key in metrics:
                        peak[key] = max(peak.get(key, 0.0), metrics[key])
        return value

    def job_started(self, job_id: str, *, with_gpu: bool | None = None) -> None:
        with self._lock:
            self._active_jobs.add(job_id)
            self.job_peaks.setdefault(job_id, {})
            while len(self.job_peaks) > MAX_JOBS:
                self.job_peaks.pop(next(iter(self.job_peaks)))
        self.sample(with_gpu=with_gpu)  # a job shorter than the interval still gets a start and an end reading

    def job_finished(self, job_id: str, *, with_gpu: bool | None = None) -> None:
        self.sample(with_gpu=with_gpu)
        with self._lock:
            self._active_jobs.discard(job_id)

    # ---- stage timings
    def observe_stages(self, job_id: str, timings: dict[str, float]) -> None:
        with self._lock:
            per_job = self.job_stages.setdefault(job_id, {})
            while len(self.job_stages) > MAX_JOBS:
                self.job_stages.pop(next(iter(self.job_stages)))
            for stage, ms in timings.items():
                for table in (self.stage_stats, per_job):
                    entry = table.setdefault(stage, {"count": 0.0, "mean_ms": 0.0, "max_ms": 0.0})
                    entry["count"] += 1
                    entry["mean_ms"] += (ms - entry["mean_ms"]) / entry["count"]
                    entry["max_ms"] = max(entry["max_ms"], ms)

    # ---- reporting
    def series(self) -> dict[str, list[tuple[float, float]]]:
        with self._lock:
            samples = list(self.samples)
        out: dict[str, list[tuple[float, float]]] = {}
        for item in samples:
            for key, value in item.metrics().items():
                out.setdefault(key, []).append((item.elapsed, value))
        return out

    def growth(self) -> GrowthReport:
        return analyse(self.profile, self.series())

    def summary(self) -> dict[str, Any]:
        with self._lock:
            latest = self.samples[-1].metrics() if self.samples else {}
            stages = {k: {m: round(v, 3) for m, v in e.items()} for k, e in sorted(self.stage_stats.items())}
            return {"samples": len(self.samples), "interval_s": self.interval,
                    "uptime_s": round(time.monotonic() - self.started, 1),
                    "latest": {k: round(v, 3) for k, v in latest.items()},
                    "peaks": {k: round(v, 3) for k, v in self.peaks.items()},
                    "job_peaks": {k: {m: round(v, 3) for m, v in p.items()} for k, p in self.job_peaks.items()},
                    "stage_timings_ms": stages}


MONITOR = ResourceMonitor()
