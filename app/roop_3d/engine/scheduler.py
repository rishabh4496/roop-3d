"""Unified bounded admission for background rendering jobs."""
from __future__ import annotations

from dataclasses import dataclass
import threading

from roop_3d.core.settings import Settings
from roop_3d.engine.vram_governor import RuntimeProfile, VRAMGovernor, VRAMLease


@dataclass
class ScheduledJob:
    job_id: str
    lease: VRAMLease

    @property
    def profile(self) -> RuntimeProfile:
        return self.lease.admission.profile

    @property
    def provider(self) -> str:
        return self.lease.admission.provider

    def release(self) -> None:
        self.lease.release()


class RuntimeScheduler:
    """Coordinate configured job limits with current per-profile VRAM capacity."""

    def __init__(self, governor: VRAMGovernor | None = None):
        self.governor = governor or VRAMGovernor()
        self._lock = threading.RLock()
        self._active: dict[str, ScheduledJob] = {}

    def try_admit(self, job_id: str, settings: Settings, *,
                  force_profile: RuntimeProfile | None = None) -> ScheduledJob | None:
        with self._lock:
            if job_id in self._active:
                return self._active[job_id]
            if len(self._active) >= settings.max_concurrent_jobs:
                return None
            lease = self.governor.try_acquire(job_id, settings, force_profile=force_profile)
            if lease is None:
                return None
            scheduled = ScheduledJob(job_id, lease)
            self._active[job_id] = scheduled
            return scheduled

    def release(self, job_id: str) -> None:
        with self._lock:
            scheduled = self._active.pop(job_id, None)
        if scheduled is not None:
            scheduled.release()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)
