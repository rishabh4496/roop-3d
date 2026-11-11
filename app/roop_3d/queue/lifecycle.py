"""The 10 user-facing job lifecycle states (Stage 23) as a lossless projection of the persisted queue state.

The queue persists the canonical states in :class:`roop_3d.queue_manager.CanonicalJobState` (12 values, with the
Stage 9 ANALYZING and RETRYING hops). Renaming or adding persisted values would invalidate every saved queue and
checkpoint, so the workstation vocabulary is derived from ``(state, progress.stage)`` instead:

========================  =====================================================
lifecycle state           derived from
========================  =====================================================
queued                    QUEUED
preparing                 PREFLIGHT, ANALYZING, RETRYING
compiling_trt             PRESCANNING/PREPASSING/RENDERING while the stage names an engine build
tracking_3d               PRESCANNING, PREPASSING (face tracking + 3D fitting)
rendering                 RENDERING
paused                    PAUSED
muxing_audio              MUXING
completed / failed / cancelled   COMPLETED / FAILED / CANCELLED
========================  =====================================================

The mapping is pure and takes strings, so it has no import cycle with ``queue_manager`` and the UI can mirror it.
"""
from __future__ import annotations

from enum import Enum
import re


class LifecycleState(str, Enum):
    queued = "queued"
    preparing = "preparing"
    compiling_trt = "compiling_trt"
    tracking_3d = "tracking_3d"
    rendering = "rendering"
    paused = "paused"
    muxing_audio = "muxing_audio"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


LIFECYCLE_ORDER: tuple[LifecycleState, ...] = tuple(LifecycleState)
TERMINAL: frozenset[LifecycleState] = frozenset({LifecycleState.completed, LifecycleState.failed, LifecycleState.cancelled})

_DIRECT = {
    "QUEUED": LifecycleState.queued,
    "PREFLIGHT": LifecycleState.preparing,
    "ANALYZING": LifecycleState.preparing,
    "RETRYING": LifecycleState.preparing,
    "PRESCANNING": LifecycleState.tracking_3d,
    "PREPASSING": LifecycleState.tracking_3d,
    "RENDERING": LifecycleState.rendering,
    "MUXING": LifecycleState.muxing_audio,
    "PAUSED": LifecycleState.paused,
    "COMPLETED": LifecycleState.completed,
    "FAILED": LifecycleState.failed,
    "CANCELLED": LifecycleState.cancelled,
}
_ENGINE_BUILD_MARKERS = ("compil", "trt_build", "engine_build", "building_engine", "tensorrt_build")
_BUILD_PHASES = frozenset({"PRESCANNING", "PREPASSING", "RENDERING"})


def lifecycle_of(state: str, stage: str | None = None) -> LifecycleState:
    """Project a canonical state (+ the progress stage text) onto the 10 workstation states."""
    key = str(getattr(state, "value", state)).strip().upper()
    if key not in _DIRECT:
        raise ValueError(f"Unrecognized canonical job state '{state}'")
    normalized = re.sub(r"[\s\-]+", "_", stage.lower()) if stage else ""
    if key in _BUILD_PHASES and any(marker in normalized for marker in _ENGINE_BUILD_MARKERS):
        return LifecycleState.compiling_trt
    return _DIRECT[key]
