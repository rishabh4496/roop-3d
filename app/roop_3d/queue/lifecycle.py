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


# --------------------------------------------------------------------------------------------- the 10 specification states
class SpecState(str, Enum):
    """The job states of the product specification: queued, validating, slicing, tracking, swapping, restoring, merging,
    remuxing, completed, and the tenth slot ``failed/cancelled``. That last slot is a PAIR of terminal outcomes that the
    persisted queue keeps distinct (a user cancel is not a failure), so this enum has 11 members: 9 single-slot states plus
    the failed/cancelled pair. :func:`spec_slot` collapses the pair back to the 10 specification slots."""
    queued = "queued"
    validating = "validating"
    slicing = "slicing"
    tracking = "tracking"
    swapping = "swapping"
    restoring = "restoring"
    merging = "merging"
    remuxing = "remuxing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


SPEC_SLOTS: tuple[str, ...] = ("queued", "validating", "slicing", "tracking", "swapping", "restoring", "merging", "remuxing",
                               "completed", "failed/cancelled")

# Every member value (lower-cased) of CanonicalJobState (queue_manager), JobState (jobs.schema) and LifecycleState.
# Cross-enum members that share a name map identically. Judgement calls:
#   * PAUSED/paused/interrupted -> queued: nothing runs, and every resume path re-queues the job (paused->queued,
#     interrupted->queued, QueueManager.resume).
#   * cancelling -> cancelled: the user intent is final; the job only winds down.
#   * running/pausing/rendering -> swapping unless the progress stage names slicing/restoring/merging.
#   * compiling_trt/retrying/analyzing -> validating: preparation before frame work starts.
_SPEC_BY_VALUE: dict[str, SpecState] = {
    # CanonicalJobState
    "queued": SpecState.queued, "preflight": SpecState.validating, "analyzing": SpecState.validating,
    "prescanning": SpecState.tracking, "prepassing": SpecState.tracking, "rendering": SpecState.swapping,
    "muxing": SpecState.remuxing, "paused": SpecState.queued, "completed": SpecState.completed,
    "failed": SpecState.failed, "retrying": SpecState.validating, "cancelled": SpecState.cancelled,
    # JobState (jobs.schema); queued/paused/cancelled/failed shared above
    "preparing": SpecState.validating, "running": SpecState.swapping, "pausing": SpecState.swapping,
    "cancelling": SpecState.cancelled, "succeeded": SpecState.completed, "interrupted": SpecState.queued,
    # LifecycleState; the remaining members are shared above
    "compiling_trt": SpecState.validating, "tracking_3d": SpecState.tracking, "muxing_audio": SpecState.remuxing,
}
_STAGE_HINT_STATES = frozenset({"rendering", "running", "pausing", "prescanning", "prepassing", "tracking_3d"})
_STAGE_HINTS: tuple[tuple[SpecState, tuple[str, ...]], ...] = (
    (SpecState.slicing, ("slic", "split", "gop")),
    (SpecState.merging, ("merg", "concat", "join", "stitch")),
    (SpecState.restoring, ("restor", "enhanc", "gfpgan", "gpen", "codeformer", "blend")),
)


def spec_state_of(state: object, stage: str | None = None) -> SpecState:
    """Total mapping from any CanonicalJobState / JobState / LifecycleState member (enum or its string value) to the spec states.

    ``stage`` (the progress stage text) refines only the frame-work states, where the persisted vocabulary cannot tell
    slicing / restoring / merging apart from swapping. Unknown values raise ``ValueError`` (fail closed)."""
    key = str(getattr(state, "value", state)).strip().lower()
    try:
        base = _SPEC_BY_VALUE[key]
    except KeyError:
        raise ValueError(f"Unrecognized job state '{state}'") from None
    if stage and key in _STAGE_HINT_STATES:
        normalized = re.sub(r"[\s\-]+", "_", stage.lower())
        for target, markers in _STAGE_HINTS:
            if any(marker in normalized for marker in markers):
                return target
    return base


def spec_slot(state: SpecState) -> str:
    """The specification's 10-slot label for a :class:`SpecState` (failed and cancelled share the last slot)."""
    return "failed/cancelled" if state in (SpecState.failed, SpecState.cancelled) else state.value
