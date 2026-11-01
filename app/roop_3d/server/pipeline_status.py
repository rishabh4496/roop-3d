"""Pipeline stage telemetry derived from job events and engine-build notifications.

The reference pipeline has a single measured stage (`render`), reported as
STAGE_3_MAIN_SWAP. Stages 1 and 2 (sparse prescan / active prepass) are not
implemented, so a job marks them SKIPPED. Quantities that nothing measures
(active ranges, verified target frames, skipped-frame percentage) stay None;
nothing here invents a figure.
"""
from __future__ import annotations

from datetime import datetime, timezone
import threading
from typing import Callable

from roop_3d.core.hardware import Hardware
from roop_3d.jobs.schema import JobState
from roop_3d.server.ws import (EngineState, EventBase, JobProgressEvent, JobStateEvent, PipelineStatusEvent,
                               StageId, StageStatus)

STAGE_ORDER: tuple[StageId, ...] = ("STAGE_1_SPARSE_PRESCAN", "STAGE_2_ACTIVE_PREPASS", "STAGE_3_MAIN_SWAP")
MAIN = "STAGE_3_MAIN_SWAP"


def _idle() -> dict[StageId, StageStatus]:
    return {stage: "PENDING" for stage in STAGE_ORDER}


class TelemetrySnapshot(Hardware):
    """`GET /system/telemetry`: the hardware profile plus the current pipeline frame."""
    pipeline: PipelineStatusEvent


class PipelineTracker:
    """Tracks the most recently active job; publishes a frame whenever it changes."""

    def __init__(self, publish: Callable[[dict], None], sequence: Callable[[], int] = lambda: 0):
        self._publish, self._sequence = publish, sequence
        self._lock = threading.Lock()
        self._state: dict = {"job_id": None, "stage": None, "stages": _idle(), "stage_progress_pct": None,
                             "eta_seconds": None, "fps": 0.0}
        self._engine: dict = {"engine_state": "IDLE", "engine_label": None, "engine_model": None,
                              "engine_precision": None, "engine_since": None}

    def snapshot(self) -> PipelineStatusEvent:
        with self._lock:
            return PipelineStatusEvent(sequence=self._sequence(), **self._state, **self._engine)

    def _emit(self) -> None:
        self._publish(self.snapshot().model_dump(mode="json", exclude={"sequence", "timestamp"}))

    # -- job events -----------------------------------------------------------------
    def observe(self, event: EventBase) -> None:
        if isinstance(event, JobStateEvent):
            changed = self._job_state(event)
        elif isinstance(event, JobProgressEvent):
            changed = self._job_progress(event)
        else:
            return
        if changed:
            self._emit()

    def _job_state(self, event: JobStateEvent) -> bool:
        with self._lock:
            state = self._state
            if event.state is JobState.preparing:
                state.update(job_id=event.job_id, stage=None, stage_progress_pct=0.0, eta_seconds=None, fps=0.0,
                             stages={"STAGE_1_SPARSE_PRESCAN": "SKIPPED", "STAGE_2_ACTIVE_PREPASS": "SKIPPED", MAIN: "PENDING"})
            elif state["job_id"] not in (None, event.job_id):
                return False  # another job owns the frame; only its own events move it
            elif event.state is JobState.running:
                state.update(job_id=event.job_id, stage=MAIN)
                state["stages"] = {**state["stages"], MAIN: "RUNNING"}
            elif event.state is JobState.succeeded:
                state.update(stage=None, stage_progress_pct=100.0, eta_seconds=0.0)
                state["stages"] = {**state["stages"], MAIN: "DONE"}
            elif event.state is JobState.failed:
                state.update(stage=None, eta_seconds=None)
                state["stages"] = {**state["stages"], MAIN: "FAILED"}
            elif event.state in (JobState.cancelled, JobState.interrupted):
                state.update(job_id=None, stage=None, stage_progress_pct=None, eta_seconds=None, fps=0.0, stages=_idle())
            else:  # queued, pausing, paused, cancelling: no stage change
                return False
            return True

    def _job_progress(self, event: JobProgressEvent) -> bool:
        if event.stage != "render" or not event.total_frames:
            return False
        with self._lock:
            if self._state["job_id"] not in (None, event.job_id):
                return False
            remaining = max(0, event.total_frames - event.frame)
            self._state.update(job_id=event.job_id, stage=MAIN, fps=event.fps,
                               stage_progress_pct=round(100 * event.frame / event.total_frames, 2),
                               eta_seconds=round(remaining / event.fps, 1) if event.fps > 0 else None)
            if self._state["stages"][MAIN] == "PENDING":
                self._state["stages"] = {**self._state["stages"], MAIN: "RUNNING"}
        return True

    # -- engine builds --------------------------------------------------------------
    def set_engine(self, state: EngineState, *, label: str | None = None, model: str | None = None,
                   precision: str | None = None) -> None:
        with self._lock:
            since = self._engine["engine_since"]
            if state == "IDLE":
                self._engine = {"engine_state": "IDLE", "engine_label": None, "engine_model": None,
                                "engine_precision": None, "engine_since": None}
            else:
                if state != self._engine["engine_state"] or since is None:
                    since = datetime.now(timezone.utc)
                self._engine = {"engine_state": state, "engine_label": label, "engine_model": model,
                                "engine_precision": precision, "engine_since": since}
        self._emit()

    def engine_observer(self, state: str, record: dict, precision: str) -> None:
        """`TensorRTBuilder(engine_observer=...)` adapter."""
        self.set_engine(state, label=f"{record.get('name')} {precision}", model=record.get("name"), precision=precision)
