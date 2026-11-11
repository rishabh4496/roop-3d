"""Measured three-stage stream telemetry and preserved reference-job status."""
from __future__ import annotations

import time
import threading
from typing import Callable

from roop_3d.core.hardware import Hardware
from roop_3d.jobs.schema import JobState
from roop_3d.server.ws import (EngineState, EventBase, JobProgressEvent, JobStateEvent, PipelineStatusEvent,
                               StageId, StageStatus, TrackingTelemetryEvent)
from roop_3d.telemetry.pinokio_terminal import HUD
from roop_3d.telemetry.trt_heartbeat import TrtCompilationHeartbeat
from roop_3d.tracking.telemetry import TrackingTelemetryPublisher

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
                              "engine_precision": None, "engine_since": None, "engine_batch_profile": None}
        self._tracking: dict | None = None
        # While an engine compiles: a 1 Hz line in the Pinokio terminal and `trt_compilation_progress` frames on the socket.
        # ONNX Runtime holds the GIL for the whole build, so the terminal line comes from a sidecar process (see trt_heartbeat).
        self._heartbeat = TrtCompilationHeartbeat(beat=HUD.trt_beat, done=HUD.trt_done, publish=publish, spawn_terminal=HUD.spawn_trt_sidecar)

    def snapshot(self) -> PipelineStatusEvent:
        with self._lock:
            return PipelineStatusEvent(sequence=self._sequence(), **self._state, **self._engine)

    # -- head / gaze tracking telemetry ------------------------------------------------
    def tracking_frame(self, payload: dict) -> None:
        """Sink for ``TrackingTelemetryPublisher``: fan the frame out, then remember it for clients that connect later.

        Raises (and remembers nothing) if the payload is not a valid `tracking.telemetry` frame."""
        self._publish(payload)
        with self._lock:
            self._tracking = payload

    def trt_snapshot(self) -> dict | None:
        """The latest heartbeat frame while a build has been running for at least one interval, else None."""
        return self._heartbeat.snapshot()

    def tracking_snapshot(self) -> TrackingTelemetryEvent | None:
        with self._lock:
            return None if self._tracking is None else TrackingTelemetryEvent.model_validate(self._tracking)

    def tracking_publisher(self, **kwargs) -> TrackingTelemetryPublisher:
        """A publisher bound to this tracker: ``VideoHeadTracker3D(on_frame=tracker.tracking_publisher(job_id=...))``."""
        return TrackingTelemetryPublisher(self.tracking_frame, **kwargs)

    def _emit(self) -> None:
        self._publish(self.snapshot().model_dump(mode="json", exclude={"sequence", "timestamp"}))

    def render_event(self, job_id: str, value: dict) -> None:
        stage, status = value["stage"], value["state"]
        if stage not in STAGE_ORDER or status not in ("RUNNING", "DONE", "RESTORED", "FAILED"):
            raise ValueError("Invalid render-stage event")
        with self._lock:
            if self._state["job_id"] not in (None, job_id):
                return
            if stage == STAGE_ORDER[0] and status in ("RUNNING", "RESTORED"):
                self._state["stages"] = _idle()
            self._state.update(job_id=job_id, stage=stage)
            self._state["stages"] = {**self._state["stages"], stage: status}
            for name in ("active_ranges_count", "verified_target_frames", "skipped_frames_pct"):
                if name in value:
                    self._state[name] = value[name]
        self._emit()

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
                             active_ranges_count=None, verified_target_frames=None, skipped_frames_pct=None,
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
                   precision: str | None = None, since: float | None = None,
                   batches: list[int] | tuple[int, ...] | None = None, opt_shape: object = None) -> None:
        with self._lock:
            previous_since = self._engine["engine_since"]
            if state == "IDLE":
                self._engine = {"engine_state": "IDLE", "engine_label": None, "engine_model": None,
                                "engine_precision": None, "engine_since": None, "engine_batch_profile": None}
            else:
                if since is None:
                    since = time.time() if state != self._engine["engine_state"] or previous_since is None else previous_since
                self._engine = {"engine_state": state, "engine_label": label, "engine_model": model,
                                "engine_precision": precision, "engine_since": since,
                                "engine_batch_profile": list(batches) if batches else None}
        self._emit()
        # After the status frame, so a client always sees `COMPILING_TRT` before its first heartbeat.
        if state == "COMPILING_TRT":
            self._heartbeat.begin(model or (label or "engine"), precision, opt_shape)
        else:
            self._heartbeat.end("compile finished; canary running" if state == "RUNNING_CANARY" else "ended")

    def compilation_event(self, payload: dict) -> None:
        """Strict builder adapter shared by both WebSocket streams."""
        self.set_engine(payload["engine_state"], label=payload["engine_label"], model=payload["engine_model"],
                        precision=payload["engine_precision"], since=payload["engine_since"],
                        batches=payload.get("engine_batch_profile"), opt_shape=payload.get("engine_opt_shape"))

    def engine_observer(self, state: str, record: dict, precision: str) -> None:
        """`TensorRTBuilder(engine_observer=...)` adapter."""
        self.set_engine(state, label=f"{record.get('name')} {precision}", model=record.get("name"), precision=precision)
