"""Live tracking telemetry: one small JSON frame per published tracker frame, throttled, never blocking the tracker.

The payload matches ``server.ws.TrackingTelemetryEvent`` (type ``tracking.telemetry``) and travels on
``/ws/v1/pipeline/status``. This module deliberately imports nothing from ``server``: the tracker library stays usable
without the web stack, and the server validates every payload it is handed.

Throttling: steady-state frames are coalesced to ``max_hz``. These always go out immediately, because a subscriber cannot
reconstruct them from later frames: the first frame, a shot cut, a status change (tracked / coasting / lost), an identity-lock
state change, and the end of the run. Every frame also carries cumulative counters (``shot_cuts``, ``shot_index``), so a frame
dropped by a slow subscriber never hides a cut.
"""
from __future__ import annotations

import math
import threading
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:   # the server imports this module at startup; the tracker itself (cv2, reconstruction) stays unloaded
    from roop_3d.tracking.video_tracker import TrackedHeadFrame

EVENT_TYPE = "tracking.telemetry"
Publish = Callable[[dict], None]


def _round(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


def telemetry_payload(frame: TrackedHeadFrame, *, job_id: str | None, track_id: int, state: str, shot_cuts: int,
                      adapted: bool) -> dict:
    """The wire form of one tracked frame. A quantity that was not measured is null, never a placeholder."""
    camera, gaze = frame.camera, frame.gaze
    identity = frame.identity
    return {
        "type": EVENT_TYPE, "job_id": job_id, "track_id": int(track_id), "state": state,
        "frame_index": int(frame.frame_index), "timestamp_s": _round(frame.timestamp, 4) if math.isfinite(frame.timestamp) else None,
        "status": frame.status, "reason": frame.reason, "confidence": _round(frame.confidence),
        "yaw_deg": _round(camera.yaw_deg, 2) if camera else None,
        "pitch_deg": _round(camera.pitch_deg, 2) if camera else None,
        "roll_deg": _round(camera.roll_deg, 2) if camera else None,
        "jaw_open_deg": _round(math.degrees(frame.jaw_open_rad), 2) if frame.jaw_open_rad is not None else None,
        "gaze_convergence_deg": _round(gaze.vergence_deg, 2) if gaze else None,
        "gaze_convergence_distance_m": _round(gaze.convergence_distance_m, 3) if gaze and gaze.convergence_distance_m else None,
        "gaze_source": gaze.source if gaze else None,
        "expression_energy": _round(frame.expression_energy),
        "shot_index": int(frame.shot_index), "shot_cuts": int(shot_cuts), "shot_cut": bool(frame.shot_cut),
        "cut_reason": frame.cut_reason,
        "identity_state": identity.state if identity else None,
        "identity_kind": identity.kind if identity else None,
        "identity_frames": int(identity.frames) if identity else 0,
        "lighting_key": [round(float(x), 3) for x in frame.lighting.key_direction] if frame.lighting else None,
        "adapted": bool(adapted),
    }


class TrackingTelemetryPublisher:
    """``VideoHeadTracker3D(on_frame=publisher)``; thread-safe, so a tracker thread and ``finish`` may race."""

    def __init__(self, publish: Publish, *, job_id: str | None = None, track_id: int = 0, max_hz: float = 15.0,
                 adapted: bool = False, clock: Callable[[], float] = time.monotonic):
        if not math.isfinite(max_hz) or max_hz <= 0:
            raise ValueError("max_hz must be positive")
        self._publish, self.job_id, self.track_id = publish, job_id, int(track_id)
        self.min_interval, self.adapted, self._clock = 1.0 / max_hz, bool(adapted), clock
        self._lock = threading.Lock()
        self.shot_cuts = 0
        self.seen = self.emitted = 0
        self._last_emit: float | None = None
        self._last_status: str | None = None
        self._last_identity: str | None = None
        self._last_payload: dict | None = None

    def __call__(self, frame: TrackedHeadFrame) -> None:
        with self._lock:
            self.seen += 1
            if frame.shot_cut:
                self.shot_cuts += 1
            identity = frame.identity.state if frame.identity else None
            now = self._clock()
            urgent = (self._last_emit is None or frame.shot_cut or frame.status != self._last_status
                      or identity != self._last_identity)
            if not urgent and now - self._last_emit < self.min_interval:
                return
            payload = telemetry_payload(frame, job_id=self.job_id, track_id=self.track_id, state="tracking",
                                        shot_cuts=self.shot_cuts, adapted=self.adapted)
            self._last_emit, self._last_status, self._last_identity = now, frame.status, identity
            self._last_payload = payload
            self.emitted += 1
        self._publish(payload)

    def finish(self) -> None:
        """Tell subscribers the run ended, so a stale last frame is not mistaken for a live one."""
        with self._lock:
            if self._last_payload is None:
                return
            payload = {**self._last_payload, "state": "ended", "shot_cut": False, "cut_reason": None}
            self._last_payload = payload
        self._publish(payload)
