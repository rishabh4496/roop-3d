"""Monocular video head and gaze tracking (Stage 4).

Exports are resolved lazily so that importing a light module (``roop_3d.tracking.telemetry`` is imported by the API server
at startup) never drags in OpenCV and the reconstruction stack.
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "VideoHeadTracker3D": "video_tracker", "TrackerSettings": "video_tracker", "FrameEvidence": "video_tracker",
    "TrackedHeadFrame": "video_tracker", "HeadAnchors": "video_tracker", "CameraPose": "video_tracker",
    "ShotBoundaryDetector": "video_tracker",
    "SceneTrackerAdaptation": "scene_adaptation", "SceneAdaptation": "scene_adaptation",
    "AdaptationReport": "scene_adaptation", "sample_frame_indices": "scene_adaptation",
    "ShotIdentityLock": "identity_lock", "MeshSHLightingEstimator": "lighting",
    "TrackFilterBank": "filters", "SmoothedChannel": "filters", "FilterSpec": "filters",
    "TrackingTelemetryPublisher": "telemetry", "telemetry_payload": "telemetry",
    "calibrate_video": "runner", "track_video": "runner", "HeadTrackingTap": "runner",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
