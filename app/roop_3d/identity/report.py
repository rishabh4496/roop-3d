"""Per-inspection identity report: what the contact sheet shows and what the render guard verifies.

The report is a local sidecar next to (not inside) the public ``FaceInspection``: it holds
ArcFace signatures, which are biometric templates. It never leaves the machine, is not
returned by any API (``public_tracks`` strips the vectors) and is bound to the inspection
by SHA-256 (``FaceInspection.identity_report_sha256``).
"""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.storage import atomic_json, owned
from roop_3d.engine.detection.face_tracker import TrackedFace
from roop_3d.engine.detection.track_policy import TrackPolicy

REPORT_VERSION = 1
MAX_SEGMENTS = 512
THUMBNAILS_PER_TRACK = 3
THUMBNAIL_FRAME_GAP = 12
MAX_SIGNATURE_ANCHORS = 6


def report_relative(inspection_id: str) -> str:
    return f"roop_3d/.runtime/identity/{inspection_id}.json"


def label_for(persistent_id: int) -> str:
    return f"track_{persistent_id:04d}"


class _Track:
    def __init__(self, persistent_id: int, frame: int, timestamp: float, scene_id: int):
        self.persistent_id = persistent_id
        self.first_frame = self.last_frame = frame
        self.first_time = self.last_time = timestamp
        self.segments: list[list] = []
        self.visible = self.swappable = 0
        self.reasons: Counter[str] = Counter()
        self.states: Counter[str] = Counter()
        self.confidence_sum = 0.0
        self.confidence_min = 1.0
        self.anchors: list[np.ndarray] = []
        self.centroid = np.zeros(512, np.float64)
        self.centroid_count = 0
        self.thumbnails: list[dict] = []
        self.scenes: set[int] = set()

    def add_segment(self, frame: int, timestamp: float) -> None:
        if self.segments and frame == self.segments[-1][1] + 1:
            self.segments[-1][1], self.segments[-1][3] = frame, timestamp
        elif len(self.segments) >= MAX_SEGMENTS:
            self.segments[-1][1], self.segments[-1][3] = frame, timestamp   # bounded: fold into the final range
        else:
            self.segments.append([frame, frame, timestamp, timestamp])


class IdentityReportCollector:
    """Fed once per decoded frame with the tracker output; produces the sidecar document."""

    def __init__(self, policy: TrackPolicy, *, detection_threshold: float, max_tracks: int = 64):
        self.policy, self.detection_threshold, self.max_tracks = policy, detection_threshold, max_tracks
        self.tracks: dict[int, _Track] = {}
        self.overflow = False

    # -- frame ingestion -------------------------------------------------------------------------
    def observe(self, frame_index: int, timestamp: float, image: np.ndarray, tracked: Iterable[TrackedFace]) -> None:
        for face in tracked:
            record = self.tracks.get(face.track_id)
            if record is None:
                if len(self.tracks) >= self.max_tracks:
                    self.overflow = True
                    continue
                record = self.tracks[face.track_id] = _Track(face.track_id, frame_index, timestamp, face.scene_id)
            record.last_frame, record.last_time = frame_index, timestamp
            record.visible += 1
            record.states[face.state] += 1
            record.scenes.add(face.scene_id)
            record.add_segment(frame_index, timestamp)
            record.confidence_sum += face.confidence
            record.confidence_min = min(record.confidence_min, face.confidence)
            if face.swap_allowed:
                record.swappable += 1
                self._learn_signature(record, face)
            else:
                record.reasons[face.reason or face.state] += 1
            self._consider_thumbnail(record, frame_index, timestamp, image, face)

    @staticmethod
    def _learn_signature(record: _Track, face: TrackedFace) -> None:
        embedding = np.asarray(face.observation.embedding, np.float32)
        record.centroid += embedding
        record.centroid_count += 1
        if len(record.anchors) < MAX_SIGNATURE_ANCHORS and (
                not record.anchors or max(float(np.dot(anchor, embedding)) for anchor in record.anchors) < 0.85):
            record.anchors.append(embedding.copy())

    @staticmethod
    def _quality(image: np.ndarray, face: TrackedFace) -> tuple[float, np.ndarray] | None:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in face.observation.box)
        margin_x, margin_y = (x2 - x1) * 0.15, (y2 - y1) * 0.15
        left, top = int(max(0, x1 - margin_x)), int(max(0, y1 - margin_y))
        right, bottom = int(min(width, x2 + margin_x)), int(min(height, y2 + margin_y))
        if right - left < 4 or bottom - top < 4:
            return None
        crop = image[top:bottom, left:right]
        gray = cv2.cvtColor(cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = variance / (variance + 150.0)
        size = min(1.0, (x2 - x1) / 96.0)
        yaw = abs(float(face.observation.pose[0]))
        frontal = max(0.0, 1.0 - yaw / 50.0)
        trusted = 1.0 if face.swap_allowed else 0.2
        return (0.1 + sharpness) * (0.2 + size) * (0.2 + frontal) * (0.3 + face.confidence) * trusted, crop

    def _consider_thumbnail(self, record: _Track, frame: int, timestamp: float, image: np.ndarray,
                            face: TrackedFace) -> None:
        scored = self._quality(image, face)
        if scored is None:
            return
        score, crop = scored
        neighbours = [item for item in record.thumbnails if abs(item["frame"] - frame) < THUMBNAIL_FRAME_GAP]
        if neighbours:
            if score <= neighbours[0]["score"]:
                return
            replace = neighbours[0]
        elif len(record.thumbnails) >= THUMBNAILS_PER_TRACK:
            replace = min(record.thumbnails, key=lambda item: item["score"])
            if score <= replace["score"]:
                return
        else:
            replace = None
        ok, jpeg = cv2.imencode(".jpg", cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA),
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        entry = {"frame": frame, "timestamp_seconds": round(float(timestamp), 6), "score": float(score),
                 "jpeg_base64": base64.b64encode(jpeg.tobytes()).decode("ascii")}
        if replace is not None:
            record.thumbnails.remove(replace)
        record.thumbnails.append(entry)

    def best_thumbnail(self, persistent_id: int) -> str | None:
        record = self.tracks.get(persistent_id)
        if record is None or not record.thumbnails:
            return None
        return max(record.thumbnails, key=lambda item: item["score"])["jpeg_base64"]

    # -- document --------------------------------------------------------------------------------
    @staticmethod
    def _missing(record: _Track) -> list[list[int]]:
        gaps = []
        for before, after in zip(record.segments, record.segments[1:]):
            gaps.append([before[1] + 1, after[0] - 1])
        return gaps

    def finalize(self, *, inspection_id: str, target_sha256: str, events, events_dropped: int,
                 scene_cut_frames: Iterable[int], decoded_frames: int, truncated: bool) -> dict:
        tracks = {}
        for identity in sorted(self.tracks):
            record = self.tracks[identity]
            centroid = record.centroid / max(float(np.linalg.norm(record.centroid)), 1e-12)
            tracks[label_for(identity)] = {
                "persistent_id": identity,
                "first_frame": record.first_frame, "last_frame": record.last_frame,
                "first_timestamp_seconds": round(float(record.first_time), 6),
                "last_timestamp_seconds": round(float(record.last_time), 6),
                "segments": [[s, e, round(float(st), 6), round(float(et), 6)] for s, e, st, et in record.segments],
                "missing_ranges": self._missing(record),
                "visible_frames": record.visible, "swappable_frames": record.swappable,
                "withheld_frames": record.visible - record.swappable,
                "withheld_reasons": dict(sorted(record.reasons.items())),
                "state_frames": dict(sorted(record.states.items())),
                "confidence_mean": round(record.confidence_sum / max(record.visible, 1), 4),
                "confidence_min": round(record.confidence_min, 4),
                "scene_ids": sorted(record.scenes),
                "thumbnails": sorted(({key: value for key, value in item.items() if key != "score"}
                                      for item in record.thumbnails), key=lambda item: item["frame"]),
                "signature": {
                    "evidence_frames": record.centroid_count,
                    "centroid": [round(float(value), 6) for value in (centroid if record.centroid_count else [])],
                    "anchors": [[round(float(value), 6) for value in anchor] for anchor in record.anchors],
                },
            }
        timeline = [{"frame": e.frame, "track_id": label_for(e.track_id), "state": e.state, "reason": e.reason,
                     "scene_id": e.scene_id} for e in events]
        return {"version": REPORT_VERSION, "inspection_id": inspection_id, "target_sha256": target_sha256,
                "tracking_policy": self.policy.as_dict(), "tracking_policy_fingerprint": self.policy.fingerprint(),
                "detection_threshold": self.detection_threshold, "decoded_frames": decoded_frames,
                "truncated": truncated, "track_limit_reached": self.overflow,
                "scene_cut_frames": sorted(int(frame) for frame in scene_cut_frames)[:256],
                "events": timeline[:2048], "events_dropped": events_dropped + max(0, len(timeline) - 2048),
                "tracks": tracks}


# -- storage -------------------------------------------------------------------------------------
def write_report(root: Path, document: dict) -> str:
    """Persist the sidecar and return the SHA-256 of the exact bytes on disk."""
    relative = report_relative(document["inspection_id"])
    atomic_json(root, relative, document)
    return hashlib.sha256(owned(root, relative).read_bytes()).hexdigest()


def read_report(root: Path, inspection_id: str, *, expected_sha256: str | None = None) -> dict:
    import json
    path = owned(root, report_relative(inspection_id))
    if not path.is_file():
        raise DomainError("IDENTITY_REPORT_NOT_FOUND", "This inspection has no identity report; inspect with the identity lock enabled", 422)
    data = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise DomainError("IDENTITY_REPORT_CHANGED", "The identity report no longer matches its inspection", 409)
    try:
        return json.loads(data)
    except ValueError:
        raise DomainError("IDENTITY_REPORT_INVALID", "The identity report is unreadable", 422) from None


def public_tracks(document: dict) -> list[dict]:
    """Contact-sheet view: everything except biometric vectors."""
    result = []
    for label, track in document["tracks"].items():
        view = {key: value for key, value in track.items() if key != "signature"}
        view["track_id"] = label
        view["signature_evidence_frames"] = track["signature"]["evidence_frames"]
        result.append(view)
    return result
