"""Drive the tracker from real video: the pre-render calibration pass, a whole-video tracking run, and an opt-in pipeline tap.

Nothing here is called by the render job runner. ``calibrate_video`` and ``track_video`` are the entry points a job (or a
script) uses; ``HeadTrackingTap.observe`` is the one call a pipeline makes per frame to share the faces its own
``FaceTracker`` already found. All three are failure-isolated from the caller's point of view: a bad frame costs that frame.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from roop_3d.tracking.scene_adaptation import SceneAdaptation, SceneTrackerAdaptation
from roop_3d.tracking.video_tracker import FrameEvidence, HeadAnchors, TrackedHeadFrame, TrackerSettings, VideoHeadTracker3D

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceFrame:
    index: int
    timestamp: float
    image: np.ndarray               # upright BGR
    raw: Any = None                 # the source's own handle, for an extractor that needs it (e.g. a FrameResult)


class FrameSource(Protocol):
    total_frames: int

    def read(self, index: int) -> SourceFrame: ...


Extractor = Callable[[SourceFrame], FrameEvidence | None]


# =========================================================================== adapters over the existing media stack
class VideoFrameServiceSource:
    """``media.frame_service.VideoFrameService`` as a FrameSource (frame-accurate seeks, the decoded-frame cache)."""

    def __init__(self, service, path: str):
        self.service, self.path = service, path
        self.info = service.info(path)
        self.total_frames = int(self.info.total_frames)
        self.fps = float(self.info.fps)

    def read(self, index: int) -> SourceFrame:
        result = self.service.read(self.path, frame_index=index)
        return SourceFrame(result.frame_index, float(result.timestamp_sec), result.image, result)


class AnalyzedFaceExtractor:
    """Evidence from ``VideoFrameService.analyzed_faces`` (detector + 106-point landmarks) for ONE face per frame.

    ``crop_margin`` (e.g. 0.6) keeps only the face region (with that margin of the box size) as ``image_bgr`` and shifts the
    landmarks to match, which is exact for a translation-equivariant camera and keeps 16 calibration frames of a 4K video
    small. Leave it None when the full frame is needed (shot-boundary detection). ``pick`` chooses among several faces
    (default: the largest), so a multi-person video calibrates the dominant face unless told otherwise."""

    def __init__(self, service, path: str, *, threshold: float = 0.5, crop_margin: float | None = None,
                 pick: Callable[[Sequence], Any] | None = None):
        self.service, self.path, self.threshold, self.crop_margin = service, path, threshold, crop_margin
        self.pick = pick or (lambda faces: max(faces, key=lambda f: float((f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))))

    def __call__(self, frame: SourceFrame) -> FrameEvidence | None:
        faces = self.service.analyzed_faces(self.path, frame.raw, self.threshold)
        if not faces:
            return None
        face = self.pick(faces)
        image, origin = frame.image, np.zeros(2)
        if self.crop_margin is not None:
            x1, y1, x2, y2 = (float(v) for v in face.box[:4])
            mx, my = self.crop_margin * (x2 - x1), self.crop_margin * (y2 - y1)
            height, width = frame.image.shape[:2]
            left, top = max(0, int(x1 - mx)), max(0, int(y1 - my))
            image = frame.image[top:min(height, int(y2 + my)), left:min(width, int(x2 + mx))]
            origin = np.array((left, top), np.float64)
        shift = lambda points: None if points is None else np.asarray(points, np.float64) - origin
        return FrameEvidence(timestamp=frame.timestamp, landmarks5=shift(face.kps5), landmarks106=shift(face.kps106),
                             image_bgr=image, frame_index=frame.index)


# =========================================================================== the passes
def calibrate_video(source: FrameSource, extractor: Extractor, adaptation: SceneTrackerAdaptation | None = None, *,
                    base: HeadAnchors | None = None) -> SceneAdaptation:
    """The optional pre-render pass: sample frames across the video, extract, adapt, all inside ONE time budget.

    Extraction (decode + detect) and the adaptation maths share the budget; when it runs out the pass adapts with the frames
    it has and reports ``budget_exhausted``. A frame that fails to decode or has no face is skipped and counted."""
    adaptation = adaptation or SceneTrackerAdaptation()
    started = adaptation.clock()
    deadline = started + adaptation.budget_s
    indices = adaptation.frame_indices(source.total_frames)
    samples: list[FrameEvidence] = []
    failures = 0
    for index in indices:
        if adaptation.clock() >= deadline:
            break
        try:
            evidence = extractor(source.read(index))
        except Exception:       # an optional pass must not take the job down; the report says what was lost
            log.warning("Calibration frame %d could not be read", index, exc_info=True)
            failures += 1
            continue
        if evidence is not None:
            samples.append(evidence)
    result = adaptation.calibrate(samples, started=started, base=base, requested=len(indices))
    if failures:
        report = dataclasses.replace(result.report, warnings=(*result.report.warnings, f"{failures} calibration frame(s) failed to decode"))
        result = dataclasses.replace(result, report=report)
    return result


def track_video(source: FrameSource, extractor: Extractor, tracker: VideoHeadTracker3D, *, start: int = 0,
                stop: int | None = None, step: int = 1, should_stop: Callable[[], bool] | None = None,
                publisher=None) -> dict:
    """Track every ``step``-th frame of ``[start, stop)``. Returns ``tracker.summary()`` plus the frames visited."""
    if step < 1 or start < 0:
        raise ValueError("start must be >= 0 and step >= 1")
    stop = source.total_frames if stop is None else min(stop, source.total_frames)
    visited = 0
    try:
        for index in range(start, stop, step):
            if should_stop is not None and should_stop():
                break
            frame = source.read(index)
            evidence = extractor(frame) or FrameEvidence(frame.timestamp, None, image_bgr=frame.image, frame_index=frame.index)
            if evidence.frame_index is None:
                evidence = dataclasses.replace(evidence, frame_index=frame.index)
            tracker.process(evidence)
            visited += 1
    finally:
        if publisher is not None:
            publisher.finish()
    return {**tracker.summary(), "visited": visited}


# =========================================================================== opt-in pipeline tap
class HeadTrackingTap:
    """Per-track trackers fed from the faces a pipeline's ``FaceTracker`` already found (no second detection pass).

    Enable it with one call after ``ReferencePipeline.track_frame``::

        tap = HeadTrackingTap(publisher_factory=lambda track_id: app.state.pipeline.tracking_publisher(job_id=job_id, track_id=track_id))
        ...
        tap.observe(image, tracked.faces, tracked.frame_time, tracked.scene_cut)

    Telemetry is published for ONE track (``telemetry_track_id``, else the first track seen) so that frames of different
    people never interleave on one stream; the others are still tracked."""

    def __init__(self, *, settings: TrackerSettings | None = None, adaptation: SceneAdaptation | None = None,
                 lighting=None, publisher_factory: Callable[[int], Callable[[TrackedHeadFrame], None]] | None = None,
                 telemetry_track_id: int | None = None, track_ids: set[int] | None = None, max_tracks: int = 4):
        self.settings, self.adaptation, self.lighting = settings, adaptation, lighting
        self.publisher_factory, self.telemetry_track_id = publisher_factory, telemetry_track_id
        self.track_ids, self.max_tracks = track_ids, max_tracks
        self.trackers: dict[int, VideoHeadTracker3D] = {}
        self.publishers: dict[int, Any] = {}

    def _tracker(self, track_id: int) -> VideoHeadTracker3D | None:
        tracker = self.trackers.get(track_id)
        if tracker is not None or len(self.trackers) >= self.max_tracks:
            return tracker
        if self.telemetry_track_id is None:
            self.telemetry_track_id = track_id
        publisher = self.publisher_factory(track_id) if self.publisher_factory and track_id == self.telemetry_track_id else None
        if publisher is not None:
            self.publishers[track_id] = publisher
        tracker = self.trackers[track_id] = VideoHeadTracker3D(self.settings, adaptation=self.adaptation,
                                                               lighting=self.lighting, on_frame=publisher)
        return tracker

    def observe(self, image: np.ndarray | None, faces: Sequence, frame_time: float, scene_cut: bool) -> Mapping[int, TrackedHeadFrame]:
        results: dict[int, TrackedHeadFrame] = {}
        for face in faces:
            track_id = int(face.track_id)
            if self.track_ids is not None and track_id not in self.track_ids:
                continue
            try:
                tracker = self._tracker(track_id)
                if tracker is None:
                    continue
                observation = face.observation
                mesh = getattr(observation, "landmarks478", None)
                results[track_id] = tracker.process(FrameEvidence(
                    timestamp=float(frame_time), landmarks5=observation.landmarks5, landmarks106=observation.landmarks106,
                    mesh478=mesh if mesh is not None and np.asarray(mesh).shape[0] == 478 else None,
                    image_bgr=image, scene_cut=bool(scene_cut),
                    pose_6dof=getattr(observation, "pose_6dof", None),
                    pose_source=getattr(observation, "pose_source", None)))
            except Exception:    # the render must never fail because tracking telemetry did
                log.exception("Head tracking failed for track %s; this frame is skipped", track_id)
        return results

    def release(self) -> None:
        for publisher in self.publishers.values():
            finish = getattr(publisher, "finish", None)
            if finish is not None:
                finish()
        self.trackers.clear()
        self.publishers.clear()
