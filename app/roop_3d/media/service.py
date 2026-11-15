"""Bounded local PyAV decoding; remote protocols and external references disabled."""
from __future__ import annotations

import base64
from contextlib import closing, contextmanager
from dataclasses import replace
from fractions import Fraction
import json
import math
from pathlib import Path
import time
import threading
from typing import Callable, Iterator
import uuid

import av
import cv2
import numpy as np

from roop_3d.audit.isolation import sha256_file, safe_path
from roop_3d.core.errors import DomainError
from roop_3d.core.pose import as_legacy_triplet, as_pose3d, pose_cell
from roop_3d.core.settings import Settings, runtime_path
from roop_3d.core.storage import atomic_json
from roop_3d.media.detector import Detector
from roop_3d.media.frame_service import VideoFrameService
from roop_3d.engine.detection.face_tracker import FaceTracker
from roop_3d.engine.detection.track_policy import TrackPolicy
from roop_3d.identity.consolidation import TrackletRecorder, consolidate
from roop_3d.identity.report import IdentityReportCollector, write_report
from roop_3d.color.description import WorkingColor, working_color
from roop_3d.media.probe import open_media, probe
from roop_3d.media.timestamps import Timestamp
from roop_3d.media.video_reader import VideoReader
from roop_3d.media.adaptive_sampling import AdaptiveSampler, FaceSnapshot
from roop_3d.media.schema import (AudioTrack, BoundingBox, FaceInspection, FaceTrack, InspectRequest, MediaProbe, PoseRange,
                                  SubtitleTrack, TrackObservation)
from roop_3d.models.catalog import ModelCatalog

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts"}
MAX_TRACK_OBSERVATIONS = 400


def _observation(frame: int, at: float, box, points, pose, score: float, width: int, height: int) -> TrackObservation:
    """A sampled detection normalised to the frame, clamped so the contract (and the UI overlay) never sees off-frame values."""
    def unit(value: float, extent: int) -> float:
        return float(min(1.0, max(0.0, value / extent)))
    x1, y1, x2, y2 = (float(v) for v in box)
    bbox = (unit(x1, width), unit(y1, height), max(unit(x2, width), unit(x1, width)), max(unit(y2, height), unit(y1, height)))
    return TrackObservation(
        frame=frame, timestamp_seconds=at, bbox=bbox,
        landmarks5=tuple((unit(float(p[0]), width), unit(float(p[1]), height)) for p in np.asarray(points).reshape(-1, 2)[:5]),
        yaw_degrees=float(pose[0]), pitch_degrees=float(pose[1]), roll_degrees=float(pose[2]),
        confidence=float(min(1.0, max(0.0, score))))


def thin_observations(items: list[TrackObservation], limit: int = MAX_TRACK_OBSERVATIONS) -> tuple[TrackObservation, ...]:
    """At most ``limit`` observations: evenly spread over time, always keeping the pose extremes (turns must stay visible)."""
    if len(items) <= limit:
        return tuple(items)
    keep = {round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)}
    for key in (lambda o: o.yaw_degrees, lambda o: -o.yaw_degrees, lambda o: o.pitch_degrees, lambda o: -o.pitch_degrees):
        keep.add(max(range(len(items)), key=lambda i: key(items[i])))
    return tuple(items[i] for i in sorted(keep))


def merge_track_records(records: dict[int, dict], canonical_of) -> tuple[dict[int, dict], dict[int, tuple[int, ...]]]:
    """Re-key per-tracker-ID inspection records to canonical IDs (see ``identity.consolidation``).

    Tracker IDs of one person fold into one record (first/last frame, pose range, counts, observations in frame order);
    IDs the consolidation discarded as noise or face-part artifacts are dropped. Returns the records and, per canonical
    ID, the tracker IDs it absorbed."""
    merged: dict[int, dict] = {}
    sources: dict[int, list[int]] = {}
    for raw in sorted(records):
        canonical = canonical_of.get(raw)
        if canonical is None:
            continue
        item = records[raw]
        sources.setdefault(canonical, []).append(raw)
        base = merged.get(canonical)
        if base is None:
            merged[canonical] = {**item, "pose_min": list(item["pose_min"]), "pose_max": list(item["pose_max"]),
                                 "observations": list(item["observations"])}
            continue
        if item["first"] < base["first"]:
            for key in ("first", "first_at", "box", "thumbnail", "scene_id"):
                base[key] = item[key]
        if item["last"] > base["last"]:
            base["last"], base["last_at"] = item["last"], item["last_at"]
        base["count"] += item["count"]
        base["pose_min"] = np.minimum(base["pose_min"], item["pose_min"]).tolist()
        base["pose_max"] = np.maximum(base["pose_max"], item["pose_max"]).tolist()
        base["observations"] += item["observations"]
    for base in merged.values():
        base["observations"].sort(key=lambda observation: observation.frame)
    return merged, {key: tuple(value) for key, value in sources.items()}



def merge_observation(items: tuple[TrackObservation, ...], new: TrackObservation) -> tuple[TrackObservation, ...]:
    """``new`` replaces any sample of the same frame, keeping the list time-ordered and bounded."""
    merged = [item for item in items if item.frame != new.frame] + [new]
    return thin_observations(sorted(merged, key=lambda item: item.frame))


def bbox_norm_of(box, width: int, height: int) -> tuple[float, float, float, float]:
    return (float(box[0]) / width, float(box[1]) / height, float(box[2]) / width, float(box[3]) / height)


def resolve_inspection_track(tracks, frame_index: int, box_norm, fps: float, *, min_iou: float = 0.4) -> str | None:
    """The inspection track whose own sampled detection nearest to ``frame_index`` overlaps ``box_norm``.

    One id domain: a face found while scrubbing keeps the track id the scan gave the same person at that moment instead of
    a registry-local ``track_NNNN`` that merely collides by number. Only detections within half a second count, and only an
    unambiguous best overlap wins; tracks that carry no observations (older scans) fall back to their first-seen box."""
    window = max(3, round(fps * 0.5))
    best: tuple[float, str] | None = None
    for track in tracks:
        samples = [o for o in track.observations if abs(o.frame - frame_index) <= window]
        if samples:
            nearest = min(abs(o.frame - frame_index) for o in samples)
            candidates = [o.bbox for o in samples if abs(o.frame - frame_index) == nearest]
        elif track.first_frame == frame_index:
            box = track.bounding_box
            candidates = [(box.x1, box.y1, box.x2, box.y2)]
        else:
            continue
        for candidate in candidates:
            score = _iou(box_norm, candidate)
            if score >= min_iou and (best is None or score > best[0]):
                best = (score, track.track_id)
    return best[1] if best else None


def _iou(a, b) -> float:
    width, height = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-12)


class MediaService:
    def __init__(self, root: Path, catalog: ModelCatalog, settings: Callable[[], Settings] = Settings):
        self.root = root
        # Inspection and the target-lock keyframe scan analyse every frame of the video next to a running job: on the CPU the
        # RetinaFace + ArcFace + 1k3d68 stack costs ~450 ms per face-frame and pegs every core.  On a >=10 GiB card they run
        # on CUDA (same output, ~10x faster); otherwise, or when CUDA cannot start, the CPU path is used unchanged.
        from roop_3d.engine.runtime import inspection_session_factory
        gpu_factory = inspection_session_factory(catalog, settings)
        # Single-pass detector wrapped by staged scale/tile/roll recovery (frames the primary pass already solves cost nothing extra).
        from roop_3d.engine.detection.robust_detector import RobustDetector, dense_validator
        self.base_detector = Detector(catalog, gpu_factory, model_id=lambda: settings().detector_model_id)
        self.detector = RobustDetector(self.base_detector)
        self.settings = settings
        self._color_cache: dict = {}
        from roop_3d.identity.target_lock import TargetLockStore
        self.target_locks = TargetLockStore()
        self._face_edit_lock = threading.RLock()
        # Persistent decoders + frame/detection caches for timeline scrubbing (see media/frame_service.py).
        from roop_3d.faceset.recognizer import Recognizer
        from roop_3d.faceset_loader import LandmarkEngine
        from roop_3d.media.face_analysis import FaceAnalyzer
        self.analyzer = FaceAnalyzer(self.detector, Recognizer(catalog, gpu_factory),
                                     LandmarkEngine(root, session_factory=gpu_factory))
        self.detector.validator = dense_validator(self.analyzer.engine)
        self.frames = VideoFrameService(self.path, self.detector, analyzer=self.analyzer)

    def path(self, value: str) -> Path:
        try:
            path = runtime_path(value, self.root)
        except ValueError:
            raise DomainError("MEDIA_PATH_DENIED", "Media must be a local file inside the owned runtime", 422) from None
        permitted = (self.root / "roop_3d/.runtime/media",
                     self.root / "roop_3d/.runtime/media_artifacts/proxies",
                     runtime_path(self.settings().output_root, self.root))
        if not any(path.is_relative_to(parent) for parent in permitted) or path.suffix.lower() not in VIDEO_SUFFIXES:
            raise DomainError("MEDIA_PATH_DENIED", "Only video files in runtime/media or runtime/outputs are accepted", 422)
        if not path.is_file():
            raise DomainError("MEDIA_NOT_FOUND", "Local video file does not exist", 404)
        return path

    @contextmanager
    def open(self, value: str) -> Iterator[av.container.InputContainer]:
        path = self.path(value)
        with open_media(path) as container:
            yield container

    def probe(self, value: str) -> MediaProbe:
        info = probe(self.path(value))
        video = info.videos[0]
        rate = video.average_rate
        duration = video.duration * video.time_base if video.duration is not None else info.duration
        audio = tuple(AudioTrack(stream_index=item.index, codec=item.codec, sample_rate=item.sample_rate,
                                 channels=item.channels, language=item.language, time_base=str(item.time_base),
                                 channel_layout=item.layout) for item in info.audio)
        return MediaProbe(container=info.container, duration_seconds=float(duration) if duration is not None else None,
                          fps=float(rate) if rate else None, fps_rational=str(rate) if rate else None,
                          width=video.width, height=video.height, video_stream_index=video.index,
                          video_codec=video.codec, declared_frame_count=video.frame_count,
                          vfr=video.vfr, vfr_state="VERIFIED_LOCAL" if video.vfr is not None else "UNKNOWN",
                          timing_sample_frames=video.sampled_frames, timing_sample_complete=video.sample_complete,
                          audio_tracks=audio, time_base=str(video.time_base), nominal_frame_rate=str(video.nominal_rate) if video.nominal_rate else None,
                          pixel_format=video.pixel_format, color_space=video.color_space_name,
                          color_range=video.color_range, rotation_degrees=video.rotation,
                          bit_depth=video.bit_depth, chroma_format=video.chroma, **self._color_fields(video),
                          subtitle_tracks=tuple(SubtitleTrack(stream_index=s.index, codec=s.codec, kind=s.kind,
                                                              language=s.language, title=s.title) for s in info.subtitles))

    @staticmethod
    def _color_fields(video) -> dict:
        from roop_3d.color import codes
        color = video.color
        return {"color_primaries": codes.name_of("primaries", color.primaries),
                "color_transfer": codes.name_of("transfer", color.transfer),
                "color_matrix": codes.name_of("matrix", color.matrix),
                "color_range_name": codes.name_of("range", color.range), "dynamic_range": color.dynamic_range,
                "color_assumed_fields": color.assumed_fields,
                "hdr_static_metadata": color.hdr.as_dict() if color.hdr else None}

    def working_color(self, value: str, hdr_policy: str = "tonemap_sdr") -> WorkingColor:
        """What the 8-bit frames of this media mean (primaries, transfer, matrix, range) under an HDR policy."""
        path = self.path(value)
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size, hdr_policy)
        cached = self._color_cache.get(key)
        if cached is None:
            video = probe(path).videos[0]
            cached = working_color(video.color, hdr_policy, (video.width, video.height))
            if len(self._color_cache) > 64:
                self._color_cache.clear()
            self._color_cache[key] = cached
        return cached

    def frame(self, value: str, timestamp: float | None, frame_index: int | None) -> np.ndarray:
        if (timestamp is None) == (frame_index is None):
            raise DomainError("FRAME_SELECTOR_INVALID", "Specify exactly one timestamp or frame_index", 422)
        if timestamp is not None and (not math.isfinite(timestamp) or not 0 <= timestamp <= 86400):
            raise DomainError("FRAME_SELECTOR_INVALID", "Timestamp must be finite and within one day", 422)
        if frame_index is not None and not 0 <= frame_index < 10000:
            raise DomainError("FRAME_SELECTOR_INVALID", "Frame index must be between 0 and 9999", 422)
        path = self.path(value)
        reader = VideoReader(path)
        if timestamp is not None:
            with self.open(value) as container:
                stream = container.streams.video[0]
                at = Fraction(str(timestamp)) + (stream.start_time or 0) * Fraction(stream.time_base)
            return reader.at(Timestamp(at.numerator, Fraction(1, at.denominator))).image
        began = time.monotonic()
        iterator = reader.frames()
        try:
            for frame in iterator:
                if time.monotonic() - began > 15:
                    raise DomainError("DECODE_BUDGET_EXCEEDED", "Requested frame exceeds the decode budget", 422)
                if frame.index == frame_index:
                    return frame.image
        finally:
            iterator.close()
        raise DomainError("FRAME_OUT_OF_RANGE", "Requested frame is beyond the video", 416)

    def preview(self, value: str, timestamp: float | None, frame_index: int | None,
                comparison_path: str | None = None) -> bytes:
        images = [self.frame(value, timestamp, frame_index)]
        if comparison_path is not None:
            images.append(self.frame(comparison_path, timestamp, frame_index))
        images = [cv2.resize(image, (max(1, int(image.shape[1] * min(720 / image.shape[0], 1280 / image.shape[1], 1))),
                                    max(1, int(image.shape[0] * min(720 / image.shape[0], 1280 / image.shape[1], 1)))))
                  for image in images]
        if len(images) == 2:
            height = min(image.shape[0] for image in images)
            images = [cv2.resize(image, (max(1, int(image.shape[1] * height / image.shape[0])), height)) for image in images]
        success, jpeg = cv2.imencode(".jpg", np.hstack(images), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise DomainError("PREVIEW_ENCODE_FAILED", "Preview JPEG could not be encoded", 422)
        return jpeg.tobytes()

    def inspect(self, request: InspectRequest) -> FaceInspection:
        path = self.path(request.path)
        _, digest = sha256_file(path)
        records: dict[int, dict] = {}
        samples = decoded = analysed = 0
        complete = True
        stop_reason = ""
        began = time.monotonic()
        detector_record = self.detector.catalog.detector()
        recognizer_record = self.detector.catalog.recognizer()
        policy = TrackPolicy() if request.identity_lock else None
        adaptive = request.sampling == "adaptive"
        # Adaptive inspection analyses sparse frames, so the tracker's temporal filters (which assume consecutive frames)
        # are off: every serialized observation is the detector's own measurement, not a smoothed prediction.
        tracker = FaceTracker(self.detector.catalog, threshold=request.detection_threshold, detector=self.detector,
                              policy=policy, stabilize=not adaptive)
        collector = (IdentityReportCollector(policy, detection_threshold=request.detection_threshold)
                     if policy is not None else None)
        recorder = TrackletRecorder() if policy is not None else None
        inspection_id = uuid.uuid4().hex
        cuts: list[int] = []
        sampler = None
        if adaptive:
            info = self.probe(request.path)
            total = info.declared_frame_count or (round(info.duration_seconds * info.fps) if info.duration_seconds and info.fps else None)
            sampler = AdaptiveSampler(info.fps or 30.0, total)
        with self.open(request.path) as container:
            stream = container.streams.video[0]
            origin = (stream.start_time or 0) * Fraction(stream.time_base)
        try:
            with closing(VideoReader(path).frames()) as frames:
                for index, frame in enumerate(frames):
                    decoded = index + 1
                    if sampler is None or sampler.wants(index):
                        image = frame.image
                        height, width = image.shape[:2]
                        tracker.frame_number = index
                        tracked = tracker.process(image)
                        if recorder is not None:
                            recorder.observe(index, tracked)
                        if tracker.scene_cut and len(cuts) < 64:
                            cuts.append(index)
                        at = max(0.0, float(frame.timestamp.seconds - origin))
                        if collector is not None:
                            collector.observe(index, at, image, tracked)
                        snapshots = []
                        for face in tracked:
                            observation = face.observation
                            box = observation.box
                            yaw, pitch, roll = pose = as_legacy_triplet(observation.pose)
                            if face.track_id not in records:
                                if len(records) >= 64:
                                    complete = False
                                    stop_reason = "the 64-track limit was reached"
                                    break
                                x1, y1, x2, y2 = box.astype(int)
                                crop = cv2.resize(image[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)], (128, 128))
                                success, jpg = cv2.imencode(".jpg", crop)
                                if not success:
                                    raise DomainError("PREVIEW_ENCODE_FAILED", "Face thumbnail could not be encoded", 422)
                                records[face.track_id] = dict(first=index, last=index, first_at=at, last_at=at,
                                    box=box, pose_min=[yaw, pitch, roll], pose_max=[yaw, pitch, roll], count=1,
                                    scene_id=face.scene_id, thumbnail=base64.b64encode(jpg).decode("ascii"),
                                    observations=[])
                            else:
                                item = records[face.track_id]
                                item.update(last=index, last_at=at, count=item["count"] + 1)
                                item["pose_min"] = np.minimum(item["pose_min"], pose).tolist()
                                item["pose_max"] = np.maximum(item["pose_max"], pose).tolist()
                            records[face.track_id]["observations"].append(
                                _observation(index, at, box, observation.landmarks5, pose, observation.score, width, height))
                            snapshots.append(FaceSnapshot(
                                face.track_id, float(yaw), float(pitch), float(roll),
                                float((box[0] + box[2]) / 2 / width), float((box[1] + box[3]) / 2 / height),
                                float(max(box[2] - box[0], box[3] - box[1]) / max(width, height))))
                        analysed += 1
                        if sampler is not None:
                            sampler.update(index, tuple(snapshots), scene_cut=bool(tracker.scene_cut),
                                           elapsed=time.monotonic() - began, budget=request.time_budget_seconds)
                    if not adaptive and (frame.key_frame or index == 0):
                        samples += 1
                    bound = None
                    if not complete:
                        bound = stop_reason
                    elif not adaptive and not request.full_video and decoded >= request.max_decoded_frames:
                        bound = f"the {request.max_decoded_frames}-frame decode limit was reached"
                    elif not adaptive and not request.full_video and samples >= request.max_samples:
                        bound = f"the {request.max_samples}-keyframe sample limit was reached"
                    elif time.monotonic() - began > request.time_budget_seconds:
                        bound = f"the {request.time_budget_seconds:g}-second time budget was exhausted"
                    if bound is not None:
                        complete, stop_reason = False, bound
                        break
        finally:
            tracker.release()
        if sha256_file(path)[1] != digest:
            raise DomainError("MEDIA_CHANGED", "Video changed during inspection")
        identity_extra: dict = {}
        identity_warnings: tuple[str, ...] = ()
        consolidation_warnings: tuple[str, ...] = ()
        sources: dict[int, tuple[int, ...]] = {}
        events = tracker.events
        if recorder is not None:
            # One person, one canonical ID: merge the tracker's IDs across turns, occlusions and cuts; drop artifacts.
            consolidation = consolidate(recorder.tracklets(), policy)
            records, sources = merge_track_records(records, consolidation.canonical_of)
            collector.consolidate(consolidation.canonical_of)
            collector.adopt_identities(consolidation.identities)
            events = [replace(event, track_id=canonical) for event in tracker.events
                      if (canonical := consolidation.canonical(event.track_id)) is not None]
            consolidation_warnings = (
                f"Global consolidation: {len(consolidation.canonical_of)} tracker ID(s) -> {consolidation.unique_ids} identit"
                f"{'y' if consolidation.unique_ids == 1 else 'ies'}; {len(consolidation.suppressed)} discarded as noise or "
                "face-part artifacts.",)
        if collector is not None:
            document = collector.finalize(inspection_id=inspection_id, target_sha256=digest, events=events,
                                          events_dropped=tracker.events_dropped, scene_cut_frames=cuts,
                                          decoded_frames=decoded, truncated=not complete)
            identity_extra = dict(tracking_policy_fingerprint=policy.fingerprint(),
                                  detection_threshold=request.detection_threshold,
                                  identity_report_sha256=write_report(self.root, document))
            identity_warnings = ("Identity lock: faces whose identity is uncertain or ambiguous are never swapped.",
                                 "Identity thresholds are uncalibrated heuristics; review the contact sheet before locking.")
        coverage_warnings: tuple[str, ...] = ()
        if not complete:
            coverage_warnings = (f"Bounded scan: stopped after {decoded} decoded frame(s) because {stop_reason}. Faces that first "
                                 "appear later are not listed; re-run with a larger time budget to traverse the whole video.",)
        elif adaptive:
            why = ", ".join(f"{name} x{count}" for name, count in sorted(sampler.reasons.items())) or "stable scene"
            coverage_warnings = (f"Adaptive scan analysed {analysed} of {decoded} decoded frame(s) across the whole video "
                                 f"(dense around: {why}).",)
        # The bounded-scan coverage warning already states that later appearances are unreviewed.
        result = FaceInspection(inspection_id=inspection_id, target_sha256=digest,
            detector_model_id=detector_record["model_id"], detector_sha256=detector_record["sha256"],
            recognizer_model_id=recognizer_record["model_id"], recognizer_sha256=recognizer_record["sha256"],
            decoded_frames=decoded, sampled_frames=analysed if adaptive else samples, truncated=not complete,
            tracking_method="identity_lock_v1" if collector is not None else "iou_arcface_hungarian",
            scene_cut_frames=tuple(cuts), **identity_extra,
            tracks=tuple(FaceTrack(track_id=f"track_{identity:04d}", persistent_id=identity,
                source_track_ids=sources.get(identity, ()),
                scene_id=item["scene_id"], landmark_count=106, first_frame=item["first"], last_frame=item["last"],
                first_timestamp_seconds=item["first_at"], last_timestamp_seconds=item["last_at"],
                bounding_box=BoundingBox(**dict(zip(("x1", "y1", "x2", "y2"), item["box"]))),
                pose_range=PoseRange(yaw_degrees=(item["pose_min"][0], item["pose_max"][0]),
                    pitch_degrees=(item["pose_min"][1], item["pose_max"][1]),
                    roll_degrees=(item["pose_min"][2], item["pose_max"][2]),
                    method="five_point_geometric_estimate", yaw_pitch_state="CANDIDATE"),
                sampled_detections=item["count"],
                observations=thin_observations(item["observations"]),
                thumbnail_base64=(collector.best_thumbnail(identity) if collector is not None else None) or item["thumbnail"])
                for identity, item in sorted(records.items())),
            warnings=("IoU and ArcFace assignment can still split or merge tracks; review mappings.",
                      *coverage_warnings,
                      "Yaw and pitch are uncalibrated geometric estimates, not reconstructed 3D pose.",
                      ("persistent_id is the canonical identity number after global tracklet consolidation "
                       "(Face 1 = the most-observed person); track_id retains the job mapping label." if recorder is not None else
                       "persistent_id is the integer engine track ID; track_id retains the job mapping label."),
                      *consolidation_warnings,
                      *identity_warnings))
        atomic_json(self.root, f"roop_3d/.runtime/inspections/{result.inspection_id}.json", result.model_dump(mode="json"))
        return result

    def detect_frame_faces(self, value: str, timestamp: float | None = None,
                           frame_index: int | None = None, threshold: float = 0.4,
                           inspection_id: str | None = None) -> dict:
        with self._face_edit_lock:
            return self._detect_frame_faces(value, timestamp, frame_index, threshold, inspection_id)

    def _detect_frame_faces(self, value: str, timestamp: float | None,
                            frame_index: int | None, threshold: float, inspection_id: str | None = None) -> dict:
        path = self.path(value)
        _, digest = sha256_file(path)

        info = self.probe(value)
        fps = info.fps or 30.0

        if timestamp is None and frame_index is None:
            timestamp = 0.0
            frame_index = 0
        elif timestamp is not None and frame_index is None:
            frame_index = int(round(timestamp * fps))
        elif frame_index is not None and timestamp is None:
            timestamp = float(frame_index / fps)

        image = self.frame(value, timestamp, None)
        scored, detector_record = self.detector.detect_scored(image, threshold)
        retained = [(box, points) for box, points, _ in scored]
        scores = [float(score) for _, _, score in scored]
        height, width = image.shape[:2]          # the decoded (rotation-corrected) frame, not the container's coded size
        recognizer_record = self.detector.catalog.recognizer()

        from roop_3d.engine.detection.dense_landmarks import geometric_pose

        inspections_dir = safe_path(self.root, "roop_3d/.runtime/inspections")
        inspections_dir.mkdir(parents=True, exist_ok=True)

        # The inspection the caller is working in (its id is the one the UI holds); without one, the newest scan of this
        # exact video. Never an arbitrary directory-order match: a stale scan would receive the new tracks.
        existing_inspection = None
        candidates = sorted(inspections_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if inspection_id:
            candidates = [p for p in candidates if p.stem == inspection_id]
        for insp_path in candidates:
            try:
                insp_data = json.loads(insp_path.read_text("utf-8"))
                if insp_data.get("target_sha256") == digest:
                    existing_inspection = FaceInspection.model_validate(insp_data)
                    break
            except Exception:
                continue
        if inspection_id and existing_inspection is None:
            raise DomainError("INSPECTION_NOT_FOUND", "The requested inspection does not exist for this video", 404)

        if not retained:
            return {
                "inspection_id": existing_inspection.inspection_id if existing_inspection else "",
                "faces": [],
                "inspection": existing_inspection.model_dump(mode="json") if existing_inspection else None,
            }

        faces_output = []
        new_tracks = list(existing_inspection.tracks) if existing_inspection else []
        from roop_3d.identity.target_lock import FaceRef
        registry = self.target_locks.registry(value)
        refs = []
        for box, points in retained:
            pose = as_pose3d(geometric_pose(points))
            embedding = self.analyzer.recognizer.embed(image, points)
            refs.append(FaceRef(embedding, pose.yaw, pose.pitch, pose.roll, box.tolist()))
        associations = registry.associate(refs, frame_index, learn=True)
        next_track = max((t.persistent_id or 0 for t in new_tracks), default=-1) + 1

        for idx, (box, points) in enumerate(retained):
            x1, y1, x2, y2 = box.astype(int)
            x1_cl = max(0, min(x1, width - 1))
            y1_cl = max(0, min(y1, height - 1))
            x2_cl = max(x1_cl + 1, min(x2, width))
            y2_cl = max(y1_cl + 1, min(y2, height))

            crop = cv2.resize(image[y1_cl:y2_cl, x1_cl:x2_cl], (128, 128))
            success, jpg = cv2.imencode(".jpg", crop)
            thumb_b64 = base64.b64encode(jpg).decode("ascii") if success else ""

            yaw, pitch, roll = geometric_pose(points)
            pose3d = as_pose3d((yaw, pitch, roll))

            assoc = associations[idx]
            if assoc.cluster_id is None or assoc.reason == "ambiguous":
                # Withheld identity: no stable track_id, no persisted track, not selectable as source/target.
                faces_output.append({
                    "id": f"uncertain_{frame_index}_{idx}",
                    "track_id": None,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "bbox": [float(x1 / width), float(y1 / height), float(x2 / width), float(y2 / height)],
                    "landmarks5": [[float(p[0] / width), float(p[1] / height)] for p in points],
                    "confidence": round(min(1.0, max(0.0, scores[idx])), 4),
                    "pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
                    "pose3d": pose3d.model_dump(), "pose_cell": pose_cell(pose3d).name,
                    "thumbnail_base64": thumb_b64,
                    "selectable": False,
                    "identity_state": "uncertain",
                    "uncertainty_reason": assoc.reason,
                    "cluster_match": {"score": round(assoc.score, 4), "margin": round(assoc.margin, 4), "reason": assoc.reason},
                })
                continue
            cluster = registry.clusters[assoc.cluster_id]
            if cluster.track_id is None:
                # Reuse an inspection track only with an observation at this frame.
                match = resolve_inspection_track(new_tracks, frame_index, bbox_norm_of(box, width, height), fps)
                cluster.track_id = match or f'track_{next_track:04d}'
                if match is None:
                    next_track += 1
            track_id = cluster.track_id
            existing = next((t for t in new_tracks if t.track_id == track_id), None)
            bbox_norm = [float(x1 / width), float(y1 / height), float(x2 / width), float(y2 / height)]
            landmarks5_norm = [[float(p[0] / width), float(p[1] / height)] for p in points]

            face_item = {
                "id": f"{track_id}_{frame_index}",
                "track_id": track_id,
                "frame_index": frame_index,
                "timestamp": timestamp,
                "bbox": bbox_norm,
                "landmarks5": landmarks5_norm,
                "confidence": round(min(1.0, max(0.0, scores[idx])), 4),
                "pose": {"yaw": yaw, "pitch": pitch, "roll": roll},   # legacy frame (pitch + = down), kept for existing clients
                "pose3d": pose3d.model_dump(), "pose_cell": pose_cell(pose3d).name,   # standard frame (pitch + = up)
                "thumbnail_base64": thumb_b64,
            }
            faces_output.append(face_item)
            observation = _observation(frame_index, timestamp, box, points, (yaw, pitch, roll), scores[idx], width, height)

            track_obj = FaceTrack(
                track_id=track_id,
                persistent_id=existing.persistent_id if existing else int(track_id.removeprefix('track_')),
                scene_id=0,
                landmark_count=5,
                first_frame=frame_index,
                last_frame=frame_index,
                first_timestamp_seconds=timestamp,
                last_timestamp_seconds=timestamp,
                bounding_box=BoundingBox(x1=float(x1_cl), y1=float(y1_cl), x2=float(x2_cl), y2=float(y2_cl)),
                pose_range=PoseRange(
                    yaw_degrees=(yaw, yaw),
                    pitch_degrees=(pitch, pitch),
                    roll_degrees=(roll, roll),
                    method="five_point_geometric_estimate",
                    yaw_pitch_state="CANDIDATE",
                ),
                thumbnail_base64=thumb_b64,
                sampled_detections=1,
                observations=(observation,),
            )
            if existing is None:
                new_tracks.append(track_obj)
            else:
                def widen(current, value):
                    # A track whose pose was never measured has no range yet: this observation starts it.
                    return (value, value) if current is None else (min(current[0], value), max(current[1], value))
                pose_range = existing.pose_range.model_copy(update={
                    'yaw_degrees': widen(existing.pose_range.yaw_degrees, yaw),
                    'pitch_degrees': widen(existing.pose_range.pitch_degrees, pitch),
                    'roll_degrees': widen(existing.pose_range.roll_degrees, roll),
                })
                new_tracks[new_tracks.index(existing)] = existing.model_copy(update={
                    'first_frame': min(existing.first_frame, frame_index), 'last_frame': max(existing.last_frame, frame_index),
                    'first_timestamp_seconds': min(existing.first_timestamp_seconds, timestamp),
                    'last_timestamp_seconds': max(existing.last_timestamp_seconds, timestamp),
                    'sampled_detections': len(cluster.seen), 'pose_range': pose_range,
                    'observations': merge_observation(existing.observations, observation),
                })

        inspection_id = existing_inspection.inspection_id if existing_inspection else uuid.uuid4().hex
        inspection_obj = FaceInspection(
            inspection_id=inspection_id,
            target_sha256=digest,
            detector_model_id=detector_record["model_id"],
            detector_sha256=detector_record["sha256"],
            recognizer_model_id=recognizer_record["model_id"],
            recognizer_sha256=recognizer_record["sha256"],
            decoded_frames=max(1, frame_index + 1, existing_inspection.decoded_frames if existing_inspection else 0),
            sampled_frames=max(1, existing_inspection.sampled_frames if existing_inspection else 0),
            truncated=False,
            tracking_method="iou_arcface_hungarian",
            scene_cut_frames=(),
            tracks=tuple(new_tracks),
            warnings=("Target face selected directly from frame preview.",),
        )
        atomic_json(self.root, f"roop_3d/.runtime/inspections/{inspection_id}.json",
                    inspection_obj.model_dump(mode="json"))

        return {
            "inspection_id": inspection_id,
            "faces": faces_output,
            "inspection": inspection_obj.model_dump(mode="json"),
        }

    def segment_boundaries(self, path_str: str, max_segment_duration: float = 15.0) -> list[dict]:
        target = self.path(path_str)
        probe_res = self.probe(path_str)
        total_duration = probe_res.duration_seconds or 0.0
        if total_duration <= 0.0:
            return [{"segment_index": 0, "start_seconds": 0.0, "end_seconds": None, "duration": 0.0}]

        if total_duration <= max_segment_duration:
            return [{"segment_index": 0, "start_seconds": 0.0, "end_seconds": round(total_duration, 3), "duration": round(total_duration, 3)}]

        keyframes: list[float] = [0.0]
        try:
            with self.open(path_str) as container:
                stream = container.streams.video[0]
                time_base = Fraction(stream.time_base)
                start_pts = stream.start_time or 0
                for packet in container.demux(stream):
                    if packet.is_keyframe and packet.pts is not None:
                        sec = float((packet.pts - start_pts) * time_base)
                        if sec > 0.05 and (not keyframes or sec - keyframes[-1] >= 0.5):
                            keyframes.append(sec)
        except Exception:
            pass

        if len(keyframes) <= 1:
            step = max_segment_duration
            cuts = [i * step for i in range(int(math.ceil(total_duration / step)) + 1)]
            cuts[-1] = total_duration
        else:
            cuts = [0.0]
            current_target = max_segment_duration
            while current_target < total_duration - 1.0:
                candidates = [k for k in keyframes if k > cuts[-1] + 1.0 and k <= current_target + max_segment_duration * 0.5]
                if candidates:
                    best = min(candidates, key=lambda k: abs(k - current_target))
                    cuts.append(best)
                    current_target = best + max_segment_duration
                else:
                    after = [k for k in keyframes if k > cuts[-1] + 1.0]
                    if after:
                        cuts.append(after[0])
                        current_target = after[0] + max_segment_duration
                    else:
                        break
            if cuts[-1] < total_duration:
                cuts.append(total_duration)

        segments = []
        for idx in range(len(cuts) - 1):
            start_t = round(cuts[idx], 3)
            end_t = round(cuts[idx + 1], 3)
            segments.append({
                "segment_index": idx,
                "start_seconds": start_t,
                "end_seconds": end_t,
                "duration": round(end_t - start_t, 3)
            })
        return segments

    def concat_segments(self, segment_paths: list[str], output_filename: str | None = None) -> dict:
        import subprocess
        import shutil
        if not segment_paths:
            raise DomainError("SEGMENTS_EMPTY", "No segment paths provided", 422)

        resolved_paths = [self.path(p) for p in segment_paths]
        for p in resolved_paths:
            if not p.is_file():
                raise DomainError("SEGMENT_FILE_MISSING", f"Segment file {p.name} does not exist", 404)

        out_name = output_filename or f"roop3d_concat_{uuid.uuid4().hex[:12]}.mp4"
        if not out_name.endswith(".mp4"):
            out_name += ".mp4"

        output_dir = runtime_path(self.settings().output_root, self.root)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / out_name

        token = uuid.uuid4().hex
        temp_list = self.root / f"roop_3d/outputs/{token}_list.txt"
        temp_list.parent.mkdir(parents=True, exist_ok=True)

        lines = [f"file '{p.resolve().as_posix()}'\n" for p in resolved_paths]
        temp_list.write_text("".join(lines), encoding="utf-8")

        ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"
        try:
            cmd = [
                ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
                "-i", str(temp_list), "-c", "copy", str(out_file)
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as err:
            temp_list.unlink(missing_ok=True)
            raise DomainError("CONCAT_FAILED", f"FFmpeg stream copy failed: {err.stderr[-200:] if err.stderr else 'unknown'}", 500)
        finally:
            temp_list.unlink(missing_ok=True)

        try:
            out_rel = out_file.relative_to(self.root).as_posix()
            out_probe = self.probe(out_rel)
            total_frames = out_probe.declared_frame_count or 0
        except Exception:
            out_rel = out_file.relative_to(self.root).as_posix()
            total_frames = 0

        return {
            "output_path": out_rel,
            "frames": total_frames,
            "segments_concatenated": len(segment_paths)
        }
