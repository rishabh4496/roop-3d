"""Session lifecycle: tracking -> identity gate -> true-3D head render (NOT YET WIRED).

The 2D identity swapper, 5-point crop-paste, canonical-strip pose blend, 2D TPS expression warp and the reenactment
painter's renderer were removed. ``ReferencePipeline`` still owns detection, landmarks, tracking, the identity gate, angle
trajectories and compositing of a rendered head layer; ``infer_frame`` raises ``True3DRendererUnavailable`` for any face that
would have to be rendered, instead of silently falling back to a 2D / 2.5D path.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import threading
import time
from collections.abc import Mapping
from typing import Any
import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.core.true3d_guard import True3DRendererUnavailable, require_true3d_renderer
from roop_3d.engine.detection.face_tracker import FaceObservation, FaceTracker, TrackedFace
from roop_3d.engine.runtime import RuntimeConfig, image_bgr
from roop_3d.engine.volumetric.angle_diagnostics import AngleDiagnostics
from roop_3d.engine.volumetric.presence import PoseFade, SwapPresence
from roop_3d.engine.volumetric.preprocess import landmark_crop, transform_points
from roop_3d.faceset.schemas import FaceSetDocument
from roop_3d.models.catalog import ModelCatalog
from roop_3d.engine.resources import guarded, inference, register
from roop_3d.temporal import smoothstep


@dataclass(frozen=True)
class StageLatency:
    detection_ms: float
    swap_ms: float
    retarget_ms: float
    total_ms: float
    projection_ms: float = 0.0


@dataclass(frozen=True)
class TrackedFrame:
    """Stage 1 output: what the tracker said about one frame, snapshotted so Stage 2 may lag the tracker.

    The tracker is stateful and runs ahead of the swap stage in the asynchronous pipeline, so everything Stage 2 would
    otherwise read back from ``tracker`` (scene cut, live track ids, frame time) travels with the frame instead.
    """
    faces: tuple
    scene_cut: bool
    active_ids: frozenset
    frame_time: float
    detection_ms: float


@dataclass(frozen=True)
class Replacement:
    track_id: int
    rendered_frame: np.ndarray
    coverage: np.ndarray
    skin_mask: np.ndarray
    hair_mask: np.ndarray
    normals: np.ndarray
    pipeline_stages: tuple[str, ...]
    full_head: bool
    occluder: np.ndarray | None = None



@dataclass(frozen=True)
class Withheld:
    """A selected face that was deliberately left untouched because its identity was not verified."""
    track_id: int
    reason: str
    state: str
    confidence: float


@dataclass(frozen=True)
class FrameResult:
    observations: tuple[TrackedFace, ...]
    replacements: tuple[Replacement, ...]
    latency: StageLatency
    scene_cut: bool
    withheld: tuple[Withheld, ...] = ()


class ReferencePipeline:
    def __init__(self, catalog: ModelCatalog | None = None, config: RuntimeConfig = RuntimeConfig()):
        self.catalog, self.config = catalog or ModelCatalog(), config
        self.tracker: FaceTracker | None = None
        self.angle_diagnostics = AngleDiagnostics()
        self.dense_enhancer_calls = 0
        self._frame_processor = None
        self.last_frame_report = None
        self.selected_ids: frozenset[int] | None = None
        self.locked_targets = None
        self.sources_by_track: dict[int, FaceSetDocument] = {}
        self.identity_guard = None
        self.prepared = False
        self.lock = threading.RLock()
        # Stages 1 (tracking) and 3 (composite) have their own locks so they can overlap the render stage.
        # Order whenever more than one is needed: track_lock -> lock -> composite_lock.
        self.track_lock = threading.RLock()
        self.composite_lock = threading.RLock()
        self._frame_context: TrackedFrame | None = None
        self.frames_tracked = 0
        self.last_latency: StageLatency | None = None
        self.frames_processed = 0
        self.default_source: FaceSetDocument | None = None
        self.identity_store = None
        self.neural_face = None          # the dense-correspondence swap engine was removed; always None
        self.full_head_engines = {}
        self.head_landmarks = None
        self.angle_trajectories: dict[int, tuple[Any, ...]] = {}
        self.angle_frame_index: int | None = None
        self.pose_hysteresis_degrees = Settings.model_fields["pose_hysteresis_degrees"].default
        self._roll_latched: dict[int, bool] = {}
        self._presence = SwapPresence()
        self._pose_fade = PoseFade()
        register(self, "release_session")

    def prepare_session(self, source: FaceSetDocument, *, track_ids: set[int] | None = None,
                        sources_by_track: Mapping[int, FaceSetDocument] | None = None,
                        track_policy=None, detection_threshold: float = 0.5, identity_store=None,
                        locked_targets=None) -> dict:
        """Prepare tracking and identity gating; rendering requires the wired true-3D renderer."""
        with self.lock:
            if track_ids is not None and (
                    len(track_ids) > 32 or any(type(i) is not int or i < 0 for i in track_ids)):
                raise ValueError("Invalid session selection/policy")
            mapped_sources = dict(sources_by_track or {})
            if (len(mapped_sources) > 32 or any(type(i) is not int or i < 0 for i in mapped_sources)
                    or track_ids is not None and not set(mapped_sources).issubset(track_ids)):
                raise ValueError("Mapped FaceSets must use selected nonnegative track IDs")
            self.release_session()
            try:
                self.tracker = FaceTracker(self.catalog, config=self.config, threshold=detection_threshold,
                                           policy=track_policy)
                self.selected_ids = frozenset(track_ids) if track_ids is not None else None
                self.locked_targets = locked_targets
                self.sources_by_track = mapped_sources
                self.default_source = source
                self.identity_store = identity_store
                self.prepared = True
                return self.capability()
            except BaseException:
                self.release_session()
                raise

    def _document_for(self, track_id: int) -> FaceSetDocument | None:
        return self.sources_by_track.get(track_id, self.default_source)

    def _frame_time(self) -> float:
        if self._frame_context is not None:
            return self._frame_context.frame_time
        seconds = getattr(self.tracker, "timestamp_seconds", None) if self.tracker is not None else None
        return float(seconds) if seconds is not None and np.isfinite(seconds) else self.frames_processed / 30.0

    @staticmethod
    def _face_sharpness(image: np.ndarray, box: np.ndarray) -> float | None:
        x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
        h, w = image.shape[:2]
        region = image[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
        if region.size == 0 or min(region.shape[:2]) < 8:
            return None
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA), cv2.CV_64F).var())

    def set_target_registry(self, registry) -> None:
        self.target_registry = registry
        # Locked canonical targets are tracked by the identity tracker; the clustering registry would re-number them.
        if self.tracker is not None and self.locked_targets is None:
            self.tracker.target_registry = registry

    def _bind_locked_targets(self, faces, fallback_frame: int):
        """Re-label the tracker's faces with the locked canonical target IDs (see ``identity.locked_target``)."""
        binder = self.locked_targets
        if binder is None:
            return faces
        return binder.bind(faces, self.angle_frame_index if self.angle_frame_index is not None else fallback_frame)

    def _active_ids(self) -> frozenset[int]:
        alive = self.tracker.active_ids
        return frozenset(alive) if self.locked_targets is None else self.locked_targets.canonical_ids(alive)

    def _persistent_target_faces(self, faces):
        registry = getattr(self, "target_registry", None)
        if (registry is None or not faces or self.locked_targets is not None
                or getattr(self.tracker, "target_registry", None) is registry):
            return faces
        from roop_3d.identity.target_lock import FaceRef
        from roop_3d.core.pose import as_pose3d
        refs = []
        for tracked in faces:
            face = tracked.observation
            pose = as_pose3d(face.pose)
            refs.append(FaceRef(face.embedding, pose.yaw, pose.pitch, pose.roll, list(face.box)))
        frame = self.angle_frame_index
        if frame is None:
            frame = max(0, self.tracker.frame_index - 1)
        associations = registry.associate(refs, frame, learn=False)
        result = []
        with registry.lock:
            for tracked, assoc in zip(faces, associations):
                if assoc.cluster_id is None:
                    continue
                cluster = registry.clusters[assoc.cluster_id]
                identity = int(cluster.track_id.removeprefix("track_"))
                result.append(replace(tracked, track_id=identity))
        return tuple(result)

    def _target_reference(self, tracked):
        registry = getattr(self, "target_registry", None)
        if registry is None or self.locked_targets is not None:
            return None
        from roop_3d.core.pose import as_pose3d
        pose = as_pose3d(tracked.observation.pose)
        with registry.lock:
            cluster = next((c for c in registry.clusters.values()
                            if c.track_id == f"track_{tracked.track_id:04d}"), None)
            return cluster.reference_for_pose(pose.yaw, pose.pitch) if cluster and cluster.locked else None

    @staticmethod
    def _bank_landmarks(reference, box):
        """Move pose-matched measured landmarks to the current frame's box for a missing-landmark fallback."""
        if reference is None or reference.landmarks3d is None:
            return None
        points = reference.landmarks3d.copy()
        before, after = np.asarray(reference.bbox), np.asarray(box)
        scale = (after[2:] - after[:2]) / np.maximum(before[2:] - before[:2], 1e-6)
        points[:, :2] = (points[:, :2] - before[:2]) * scale + after[:2]
        if points.shape[1] == 3:
            points[:, 2] *= float(scale.mean())
        return points

    def set_identity_guard(self, guard) -> None:
        """Install the render-time identity lock check (see roop_3d.identity.guard)."""
        with self.lock:
            self.identity_guard = guard

    def configure_temporal(self, settings) -> None:
        from roop_3d.temporal import TrackTemporalStabilizer
        config = dict(min_cutoff=settings.temporal_min_cutoff, beta=settings.temporal_beta,
                      derivative_cutoff=settings.temporal_d_cutoff)
        if self.tracker is not None:
            self.tracker.stabilizer = TrackTemporalStabilizer(**config)
        self.pose_hysteresis_degrees = settings.pose_hysteresis_degrees

    def set_frame_timestamp(self, seconds: float) -> None:
        if not np.isfinite(seconds):
            raise ValueError("Frame timestamp must be finite")
        if self.tracker is not None:
            self.tracker.timestamp_seconds = seconds

    def set_angle_trajectories(self, trajectories: Mapping[int, tuple[Any, ...]]) -> None:
        """Install the Stage 13 scan before rendering begins; keys are replayed tracker IDs."""
        if any(type(track_id) is not int or track_id < 0 for track_id in trajectories):
            raise ValueError("Angle trajectory IDs must be nonnegative integers")
        self.angle_trajectories = {track_id: tuple(frames) for track_id, frames in trajectories.items()}

    def set_angle_frame(self, frame_index: int) -> None:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("Angle scan frame index must be a nonnegative integer")
        self.angle_frame_index = frame_index
        if self.tracker is not None:
            self.tracker.frame_number = frame_index
            if getattr(self, "target_registry", None) is not None:
                self.tracker.registry_frame_index = frame_index

    def scan_locked_targets(self, video_path, *, hdr_policy: str = "tonemap_sdr", on_frame=None) -> dict[str, int]:
        """Track the whole timeline once so the locked canonical IDs are bound by the *global* consolidation.

        The tracker runs over every frame, the binder records its tracklets and, at the end, consolidates and matches them to
        the locked signatures (``LockedTargetBinder.finish_prepass``). Association state is then reset, so the render replays
        the same tracker deterministically and looks each face up by frame and box. Returns the pre-pass summary."""
        if not self.prepared or self.tracker is None:
            raise DomainError("SESSION_NOT_PREPARED", "Prepare a session before scanning target identities", 409)
        binder = self.locked_targets
        if binder is None:
            return {}
        from roop_3d.media.video_reader import VideoReader
        frames = VideoReader(video_path, hardware="auto", hdr_policy=hdr_policy).frames()
        binder.start_prepass()
        try:
            for frame_index, decoded in enumerate(frames):
                if decoded.image is not None:
                    self.set_frame_timestamp(float(decoded.timestamp.seconds))
                    self.tracker.frame_number = frame_index
                    self._bind_locked_targets(self.tracker.process(decoded.image), frame_index)
                if on_frame is not None:
                    on_frame(frame_index + 1)
        finally:
            frames.close()
            self.tracker.reset_tracking()
            self.tracker.frame_number = None
            binder.finish_prepass()
        return dict(binder.prepass_summary)

    def scan_angle_trajectories(self, video_path, track_ids: set[int], *, fps: float = 30.0,
                                hdr_policy: str = "tonemap_sdr", min_cutoff: float = 0.15,
                                beta: float = 0.04, derivative_cutoff: float = 1.0,
                                on_frame=None, window: tuple[Any, Any] | None = None) -> dict[int, tuple[Any, ...]]:
        """Run Stage A/B/C over the video, then reset association state for deterministic render replay.

        ``window`` is the job's ``(start, end)`` trim in source seconds (start inclusive, end exclusive, either
        may be None). Frames outside it are not analysed - they get empty detections so every index still
        lines up with the render - and decoding stops at ``end``: a 6 s trim of a 7 min file no longer pays for
        a 7 min tracker pass.
        """
        if not self.prepared or self.tracker is None:
            raise DomainError("SESSION_NOT_PREPARED", "Prepare a session before scanning target angles", 409)
        if any(type(track_id) is not int or track_id < 0 for track_id in track_ids):
            raise ValueError("Angle scan IDs must be nonnegative integers")
        from roop_3d.angle_tracker import (AngleDetection, AngleTrajectoryTracer,
                                           optical_flow_landmarks_3d)
        from roop_3d.faceset_loader import LandmarkEngine
        from roop_3d.identity.target_lock import BankEntry
        from roop_3d.media.video_reader import VideoReader

        detections: list[dict[int, AngleDetection]] = []
        barriers: list[set[int]] = []
        flow_refinements: dict[tuple[int, int, int], dict[int, np.ndarray]] = {}
        last_detection: dict[int, AngleDetection] = {}
        last_gray: dict[int, np.ndarray] = {}
        pending_gray: dict[int, list[np.ndarray]] = {}
        flow_scale = 1.0
        # The 68-point model runs once per scanned frame: on the job's GPU (CUDA, no TensorRT engine build) instead of the CPU.
        from roop_3d.engine.runtime import inspection_session_factory
        landmark_factory = (None if self.config.provider == "CPUExecutionProvider" else inspection_session_factory(
            self.catalog, config=replace(self.config, provider="CUDAExecutionProvider")))
        landmark_engine = LandmarkEngine(root=Path(__file__).resolve().parents[3], session_factory=landmark_factory)
        has_landmarks68 = landmark_engine.has_68
        reader = VideoReader(video_path, hardware="auto", hdr_policy=hdr_policy)
        frames = reader.frames()
        if self.locked_targets is not None:
            self.locked_targets.start_prepass()
        try:
            window_start, window_end = window if window is not None else (None, None)
            for frame_index, decoded in enumerate(frames):
                if window_end is not None and decoded.timestamp.seconds >= window_end:
                    break
                if window_start is not None and decoded.timestamp.seconds < window_start:
                    detections.append({})
                    barriers.append(set())
                    continue
                if decoded.image is None:
                    detections.append({})
                    barriers.append(set(track_ids))
                    continue
                gray = cv2.cvtColor(decoded.image, cv2.COLOR_BGR2GRAY)
                flow_scale = min(1.0, 640.0 / max(gray.shape))
                flow_image = (cv2.resize(gray, None, fx=flow_scale, fy=flow_scale,
                                          interpolation=cv2.INTER_AREA)
                              if flow_scale < 1.0 else gray)
                self.set_frame_timestamp(float(decoded.timestamp.seconds))
                self.tracker.frame_number = frame_index
                tracked_faces = self._persistent_target_faces(
                    self._bind_locked_targets(self.tracker.process(decoded.image), frame_index))
                frame_detections: dict[int, AngleDetection] = {}
                frame_barriers = set(track_ids) if self.tracker.scene_cut else set()
                if self.tracker.scene_cut:
                    last_detection.clear()
                    last_gray.clear()
                    pending_gray.clear()
                for tracked in tracked_faces:
                    if tracked.track_id not in track_ids:
                        continue
                    face = tracked.observation
                    if not tracked.swap_allowed:
                        frame_barriers.add(tracked.track_id)
                        last_detection.pop(tracked.track_id, None)
                        last_gray.pop(tracked.track_id, None)
                        pending_gray.pop(tracked.track_id, None)
                        continue
                    from roop_3d.core.pose import as_pose3d
                    pose = as_pose3d(face.pose).as_tuple()
                    landmarks68 = (landmark_engine.landmarks68(decoded.image, face.box)
                                   if has_landmarks68 else None)
                    frame_detections[tracked.track_id] = AngleDetection(
                        face.box, face.landmarks5, pose, face.embedding,
                        landmarks68, face.landmarks106, tracked.confidence, frame_index)
                for track_id, current in frame_detections.items():
                    previous = last_detection.get(track_id)
                    missed = pending_gray.pop(track_id, [])
                    gap = frame_index - previous.frame_index - 1 if previous is not None else 0
                    if (previous is not None and previous.landmarks68_3d is not None
                            and current.landmarks68_3d is not None and 1 <= gap <= 8
                            and len(missed) == gap and track_id in last_gray):
                        images = [last_gray[track_id], *missed, flow_image]

                        def flow(before, after, points):
                            scaled = points.copy()
                            scaled[:, :2] *= flow_scale
                            moved = optical_flow_landmarks_3d(before, after, scaled)
                            moved[:, :2] /= flow_scale
                            return moved

                        try:
                            forward = {}
                            moved = previous.landmarks68_3d.copy()
                            for step in range(1, gap + 1):
                                moved = flow(images[step - 1], images[step], moved)
                                forward[step] = moved.copy()
                            backward = {}
                            moved = current.landmarks68_3d.copy()
                            for step in range(gap, 0, -1):
                                moved = flow(images[step + 1], images[step], moved)
                                backward[step] = moved.copy()
                            refinements = {}
                            for step in range(1, gap + 1):
                                amount = step / (gap + 1)
                                points = (1.0 - amount) * forward[step] + amount * backward[step]
                                points[:, 2] = ((1.0 - amount) * previous.landmarks68_3d[:, 2]
                                                + amount * current.landmarks68_3d[:, 2])
                                refinements[previous.frame_index + step] = points.astype(np.float32)
                            flow_refinements[(track_id, previous.frame_index, frame_index)] = refinements
                        except (ValueError, cv2.error):
                            pass
                    last_detection[track_id] = current
                    last_gray[track_id] = flow_image.copy()
                for track_id in track_ids:
                    if track_id in frame_detections:
                        continue
                    if track_id in frame_barriers:
                        last_detection.pop(track_id, None)
                        last_gray.pop(track_id, None)
                        pending_gray.pop(track_id, None)
                        continue
                    previous = last_detection.get(track_id)
                    if (previous is not None and previous.landmarks68_3d is not None
                            and frame_index - previous.frame_index <= 8):
                        pending_gray.setdefault(track_id, []).append(flow_image.copy())
                    else:
                        pending_gray.pop(track_id, None)
                detections.append(frame_detections)
                barriers.append(frame_barriers)
                if on_frame is not None:
                    on_frame(frame_index + 1)
        finally:
            frames.close()
            self.tracker.reset_tracking()
            self.tracker.frame_number = None
            self.angle_frame_index = None
            if self.locked_targets is not None:
                self.locked_targets.finish_prepass()

        tracer = AngleTrajectoryTracer(min_cutoff=min_cutoff, beta=beta,
                                       derivative_cutoff=derivative_cutoff)
        trajectories: dict[int, tuple[Any, ...]] = {}
        for track_id in sorted(track_ids):
            output: list[Any] = [None] * len(detections)
            references: list[BankEntry] = []
            for frame_index, frame in enumerate(detections):
                face = frame.get(track_id)
                if face is None:
                    continue
                if all(np.hypot(face.pose[0] - item.yaw, face.pose[1] - item.pitch) >= 12.0
                       for item in references):
                    references.append(BankEntry(face.embedding.copy(), face.pose[0], face.pose[1],
                                                face.pose[2], frame_index, face.bbox.tolist(), "scan"))
                    if len(references) >= 12:
                        break
            start = 0
            for boundary in [index for index, row in enumerate(barriers) if track_id in row] + [len(detections)]:
                if boundary > start:
                    sequence = [[row[track_id]] if track_id in row else []
                                for row in detections[start:boundary]]
                    if any(sequence):
                        def gap_flow(left, right, _left_points, _right_points):
                            return flow_refinements.get((track_id, start + left, start + right), {})

                        result = tracer.scan(sequence, reference_bank=references, fps=fps,
                                             gap_flow=gap_flow)
                        for offset, point in enumerate(result.target_frames):
                            if point is not None:
                                output[start + offset] = replace(point, frame_index=start + offset,
                                    track_id=track_id, alignment_matrix=point.alignment_matrix.copy())
                # A scene-cut frame may already contain the reacquired face. Keep that observation
                # as the first point of a fresh segment; explicitly withheld/missing frames stay empty.
                start = (boundary if boundary < len(detections) and track_id in detections[boundary]
                         else boundary + 1)
            trajectories[track_id] = tuple(output)
        return trajectories

    def _apply_angle_trajectories(self, faces: tuple[TrackedFace, ...]) -> tuple[TrackedFace, ...]:
        frame_index = self.angle_frame_index
        if frame_index is None or not self.angle_trajectories:
            return faces
        from roop_3d.core.pose import as_pose3d
        by_id = {item.track_id: item for item in faces}
        for track_id, trajectory in self.angle_trajectories.items():
            if frame_index >= len(trajectory):
                continue
            point = trajectory[frame_index]
            if point is None or point.landmarks106 is None or point.embedding is None:
                continue
            current = by_id.get(track_id)
            pose = as_pose3d(point.pose, frame="standard").legacy_tuple()
            if current is None:
                observation = FaceObservation(point.bbox.copy(), point.landmarks5.copy(),
                    point.landmarks106.copy(), point.embedding.copy(), pose)
                by_id[track_id] = TrackedFace(track_id, observation, 1,
                    self.tracker.scene_id if self.tracker is not None else 0,
                    state="TRACKED", confidence=min(0.65, float(point.confidence)), swap_allowed=True)
                continue
            observation = replace(current.observation, box=point.bbox.copy(),
                landmarks5=point.landmarks5.copy(), landmarks106=point.landmarks106.copy(),
                embedding=point.embedding.copy(), pose=pose)
            by_id[track_id] = replace(current, observation=observation)
        return tuple(by_id[key] for key in sorted(by_id))

    @inference
    def infer_frame(self, image: np.ndarray, *, tracked: TrackedFrame | None = None) -> FrameResult:
        """Track, gate, then render only with the wired true-3D FullHeadEngine; no alternate renderer is accepted.

        ``tracked`` (from :meth:`track_frame`) lets Stage 1 run ahead; without it tracking happens inline. A frame with no
        selected, gate-passing face returns an empty result (nothing to render, so nothing is falsely reported as rendered)."""
        with self.lock:
            if not self.prepared or self.tracker is None:
                raise DomainError("SESSION_NOT_PREPARED", "Prepare a source before frame inference", 409)
            image_bgr(image)
            began = time.perf_counter()
            if tracked is None:
                self._frame_context = None
                faces = self.tracker.process(image, include203=False)
                faces = self._persistent_target_faces(self._apply_angle_trajectories(
                    self._bind_locked_targets(faces, self.frames_processed)))
                detection_end = time.perf_counter()
                tracked = TrackedFrame(faces, bool(self.tracker.scene_cut), self._active_ids(),
                                       self._frame_time(), (detection_end - began) * 1000)
            else:
                faces, detection_end = tracked.faces, began + tracked.detection_ms / 1000
            self._frame_context = tracked
            scene_cut, active_ids = tracked.scene_cut, tracked.active_ids
            for track_id in set(self.full_head_engines) - set(active_ids):
                self.full_head_engines.pop(track_id).rasterizer.release()
            if scene_cut:
                self._roll_latched.clear()
                self._presence.reset()
                self._pose_fade.reset()
            else:
                self._roll_latched = {key: value for key, value in self._roll_latched.items() if key in active_ids}
                self._pose_fade.retain(active_ids)
            withheld: list[Withheld] = []
            renderable: list[TrackedFace] = []
            for face in faces:
                if self.selected_ids is not None and face.track_id not in self.selected_ids:
                    continue
                if not face.swap_allowed:
                    withheld.append(Withheld(face.track_id, face.reason or face.state, face.state, face.confidence))
                    continue
                if self.identity_guard is not None:
                    verdict = self.identity_guard.check(face)
                    if not verdict.allowed:
                        withheld.append(Withheld(face.track_id, verdict.reason or "IDENTITY_UNVERIFIED",
                                                 face.state, verdict.similarity))
                        continue
                renderable.append(face)
            if renderable:
                require_true3d_renderer(f"render {len(renderable)} tracked face(s)")
                raise True3DRendererUnavailable(
                    "3D renderer not wired: FullHeadEngine rendering is not connected to ReferencePipeline.infer_frame",
                    {"track_ids": [face.track_id for face in renderable]})
            ended = time.perf_counter()
            self.frames_processed += 1
            self.last_latency = StageLatency((detection_end - began) * 1000, 0.0, 0.0, (ended - began) * 1000, 0.0)
            return FrameResult(faces, (), self.last_latency, scene_cut, tuple(withheld))

    def track_frame(self, image: np.ndarray) -> TrackedFrame:
        """Stage 1: detection, landmarks, pose and track association for one frame (no render work).

        Pass the result to :meth:`infer_frame` as ``tracked``. Frames must be tracked in order; only this method
        touches the tracker, so it may safely run on a thread of its own while Stage 2 works on an earlier frame.
        """
        with self.track_lock:
            if not self.prepared or self.tracker is None:
                raise DomainError("SESSION_NOT_PREPARED", "Prepare a source before frame inference", 409)
            image_bgr(image)
            began = time.perf_counter()
            faces = self.tracker.process(image, include203=False)
            faces = self._persistent_target_faces(self._apply_angle_trajectories(
                self._bind_locked_targets(faces, self.frames_tracked)))
            seconds = getattr(self.tracker, "timestamp_seconds", None)
            frame_time = float(seconds) if seconds is not None and np.isfinite(seconds) else self.frames_tracked / 30.0
            self.frames_tracked += 1
            return TrackedFrame(faces, bool(self.tracker.scene_cut), self._active_ids(), frame_time,
                                (time.perf_counter() - began) * 1000)

    def _detect_landmarks106(self, crop: np.ndarray) -> np.ndarray | None:
        """106 dense landmarks of a crop (the crop is a tight face, so the box is the whole image)."""
        if self.tracker is None:
            return None
        h, w = crop.shape[:2]
        return self.tracker.dense.infer106(crop, np.array((0, 0, w, h), np.float32))

    def capability(self) -> dict:
        from roop_3d.pipeline import true3d_renderer
        wired = true3d_renderer.is_wired()
        return {"prepared": self.prepared, "frames_processed": self.frames_processed,
                "renderer": {"wired": wired, "message": "ready" if wired else "3D renderer not wired"},
                "profile": self.config.hardware_profile,
                "selected_track_ids": sorted(self.selected_ids) if self.selected_ids is not None else None,
                "dense_landmarks": (68, 106, 203),
                "full_head": {"active": False, "renderer": "not_wired"},
                "compositor": {"architecture": "full_frame_true3d_only", "wired": wired}}

    def composite_frame(self, image: np.ndarray, result: FrameResult, *,
                        options=None, frame_index: int | None = None,
                        timestamp_seconds: float | None = None) -> np.ndarray:
        """Composite only a complete 3D mesh raster with full-frame semantic masks."""
        if timestamp_seconds is not None and not np.isfinite(timestamp_seconds):
            raise ValueError("Frame timestamp must be finite")
        if not result.replacements:
            return image
        with self.composite_lock:
            from roop_3d.pipeline.photorealistic_swap_core import Photorealistic3DSwapCore, SwapCoreOptions
            from roop_3d.pipeline.session_adapter import SessionFrameProcessor
            if self._frame_processor is None:
                self._frame_processor = SessionFrameProcessor(self, Photorealistic3DSwapCore())
            core_options = SwapCoreOptions(
                feather_px=float(getattr(options, "feather_px", 5.0)),
                pyramid_levels=int(getattr(options, "pyramid_levels", 4)),
            )
            self._frame_processor.core.options = core_options
            index = max(0, self.frames_processed - 1) if frame_index is None else frame_index
            output, report = self._frame_processor.process_result(image, result, index)
            self.last_frame_report = report
            return output
    @guarded
    def release_session(self) -> None:
        with self.track_lock, self.lock, self.composite_lock:
            self._frame_context = None
            self.frames_tracked = 0
            if self.tracker is not None:
                self.tracker.release()
            self.tracker = None
            if self._frame_processor is not None:
                self._frame_processor.release()
                self._frame_processor = None
            self.last_frame_report = None
            self.selected_ids = None
            self.sources_by_track.clear()
            self.identity_guard = None
            self.locked_targets = None
            self.prepared = False
            self.last_latency = None
            self.frames_processed = 0
            self.default_source, self.identity_store = None, None
            for engine in self.full_head_engines.values():
                engine.rasterizer.release()
            self.full_head_engines.clear()
            self.head_landmarks = None
            self.angle_trajectories.clear()
            self.angle_frame_index = None
            self._roll_latched.clear()
            self._presence.reset()
            self._pose_fade.reset()


def prepare_session(source: FaceSetDocument, *, catalog: ModelCatalog | None = None,
                    config: RuntimeConfig = RuntimeConfig(), **kwargs) -> ReferencePipeline:
    session = ReferencePipeline(catalog, config)
    session.prepare_session(source, **kwargs)
    return session


def infer_frame(session: ReferencePipeline, image: np.ndarray, **kwargs) -> FrameResult:
    return session.infer_frame(image, **kwargs)


def release_session(session: ReferencePipeline) -> None:
    session.release_session()
