"""Stage 06 lifecycle: tracking -> identity transfer -> optional 3D reenactment.

Returns model crops and exact crop-to-frame matrices; the full-frame compositor
and its occlusion/segmentation logic remain Stage 09 work.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import threading
import time
from collections.abc import Mapping
from typing import Literal
import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.engine.detection.face_tracker import FaceObservation, FaceTracker, TrackedFace
from roop_3d.engine.runtime import RuntimeConfig, image_bgr
from roop_3d.engine.volumetric.angle_diagnostics import AngleDiagnostics
from roop_3d.engine.volumetric.identity_swapper import IdentitySwapper, SwapResult
from roop_3d.engine.volumetric.presence import PoseFade, SwapPresence
from roop_3d.engine.volumetric.preprocess import AlignedCrop, landmark_crop, transform_points
from roop_3d.engine.volumetric.reference_adapter import LivePortraitAdapter
from roop_3d.faceset.schemas import FaceSetDocument
from roop_3d.models.catalog import ModelCatalog
from roop_3d.engine.resources import guarded, inference, register
from roop_3d.temporal import smoothstep
from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor
from roop_3d.engine.audio_sync.lip_sync_adapter import LipSyncAdapter, LipSyncFrameInput
from roop_3d.reconstruction.expression_tps import apply_expression_to_crop, group, layout_is_plausible
from roop_3d.reconstruction.pose_blend import PoseBlendProjector, PoseBlendResult
from roop_3d.reconstruction.reenactment import (RENDER_MODES, IdentityRig, ReenactmentEngine, ReenactmentFrame,
                                                ReenactmentSettings, TargetObservation)


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
    image: np.ndarray
    inverse: np.ndarray
    reference_index: int
    method: str
    coverage: np.ndarray | None = None
    projection: PoseBlendResult | None = None
    landmarks5: np.ndarray | None = None
    # Stage 4: what ran, and why the proven path was used instead of the requested mode (never silent).
    render_mode: str = "standard"
    fallback_reason: str | None = None
    reenactment: ReenactmentFrame | None = None
    full_head: bool = False
    target_silhouette: np.ndarray | None = None
    source_silhouette: np.ndarray | None = None
    photometry: object = None
    face_detail: object = None
    # Fade factor of a face whose swap is held through, or returning from, a one/two-frame tracker gate.
    presence: float = 1.0
    # The crop already went through the restorer at its native resolution (the dense path restores before it composites).
    restored: bool = False


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


def swapped_portrait(portrait: AlignedCrop, swap: SwapResult) -> np.ndarray:
    """Place the identity crop within a portrait for the appearance encoder only."""
    portrait_matrix = np.vstack((portrait.matrix, (0, 0, 1)))
    swap_inverse = np.vstack((swap.alignment.inverse, (0, 0, 1)))
    local = (portrait_matrix @ swap_inverse)[:2]
    height, width = swap.image.shape[:2]
    mask = np.zeros((height, width), np.float32)
    cv2.ellipse(mask, (round(width * 0.5), round(height * 0.555)),
                (round(width * 0.39), round(height * 0.46)), 0, 0, 360, 1, -1)
    blur = max(3, round(min(height, width) * 0.117)) | 1
    mask = cv2.GaussianBlur(mask, (blur, blur), max(1.0, height * 0.0234))
    size = (portrait.image.shape[1], portrait.image.shape[0])
    weight = cv2.warpAffine(mask, local, size)[:, :, None]
    fake = cv2.warpAffine(swap.image, local, size)
    return np.clip(fake * weight + portrait.image * (1 - weight), 0, 255).astype(np.uint8)


PROFILE_TRIM_START, PROFILE_TRIM_FULL = 30.0, 40.0   # |yaw| between which the far-cheek trim fades in (degrees)


def profile_aware_mask(shape: tuple[int, int], landmarks5: np.ndarray, yaw: float) -> np.ndarray:
    """Tighten the far cheek edge for turned faces while leaving the near side untouched.

    The trim fades in over ``PROFILE_TRIM_START``..``PROFILE_TRIM_FULL`` degrees of yaw: a threshold would pop the cheek
    edge in a single frame wherever the head crosses it.
    """
    height, width = shape
    result = np.ones((height, width), np.float32)
    amount = smoothstep((abs(float(yaw)) - PROFILE_TRIM_START) / (PROFILE_TRIM_FULL - PROFILE_TRIM_START))
    if amount <= 0.0 or width < 16 or height < 16:
        return result
    points = np.asarray(landmarks5, np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return result
    nose_x = float(points[2, 0])
    eye_distance = max(1.0, float(np.linalg.norm(points[1] - points[0])))
    # The far cheek silhouette sits about two thirds of an inter-ocular distance past the bridge.
    # A cosine feather keeps the bridge and eye coverage continuous at the cutoff.
    feather = max(4.0, width * 0.07)
    if yaw > 0:
        edge = nose_x - 0.65 * eye_distance
        x = np.arange(width, dtype=np.float32)
        ramp = np.clip((x - (edge - feather)) / feather, 0.0, 1.0)
    else:
        edge = nose_x + 0.65 * eye_distance
        x = np.arange(width, dtype=np.float32)
        ramp = np.clip(((edge + feather) - x) / feather, 0.0, 1.0)
    ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)
    result *= (1.0 - amount * (1.0 - ramp))[None, :]
    return result


class ReferencePipeline:
    def __init__(self, catalog: ModelCatalog | None = None, config: RuntimeConfig = RuntimeConfig()):
        self.catalog, self.config = catalog or ModelCatalog(), config
        self.tracker: FaceTracker | None = None
        self.swapper = IdentitySwapper(self.catalog, config)
        self.portrait = LivePortraitAdapter(self.catalog, config)
        # The compositor and its parser/restorer adapters are lazy; session creation
        # does not activate any additional inference model.
        self.compositor = FrameCompositor(self.catalog, config)
        self.compositor.expression_portrait = self.portrait
        self.angle_diagnostics = AngleDiagnostics()
        self.dense_enhancer_calls = 0
        self.compositor.angle_diagnostics = self.angle_diagnostics
        self.lip_sync = LipSyncAdapter(self.catalog)
        self.selected_ids: frozenset[int] | None = None
        self.sources_by_track: dict[int, FaceSetDocument] = {}
        self.identity_guard = None
        self.prepared = False
        self.lock = threading.RLock()
        # Stages 1 (tracking) and 3 (composite) have their own locks so they can overlap the swap stage.
        # Order whenever more than one is needed: track_lock -> lock -> composite_lock.
        self.track_lock = threading.RLock()
        self.composite_lock = threading.RLock()
        self._frame_context: TrackedFrame | None = None
        self.frames_tracked = 0
        self.last_latency: StageLatency | None = None
        self.frames_processed = 0
        self.pose_projector: PoseBlendProjector | None = None
        # Stage 4 render modes: "standard" (existing identity swap), "pose_3d", "reenactment_3d"
        self.render_mode = "standard"
        self.default_source: FaceSetDocument | None = None
        self.rigs: dict[str, IdentityRig] = {}
        self.rig_status: dict[str, dict] = {}
        self.reenactors: dict[int, ReenactmentEngine] = {}
        self.pose_projectors: dict[str, PoseBlendProjector] = {}
        self.full_head_engines = {}
        self.head_landmarks = None
        self.neural_face = None
        # Interactive previews may opt in to degrading to the standard 2D swap when the dense facial bundle is not
        # installed. Renders keep the strict default (explicit DENSE_FACE_ASSETS_MISSING error, never a silent change).
        self.dense_face_optional = False
        self.dense_face_unavailable: str | None = None
        self.angle_trajectories: dict[int, tuple[Any, ...]] = {}
        self.angle_frame_index: int | None = None
        self.reenactment_settings = ReenactmentSettings()
        self.pose_hysteresis_degrees = Settings.model_fields["pose_hysteresis_degrees"].default
        self._roll_latched: dict[int, bool] = {}
        self._presence = SwapPresence()
        self._pose_fade = PoseFade()
        register(self, "release_session")

    def select_swapper(self, model_id: str) -> None:
        """Select a registered swapper before source preparation/model activation."""
        if self.prepared:
            raise DomainError("SESSION_ALREADY_PREPARED", "Swapper selection is fixed for a prepared render session", 409)
        if self.catalog.swapper_spec(model_id) is None:
            raise DomainError("SWAPPER_MODEL_UNSUPPORTED", "Selected model is not a registered swapper", 422)
        self.swapper.model_id = model_id

    @guarded
    def prepare_session(self, source: FaceSetDocument, *, track_ids: set[int] | None = None,
                        sources_by_track: Mapping[int, FaceSetDocument] | None = None,
                        retarget: Literal["auto", "off", "required"] = "auto",
                        track_policy=None, detection_threshold: float = 0.5,
                        render_mode: str = "standard", identity_store=None,
                        reenactment_settings: ReenactmentSettings | None = None) -> dict:
        with self.lock:
            if render_mode not in RENDER_MODES:
                raise ValueError(f"render_mode must be one of {RENDER_MODES}")
            if retarget not in ("auto", "off", "required") or track_ids is not None and (
                    len(track_ids) > 32 or any(type(i) is not int or i < 0 for i in track_ids)):
                raise ValueError("Invalid session selection/policy")
            mapped_sources = dict(sources_by_track or {})
            if (len(mapped_sources) > 32 or any(type(i) is not int or i < 0 for i in mapped_sources)
                    or track_ids is not None and not set(mapped_sources).issubset(track_ids)):
                raise ValueError("Mapped FaceSets must use selected nonnegative track IDs")
            self.release_session()
            try:
                self.swapper.prepare_session(source, sources_by_track=mapped_sources)
                self.tracker = FaceTracker(self.catalog, config=self.config, threshold=detection_threshold,
                                           policy=track_policy)
                self.selected_ids = frozenset(track_ids) if track_ids is not None else None
                self.sources_by_track = mapped_sources
                self.default_source, self.render_mode = source, render_mode
                self.reenactment_settings = reenactment_settings or ReenactmentSettings()
                self.dense_face_unavailable = None
                if render_mode != "standard":
                    self._prepare_identity_rigs(source, mapped_sources, identity_store)
                if render_mode != "standard" and self.dense_face_unavailable:
                    self.portrait.fallback_reason = self.dense_face_unavailable
                elif render_mode != "standard":
                    self.portrait.fallback_reason = "Three-tier reconstruction supplies target expressions; dense warp uses neural swapper and UltraMax"
                elif retarget != "off":
                    self.portrait.prepare_session(required=retarget == "required")
                else:
                    self.portrait.fallback_reason = "LivePortrait disabled explicitly; identity-only path selected"
                self.prepared = True
                return self.capability()
            except BaseException:
                self.release_session()
                raise

    def _prepare_identity_rigs(self, source: FaceSetDocument, mapped: Mapping[int, FaceSetDocument], store=None) -> None:
        """Prepare face-only 3D correspondence; no MakeHuman render path."""
        from roop_3d.reconstruction.neural_face import NeuralFaceEngine, WARM_ENGINE
        from roop_3d.core.ort_session import pool_scope_active
        if self.neural_face is None:
            try:
                self._warm_key = (id(self.catalog), self.config.provider, self.config.hardware_profile,
                                  self.config.precision_policy, self.config.cpu_threads) if pool_scope_active() else None
                warm = WARM_ENGINE.take(self._warm_key) if self._warm_key is not None else None
                self.neural_face = warm if warm is not None else NeuralFaceEngine.load(self.catalog, self.config)
            except DomainError as error:
                if not self.dense_face_optional or error.code != "DENSE_FACE_ASSETS_MISSING":
                    raise
                self.dense_face_unavailable = f"{error.code}: standard 2D identity swap used instead. {error.message}"
                return
        for document in {d.faceset_id: d for d in (source, *mapped.values())}.values():
            self.neural_face.prepare(document)
            self.rig_status[document.faceset_id] = self.neural_face.capability()

    def _document_for(self, track_id: int) -> FaceSetDocument | None:
        return self.sources_by_track.get(track_id, self.default_source)

    def _rig_for(self, track_id: int) -> IdentityRig | None:
        document = self._document_for(track_id)
        return self.rigs.get(document.faceset_id) if document is not None else None

    def _unavailable_reason(self, track_id: int) -> str:
        document = self._document_for(track_id)
        status = self.rig_status.get(document.faceset_id) if document is not None else None
        return "IDENTITY_ASSET_UNAVAILABLE" + (f": {status['reason']}" if status and status.get("reason") else "")

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
        if self.tracker is not None:
            self.tracker.target_registry = registry

    def _persistent_target_faces(self, faces):
        registry = getattr(self, "target_registry", None)
        if registry is None or not faces or getattr(self.tracker, "target_registry", None) is registry:
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
        if registry is None:
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

    def _roll_exceeded(self, track_id: int, roll: float) -> bool:
        """The dense path hands a face to the 2D swap above ``dense_3d_max_roll``; leaving that fallback needs the head to
        come ``pose_hysteresis_degrees`` back inside the limit, so a roll hovering at the limit does not flip the method
        (and the whole look of the face) from frame to frame."""
        limit = dense_3d_max_roll()
        exceeded = abs(float(roll)) > limit - (self.pose_hysteresis_degrees if self._roll_latched.get(track_id) else 0.0)
        self._roll_latched[track_id] = exceeded
        return exceeded

    def configure_temporal(self, settings) -> None:
        from roop_3d.temporal import TrackTemporalStabilizer
        from roop_3d.temporal_stabilizer import TemporalStabilizer
        config = dict(min_cutoff=settings.temporal_min_cutoff, beta=settings.temporal_beta,
                      derivative_cutoff=settings.temporal_d_cutoff)
        if self.tracker is not None:
            self.tracker.stabilizer = TrackTemporalStabilizer(**config)
        # The histogram stabilizer's One-Euro filter names the derivative cutoff `d_cutoff`.
        self.compositor.histogram_stabilizer = TemporalStabilizer(
            min_cutoff=config['min_cutoff'], beta=config['beta'], d_cutoff=config['derivative_cutoff'])
        self.pose_hysteresis_degrees = settings.pose_hysteresis_degrees

    def set_frame_timestamp(self, seconds: float, *, compositor: bool = True) -> None:
        """``compositor=False`` is for the asynchronous pipeline, where the composite stage sets its own per frame."""
        if not np.isfinite(seconds):
            raise ValueError('Frame timestamp must be finite')
        if self.tracker is not None:
            self.tracker.timestamp_seconds = seconds
        if compositor:
            self.compositor.timestamp_seconds = seconds

    def set_angle_trajectories(self, trajectories: Mapping[int, tuple[Any, ...]]) -> None:
        """Install the Stage 13 scan before rendering begins; keys are replayed tracker IDs."""
        if any(type(track_id) is not int or track_id < 0 for track_id in trajectories):
            raise ValueError("Angle trajectory IDs must be nonnegative integers")
        self.angle_trajectories = {track_id: tuple(frames) for track_id, frames in trajectories.items()}

    def set_angle_frame(self, frame_index: int) -> None:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("Angle scan frame index must be a nonnegative integer")
        self.angle_frame_index = frame_index
        if self.tracker is not None and getattr(self, "target_registry", None) is not None:
            self.tracker.registry_frame_index = frame_index

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
                tracked_faces = self._persistent_target_faces(self.tracker.process(decoded.image))
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
            self.angle_frame_index = None

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
    def infer_frame(self, image: np.ndarray, *, strength: float = 0.35, pose_blend: float | None = None,
                    frontalization: float = 0.0, tps_expression: float = 0.0, identity_blend_t: float | None = None,
                    render_mode: str | None = None, tracked: TrackedFrame | None = None,
                    jaw_adaptation: bool = False) -> FrameResult:
        """``tracked`` (from :meth:`track_frame`) lets Stage 1 run ahead; without it tracking happens inline as before.

        ``jaw_adaptation`` (opt-in) fits the canonical proxy's jaw width to the target's 106-point face contour in the mesh
        pose-blend path (``pose_blend`` given); the outcome, applied or not and why, is on ``replacement.projection.jaw``."""
        with self.lock:
            if not self.prepared or self.tracker is None:
                raise DomainError("SESSION_NOT_PREPARED", "Prepare a source before frame inference", 409)
            image_bgr(image)
            if not np.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("Retargeting strength must be within 0..1")
            if pose_blend is not None and (not np.isfinite(pose_blend) or not 0 <= pose_blend <= 1):
                raise ValueError('Mesh pose blend must be within 0..1')
            for name, amount in (('Pose frontalization', frontalization), ('TPS expression weight', tps_expression)):
                if not np.isfinite(amount) or not 0 <= amount <= 1:
                    raise ValueError(f'{name} must be within 0..1')
            mode = self.render_mode if render_mode is None else render_mode
            if mode not in RENDER_MODES:
                raise ValueError(f"render_mode must be one of {RENDER_MODES}")
            if mode != "standard" and self.dense_face_unavailable:
                mode = "standard"
            if mode == "pose_3d" and pose_blend is None:
                pose_blend = 1.0
            began = time.perf_counter()
            if tracked is None:
                self._frame_context = None
                faces = self.tracker.process(image, include203=self.portrait.available and pose_blend is None)
                faces = self._persistent_target_faces(self._apply_angle_trajectories(faces))
                detection_end = time.perf_counter()
                tracked = TrackedFrame(faces, bool(self.tracker.scene_cut), frozenset(self.tracker.active_ids),
                                       self._frame_time(), (detection_end - began) * 1000)
            else:
                faces, detection_end = tracked.faces, began + tracked.detection_ms / 1000
            self._frame_context = tracked
            scene_cut, active_ids = tracked.scene_cut, tracked.active_ids
            frame_time = tracked.frame_time
            self.portrait.retain_tracks(active_ids)
            for track_id in set(self.reenactors) - set(active_ids):
                self.reenactors.pop(track_id).release()
            for track_id in set(self.full_head_engines) - set(active_ids):
                self.full_head_engines.pop(track_id).rasterizer.release()
            if scene_cut:
                self._roll_latched.clear()
                self._presence.reset()
                self._pose_fade.reset()
            else:
                self._roll_latched = {key: value for key, value in self._roll_latched.items() if key in active_ids}
                self._pose_fade.retain(active_ids)
            if self.neural_face is not None:
                self.neural_face.retain_tracks(active_ids)
            replacements, swap_seconds, retarget_seconds, projection_seconds = [], 0.0, 0.0, 0.0
            withheld: list[Withheld] = []
            gated: set[int] = set()   # tracker policy closed the gate this frame (not an identity verdict)
            for tracked in faces:
                if self.selected_ids is not None and tracked.track_id not in self.selected_ids:
                    continue
                if not tracked.swap_allowed:
                    gated.add(tracked.track_id)
                    withheld.append(Withheld(tracked.track_id, tracked.reason or tracked.state, tracked.state, tracked.confidence))
                    continue
                if self.identity_guard is not None:
                    verdict = self.identity_guard.check(tracked)
                    if not verdict.allowed:
                        withheld.append(Withheld(tracked.track_id, verdict.reason or "IDENTITY_UNVERIFIED",
                                                 tracked.state, verdict.similarity))
                        continue
                face = tracked.observation
                target_reference = self._target_reference(tracked)
                if face.landmarks68 is None:
                    measured = self._bank_landmarks(target_reference, face.box)
                    if measured is not None:
                        face = replace(face, landmarks68=measured)
                        tracked = replace(tracked, observation=face)
                at = time.perf_counter()
                yaw_limit, pitch_limit = auto_3d_limits()
                extreme = abs(face.pose[0]) > yaw_limit or abs(face.pose[1]) > pitch_limit
                face_mode = "pose_3d" if mode == "standard" and extreme else mode
                dense_fallback = self.dense_face_unavailable if face_mode != "standard" else None
                if face_mode != "standard" and not dense_fallback and self._roll_exceeded(tracked.track_id, face.pose[2]):
                    # The dense mesh fit breaks down on a strongly rolled head (a close-up lying on a pillow produced a duplicated
                    # eye and a warped mouth, which the compositor then rightly masked away: no swap at all). The standard 2D swap
                    # aligns by the five landmarks, so it copes with roll.
                    dense_fallback = f"DENSE_3D_ROLL_LIMIT: head roll {face.pose[2]:.0f} deg exceeds {dense_3d_max_roll():.0f}; standard 2D swap used"
                if face_mode != "standard" and not dense_fallback:
                    document = self._document_for(tracked.track_id)
                    if document is None:
                        raise DomainError("DENSE_FACE_SOURCE_REQUIRED", "3D correspondence requires a source identity", 409)
                    if self.neural_face is None or document.faceset_id not in self.neural_face.sources:
                        self._prepare_identity_rigs(document, {})
                    dense_fallback = self.dense_face_unavailable
                if dense_fallback:
                    face_mode = "standard"
                if face_mode != "standard":
                    fade = self._pose_fade.level(tracked.track_id, face.pose, self.pose_hysteresis_degrees)
                    if fade <= 0.0:
                        continue          # the head is turned away from the camera: no face is drawn on it
                    generated = self.neural_face.process(image, face, document, self.swapper, track_id=tracked.track_id,
                                                         frame_index=self.frames_processed, timestamp=frame_time,
                                                         scene_cut=scene_cut, pose_confidence=tracked.confidence)
                    self.dense_enhancer_calls += 1
                    if self.neural_face.last_diag is not None:
                        self.angle_diagnostics.record_swap(self.frames_processed, tracked.track_id,
                                                           {**self.neural_face.last_diag, "presence": fade}, pose=face.pose)
                    projection_seconds += time.perf_counter() - at
                    replacements.append(Replacement(tracked.track_id, generated.image, generated.inverse, -1,
                        "dense_3d_correspondence_neural_ultramax", coverage=generated.coverage,
                        landmarks5=generated.landmarks5, render_mode=face_mode, restored=True, presence=fade))
                    continue
                swap = self.swapper.infer_frame(image, face.landmarks5, pose=face.pose,
                                                track_id=tracked.track_id, frontalization=frontalization,
                                                **({"identity_blend_t": identity_blend_t} if identity_blend_t is not None else {}))
                swap_seconds += time.perf_counter() - at
                if self.portrait.available and pose_blend is None and face.landmarks203 is not None:
                    at = time.perf_counter()
                    crop = landmark_crop(image, face.landmarks106, 512, expansion=2.3, vertical_shift=-0.15)
                    driving_points = transform_points(face.landmarks203, crop.matrix).astype(np.float32)
                    if tracked.track_id not in self.portrait.tracks:
                        source_portrait = swapped_portrait(crop, swap)
                        source_points = self.tracker.dense.infer203(source_portrait,
                            transform_points(face.landmarks106, crop.matrix).astype(np.float32))
                        self.portrait.prepare_track(tracked.track_id, source_portrait, crop.image, source_points, driving_points)
                    generated = self.portrait.infer_frame(tracked.track_id, crop.image, driving_points, strength=strength)
                    inverse, method = crop.inverse, "identity_swap_liveportrait"
                    retarget_seconds += time.perf_counter() - at
                else:
                    generated, inverse, method = swap.image, swap.alignment.inverse, "identity_swap"
                if tps_expression > 0:
                    at = time.perf_counter()
                    warp = apply_expression_to_crop(generated, inverse, face.landmarks106, tps_expression, self._detect_landmarks106)
                    if warp.applied:
                        generated, method = warp.image, method + '_tps_expression'
                    projection_seconds += time.perf_counter() - at
                projection = None
                fallback_reason, reenactment, coverage = dense_fallback, None, None
                if pose_blend is not None:
                    at = time.perf_counter()
                    try:
                        if mode == "pose_3d":
                            projector, fallback_reason = self._pose_3d_projector(tracked.track_id)
                        else:
                            if self.pose_projector is None:
                                self.pose_projector = PoseBlendProjector(
                                    sampling_device="cuda" if self.config.provider == "CUDAExecutionProvider" else "cpu")
                            projector = self.pose_projector
                        observed = transform_points(face.landmarks5, cv2.invertAffineTransform(inverse))
                        projection = projector.render(generated, observed, pose_blend,
                                                      **({"jaw_contour": self._jaw_contour(face, inverse)} if jaw_adaptation else {}))
                        generated, method = projection.image, ('identity_swap_pose_3d' if mode == "pose_3d" else 'identity_swap_tps_mesh_blend')
                        coverage = projection.coverage
                    except DomainError as error:
                        if mode != "pose_3d":
                            raise
                        # 3D Pose never forces a projection the landmarks cannot support: keep the proven swap crop.
                        projection, fallback_reason = None, f"POSE_PROJECTION_REJECTED: {error.code}"
                    projection_seconds += time.perf_counter() - at
                if mode == "reenactment_3d":
                    at = time.perf_counter()
                    generated, coverage, method, fallback_reason, reenactment = self._reenact(
                        tracked, image, generated, inverse, coverage, method)
                    projection_seconds += time.perf_counter() - at
                if abs(face.pose[0]) > PROFILE_TRIM_START:
                    crop_landmarks = transform_points(face.landmarks5, cv2.invertAffineTransform(inverse))
                    profile_mask = profile_aware_mask(generated.shape[:2], crop_landmarks, face.pose[0])
                    coverage = profile_mask if coverage is None else coverage * profile_mask
                replacements.append(Replacement(tracked.track_id, generated, inverse.copy(), swap.reference_index,
                                                method, coverage, projection,
                                                transform_points(face.landmarks5, cv2.invertAffineTransform(inverse)),
                                                mode if fallback_reason is None else "standard", fallback_reason, reenactment))
            replacements = self._presence.apply(self.frames_processed, replacements, gated)
            ended = time.perf_counter()
            self.frames_processed += 1
            self.last_latency = StageLatency((detection_end - began) * 1000, swap_seconds * 1000,
                                             retarget_seconds * 1000, (ended - began) * 1000, projection_seconds * 1000)
            return FrameResult(faces, tuple(replacements), self.last_latency, scene_cut, tuple(withheld))

    def track_frame(self, image: np.ndarray, *, pose_blend: float | None = None,
                    render_mode: str | None = None) -> TrackedFrame:
        """Stage 1: detection, landmarks, pose and track association for one frame (no swap work).

        Pass the result to :meth:`infer_frame` as ``tracked``.  Frames must be tracked in order; only this method
        touches the tracker, so it may safely run on a thread of its own while Stage 2 works on an earlier frame.
        """
        with self.track_lock:
            if not self.prepared or self.tracker is None:
                raise DomainError("SESSION_NOT_PREPARED", "Prepare a source before frame inference", 409)
            image_bgr(image)
            mode = self.render_mode if render_mode is None else render_mode
            if mode not in RENDER_MODES:
                raise ValueError(f"render_mode must be one of {RENDER_MODES}")
            if mode != "standard" and self.dense_face_unavailable:
                mode = "standard"
            if mode == "pose_3d" and pose_blend is None:
                pose_blend = 1.0
            began = time.perf_counter()
            faces = self.tracker.process(image, include203=self.portrait.available and pose_blend is None)
            faces = self._persistent_target_faces(self._apply_angle_trajectories(faces))
            seconds = getattr(self.tracker, "timestamp_seconds", None)
            frame_time = float(seconds) if seconds is not None and np.isfinite(seconds) else self.frames_tracked / 30.0
            self.frames_tracked += 1
            return TrackedFrame(faces, bool(self.tracker.scene_cut), frozenset(self.tracker.active_ids), frame_time,
                                (time.perf_counter() - began) * 1000)

    def begin_fit_ahead(self) -> None:
        """A pipelined render is about to start: let the track thread fit target faces for the swap stage ahead of time."""
        if self.neural_face is not None:
            self.neural_face.begin_fit_ahead(self.frames_processed)

    def prefetch_target_fits(self, image: np.ndarray, tracked: TrackedFrame, *, render_mode: str | None = None) -> None:
        """Stage 1 helper: fit the faces Stage 2 is likely to swap with the dense-3D path (``fit_ahead``).

        Only a *prediction*: Stage 2 re-derives its own decision (identity guard, pose fade, roll latch) and uses a
        precomputed fit only when it was made from identical inputs, so a face predicted wrongly just wastes this work.
        Read-only on session state; the one call per frame also keeps the lookahead's frame counter in step."""
        engine = self.neural_face
        if engine is None or engine.fit_ahead is None or not self.prepared:
            return
        mode = self.render_mode if render_mode is None else render_mode
        candidates = []                                         # an unknown mode or missing dense assets: nothing to fit
        if mode in RENDER_MODES and not self.dense_face_unavailable:
            yaw_limit, pitch_limit = auto_3d_limits()
            roll_limit = dense_3d_max_roll()
            for face in tracked.faces:
                if (self.selected_ids is not None and face.track_id not in self.selected_ids) or not face.swap_allowed:
                    continue
                pose = face.observation.pose
                if mode == "standard" and abs(pose[0]) <= yaw_limit and abs(pose[1]) <= pitch_limit:
                    continue                                    # the standard 2D swap handles it: no dense fit needed
                if abs(pose[2]) > roll_limit:
                    continue
                document = self._document_for(face.track_id)
                if document is None or document.faceset_id not in engine.sources:
                    continue
                candidates.append((face.track_id, face.observation.landmarks5))
        engine.fit_ahead.submit(image, candidates, timestamp=tracked.frame_time, scene_cut=tracked.scene_cut,
                                active_ids=tracked.active_ids)

    def _refine_full_head(self, tracked, image, rendered, coverage, inverse, strength):
        """Condition the portrait appearance encoder on the entire UV render.

        SPADE produces a full 512px portrait. Transfer its fine-scale residual,
        retaining the UV render's low-frequency identity shape and silhouette.
        This uses existing portrait weights; it is not a newly trained head model.
        """
        from roop_3d.reconstruction.full_head import bridge_neural_detail
        side = rendered.shape[0]
        to_crop = cv2.invertAffineTransform(inverse)
        points106 = transform_points(tracked.observation.landmarks106, to_crop) * (512 / side)
        source = cv2.resize(rendered, (512, 512))
        target = self.compositor._crop_to_frame(image, inverse, rendered.shape[:2])
        target = cv2.resize(target, (512, 512))
        target203 = self.tracker.dense.infer203(target, points106.astype(np.float32))
        # Re-encode the current pose-dependent UV render: cached features from a
        # different pose would reintroduce wrong silhouettes at extreme turns.
        source203 = self.tracker.dense.infer203(source, points106.astype(np.float32))
        self.portrait.prepare_track(tracked.track_id, source, target, source203, target203)
        neural = self.portrait.infer_frame(tracked.track_id, target, target203, strength=strength)
        return bridge_neural_detail(rendered, cv2.resize(neural, (side, side)), coverage)

    def _pose_3d_projector(self, track_id: int):
        """(projector, fallback_reason): the source identity's own geometry when an asset exists, else the proven proxy."""
        document = self._document_for(track_id)
        rig = self._rig_for(track_id)
        if document is None or rig is None:
            if self.pose_projector is None:
                self.pose_projector = PoseBlendProjector(
                    sampling_device="cuda" if self.config.provider == "CUDAExecutionProvider" else "cpu")
            return self.pose_projector, self._unavailable_reason(track_id)
        projector = self.pose_projectors.get(document.faceset_id)
        if projector is None:
            projector = self.pose_projectors[document.faceset_id] = PoseBlendProjector.from_identity_asset(
                rig.asset, sampling_device="cuda" if self.config.provider == "CUDAExecutionProvider" else "cpu")
        return projector, None

    def _reenact(self, tracked: TrackedFace, image: np.ndarray, generated: np.ndarray, inverse: np.ndarray,
                 coverage: np.ndarray | None, method: str):
        """3D Reenactment of one face; returns (image, coverage, method, fallback_reason, frame).

        Engaged: the engine render replaces the crop, cross-faded with the proven path's crop while it ramps in/out, and
        its coverage (zero where the atlas has no pixels, e.g. an open mouth) lets the compositor keep the target there.
        Not engaged (or no asset): the proven crop is returned untouched, with the reason."""
        rig = self._rig_for(tracked.track_id)
        if rig is None:
            return generated, coverage, method, self._unavailable_reason(tracked.track_id), None
        engine = self.reenactors.get(tracked.track_id)
        if engine is None:
            engine = self.reenactors[tracked.track_id] = ReenactmentEngine(rig, self.reenactment_settings)
            engine.sampling_device = "cuda" if self.config.provider == "CUDAExecutionProvider" else "cpu"
            engine.renderer_profile = self.config.hardware_profile
        face = tracked.observation
        to_crop = cv2.invertAffineTransform(inverse)
        cut = (self._frame_context.scene_cut if self._frame_context is not None
               else bool(self.tracker is not None and self.tracker.scene_cut))
        try:
            observation = TargetObservation(
                landmarks5=transform_points(face.landmarks5, to_crop), landmarks106=transform_points(face.landmarks106, to_crop),
                timestamp=self._frame_time(), scene_cut=cut, sharpness=self._face_sharpness(image, face.box))
        except (ValueError, TypeError):
            observation = TargetObservation(None, None, self._frame_time(), cut)
        frame = engine.process(observation, generated.shape[:2])
        if not frame.engaged or frame.image is None or frame.coverage is None:
            return generated, coverage, method, frame.reason, frame
        weight = float(frame.weight)
        engine_alpha = frame.coverage * weight
        base_alpha = (np.ones(generated.shape[:2], np.float32) if coverage is None else coverage.astype(np.float32)) * (1.0 - weight)
        total = engine_alpha + base_alpha
        mixed = ((frame.image.astype(np.float32) * engine_alpha[..., None] + generated.astype(np.float32) * base_alpha[..., None])
                 / np.maximum(total[..., None], 1e-6))
        return (np.clip(mixed, 0, 255).astype(np.uint8), np.clip(total, 0.0, 1.0).astype(np.float32),
                "identity_reenactment_3d" + ("" if weight >= 1.0 else "_blend"), None, frame)

    @staticmethod
    def _jaw_contour(face, inverse: np.ndarray) -> np.ndarray:
        """The detector's 33-point face contour in replacement-crop pixels. An implausible 106 layout is never trusted: NaNs make
        the projector report ``contour_invalid`` instead of silently skipping the requested fit."""
        points = None if face.landmarks106 is None else np.asarray(face.landmarks106, np.float64)
        if points is None or points.shape != (106, 2) or not layout_is_plausible(points):
            return np.full((33, 2), np.nan)
        return transform_points(group(points, "contour"), cv2.invertAffineTransform(inverse))

    def _detect_landmarks106(self, crop: np.ndarray) -> np.ndarray | None:
        """106 dense landmarks of a generated crop (the crop is a tight face, so the box is the whole image)."""
        if self.tracker is None:
            return None
        h, w = crop.shape[:2]
        return self.tracker.dense.infer106(crop, np.array((0, 0, w, h), np.float32))

    def capability(self) -> dict:
        return {"prepared": self.prepared, "frames_processed": self.frames_processed,
                "identity_swapper": self.swapper.model.capability() if self.swapper.model else None,
                "portrait": self.portrait.capability(), "profile": self.config.hardware_profile,
                "selected_track_ids": sorted(self.selected_ids) if self.selected_ids is not None else None,
                "pose_method": "three_tier_morphable_model" if self.neural_face is not None else "five_point_geometric_estimate", "dense_landmarks": (68, 106, 203),
                "full_head": {"active": False, "renderer": "disabled_legacy_cg"},
                "hybrid_mesh_renderer": {str(track_id): engine.renderer_status()
                                         for track_id, engine in self.reenactors.items()},
                "dense_face": self.neural_face.capability() if self.neural_face is not None else {
                    "available": False, "active": False,
                    "reason": self.dense_face_unavailable or "Dense face bundle has not been loaded",
                    "raw_mesh_compositing": False},
                "pose_projection": {"state": "AVAILABLE" if self.neural_face is not None or self.pose_projector is not None else "NOT_LOADED",
                                    "geometry": "SEMANTIC_ANATOMICAL_FACE" if self.neural_face is not None else "ESTIMATED_CANONICAL_PROXY_68",
                                    "camera": "FITTED_EXPORT_PINHOLE" if self.neural_face is not None else "ESTIMATED_CROP_PINHOLE",
                                    "active": self.neural_face is not None or self.pose_projector is not None,
                                    "jaw_adaptation": "OPT_IN_CANONICAL_PROXY_JAWLINE_FROM_106_CONTOUR"},
                "full_frame_compositor": self.compositor.capability(),
                "lip_sync": self.lip_sync.capability(),
                "render_modes": {"available": list(RENDER_MODES), "default": self.render_mode,
                                 "fallback": "3D modes require a compatible dense facial model bundle, neural swapper and UltraMax; 6 GB uses perspective UV warping and 12 GB reports hardware-raster failures before falling back",
                                 "identity_assets": dict(self.rig_status),
                                 "active_reenactors": len(self.reenactors)}}

    def prefetch_composite_inputs(self, image: np.ndarray, result: FrameResult, options: CompositeOptions = CompositeOptions()) -> int:
        """Warm the compositor's face-parser cache for ``result`` (no state is touched; safe next to ``composite_frame``)."""
        if not result.replacements or getattr(self, "compositor", None) is None:
            return 0
        return self.compositor.prefetch_parser(image, result.replacements, options)

    def composite_frame(self, image: np.ndarray, result: FrameResult, *,
                        options: CompositeOptions = CompositeOptions(),
                        occluder_masks: dict[int, np.ndarray] | None = None,
                        mask_temporal_context: Mapping[int, Mapping[str, np.ndarray]] | None = None,
                        lip_sync_enabled: bool = False,
                        lip_sync_inputs: Mapping[int, LipSyncFrameInput] | None = None,
                        observer=None, frame_index: int | None = None,
                        timestamp_seconds: float | None = None) -> np.ndarray:
        """Run optional post-swap lip sync, then fuse mouth/face regions into the frame.

        ``frame_index`` / ``timestamp_seconds`` make the call independent of how far Stage 2 has advanced, which the
        asynchronous pipeline needs (defaults reproduce the inline behaviour).  Uses ``composite_lock`` so it can
        overlap :meth:`infer_frame` on another thread. ``mask_temporal_context`` is an optional per-track buffered
        lookahead supplied by offline callers; it contains previous/next masks and their guide images so the
        compositor can repair a verified one-frame matte dropout. The realtime path does not add one-frame latency.
        """
        with self.composite_lock:
            if timestamp_seconds is not None:
                if not np.isfinite(timestamp_seconds):
                    raise ValueError('Frame timestamp must be finite')
                self.compositor.timestamp_seconds = timestamp_seconds
            landmarks = {item.track_id: item.observation.landmarks106 for item in result.observations}
            lip_results = {}
            active_tracks = {item.track_id for item in result.replacements}
            self.lip_sync.gate.begin_frame(active_track_ids=active_tracks if lip_sync_enabled else set(),
                                           scene_cut=result.scene_cut)
            if lip_sync_enabled:
                supplied = lip_sync_inputs or {}
                for replacement in result.replacements:
                    frame_input = supplied.get(replacement.track_id)
                    if frame_input is None:
                        frame_input = LipSyncFrameInput(np.zeros((80, 1), np.float32), 0.0, 0.0)
                    lip_results[replacement.track_id] = self.lip_sync.process_frame(
                        replacement.image, replacement.track_id, frame_input, scene_cut=result.scene_cut)
            return self.compositor.compose(image, result.replacements, landmarks,
                                           scene_cut=result.scene_cut,
                                           frame_index=max(0, self.frames_processed - 1) if frame_index is None else frame_index,
                                           occluder_masks=occluder_masks, mask_temporal_context=mask_temporal_context,
                                           lip_sync_results=lip_results,
                                           options=options, observer=observer)

    @guarded
    def release_session(self) -> None:
        with self.track_lock, self.lock, self.composite_lock:
            self._frame_context = None
            self.frames_tracked = 0
            if self.tracker is not None:
                self.tracker.release()
            self.tracker = None
            self.swapper.release_session()
            self.portrait.release_session()
            self.compositor.release()
            self.lip_sync.release()
            self.selected_ids = None
            self.sources_by_track.clear()
            self.identity_guard = None
            self.prepared = False
            self.last_latency = None
            self.frames_processed = 0
            self.pose_projector = None
            self.render_mode, self.default_source = "standard", None
            self.rigs.clear()
            self.rig_status.clear()
            for engine in self.reenactors.values():
                engine.release()
            self.reenactors.clear()
            self.pose_projectors.clear()
            for engine in self.full_head_engines.values():
                engine.rasterizer.release()
            self.full_head_engines.clear()
            self.head_landmarks = None
            if self.neural_face is not None:
                warm_key = getattr(self, "_warm_key", None)
                if warm_key is not None:
                    # A live preview: keep the engine warm for the next one instead of tearing the 3D head down.
                    from roop_3d.reconstruction.neural_face import WARM_ENGINE
                    self.neural_face.reset_tracks()
                    WARM_ENGINE.put(warm_key, self.neural_face)
                else:
                    self.neural_face.release()
                self.neural_face = None
                self._warm_key = None
            self.angle_trajectories.clear()
            self.angle_frame_index = None
            self._roll_latched.clear()
            self._presence.reset()
            self._pose_fade.reset()


def dense_3d_max_roll() -> float:
    """Head roll (degrees) beyond which the dense 3D path hands the face to the standard swap (``ROOP3D_3D_MAX_ROLL``, default 40)."""
    import os
    try:
        return float(os.environ.get("ROOP3D_3D_MAX_ROLL", 40.0))
    except ValueError:
        return 40.0


def auto_3d_limits() -> tuple[float, float]:
    """(yaw, pitch) in degrees beyond which a face leaves the 2D swap for the dense 3D path (default 30 / 25).

    ``ROOP3D_AUTO_3D_YAW`` / ``ROOP3D_AUTO_3D_PITCH`` change them; a value of 180 disables the automatic switch.
    """
    import os

    def read(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except ValueError:
            return default
    return read("ROOP3D_AUTO_3D_YAW", 30.0), read("ROOP3D_AUTO_3D_PITCH", 25.0)


def prepare_session(source: FaceSetDocument, *, catalog: ModelCatalog | None = None,
                    config: RuntimeConfig = RuntimeConfig(), **kwargs) -> ReferencePipeline:
    session = ReferencePipeline(catalog, config)
    session.prepare_session(source, **kwargs)
    return session


def infer_frame(session: ReferencePipeline, image: np.ndarray, **kwargs) -> FrameResult:
    return session.infer_frame(image, **kwargs)


def release_session(session: ReferencePipeline) -> None:
    session.release_session()
