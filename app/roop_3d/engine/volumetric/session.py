"""Stage 06 lifecycle: tracking -> identity transfer -> optional 3D reenactment.

Returns model crops and exact crop-to-frame matrices; the full-frame compositor
and its occlusion/segmentation logic remain Stage 09 work.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from collections.abc import Mapping
from typing import Literal
import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.detection.face_tracker import FaceTracker, TrackedFace
from roop_3d.engine.runtime import RuntimeConfig, image_bgr
from roop_3d.engine.volumetric.identity_swapper import IdentitySwapper, SwapResult
from roop_3d.engine.volumetric.preprocess import AlignedCrop, landmark_crop, transform_points
from roop_3d.engine.volumetric.reference_adapter import LivePortraitAdapter
from roop_3d.faceset.schemas import FaceSetDocument
from roop_3d.models.catalog import ModelCatalog
from roop_3d.engine.resources import guarded, register
from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor
from roop_3d.engine.audio_sync.lip_sync_adapter import LipSyncAdapter, LipSyncFrameInput
from roop_3d.reconstruction.expression_tps import apply_expression_to_crop
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
    mask = np.zeros((128, 128), np.float32)
    cv2.ellipse(mask, (64, 71), (50, 59), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 3)
    size = (portrait.image.shape[1], portrait.image.shape[0])
    weight = cv2.warpAffine(mask, local, size)[:, :, None]
    fake = cv2.warpAffine(swap.image, local, size)
    return np.clip(fake * weight + portrait.image * (1 - weight), 0, 255).astype(np.uint8)


class ReferencePipeline:
    def __init__(self, catalog: ModelCatalog | None = None, config: RuntimeConfig = RuntimeConfig()):
        self.catalog, self.config = catalog or ModelCatalog(), config
        self.tracker: FaceTracker | None = None
        self.swapper = IdentitySwapper(self.catalog, config)
        self.portrait = LivePortraitAdapter(self.catalog, config)
        # The compositor and its parser/restorer adapters are lazy; session creation
        # does not activate any additional inference model.
        self.compositor = FrameCompositor(self.catalog, config)
        self.lip_sync = LipSyncAdapter(self.catalog)
        self.selected_ids: frozenset[int] | None = None
        self.sources_by_track: dict[int, FaceSetDocument] = {}
        self.identity_guard = None
        self.prepared = False
        self.lock = threading.RLock()
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
        self.reenactment_settings = ReenactmentSettings()
        register(self, "release_session")

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
                if render_mode != "standard":
                    self._prepare_identity_rigs(source, mapped_sources, identity_store)
                if retarget != "off":
                    self.portrait.prepare_session(required=retarget == "required")
                else:
                    self.portrait.fallback_reason = "LivePortrait disabled explicitly; identity-only path selected"
                self.prepared = True
                return self.capability()
            except BaseException:
                self.release_session()
                raise

    def _prepare_identity_rigs(self, source: FaceSetDocument, mapped: Mapping[int, FaceSetDocument], store=None) -> None:
        """Build / reuse the Stage 3 identity asset of every FaceSet in this session. A FaceSet that cannot support one
        is recorded with its reason and keeps the proven path: the session still prepares."""
        from roop_3d.reconstruction.identity_asset import IdentityAssetStore
        store = store or IdentityAssetStore()
        for document in {d.faceset_id: d for d in (source, *mapped.values())}.values():
            try:
                asset, reused = store.get_or_build(document)
                self.rigs[document.faceset_id] = IdentityRig(asset)
                self.rig_status[document.faceset_id] = {"available": True, "reused": reused, "build_key": asset.build_key,
                                                        "level": asset.manifest.reconstruction_level,
                                                        "layout": asset.manifest.landmark_layout}
            except DomainError as error:
                self.rig_status[document.faceset_id] = {"available": False, "reason": f"{error.code}: {error.message}"[:300]}

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
        self.compositor.histogram_stabilizer = TemporalStabilizer(**config)
        self.swapper.pose_deadband = settings.pose_hysteresis_degrees
        self.swapper._hysteresis = {}

    def set_frame_timestamp(self, seconds: float) -> None:
        if not np.isfinite(seconds):
            raise ValueError('Frame timestamp must be finite')
        if self.tracker is not None:
            self.tracker.timestamp_seconds = seconds
        self.compositor.timestamp_seconds = seconds

    @guarded
    def infer_frame(self, image: np.ndarray, *, strength: float = 0.35, pose_blend: float | None = None,
                    frontalization: float = 0.0, tps_expression: float = 0.0, identity_blend_t: float | None = None,
                    render_mode: str | None = None) -> FrameResult:
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
            if mode == "pose_3d" and pose_blend is None:
                pose_blend = 1.0
            began = time.perf_counter()
            faces = self.tracker.process(image, include203=self.portrait.available and pose_blend is None)
            detection_end = time.perf_counter()
            self.portrait.retain_tracks(self.tracker.active_ids)
            self.reenactors = {key: engine for key, engine in self.reenactors.items() if key in self.tracker.active_ids}
            if self.tracker.scene_cut:
                self.swapper._hysteresis = {}
            elif hasattr(self.swapper, '_hysteresis'):
                self.swapper._hysteresis = {key:value for key,value in self.swapper._hysteresis.items() if key in self.tracker.active_ids}
            replacements, swap_seconds, retarget_seconds, projection_seconds = [], 0.0, 0.0, 0.0
            withheld: list[Withheld] = []
            for tracked in faces:
                if self.selected_ids is not None and tracked.track_id not in self.selected_ids:
                    continue
                if not tracked.swap_allowed:
                    withheld.append(Withheld(tracked.track_id, tracked.reason or tracked.state, tracked.state, tracked.confidence))
                    continue
                if self.identity_guard is not None:
                    verdict = self.identity_guard.check(tracked)
                    if not verdict.allowed:
                        withheld.append(Withheld(tracked.track_id, verdict.reason or "IDENTITY_UNVERIFIED",
                                                 tracked.state, verdict.similarity))
                        continue
                face = tracked.observation
                at = time.perf_counter()
                swap = self.swapper.infer_frame(image, face.landmarks5, pose=face.pose,
                                                track_id=tracked.track_id, frontalization=frontalization,
                                                **({"identity_blend_t": identity_blend_t} if identity_blend_t is not None else {}))
                swap_seconds += time.perf_counter() - at
                if self.portrait.available and pose_blend is None:
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
                fallback_reason, reenactment, coverage = None, None, None
                if pose_blend is not None:
                    at = time.perf_counter()
                    try:
                        if mode == "pose_3d":
                            projector, fallback_reason = self._pose_3d_projector(tracked.track_id)
                        else:
                            if self.pose_projector is None:
                                self.pose_projector = PoseBlendProjector()
                            projector = self.pose_projector
                        observed = transform_points(face.landmarks5, cv2.invertAffineTransform(inverse))
                        projection = projector.render(generated, observed, pose_blend)
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
                replacements.append(Replacement(tracked.track_id, generated, inverse.copy(), swap.reference_index,
                                                method, coverage, projection,
                                                transform_points(face.landmarks5, cv2.invertAffineTransform(inverse)),
                                                mode if fallback_reason is None else "standard", fallback_reason, reenactment))
            ended = time.perf_counter()
            self.frames_processed += 1
            self.last_latency = StageLatency((detection_end - began) * 1000, swap_seconds * 1000,
                                             retarget_seconds * 1000, (ended - began) * 1000, projection_seconds * 1000)
            return FrameResult(faces, tuple(replacements), self.last_latency, self.tracker.scene_cut, tuple(withheld))

    def _pose_3d_projector(self, track_id: int):
        """(projector, fallback_reason): the source identity's own geometry when an asset exists, else the proven proxy."""
        document = self._document_for(track_id)
        rig = self._rig_for(track_id)
        if document is None or rig is None:
            if self.pose_projector is None:
                self.pose_projector = PoseBlendProjector()
            return self.pose_projector, self._unavailable_reason(track_id)
        projector = self.pose_projectors.get(document.faceset_id)
        if projector is None:
            projector = self.pose_projectors[document.faceset_id] = PoseBlendProjector.from_identity_asset(rig.asset)
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
        face = tracked.observation
        to_crop = cv2.invertAffineTransform(inverse)
        cut = bool(self.tracker is not None and self.tracker.scene_cut)
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
                "pose_method": "five_point_geometric_estimate", "dense_landmarks": (106, 203),
                "pose_projection": {"state": "AVAILABLE", "geometry": "ESTIMATED_CANONICAL_PROXY_68",
                                    "camera": "ESTIMATED_CROP_PINHOLE", "active": self.pose_projector is not None},
                "full_frame_compositor": self.compositor.capability(),
                "lip_sync": self.lip_sync.capability(),
                "render_modes": {"available": list(RENDER_MODES), "default": self.render_mode,
                                 "fallback": "deterministic: the proven identity-swap path whenever the 3D geometry is not reliable",
                                 "identity_assets": dict(self.rig_status),
                                 "active_reenactors": len(self.reenactors)}}

    def composite_frame(self, image: np.ndarray, result: FrameResult, *,
                        options: CompositeOptions = CompositeOptions(),
                        occluder_masks: dict[int, np.ndarray] | None = None,
                        lip_sync_enabled: bool = False,
                        lip_sync_inputs: Mapping[int, LipSyncFrameInput] | None = None,
                        observer=None) -> np.ndarray:
        """Run optional post-swap lip sync, then fuse mouth/face regions into the frame."""
        with self.lock:
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
                                           frame_index=max(0, self.frames_processed - 1),
                                           occluder_masks=occluder_masks, lip_sync_results=lip_results,
                                           options=options, observer=observer)

    @guarded
    def release_session(self) -> None:
        with self.lock:
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
            self.reenactors.clear()
            self.pose_projectors.clear()


def prepare_session(source: FaceSetDocument, *, catalog: ModelCatalog | None = None,
                    config: RuntimeConfig = RuntimeConfig(), **kwargs) -> ReferencePipeline:
    session = ReferencePipeline(catalog, config)
    session.prepare_session(source, **kwargs)
    return session


def infer_frame(session: ReferencePipeline, image: np.ndarray, **kwargs) -> FrameResult:
    return session.infer_frame(image, **kwargs)


def release_session(session: ReferencePipeline) -> None:
    session.release_session()
