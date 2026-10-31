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
from roop_3d.reconstruction.pose_blend import PoseBlendProjector, PoseBlendResult


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


@dataclass(frozen=True)
class FrameResult:
    observations: tuple[TrackedFace, ...]
    replacements: tuple[Replacement, ...]
    latency: StageLatency
    scene_cut: bool


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
        self.prepared = False
        self.lock = threading.RLock()
        self.last_latency: StageLatency | None = None
        self.frames_processed = 0
        self.pose_projector: PoseBlendProjector | None = None
        register(self, "release_session")

    @guarded
    def prepare_session(self, source: FaceSetDocument, *, track_ids: set[int] | None = None,
                        sources_by_track: Mapping[int, FaceSetDocument] | None = None,
                        retarget: Literal["auto", "off", "required"] = "auto") -> dict:
        with self.lock:
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
                self.tracker = FaceTracker(self.catalog, config=self.config)
                self.selected_ids = frozenset(track_ids) if track_ids is not None else None
                self.sources_by_track = mapped_sources
                if retarget != "off":
                    self.portrait.prepare_session(required=retarget == "required")
                else:
                    self.portrait.fallback_reason = "LivePortrait disabled explicitly; identity-only path selected"
                self.prepared = True
                return self.capability()
            except BaseException:
                self.release_session()
                raise

    @guarded
    def infer_frame(self, image: np.ndarray, *, strength: float = 0.35, pose_blend: float | None = None) -> FrameResult:
        with self.lock:
            if not self.prepared or self.tracker is None:
                raise DomainError("SESSION_NOT_PREPARED", "Prepare a source before frame inference", 409)
            image_bgr(image)
            if not np.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("Retargeting strength must be within 0..1")
            if pose_blend is not None and (not np.isfinite(pose_blend) or not 0 <= pose_blend <= 1):
                raise ValueError('Mesh pose blend must be within 0..1')
            began = time.perf_counter()
            faces = self.tracker.process(image, include203=self.portrait.available and pose_blend is None)
            detection_end = time.perf_counter()
            self.portrait.retain_tracks(self.tracker.active_ids)
            replacements, swap_seconds, retarget_seconds, projection_seconds = [], 0.0, 0.0, 0.0
            for tracked in faces:
                if self.selected_ids is not None and tracked.track_id not in self.selected_ids:
                    continue
                face = tracked.observation
                at = time.perf_counter()
                swap = self.swapper.infer_frame(image, face.landmarks5, pose=face.pose,
                                                track_id=tracked.track_id)
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
                projection = None
                if pose_blend is not None:
                    at = time.perf_counter()
                    if self.pose_projector is None:
                        self.pose_projector = PoseBlendProjector()
                    observed = transform_points(face.landmarks5, cv2.invertAffineTransform(inverse))
                    projection = self.pose_projector.render(generated, observed, pose_blend)
                    generated, method = projection.image, 'identity_swap_tps_mesh_blend'
                    projection_seconds += time.perf_counter() - at
                replacements.append(Replacement(tracked.track_id, generated, inverse.copy(), swap.reference_index,
                                                method, projection.coverage if projection else None, projection))
            ended = time.perf_counter()
            self.frames_processed += 1
            self.last_latency = StageLatency((detection_end - began) * 1000, swap_seconds * 1000,
                                             retarget_seconds * 1000, (ended - began) * 1000, projection_seconds * 1000)
            return FrameResult(faces, tuple(replacements), self.last_latency, self.tracker.scene_cut)

    def capability(self) -> dict:
        return {"prepared": self.prepared, "frames_processed": self.frames_processed,
                "identity_swapper": self.swapper.model.capability() if self.swapper.model else None,
                "portrait": self.portrait.capability(), "profile": self.config.hardware_profile,
                "selected_track_ids": sorted(self.selected_ids) if self.selected_ids is not None else None,
                "pose_method": "five_point_geometric_estimate", "dense_landmarks": (106, 203),
                "pose_projection": {"state": "AVAILABLE", "geometry": "ESTIMATED_CANONICAL_PROXY_68",
                                    "camera": "ESTIMATED_CROP_PINHOLE", "active": self.pose_projector is not None},
                "full_frame_compositor": self.compositor.capability(),
                "lip_sync": self.lip_sync.capability()}

    def composite_frame(self, image: np.ndarray, result: FrameResult, *,
                        options: CompositeOptions = CompositeOptions(),
                        occluder_masks: dict[int, np.ndarray] | None = None,
                        lip_sync_enabled: bool = False,
                        lip_sync_inputs: Mapping[int, LipSyncFrameInput] | None = None) -> np.ndarray:
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
                                           options=options)

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
            self.prepared = False
            self.last_latency = None
            self.frames_processed = 0
            self.pose_projector = None


def prepare_session(source: FaceSetDocument, *, catalog: ModelCatalog | None = None,
                    config: RuntimeConfig = RuntimeConfig(), **kwargs) -> ReferencePipeline:
    session = ReferencePipeline(catalog, config)
    session.prepare_session(source, **kwargs)
    return session


def infer_frame(session: ReferencePipeline, image: np.ndarray, **kwargs) -> FrameResult:
    return session.infer_frame(image, **kwargs)


def release_session(session: ReferencePipeline) -> None:
    session.release_session()
