"""Stage 06 lifecycle: tracking -> identity transfer -> optional 3D reenactment.

Returns model crops and exact crop-to-frame matrices; the full-frame compositor
and its occlusion/segmentation logic remain Stage 09 work.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
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


@dataclass(frozen=True)
class StageLatency:
    detection_ms: float
    swap_ms: float
    retarget_ms: float
    total_ms: float


@dataclass(frozen=True)
class Replacement:
    track_id: int
    image: np.ndarray
    inverse: np.ndarray
    reference_index: int
    method: str


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
        self.selected_ids: frozenset[int] | None = None
        self.prepared = False
        self.lock = threading.RLock()
        self.last_latency: StageLatency | None = None
        self.frames_processed = 0
        register(self, "release_session")

    @guarded
    def prepare_session(self, source: FaceSetDocument, *, track_ids: set[int] | None = None,
                        retarget: Literal["auto", "off", "required"] = "auto") -> dict:
        with self.lock:
            if retarget not in ("auto", "off", "required") or track_ids is not None and (
                    len(track_ids) > 32 or any(type(i) is not int or i < 0 for i in track_ids)):
                raise ValueError("Invalid session selection/policy")
            self.release_session()
            try:
                self.swapper.prepare_session(source)
                self.tracker = FaceTracker(self.catalog, config=self.config)
                self.selected_ids = frozenset(track_ids) if track_ids is not None else None
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
    def infer_frame(self, image: np.ndarray, *, strength: float = 0.35) -> FrameResult:
        with self.lock:
            if not self.prepared or self.tracker is None:
                raise DomainError("SESSION_NOT_PREPARED", "Prepare a source before frame inference", 409)
            image_bgr(image)
            if not np.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("Retargeting strength must be within 0..1")
            began = time.perf_counter()
            faces = self.tracker.process(image, include203=self.portrait.available)
            detection_end = time.perf_counter()
            self.portrait.retain_tracks(self.tracker.active_ids)
            replacements, swap_seconds, retarget_seconds = [], 0.0, 0.0
            for tracked in faces:
                if self.selected_ids is not None and tracked.track_id not in self.selected_ids:
                    continue
                face = tracked.observation
                at = time.perf_counter()
                swap = self.swapper.infer_frame(image, face.landmarks5, pose=face.pose)
                swap_seconds += time.perf_counter() - at
                if self.portrait.available:
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
                replacements.append(Replacement(tracked.track_id, generated, inverse.copy(), swap.reference_index, method))
            ended = time.perf_counter()
            self.frames_processed += 1
            self.last_latency = StageLatency((detection_end - began) * 1000, swap_seconds * 1000,
                                             retarget_seconds * 1000, (ended - began) * 1000)
            return FrameResult(faces, tuple(replacements), self.last_latency, self.tracker.scene_cut)

    def capability(self) -> dict:
        return {"prepared": self.prepared, "frames_processed": self.frames_processed,
                "identity_swapper": self.swapper.model.capability() if self.swapper.model else None,
                "portrait": self.portrait.capability(), "profile": self.config.hardware_profile,
                "selected_track_ids": sorted(self.selected_ids) if self.selected_ids is not None else None,
                "pose_method": "five_point_geometric_estimate", "dense_landmarks": (106, 203),
                "full_frame_compositor": "Stage 09 pending"}

    @guarded
    def release_session(self) -> None:
        with self.lock:
            if self.tracker is not None:
                self.tracker.release()
            self.tracker = None
            self.swapper.release_session()
            self.portrait.release_session()
            self.selected_ids = None
            self.prepared = False
            self.last_latency = None
            self.frames_processed = 0


def prepare_session(source: FaceSetDocument, *, catalog: ModelCatalog | None = None,
                    config: RuntimeConfig = RuntimeConfig(), **kwargs) -> ReferencePipeline:
    session = ReferencePipeline(catalog, config)
    session.prepare_session(source, **kwargs)
    return session


def infer_frame(session: ReferencePipeline, image: np.ndarray, **kwargs) -> FrameResult:
    return session.infer_frame(image, **kwargs)


def release_session(session: ReferencePipeline) -> None:
    session.release_session()
