"""Orchestrate per-face parsing, restoration, harmonization and full-frame fusion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from collections.abc import Callable

import cv2
import numpy as np

from roop_3d.compositor.boundary_fusion import BlendMode, fuse_crop
from roop_3d.compositor.color_match import ColorStats, apply_lab_stats, estimate_lab_stats, masked_histogram_match
from roop_3d.compositor.lighting import relight_spherical_harmonics
from roop_3d.compositor.masks import BisenetFaceParser, FaceParser, build_face_mask, landmark_hull_mask
from roop_3d.compositor.mouth import composite_mouth_region, lower_face_soft_mask
from roop_3d.compositor.occlusion import combine_occluders, semantic_occluder_mask, subtract_occluders
from roop_3d.compositor.restoration import GpenRestorer, blend_restored_roi
from roop_3d.compositor.temporal import TemporalSmoother
from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import RuntimeConfig, image_bgr


@dataclass(frozen=True)
class CompositeOptions:
    mode: BlendMode = BlendMode.FEATHER
    restoration_blend: float = 0.0
    color_method: str = "lab"
    color_strength: float = 0.65
    relight: bool = False
    semantic_confidence_threshold: float = 0.55
    feather_px: float = 5.0
    pyramid_levels: int = 4
    swap_weight: float = 1.0
    erosion_px: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.restoration_blend <= 1 or not 0 <= self.color_strength <= 1:
            raise ValueError("Blend ratios must be within 0..1")
        if not np.isfinite(self.swap_weight) or not 0 <= self.swap_weight <= 1 or not np.isfinite(self.erosion_px) or not -32 <= self.erosion_px <= 32:
            raise ValueError('Swap weight or erosion is outside its bounds')
        if self.color_method not in {"lab", "histogram"}:
            raise ValueError("color_method must be 'lab' or 'histogram'")
        if not 0 <= self.semantic_confidence_threshold <= 1 or self.feather_px < 0 or self.pyramid_levels < 1:
            raise ValueError("Invalid mask or pyramid configuration")


class FrameCompositor:
    """Reusable frame compositor; neural sessions are loaded only on first use."""

    def __init__(self, catalog=None, config: RuntimeConfig = RuntimeConfig(), *, parser: FaceParser | None = None,
                 restorer: GpenRestorer | None = None, temporal: TemporalSmoother | None = None):
        self.catalog = catalog
        self.config = config
        self.parser = parser if parser is not None else BisenetFaceParser(catalog, config)
        self.restorer = restorer
        self.temporal = temporal or TemporalSmoother()
        self.frames_composited = 0

    @staticmethod
    def _crop_to_frame(frame: np.ndarray, inverse: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        frame_to_crop = cv2.invertAffineTransform(np.asarray(inverse, dtype=np.float32))
        return cv2.warpAffine(frame, frame_to_crop, (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    @staticmethod
    def _landmarks_to_crop(landmarks: np.ndarray | None, inverse: np.ndarray) -> np.ndarray | None:
        if landmarks is None:
            return None
        points = np.asarray(landmarks, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise DomainError("COMPOSITOR_LANDMARKS_INVALID", "Target landmarks must be finite Nx2 points", 422)
        frame_to_crop = cv2.invertAffineTransform(np.asarray(inverse, dtype=np.float32))
        return cv2.transform(points[None], frame_to_crop)[0]

    @staticmethod
    def _color_process(swapped: np.ndarray, target: np.ndarray, alpha: np.ndarray, method: str,
                       strength: float) -> tuple[np.ndarray, ColorStats]:
        stats = estimate_lab_stats(swapped, target, alpha)
        if method == "histogram":
            matched = masked_histogram_match(swapped, target, alpha)
            if strength < 1:
                active = alpha > 0
                partial = swapped.copy()
                partial[active] = (swapped.astype(np.float32)[active] * (1 - strength)
                                   + matched.astype(np.float32)[active] * strength).round().astype(np.uint8)
                matched = partial
            return matched, stats
        return apply_lab_stats(swapped, stats, alpha, strength=strength), stats

    def compose(self, frame: np.ndarray, replacements, landmarks_by_track: Mapping[int, np.ndarray] | None = None,
                *, scene_cut: bool = False, frame_index: int | None = None,
                occluder_masks: Mapping[int, np.ndarray] | None = None,
                lip_sync_results: Mapping[int, object] | None = None,
                options: CompositeOptions = CompositeOptions(),
                observer: Callable | None = None) -> np.ndarray:
        image_bgr(frame)
        landmarks_by_track = landmarks_by_track or {}
        occluder_masks = occluder_masks or {}
        lip_sync_results = lip_sync_results or {}
        active_tracks = set(landmarks_by_track) | {item.track_id for item in replacements}
        self.temporal.begin_frame(frame_index, active_track_ids=active_tracks, scene_cut=scene_cut)
        output = frame.copy()
        for replacement in sorted(replacements, key=lambda item: item.track_id):
            crop = np.asarray(replacement.image)
            if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3:
                raise DomainError("COMPOSITOR_CROP_INVALID", "Replacement crop must be uint8 BGR", 422)
            target_crop = self._crop_to_frame(frame, replacement.inverse, crop.shape[:2])
            points = self._landmarks_to_crop(landmarks_by_track.get(replacement.track_id), replacement.inverse)
            hull = landmark_hull_mask(crop.shape, points, feather_px=options.feather_px)

            probabilities = None
            try:
                probabilities = self.parser.predict(target_crop) if self.parser is not None else None
            except (DomainError, RuntimeError, OSError, ValueError):
                # A missing or unsupported optional local parser selects the documented hull fallback.
                probabilities = None
            face_mask = build_face_mask(crop.shape, points, probabilities=probabilities,
                                        confidence_threshold=options.semantic_confidence_threshold,
                                        feather_px=options.feather_px)
            semantic_front = (semantic_occluder_mask(probabilities)
                              if probabilities is not None and face_mask.source.startswith("bisenet") else None)
            explicit_front = occluder_masks.get(replacement.track_id)
            front = combine_occluders(semantic_front, explicit_front)
            alpha = subtract_occluders(face_mask.alpha, front)
            if options.erosion_px:
                radius = max(1, round(abs(options.erosion_px)))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
                alpha = cv2.erode(alpha, kernel) if options.erosion_px > 0 else cv2.dilate(alpha, kernel)
            alpha = alpha * options.swap_weight
            if observer is not None:
                observer('swapper', crop, replacement.inverse, alpha)
                observer('matte', (alpha * 255).round().astype(np.uint8), replacement.inverse, alpha)
            if not alpha.any():
                continue

            enhanced = crop
            if options.restoration_blend > 0:
                if self.restorer is None:
                    raise DomainError("RESTORER_UNAVAILABLE", "A positive restoration blend requires an explicit local restorer", 503)
                restored = self.restorer.restore(crop)
                enhanced = blend_restored_roi(crop, restored, alpha, options.restoration_blend)

            lip_result = lip_sync_results.get(replacement.track_id)
            if observer is not None:
                observer('restorer', enhanced, replacement.inverse, alpha)
            if lip_result is not None and bool(getattr(lip_result, "applied", False)):
                lip_image = np.asarray(getattr(lip_result, "image", None))
                lip_blend = float(getattr(lip_result, "blend", 0.0))
                mouth_mask = getattr(lip_result, "mouth_mask", None)
                if mouth_mask is None:
                    mouth_mask = lower_face_soft_mask(crop.shape, probabilities, feather_px=options.feather_px)
                enhanced = composite_mouth_region(enhanced, lip_image, mouth_mask, lip_blend)

            harmonized, color_stats = self._color_process(enhanced, target_crop, alpha,
                                                           options.color_method, options.color_strength)
            if options.relight:
                skin = face_mask.skin if face_mask.source.startswith("bisenet") else alpha
                harmonized = relight_spherical_harmonics(harmonized, target_crop, skin, alpha)
            temporal = self.temporal.update(replacement.track_id, replacement.inverse, alpha / max(options.swap_weight, 1e-8), color_stats,
                                            frame_index=frame_index, scene_cut=False)
            if observer is not None:
                observer('color', harmonized, replacement.inverse, alpha)
            effective_alpha = temporal.alpha * options.swap_weight
            output = fuse_crop(output, harmonized, temporal.inverse, effective_alpha,
                               mode=options.mode, levels=options.pyramid_levels)
        self.frames_composited += 1
        return output

    def capability(self) -> dict:
        parser = self.parser.capability() if hasattr(self.parser, "capability") else {
            "weights": "UNKNOWN", "provenance": "UNKNOWN", "license": "UNKNOWN", "inference": "UNKNOWN"}
        restorer = self.restorer.capability() if self.restorer is not None else {
            "weights": "NOT_CONFIGURED", "inference": "UNKNOWN"}
        return {"state": "AVAILABLE", "parser": parser, "restorer": restorer,
                "temporal_tracks": self.temporal.track_count,
                "temporal_cache_bytes": self.temporal.cache_bytes,
                "temporal_cache_limit_bytes": self.temporal.max_cache_bytes,
                "frames_composited": self.frames_composited}

    def release(self) -> None:
        if self.parser is not None and hasattr(self.parser, "release"):
            self.parser.release()
        if self.restorer is not None:
            self.restorer.release()
        self.temporal.clear()
