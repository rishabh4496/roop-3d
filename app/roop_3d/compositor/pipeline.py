"""Orchestrate per-face parsing, restoration, harmonization and full-frame fusion.

Two mask/fusion strategies share every other stage (restoration, mouth, colour, relighting, temporal
state, boundary fusion):

* ``occlusion_mode="standard"`` (default): the original Stage 09 behaviour, unchanged.
* ``occlusion_mode="robust"``: Stage 5 unified face + occlusion analysis (``occlusion_analysis``),
  flow-aware temporal matte (``occlusion_tracker``), depth-ordered multi-face compositing and
  frame-space matte refinement / quality fusion (``robust_fusion``). Every dependency that is absent
  is reported through ``capability()`` / ``last_reports``; nothing is simulated.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import functools
import time
from typing import Mapping
from collections.abc import Callable

import cv2
import numpy as np

from roop_3d.compositor.boundary_fusion import BlendMode, fuse_crop
from roop_3d.compositor.color_match import ColorStats, apply_lab_stats, estimate_lab_stats, masked_histogram_match
from roop_3d.compositor.lighting import relight_spherical_harmonics
from roop_3d.compositor.matte import border_fade
from roop_3d.compositor.masks import BisenetFaceParser, FaceParser, build_face_mask, landmark_hull_mask
from roop_3d.compositor.mouth import composite_mouth_region, lower_face_soft_mask
from roop_3d.compositor.occlusion import combine_occluders, semantic_occluder_mask, subtract_occluders
from roop_3d.compositor.restoration import GpenRestorer, blend_restored_roi
from roop_3d.compositor.temporal import TemporalSmoother
from roop_3d.temporal import TemporalMaskEMA
from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import RuntimeConfig, image_bgr

OCCLUSION_MODES = ("standard", "robust")


@dataclass(frozen=True)
class RobustOptions:
    """Knobs of the Stage 5 path; only read when ``occlusion_mode == "robust"``."""
    analysis_size: int = 256
    refine_matte: bool = True
    edge_decontamination: bool = True
    sharpness_matching: bool = True
    local_illumination: float = 0.6
    texture_strength: float = 0.7
    restoration_guard: bool = True
    linear_light: bool = True
    cross_face_occlusion: bool = True
    network_border_trust: float = 0.6     # weight of the networks' "not face" vote at the landmark-hull border
    matte_contrast: float = 1.5           # plateau tightening applied after guided matte refinement (1 = off)
    occlusion_release: float = 0.75       # lingering of a dropped occluder pixel (0 = none); never delays onset. 0.75 measured best on real clips

    def __post_init__(self) -> None:
        if not 64 <= self.analysis_size <= 1024:
            raise ValueError("analysis_size must be within 64..1024")
        if not 1.0 <= self.matte_contrast <= 4.0:
            raise ValueError("matte_contrast must be within 1..4")
        for name in ("local_illumination", "texture_strength", "network_border_trust", "occlusion_release"):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be within 0..1")


@dataclass(frozen=True)
class CompositeOptions:
    mode: BlendMode = BlendMode.FEATHER
    restoration_blend: float = 0.0
    fidelity_weight: float = 0.5
    color_method: str = "lab"
    color_strength: float = 0.65
    relight: bool = False
    relight_strength: float = 0.8
    specular_sheen: float = 0.25
    shadow_preservation: float = 0.85
    ultramax_small_face_mode: str = "bypass"
    ultramax_min_native_pixels: int = 320
    lighting_mode: str = "auto"
    manual_light_yaw: float = 35.0
    manual_light_pitch: float = 25.0
    semantic_confidence_threshold: float = 0.55
    feather_px: float = 5.0
    pyramid_levels: int = 4
    swap_weight: float = 1.0
    erosion_px: float = 0.0
    occlusion_mode: str = "standard"
    robust: RobustOptions = field(default_factory=RobustOptions)
    # Stage 15: blend arithmetic in linear light, with the display transfer named explicitly (never assumed sRGB).
    linear_light: bool = False
    blend_transfer: str = "srgb"

    def __post_init__(self) -> None:
        if not np.isfinite(self.fidelity_weight) or not 0 <= self.fidelity_weight <= 1:
            raise ValueError("CodeFormer fidelity must be within 0..1")
        if not 0 <= self.restoration_blend <= 1 or not 0 <= self.color_strength <= 1:
            raise ValueError("Blend ratios must be within 0..1")
        if not np.isfinite(self.swap_weight) or not 0 <= self.swap_weight <= 1 or not np.isfinite(self.erosion_px) or not -32 <= self.erosion_px <= 32:
            raise ValueError('Swap weight or erosion is outside its bounds')
        if self.color_method not in {"lab", "reinhard_lab", "histogram"}:
            raise ValueError("color_method must be 'lab', 'reinhard_lab', or 'histogram'")
        if self.ultramax_small_face_mode not in {"bypass", "adaptive", "off"} or not 64 <= self.ultramax_min_native_pixels <= 1024:
            raise ValueError("Invalid UltraMax small-face policy")
        if not 0 <= self.semantic_confidence_threshold <= 1 or self.feather_px < 0 or self.pyramid_levels < 1:
            raise ValueError("Invalid mask or pyramid configuration")
        if self.occlusion_mode not in OCCLUSION_MODES:
            raise ValueError(f"occlusion_mode must be one of {OCCLUSION_MODES}")
        from roop_3d.color.transfer import BLEND_TRANSFERS
        if self.blend_transfer not in BLEND_TRANSFERS:
            raise ValueError(f"blend_transfer must be one of {BLEND_TRANSFERS}")


@dataclass
class _FaceContext:
    """Per-face state of the robust path between its analysis pass and its composite pass."""
    replacement: object
    crop: np.ndarray
    target_crop: np.ndarray            # native-resolution target crop (colour statistics)
    work_crop: np.ndarray              # analysis-resolution target crop
    work_inverse: np.ndarray           # crop(work) -> frame
    scale: float
    analysis: object
    front: np.ndarray | None
    motion: tuple[float, float]        # face displacement in FRAME pixels per frame (drives motion-blur matching)
    state: object
    head: np.ndarray                   # analysis-resolution head mask (hull + hair + ears), for depth ordering
    depth: float
    centre_y: float
    report: dict = field(default_factory=dict)


class FrameCompositor:
    """Reusable frame compositor; neural sessions are loaded only on first use."""

    def __init__(self, catalog=None, config: RuntimeConfig = RuntimeConfig(), *, parser: FaceParser | None = None,
                 restorer: GpenRestorer | None = None, temporal: TemporalSmoother | None = None,
                 analyzer=None):
        self.catalog = catalog
        from roop_3d.compositor.stream_restoration import MaskSeamWatchdog
        self.watchdog = MaskSeamWatchdog()
        self.config = config
        self._parser_injected = parser is not None
        self.parser = parser if parser is not None else BisenetFaceParser(catalog, config)
        self.restorer = restorer
        self.temporal = temporal or TemporalSmoother()
        self.mask_ema = TemporalMaskEMA(alpha=.65, max_tracks=32)
        from roop_3d.temporal_stabilizer import TemporalStabilizer
        self.histogram_stabilizer = TemporalStabilizer(min_cutoff=.15, beta=.04)
        self._analyzer = analyzer
        self._occlusion_tracker = None
        self._face_centres: dict[int, tuple[int | None, np.ndarray]] = {}
        self.frames_composited = 0
        self.last_timing: dict[str, float] = {}
        # Stage 5: per-track report of the last robust frame (tier, confidence, reasons, fusion metrics).
        self.last_reports: dict[int, dict] = {}

    # -- lazy Stage 5 components ----------------------------------------------------------------------
    @property
    def analyzer(self):
        if self._analyzer is None:
            from roop_3d.compositor.occlusion_analysis import OcclusionAnalyzer
            self._analyzer = OcclusionAnalyzer(self.catalog, self.config,
                                               parser=self.parser if self._parser_injected else None)
        return self._analyzer

    @property
    def occlusion_tracker(self):
        if self._occlusion_tracker is None:
            from roop_3d.compositor.occlusion_tracker import OcclusionTracker
            self._occlusion_tracker = OcclusionTracker()
        return self._occlusion_tracker

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
                       strength: float, smooth_histogram=None) -> tuple[np.ndarray, ColorStats]:
        stats = estimate_lab_stats(swapped, target, alpha)
        if method == "histogram":
            matched = masked_histogram_match(swapped, target, alpha, smooth_histogram=smooth_histogram)
            if strength < 1:
                active = alpha > 0
                partial = swapped.copy()
                partial[active] = (swapped.astype(np.float32)[active] * (1 - strength)
                                   + matched.astype(np.float32)[active] * strength).round().astype(np.uint8)
                matched = partial
            return matched, stats
        return apply_lab_stats(swapped, stats, alpha, strength=strength), stats

    @staticmethod
    def _manual_mask(value, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return (include, exclude); legacy grayscale masks remain exclude-only."""
        zeros = np.zeros(shape, np.float32)
        if value is None:
            return zeros, zeros.copy()
        if isinstance(value, (tuple, list)) and len(value) == 2:
            include, exclude = (np.asarray(item, np.float32) for item in value)
        else:
            include, exclude = zeros, np.asarray(value, np.float32)
        if (include.shape != shape or exclude.shape != shape or not np.isfinite(include).all()
                or not np.isfinite(exclude).all() or np.any((include < 0) | (include > 1))
                or np.any((exclude < 0) | (exclude > 1))):
            raise DomainError("MANUAL_MASK_INVALID", "Manual include/exclude maps must match the face crop and stay within 0..1", 422)
        return include, exclude

    # -- shared enhancement: restoration, mouth, colour, relighting -----------------------------------
    def _enhance(self, replacement, crop, target_crop, alpha, *, probabilities, skin, options, observer,
                 lip_result, frame_index, protect=None, mouth_mask=None, local_illumination: float = 0.0):
        restorer_seconds = 0.0
        enhanced = crop
        if options.restoration_blend > 0:
            if self.restorer is None:
                raise DomainError("RESTORER_UNAVAILABLE", "A positive restoration blend requires an explicit local restorer", 503)
            restore_began = time.perf_counter()
            if hasattr(self.restorer, 'fidelity_weight'):
                self.restorer.fidelity_weight = options.fidelity_weight
            elif options.fidelity_weight != .5:
                raise DomainError('FIDELITY_UNAVAILABLE', 'The selected restorer has no CodeFormer fidelity input', 422)
            if hasattr(self.restorer, 'restore_aligned'):
                if hasattr(self.restorer, 'small_face_mode'):
                    self.restorer.small_face_mode = options.ultramax_small_face_mode
                    self.restorer.min_native_pixels = options.ultramax_min_native_pixels
                    linear = np.asarray(replacement.inverse, np.float32)[:, :2]
                    native_side = int(round(min(crop.shape[1] * np.linalg.norm(linear[:, 0]),
                                                crop.shape[0] * np.linalg.norm(linear[:, 1]))))
                    restored = self.restorer.restore_aligned(crop, getattr(replacement, 'landmarks5', None),
                                                              native_face_size_px=native_side)
                else:
                    restored = self.restorer.restore_aligned(crop, getattr(replacement, 'landmarks5', None))
            else:
                restored = self.restorer.restore(crop)
            enhanced = blend_restored_roi(crop, restored, alpha, options.restoration_blend, protect=protect)
            restorer_seconds += time.perf_counter() - restore_began

        if observer is not None:
            observer('restorer', enhanced, replacement.inverse, alpha)
        if lip_result is not None and bool(getattr(lip_result, "applied", False)):
            lip_image = np.asarray(getattr(lip_result, "image", None))
            lip_blend = float(getattr(lip_result, "blend", 0.0))
            lip_mask = getattr(lip_result, "mouth_mask", None)
            if lip_mask is None:
                lip_mask = mouth_mask if mouth_mask is not None else lower_face_soft_mask(
                    crop.shape, probabilities, feather_px=options.feather_px)
            enhanced = composite_mouth_region(enhanced, lip_image, lip_mask, lip_blend)

        harmonized, color_stats = self._color_process(
            enhanced, target_crop, alpha, options.color_method, options.color_strength,
            lambda histogram: self.histogram_stabilizer.filter(
                replacement.track_id, "histogram", histogram,
                getattr(self, "timestamp_seconds", (frame_index if frame_index is not None else self.frames_composited) / 30.0)))
        if local_illumination > 0:
            from roop_3d.compositor.quality import local_illumination_match
            harmonized = local_illumination_match(harmonized, target_crop, alpha, strength=local_illumination)
        if options.relight:
            from roop_3d.relighting_3d import relighter
            relit_crop, normal_vis, sh_vis, _ = relighter.relight(
                harmonized, target_crop, skin_mask=skin, face_mask=alpha,
                relight_strength=getattr(options, "relight_strength", 0.8),
                specular_sheen=getattr(options, "specular_sheen", 0.25),
                shadow_preservation=getattr(options, "shadow_preservation", 0.85),
                lighting_mode=getattr(options, "lighting_mode", "auto"),
                manual_light_yaw=getattr(options, "manual_light_yaw", 35.0),
                manual_light_pitch=getattr(options, "manual_light_pitch", 25.0))
            harmonized = relit_crop
            if observer is not None:
                observer('normal_map', normal_vis, replacement.inverse, alpha)
                observer('sh_lighting', sh_vis, replacement.inverse, alpha)
        return harmonized, color_stats, restorer_seconds

    def compose(self, frame: np.ndarray, replacements, landmarks_by_track: Mapping[int, np.ndarray] | None = None,
                *, scene_cut: bool = False, frame_index: int | None = None,
                occluder_masks: Mapping[int, np.ndarray] | None = None,
                lip_sync_results: Mapping[int, object] | None = None,
                options: CompositeOptions = CompositeOptions(),
                observer: Callable | None = None) -> np.ndarray:
        image_bgr(frame)
        if any(getattr(item, "full_head", False) for item in replacements) and options.occlusion_mode == "standard":
            options = replace(options, occlusion_mode="robust")
        landmarks_by_track = landmarks_by_track or {}
        occluder_masks = occluder_masks or {}
        lip_sync_results = lip_sync_results or {}
        active_tracks = set(landmarks_by_track) | {item.track_id for item in replacements}
        self.temporal.begin_frame(frame_index, active_track_ids=active_tracks, scene_cut=scene_cut)
        self.mask_ema.retain(active_tracks)
        if scene_cut:
            self.mask_ema.clear()
        if scene_cut: self.histogram_stabilizer.reset()
        self.histogram_stabilizer.retain(active_tracks)
        if options.occlusion_mode == "robust":
            return self._compose_robust(frame, replacements, landmarks_by_track, active_tracks, scene_cut, frame_index,
                                        occluder_masks, lip_sync_results, options, observer)
        output = frame.copy()
        compose_began, restorer_seconds = time.perf_counter(), 0.0
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
            if getattr(replacement, "full_head", False):
                # Mesh coverage defines source anatomy. Target parsing can veto
                # foreground occlusions but cannot shrink it to the target hull.
                face_mask = replace(face_mask, alpha=np.ones(crop.shape[:2], np.float32))
            semantic_front = (semantic_occluder_mask(probabilities)
                              if probabilities is not None and face_mask.source.startswith("bisenet") else None)
            manual_include, manual_exclude = self._manual_mask(occluder_masks.get(replacement.track_id), crop.shape[:2])
            explicit_front = manual_exclude if manual_exclude.any() else None
            front = combine_occluders(semantic_front, explicit_front)
            self.watchdog.observe(replacement.track_id, face_mask.alpha, kind="geometric")
            if front is not None:
                self.watchdog.observe(replacement.track_id, np.clip(front, 0, 1), kind="occluder")
            if observer is not None:
                observer('occluder', front if front is not None else np.zeros(crop.shape[:2], np.float32), replacement.inverse, face_mask.alpha)
            geometric = self.mask_ema.update(replacement.track_id, face_mask.alpha, frame=frame_index, kind="geometric")
            manual_include = self.mask_ema.update(replacement.track_id, manual_include, frame=frame_index, kind="manual_include")
            if front is not None:
                front = self.mask_ema.update(replacement.track_id, front, frame=frame_index, kind="occluder")
            alpha = subtract_occluders(np.maximum(geometric, manual_include), front)
            alpha = self.watchdog.feather(replacement.track_id, alpha, kind="composite")
            # Feathering may cross an occlusion boundary; restore the foreground veto exactly.
            if front is not None:
                alpha *= (1.0 - front)
            if options.erosion_px:
                radius = max(1, round(abs(options.erosion_px)))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
                alpha = cv2.erode(alpha, kernel) if options.erosion_px > 0 else cv2.dilate(alpha, kernel)
            alpha = alpha * options.swap_weight
            coverage = getattr(replacement, 'coverage', None)
            if coverage is not None:
                coverage = np.asarray(coverage, np.float32)
                if coverage.shape != alpha.shape or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
                    raise DomainError('PROJECTION_COVERAGE_INVALID', 'Projected coverage must be finite crop-space values in 0..1', 422)
                alpha *= coverage
            if observer is not None:
                observer('swapper', crop, replacement.inverse, alpha)
                observer('matte', (alpha * 255).round().astype(np.uint8), replacement.inverse, alpha)
            if not alpha.any():
                continue

            skin = face_mask.skin if getattr(face_mask, "source", "").startswith("bisenet") else alpha
            harmonized, color_stats, spent = self._enhance(
                replacement, crop, target_crop, alpha, probabilities=probabilities, skin=skin, options=options,
                observer=observer, lip_result=lip_sync_results.get(replacement.track_id), frame_index=frame_index)
            restorer_seconds += spent
            temporal = self.temporal.update(replacement.track_id, replacement.inverse, alpha / max(options.swap_weight, 1e-8), color_stats,
                                            frame_index=frame_index, scene_cut=False)
            if observer is not None:
                observer('color', harmonized, replacement.inverse, alpha)
            effective_alpha = temporal.alpha * options.swap_weight
            if coverage is not None:
                effective_alpha = np.minimum(effective_alpha, coverage)
            if observer is not None:
                observer('vfx', crop, temporal.inverse, effective_alpha)
            output = fuse_crop(output, harmonized, temporal.inverse, effective_alpha,
                               mode=options.mode, levels=options.pyramid_levels,
                               linear_light=options.linear_light, transfer=options.blend_transfer)
        self.frames_composited += 1
        total_ms = (time.perf_counter() - compose_began) * 1000
        restorer_ms = restorer_seconds * 1000
        # Masking = parser + matte + occluders + colour/temporal/fusion work that is not the restorer.
        self.last_timing = {"masking": max(0.0, total_ms - restorer_ms), "restorer": restorer_ms}
        return output

    # -- Stage 5: robust occlusion path ---------------------------------------------------------------
    @staticmethod
    def _scaled_inverse(inverse: np.ndarray, scale: float) -> np.ndarray:
        scaled = np.asarray(inverse, np.float32).copy()
        scaled[:, :2] /= scale
        return scaled

    @staticmethod
    def _compose_affine(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
        """2x3 matrix applying ``inner`` first, then ``outer``."""
        a = np.vstack((np.asarray(outer, np.float64), (0, 0, 1)))
        b = np.vstack((np.asarray(inner, np.float64), (0, 0, 1)))
        return (a @ b)[:2].astype(np.float32)

    def _analyse_face(self, frame, replacement, landmarks, occluder_masks, options, scene_cut, frame_index) -> _FaceContext:
        crop = np.asarray(replacement.image)
        if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3:
            raise DomainError("COMPOSITOR_CROP_INVALID", "Replacement crop must be uint8 BGR", 422)
        if crop.shape[0] != crop.shape[1]:
            raise DomainError("COMPOSITOR_CROP_INVALID", "The robust path requires a square replacement crop", 422)
        side = crop.shape[0]
        work = max(side, options.robust.analysis_size)
        scale = work / side
        work_inverse = self._scaled_inverse(replacement.inverse, scale)
        target_crop = self._crop_to_frame(frame, replacement.inverse, crop.shape[:2])
        work_crop = self._crop_to_frame(frame, work_inverse, (work, work))
        points = self._landmarks_to_crop(landmarks, work_inverse)
        manual_include, explicit = self._manual_mask(occluder_masks.get(replacement.track_id), (side, side))
        if work != side:
            manual_include = cv2.resize(manual_include, (work, work), interpolation=cv2.INTER_LINEAR)
            explicit = cv2.resize(explicit, (work, work), interpolation=cv2.INTER_LINEAR)
        explicit = explicit if explicit.any() else None
        analysis = self.analyzer.analyze(work_crop, points, feather_px=options.feather_px * scale,
                                         confidence_threshold=options.semantic_confidence_threshold,
                                         explicit_front=explicit, border_trust=options.robust.network_border_trust,
                                         track_id=replacement.track_id,
                                         **({"mesh_support": cv2.resize(replacement.coverage, (work, work))}
                                            if getattr(replacement, "full_head", False) else {}))
        if getattr(replacement, "full_head", False):
            analysis = replace(analysis, face_alpha=np.ones((work, work), np.float32))
        # Pixels outside the frame are reflected copies for the models; they must never count as foreground.
        validity = cv2.warpAffine(np.ones(frame.shape[:2], np.float32), cv2.invertAffineTransform(work_inverse),
                                  (work, work), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        validity = cv2.GaussianBlur(validity, (5, 5), 1.0)
        front = analysis.occluder
        if front is not None:
            front = front * (validity > 0.5)
        state = self.occlusion_tracker.update(replacement.track_id, front, work_crop, frame_index=frame_index,
                                              scene_cut=scene_cut, release=min(options.robust.occlusion_release, 0.99))
        front = state.occluder
        stable_face = self.mask_ema.update(replacement.track_id, analysis.face_alpha, frame=frame_index, kind="geometric")
        stable_include = self.mask_ema.update(replacement.track_id, manual_include, frame=frame_index, kind="manual_include")
        stable_face = np.maximum(stable_face, stable_include)
        if front is not None:
            front = self.mask_ema.update(replacement.track_id, front, frame=frame_index, kind="occluder")
        analysis = replace(analysis, face_alpha=stable_face)
        outside = float(1.0 - validity.mean())
        head = landmark_hull_mask((work, work), points, feather_px=options.feather_px * scale)
        head = cv2.dilate(head, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(work * 0.06)) + 1,) * 2))
        if analysis.regions is not None:
            head = np.maximum(head, np.maximum(analysis.regions.get("hair"), analysis.regions.get("ears")))
        determinant = abs(float(np.linalg.det(np.asarray(replacement.inverse, np.float64)[:, :2])))
        centre = cv2.transform(np.array([[[side / 2, side / 2]]], np.float32), np.asarray(replacement.inverse, np.float32))[0, 0]
        # Face velocity in frame pixels: the aligned crop is stabilised, so the blur the camera recorded on the face
        # has to come from how far the crop moved through the frame, not from the flow inside the crop.
        velocity = (0.0, 0.0)
        previous = self._face_centres.get(replacement.track_id)
        if previous is not None and not scene_cut and (frame_index is None or previous[0] is None
                                                       or 0 < frame_index - previous[0] <= 2):
            velocity = (float(centre[0] - previous[1][0]), float(centre[1] - previous[1][1]))
        self._face_centres[replacement.track_id] = (frame_index, np.asarray(centre, np.float32))
        report = {"tier": analysis.tier, "confidence": analysis.confidence, "reasons": list(analysis.reasons),
                  "networks": list(analysis.nets_used), "parser_trusted": analysis.parser_trusted,
                  "visible_fraction": analysis.visible_fraction, "outside_frame_fraction": outside,
                  "temporal": {"held": state.held, "reset": state.reset_reason, "history_influence": state.history_influence,
                               "content_motion_px": list(state.motion), "face_motion_px": list(velocity)},
                  "analysis_ms": analysis.timing_ms}
        if outside > 0.02:
            report["reasons"].append("PARTIALLY_OUTSIDE_FRAME")
        return _FaceContext(replacement, crop, target_crop, work_crop, work_inverse, scale, analysis, front,
                            velocity, state, head.astype(np.float32), determinant, float(centre[1]), report)

    @staticmethod
    def _is_nearer(candidate: "_FaceContext", reference: "_FaceContext") -> bool:
        """Depth heuristic: a larger face is nearer; near-equal sizes fall back to the lower face in the frame."""
        ratio = candidate.depth / max(reference.depth, 1e-9)
        return ratio > 1.08 or (0.92 <= ratio <= 1.08 and candidate.centre_y > reference.centre_y)

    def _depth_order(self, a: "_FaceContext", b: "_FaceContext") -> int:
        if self._is_nearer(a, b):
            return 1
        if self._is_nearer(b, a):
            return -1
        return (a.replacement.track_id > b.replacement.track_id) - (a.replacement.track_id < b.replacement.track_id)

    def _cross_face_front(self, context: _FaceContext, others: list[_FaceContext], work: int) -> np.ndarray | None:
        """Heads of nearer faces that overlap this face. Depth is a scale heuristic (no depth model is installed)."""
        total = None
        for other in others:
            if other is context:
                continue
            if not self._is_nearer(other, context):
                continue
            matrix = self._compose_affine(cv2.invertAffineTransform(context.work_inverse), other.work_inverse)
            warped = cv2.warpAffine(other.head, matrix, (work, work), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if warped.max() > 0.02:
                total = warped if total is None else np.maximum(total, warped)
        return total

    def _compose_robust(self, frame, replacements, landmarks_by_track, active_tracks, scene_cut, frame_index,
                        occluder_masks, lip_sync_results, options, observer) -> np.ndarray:
        from roop_3d.compositor.robust_fusion import RobustFusionOptions, fuse_robust
        robust = options.robust
        self.occlusion_tracker.retain(active_tracks)
        self.analyzer.retain(active_tracks)
        self._face_centres = {key: value for key, value in self._face_centres.items() if key in active_tracks}
        if scene_cut:
            self.occlusion_tracker.clear()
            self.analyzer.retain(())
        output = frame.copy()
        compose_began, restorer_seconds = time.perf_counter(), 0.0
        contexts = [self._analyse_face(frame, replacement, landmarks_by_track.get(replacement.track_id), occluder_masks,
                                       options, scene_cut, frame_index) for replacement in replacements]
        # Back-to-front: nearer (larger) faces are composited last so they correctly cover farther ones.
        contexts.sort(key=functools.cmp_to_key(self._depth_order))
        fusion_options = RobustFusionOptions(refine_matte=robust.refine_matte, decontaminate=robust.edge_decontamination,
                                             match_sharpness=robust.sharpness_matching, texture_strength=robust.texture_strength,
                                             linear_light=robust.linear_light or options.linear_light,
                                             transfer=options.blend_transfer if options.linear_light else "srgb",
                                             matte_contrast=robust.matte_contrast)
        self.last_reports = {}
        for context in contexts:
            replacement, crop, analysis = context.replacement, context.crop, context.analysis
            side, work, scale = crop.shape[0], context.work_crop.shape[0], context.scale
            front = context.front
            if robust.cross_face_occlusion and len(contexts) > 1:
                cross = self._cross_face_front(context, contexts, work)
                if cross is not None:
                    front = cross if front is None else combine_occluders(front, cross)
                    context.report["reasons"].append("OCCLUDED_BY_NEARER_FACE")
            face_alpha = analysis.face_alpha * border_fade(analysis.face_alpha.shape)
            if options.erosion_px:
                radius = max(1, round(abs(options.erosion_px) * scale))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
                face_alpha = cv2.erode(face_alpha, kernel) if options.erosion_px > 0 else cv2.dilate(face_alpha, kernel)
            coverage = getattr(replacement, 'coverage', None)
            if coverage is not None:
                coverage = np.asarray(coverage, np.float32)
                if coverage.shape != crop.shape[:2] or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
                    raise DomainError('PROJECTION_COVERAGE_INVALID', 'Projected coverage must be finite crop-space values in 0..1', 422)
                coverage = cv2.resize(coverage, (work, work), interpolation=cv2.INTER_LINEAR)
            visible_alpha = face_alpha * (1.0 - (front if front is not None else 0.0)) * options.swap_weight
            visible_alpha = self.watchdog.feather(replacement.track_id, visible_alpha, kind="composite")
            if front is not None:
                visible_alpha *= (1.0 - front)
            if coverage is not None:
                visible_alpha = visible_alpha * coverage
            alpha_native = cv2.resize(visible_alpha, (side, side), interpolation=cv2.INTER_AREA) if work != side else visible_alpha
            if observer is not None:
                observer('occluder', cv2.resize(front, (side, side), interpolation=cv2.INTER_AREA) if front is not None else np.zeros((side, side), np.float32),
                         replacement.inverse, alpha_native)
                observer('swapper', crop, replacement.inverse, alpha_native)
                observer('matte', (alpha_native * 255).round().astype(np.uint8), replacement.inverse, alpha_native)
            self.watchdog.observe(replacement.track_id, face_alpha, kind="geometric")
            if front is not None:
                self.watchdog.observe(replacement.track_id, np.clip(front, 0, 1), kind="occluder")
            self.last_reports[replacement.track_id] = context.report
            if not alpha_native.any():
                context.report["skipped"] = "FACE_FULLY_OCCLUDED_OR_EMPTY"
                continue

            protect = None
            if robust.restoration_guard and analysis.protect is not None:
                protect = cv2.resize(analysis.protect, (side, side), interpolation=cv2.INTER_AREA) if work != side else analysis.protect
            skin_native = cv2.resize(analysis.skin, (side, side), interpolation=cv2.INTER_AREA) if work != side else analysis.skin
            mouth = None
            if analysis.regions is not None:
                mouth_work = np.clip(analysis.regions.get("mouth_interior") + analysis.regions.get("lips"), 0, 1)
                mouth = cv2.GaussianBlur(cv2.resize(mouth_work, (side, side), interpolation=cv2.INTER_AREA), (7, 7), 2.0)
            harmonized, color_stats, spent = self._enhance(
                replacement, crop, context.target_crop, alpha_native, probabilities=None,
                skin=skin_native if analysis.parser_trusted else alpha_native, options=options, observer=observer,
                lip_result=lip_sync_results.get(replacement.track_id), frame_index=frame_index, protect=protect,
                mouth_mask=mouth, local_illumination=robust.local_illumination)
            restorer_seconds += spent
            temporal = self.temporal.update(replacement.track_id, replacement.inverse, face_alpha, color_stats,
                                            frame_index=frame_index, scene_cut=False)
            if observer is not None:
                observer('color', harmonized, replacement.inverse, alpha_native)
            smoothed_face = temporal.alpha * options.swap_weight
            if coverage is not None:
                smoothed_face = np.minimum(smoothed_face, coverage)
            source = cv2.resize(harmonized, (work, work), interpolation=cv2.INTER_CUBIC) if work != side else harmonized
            work_inverse = self._scaled_inverse(temporal.inverse, scale)
            if observer is not None:
                observer('vfx', crop, temporal.inverse, cv2.resize(smoothed_face, (side, side), interpolation=cv2.INTER_AREA) if work != side else smoothed_face)
            output, fusion = fuse_robust(output, source, work_inverse, smoothed_face, front,
                                         analysis.skin if analysis.parser_trusted else None,
                                         frame_motion=context.motion, mode=options.mode, levels=options.pyramid_levels,
                                         options=fusion_options)
            context.report["fusion"] = {"sharpness": fusion.sharpness, "texture_gain": fusion.texture_gain,
                                        "occluded_fraction": fusion.occluded_fraction}
        self.frames_composited += 1
        total_ms = (time.perf_counter() - compose_began) * 1000
        restorer_ms = restorer_seconds * 1000
        self.last_timing = {"masking": max(0.0, total_ms - restorer_ms), "restorer": restorer_ms}
        return output

    def capability(self) -> dict:
        parser = self.parser.capability() if hasattr(self.parser, "capability") else {
            "weights": "UNKNOWN", "provenance": "UNKNOWN", "license": "UNKNOWN", "inference": "UNKNOWN"}
        restorer = self.restorer.capability() if self.restorer is not None else {
            "weights": "NOT_CONFIGURED", "inference": "UNKNOWN"}
        result = {"state": "AVAILABLE", "parser": parser, "restorer": restorer,
                  "temporal_tracks": self.temporal.track_count,
                  "temporal_cache_bytes": self.temporal.cache_bytes,
                  "temporal_cache_limit_bytes": self.temporal.max_cache_bytes,
                  "frames_composited": self.frames_composited,
                  "occlusion_modes": {"default": "standard", "available": list(OCCLUSION_MODES)}}
        if self._analyzer is not None:
            result["robust_occlusion"] = self._analyzer.capability()
        return result

    def release(self) -> None:
        if self.parser is not None and hasattr(self.parser, "release"):
            self.parser.release()
        if self._analyzer is not None:
            self._analyzer.release()
        if self.restorer is not None:
            self.restorer.release()
        if self._occlusion_tracker is not None:
            self._occlusion_tracker.clear()
        self.temporal.clear()
        self.mask_ema.clear()
        self.histogram_stabilizer.reset()
        self.watchdog.clear()
