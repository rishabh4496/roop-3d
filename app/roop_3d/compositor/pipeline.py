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
import logging
import time
from typing import Mapping
from collections.abc import Callable, Sequence

import cv2
import numpy as np

from roop_3d.compositor.boundary_fusion import BlendMode, fuse_crop
from roop_3d.compositor.color_match import ColorStats, apply_lab_stats, estimate_lab_stats, masked_histogram_match
from roop_3d.compositor.hairline import offset_hairline, refine_hairline_matte
from roop_3d.compositor.lighting import relight_spherical_harmonics
from roop_3d.compositor.matte import border_fade
from roop_3d.compositor.strand_matte import StrandMatteOptions
from roop_3d.compositor.masks import (
    BisenetFaceParser, FaceParser, build_face_mask, landmark_hull_mask,
    compute_dual_bisenet_forbidden_mask, compose_active_mask, apply_submental_jaw_cutoff,
    fill_mask_holes, confident_forbidden, turn_ratio, chin_line_clamp, JAW_CUTOFF_MIN_TURN_RATIO,
    INCLUDED_SKIN_FACE_CLASSES, STRICT_FORBIDDEN_CLASSES, OCCLUDER_CLASS_IDS
)

#: BiSeNet 19-class ids behind the Hair & Neck Protection toggles (see masks.STRICT_FORBIDDEN_CLASSES_DICT).
NECK_CLASSES = (14, 15, 16)             # neck, neck_l, cloth
EAR_GLASSES_CLASSES = (6, 7, 8, 9)      # eye_g, l_ear, r_ear, ear_r


def protect_matte(mask: np.ndarray, guide_bgr: np.ndarray, landmarks, options: "CompositeOptions", scale: float = 1.0, occluder_mask: np.ndarray | None = None) -> np.ndarray:
    """Jaw-to-neck clamp then hairline/boundary-ribbon matting, each behind its toggle. With both on this is the Stage 3 matte exactly;
    with matting off the edge is feathered by a plain Gaussian (the behaviour before the guided filter)."""
    if options.neck_protection:
        mask = apply_submental_jaw_cutoff(mask, landmarks, falloff_px=options.jaw_falloff(scale))
    if getattr(options, "boundary_ribbon_matting", False):
        from roop_3d.compositor.matte import refine_boundary_ribbon_matte
        return refine_boundary_ribbon_matte(guide_bgr, mask, occluder_mask=occluder_mask, radius=max(2, int(round(6 * scale))))
    if options.hair_matting:
        return refine_hairline_matte(mask, guide_bgr, radius=8, eps=1e-4, smooth_feather_px=options.feather_px * scale)
    sigma = options.feather_px * scale
    if sigma <= 0:
        return np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
    radius = max(1, int(np.ceil(sigma * 3)))
    return np.clip(cv2.GaussianBlur(np.asarray(mask, np.float32), (2 * radius + 1, 2 * radius + 1), sigmaX=sigma), 0.0, 1.0).astype(np.float32)
from roop_3d.compositor.mouth import composite_mouth_region, lower_face_soft_mask
from roop_3d.compositor.occlusion import combine_occluders, semantic_occluder_mask, subtract_occluders
from roop_3d.compositor.quality import inject_target_skin_micro_texture
from roop_3d.compositor.restoration import GpenRestorer, blend_restored_roi
from roop_3d.compositor.temporal import TemporalSmoother
from roop_3d.compositor.detail_temporal import TemporalDetailStabilizer
from roop_3d.temporal import TemporalMaskEMA
from roop_3d.compositor.mask_flow import FlowMaskStabilizer
from roop_3d.pipeline.subpixel import warp_affine
from roop_3d.tracking.scene_cut import SceneCutDetector
from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import RuntimeConfig, image_bgr
from roop_3d.hires_handoff import destination_face_side, fit_hires_to_frame, scale_matrix_input

log = logging.getLogger(__name__)

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
    # Opt-in directional strand matte (hair strands / bangs over the face), computed at FRAME resolution inside fuse_robust from
    # the parser's hair probability. Off by default: it adds a 12-orientation Gabor bank over the face region per face per frame.
    strand_matte: bool = False
    # Flow-warped history for the face matte (compositor/mask_flow.py, c_max 0.6, reset on cut/flash) instead of the plain EMA.
    mask_flow: bool = True
    strand: StrandMatteOptions = field(default_factory=StrandMatteOptions)

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
    expression_restore_factor: float = 0.0
    face_expression_overrides: Mapping = field(default_factory=dict)
    mode: BlendMode = BlendMode.FEATHER
    restoration_blend: float = 0.0
    # Internal pressure governor cap; zero keeps the restorer's native output resolution.
    roi_resolution_limit: int = 0
    fidelity_weight: float = 0.5
    color_method: str = "lab"
    color_strength: float = 0.65
    relight: bool = False
    relight_strength: float = 0.8
    specular_sheen: float = 0.25
    shadow_preservation: float = 0.85
    # 0 is an explicit diagnostic bypass; production defaults exceed the .78 gate.
    photometric_identity_floor: float = .8
    # UltraMax runs on every face by default. "bypass"/"adaptive" are explicit opt-ins judged on the face's
    # destination size in the target frame.
    ultramax_small_face_mode: str = "off"
    ultramax_min_native_pixels: int = 320
    lighting_mode: str = "auto"
    manual_light_yaw: float = 35.0
    manual_light_pitch: float = 25.0
    # Ambient (SH band-0) fill of the manual virtual key light; the auto light is estimated from the target instead.
    ambient_fill: float = 0.3
    semantic_confidence_threshold: float = 0.55
    feather_px: float = 5.0
    pyramid_levels: int = 4
    subpixel_interpolation: str = "lanczos4"
    swap_weight: float = 1.0
    erosion_px: float = 0.0
    # Moves only the forehead / hairline band of the matte (see compositor/hairline.py), in 512-class crop pixels.
    hairline_offset_px: float = 0.0
    # Hair / neck / ear protection (the Stage 3 anti-smudge matte). Every default is the compositor's previous hard-coded behaviour.
    neck_protection: bool = True              # neck + cloth are forbidden classes and the matte is clamped along the 3D jawline
    hair_matting: bool = True                 # guided-filter matte on the forehead / hairline instead of a plain Gaussian feather
    preserve_ears_glasses: bool = True        # ears and eyeglasses stay the target's
    jaw_feather_px: float = 6.0               # inward falloff of the jaw-to-neck clamp, in 512-class crop pixels
    skin_texture_retention: float = 0.25      # share of the target's pore / specular residual re-injected after UltraMax
    boundary_ribbon_matting: bool = False     # 12px morphological boundary-ribbon sub-pixel alpha matting
    submental_contact_shadow: bool = False    # soft under-chin contact shadow projection on upper neck
    contact_shadow_opacity: float = 0.12      # contact shadow peak opacity (0..0.15)
    contact_shadow_falloff_px: float = 10.0   # contact shadow falloff distance (6..14 px)
    sss_shadow_warmth: bool = False           # subsurface scattering shadow-terminator warmth
    sss_warmth_amount: float = 0.5            # SSS warmth intensity
    occlusion_mode: str = "standard"
    robust: RobustOptions = field(default_factory=RobustOptions)
    # Stage 15: blend arithmetic in linear light, with the display transfer named explicitly (never assumed sRGB).
    linear_light: bool = False
    blend_transfer: str = "srgb"
    sensor_shutter_angle: float = 180.0
    sensor_grain_strength: float = 100.0
    # Dense 3D replacements only: share of the target's low-frequency lightness (lit side, rim) the synthesised face takes on.
    dense_shading: float = 1.0
    # Stage 6B swap scope (rendering/silhouette_replacement.py). Only read for full-head (3D silhouette) replacements.
    # full_craniofacial: the old silhouette is removed (hair / ears / neck warped onto the new perimeter, overhang filled with
    # background); adaptive_skull: the same for the blended skull the engine rendered; inner_face: legacy, the swap stays inside
    # the target's own face outline and nothing is filled or warped.
    swap_scope: str = "full_craniofacial"
    skull_blend: float = 1.0                  # adaptive_skull: share of beta_source in the skull (-> source_identity_lock)
    silhouette_history: int = 2               # past unswapped frames kept per track as temporal background donors (0 = spatial only)

    def __post_init__(self) -> None:
        from roop_3d.rendering.silhouette_replacement import resolve_scope
        try:
            resolve_scope(self.swap_scope, self.skull_blend)
        except DomainError as error:
            raise ValueError(error.message) from error
        if type(self.silhouette_history) is not int or not 0 <= self.silhouette_history <= 8:
            raise ValueError("silhouette_history must be an integer within 0..8")
        if not np.isfinite(self.expression_restore_factor) or not 0 <= self.expression_restore_factor <= 1:
            raise ValueError('expression_restore_factor must be within 0..1')
        if self.photometric_identity_floor != 0 and not .78 < self.photometric_identity_floor < 1:
            raise ValueError('Photometric identity floor must be 0 or in (.78, 1)')
        if not np.isfinite(self.fidelity_weight) or not 0 <= self.fidelity_weight <= 1:
            raise ValueError("CodeFormer fidelity must be within 0..1")
        if not 0 <= self.restoration_blend <= 1 or not 0 <= self.color_strength <= 1:
            raise ValueError("Blend ratios must be within 0..1")
        if (type(self.roi_resolution_limit) is not int
                or (self.roi_resolution_limit != 0 and not 128 <= self.roi_resolution_limit <= 2048)):
            raise ValueError("ROI resolution cap must be zero or within 128..2048")
        if not np.isfinite(self.swap_weight) or not 0 <= self.swap_weight <= 1 or not np.isfinite(self.erosion_px) or not -32 <= self.erosion_px <= 32:
            raise ValueError('Swap weight or erosion is outside its bounds')
        if not np.isfinite(self.ambient_fill) or not 0 <= self.ambient_fill <= 1:
            raise ValueError('Ambient fill must be within 0..1')
        if not np.isfinite(self.hairline_offset_px) or not -32 <= self.hairline_offset_px <= 32:
            raise ValueError('Hairline offset is outside its bounds')
        if not np.isfinite(self.jaw_feather_px) or not 2.0 <= self.jaw_feather_px <= 24.0:
            raise ValueError('Jaw feather falloff must be within 2..24 px')
        if not np.isfinite(self.skin_texture_retention) or not 0 <= self.skin_texture_retention <= 1:
            raise ValueError('Skin texture retention must be within 0..1')
        if not np.isfinite(self.contact_shadow_opacity) or not 0 <= self.contact_shadow_opacity <= 0.15:
            raise ValueError('contact_shadow_opacity must be within 0..0.15')
        if not np.isfinite(self.contact_shadow_falloff_px) or not 6.0 <= self.contact_shadow_falloff_px <= 14.0:
            raise ValueError('contact_shadow_falloff_px must be within 6..14 px')
        if self.color_method not in {"lab", "reinhard_lab", "histogram", "ycbcr", "oklab", "oklab_wasserstein"}:
            raise ValueError("color_method must be 'lab', 'reinhard_lab', 'ycbcr', 'histogram', 'oklab', or 'oklab_wasserstein'")
        if self.ultramax_small_face_mode not in {"bypass", "adaptive", "off"} or not 64 <= self.ultramax_min_native_pixels <= 1024:
            raise ValueError("Invalid UltraMax small-face policy")
        if not 0 <= self.semantic_confidence_threshold <= 1 or self.feather_px < 0 or self.pyramid_levels < 1:
            raise ValueError("Invalid mask or pyramid configuration")
        if self.subpixel_interpolation not in {"lanczos4", "cubic", "linear"}:
            raise ValueError("subpixel_interpolation must be 'lanczos4', 'cubic', or 'linear'")
        if self.occlusion_mode not in OCCLUSION_MODES:
            raise ValueError(f"occlusion_mode must be one of {OCCLUSION_MODES}")
        from roop_3d.color.transfer import BLEND_TRANSFERS
        if self.blend_transfer not in BLEND_TRANSFERS:
            raise ValueError(f"blend_transfer must be one of {BLEND_TRANSFERS}")

    def forbidden_classes(self) -> tuple[int, ...]:
        """BiSeNet classes the swap may never cover. Hair and hats are always protected; neck/cloth and ears/glasses follow the toggles."""
        dropped = set() if self.neck_protection else set(NECK_CLASSES)
        if not self.preserve_ears_glasses:
            dropped |= set(EAR_GLASSES_CLASSES)
        return tuple(item for item in STRICT_FORBIDDEN_CLASSES if item not in dropped)

    def front_classes(self) -> tuple[int, ...]:
        """BiSeNet classes that stay in front of the swap as semantic occluders (glasses, hair, hat)."""
        return tuple(item for item in OCCLUDER_CLASS_IDS if self.preserve_ears_glasses or item not in EAR_GLASSES_CLASSES)

    def jaw_falloff(self, scale: float = 1.0) -> float:
        return float(min(32.0, max(2.0, self.jaw_feather_px * scale)))
        if not np.isfinite(self.sensor_shutter_angle) or not 0 <= self.sensor_shutter_angle <= 360:
            raise ValueError("sensor_shutter_angle must be within 0..360 degrees")
        if not np.isfinite(self.sensor_grain_strength) or not 0 <= self.sensor_grain_strength <= 100:
            raise ValueError("sensor_grain_strength must be within 0..100 percent")


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
                 analyzer=None, scene_cut_detector: SceneCutDetector | None = None):
        """``scene_cut_detector`` (opt-in) lets the compositor detect shot changes and exposure flashes itself from the
        frames it is given, in addition to the ``scene_cut`` flag callers pass (the volumetric session passes the face
        tracker's, which uses the same detector). A cut resets every temporal state; a flash resets the colour/detail/matte
        filters but keeps tracks and occluder history."""
        self.catalog = catalog
        self.scene_cut_detector = scene_cut_detector
        self.last_cut_event = None
        self.mask_flow = FlowMaskStabilizer()
        from roop_3d.compositor.stream_restoration import MaskSeamWatchdog
        self.watchdog = MaskSeamWatchdog()
        self.config = config
        self._parser_injected = parser is not None
        self.parser = parser if parser is not None else BisenetFaceParser(catalog, config)
        self.restorer = restorer
        self.expression_restorer = None
        self.expression_portrait = None
        self.expression_reports = {}
        self.track_labels = {}
        self.temporal = temporal or TemporalSmoother()
        self.mask_ema = TemporalMaskEMA(alpha=.65, max_tracks=32)
        # Alpha is the current-detail coefficient; its complementary history weight is
        # flow/visibility gated and reset at scene cuts.
        self.detail_temporal = TemporalDetailStabilizer(temporal_alpha=.25)
        from roop_3d.temporal_stabilizer import TemporalStabilizer
        self.histogram_stabilizer = TemporalStabilizer(min_cutoff=.15, beta=.04)
        self._analyzer = analyzer
        self._occlusion_tracker = None
        self._face_centres: dict[int, tuple[int | None, np.ndarray]] = {}
        self.frames_composited = 0
        self.angle_diagnostics = None      # AngleDiagnostics installed by the session; only its "full" level reads frames
        self.last_timing: dict[str, float] = {}
        # Stage 5: per-track report of the last robust frame (tier, confidence, reasons, fusion metrics).
        self.last_reports: dict[int, dict] = {}
        self.last_relighting: dict[int, dict] = {}
        self._relight_recognizer = None
        # Stage 6B: silhouette replacement (built on first full-head frame) and the unswapped-frame history that feeds it.
        self._silhouette_replacer = None
        self._silhouette_history = None
        self._background_donors: Mapping[int, Sequence] = {}

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
        return warp_affine(frame, frame_to_crop, (shape[1], shape[0]), interpolation="cubic", border="reflect101")

    def prefetch_parser(self, frame: np.ndarray, replacements, options: "CompositeOptions") -> int:
        """Run the face parser on the crops :meth:`compose` will ask about, so it can start on the next frame at once.

        Called from the (light) swap stage while the composite stage is still busy with the previous frame.  The parser
        is stateless and its results are cached by image content, so this changes no output: a miss merely costs the
        same inference later.  Only the standard occlusion path parses these crops.  Returns the crops parsed.
        """
        parser = self.parser
        if parser is None or not getattr(parser, "cache_enabled", False) or options.occlusion_mode != "standard":
            return 0
        done = 0
        for replacement in replacements:
            try:
                if parser.prefetch_pressure() >= parser.PREFETCH_LEAD_CROPS:
                    return done          # far enough ahead of the composite stage; more would only cost memory
                crop = np.asarray(replacement.image)
                if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3:
                    continue
                parser.predict(self._crop_to_frame(frame, replacement.inverse, crop.shape[:2]), prefetch=True)
                parser.predict(crop, prefetch=True)
                done += 2
            except Exception:
                return done      # the composite stage reports real errors itself
        return done

    @staticmethod
    def _landmarks_to_crop(landmarks: np.ndarray | None, inverse: np.ndarray) -> np.ndarray | None:
        if landmarks is None:
            return None
        points = np.asarray(landmarks, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise DomainError("COMPOSITOR_LANDMARKS_INVALID", "Target landmarks must be finite Nx2 points", 422)
        frame_to_crop = cv2.invertAffineTransform(np.asarray(inverse, dtype=np.float32))
        return cv2.transform(points[None], frame_to_crop)[0]

    def _face_motion(self, replacement, frame_index: int | None, scene_cut: bool) -> tuple[float, float]:
        """Per-frame crop-center displacement, used when a dense motion field is unavailable."""
        side = int(np.asarray(replacement.image).shape[0])
        centre = cv2.transform(np.array([[[side / 2, side / 2]]], np.float32),
                               np.asarray(replacement.inverse, np.float32))[0, 0]
        velocity = (0.0, 0.0)
        previous = self._face_centres.get(replacement.track_id)
        if previous is not None and not scene_cut and (frame_index is None or previous[0] is None
                                                       or 0 < frame_index - previous[0] <= 2):
            velocity = (float(centre[0] - previous[1][0]), float(centre[1] - previous[1][1]))
        self._face_centres[replacement.track_id] = (frame_index, np.asarray(centre, np.float32))
        return velocity

    @staticmethod
    def _crop_motion(frame_motion: tuple[float, float], inverse: np.ndarray, crop_side: int,
                     output_side: int) -> tuple[float, float]:
        linear = np.asarray(inverse, np.float64)[:, :2]
        try:
            motion = np.linalg.solve(linear, np.asarray(frame_motion, np.float64))
        except np.linalg.LinAlgError:
            return (0.0, 0.0)
        motion *= output_side / max(crop_side, 1)
        return float(motion[0]), float(motion[1])

    def _finish_sensor(self, replacement, enhanced, target, alpha, skin, frame_motion, options, frame_index,
                       *, grain_strength: float | None = None, match_optics: bool = True):
        """Run the target sensor model on the restored face before it reaches the final mask blend."""
        if options.restoration_blend <= 0 or self.restorer is None:
            return enhanced, {"backend": "skipped", "reason": "restoration_not_active"}
        from roop_3d.compositor.sensor_physics import finish_target_sensor
        motion = self._crop_motion(frame_motion, replacement.inverse, np.asarray(replacement.image).shape[0],
                                  enhanced.shape[0])
        output, report = finish_target_sensor(
            enhanced, target, alpha, skin, flow=self.detail_temporal.flow_for(replacement.track_id),
            crop_motion=motion, shutter_angle=options.sensor_shutter_angle,
            grain_strength=options.sensor_grain_strength if grain_strength is None else grain_strength,
            seed=((frame_index or 0) * 1_000_003 + replacement.track_id * 9_176) % (2**31 - 1),
            match_optics=match_optics, transfer=options.blend_transfer)
        return output, report

    @staticmethod
    def _color_process(swapped: np.ndarray, target: np.ndarray, alpha: np.ndarray, method: str,
                       strength: float, smooth_histogram=None, stats_mask: np.ndarray | None = None
                       ) -> tuple[np.ndarray, ColorStats]:
        """LAB mean/std (or histogram) transfer. ``stats_mask`` (the parser's skin class inside the face) decides which
        pixels the statistics come from, so eyes, brows, lips and hair do not skew the skin tone; ``alpha`` is where the
        correction is written (mean/std) and the fallback when no skin class is available."""
        basis = alpha if stats_mask is None else stats_mask
        if method in {"oklab", "oklab_wasserstein"}:
            from roop_3d.compositor.color_match import sliced_wasserstein_color_transfer_oklab
            matched = sliced_wasserstein_color_transfer_oklab(
                swapped, target, basis, strength=strength, preserve_specular=True
            )
            stats = estimate_lab_stats(swapped, target, basis)
            return matched, stats
        if method == "ycbcr":
            from roop_3d.compositor.color_match import apply_ycbcr_stats, estimate_ycbcr_stats
            stats = estimate_ycbcr_stats(swapped, target, basis)
            return apply_ycbcr_stats(swapped, stats, alpha, strength=strength), stats
        stats = estimate_lab_stats(swapped, target, basis)
        if method == "histogram":
            matched = masked_histogram_match(swapped, target, basis, smooth_histogram=smooth_histogram)
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
    @staticmethod
    def _resize_plane(plane, side: int):
        if plane is None or plane.shape[0] == side:
            return plane
        return np.clip(cv2.resize(np.asarray(plane, np.float32), (side, side), interpolation=cv2.INTER_CUBIC), 0, 1)

    def _run_restorer(self, crop, replacement, alpha, options):
        """Restored face at the enhancer's NATIVE resolution (square, >= crop) or ``None`` for an explicit bypass."""
        restorer = self.restorer
        if hasattr(restorer, 'fidelity_weight'):
            restorer.fidelity_weight = options.fidelity_weight
        elif options.fidelity_weight != .5:
            raise DomainError('FIDELITY_UNAVAILABLE', 'The selected restorer has no CodeFormer fidelity input', 422)
        landmarks = getattr(replacement, 'landmarks5', None)
        try:
            if hasattr(restorer, 'restore_hires'):
                kwargs = {}
                if hasattr(restorer, 'small_face_mode'):
                    restorer.small_face_mode = options.ultramax_small_face_mode
                    restorer.min_native_pixels = options.ultramax_min_native_pixels
                    if options.ultramax_small_face_mode != 'off':
                        # The destination bounding box in the TARGET frame, not the size of the swap crop.
                        kwargs['native_face_size_px'] = destination_face_side(replacement.inverse, alpha)
                restored = restorer.restore_hires(crop, landmarks, **kwargs)
            elif hasattr(restorer, 'restore_aligned'):
                restored = restorer.restore_aligned(crop, landmarks)
            else:
                restored = restorer.restore(crop)
            limit = options.roi_resolution_limit
            if limit and restored.shape[0] > limit:
                side = max(int(limit), int(crop.shape[0]))
                restored = cv2.resize(restored, (side, side), interpolation=cv2.INTER_AREA)
            return restored
        except DomainError:
            raise
        except Exception as error:
            # Never fall back to the unenhanced crop: that would hide a shape / provider mismatch behind a
            # normal-looking (low resolution) swap.
            log.critical("Restorer %s failed on a %s crop: %s", type(restorer).__name__, crop.shape, error, exc_info=True)
            raise DomainError('RESTORER_FAILED', f'{type(restorer).__name__} failed: {error}', 503) from error

    def _enhance(self, replacement, crop, target_crop, alpha, *, probabilities, skin, options, observer,
                 lip_result, frame_index, protect=None, mouth_mask=None, local_illumination: float = 0.0,
                 frame=None):
        """Restoration, mouth, colour and relighting.

        When the restorer returns more pixels than the swap crop (GPEN/CodeFormer/GFPGAN 512, UltraMax 512/1024), every
        later step runs at that resolution and the returned image is the large one. Callers re-base the crop->frame
        matrix with ``scale_matrix_input(inverse, harmonized.shape[0] / crop.shape[0])``.
        """
        restorer_seconds = 0.0
        self.last_relighting.pop(replacement.track_id,None)
        from roop_3d.core.face_editor import FaceEditor
        label = self.track_labels.get(replacement.track_id, f'track_{replacement.track_id:04d}')
        edit = FaceEditor.model_validate(options.face_expression_overrides.get(label, {}))
        expression_active = options.expression_restore_factor > 0 or edit.active()
        if expression_active:
            from roop_3d.expression_restorer import ExpressionRestorer
            if self.expression_restorer is None:
                shared = self.expression_portrait if getattr(self.expression_portrait, 'available', False) else None
                self.expression_restorer = ExpressionRestorer(self.catalog, self.config, portrait=shared)
            began = time.perf_counter()
            crop = self.expression_restorer.restore(crop, target_crop, factor=options.expression_restore_factor, edit=edit)
            self.expression_reports[replacement.track_id] = dict(self.expression_restorer.last_report)
            restorer_seconds += time.perf_counter()-began
            if observer is not None:
                observer('expression', crop, replacement.inverse, alpha)
        else:
            self.expression_reports.pop(replacement.track_id, None)
        if lip_result is not None and bool(getattr(lip_result, "applied", False)):
            lip_image = np.asarray(getattr(lip_result, "image", None))
            lip_blend = float(getattr(lip_result, "blend", 0.0))
            lip_mask = getattr(lip_result, "mouth_mask", None)
            if lip_mask is None:
                lip_mask = mouth_mask if mouth_mask is not None else lower_face_soft_mask(
                    crop.shape, probabilities, feather_px=options.feather_px)
            if lip_image.shape[0] != crop.shape[0]:
                lip_image = cv2.resize(lip_image, (crop.shape[0], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
            lip_mask = self._resize_plane(np.asarray(lip_mask, np.float32), crop.shape[0])
            crop = composite_mouth_region(crop, lip_image, lip_mask, lip_blend)

        enhanced = crop
        side = crop.shape[0]
        current_alpha, current_target, current_skin, current_protect, current_mouth = alpha, target_crop, skin, protect, mouth_mask
        current_inverse = replacement.inverse
        if options.restoration_blend > 0:
            if self.restorer is None:
                raise DomainError("RESTORER_UNAVAILABLE", "A positive restoration blend requires an explicit local restorer", 503)
            restore_began = time.perf_counter()
            # A replacement the dense engine already restored (UltraMax at native resolution, aligned to the swapped
            # geometry) is not run through a second restorer: that only re-softened and re-sharpened an already restored face.
            restored = None if getattr(replacement, "restored", False) else self._run_restorer(crop, replacement, alpha, options)
            if restored is not None:
                size = restored.shape[0]
                if restored.shape[:2] != (size, size) or size < side:
                    raise DomainError('RESTORATION_OUTPUT_INVALID', 'Restorer must return a square image no smaller than the crop', 503)
                if size != side:
                    scale = size / side
                    current_inverse = scale_matrix_input(replacement.inverse, scale)
                    base = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
                    current_alpha = self._resize_plane(alpha, size)
                    current_protect = self._resize_plane(protect, size)
                    current_skin = self._resize_plane(skin, size)
                    current_mouth = self._resize_plane(mouth_mask, size)
                    # Real target pixels at the enhanced resolution (not an upsample of the 128 px sample).
                    current_target = (self._crop_to_frame(frame, current_inverse, (size, size)) if frame is not None
                                      else cv2.resize(target_crop, (size, size), interpolation=cv2.INTER_CUBIC))
                else:
                    base = crop
                enhanced = blend_restored_roi(base, restored, current_alpha, options.restoration_blend, protect=current_protect)
            restorer_seconds += time.perf_counter() - restore_began
        size = enhanced.shape[0]

        # Re-inject 25% of target skin's high-frequency pore/specular residual (Target_HighPass_Skin)
        # into UltraMax's 1024px output over unoccluded Class 1 (Skin) pixels
        is_ultramax = (
            getattr(replacement, 'method', '') == "dense_3d_correspondence_neural_ultramax"
            or hasattr(self.restorer, 'restore_hires')
            or type(self.restorer).__name__.startswith("UltraMax")
        )
        if is_ultramax and options.skin_texture_retention > 0 and current_skin is not None and np.any(current_skin > 0.05):
            enhanced = inject_target_skin_micro_texture(
                enhanced, current_target, current_skin, current_alpha, injection_rate=options.skin_texture_retention
            )

        if observer is not None:
            observer('restorer', enhanced, current_inverse, current_alpha)

        stats_mask = None
        if current_skin is not None and current_skin is not current_alpha:
            candidate = np.minimum(np.asarray(current_alpha, np.float32), np.asarray(current_skin, np.float32))
            # Never sample dark hair or neck shadows when computing skin color statistics
            target_luma = (current_target.astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)) / 255.0
            candidate[target_luma < 0.08] = 0.0
            if np.count_nonzero(candidate > 0.15) >= 32:      # unoccluded skin class
                stats_mask = candidate
        harmonized, color_stats = self._color_process(
            enhanced, current_target, current_alpha, options.color_method,
            0. if options.relight and getattr(replacement,'photometry',None) is not None else options.color_strength,
            lambda histogram: self.histogram_stabilizer.filter(
                replacement.track_id, "histogram", histogram,
                getattr(self, "timestamp_seconds", (frame_index if frame_index is not None else self.frames_composited) / 30.0)),
            stats_mask=stats_mask)
        photo = getattr(replacement, 'photometry', None)
        if (str(getattr(replacement, "method", "")).startswith("dense_3d") and options.dense_shading > 0
                and not (options.relight and photo is not None)):
            from roop_3d.compositor.quality import shading_gain_match
            harmonized = shading_gain_match(harmonized, current_target, current_alpha, strength=options.dense_shading)
        if local_illumination > 0 and not (options.relight and photo is not None):
            from roop_3d.compositor.quality import local_illumination_match
            harmonized = local_illumination_match(harmonized, current_target, current_alpha, strength=local_illumination)
        if options.relight and photo is not None:
            # Enhanced pixels remain at their native resolution through the
            # GPU normal-ratio shader and Stage 4's single inverse warp.
            harmonized, normal_vis, sh_vis = photo.apply(
                enhanced, current_target, current_skin, current_alpha, options)
            if observer is not None:
                observer('normal_map', normal_vis, current_inverse, current_alpha)
                observer('sh_lighting', sh_vis, current_inverse, current_alpha)
        elif options.relight:
            from roop_3d.relighting_3d import relighter
            relit_crop, normal_vis, sh_vis, _ = relighter.relight(
                harmonized, current_target, skin_mask=current_skin, face_mask=current_alpha,
                relight_strength=getattr(options, "relight_strength", 0.8),
                specular_sheen=getattr(options, "specular_sheen", 0.25),
                shadow_preservation=getattr(options, "shadow_preservation", 0.85),
                lighting_mode=getattr(options, "lighting_mode", "auto"),
                manual_light_yaw=getattr(options, "manual_light_yaw", 35.0),
                manual_light_pitch=getattr(options, "manual_light_pitch", 25.0),
                ambient_fill=getattr(options, "ambient_fill", 0.3))
            harmonized = relit_crop
            if observer is not None:
                observer('normal_map', normal_vis, current_inverse, current_alpha)
                observer('sh_lighting', sh_vis, current_inverse, current_alpha)
        if getattr(options, "sss_shadow_warmth", False):
            from roop_3d.compositor.color_match import apply_sss_shadow_warmth
            from roop_3d.relighting_3d import SurfaceNormalEstimator
            estimator = SurfaceNormalEstimator()
            crop_normals = estimator.compute_normals(harmonized)
            l_dir = np.array([0.0, 0.85, 0.52], dtype=np.float32)
            harmonized = apply_sss_shadow_warmth(
                harmonized, crop_normals, l_dir, skin_mask=current_skin,
                warmth_amount=getattr(options, "sss_warmth_amount", 0.5),
            )
        if getattr(options, "submental_contact_shadow", False):
            from roop_3d.compositor.lighting import project_submental_contact_shadow
            lms = getattr(replacement, "landmarks", None)
            harmonized = project_submental_contact_shadow(
                harmonized,
                lms,
                light_direction=np.array([0.0, 0.85, 0.52], dtype=np.float32),
                max_opacity=getattr(options, "contact_shadow_opacity", 0.12),
                falloff_px=getattr(options, "contact_shadow_falloff_px", 10.0),
            )
        detail = getattr(replacement, 'face_detail', None)
        if detail is not None:
            harmonized = detail.condition(harmonized, current_target)
        if options.relight and photo is not None and options.photometric_identity_floor:
            from roop_3d.faceset.recognizer import Recognizer
            from roop_3d.compositor.photometric_identity import constrain_identity
            if self._relight_recognizer is None:
                self._relight_recognizer = Recognizer(self.catalog)
            baseline = detail.condition(enhanced,current_target) if detail is not None else enhanced
            points = getattr(replacement,'landmarks5',None)
            points = np.asarray(points,np.float32) * (size/side) if points is not None else None
            harmonized, constraint = constrain_identity(baseline,harmonized,points,self._relight_recognizer,
                                                       floor=options.photometric_identity_floor)
            self.last_relighting[replacement.track_id] = constraint.metadata()
            self.last_relighting[replacement.track_id].update(
                requested_strength=options.relight_strength,
                effective_strength=options.relight_strength * constraint.applied_fraction)
        return harmonized, color_stats, restorer_seconds

    def _stabilize_detail(self, track_id, enhanced, base, guide, valid_mask, *, frame_index, scene_cut,
                          protect=None):
        """Stabilize only the enhancer-added high-pass band, never the full face image."""
        height, width = enhanced.shape[:2]
        if base.shape[:2] != (height, width):
            base = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
        if guide.shape[:2] != (height, width):
            guide = cv2.resize(guide, (width, height), interpolation=cv2.INTER_CUBIC)
        validity = self._resize_plane(valid_mask, height)
        if protect is not None:
            validity *= 1.0 - self._resize_plane(np.asarray(protect, np.float32), height)
        result, report = self.detail_temporal.update(track_id, enhanced, base, guide, valid_mask=validity,
                                                    frame_index=frame_index, scene_cut=scene_cut)
        if track_id in self.expression_reports and self.expression_restorer is not None:
            self.expression_restorer.last_report = self.expression_reports[track_id]
            result = self.expression_restorer.enforce(result)
        return result, report

    @staticmethod
    def _apply_recovered_alpha(recovered, swap_weight, coverage=None):
        """Reapply compositor controls to a matte interpolated from neighboring frames."""
        alpha = np.asarray(recovered, np.float32) * float(swap_weight)
        if coverage is not None:
            alpha = np.minimum(alpha, np.asarray(coverage, np.float32) * float(swap_weight))
        return alpha

    def compose(self, frame: np.ndarray, replacements, landmarks_by_track: Mapping[int, np.ndarray] | None = None,
                *, scene_cut: bool = False, frame_index: int | None = None,
                occluder_masks: Mapping[int, np.ndarray] | None = None,
                mask_temporal_context: Mapping[int, Mapping[str, np.ndarray]] | None = None,
                lip_sync_results: Mapping[int, object] | None = None,
                options: CompositeOptions = CompositeOptions(),
                observer: Callable | None = None,
                background_donors: Mapping[int, Sequence] | None = None) -> np.ndarray:
        """Composite one frame with optional per-track one-frame mask lookahead.

        ``background_donors`` (track id -> ``DonorFrame`` list, frame space) adds frames the caller can see but this causal
        compositor cannot, typically the NEXT frames of an offline job, to the past frames it keeps itself: silhouette
        completion then draws on t-k .. t+k.

        Each context entry needs ``previous_mask``, ``next_mask``, ``previous_guide``
        and ``next_guide`` in the same face-crop coordinate system as this frame.
        The current guide and current foreground-occluder veto are taken from the
        active compositor call. Realtime callers can omit the context and remain
        causal; buffered/offline callers may supply it to repair a verified dropout.
        """
        image_bgr(frame)
        if any(getattr(item, "full_head", False) for item in replacements) and options.occlusion_mode == "standard":
            options = replace(options, occlusion_mode="robust")
        landmarks_by_track = landmarks_by_track or {}
        occluder_masks = occluder_masks or {}
        mask_temporal_context = mask_temporal_context or {}
        lip_sync_results = lip_sync_results or {}
        active_tracks = set(landmarks_by_track) | {item.track_id for item in replacements}
        self.expression_reports = {key: value for key, value in self.expression_reports.items() if key in active_tracks}
        self.last_relighting = {key:value for key,value in self.last_relighting.items() if key in active_tracks}
        flash = False
        if self.scene_cut_detector is not None:
            self.last_cut_event = event = self.scene_cut_detector.observe(frame, frame_index=frame_index)
            scene_cut, flash = scene_cut or event.kind == "cut", event.kind == "flash"
        flush = scene_cut or flash
        self.temporal.begin_frame(frame_index, active_track_ids=active_tracks, scene_cut=flush)
        self.mask_ema.retain(active_tracks)
        self.detail_temporal.retain(active_tracks)
        self.mask_flow.retain(active_tracks)
        if flush:
            self.mask_ema.clear()
            self.detail_temporal.clear()
            self.watchdog.clear()
            self.mask_flow.reset()
            self.histogram_stabilizer.reset()
        self.histogram_stabilizer.retain(active_tracks)
        self.last_reports = {}
        if options.occlusion_mode == "robust":
            return self._compose_robust(frame, replacements, landmarks_by_track, active_tracks, scene_cut, frame_index,
                                        occluder_masks, mask_temporal_context, lip_sync_results, options, observer,
                                        background_donors or {})
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
                                        feather_px=options.feather_px,
                                        occluder_class_ids=options.forbidden_classes())
            if getattr(replacement, "full_head", False):
                # Mesh coverage defines source anatomy. Target parsing can veto
                # foreground occlusions but cannot shrink it to the target hull.
                face_mask = replace(face_mask, alpha=np.ones(crop.shape[:2], np.float32))
            semantic_front = (semantic_occluder_mask(probabilities, options.front_classes())
                              if probabilities is not None and face_mask.source.startswith("bisenet") else None)
            # Run BiSeNet on both target ROI and 3D swapped face crop for dual forbidden extraction
            target_forbidden, swap_forbidden, dual_forbidden = compute_dual_bisenet_forbidden_mask(
                self.parser, target_crop, crop, forbidden_classes=options.forbidden_classes()
            )
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
            if options.hairline_offset_px:
                alpha = offset_hairline(alpha, options.hairline_offset_px)

            coverage = getattr(replacement, 'coverage', None)
            if coverage is not None:
                coverage = np.asarray(coverage, np.float32)
                if coverage.shape != alpha.shape or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
                    raise DomainError('PROJECTION_COVERAGE_INVALID', 'Projected coverage must be finite crop-space values in 0..1', 422)
                # Hard-subtract STRICT_FORBIDDEN_CLASSES from 3D mesh silhouette mask:
                # active_mask = mesh_face_only_mask * xseg_face_mask * (1.0 - forbidden_hair_neck_ear_mask)
                xseg_mask = (1.0 - front) if front is not None else np.ones(crop.shape[:2], np.float32)
                if str(getattr(replacement, "method", "")).startswith("dense_3d"):
                    # The dense 3D crop is synthesised, so the parser's weak opinion of it is noise: close the eye-socket
                    # gaps in the mesh silhouette and keep only confident hair/neck/ear evidence from either crop.
                    coverage = fill_mask_holes(coverage)
                    # Coverage comes from a per-frame monocular fit and lives in the template-registered crop, where an
                    # EMA blends without ghosting; the geometric/occluder matte above never saw it.
                    coverage = self.mask_ema.update(replacement.track_id, np.clip(coverage, 0.0, 1.0), frame=frame_index,
                                                    kind="dense_coverage")
                    dual_forbidden = (target_forbidden if swap_forbidden is None
                                      else np.maximum(confident_forbidden(swap_forbidden), target_forbidden))
                active_mask = compose_active_mask(coverage, xseg_mask, dual_forbidden)
                matte_options = options
                if str(getattr(replacement, "method", "")).startswith("dense_3d") and options.neck_protection:
                    # The 106-point jaw contour behind the submental cutoff is unreliable as soon as the head turns (it already
                    # drew a straight diagonal through the swapped cheek at 30 degrees yaw, and collapsed entirely in profile).
                    # The dense path's mesh silhouette bounds the swap everywhere else, so below the chin a five-point line
                    # (reliable at any yaw) keeps the neck and the source photo's collar out instead.
                    matte_options = replace(options, neck_protection=False)
                    active_mask = chin_line_clamp(active_mask, getattr(replacement, "landmarks5", None))
                # Submental Chin-to-Neck Gradient Cutoff (gonion-to-menton curve with an inward SDF falloff), then the
                # edge-preserving Fast Guided Filter for hairline and bangs; each is behind its toggle.
                active_mask = protect_matte(active_mask, target_crop, points, matte_options)
                alpha = active_mask * options.swap_weight
            else:
                alpha = alpha * options.swap_weight
            if front is not None:
                alpha *= (1.0 - front)
            recovery_context = mask_temporal_context.get(replacement.track_id)
            if recovery_context is not None:
                recovery_context = dict(recovery_context)
                recovery_context.setdefault("current_guide", target_crop)
                if front is not None:
                    recovery_context["occluder_mask"] = front
                else:
                    recovery_context.pop("occluder_mask", None)
            mask_report, recovered_alpha = self.watchdog.observe_and_recover(
                replacement.track_id, alpha, frame_index=frame_index, scene_cut=scene_cut,
                context=recovery_context)
            if mask_report["recovery"] == "bidirectional_optical_flow":
                # The watchdog interpolates a matte, but the compositor alpha also
                # carries the user-selected swap strength. Reapply it after replacing
                # the anomalous current matte so a recovery cannot silently force a
                # full-strength swap.
                alpha = self._apply_recovered_alpha(recovered_alpha, options.swap_weight, coverage)
                log.info("Temporal mask recovery track=%s frame=%s diagnostics=%s",
                         replacement.track_id, frame_index, mask_report)
            if mask_report["anomaly"]:
                log.warning("Temporal mask anomaly track=%s frame=%s diagnostics=%s", replacement.track_id,
                            frame_index, mask_report)
            if observer is not None:
                observer('swapper', crop, replacement.inverse, alpha)
                observer('matte', (alpha * 255).round().astype(np.uint8), replacement.inverse, alpha)
            if not alpha.any():
                self.last_reports[replacement.track_id] = {"mask_watchdog": mask_report}
                continue

            skin = face_mask.skin if getattr(face_mask, "source", "").startswith("bisenet") else alpha
            frame_motion = self._face_motion(replacement, frame_index, scene_cut)
            harmonized, color_stats, spent = self._enhance(
                replacement, crop, target_crop, alpha, probabilities=probabilities, skin=skin, options=options,
                observer=observer, lip_result=lip_sync_results.get(replacement.track_id), frame_index=frame_index,
                frame=frame)
            restorer_seconds += spent
            harmonized, detail_report = self._stabilize_detail(
                replacement.track_id, harmonized, crop, target_crop, alpha,
                frame_index=frame_index, scene_cut=scene_cut)
            harmonized, sensor_report = self._finish_sensor(replacement, harmonized, target_crop, alpha, skin,
                                                            frame_motion, options, frame_index)
            self.last_reports[replacement.track_id] = {
                "temporal_detail": detail_report,
                "target_sensor": sensor_report,
                "mask_watchdog": mask_report,
            }
            temporal = self.temporal.update(replacement.track_id, replacement.inverse, alpha / max(options.swap_weight, 1e-8), color_stats,
                                            frame_index=frame_index, scene_cut=False)
            if observer is not None:
                hr = harmonized.shape[0] != crop.shape[0]
                observer('color', harmonized, scale_matrix_input(replacement.inverse, harmonized.shape[0] / crop.shape[0]) if hr else replacement.inverse,
                         self._resize_plane(alpha, harmonized.shape[0]) if hr else alpha)
            effective_alpha = temporal.alpha * (options.swap_weight * getattr(replacement, "presence", 1.0))
            if coverage is not None:
                effective_alpha = np.minimum(effective_alpha, coverage * getattr(replacement, "presence", 1.0))
            if observer is not None:
                observer('vfx', crop, temporal.inverse, effective_alpha)
            fuse_image, fuse_alpha, fuse_inverse = harmonized, effective_alpha, temporal.inverse
            if harmonized.shape[0] != crop.shape[0]:
                # Warp the native-resolution face straight into the frame with M_hr (basis and translation re-based by
                # S = enhanced / crop); it is never reduced to the swap-crop size first.
                fuse_inverse = scale_matrix_input(temporal.inverse, harmonized.shape[0] / crop.shape[0])
                fuse_alpha = self._resize_plane(effective_alpha, harmonized.shape[0])
                fuse_image, fuse_alpha, fuse_inverse = fit_hires_to_frame(harmonized, fuse_alpha, fuse_inverse)
            output = fuse_crop(output, fuse_image, fuse_inverse, fuse_alpha,
                               mode=options.mode, levels=options.pyramid_levels,
                               linear_light=options.linear_light, transfer=options.blend_transfer,
                               interpolation=options.subpixel_interpolation)
            diagnostics = self.angle_diagnostics
            if diagnostics is not None and diagnostics.full and frame_index is not None:
                from roop_3d.engine.volumetric.angle_diagnostics import frame_metrics
                diagnostics.record_composite(frame_index, replacement.track_id,
                                             frame_metrics(frame, output, fuse_alpha, fuse_inverse))
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
        if getattr(replacement, "full_head", False) and options.swap_scope != "inner_face":
            # Legacy inner-face scope keeps the analysis matte (target landmark hull AND semantic face): the swap stays inside it.
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
        if options.robust.mask_flow:
            # Flow-aligned history instead of the position-blind EMA; the occluder above already went through the
            # flow-based OcclusionTracker (same FlowEstimator), so only the face matte needs this.
            stable_face = self.mask_flow.update(replacement.track_id, analysis.face_alpha,
                                                cv2.cvtColor(work_crop, cv2.COLOR_BGR2GRAY),
                                                frame_index=frame_index, scene_cut=scene_cut).alpha
        else:
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
        velocity = self._face_motion(replacement, frame_index, scene_cut)
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
                        occluder_masks, mask_temporal_context, lip_sync_results, options, observer,
                        background_donors=None) -> np.ndarray:
        from roop_3d.compositor.robust_fusion import RobustFusionOptions, fuse_robust
        robust = options.robust
        self._background_donors = background_donors or {}
        if self._silhouette_history is not None:
            self._silhouette_history.retain(active_tracks)
            if scene_cut:
                self._silhouette_history.clear()
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
        sensor_active = options.restoration_blend > 0 and self.restorer is not None
        fusion_options = RobustFusionOptions(refine_matte=robust.refine_matte, decontaminate=robust.edge_decontamination,
                                             match_sharpness=robust.sharpness_matching and not sensor_active,
                                             texture_strength=0.0 if sensor_active else robust.texture_strength,
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
            if options.hairline_offset_px:
                face_alpha = offset_hairline(face_alpha, options.hairline_offset_px, scale)
            coverage = getattr(replacement, 'coverage', None)
            if coverage is not None:
                coverage = np.asarray(coverage, np.float32)
                if coverage.shape != crop.shape[:2] or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
                    raise DomainError('PROJECTION_COVERAGE_INVALID', 'Projected coverage must be finite crop-space values in 0..1', 422)
                coverage = cv2.resize(coverage, (work, work), interpolation=cv2.INTER_LINEAR)
            if getattr(replacement, "full_head", False):
                output = self._remove_old_silhouette(output, context, front, options, original=frame, frame_index=frame_index)

            # Dual BiSeNet forbidden extraction on robust analysis crops
            _, _, robust_forbidden = compute_dual_bisenet_forbidden_mask(
                self.parser, context.work_crop, crop, forbidden_classes=options.forbidden_classes())
            work_points = self._landmarks_to_crop(landmarks_by_track.get(replacement.track_id), context.work_inverse)
            if coverage is not None:
                xseg_mask = (1.0 - (front if front is not None else 0.0))
                active_mask = compose_active_mask(coverage, xseg_mask, robust_forbidden)
                active_mask = protect_matte(active_mask, context.work_crop, work_points, options, scale)
                visible_alpha = active_mask * options.swap_weight
            else:
                visible_alpha = face_alpha * (1.0 - (front if front is not None else 0.0)) * options.swap_weight
                if robust_forbidden.any():
                    visible_alpha *= (1.0 - robust_forbidden)
                visible_alpha = protect_matte(visible_alpha, context.work_crop, work_points, options, scale)
            if front is not None:
                visible_alpha *= (1.0 - front)
            recovery_context = mask_temporal_context.get(replacement.track_id)
            if recovery_context is not None:
                recovery_context = dict(recovery_context)
                recovery_context.setdefault("current_guide", context.target_crop)
                if front is not None:
                    recovery_context["occluder_mask"] = front
                else:
                    recovery_context.pop("occluder_mask", None)
            mask_report, recovered_alpha = self.watchdog.observe_and_recover(
                replacement.track_id, visible_alpha, frame_index=frame_index, scene_cut=scene_cut,
                context=recovery_context)
            if mask_report["recovery"] == "bidirectional_optical_flow":
                # Keep recovery from overriding the global blend control. The input
                # to the watchdog contains swap_weight, while its interpolated
                # context masks may be unweighted matte probabilities.
                visible_alpha = self._apply_recovered_alpha(recovered_alpha, options.swap_weight, coverage)
                log.info("Temporal mask recovery track=%s frame=%s diagnostics=%s",
                         replacement.track_id, frame_index, mask_report)
            if mask_report["anomaly"]:
                log.warning("Temporal mask anomaly track=%s frame=%s diagnostics=%s", replacement.track_id,
                            frame_index, mask_report)
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
                context.report["mask_watchdog"] = mask_report
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
                mouth_mask=mouth, local_illumination=robust.local_illumination, frame=frame)
            restorer_seconds += spent
            harmonized, detail_report = self._stabilize_detail(
                replacement.track_id, harmonized, crop, context.target_crop, alpha_native,
                frame_index=frame_index, scene_cut=scene_cut, protect=protect)
            sensor_alpha = alpha_native
            if analysis.regions is not None:
                ears = self._resize_plane(analysis.regions.get("ears"), side)
                sensor_alpha = np.maximum(sensor_alpha, ears)
            if front is not None:
                sensor_front = self._resize_plane(front, side)
                sensor_alpha = sensor_alpha * (1.0 - sensor_front)
            harmonized, sensor_report = self._finish_sensor(
                replacement, harmonized, context.target_crop, sensor_alpha, skin_native, context.motion, options,
                frame_index, grain_strength=options.sensor_grain_strength * robust.texture_strength,
                match_optics=robust.sharpness_matching)
            context.report["target_sensor"] = sensor_report
            context.report["temporal_detail"] = detail_report
            context.report["mask_watchdog"] = mask_report
            temporal = self.temporal.update(replacement.track_id, replacement.inverse, face_alpha, color_stats,
                                            frame_index=frame_index, scene_cut=False)
            if getattr(replacement, "full_head", False):
                # Pose was already filtered in 3D before rasterization. Filtering
                # this ROI transform again detaches the rendered depth/silhouette
                # from its camera and from the target-minus-source ribbon.
                temporal = replace(temporal, inverse=replacement.inverse)
            enhanced_side = harmonized.shape[0]
            if observer is not None:
                observer('color', harmonized,
                         scale_matrix_input(replacement.inverse, enhanced_side / side) if enhanced_side != side else replacement.inverse,
                         self._resize_plane(alpha_native, enhanced_side) if enhanced_side != side else alpha_native)
            smoothed_face = temporal.alpha * options.swap_weight
            if coverage is not None:
                smoothed_face = np.minimum(smoothed_face, coverage)
            if observer is not None:
                observer('vfx', crop, temporal.inverse, cv2.resize(smoothed_face, (side, side), interpolation=cv2.INTER_AREA) if work != side else smoothed_face)
            skin_for_fusion = analysis.skin if analysis.parser_trusted else None
            if enhanced_side != side:
                # Fuse at the enhancer's resolution (capped at ~1.25x the face's footprint in the frame so the warp does
                # not alias): the masks follow the image, and M_hr re-bases the crop->frame matrix by S.
                footprint = float(np.sqrt(abs(np.linalg.det(np.asarray(temporal.inverse, np.float64)[:, :2]))))
                fuse_side = max(work, min(enhanced_side, int(round(side * footprint * 1.25))))
                source = (harmonized if fuse_side == enhanced_side else
                          cv2.resize(harmonized, (fuse_side, fuse_side), interpolation=cv2.INTER_AREA))
                smoothed_face = self._resize_plane(smoothed_face, fuse_side)
                front = self._resize_plane(front, fuse_side)
                skin_for_fusion = self._resize_plane(skin_for_fusion, fuse_side)
                work_inverse = scale_matrix_input(temporal.inverse, fuse_side / side)
            else:
                source = cv2.resize(harmonized, (work, work), interpolation=cv2.INTER_CUBIC) if work != side else harmonized
                work_inverse = self._scaled_inverse(temporal.inverse, scale)
            strand_kwargs = {}
            if robust.strand_matte:
                if analysis.regions is not None and analysis.parser_trusted:
                    strand_kwargs = {"hair": self._resize_plane(analysis.regions.get("hair"), source.shape[0]),
                                     "strand": robust.strand}
                else:
                    context.report["reasons"].append("STRAND_MATTE_NEEDS_TRUSTED_PARSER")
            output, fusion = fuse_robust(output, source, work_inverse, smoothed_face, front,
                                         skin_for_fusion,
                                         frame_motion=context.motion, mode=options.mode, levels=options.pyramid_levels,
                                         options=fusion_options, **strand_kwargs)
            context.report["fusion"] = {"sharpness": fusion.sharpness, "texture_gain": fusion.texture_gain,
                                        "occluded_fraction": fusion.occluded_fraction,
                                        "strand_fraction": fusion.strand_fraction}
        self.frames_composited += 1
        total_ms = (time.perf_counter() - compose_began) * 1000
        restorer_ms = restorer_seconds * 1000
        self.last_timing = {"masking": max(0.0, total_ms - restorer_ms), "restorer": restorer_ms}
        return output

    def _remove_old_silhouette(self, frame, context, front, options, *, original, frame_index=None):
        """Stage 6B: replace the target's silhouette with the source head's, BEFORE the source face is fused.

        The overhang ``dilate(M_tgt) & ~M_src`` and whatever the hair / ears / neck vacate when they are warped onto the new
        perimeter are filled with background (temporal donors, then a multi-scale spatial fill); facial skin never donates.
        ``original`` is the UNSWAPPED input frame: it is what the donor history keeps."""
        from roop_3d.rendering.silhouette_replacement import DonorHistory, SilhouetteReplacer, resolve_scope
        from roop_3d.rendering.silhouette_inpaint import DonorFrame
        plan = resolve_scope(options.swap_scope, options.skull_blend)
        item = context.replacement
        if not plan.replace_silhouette:
            context.report["silhouette_completion"] = {"scope": plan.scope, "applied": False,
                                                       "reason": "SCOPE_KEEPS_TARGET_SILHOUETTE"}
            return frame
        target = getattr(item, "target_silhouette", None)
        source = getattr(item, "source_silhouette", None)
        if source is None:
            source = item.coverage
        if target is None or source is None or options.swap_weight == 0:
            return frame
        side = context.work_crop.shape[0]
        planes = []
        for plane in (target, source):
            plane = np.asarray(plane, np.float32)
            if plane.shape != item.image.shape[:2] or not np.isfinite(plane).all() or np.any((plane < 0) | (plane > 1)):
                raise DomainError("HEAD_SILHOUETTE_INVALID", "Head silhouettes must be finite native crop masks in 0..1", 422)
            planes.append(cv2.resize(plane, (side, side)))
        target, source = planes
        analysis = context.analysis
        structure = None
        # Semantic skin may extend the measured jaw into the forehead/temples.
        if analysis.parser_trusted:
            target = np.maximum(target, analysis.skin)
            if analysis.regions is not None:
                structure = np.clip(np.maximum.reduce([analysis.regions.get(name) for name in ("hair", "ears", "neck")]), 0, 1)
        # Reflected out-of-frame model padding is never valid background.
        valid = cv2.warpAffine(np.ones(frame.shape[:2], np.uint8),
                               cv2.invertAffineTransform(context.work_inverse), (side, side)) > 0
        if self._silhouette_replacer is None:
            self._silhouette_replacer = SilhouetteReplacer()
        history = self._silhouette_history
        if history is None or history.depth != options.silhouette_history:
            history = self._silhouette_history = DonorHistory(options.silhouette_history)
        track = item.track_id
        past = history.donors(track, original.shape[:2], frame_index, max_gap=max(2, 2 * history.depth))
        donors = []
        for donor in (*past, *self._background_donors.get(track, ())):
            image = np.asarray(donor.image)
            if image.shape != original.shape or image.dtype != np.uint8:
                continue
            head = np.zeros((side, side), np.float32)
            if donor.head_mask is not None and np.asarray(donor.head_mask).shape == original.shape[:2]:
                head = self._crop_to_frame(np.asarray(donor.head_mask, np.float32), context.work_inverse, (side, side))
            donors.append(DonorFrame(self._crop_to_frame(image, context.work_inverse, (side, side)),
                                     np.maximum(head, (~valid).astype(np.float32))))
        perimeter = None
        source_xy, target_xy = getattr(item, "source_perimeter", None), getattr(item, "target_perimeter", None)
        if source_xy is not None and target_xy is not None:
            k = side / item.image.shape[0]
            perimeter = (np.asarray(source_xy, np.float64) * k, np.asarray(target_xy, np.float64) * k)
        edited, changed, outcome = self._silhouette_replacer.apply(
            context.work_crop, target_mask=target, source_mask=source, plan=plan, structure_mask=structure, occluder=front,
            donors=donors, perimeter=perimeter, invalid=~valid)
        head = np.maximum(target, structure) if structure is not None else target
        history.push(track, original, cv2.warpAffine(head.astype(np.float32), np.asarray(context.work_inverse, np.float32),
                                                     (original.shape[1], original.shape[0])), frame_index)
        context.report["silhouette_completion"] = {"method": "silhouette_delta", "pixels": int(changed.sum()), **outcome.as_dict()}
        if not changed.any():
            return frame
        alpha = np.maximum(changed, cv2.GaussianBlur(changed, (0, 0), 1.0)) * options.swap_weight
        return fuse_crop(frame, edited, context.work_inverse, alpha, mode=BlendMode.FEATHER,
                         interpolation=options.subpixel_interpolation)

    def capability(self) -> dict:
        parser = self.parser.capability() if hasattr(self.parser, "capability") else {
            "weights": "UNKNOWN", "provenance": "UNKNOWN", "license": "UNKNOWN", "inference": "UNKNOWN"}
        restorer = self.restorer.capability() if self.restorer is not None else {
            "weights": "NOT_CONFIGURED", "inference": "UNKNOWN"}
        result = {"state": "AVAILABLE", "parser": parser, "restorer": restorer,
                  "expression_restorer": {"reports": dict(self.expression_reports),
                      "models": self.expression_restorer.portrait.capability() if self.expression_restorer and self.expression_restorer.portrait else None},
                  "temporal_tracks": self.temporal.track_count,
                  "temporal_cache_bytes": self.temporal.cache_bytes,
                  "temporal_cache_limit_bytes": self.temporal.max_cache_bytes,
                  "temporal_detail": self.detail_temporal.capability(),
                  "temporal_diagnostics": dict(self.last_reports),
                  "frames_composited": self.frames_composited,
                  "photometric_identity": dict(self.last_relighting),
                  "occlusion_modes": {"default": "standard", "available": list(OCCLUSION_MODES)}}
        if self._analyzer is not None:
            result["robust_occlusion"] = self._analyzer.capability()
        return result

    def release(self) -> None:
        if self.expression_restorer is not None:
            self.expression_restorer.release()
            self.expression_restorer = None
        self.expression_reports.clear()
        from roop_3d.reconstruction.photometric_gpu import release_renderer
        release_renderer()
        if self._relight_recognizer is not None:
            self._relight_recognizer.release_for_build()
            self._relight_recognizer = None
        self.last_relighting.clear()
        if self._silhouette_history is not None:
            self._silhouette_history.clear()
        if self.parser is not None and hasattr(self.parser, "release"):
            self.parser.release()
        if self._analyzer is not None:
            self._analyzer.release()
        if self.restorer is not None:
            self.restorer.release()
        if self._occlusion_tracker is not None:
            self._occlusion_tracker.clear()
        self.temporal.clear()
        self.detail_temporal.release()
        self.mask_ema.clear()
        self.mask_flow.reset()
        self.histogram_stabilizer.reset()
        self.watchdog.clear()
