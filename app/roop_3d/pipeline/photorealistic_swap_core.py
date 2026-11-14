"""Photorealistic 3D head compositor: silhouette geometry + rendered head layer + quotient-image relight + Laplacian composite.

The input is a rendered true-3D head layer (``FacePacket.rendered_crop`` with its coverage plane). The former 2D neural-swap
crop, landmark thin-plate rectification, FaceSet photograph patch and aligned-crop GAN restorer were removed.

Steps per face (``Photorealistic3DSwapCore.process_face``)::

  1. geometry   target outline vs source-locked outline -> overhang inpaint + perimeter-cage warp of hair / ears / neck
                 (rendering.silhouette_replacement.SilhouetteReplacer; screen-space, kept pending the full-head render path)
  2. rectify    the rendered layer is placed in the ROI (affine of the layer canvas; no TPS, no photograph patch)
  3. relight    R_sh = Render_SH(N, L_target) / (Render_SH(N, L_source) + 1e-4) multiplied onto the layer in linear light,
                 then a bounded TRUST-style skin-tone gain
  4. optics     camera focus + grain matching, occluder-aware alpha, then a 4-band Laplacian composite onto the step-1 plate
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Callable, Sequence

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.pipeline import image_ops as ops
from roop_3d.pipeline.image_ops import LUMA_BGR
from roop_3d.reconstruction.landmark_layout import BLOCKS, layout_is_plausible
from roop_3d.relighting_3d import SphericalHarmonicsLighting
from roop_3d.rendering.silhouette_inpaint import DonorFrame
from roop_3d.rendering.silhouette_replacement import SilhouetteReplacer, resolve_scope
from roop_3d.rendering.skin_calibration import (SceneIllumination, UnbiasedSkinToneCalibrator, linear_to_srgb,
                                                srgb_to_linear)

MAX_CONTOUR_SHIFT = 0.30        # per-point cap on the jaw move, as a fraction of face width (the spline stays fold-free)
STEP_NAMES = ("geometry", "rectify", "relight", "restore", "blend")   # "rectify" = layer placement, "restore" = optics matching


@dataclass(frozen=True)
class SwapCoreOptions:
    swap_scope: str = "full_craniofacial"        # full_craniofacial | adaptive_skull | inner_face
    skull_blend: float = 1.0                     # adaptive_skull: share of beta_source in the skull
    forehead_lift: float = 0.42                  # forehead arc height, as a fraction of the brow-to-chin distance
    relight_strength: float = 0.85
    sh_epsilon: float = 1e-4
    ratio_clip: tuple[float, float] = (0.4, 2.2)
    ratio_sigma_frac: float = 0.02               # smoothing of R_sh, as a fraction of the face width (never below 3 px)
    skin_tone_strength: float = 0.6              # TRUST gain exponent (0 = off)
    skin_tone_max_gain: float = 1.15
    match_focus: bool = True
    match_grain: bool = True
    pyramid_levels: int = 4
    feather_frac: float = 0.02                   # alpha feather, as a fraction of the face width
    roi_margin_frac: float = 0.45

    def __post_init__(self) -> None:
        resolve_scope(self.swap_scope, self.skull_blend)
        if not (0 <= self.relight_strength <= 1
                and 0 <= self.skin_tone_strength <= 1 and self.skin_tone_max_gain >= 1 and 1 <= self.pyramid_levels <= 6
                and 0 <= self.forehead_lift <= 1):
            raise ValueError("invalid swap-core options")


@dataclass
class FacePacket:
    """Everything the core needs about ONE face in ONE frame. Arrays are in FRAME pixels unless the name says ``crop``."""
    track_id: int
    rendered_crop: np.ndarray                    # uint8 BGR, the rendered true-3D head layer in its ROI canvas (S x S)
    crop_to_frame: np.ndarray                    # 2x3 affine, crop px -> frame px
    target_landmarks: np.ndarray                 # (106, 2) frame px
    crop_coverage: np.ndarray | None = None      # float (S, S): where the swap is valid; None -> whole crop
    source_landmarks: np.ndarray | None = None   # (106, 2) in ``source_view`` pixels; None -> geometry step skipped
    source_view: np.ndarray | None = None        # uint8 BGR photograph of the source from the nearest pose
    structure_mask: np.ndarray | None = None     # float (H, W) hair + ears + neck of the target (parser), frame space
    skin_mask: np.ndarray | None = None          # float (H, W) target skin (parser), frame space
    occluder: np.ndarray | None = None           # float (H, W) hands / glasses / strands in front of the face
    normals: np.ndarray | None = None            # (H, W, 3) mesh normals in frame space; None -> silhouette ellipsoid
    projected: bool = False                     # upstream mesh/projection owns framing and coverage; never rectify it again
    presence: float = 1.0                       # fade the whole face edit, including silhouette replacement
    mask_diagnostics: dict = field(default_factory=dict)


@dataclass
class FaceReport:
    track_id: int
    steps_ms: dict[str, float] = field(default_factory=dict)
    geometry: dict = field(default_factory=dict)
    rectify: dict = field(default_factory=dict)
    relight: dict = field(default_factory=dict)
    restore: dict = field(default_factory=dict)
    blend: dict = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"track_id": self.track_id, "steps_ms": {k: round(v, 2) for k, v in self.steps_ms.items()},
                "geometry": self.geometry, "rectify": self.rectify, "relight": self.relight, "restore": self.restore,
                "blend": self.blend, "fallbacks": list(self.fallbacks)}


@dataclass
class _Geometry:
    """Step-1 plan in ROI coordinates."""
    enabled: bool
    reason: str
    roi: tuple[int, int, int, int]
    target_pts: np.ndarray
    locked_pts: np.ndarray
    p_src: np.ndarray | None
    similarity: np.ndarray | None                # source view px -> ROI px
    lock: float
    face_width: float
    target_mask: np.ndarray
    locked_mask: np.ndarray
    contour_from: np.ndarray
    contour_to: np.ndarray
    max_shift_px: float
    down: np.ndarray = None            # unit vector brow -> chin (ROI axes)
    brow_mid: np.ndarray = None        # brow centre, ROI px
    face_height: float = 0.0


@dataclass
class _Layer:
    inner: np.ndarray                            # float32 BGR 0..255, ROI
    inner_alpha: np.ndarray
    patch: np.ndarray | None
    patch_alpha: np.ndarray | None


class HairDonorReplacer(SilhouetteReplacer):
    """``SilhouetteReplacer`` with one change: warped-away hair / ears / neck may DONATE to the fill.

    The stock step excludes the structure layer from the donor set, so when hair lies behind the old jaw the nearest "background" it
    can propagate from is whatever lies beyond the hair (a backdrop, a border): a fringe of the wrong colour along the hair edge.
    Hair is what really is behind a jaw, so here it continues into the hole. Everything else (cage field, structure warp, ``keep``
    protection of the structure, occluder handling) is the stock behaviour."""

    def apply(self, frame, *, target_mask, source_mask, plan, structure_mask=None, occluder=None, donors=(), perimeter=None, invalid=None):
        from roop_3d.rendering.perimeter_cage import perimeter_from_masks
        from roop_3d.rendering.silhouette_inpaint import SilhouetteDeltaInpainter, as_mask
        from roop_3d.rendering.silhouette_replacement import SilhouetteReplacementReport
        image = SilhouetteDeltaInpainter._image(frame)
        shape = image.shape[:2]
        if not plan.replace_silhouette:
            return super().apply(frame, target_mask=target_mask, source_mask=source_mask, plan=plan, structure_mask=structure_mask,
                                 occluder=occluder, donors=donors, perimeter=perimeter, invalid=invalid)
        target, source = as_mask(target_mask, shape, "target mask"), as_mask(source_mask, shape, "source mask")
        front = np.zeros(shape, bool) if occluder is None else as_mask(occluder, shape, "occluder") > 0.01
        structure = np.zeros(shape, np.float32) if structure_mask is None else as_mask(structure_mask, shape, "structure mask") * (~front)
        has_structure = bool((structure > 0.5).any())
        warped_structure = warped_image = cage = None
        origin, vacated = "none", np.zeros(shape, bool)
        near_source = cv2.dilate((source > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if has_structure:
            src_xy, tgt_xy = perimeter if perimeter is not None else perimeter_from_masks(source, target)
            origin = "mesh" if perimeter is not None else "silhouettes"
            field = self.warper.estimate(src_xy, tgt_xy, shape)
            cage = field.report
            if not field.identity:
                warped_structure = self.warper.warp(field, structure) * (~near_source) * (~front)
                warped_image = self.warper.warp(field, image)
                vacated = (structure > 0.5) & (warped_structure < 0.5) & ~near_source & ~front
        excluded = np.zeros(shape, bool) if invalid is None else as_mask(invalid, shape, "invalid mask") > 0.5
        filled, inpaint = self.inpainter.fill(image, target, source, occluder=front if front.any() else None, donors=donors,
                                              extra_hole=vacated if vacated.any() else None,
                                              extra_excluded=excluded if excluded.any() else None,
                                              keep=((structure > 0.5) & ~vacated) if has_structure else None)
        out = filled
        if warped_structure is not None:
            alpha = np.clip(warped_structure, 0, 1)[..., None]
            out = np.clip(np.rint(filled.astype(np.float32) * (1 - alpha) + warped_image.astype(np.float32) * alpha), 0, 255).astype(np.uint8)
        changed = np.any(out != image, axis=2).astype(np.float32)
        report = SilhouetteReplacementReport(plan.scope, True, perimeter_source=origin, structure_pixels=int((structure > 0.5).sum()),
                                             vacated_pixels=int(vacated.sum()), cage=cage, inpaint=inpaint)
        return out, changed, report


class Photorealistic3DSwapCore:
    def __init__(self, options: SwapCoreOptions | None = None, *, replacer: SilhouetteReplacer | None = None,
                 calibrator: UnbiasedSkinToneCalibrator | None = None, lighting: SphericalHarmonicsLighting | None = None):
        self.options = options or SwapCoreOptions()
        self.replacer = replacer or HairDonorReplacer()
        self.calibrator = calibrator or UnbiasedSkinToneCalibrator()
        self.lighting = lighting or SphericalHarmonicsLighting()

    # ================================================================================================ public API
    def process_face(self, frame: np.ndarray, packet: FacePacket, *, donors: Sequence[DonorFrame] = (),
                     frame_index: int = 0) -> tuple[np.ndarray, FaceReport]:
        """Return ``(frame with this face swapped, report)``. ``frame`` is never modified."""
        _check_frame(frame)
        crop = np.asarray(packet.rendered_crop)
        matrix = np.asarray(packet.crop_to_frame, np.float64)
        if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3 or min(crop.shape[:2]) < 1:
            raise DomainError("SWAP_CORE_CROP_INVALID", "The neural swap crop must be nonempty uint8 BGR", 422)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-12:
            raise DomainError("SWAP_CORE_TRANSFORM_INVALID", "Crop-to-frame transform must be finite and invertible", 422)
        if packet.crop_coverage is not None:
            coverage = np.asarray(packet.crop_coverage, np.float32)
            if coverage.shape != crop.shape[:2] or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
                raise DomainError("SWAP_CORE_COVERAGE_INVALID", "Coverage must be a finite crop-sized probability plane", 422)
        elif packet.projected:
            raise DomainError("SWAP_CORE_COVERAGE_REQUIRED", "Projected replacements require rasterized coverage", 422)
        if not np.isfinite(packet.presence) or not 0 <= packet.presence <= 1:
            raise DomainError("SWAP_CORE_PRESENCE_INVALID", "Presence must be within 0..1", 422)
        report = FaceReport(packet.track_id)
        height, width = frame.shape[:2]
        began = time.perf_counter()
        geo = self._plan(frame, packet, report)
        x0, y0, x1, y1 = geo.roi
        roi = np.ascontiguousarray(frame[y0:y1, x0:x1])
        occluder = _crop(packet.occluder, geo.roi)
        structure = _crop(packet.structure_mask, geo.roi)
        skin = _crop(packet.skin_mask, geo.roi)

        # -- 1. geometry: outline vs outline -> plate with the old overhang filled and hair / ears / neck on the new skull
        plate = self._geometry_plate(roi, geo, structure, occluder, donors, report)
        report.steps_ms["geometry"] = (time.perf_counter() - began) * 1000

        # -- 2. rectify: neural swap + FaceSet photograph onto the source-locked skull
        began = time.perf_counter()
        layer = self._rectify(roi, packet, geo, structure, report)
        report.steps_ms["rectify"] = (time.perf_counter() - began) * 1000

        # -- 3. relight: quotient image + skin tone
        began = time.perf_counter()
        photo, alpha = self._relight(roi, layer, packet, geo, skin, report)
        report.steps_ms["relight"] = (time.perf_counter() - began) * 1000

        # -- 4a/b. restore + detail + optics
        began = time.perf_counter()
        photo = self._restore(roi, photo, alpha, layer, packet, geo, skin, frame_index, report)
        report.steps_ms["restore"] = (time.perf_counter() - began) * 1000

        # -- 4c. semantic alpha + 4-band Laplacian composite onto the plate
        began = time.perf_counter()
        if occluder is not None:
            alpha = alpha * (1.0 - np.clip(occluder, 0.0, 1.0))
        out_roi = ops.laplacian_pyramid_blend(photo, plate, alpha, self.options.pyramid_levels)
        if packet.presence < 1.0:
            out_roi = roi.astype(np.float32) + (out_roi.astype(np.float32) - roi) * packet.presence
        report.blend = {"levels": self.options.pyramid_levels, "alpha_mean": round(float(alpha.mean()), 4),
                        "alpha_pixels": int((alpha > 0.5).sum()), "occluder_applied": occluder is not None,
                        "presence": float(packet.presence), "analysis": packet.mask_diagnostics}
        result = frame.copy()
        result[y0:y1, x0:x1] = out_roi
        report.steps_ms["blend"] = (time.perf_counter() - began) * 1000
        return result, report

    # ================================================================================================ step 1
    def _plan(self, frame: np.ndarray, packet: FacePacket, report: FaceReport) -> _Geometry:
        opt = self.options
        height, width = frame.shape[:2]
        target = np.asarray(packet.target_landmarks, np.float64)
        plan = resolve_scope(opt.swap_scope, opt.skull_blend)
        lock = float(plan.source_identity_lock)
        reason = ""
        p_src = similarity = None
        if not layout_is_plausible(target):
            raise DomainError("SWAP_CORE_LANDMARKS_INVALID", "Target 106-point layout is not plausible; the face is not swapped", 422)
        frame_axes = ops.head_frame(target)
        if packet.projected:
            reason = "UPSTREAM_PROJECTION_OWNS_GEOMETRY"
        elif not plan.replace_silhouette:
            reason = "SCOPE_KEEPS_TARGET_SILHOUETTE"
        elif packet.source_landmarks is None or packet.source_view is None:
            reason = "NO_SOURCE_GEOMETRY"
        elif not layout_is_plausible(np.asarray(packet.source_landmarks, np.float64)):
            reason = "SOURCE_LANDMARKS_IMPLAUSIBLE"
        else:
            try:
                anchors = ops.inner_anchor_indices()
                similarity = ops.similarity_fit(np.asarray(packet.source_landmarks, np.float64)[anchors], target[anchors])
                p_src = ops.apply_affine(similarity, packet.source_landmarks)
            except ValueError as error:
                reason = f"SOURCE_ALIGNMENT_FAILED: {error}"
        enabled = p_src is not None and not reason
        # source-locked landmark set: contour (and brows) interpolated from target to the aligned source by `lock`
        locked = target.copy()
        contour_ids = np.array(BLOCKS["contour"])
        shift = np.zeros(len(contour_ids))
        if enabled:
            move = p_src[contour_ids] - target[contour_ids]
            cap = MAX_CONTOUR_SHIFT * frame_axes.width
            norms = np.linalg.norm(move, axis=1, keepdims=True)
            move = move * np.minimum(1.0, cap / np.maximum(norms, 1e-9)) * lock
            locked[contour_ids] = target[contour_ids] + move
            shift = np.linalg.norm(move, axis=1)
        lift = opt.forehead_lift
        tgt_poly = ops.head_polygon(target, lift)
        arc_t = ops.forehead_arc(target, lift)
        if enabled:
            arc_s = ops.forehead_arc(p_src, lift)
            arc_l = (1 - lock) * arc_t + lock * arc_s
            locked_poly = cv2.convexHull(np.vstack((locked[contour_ids], arc_l)).astype(np.float32))[:, 0, :].astype(np.float64)
        else:
            locked_poly = tgt_poly
        union = np.vstack((tgt_poly, locked_poly))
        if packet.projected:
            ys, xs = np.nonzero(np.asarray(packet.crop_coverage) > 0)
            if len(xs) == 0:
                raise DomainError("SWAP_CORE_COVERAGE_EMPTY", "Projected replacement has no visible surface", 422)
            bounds = np.array([[xs.min(), ys.min()], [xs.max() + 1, ys.min()],
                               [xs.max() + 1, ys.max() + 1], [xs.min(), ys.max() + 1]], np.float64)
            union = np.vstack((union, ops.apply_affine(packet.crop_to_frame, bounds)))
        margin = opt.roi_margin_frac * frame_axes.width
        x0 = int(max(0, np.floor(union[:, 0].min() - margin)))
        y0 = int(max(0, np.floor(union[:, 1].min() - margin)))
        x1 = int(min(frame.shape[1], np.ceil(union[:, 0].max() + margin)))
        y1 = int(min(frame.shape[0], np.ceil(union[:, 1].max() + margin)))
        if x1 - x0 < 32 or y1 - y0 < 32:
            raise DomainError("SWAP_CORE_ROI_EMPTY", "The face lies outside the frame", 422)
        origin = np.array((x0, y0), np.float64)
        shape = (y1 - y0, x1 - x0)
        feather = max(0.8, opt.feather_frac * frame_axes.width)
        geo = _Geometry(
            enabled=enabled, reason=reason, roi=(x0, y0, x1, y1), target_pts=target - origin, locked_pts=locked - origin,
            p_src=None if p_src is None else p_src - origin,
            similarity=None if similarity is None else ops.compose_affine(np.array([[1, 0, -x0], [0, 1, -y0]], np.float64), similarity),
            lock=lock, face_width=frame_axes.width,
            target_mask=ops.polygon_mask(tgt_poly - origin, shape, 0.0),
            locked_mask=ops.polygon_mask(locked_poly - origin, shape, 0.0),
            contour_from=target[contour_ids] - origin, contour_to=locked[contour_ids] - origin,
            max_shift_px=float(shift.max()) if len(shift) else 0.0,
            down=frame_axes.down, brow_mid=frame_axes.brow_mid - origin, face_height=frame_axes.height)
        if packet.projected:
            local = np.asarray(packet.crop_to_frame, np.float64).copy()
            local[:, 2] -= origin
            geo.locked_mask = cv2.warpAffine(np.asarray(packet.crop_coverage, np.float32), local, (shape[1], shape[0]),
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        report.geometry = {"mode": "projected_coverage" if packet.projected else "landmark_silhouette" if enabled else f"skipped:{reason}", "scope": opt.swap_scope,
                           "source_identity_lock": lock, "face_width_px": round(frame_axes.width, 1),
                           "max_contour_shift_px": round(geo.max_shift_px, 2), "roi": [x0, y0, x1, y1],
                           "skull_source": "upstream_projection" if packet.projected else
                                           "faceset_view_106_similarity_aligned" if enabled else "none"}
        if reason and not packet.projected:
            report.fallbacks.append(f"geometry:{reason}")
        return geo

    def _geometry_plate(self, roi: np.ndarray, geo: _Geometry, structure, occluder, donors, report: FaceReport) -> np.ndarray:
        if not geo.enabled or geo.max_shift_px < 0.5:
            if geo.enabled:
                report.geometry["silhouette"] = {"applied": False, "reason": "SILHOUETTES_ALREADY_MATCH"}
            return roi
        local = tuple(self._crop_donor(d, geo.roi) for d in donors)
        plan = resolve_scope(self.options.swap_scope, self.options.skull_blend)
        # Only the part of the outline BELOW the brow line is edited. The forehead arc is a heuristic (the 106 layout stops at the
        # brows) and the area above it is hair or backdrop, never "old skin to erase": cutting both masks identically keeps the
        # overhang a jaw / cheek / chin quantity, and the FaceSet photograph already covers the forehead.
        h, w = roi.shape[:2]
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        below = ((xx - geo.brow_mid[0]) * geo.down[0] + (yy - geo.brow_mid[1]) * geo.down[1] >= -0.02 * geo.face_height).astype(np.float32)
        keep = structure
        if structure is not None:
            # hair edge pixels are anti-aliased: protect them with a margin so they are never "filled with background"
            grow = max(2, int(round(0.025 * geo.face_width)))
            keep = cv2.dilate(structure.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
        plate, _, silhouette = self.replacer.apply(
            roi, target_mask=geo.target_mask * below, source_mask=geo.locked_mask * below, plan=plan, structure_mask=keep,
            occluder=occluder, donors=local, perimeter=(geo.contour_to, geo.contour_from))
        report.geometry["silhouette"] = silhouette.as_dict()
        return plate

    @staticmethod
    def _crop_donor(donor: DonorFrame, roi: tuple[int, int, int, int]) -> DonorFrame:
        x0, y0, x1, y1 = roi
        mask = None if donor.head_mask is None else np.asarray(donor.head_mask)[y0:y1, x0:x1]
        return DonorFrame(np.ascontiguousarray(np.asarray(donor.image)[y0:y1, x0:x1]), mask)

    # ================================================================================================ step 2
    def _rectify(self, roi: np.ndarray, packet: FacePacket, geo: _Geometry, structure, report: FaceReport) -> _Layer:
        opt = self.options
        h, w = roi.shape[:2]
        x0, y0 = geo.roi[0], geo.roi[1]
        crop = np.asarray(packet.rendered_crop)
        if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3:
            raise DomainError("SWAP_CORE_CROP_INVALID", "The neural swap crop must be uint8 BGR", 422)
        matrix = np.asarray(packet.crop_to_frame, np.float64).copy()
        matrix[:, 2] -= (x0, y0)
        swap = cv2.warpAffine(crop, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        coverage = np.ones(crop.shape[:2], np.float32) if packet.crop_coverage is None else np.asarray(packet.crop_coverage, np.float32)
        valid = cv2.warpAffine(np.clip(coverage, 0, 1), matrix, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if packet.projected:
            report.rectify = {"tps": {"applied": False, "reason": "UPSTREAM_PROJECTION_OWNS_GEOMETRY"},
                              "patch": {"used": False, "reason": "UNPROJECTED_PHOTO_FILL_FORBIDDEN"},
                              "inner_pixels": int((valid > 0.5).sum())}
            return _Layer(swap, valid, None, None)
        # The rendered head layer carries its own coverage; there is no landmark thin-plate field and no unprojected
        # photograph patch (both were 2D / 2.5D screen-space reconstructions of the face).
        report.rectify = {"tps": {"applied": False, "reason": "REMOVED_2D_SCREEN_SPACE_WARP"},
                          "patch": {"used": False, "reason": "UNPROJECTED_PHOTO_FILL_FORBIDDEN"},
                          "inner_pixels": int((valid > 0.5).sum())}
        return _Layer(swap, valid, None, None)

    # ================================================================================================ step 3
    def _relight(self, roi: np.ndarray, layer: _Layer, packet: FacePacket, geo: _Geometry, skin, report: FaceReport
                 ) -> tuple[np.ndarray, np.ndarray]:
        opt = self.options
        h, w = roi.shape[:2]
        mesh_normals = _crop(packet.normals, geo.roi, channels=3)
        normals = mesh_normals if mesh_normals is not None else ops.silhouette_normals(geo.locked_mask)
        sigma = max(3.0, opt.ratio_sigma_frac * geo.face_width)
        interior = cv2.erode((geo.locked_mask > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        target_skin = interior & ((skin > 0.5) if skin is not None else True)
        target_lin = srgb_to_linear(np.clip(roi.astype(np.float32) / 255.0, 0, 1))
        light_t = self.lighting.estimate_lighting(normals, target_lin * 255.0, target_skin)
        shading_t = self.lighting.evaluate_shading(normals, light_t)
        info: dict = {"normals": "mesh" if mesh_normals is not None else "silhouette_ellipsoid", "ratio_sigma_px": round(sigma, 2),
                      "strength": opt.relight_strength,
                      "key_light_dir": [round(float(v), 3) for v in self.lighting.extract_key_light_direction(light_t)]}

        def relight_part(part: np.ndarray, part_alpha: np.ndarray, name: str) -> np.ndarray:
            linear = srgb_to_linear(np.clip(part / 255.0, 0, 1))
            region = interior & (part_alpha > 0.5)
            light_s = self.lighting.estimate_lighting(normals, linear * 255.0, region)
            shading_s = self.lighting.evaluate_shading(normals, light_s)
            ratio = ops.quotient_ratio(shading_t, shading_s, epsilon=opt.sh_epsilon, clip=opt.ratio_clip, sigma=sigma)
            m = ratio[region] if region.any() else ratio.reshape(-1, 3)
            info[name] = {"ratio_min": round(float(m.min()), 3), "ratio_max": round(float(m.max()), 3),
                          "ratio_mean": round(float(m.mean()), 3)}
            return ops.apply_ratio(linear, ratio, opt.relight_strength)

        inner_lin = relight_part(layer.inner, layer.inner_alpha, "inner")
        patch_lin = relight_part(layer.patch, layer.patch_alpha, "patch") if layer.patch is not None else None

        # -- TRUST-style skin tone: intrinsic (albedo) tone of the source photograph under the TARGET's scene light
        info["trust"] = {"applied": False, "reason": "DISABLED" if opt.skin_tone_strength <= 0 else "NO_SOURCE_PHOTOGRAPH"}
        if opt.skin_tone_strength > 0 and packet.source_view is not None:
            try:
                gain, confidence = self._skin_tone_gain(inner_lin, layer.inner_alpha, target_lin, packet.source_view, skin, interior)
                inner_lin = np.clip(inner_lin * gain, 0, 1)
                patch_lin = None if patch_lin is None else np.clip(patch_lin * gain, 0, 1)
                info["trust"] = {"applied": True, "gain_bgr": [round(float(v), 4) for v in gain],
                                 "illuminant_confidence": round(confidence, 2)}
            except (ValueError, DomainError) as error:
                info["trust"] = {"applied": False, "reason": str(error)[:80]}
                report.fallbacks.append("relight:TRUST_SKIPPED")
        inner = linear_to_srgb(inner_lin) * 255.0
        patch = None if patch_lin is None else linear_to_srgb(patch_lin) * 255.0
        photo, alpha = self._merge_layers(inner, layer.inner_alpha, patch, layer.patch_alpha, info)
        report.relight = info
        return photo, alpha

    def _skin_tone_gain(self, inner_lin, inner_alpha, target_lin, source_view, skin, interior) -> np.ndarray:
        opt = self.options
        # Exposure is the quotient image's job: both illuminants are pinned to unit intensity so only the WHITE BALANCE
        # (skin chromaticity under the scene light) can differ, and the result is luminance-normalised below.
        target_light = SceneIllumination.from_scene(np.clip(linear_to_srgb(target_lin), 0, 1))
        source_view = np.asarray(source_view)
        source_light = SceneIllumination.from_scene(source_view)
        confidence = float(min(target_light.confidence, source_light.confidence))
        target_light = replace(target_light, intensity=1.0)
        source_light = replace(source_light, intensity=1.0)
        src_h, src_w = source_view.shape[:2]
        yy, xx = np.mgrid[:src_h, :src_w]
        source_skin = ((((xx - (src_w - 1) / 2) / (src_w * 0.30)) ** 2 + ((yy - (src_h * 0.55)) / (src_h * 0.30)) ** 2) <= 1)
        swap_skin = (inner_alpha > 0.7) & interior & ((skin > 0.5) if skin is not None else True)
        if int(swap_skin.sum()) < 64:
            raise ValueError("TOO_FEW_SKIN_PIXELS")
        swap_srgb = np.clip(linear_to_srgb(inner_lin), 0, 1)
        swap_est = self.calibrator.decompose(swap_srgb, skin_mask=swap_skin, illumination=target_light)
        source_est = self.calibrator.decompose(source_view, skin_mask=source_skin, illumination=source_light)
        swap_albedo = np.median(swap_est.albedo_linear_bgr[swap_skin], axis=0)
        source_albedo = np.median(source_est.albedo_linear_bgr[source_skin], axis=0)
        # an illuminant guessed from a handful of pixels must not move skin tone: strength follows the estimate's confidence
        strength = opt.skin_tone_strength * confidence
        gain = (np.maximum(source_albedo, 1e-4) / np.maximum(swap_albedo, 1e-4)) ** strength
        gain = gain / max(float(gain @ LUMA_BGR), 1e-6)                    # chromatic only: luminance is left to the quotient image
        return np.clip(gain, 1.0 / opt.skin_tone_max_gain, opt.skin_tone_max_gain).astype(np.float32), confidence

    def _merge_layers(self, inner, inner_alpha, patch, patch_alpha, info: dict) -> tuple[np.ndarray, np.ndarray]:
        """Inner swap on top of the photograph; the photograph is colour-matched to the swap along their shared ring."""
        if patch is None:
            return inner.astype(np.float32), inner_alpha
        ring = (inner_alpha > 0.25) & (inner_alpha < 0.95) & (patch_alpha > 0.5)
        gain = np.ones(3, np.float32)
        if int(ring.sum()) >= 48:
            a = np.median(inner[ring], axis=0)
            b = np.median(patch[ring], axis=0)
            gain = np.clip(a / np.maximum(b, 1.0), 0.8, 1.25).astype(np.float32)
            patch = np.clip(patch * gain, 0, 255)
        info["patch_colour_match_gain"] = [round(float(v), 3) for v in gain]
        # Straight colour must be normalized by the union alpha. Otherwise the final composite applies inner_alpha twice
        # wherever patch_alpha is partial/zero, producing a dark or source-border-coloured jaw/forehead seam.
        patch_weight = (1.0 - inner_alpha) * patch_alpha
        alpha = np.clip(inner_alpha + patch_weight, 0.0, 1.0)
        photo = np.divide(inner * inner_alpha[:, :, None] + patch * patch_weight[:, :, None], alpha[:, :, None],
                          out=inner.copy(), where=alpha[:, :, None] > 0)
        return photo.astype(np.float32), alpha.astype(np.float32)

    # ================================================================================================ step 4
    def _restore(self, roi, photo, alpha, layer: _Layer, packet: FacePacket, geo: _Geometry, skin, frame_index: int,
                 report: FaceReport) -> np.ndarray:
        opt = self.options
        h, w = roi.shape[:2]
        info: dict = {"restorer": "none", "ran": False}
        photo_u8 = np.clip(photo, 0, 255).astype(np.uint8)
        info["reason"] = "NO_2D_RESTORATION"   # screen-space GAN restoration was removed; detail lives in the UV albedo

        skin_alpha = alpha * (1.0 if skin is None else np.clip(skin, 0, 1))
        photo_u8 = np.clip(photo, 0, 255).astype(np.uint8)
        target_plane = ops.luma(roi)
        target_skin = (cv2.erode((geo.target_mask > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        if skin is not None:
            target_skin &= skin > 0.5
        if opt.match_focus:
            photo_u8, sigma = ops.match_focus(photo_u8, skin_alpha, ops.laplacian_variance(target_plane, target_skin))
            info["focus_sigma_px"] = round(sigma, 2)
        if opt.match_grain:
            target_noise = ops.noise_sigma(target_plane, target_skin)
            layer_noise = ops.noise_sigma(ops.luma(photo_u8), skin_alpha > 0.5)
            extra = float(np.sqrt(max(target_noise ** 2 - layer_noise ** 2, 0.0)))
            extra = min(extra, 4.0)
            photo_u8 = ops.add_grain(photo_u8, skin_alpha, extra, seed=frame_index * 7919 + packet.track_id)
            info["grain"] = {"target_sigma": round(target_noise, 2), "layer_sigma": round(layer_noise, 2), "added_sigma": round(extra, 2)}
        report.restore = info
        return photo_u8.astype(np.float32)

def _check_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise DomainError("SWAP_CORE_FRAME_INVALID", "Frame must be a uint8 H x W x 3 BGR array", 422)


def _crop(value, roi: tuple[int, int, int, int], channels: int | None = None):
    if value is None:
        return None
    x0, y0, x1, y1 = roi
    array = np.asarray(value, np.float32)
    if channels is None:
        if array.ndim != 2:
            raise DomainError("SWAP_CORE_MASK_INVALID", "Masks must be 2-D float planes", 422)
        return np.ascontiguousarray(array[y0:y1, x0:x1])
    if array.ndim != 3 or array.shape[2] != channels:
        raise DomainError("SWAP_CORE_MASK_INVALID", f"Expected an H x W x {channels} array", 422)
    return np.ascontiguousarray(array[y0:y1, x0:x1])
