# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""StrandAwareOcclusionCompositor: four independent occlusion cues, plus a directional strand matte, in one swap alpha.

NeuralHaircut-inspired in its *goal* - treat hair as thin oriented structures instead of a blob - but not in its method:
NeuralHaircut optimises 3D strands from multi-view video; this module has one frame, so it works in 2D with an
orientation-selective filter bank. It does not reconstruct hair and does not claim to.

The final swap alpha is ``swap_alpha * grazing * (1 - occluder)`` where ``occluder`` is the union (noisy-OR) of

depth      3D Z-buffer test. A pixel whose *occluder* depth (monocular scene depth, or a rendered proxy for glasses / a hand)
           is nearer than the swapped face surface by more than a tolerance is in front of the face. The face depth is the
           rendered mesh z-buffer, so self-occlusion (a cheek hiding the far ear) is already resolved by the rasteriser.
semantic   face-parsing probabilities for glasses, hat and hair (``compositor.face_regions``).
object     a caller-supplied map for what no local parser knows (hands, microphones, props). No object-class model is
           installed (``docs/OCCLUSION_STAGE5.md``), so this module never invents one.
strand     high-frequency directional matting along hairline bangs and facial hair: a bank of even Gabor filters finds
           thin ridges, an orientation-coherence term (structure tensor) rejects isotropic skin texture and noise, the
           hair-flow direction measured inside the hair region gates the orientation, and a local colour line between the
           skin and the hair colour gates the contrast. Only within ``strand_reach_px`` of hair, so skin pores far from any
           hair are never promoted.

``grazing`` is not an occluder: the swapped texture is stretched where the surface turns away from the camera, so the swap is
faded out as N.V falls from ``grazing_cos_high`` to ``grazing_cos_low`` and the target's own pixels take over.

Inputs are pixel-aligned arrays of the target frame (depth/normals from ``PhysicalBuffers`` or ``vertex_raster``). Everything
is computed on the bounding box of the swap, so a 4K frame with a small face costs a small crop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from pydantic import Field

from roop_3d.compositor.boundary_fusion import BlendMode, fuse_warped
from roop_3d.compositor.face_regions import FaceRegions
from roop_3d.compositor.matte import fast_guided_filter
from roop_3d.core.errors import DomainError
from roop_3d.compositor.strand_matte import (  # noqa: F401  (re-exported: the public home of these names is here)
    _LUMA, StrandMatteOptions, _normalised_blur, _smoothstep, gabor_bank, strand_evidence)


class StrandOcclusionOptions(StrandMatteOptions):
    # 3D depth test
    depth_tolerance_m: float = Field(default=0.006, ge=0.0, le=0.1, description="Occluder must be this much nearer than the face")
    depth_softness_m: float = Field(default=0.004, gt=0.0, le=0.1)
    # grazing angle
    grazing_cos_low: float = Field(default=0.10, ge=0.0, lt=1.0, description="N.V at or below which the swap is fully faded")
    grazing_cos_high: float = Field(default=0.45, gt=0.0, le=1.0, description="N.V at or above which the swap is untouched")
    # semantic weights
    glasses_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    hat_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    hair_weight: float = Field(default=0.85, ge=0.0, le=1.0)
    # strand matting (the filter parameters are inherited from StrandMatteOptions)
    strand_enabled: bool = True
    # boundary snap
    boundary_snap: bool = True
    snap_radius_px: int = Field(default=3, ge=1, le=16)
    snap_eps: float = Field(default=2e-4, gt=0.0, le=1.0)


@dataclass(frozen=True)
class OcclusionInputs:
    target_bgr: np.ndarray                       # (H, W, 3) uint8, the target frame
    swap_alpha: np.ndarray                       # (H, W) 0..1: depth-tested coverage x the 3D swap mask
    face_depth: np.ndarray | None = None         # (H, W) camera depth of the swapped surface, smaller = closer, inf outside
    face_normals: np.ndarray | None = None       # (H, W, 3) camera-space unit normals
    regions: FaceRegions | None = None           # parser maps at frame resolution
    occluder_depth: np.ndarray | None = None     # (H, W) nearest occluder depth in the same units; inf/NaN = none known
    object_occluder: np.ndarray | None = None    # (H, W) hands, microphones, props
    facial_hair: np.ndarray | None = None        # (H, W) beard/moustache prior (a parser or a user brush)
    view_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)   # towards the viewer in the normals' frame


@dataclass(frozen=True)
class OcclusionReport:
    roi: tuple[int, int, int, int] | None
    swap_area_px: float
    removed_share: dict[str, float]              # per cue: share of the swap area (alpha-weighted) it covers
    occluded_share: float                        # share removed by the union of the occluders
    grazing_share: float                         # share faded by the grazing-angle term
    strand_pixels: int
    cues_available: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OcclusionResult:
    alpha: np.ndarray                            # (H, W) final swap alpha
    occluder: np.ndarray                         # (H, W) union of the occluders
    grazing: np.ndarray                          # (H, W) 1 = untouched, 0 = fully faded
    components: dict[str, np.ndarray]            # depth / semantic / object / strand (H, W) each
    swap_alpha: np.ndarray                       # the input coverage, for reference
    report: OcclusionReport = field(default=None)

    def occlusion_in(self, region: np.ndarray) -> float:
        """Share (0..1) of the swapped area inside ``region`` that the occluders remove: e.g. the mouth, for audio lock-in."""
        weight = np.clip(np.asarray(region, np.float32), 0.0, 1.0) * self.swap_alpha
        total = float(weight.sum())
        return float((weight * self.occluder).sum() / total) if total > 1e-6 else 0.0


def _noisy_or(*maps: np.ndarray | None) -> np.ndarray:
    result = None
    for item in maps:
        if item is None:
            continue
        result = item.copy() if result is None else 1.0 - (1.0 - result) * (1.0 - item)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


class StrandAwareOcclusionCompositor:
    def __init__(self, options: StrandOcclusionOptions | None = None) -> None:
        self.options = options or StrandOcclusionOptions()
        if self.options.grazing_cos_low >= self.options.grazing_cos_high:
            raise ValueError("grazing_cos_low must be below grazing_cos_high")
        if self.options.strand_response_low >= self.options.strand_response_high:
            raise ValueError("strand_response_low must be below strand_response_high")

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _check(inputs: OcclusionInputs) -> tuple[int, int]:
        target = np.asarray(inputs.target_bgr)
        if target.ndim != 3 or target.shape[2] != 3 or target.dtype != np.uint8:
            raise DomainError("OCCLUSION_INPUT_INVALID", "Target frame must be uint8 BGR", 422)
        shape = target.shape[:2]
        alpha = np.asarray(inputs.swap_alpha)
        if alpha.shape != shape or not np.isfinite(alpha).all():
            raise DomainError("OCCLUSION_INPUT_INVALID", "swap_alpha must be a finite map matching the frame", 422)
        for name in ("face_depth", "occluder_depth", "object_occluder", "facial_hair"):
            value = getattr(inputs, name)
            if value is not None and np.asarray(value).shape != shape:
                raise DomainError("OCCLUSION_INPUT_INVALID", f"{name} must match the frame", 422)
        if inputs.face_normals is not None and np.asarray(inputs.face_normals).shape != (*shape, 3):
            raise DomainError("OCCLUSION_INPUT_INVALID", "face_normals must be (H, W, 3)", 422)
        if inputs.regions is not None and tuple(inputs.regions.shape) != tuple(shape):
            raise DomainError("OCCLUSION_INPUT_INVALID", "Parser regions must be at frame resolution", 422)
        view = np.asarray(inputs.view_direction, np.float64)
        if view.shape != (3,) or not np.isfinite(view).all() or np.linalg.norm(view) < 1e-9:
            raise DomainError("OCCLUSION_INPUT_INVALID", "view_direction must be a finite non-zero 3-vector", 422)
        return shape

    # ------------------------------------------------------------------ cues
    def grazing_falloff(self, normals: np.ndarray, view_direction=(0.0, 0.0, 1.0)) -> np.ndarray:
        """1 where the surface faces the camera, falling to 0 as N.V drops from ``grazing_cos_high`` to ``grazing_cos_low``."""
        view = np.asarray(view_direction, np.float32)
        view = view / np.linalg.norm(view)
        n = np.asarray(normals, np.float32)
        length = np.linalg.norm(n, axis=-1)
        cosine = np.where(length > 1e-6, (n @ view) / np.maximum(length, 1e-6), 0.0)
        return _smoothstep(cosine, self.options.grazing_cos_low, self.options.grazing_cos_high)

    def depth_occlusion(self, face_depth: np.ndarray, occluder_depth: np.ndarray) -> np.ndarray:
        """Soft 0..1 where the occluder is nearer than the face by more than the tolerance (unknown depth = no occlusion)."""
        face = np.asarray(face_depth, np.float32)
        front = np.asarray(occluder_depth, np.float32)
        valid = np.isfinite(face) & np.isfinite(front)
        gap = np.full(face.shape, -np.inf, np.float32)
        gap[valid] = face[valid] - front[valid]
        o = self.options
        return _smoothstep(gap, o.depth_tolerance_m, o.depth_tolerance_m + o.depth_softness_m)

    def semantic_occlusion(self, regions: FaceRegions) -> np.ndarray:
        o = self.options
        return _noisy_or(regions.get("glasses") * o.glasses_weight, regions.get("hat") * o.hat_weight,
                         regions.get("hair") * o.hair_weight)

    # ------------------------------------------------------------------ main
    def evaluate(self, inputs: OcclusionInputs) -> OcclusionResult:
        shape = self._check(inputs)
        o = self.options
        swap = np.clip(np.asarray(inputs.swap_alpha, np.float32), 0.0, 1.0)
        zeros = np.zeros(shape, np.float32)
        cues = []
        notes = []
        ys, xs = np.nonzero(swap > 1e-4)
        if not len(xs):
            report = OcclusionReport(None, 0.0, {k: 0.0 for k in ("depth", "semantic", "object", "strand")}, 0.0, 0.0, 0,
                                     (), ("empty swap",))
            return OcclusionResult(zeros.copy(), zeros.copy(), np.ones(shape, np.float32),
                                   {k: zeros.copy() for k in ("depth", "semantic", "object", "strand")}, swap, report)
        margin = int(math.ceil(o.strand_reach_px + 3 * o.gabor_sigma_px + o.snap_radius_px + 4))
        x0, y0 = max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin)
        x1, y1 = min(shape[1], int(xs.max()) + margin + 1), min(shape[0], int(ys.max()) + margin + 1)
        window = (slice(y0, y1), slice(x0, x1))
        crop = lambda a: None if a is None else np.asarray(a)[window]
        swap_c = swap[window]
        target_c = np.ascontiguousarray(np.asarray(inputs.target_bgr)[window])

        grazing_c = np.ones_like(swap_c)
        if inputs.face_normals is not None:
            grazing_c = self.grazing_falloff(crop(inputs.face_normals), inputs.view_direction)
            cues.append("grazing")
        depth_c = np.zeros_like(swap_c)
        if inputs.occluder_depth is not None:
            if inputs.face_depth is None:
                notes.append("occluder_depth given without face_depth: depth cue skipped")
            else:
                depth_c = self.depth_occlusion(crop(inputs.face_depth), crop(inputs.occluder_depth))
                cues.append("depth")
        semantic_c = np.zeros_like(swap_c)
        hair_c = np.zeros_like(swap_c)
        if inputs.regions is not None:
            sliced = FaceRegions({k: v[window] for k, v in inputs.regions.maps.items()}, swap_c.shape)
            semantic_c = self.semantic_occlusion(sliced)
            hair_c = sliced.get("hair")
            cues.append("semantic")
        object_c = np.zeros_like(swap_c)
        if inputs.object_occluder is not None:
            object_c = np.clip(np.asarray(crop(inputs.object_occluder), np.float32), 0.0, 1.0)
            cues.append("object")
        strand_c = np.zeros_like(swap_c)
        facial_c = None if inputs.facial_hair is None else np.clip(np.asarray(crop(inputs.facial_hair), np.float32), 0.0, 1.0)
        if o.strand_enabled:
            prior = hair_c if facial_c is None else np.maximum(hair_c, facial_c)
            if prior.max() > 0.5:
                skin = np.clip(swap_c * (1.0 - prior), 0.0, 1.0)
                strand_c, _ = strand_evidence(target_c, prior, skin, o)
                cues.append("strand")
            else:
                notes.append("no hair or facial-hair prior: strand matting skipped")

        occluder_c = _noisy_or(depth_c, semantic_c, object_c, strand_c)
        alpha_c = swap_c * grazing_c * (1.0 - occluder_c)
        # Snap only where alpha actually CHANGES (its edges, widened by the filter radius) to the real image edges: a flat
        # partial plateau (a hair weight of 0.5, a grazing fade) is a decision, not a boundary, and a guided filter would
        # blur it. Never over strands, never above the geometric coverage.
        grad_y, grad_x = np.gradient(alpha_c)
        edge = np.clip(np.hypot(grad_x, grad_y) / 0.08, 0.0, 1.0).astype(np.float32)
        size = 2 * o.snap_radius_px + 1
        luma = (target_c.astype(np.float32) @ _LUMA) / 255.0
        # ... and only where the guide has structure: on a flat guide the filter degenerates into a box blur of radius ~2r
        # that would erase thin occluders (glasses frames, a strand) for no image edge to snap to.
        mean = cv2.boxFilter(luma, cv2.CV_32F, (size, size), borderType=cv2.BORDER_REPLICATE)
        spread = np.sqrt(np.maximum(cv2.boxFilter(luma * luma, cv2.CV_32F, (size, size), borderType=cv2.BORDER_REPLICATE)
                                    - mean * mean, 0.0))
        structure = _smoothstep(spread, 0.015, 0.05)
        ribbon = cv2.dilate(edge, np.ones((size, size), np.uint8)) * (1.0 - strand_c) * structure
        if o.boundary_snap and ribbon.max() > 0.05:
            snapped = fast_guided_filter(luma, alpha_c, radius=o.snap_radius_px, eps=o.snap_eps, s=1)
            alpha_c = alpha_c + (snapped - alpha_c) * ribbon
        alpha_c = np.clip(np.minimum(alpha_c, swap_c * grazing_c), 0.0, 1.0)

        def paste(small: np.ndarray, fill: float = 0.0) -> np.ndarray:
            full = np.full(shape, fill, np.float32)
            full[window] = small
            return full

        components = {"depth": paste(depth_c), "semantic": paste(semantic_c), "object": paste(object_c), "strand": paste(strand_c)}
        area = float(swap_c.sum())
        removed = {name: float((swap_c * crop(components[name])).sum() / area) if area > 1e-6 else 0.0 for name in components}
        report = OcclusionReport(
            (x0, y0, x1, y1), area, removed,
            float((swap_c * occluder_c).sum() / area) if area > 1e-6 else 0.0,
            float((swap_c * (1.0 - grazing_c)).sum() / area) if area > 1e-6 else 0.0,
            int(((strand_c > 0.5) & (swap_c > 0.05)).sum()), tuple(cues), tuple(notes))
        return OcclusionResult(paste(alpha_c), paste(occluder_c), paste(grazing_c, 1.0), components, swap, report)

    def composite(self, inputs: OcclusionInputs, swap_bgr: np.ndarray, *, mode: BlendMode | str = BlendMode.FEATHER,
                  levels: int = 4, linear_light: bool = True) -> tuple[np.ndarray, OcclusionResult]:
        """Evaluate the cues and blend ``swap_bgr`` into the target with the resulting alpha (shared fusion code)."""
        result = self.evaluate(inputs)
        swap = np.asarray(swap_bgr)
        if swap.shape != np.asarray(inputs.target_bgr).shape or swap.dtype != np.uint8:
            raise DomainError("OCCLUSION_INPUT_INVALID", "swap_bgr must be a uint8 image matching the target frame", 422)
        out = fuse_warped(np.asarray(inputs.target_bgr), swap, result.alpha, mode=mode, levels=levels, linear_light=linear_light)
        return out, result
