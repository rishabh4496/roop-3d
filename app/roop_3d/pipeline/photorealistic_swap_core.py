"""Composite a rasterized, shaded full-head mesh through a semantic matte.

The compositor accepts only render layers carrying the ordered true-3D stage
trace. Skin-only scene photometry and measured detail finishing follow rendering;
all neural restoration belongs to the canonical UV atlas, never a target crop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roop_3d.compositor.stage4 import anatomical_alpha, finish_skin_detail, local_lab_skin_match
from roop_3d.core.errors import DomainError
from roop_3d.core.true3d_guard import FRAME_PIPELINE_STAGES, require_true3d_frame_pipeline
from roop_3d.pipeline import image_ops as ops

STEP_NAMES = ("geometry", "rectify", "relight", "restore", "blend")




@dataclass(frozen=True)
class SwapCoreOptions:
    feather_px: float = 5.0
    pyramid_levels: int = 4
    restoration_weight: float = 0.35
    erosion_px: float = 0.35
    dilation_px: float = 0.65

    def __post_init__(self) -> None:
        if not np.isfinite(self.feather_px) or not 0 <= self.feather_px <= 64:
            raise ValueError("feather_px must be within 0..64")
        if not 1 <= self.pyramid_levels <= 6:
            raise ValueError("pyramid_levels must be within 1..6")
        if not np.isfinite(self.restoration_weight) or not 0 <= self.restoration_weight <= 1:
            raise ValueError("restoration_weight must be within 0..1")
        for name in ("erosion_px", "dilation_px"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 4:
                raise ValueError(f"{name} must be within 0..4")


@dataclass
class FacePacket:
    """Full-frame buffers emitted by a true-3D rasterizer and semantic parser."""
    track_id: int
    rendered_frame: np.ndarray
    coverage: np.ndarray
    skin_mask: np.ndarray
    hair_mask: np.ndarray
    normals: np.ndarray
    pipeline_stages: tuple[str, ...]
    occluder: np.ndarray | None = None
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
    pipeline_stages: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"track_id": self.track_id, "steps_ms": {k: round(v, 2) for k, v in self.steps_ms.items()},
                "geometry": self.geometry, "blend": self.blend, "fallbacks": list(self.fallbacks),
                "pipeline_stages": list(self.pipeline_stages)}


class Photorealistic3DSwapCore:
    """Blend a full-frame shaded mesh raster using full-frame semantic evidence."""

    def __init__(self, options: SwapCoreOptions | None = None):
        self.options = options or SwapCoreOptions()
        self.last_alpha: np.ndarray | None = None

    def process_face(self, frame: np.ndarray, packet: FacePacket, *, frame_index: int = 0) -> tuple[np.ndarray, FaceReport]:
        _check_frame(frame)
        self.last_alpha = None
        stages = require_true3d_frame_pipeline(packet.pipeline_stages, through=FRAME_PIPELINE_STAGES[3])
        shape = frame.shape[:2]
        rendered = np.asarray(packet.rendered_frame)
        if rendered.dtype != np.uint8 or rendered.shape != frame.shape:
            raise DomainError("TRUE3D_RENDER_INVALID", "The 3D renderer must return a full-frame uint8 BGR raster", 422)
        normals = np.asarray(packet.normals, np.float32)
        if normals.shape != (*shape, 3) or not np.isfinite(normals).all():
            raise DomainError("TRUE3D_MESH_NORMALS_REQUIRED", "Full-frame surface normals are required", 422)
        normal_lengths = np.linalg.norm(normals, axis=2)

        planes = {}
        for name in ("coverage", "skin_mask", "hair_mask"):
            plane = np.asarray(getattr(packet, name), np.float32)
            if plane.shape != shape or not np.isfinite(plane).all() or np.any((plane < 0) | (plane > 1)):
                raise DomainError("TRUE3D_SEMANTIC_MASK_INVALID", f"{name} must be a finite full-frame plane in 0..1", 422)
            planes[name] = plane
        coverage, skin, hair = (planes[name] for name in ("coverage", "skin_mask", "hair_mask"))
        if np.any((coverage > 0.5) & ((normal_lengths < 0.5) | (normal_lengths > 1.5))):
            raise DomainError("TRUE3D_MESH_NORMALS_REQUIRED", "Covered mesh pixels must carry normalized surface normals", 422)
        alpha = coverage * skin * (1.0 - hair)
        if packet.occluder is not None:
            occluder = np.asarray(packet.occluder, np.float32)
            if occluder.shape != shape or not np.isfinite(occluder).all() or np.any((occluder < 0) | (occluder > 1)):
                raise DomainError("TRUE3D_OCCLUDER_MASK_INVALID", "Occluder mask must be a finite full-frame plane in 0..1", 422)
            alpha *= 1.0 - occluder
        if not np.any(alpha > 0):
            raise DomainError("TRUE3D_SEMANTIC_MASK_EMPTY", "Hair-excluding semantic skin mask is empty", 422)

        allowed = (coverage > 0) & (skin > 0) & (hair < 0.5)
        if packet.occluder is not None:
            allowed &= occluder < 0.5
        alpha = anatomical_alpha(alpha, allowed, feather_px=self.options.feather_px,
                                 erosion_px=self.options.erosion_px, dilation_px=self.options.dilation_px)
        if not np.any(alpha > 0):
            raise DomainError("TRUE3D_SEMANTIC_MASK_EMPTY", "Anatomical feather removed all visible face pixels", 422)
        self.last_alpha = alpha
        skin_only = np.asarray(packet.mask_diagnostics.get("_skin_only_mask", skin), np.float32)
        if (skin_only.shape != shape or not np.isfinite(skin_only).all()
                or np.any((skin_only < 0) | (skin_only > 1))):
            raise DomainError("TRUE3D_SEMANTIC_MASK_INVALID", "Skin-only LAB support must be a finite full-frame plane in 0..1", 422)
        skin_only = np.minimum(skin_only, alpha)
        # This is only a computational ROI. The mesh has already been projected
        # in full-frame perspective coordinates; no target image is warped.
        ys, xs = np.nonzero(alpha > 0)
        margin = 2 ** (self.options.pyramid_levels + 1)
        x0, x1 = max(0, int(xs.min()) - margin), min(shape[1], int(xs.max()) + margin + 1)
        y0, y1 = max(0, int(ys.min()) - margin), min(shape[0], int(ys.max()) + margin + 1)
        roi = np.s_[y0:y1, x0:x1]
        local_skin = skin_only[roi]
        corrected, photometry = local_lab_skin_match(rendered[roi], frame[roi], local_skin)
        detailed, detail = finish_skin_detail(rendered[roi], corrected, frame[roi], local_skin,
                                              seed=int(frame_index) * 1000003 + int(packet.track_id))
        # Undefined raster pixels are scene pixels before any coarse pyramid
        # filtering. Zero-alpha pixels are restored again after reconstruction.
        source = np.where((alpha[roi] > 0)[..., None], detailed, frame[roi])
        blended = ops.laplacian_pyramid_blend(source, frame[roi], alpha[roi], self.options.pyramid_levels)
        blended[alpha[roi] == 0] = frame[roi][alpha[roi] == 0]
        output = frame.copy()
        output[roi] = blended
        report = FaceReport(
            packet.track_id,
            geometry={"mode": "full_frame_3d_raster", "covered_pixels": int(np.count_nonzero(coverage > 0.5))},
            relight=photometry,
            restore={**detail, "neural_restoration_space": "canonical_uv_only",
                     "requested_uv_restoration_weight": self.options.restoration_weight},
            blend={"levels": self.options.pyramid_levels, "alpha_pixels": int(np.count_nonzero(alpha > 0.5)),
                   "hair_excluded": True, "occluder_applied": packet.occluder is not None,
                   "mask_method": "bounded_multiscale_signed_distance",
                   "analysis": {key: value for key, value in packet.mask_diagnostics.items() if not key.startswith("_")}},
            pipeline_stages=(*stages, FRAME_PIPELINE_STAGES[4]),
        )
        return output, report


def _check_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise DomainError("TRUE3D_FRAME_INVALID", "Frame must be a uint8 H x W x 3 BGR array", 422)
