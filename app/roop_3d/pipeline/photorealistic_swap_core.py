"""Composite a rasterized, shaded full-head mesh through a semantic matte.

The compositor accepts only render layers carrying the ordered true-3D stage
trace. It performs frame placement, scene relighting, detail finishing, and a
soft Laplacian edge blend; it has no image-warp or silhouette-replacement path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.true3d_guard import FRAME_PIPELINE_STAGES, require_true3d_frame_pipeline
from roop_3d.pipeline import image_ops as ops

STEP_NAMES = ("geometry", "rectify", "relight", "restore", "blend")




@dataclass(frozen=True)
class SwapCoreOptions:
    feather_px: float = 5.0
    pyramid_levels: int = 4

    def __post_init__(self) -> None:
        if not np.isfinite(self.feather_px) or not 0 <= self.feather_px <= 64:
            raise ValueError("feather_px must be within 0..64")
        if not 1 <= self.pyramid_levels <= 6:
            raise ValueError("pyramid_levels must be within 1..6")


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

    def process_face(self, frame: np.ndarray, packet: FacePacket, *, frame_index: int = 0) -> tuple[np.ndarray, FaceReport]:
        del frame_index
        _check_frame(frame)
        stages = require_true3d_frame_pipeline(packet.pipeline_stages, through=FRAME_PIPELINE_STAGES[3])
        shape = frame.shape[:2]
        rendered = np.asarray(packet.rendered_frame)
        if rendered.dtype != np.uint8 or rendered.shape != frame.shape:
            raise DomainError("TRUE3D_RENDER_INVALID", "The 3D renderer must return a full-frame uint8 BGR raster", 422)
        normals = np.asarray(packet.normals, np.float32)
        if normals.shape != (*shape, 3) or not np.isfinite(normals).all():
            raise DomainError("TRUE3D_MESH_NORMALS_REQUIRED", "Full-frame surface normals are required", 422)

        planes = {}
        for name in ("coverage", "skin_mask", "hair_mask"):
            plane = np.asarray(getattr(packet, name), np.float32)
            if plane.shape != shape or not np.isfinite(plane).all() or np.any((plane < 0) | (plane > 1)):
                raise DomainError("TRUE3D_SEMANTIC_MASK_INVALID", f"{name} must be a finite full-frame plane in 0..1", 422)
            planes[name] = plane
        coverage, skin, hair = (planes[name] for name in ("coverage", "skin_mask", "hair_mask"))
        alpha = coverage * skin * (1.0 - hair)
        if packet.occluder is not None:
            occluder = np.asarray(packet.occluder, np.float32)
            if occluder.shape != shape or not np.isfinite(occluder).all() or np.any((occluder < 0) | (occluder > 1)):
                raise DomainError("TRUE3D_OCCLUDER_MASK_INVALID", "Occluder mask must be a finite full-frame plane in 0..1", 422)
            alpha *= 1.0 - occluder
        if not np.any(alpha > 0):
            raise DomainError("TRUE3D_SEMANTIC_MASK_EMPTY", "Hair-excluding semantic skin mask is empty", 422)

        if self.options.feather_px:
            alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=self.options.feather_px)
            alpha = np.minimum(alpha, coverage) * skin * (1.0 - hair)
            if packet.occluder is not None:
                alpha *= 1.0 - occluder
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        output = ops.laplacian_pyramid_blend(rendered, frame, alpha, self.options.pyramid_levels)
        report = FaceReport(
            packet.track_id,
            geometry={"mode": "full_frame_3d_raster", "covered_pixels": int(np.count_nonzero(coverage > 0.5))},
            blend={"levels": self.options.pyramid_levels, "alpha_pixels": int(np.count_nonzero(alpha > 0.5)),
                   "hair_excluded": True, "occluder_applied": packet.occluder is not None,
                   "analysis": packet.mask_diagnostics},
            pipeline_stages=(*stages, FRAME_PIPELINE_STAGES[4]),
        )
        return output, report


def _check_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise DomainError("TRUE3D_FRAME_INVALID", "Frame must be a uint8 H x W x 3 BGR array", 422)
