"""Frame-space fusion for the robust compositor path.

The occluder matte is estimated on a small face crop; its edges are only as sharp as that crop. This
step warps the matte to the frame, snaps it to the real edges of the full-resolution target with a
guided filter, decontaminates the swapped colour at the feathered boundary, matches defocus/motion
blur and re-uses the target's own fine texture, and only then blends. It reuses ``fuse_warped``
(the same blend implementation as the standard path), so feather/multi-band/Poisson all still work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from roop_3d.compositor.boundary_fusion import BlendMode, fuse_warped
from roop_3d.compositor.matte import decontaminate_edges, refine_occluder
from roop_3d.compositor.quality import match_sharpness, transfer_fine_texture
from roop_3d.core.errors import DomainError


@dataclass(frozen=True)
class RobustFusionOptions:
    refine_matte: bool = True
    refine_radius_fraction: float = 0.012     # of the larger frame-ROI side
    matte_contrast: float = 1.5
    decontaminate: bool = True
    match_sharpness: bool = True
    texture_strength: float = 0.7
    linear_light: bool = True
    shutter: float = 0.5


@dataclass(frozen=True)
class FusionReport:
    roi: tuple[int, int, int, int] | None = None
    sharpness: dict[str, float] = field(default_factory=dict)
    texture_gain: float = 0.0
    occluded_fraction: float = 0.0


def _roi(matrix: np.ndarray, crop_shape: tuple[int, int], frame_shape: tuple[int, int], margin: int):
    h, w = crop_shape
    corners = np.array(((0, 0), (w, 0), (w, h), (0, h)), np.float32)
    mapped = cv2.transform(corners[None], matrix)[0]
    x0 = max(0, int(np.floor(mapped[:, 0].min())) - margin)
    y0 = max(0, int(np.floor(mapped[:, 1].min())) - margin)
    x1 = min(frame_shape[1], int(np.ceil(mapped[:, 0].max())) + margin)
    y1 = min(frame_shape[0], int(np.ceil(mapped[:, 1].max())) + margin)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def fuse_robust(frame: np.ndarray, crop: np.ndarray, inverse: np.ndarray, face_alpha: np.ndarray,
                occluder: np.ndarray | None, skin: np.ndarray | None, *, frame_motion: tuple[float, float] = (0.0, 0.0),
                mode: BlendMode | str = BlendMode.FEATHER, levels: int = 4,
                options: RobustFusionOptions = RobustFusionOptions()) -> tuple[np.ndarray, FusionReport]:
    target = np.asarray(frame)
    image = np.asarray(crop)
    matrix = np.asarray(inverse, np.float32)
    if (target.ndim != 3 or target.shape[2] != 3 or target.dtype != np.uint8 or image.dtype != np.uint8
            or image.ndim != 3 or matrix.shape != (2, 3) or not np.isfinite(matrix).all()
            or abs(float(np.linalg.det(matrix[:, :2]))) < 1e-8):
        raise DomainError("COMPOSITOR_WARP_INVALID", "Robust fusion inputs are invalid", 422)
    alpha_crop = np.asarray(face_alpha, np.float32)
    if alpha_crop.shape != image.shape[:2] or not np.isfinite(alpha_crop).all():
        raise DomainError("COMPOSITOR_MASK_INVALID", "Face alpha must match the replacement crop", 422)
    for name, value in (("occluder", occluder), ("skin", skin)):
        if value is not None and np.asarray(value).shape != image.shape[:2]:
            raise DomainError("COMPOSITOR_MASK_INVALID", f"{name} must match the replacement crop", 422)
    box = _roi(matrix, image.shape[:2], target.shape[:2], margin=6)
    if box is None or not alpha_crop.any():
        return target.copy(), FusionReport()
    x0, y0, x1, y1 = box
    shifted = matrix.copy()
    shifted[:, 2] -= (x0, y0)
    size = (x1 - x0, y1 - y0)

    def warp(data, flag=cv2.INTER_LINEAR):
        return cv2.warpAffine(data, shifted, size, flags=flag, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    source = warp(image)
    alpha = np.clip(warp(np.clip(alpha_crop, 0, 1)), 0, 1)
    footprint = warp(np.ones(image.shape[:2], np.float32))
    target_roi = target[y0:y1, x0:x1]
    # Decontamination is defined on the GEOMETRIC face edge only. Using the post-occlusion alpha would treat every
    # partially-occluded interior pixel as "edge", flatten its texture and drag its tone toward the few opaque pixels.
    face_edge_alpha = alpha.copy()
    occluded_fraction = 0.0
    if occluder is not None:
        front = np.clip(warp(np.clip(np.asarray(occluder, np.float32), 0, 1)), 0, 1)
        if options.refine_matte:
            radius = max(2, int(round(max(size) * options.refine_radius_fraction)))
            front = refine_occluder(front, target_roi, radius=radius, contrast=options.matte_contrast)
        base_sum = float(alpha.sum())
        alpha = alpha * (1.0 - front)
        occluded_fraction = float(1.0 - alpha.sum() / base_sum) if base_sum > 1e-6 else 0.0
    if options.decontaminate:
        source = decontaminate_edges(source, face_edge_alpha)
    report_sharp: dict[str, float] = {}
    if options.match_sharpness:
        # ``frame_motion`` is the face's displacement in FRAME pixels per frame (the aligned crop itself is
        # motion-stabilised, so its content flow says nothing about the blur the camera recorded on the face).
        source, report_sharp = match_sharpness(source, target_roi, alpha, footprint, motion_px=frame_motion,
                                               shutter=options.shutter)
    gain = 0.0
    if options.texture_strength > 0 and skin is not None and np.asarray(skin).any():
        skin_roi = np.clip(warp(np.clip(np.asarray(skin, np.float32), 0, 1)), 0, 1)
        source, gain = transfer_fine_texture(source, target_roi, alpha, skin_roi, footprint, strength=options.texture_strength)
    fused = fuse_warped(target_roi, source, alpha, mode=mode, levels=levels, linear_light=options.linear_light)
    output = target.copy()
    output[y0:y1, x0:x1] = fused
    return output, FusionReport(box, report_sharp, gain, occluded_fraction)
