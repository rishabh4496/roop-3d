"""Forehead / hairline boundary offset for the face matte.

The matte lives in the aligned face crop, which is upright: the top rows are the forehead and hairline. ``offset_hairline`` grows
(``offset_px > 0``) or pulls in (``< 0``) the matte boundary only inside the forehead band, fading to no change below it, so the jaw,
cheeks and mouth edges keep the global ``erosion_px`` behaviour. The offset is in pixels of the 512-class crop and is scaled with the
working resolution by the caller.
"""
from __future__ import annotations

import cv2
import numpy as np

#: Fraction of the crop height, measured from the top, that the offset acts on (it fades out to nothing at the bottom of the band).
FOREHEAD_BAND = 0.42
MAX_OFFSET_PX = 32.0


def hairline_weight(height: int, band: float = FOREHEAD_BAND) -> np.ndarray:
    """A (height, 1) float32 column: 1 at the top row, smoothly 0 from ``band * height`` downward."""
    if height <= 0:
        raise ValueError("height must be positive")
    rows = (np.arange(height, dtype=np.float32) + 0.5) / (band * height)
    t = np.clip(1.0 - rows, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).reshape(height, 1)


def offset_hairline(alpha: np.ndarray, offset_px: float, scale: float = 1.0) -> np.ndarray:
    """The matte with its forehead boundary moved by ``offset_px`` (+ grows, - shrinks). ``alpha`` is returned unchanged for 0."""
    if not np.isfinite(offset_px) or abs(offset_px) > MAX_OFFSET_PX:
        raise ValueError(f"hairline offset must be finite and within +-{MAX_OFFSET_PX:g} px")
    if not offset_px:
        return alpha
    alpha = np.asarray(alpha, np.float32)
    if alpha.ndim != 2:
        raise ValueError("the matte must be a single-channel plane")
    radius = max(1, round(abs(offset_px) * scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    moved = cv2.dilate(alpha, kernel) if offset_px > 0 else cv2.erode(alpha, kernel)
    weight = hairline_weight(alpha.shape[0])
    return (alpha * (1.0 - weight) + moved * weight).astype(np.float32)


def refine_hairline_matte(
    alpha: np.ndarray,
    guide_bgr: np.ndarray,
    *,
    radius: int = 8,
    eps: float = 1e-4,
    band: float = FOREHEAD_BAND,
    smooth_feather_px: float = 5.0,
) -> np.ndarray:
    """Edge-Preserving Fast Guided Filter for Hairline & Bangs (Zero Smudge Blur).

    Replaces standard cv2.GaussianBlur mask feathering along the upper forehead/hairline with a
    Fast Guided Filter (guidedFilter) using the target RGB frame's luminance channel as the
    guidance image (r=8, eps=1e-4).

    This locks the alpha matte directly to individual hair strands and bangs crossing the forehead—
    eliminating fuzzy halo smudging around hair while keeping skin-to-skin transitions along the
    jaw and temples smooth.
    """
    from roop_3d.compositor.matte import fast_guided_filter

    m = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    image = np.asarray(guide_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != m.shape:
        raise ValueError("Guide image must match alpha shape")

    # Guidance image: target frame's luminance channel in [0, 1]
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Smooth base feathering
    if smooth_feather_px > 0:
        rad = max(1, int(np.ceil(smooth_feather_px * 3)))
        smooth_m = cv2.GaussianBlur(m, (2 * rad + 1, 2 * rad + 1), sigmaX=smooth_feather_px)
    else:
        smooth_m = m

    # Fast Guided Filter along hairline and bangs on the feathered boundary
    guided_m = fast_guided_filter(luma, smooth_m, radius=radius, eps=eps)

    # Protect solid interior and background: core must stay 1.0, background 0.0
    ksize = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bin_m = (m > 0.5).astype(np.uint8)
    core = cv2.erode(bin_m, kernel).astype(np.float32)
    bg = (cv2.dilate(bin_m, kernel) == 0).astype(np.float32)
    uncertain = np.clip(1.0 - core - bg, 0.0, 1.0)
    boundary_guided = core * 1.0 + uncertain * guided_m

    # Hairline region weight: 1.0 at upper forehead / hairline, smoothly 0.0 below band
    hw = hairline_weight(m.shape[0], band=band)

    # Forehead/hairline uses boundary_guided, jaw and temples use smooth_m
    result = smooth_m * (1.0 - hw) + boundary_guided * hw
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# Re-export boundary ribbon matting for clean access from hairline/compositor
from roop_3d.compositor.matte import (
    build_boundary_ribbon_trimap,
    refine_boundary_ribbon_matte,
)
