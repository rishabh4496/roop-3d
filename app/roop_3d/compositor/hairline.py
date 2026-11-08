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
