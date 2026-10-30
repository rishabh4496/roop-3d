"""Optional, clamped spherical-harmonic luminance relighting."""
from __future__ import annotations

import numpy as np

from roop_3d.core.errors import DomainError


def _basis(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    radius = x * x + y * y
    z = np.sqrt(np.maximum(1.0 - radius, 0.0))
    return np.stack((np.ones_like(x), x, y, z, x * y, x * z, y * z, x * x - y * y, 3 * z * z - 1), axis=-1).astype(np.float32)


def relight_spherical_harmonics(
    swapped_bgr: np.ndarray,
    target_bgr: np.ndarray,
    skin_mask: np.ndarray,
    face_mask: np.ndarray,
    *,
    gain_range: tuple[float, float] = (0.72, 1.35),
    minimum_samples: int = 96,
) -> np.ndarray:
    """Match low-order directional luminance; fit coefficients on skin pixels only."""
    source = np.asarray(swapped_bgr)
    target = np.asarray(target_bgr)
    skin = np.asarray(skin_mask, dtype=np.float32)
    face = np.asarray(face_mask, dtype=np.float32)
    if (source.ndim != 3 or source.shape[2] != 3 or target.shape != source.shape
            or skin.shape != source.shape[:2] or face.shape != source.shape[:2]
            or not all(np.isfinite(value).all() for value in (source, target, skin, face))):
        raise DomainError("LIGHTING_INPUT_INVALID", "Relighting inputs have incompatible shapes", 422)
    low, high = gain_range
    if not (0 < low <= high and np.isfinite((low, high)).all()):
        raise ValueError("gain_range must be finite, positive and ordered")
    samples = (skin > 0.5) & (face > 0.05)
    if np.count_nonzero(samples) < minimum_samples:
        return source.copy()
    basis = _basis(source.shape[0], source.shape[1])
    source_rgb = source[..., ::-1].astype(np.float32) / 255.0
    target_rgb = target[..., ::-1].astype(np.float32) / 255.0
    source_luma = source_rgb @ np.array((0.2126, 0.7152, 0.0722), np.float32)
    target_luma = target_rgb @ np.array((0.2126, 0.7152, 0.0722), np.float32)
    source_coeff, *_ = np.linalg.lstsq(basis[samples], source_luma[samples], rcond=1e-4)
    target_coeff, *_ = np.linalg.lstsq(basis[samples], target_luma[samples], rcond=1e-4)
    source_shade = np.maximum(basis @ source_coeff, 0.05)
    target_shade = np.maximum(basis @ target_coeff, 0.05)
    gain = np.clip(target_shade / source_shade, low, high)
    output = source.astype(np.float32, copy=True)
    active = (skin > 0.05) & (face > 0)
    output[active] *= gain[active, None]
    return np.clip(np.nan_to_num(output), 0, 255).astype(np.uint8)
