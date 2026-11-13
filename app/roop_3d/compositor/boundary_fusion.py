"""Finite-safe crop fusion: four-band Laplacian blending and inward anatomical SDF feathers.

Anatomy comes from the caller's measured landmarks, semantic face mask or rasterized
coverage, never from an inferred unseen silhouette. Pyramid reconstruction cannot
write into a zero-alpha foreground veto or outside the finite blend support.
"""
from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
from roop_3d.core.errors import DomainError



class BlendMode(str, Enum):
    FEATHER = "feather"
    MULTI_BAND = "multi_band"
    POISSON = "poisson"
    # Laplacian-pyramid split that keeps the target's fine detail and Lab-matches the swap's base (compositor/frequency_blend.py)
    FREQUENCY = "frequency_separated"


def warp_crop(crop: np.ndarray, inverse: np.ndarray, frame_shape: tuple[int, ...],
              mask: np.ndarray | None = None, *, interpolation: str = "lanczos4") -> tuple[np.ndarray, np.ndarray]:
    """Warp crop coordinates into the full frame with sub-pixel image resampling and a linear matte."""
    image = np.asarray(crop)
    matrix = np.asarray(inverse, dtype=np.float64)
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if (image.ndim != 3 or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0
            or matrix.shape != (2, 3) or not np.isfinite(matrix).all()
            or height <= 0 or width <= 0 or not np.isfinite(image).all()):
        raise DomainError("COMPOSITOR_WARP_INVALID", "Crop, inverse matrix or frame dimensions are invalid", 422)
    determinant = float(np.linalg.det(matrix[:, :2]))
    if abs(determinant) < 1e-8:
        raise DomainError("COMPOSITOR_WARP_INVALID", "Crop inverse matrix is singular", 422)
    alpha = np.ones(image.shape[:2], np.float32) if mask is None else np.asarray(mask, dtype=np.float32)
    if alpha.shape != image.shape[:2] or not np.isfinite(alpha).all():
        raise DomainError("COMPOSITOR_MASK_INVALID", "Crop alpha must match the replacement image", 422)
    from roop_3d.pipeline.subpixel import warp_affine as warp_subpixel_affine

    warped_image = warp_subpixel_affine(image, matrix, (width, height), interpolation=interpolation,
                                        border="constant", border_value=(0, 0, 0))
    warped_alpha = cv2.warpAffine(np.clip(alpha, 0, 1), matrix.astype(np.float32), (width, height),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_image, np.clip(np.nan_to_num(warped_alpha), 0.0, 1.0).astype(np.float32)




def feather_anatomical_mask(mask: np.ndarray, feather_px: float) -> np.ndarray:
    """Smoothstep an observed contour's signed distance inward without growing its support.

    Soft probabilities remain an upper bound. Without a confident contour (> 0.5)
    there is no anatomical edge to reconstruct, so the supplied matte is retained.
    Zero padding treats a crop edge as an actual limit rather than infinite skin.
    """
    value = np.asarray(mask, np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise DomainError("COMPOSITOR_MASK_INVALID", "Anatomical alpha must be a finite 2-D mask", 422)
    if not np.isfinite(feather_px) or feather_px < 0:
        raise ValueError("Anatomical feather must be finite and nonnegative")
    value = np.clip(value, 0.0, 1.0)
    support = value > 0.5
    if feather_px == 0 or not support.any():
        return value
    padded = np.pad(support.astype(np.uint8), 1, mode="constant")
    inside = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - padded, cv2.DIST_L2, 5)
    signed_distance = (inside - outside)[1:-1, 1:-1]
    ramp = np.clip(signed_distance / float(feather_px), 0.0, 1.0)
    ramp *= ramp * (3.0 - 2.0 * ramp)
    return np.minimum(value, ramp).astype(np.float32)


def _bounds(alpha: np.ndarray, margin: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(alpha > 1e-5)
    if len(xs) == 0:
        return None
    height, width = alpha.shape
    return (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(width, int(xs.max()) + margin + 1), min(height, int(ys.max()) + margin + 1))


def _pyramid_blend(target: np.ndarray, source: np.ndarray, alpha: np.ndarray, levels: int) -> np.ndarray:
    max_levels = max(1, int(np.floor(np.log2(max(2, min(target.shape[:2]))))) - 1)
    count = max(1, min(int(levels), max_levels))
    target_f, source_f = target.astype(np.float32), source.astype(np.float32)
    # Build the source's Gaussian levels from observed samples only. Filling
    # hidden/warped-out pixels with the target bakes an artificial target-colour
    # edge into the source Laplacian bands, including their finest detail.
    source_weight = np.where(alpha > 1e-5, alpha, 0).astype(np.float32)
    numerator = source_f * source_weight[..., None]
    gauss_t, gauss_s, gauss_m = [target_f], [source_f], [source_weight]
    # ``levels`` counts the base Gaussian too: four levels mean three
    # reductions, not a fifth/coarser band bleeding target light into the face.
    for _ in range(count - 1):
        if min(gauss_t[-1].shape[:2]) < 4:
            break
        gauss_t.append(cv2.pyrDown(gauss_t[-1]))
        gauss_m.append(cv2.pyrDown(gauss_m[-1]))
        numerator = cv2.pyrDown(numerator)
        gauss_s.append(numerator / np.maximum(gauss_m[-1], 1e-8)[..., None])
    lap_t, lap_s = [], []
    for stack, laps in ((gauss_t, lap_t), (gauss_s, lap_s)):
        for index in range(len(stack) - 1):
            up = cv2.pyrUp(stack[index + 1], dstsize=(stack[index].shape[1], stack[index].shape[0]))
            laps.append(stack[index] - up)
        laps.append(stack[-1])
    mixed = []
    for t_level, s_level, mask_level in zip(lap_t, lap_s, gauss_m):
        weight = np.clip(mask_level, 0, 1)[..., None]
        mixed.append(s_level * weight + t_level * (1 - weight))
    result = mixed[-1]
    for index in range(len(mixed) - 2, -1, -1):
        result = cv2.pyrUp(result, dstsize=(mixed[index].shape[1], mixed[index].shape[0])) + mixed[index]
    return result


def fuse_crop(frame: np.ndarray, crop: np.ndarray, inverse: np.ndarray, mask: np.ndarray, *,
              mode: BlendMode | str = BlendMode.MULTI_BAND, levels: int = 4,
              linear_light: bool = False, transfer: str = "srgb", interpolation: str = "lanczos4") -> np.ndarray:
    """Composite a transformed crop and guarantee byte-identical pixels outside its dilated ROI."""
    target = np.asarray(frame)
    if target.ndim != 3 or target.shape[2] != 3 or target.dtype != np.uint8:
        raise DomainError("COMPOSITOR_FRAME_INVALID", "Target frame must be uint8 BGR", 422)
    try:
        blend_mode = BlendMode(mode)
    except ValueError:
        raise ValueError("Unknown boundary fusion mode") from None
    source, alpha = warp_crop(crop, inverse, target.shape, mask, interpolation=interpolation)
    return fuse_warped(target, source, alpha, mode=blend_mode, levels=levels, linear_light=linear_light,
                       transfer=transfer)


def fuse_warped(target: np.ndarray, source: np.ndarray, alpha: np.ndarray, *,
                mode: BlendMode | str = BlendMode.MULTI_BAND, levels: int = 4, linear_light: bool = False,
                transfer: str = "srgb") -> np.ndarray:
    """Blend an already frame-space ``source``/``alpha`` pair into ``target`` (uint8 BGR).

    ``linear_light`` performs the blend arithmetic on linear light instead of on encoded code values, using
    the named display transfer (``color.transfer.BLEND_TRANSFERS``; the default is sRGB, matching earlier
    behaviour). It applies to every mode: feather mixes in linear light; multi-band builds and merges its
    pyramids in linear light; Poisson solves its gradient field on the encoded image (gradient-domain
    cloning is perceptual by construction) and then performs the final alpha mix in linear light.
    Pixels outside the dilated alpha support are returned byte-identical.
    """
    from roop_3d.compositor.quality import encode_linear, linear_lut, mix_linear_light
    blend_mode = BlendMode(mode)
    if blend_mode == BlendMode.FREQUENCY and linear_light:
        raise ValueError("Frequency-separated blending works on encoded values and cannot be combined with linear-light blending")
    margin = max(2, (2 ** max(1, min(levels, 8))) if blend_mode in (BlendMode.MULTI_BAND, BlendMode.FREQUENCY) else 2)
    bounds = _bounds(alpha, margin)
    if bounds is None:
        return target.copy()
    x0, y0, x1, y1 = bounds
    target_roi = target[y0:y1, x0:x1]
    source_roi = source[y0:y1, x0:x1]
    alpha_roi = alpha[y0:y1, x0:x1]

    def mix(foreground: np.ndarray) -> np.ndarray:
        if linear_light:
            return mix_linear_light(target_roi, foreground, alpha_roi, transfer)
        return target_roi.astype(np.float32) * (1 - alpha_roi[..., None]) + foreground.astype(np.float32) * alpha_roi[..., None]

    if blend_mode == BlendMode.MULTI_BAND:
        if linear_light:
            lut = linear_lut(transfer)
            clipped = np.clip(np.rint(source_roi), 0, 255).astype(np.uint8)
            mixed = encode_linear(transfer, _pyramid_blend(lut[target_roi], lut[clipped], alpha_roi, levels))
        else:
            mixed = _pyramid_blend(target_roi, source_roi, alpha_roi, levels)
    elif blend_mode == BlendMode.FREQUENCY:
        from roop_3d.compositor.frequency_blend import FrequencyBlendOptions, frequency_blend
        # The compositor already performs skin/nose-only colour transfer. Do not
        # run a second, face-wide LAB estimate over eyes, lips or occluders here.
        mixed = frequency_blend(target_roi, source_roi, alpha_roi,
                                FrequencyBlendOptions(levels=max(2, min(levels, 8)), color_strength=0.0))
    elif blend_mode == BlendMode.POISSON:
        binary = (alpha_roi > 0.05).astype(np.uint8) * 255
        if int(binary.sum()) == 0 or min(binary.shape) < 2:
            mixed = mix(source_roi)
        else:
            try:
                center = ((x1 - x0) // 2, (y1 - y0) // 2)
                mixed = mix(cv2.seamlessClone(source_roi, target_roi, binary, center, cv2.NORMAL_CLONE))
            except cv2.error:
                mixed = mix(source_roi)
    else:
        mixed = mix(source_roi)
    # Preserve explicit/semantic foreground holes even inside the pyramid ROI.
    allowed = alpha_roi > 1e-5
    mixed[~allowed] = target_roi[~allowed]
    output = target.copy()
    output[y0:y1, x0:x1] = np.clip(np.nan_to_num(mixed, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).round().astype(np.uint8)
    return output
