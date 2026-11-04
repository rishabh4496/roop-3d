"""Finite-safe crop warping and feather, multi-band or Poisson boundary fusion."""
from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from roop_3d.core.errors import DomainError


class BlendMode(str, Enum):
    FEATHER = "feather"
    MULTI_BAND = "multi_band"
    POISSON = "poisson"


def warp_crop(crop: np.ndarray, inverse: np.ndarray, frame_shape: tuple[int, ...],
              mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Warp crop coordinates into the full frame with a crop-to-frame Minv matrix."""
    image = np.asarray(crop)
    matrix = np.asarray(inverse, dtype=np.float32)
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
    warped_image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_alpha = cv2.warpAffine(np.clip(alpha, 0, 1), matrix, (width, height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_image, np.clip(np.nan_to_num(warped_alpha), 0.0, 1.0).astype(np.float32)


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
    gauss_t, gauss_s, gauss_m = [target_f], [source_f], [alpha.astype(np.float32)]
    for _ in range(count):
        if min(gauss_t[-1].shape[:2]) < 4:
            break
        gauss_t.append(cv2.pyrDown(gauss_t[-1]))
        gauss_s.append(cv2.pyrDown(gauss_s[-1]))
        gauss_m.append(cv2.pyrDown(gauss_m[-1]))
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
              mode: BlendMode | str = BlendMode.FEATHER, levels: int = 4,
              linear_light: bool = False, transfer: str = "srgb") -> np.ndarray:
    """Composite a transformed crop and guarantee byte-identical pixels outside its dilated ROI."""
    target = np.asarray(frame)
    if target.ndim != 3 or target.shape[2] != 3 or target.dtype != np.uint8:
        raise DomainError("COMPOSITOR_FRAME_INVALID", "Target frame must be uint8 BGR", 422)
    try:
        blend_mode = BlendMode(mode)
    except ValueError:
        raise ValueError("Unknown boundary fusion mode") from None
    source, alpha = warp_crop(crop, inverse, target.shape, mask)
    return fuse_warped(target, source, alpha, mode=blend_mode, levels=levels, linear_light=linear_light,
                       transfer=transfer)


def fuse_warped(target: np.ndarray, source: np.ndarray, alpha: np.ndarray, *,
                mode: BlendMode | str = BlendMode.FEATHER, levels: int = 4, linear_light: bool = False,
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
    margin = max(2, (2 ** max(1, min(levels, 8))) if blend_mode == BlendMode.MULTI_BAND else 2)
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
    # Pyramid reconstruction and Poisson gradients can spread beyond nonzero alpha.
    # Clamp their write footprint to the explicitly dilated blend support.
    radius = max(1, margin)
    dilate_kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    allowed = cv2.dilate((alpha > 1e-5).astype(np.uint8), dilate_kernel)[y0:y1, x0:x1] > 0
    mixed[~allowed] = target_roi[~allowed]
    output = target.copy()
    output[y0:y1, x0:x1] = np.clip(np.nan_to_num(mixed, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).round().astype(np.uint8)
    return output
