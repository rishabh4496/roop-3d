"""Fidelity helpers for the robust compositor: illumination, sharpness/motion blur, fine texture.

Everything here is deterministic image processing on the real target pixels; nothing synthesises
detail that is absent from the target. Fine texture (pores, grain) is *taken from the target frame
itself*, aligned by construction because the swapped face is composited back onto that same frame.

Frame-space functions take ROIs cut from the warped replacement (``source``), the original target
(``target``) and the warped ``footprint`` (1 where the crop actually exists, so blurs never drag the
black warp border into the face).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from roop_3d.core.errors import DomainError

_LUMA = np.array((0.114, 0.587, 0.299), np.float32)   # BGR order


def _gray01(image: np.ndarray) -> np.ndarray:
    return (np.asarray(image, np.float32) @ _LUMA) / 255.0


def _normalised_blur(image: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that only averages pixels where ``weight`` > 0 (works for HxW and HxWxC)."""
    radius = max(1, int(math.ceil(sigma * 3)))
    ksize = (2 * radius + 1, 2 * radius + 1)
    w = np.clip(weight.astype(np.float32), 0.0, 1.0)
    denominator = cv2.GaussianBlur(w, ksize, sigma, borderType=cv2.BORDER_REPLICATE)
    data = image.astype(np.float32) * (w[..., None] if image.ndim == 3 else w)
    numerator = cv2.GaussianBlur(data, ksize, sigma, borderType=cv2.BORDER_REPLICATE)
    return numerator / np.maximum(denominator, 1e-4)[..., None] if image.ndim == 3 else numerator / np.maximum(denominator, 1e-4)


def local_illumination_match(source: np.ndarray, target: np.ndarray, alpha: np.ndarray, *, strength: float = 0.6,
                             sigma_fraction: float = 0.16, gain_range: tuple[float, float] = (0.7, 1.4)) -> np.ndarray:
    """Match the *spatial* shading of the target (one-sided light, hair shadow) on lightness only.

    The global brightness is left to the LAB colour step; only the spatially varying part of the
    target/source lightness ratio is applied, so the two stages do not double-count.
    """
    image, reference = np.asarray(source), np.asarray(target)
    weight = np.asarray(alpha, np.float32)
    if (image.ndim != 3 or image.shape[2] != 3 or reference.shape != image.shape or weight.shape != image.shape[:2]
            or not np.isfinite(weight).all()):
        raise DomainError("QUALITY_INPUT_INVALID", "Illumination inputs must have matching shapes", 422)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be within 0..1")
    mask = np.clip(weight, 0.0, 1.0)
    if strength == 0.0 or int((mask > 0.3).sum()) < 64:
        return image.copy()
    sigma = max(2.0, sigma_fraction * min(image.shape[:2]))
    lab_s = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lab_t = cv2.cvtColor(reference.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    low_s = _normalised_blur(lab_s[..., 0], mask, sigma) + 5.0
    low_t = _normalised_blur(lab_t[..., 0], mask, sigma) + 5.0
    log_gain = np.log(np.clip(low_t / low_s, 0.05, 20.0))
    mean = float((log_gain * mask).sum() / max(float(mask.sum()), 1e-6))
    gain = np.exp(np.clip(log_gain - mean, math.log(gain_range[0]), math.log(gain_range[1])))
    lab_s[..., 0] = np.clip(lab_s[..., 0] * (1.0 + (gain - 1.0) * strength * np.clip(mask * 2.0, 0.0, 1.0)), 0.0, 100.0)
    out = cv2.cvtColor(lab_s, cv2.COLOR_LAB2BGR) * 255.0
    return np.clip(np.nan_to_num(out), 0, 255).round().astype(np.uint8)


def shading_gain_match(source: np.ndarray, target: np.ndarray, alpha: np.ndarray, *, strength: float = 1.0,
                       sigma_fraction: float = 0.03, reach_fraction: float = 24 / 512,
                       gain_range: tuple[float, float] = (0.6, 1.5), work_side: int = 192) -> np.ndarray:
    """Give a synthesised face the target's low-frequency shading, continued across the mask boundary.

    A neural/restored face carries the lighting of its training data: flat, or lit from the wrong side, so it reads as a
    2D sticker and its rim is brighter or darker than the head around it. The target's own lightness, low-passed, is the
    lighting to follow. Both images are low-passed over the same domain, the mask DILATED outward by ``reach_fraction`` of
    the crop (so the gain at the rim is set by what surrounds the face, not only by what is inside it, and the
    transition continues smoothly under the feather), and their ratio is a per-pixel gain on lightness only. The mean gain
    inside the mask is divided out (a mean, so brightness is not shifted): global brightness and colour already belong to the colour-match step, and
    high-frequency detail is untouched because the gain is a low-pass field.
    """
    image, reference = np.asarray(source), np.asarray(target)
    weight = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    if (image.ndim != 3 or image.shape[2] != 3 or image.shape[0] != image.shape[1] or reference.shape != image.shape
            or weight.shape != image.shape[:2] or not np.isfinite(weight).all()):
        raise DomainError("QUALITY_INPUT_INVALID", "Shading inputs must be matching square crops", 422)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be within 0..1")
    inside = weight > 0.5
    if strength == 0.0 or int(inside.sum()) < 64:
        return image.copy()
    side = image.shape[0]
    work = min(work_side, side)
    shrink = lambda plane: cv2.resize(plane, (work, work), interpolation=cv2.INTER_AREA)
    lab_s = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lab_t = cv2.cvtColor(reference.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    small_inside = shrink(inside.astype(np.float32)) > 0.5
    reach = max(1, int(round(reach_fraction * work)))
    domain = cv2.dilate(small_inside.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1)))
    sigma = max(1.5, sigma_fraction * work)
    low_s = _normalised_blur(shrink(lab_s[..., 0]), domain.astype(np.float32), sigma) + 5.0
    low_t = _normalised_blur(shrink(lab_t[..., 0]), domain.astype(np.float32), sigma) + 5.0
    gain = np.clip(low_t / low_s, 0.05, 20.0)
    gain = np.clip(gain / float(gain[small_inside].mean()), gain_range[0], gain_range[1])
    gain = cv2.resize(gain, (side, side), interpolation=cv2.INTER_CUBIC)
    lab_s[..., 0] = np.clip(lab_s[..., 0] * (1.0 + (gain - 1.0) * strength), 0.0, 100.0)
    out = cv2.cvtColor(lab_s, cv2.COLOR_LAB2BGR) * 255.0
    return np.clip(np.nan_to_num(out), 0, 255).round().astype(np.uint8)


def _motion_kernel(dx: float, dy: float, shutter: float, max_length: float) -> np.ndarray | None:
    length = min(math.hypot(dx, dy) * shutter, max_length)
    if length < 1.5:
        return None
    size = int(math.ceil(length)) | 1
    kernel = np.zeros((size, size), np.float32)
    centre = size // 2
    ux, uy = dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)
    cv2.line(kernel, (int(round(centre - ux * length / 2)), int(round(centre - uy * length / 2))),
             (int(round(centre + ux * length / 2)), int(round(centre + uy * length / 2))), 1.0, 1, cv2.LINE_AA)
    total = float(kernel.sum())
    return kernel / total if total > 0 else None


def _gradient_energy(gray: np.ndarray, weight: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    total = float(weight.sum())
    return float(((gx * gx + gy * gy) * weight).sum() / total) if total > 1.0 else 0.0


def match_sharpness(source: np.ndarray, target: np.ndarray, alpha: np.ndarray, footprint: np.ndarray, *,
                    motion_px: tuple[float, float] = (0.0, 0.0), shutter: float = 0.5, max_sigma: float = 2.5,
                    strength: float = 1.0) -> tuple[np.ndarray, dict[str, float]]:
    """Blur the swapped face only as much as the target is blurred (defocus and motion blur).

    Returns the matched source and the applied parameters. A target that is sharper than the source
    is left alone; sharpening is never invented here.
    """
    image, reference = np.asarray(source), np.asarray(target)
    weight = np.clip(np.asarray(alpha, np.float32), 0, 1) * (np.asarray(footprint, np.float32) > 0.5)
    if image.shape != reference.shape or weight.shape != image.shape[:2]:
        raise DomainError("QUALITY_INPUT_INVALID", "Sharpness inputs must have matching shapes", 422)
    info = {"sigma": 0.0, "motion_length": 0.0}
    core = weight > 0.6
    if strength == 0.0 or int(core.sum()) < 64:
        return image.copy(), info
    foot = (np.asarray(footprint, np.float32) > 0.5).astype(np.float32)
    core_f = core.astype(np.float32)
    working = image.astype(np.float32)
    target_energy = _gradient_energy(_gray01(reference), core_f)
    kernel = _motion_kernel(motion_px[0], motion_px[1], shutter, 15.0)
    # Directional blur only when the target really is softer than the swap; sharpening is never invented.
    if kernel is not None and target_energy < _gradient_energy(_gray01(working), core_f):
        radius = kernel.shape[0] // 2
        weight_padded = cv2.copyMakeBorder(foot, radius, radius, radius, radius, cv2.BORDER_REPLICATE)
        blurred_w = cv2.filter2D(weight_padded, -1, kernel)[radius:-radius, radius:-radius]
        data = cv2.copyMakeBorder(working * foot[..., None], radius, radius, radius, radius, cv2.BORDER_REPLICATE)
        working = cv2.filter2D(data, -1, kernel)[radius:-radius, radius:-radius] / np.maximum(blurred_w, 1e-4)[..., None]
        info["motion_length"] = float(math.hypot(*motion_px) * shutter)
    best_sigma, best_error = 0.0, abs(_gradient_energy(_gray01(working), core_f) - target_energy)
    sigma = 0.4
    while sigma <= max_sigma + 1e-6:
        error = abs(_gradient_energy(_gray01(_normalised_blur(working, foot, sigma)), core_f) - target_energy)
        if error < best_error:
            best_sigma, best_error = sigma, error
        sigma += 0.3
    if best_sigma > 0:
        working = _normalised_blur(working, foot, best_sigma)
        info["sigma"] = float(best_sigma)
    if info["sigma"] == 0.0 and info["motion_length"] == 0.0:
        return image.copy(), info
    mixed = image.astype(np.float32) + (working - image.astype(np.float32)) * strength
    return np.clip(mixed, 0, 255).round().astype(np.uint8), info


def transfer_fine_texture(source: np.ndarray, target: np.ndarray, alpha: np.ndarray, skin: np.ndarray,
                          footprint: np.ndarray, *, strength: float = 0.7, fine_sigma: float = 0.8,
                          coarse_sigma: float = 2.2) -> tuple[np.ndarray, float]:
    """Add the target's own fine luminance band (pores, skin grain) where the swap is smoother.

    Applied on skin only (never eyes, brows, lips, teeth) and only up to the deficit between target
    and source fine-band energy, so already-textured source pixels are not doubled up and a
    motion-blurred target (no fine band) contributes nothing. Returns the image and the mean gain.
    """
    image, reference = np.asarray(source), np.asarray(target)
    weight = (np.clip(np.asarray(alpha, np.float32), 0, 1) * np.clip(np.asarray(skin, np.float32), 0, 1)
              * (np.asarray(footprint, np.float32) > 0.5))
    if image.shape != reference.shape or weight.shape != image.shape[:2]:
        raise DomainError("QUALITY_INPUT_INVALID", "Texture inputs must have matching shapes", 422)
    if strength == 0.0 or int((weight > 0.5).sum()) < 64:
        return image.copy(), 0.0
    lum_t, lum_s = _gray01(reference), _gray01(image)
    band_t = cv2.GaussianBlur(lum_t, (0, 0), fine_sigma) - cv2.GaussianBlur(lum_t, (0, 0), coarse_sigma)
    band_s = cv2.GaussianBlur(lum_s, (0, 0), fine_sigma) - cv2.GaussianBlur(lum_s, (0, 0), coarse_sigma)
    energy_t = cv2.GaussianBlur(band_t * band_t, (0, 0), 6.0)
    energy_s = cv2.GaussianBlur(band_s * band_s, (0, 0), 6.0)
    deficit = np.sqrt(np.maximum(energy_t - energy_s, 0.0) / np.maximum(energy_t, 1e-8))
    gain = np.clip(deficit, 0.0, 1.0) * weight * strength
    offset = (band_t * gain * 255.0)[..., None]
    out = np.clip(image.astype(np.float32) + offset, 0, 255).round().astype(np.uint8)
    return out, float(gain[weight > 0.5].mean()) if (weight > 0.5).any() else 0.0


# Display-referred code value <-> linear light, used for physically meaningful edge blending. The decode
# curve is a parameter: material is not always sRGB (BT.709/601/2020 video is graded on a BT.1886 display).
_LINEAR_LUTS: dict[str, np.ndarray] = {}


def linear_lut(transfer: str) -> np.ndarray:
    """256-entry float32 table: 8-bit code -> relative linear light, for one of ``color.transfer.BLEND_TRANSFERS``."""
    if transfer not in _LINEAR_LUTS:
        from roop_3d.color import transfer as curves
        _LINEAR_LUTS[transfer] = curves.lut_to_linear(transfer, 8)
    return _LINEAR_LUTS[transfer]


def encode_linear(transfer: str, linear: np.ndarray) -> np.ndarray:
    """Relative linear light 0..1 -> code value 0..255 (float32, unrounded)."""
    from roop_3d.color import transfer as curves
    return (curves.from_linear(transfer, np.clip(linear, 0.0, 1.0)) * 255.0).astype(np.float32)


def mix_linear_light(target: np.ndarray, source: np.ndarray, alpha: np.ndarray, transfer: str = "srgb") -> np.ndarray:
    """alpha-mix two images in linear light. ``source`` may be float 0..255; it is clipped and rounded to codes."""
    lut = linear_lut(transfer)
    t_lin = lut[target]
    s_lin = lut[np.clip(np.rint(source), 0, 255).astype(np.uint8)]
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    return encode_linear(transfer, t_lin * (1.0 - a) + s_lin * a)


def blend_linear_light(target: np.ndarray, source: np.ndarray, alpha: np.ndarray, transfer: str = "srgb") -> np.ndarray:
    """alpha-blend two uint8 BGR images in linear light; returns float32 code values 0..255 (sRGB by default)."""
    return mix_linear_light(target, source, alpha, transfer)


def inject_target_skin_micro_texture(
    enhanced: np.ndarray,
    target: np.ndarray,
    skin_mask: np.ndarray,
    alpha_mask: np.ndarray | None = None,
    *,
    injection_rate: float = 0.25,
    highpass_sigma: float = 1.5,
) -> np.ndarray:
    """Re-inject 25% of the target skin's high-frequency pore/specular residual (Target_HighPass_Skin).

    Blended with UltraMax's 1024px output exclusively over unoccluded Class 1 (Skin) pixels
    so the swapped skin retains authentic camera micro-texture under scene lighting.
    """
    enh = np.asarray(enhanced)
    tgt = np.asarray(target)
    if enh.ndim != 3 or enh.shape[2] != 3 or not np.isfinite(enh).all():
        raise DomainError("QUALITY_INPUT_INVALID", "Enhanced image must be uint8 or float32 BGR", 422)
    h, w = enh.shape[:2]
    if tgt.shape[:2] != (h, w):
        tgt = cv2.resize(tgt, (w, h), interpolation=cv2.INTER_CUBIC)

    skin = np.clip(np.asarray(skin_mask, dtype=np.float32), 0.0, 1.0)
    if skin.shape != (h, w):
        skin = cv2.resize(skin, (w, h), interpolation=cv2.INTER_LINEAR)

    if alpha_mask is not None:
        alpha = np.clip(np.asarray(alpha_mask, dtype=np.float32), 0.0, 1.0)
        if alpha.shape != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        skin = skin * alpha

    if not skin.any() or injection_rate <= 0:
        return enh.copy()

    # Extract target skin high-frequency micro-texture (pore/specular residual)
    target_low = cv2.GaussianBlur(tgt.astype(np.float32), (0, 0), sigmaX=highpass_sigma, sigmaY=highpass_sigma)
    target_highpass = tgt.astype(np.float32) - target_low
    target_highpass_skin = target_highpass * skin[..., None]

    blended = enh.astype(np.float32) + float(injection_rate) * target_highpass_skin
    if np.issubdtype(enh.dtype, np.integer):
        return np.clip(blended, 0.0, 255.0).round().astype(enh.dtype)
    return np.clip(blended, 0.0, 255.0).astype(enh.dtype)
