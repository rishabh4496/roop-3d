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
    out = np.clip(np.nan_to_num(out), 0, 255).round().astype(np.uint8)
    out[mask <= 0] = image[mask <= 0]
    return out


def shading_gain_match(source: np.ndarray, target: np.ndarray, alpha: np.ndarray, *, strength: float = 1.0,
                       sigma_fraction: float = 0.03,
                       gain_range: tuple[float, float] = (0.6, 1.5), work_side: int = 192) -> np.ndarray:
    """Match low-frequency target lightness using only the supplied skin/nose mask.

    Normalized blurs do not sample adjoining eyes, lips, hair or foreground.
    The rim follows observed lightness through a zero-mean gain correction;
    global brightness remains the color step's job.
    Fine detail and all pixels outside skin are preserved.
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
    domain = shrink(weight)
    sigma = max(1.5, sigma_fraction * work)
    # Premultiply before reducing resolution, so area resampling cannot mix
    # adjoining non-skin pixels into a partially covered skin sample.
    small_s = shrink(lab_s[..., 0] * weight) / np.maximum(domain, 1e-4)
    small_t = shrink(lab_t[..., 0] * weight) / np.maximum(domain, 1e-4)
    low_s = _normalised_blur(small_s, domain, sigma) + 5.0
    low_t = _normalised_blur(small_t, domain, sigma) + 5.0
    gain = np.clip(low_t / low_s, 0.05, 20.0)
    # Normalize the spatial shading first, then anchor the rim to the observed
    # ratio. Remove the correction's DC component uniformly: concentrating that
    # compensation in the interior invents a bright dome absent from the target.
    mean_gain = float((gain * domain).sum() / max(float(domain.sum()), 1e-6))
    spatial_gain = gain / max(mean_gain, 1e-6)
    distance = cv2.distanceTransform(np.pad((domain > 0.5).astype(np.uint8), 1),
                                     cv2.DIST_L2, 5)[1:-1, 1:-1]
    interior = np.clip(distance / (3.0 * sigma), 0.0, 1.0)
    interior *= interior * (3.0 - 2.0 * interior)
    gain = spatial_gain + (gain - spatial_gain) * (1.0 - interior)
    # Account for both the source lightness and the mask applied below, rather
    # than preserving an unweighted mean of ratios that is not image brightness.
    lightness_weight = small_s * domain * domain
    mean_delta = float(((gain - 1.0) * lightness_weight).sum()
                       / max(float(lightness_weight.sum()), 1e-6))
    gain -= mean_delta
    gain = np.clip(gain, gain_range[0], gain_range[1])
    gain = cv2.resize(gain, (side, side), interpolation=cv2.INTER_CUBIC)
    lab_s[..., 0] = np.clip(lab_s[..., 0] * (1.0 + (gain - 1.0) * strength * weight), 0.0, 100.0)
    out = cv2.cvtColor(lab_s, cv2.COLOR_LAB2BGR) * 255.0
    out = np.clip(np.nan_to_num(out), 0, 255).round().astype(np.uint8)
    out[weight <= 0] = image[weight <= 0]
    return out


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

    Returns the matched source, applied ``sigma``/``motion_length``, and measured
    ``target_energy``/``optics_observable`` when the mask supports measurement.
    A sharper source is never invented; a featureless target is unobservable.
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
    # A featureless target does not identify an optical PSF. Driving its energy
    # to zero would erase genuine enhancer detail, not match measured defocus.
    info["target_energy"] = target_energy
    if target_energy <= 1e-8:
        info["optics_observable"] = 0.0
        return image.copy(), info
    info["optics_observable"] = 1.0
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
    """Add observed target DoG luminance detail only within unoccluded skin/nose.

    Pores and small pigment marks are retained up to the source's detail-energy
    deficit, not detected or invented. Skin-normalized filters cannot import
    adjoining eyes, lips or foreground edges into the texture band.
    """
    image, reference = np.asarray(source), np.asarray(target)
    weight = (np.clip(np.asarray(alpha, np.float32), 0, 1) * np.clip(np.asarray(skin, np.float32), 0, 1)
              * (np.asarray(footprint, np.float32) > 0.5))
    if (image.ndim != 3 or image.shape[2] != 3 or image.shape != reference.shape
            or weight.shape != image.shape[:2] or not np.isfinite(weight).all()
            or not np.isfinite(image).all() or not np.isfinite(reference).all()):
        raise DomainError("QUALITY_INPUT_INVALID", "Texture inputs must have matching finite shapes", 422)
    if (not np.isfinite(strength) or not 0 <= strength <= 1 or not np.isfinite(fine_sigma)
            or not np.isfinite(coarse_sigma) or not 0 < fine_sigma < coarse_sigma):
        raise ValueError("Texture strength must be within 0..1 and 0 < fine_sigma < coarse_sigma")
    if strength == 0.0 or int((weight > 0.5).sum()) < 64:
        return image.copy(), 0.0
    lum_t, lum_s = _gray01(reference), _gray01(image)
    band_t = _normalised_blur(lum_t, weight, fine_sigma) - _normalised_blur(lum_t, weight, coarse_sigma)
    band_s = _normalised_blur(lum_s, weight, fine_sigma) - _normalised_blur(lum_s, weight, coarse_sigma)
    energy_t = _normalised_blur(band_t * band_t, weight, 6.0)
    energy_s = _normalised_blur(band_s * band_s, weight, 6.0)
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
    """Retain the target's observed pore/freckle band after restoration, on skin/nose only.

    A skin-normalized difference of Gaussians rejects lighting gradients and
    protected feature edges. The energy-deficit gate avoids doubling already
    textured source skin. This is not a mole detector or source-pigment projection.
    """
    enh = np.asarray(enhanced)
    tgt = np.asarray(target)
    if (not np.isfinite(injection_rate) or not 0 <= injection_rate <= 1
            or not np.isfinite(highpass_sigma) or highpass_sigma <= 0):
        raise ValueError("Texture retention must be within 0..1 and its scale must be positive")
    if tgt.ndim != 3 or tgt.shape[2] != 3 or not np.isfinite(tgt).all():
        raise DomainError("QUALITY_INPUT_INVALID", "Target image must be finite BGR", 422)
    if enh.ndim != 3 or enh.shape[2] != 3 or not np.isfinite(enh).all():
        raise DomainError("QUALITY_INPUT_INVALID", "Enhanced image must be uint8 or float32 BGR", 422)
    h, w = enh.shape[:2]
    if tgt.shape[:2] != (h, w):
        tgt = cv2.resize(tgt, (w, h), interpolation=cv2.INTER_CUBIC)

    skin = np.clip(np.asarray(skin_mask, dtype=np.float32), 0.0, 1.0)
    if skin.ndim != 2 or not np.isfinite(skin).all():
        raise DomainError("QUALITY_INPUT_INVALID", "Skin texture mask must be a finite 2-D map", 422)
    if skin.shape != (h, w):
        skin = cv2.resize(skin, (w, h), interpolation=cv2.INTER_LINEAR)

    if alpha_mask is not None:
        alpha = np.clip(np.asarray(alpha_mask, dtype=np.float32), 0.0, 1.0)
        if alpha.ndim != 2 or not np.isfinite(alpha).all():
            raise DomainError("QUALITY_INPUT_INVALID", "Texture alpha must be a finite 2-D map", 422)
        if alpha.shape != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        skin = skin * alpha

    if not skin.any() or injection_rate <= 0:
        return enh.copy()

    fine_sigma = highpass_sigma / 3.0
    target_f, enhanced_f = tgt.astype(np.float32), enh.astype(np.float32)
    target_band = (_normalised_blur(target_f, skin, fine_sigma)
                   - _normalised_blur(target_f, skin, highpass_sigma))
    source_band = (_normalised_blur(enhanced_f, skin, fine_sigma)
                   - _normalised_blur(enhanced_f, skin, highpass_sigma))
    energy_sigma = max(1.0, highpass_sigma * 3.0)
    target_energy = _normalised_blur(np.mean(target_band * target_band, axis=-1), skin, energy_sigma)
    source_energy = _normalised_blur(np.mean(source_band * source_band, axis=-1), skin, energy_sigma)
    deficit = np.sqrt(np.maximum(target_energy - source_energy, 0.0) / np.maximum(target_energy, 1e-6))
    gain = float(injection_rate) * skin * np.clip(deficit, 0.0, 1.0)
    blended = enhanced_f + target_band * gain[..., None]
    if np.issubdtype(enh.dtype, np.integer):
        return np.clip(blended, 0.0, 255.0).round().astype(enh.dtype)
    return np.clip(blended, 0.0, 255.0).astype(enh.dtype)
