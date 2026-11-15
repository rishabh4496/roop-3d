"""Full-frame mesh mattes and skin photometry; no face warping or portrait restoration.

The 19-class CelebAMask-HQ partition deliberately includes both ears. Background,
glasses, earrings, neck, necklaces, clothes, hair and hats are hard exclusions.
Every morphology and pyramid operation remains bounded by that observed support.
"""
from __future__ import annotations

import cv2
import numpy as np

from roop_3d.core.errors import DomainError

FACE_CLASSES = (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13)
SKIN_CLASSES = (1, 7, 8, 10)
FORBIDDEN_CLASSES = (0, 6, 9, 14, 15, 16, 17, 18)


def semantic_face_masks(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return face confidence, hard exclusions, skin-only confidence and ear support."""
    scores = np.asarray(probabilities, np.float32)
    if (scores.ndim != 3 or scores.shape[2] != 19 or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))
            or not np.allclose(scores.sum(axis=2), 1.0, atol=2e-3)):
        raise DomainError("TRUE3D_SEMANTIC_MASK_INVALID", "BiSeNet must return normalized H x W x 19 probabilities", 503)
    labels = np.argmax(scores, axis=2)
    forbidden_probability = scores[..., FORBIDDEN_CLASSES].sum(axis=2)
    forbidden = (np.isin(labels, FORBIDDEN_CLASSES) | (forbidden_probability >= 0.4)
                 | (scores[..., 17] >= 0.2) | (scores[..., 6] >= 0.2) | (scores[..., 9] >= 0.2))
    face = scores[..., FACE_CLASSES].sum(axis=2) * ~forbidden
    skin = scores[..., SKIN_CLASSES].sum(axis=2) * np.isin(labels, SKIN_CLASSES) * ~forbidden
    ears = np.isin(labels, (7, 8)) & ~forbidden
    return face.astype(np.float32), forbidden.astype(np.float32), skin.astype(np.float32), ears


def anatomical_alpha(alpha: np.ndarray, allowed: np.ndarray, *, feather_px: float,
                     erosion_px: float, dilation_px: float) -> np.ndarray:
    """Distance-field opening and multiscale inward feather, never beyond allowed pixels.

    Subpixel erosion/dilation cleans only the confidence contour, not the forehead
    or chin via a landmark cutoff. Feather widths are capped relative to the face
    extent, so a small or profile face does not lose its entire jaw to a 5px ramp.
    """
    value = np.asarray(alpha, np.float32)
    support = np.asarray(allowed, bool)
    if value.ndim != 2 or support.shape != value.shape or not np.isfinite(value).all():
        raise DomainError("TRUE3D_SEMANTIC_MASK_INVALID", "Anatomical matte and support must be matching finite planes", 422)
    contour = (value > 0.15) & support
    if not contour.any():
        raise DomainError("TRUE3D_SEMANTIC_MASK_EMPTY", "No visible semantic face intersects the head mesh", 422)
    padded = np.pad(contour.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    eroded = distance > erosion_px
    outside = cv2.distanceTransform((~eroded).astype(np.uint8), cv2.DIST_L2, 5)
    cleaned = (eroded | (outside <= dilation_px))[1:-1, 1:-1] & support
    value = np.where(cleaned, np.clip(value, 0.0, 1.0), 0.0)
    if feather_px == 0:
        return value.astype(np.float32)
    padded = np.pad(cleaned.astype(np.uint8), 1)
    inside = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - padded, cv2.DIST_L2, 5)
    signed = (inside - outside)[1:-1, 1:-1]
    ys, xs = np.nonzero(cleaned)
    if not len(xs):
        raise DomainError("TRUE3D_SEMANTIC_MASK_EMPTY", "Anatomical morphology removed all visible skin", 422)
    extent = min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    width = min(feather_px, max(0.75, 0.035 * extent))
    ramp = np.zeros_like(value)
    for scale, weight in ((0.5, 0.55), (1.0, 0.3), (2.0, 0.15)):
        t = np.clip(signed / (width * scale), 0.0, 1.0)
        ramp += weight * t * t * (3.0 - 2.0 * t)
    return np.minimum(value, ramp).astype(np.float32)


def local_lab_skin_match(source: np.ndarray, target: np.ndarray,
                         skin: np.ndarray) -> tuple[np.ndarray, dict]:
    """Masked Reinhard transfer plus broad local LAB residuals, sampling only skin.

    Eyes, brows, lips, mouth, hair and occluders neither contribute statistics nor
    receive colour edits. Surface-normal lighting is already in the input raster;
    only low-frequency scene colour/exposure residuals are estimated here.
    """
    weight = np.asarray(skin, np.float32)
    selected = weight > 0.15
    if np.count_nonzero(selected) < 8:
        raise DomainError("TRUE3D_SKIN_SAMPLES_INSUFFICIENT", "LAB matching needs at least eight visible skin samples", 422)
    weight = np.where(selected, weight, 0.0)
    total = float(weight.sum())
    a = cv2.cvtColor(source.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    b = cv2.cvtColor(target.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    means, deviations = [], []
    for image in (a, b):
        mean = (image * weight[..., None]).sum(axis=(0, 1)) / total
        variance = ((image - mean) ** 2 * weight[..., None]).sum(axis=(0, 1)) / total
        means.append(mean)
        deviations.append(np.sqrt(np.maximum(variance, 0.25)))
    gain = np.clip(deviations[1] / deviations[0], 0.65, 1.6)
    matched = (a - means[0]) * gain + means[1]
    ys, xs = np.nonzero(selected)
    sigma = max(2.0, 0.12 * min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
    denominator = cv2.GaussianBlur(weight, (0, 0), sigma)
    difference = cv2.GaussianBlur((b - matched) * weight[..., None], (0, 0), sigma)
    local = difference / np.maximum(denominator[..., None], 1e-6)
    local = np.clip(local, (-8.0, -5.0, -5.0), (8.0, 5.0, 5.0))
    adjusted = a + (matched + 0.35 * local - a) * weight[..., None]
    result = cv2.cvtColor(adjusted.astype(np.float32), cv2.COLOR_LAB2BGR)
    result = np.clip(result * 255.0, 0, 255).round().astype(np.uint8)
    result[weight == 0] = source[weight == 0]
    return result, {"method": "skin_only_local_lab_reinhard", "skin_samples": int(selected.sum()),
                    "gain": gain.astype(float).tolist(), "local_sigma_px": float(sigma)}


def finish_skin_detail(source: np.ndarray, matched: np.ndarray, target: np.ndarray,
                       skin: np.ndarray, *, seed: int) -> tuple[np.ndarray, dict]:
    """Retain rendered identity DoG detail, then match measured scene grain deficit.

    No target mole/freckle pattern is transplanted. Neural restoration is exclusively
    canonical-UV work; this operation preserves the raster's measured luminance bands.
    """
    weight = np.asarray(skin, np.float32)
    selected = weight > 0.25
    luma = lambda image: cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
    original, corrected, scene = (luma(image) for image in (source, matched, target))
    dog = lambda plane: cv2.GaussianBlur(plane, (0, 0), 0.55) - cv2.GaussianBlur(plane, (0, 0), 2.2)
    delta = (dog(original) - dog(corrected)) * weight
    detailed = np.clip(matched.astype(np.float32) + delta[..., None], 0, 255)
    # Avoid mistaking pores, lips, edges or a cast shadow for camera sensor noise.
    smooth = cv2.GaussianBlur(scene, (0, 0), 1.2)
    gradient = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
    flat = selected & (gradient < 12.0)
    grain = 0.0
    if np.count_nonzero(flat) >= 8:
        def noise(plane):
            residual = (plane - cv2.GaussianBlur(plane, (0, 0), 0.6))[flat]
            return float(1.4826 * np.median(np.abs(residual - np.median(residual))))
        scene_sigma, source_sigma = noise(scene), noise(luma(detailed))
        grain = min(8.0, float(np.sqrt(max(0.0, scene_sigma ** 2 - source_sigma ** 2))))
        if grain > 0:
            random = np.random.default_rng(seed & 0xFFFFFFFF)
            detailed += random.normal(0.0, grain, weight.shape).astype(np.float32)[..., None] * weight[..., None]
    output = np.clip(detailed, 0, 255).round().astype(np.uint8)
    output[weight == 0] = matched[weight == 0]
    return output, {"method": "identity_dog_and_measured_scene_grain", "dog_sigmas_px": [0.55, 2.2],
                    "grain_sigma": grain, "grain_samples": int(flat.sum()),
                    "grain_state": "measured" if np.count_nonzero(flat) >= 8 else "insufficient_flat_skin"}
