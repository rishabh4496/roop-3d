"""Luminance-chrominance decoupled spherical-harmonic relighting."""
from __future__ import annotations

import cv2
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
    occluder_mask: np.ndarray | None = None,
    color_space: str = "ycbcr",
) -> np.ndarray:
    """Decoupled luminance relighting over unoccluded Class 1 (Skin) pixels.

    Replaces flat RGB multiplication with YCbCr/LAB luminance-chrominance decoupled matching.
    Never samples dark hair or neck shadows when computing skin illumination statistics.
    """
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

    source_rgb = source[..., ::-1].astype(np.float32) / 255.0
    target_rgb = target[..., ::-1].astype(np.float32) / 255.0
    source_luma = source_rgb @ np.array((0.2126, 0.7152, 0.0722), np.float32)
    target_luma = target_rgb @ np.array((0.2126, 0.7152, 0.0722), np.float32)

    # Exclusively sample unoccluded Class 1 (Skin) pixels; exclude dark hair and neck shadows (< 0.08)
    samples = (skin > 0.5) & (face > 0.05) & (target_luma > 0.08) & (source_luma > 0.08)
    if occluder_mask is not None:
        occ = np.asarray(occluder_mask, dtype=np.float32)
        if occ.shape == samples.shape:
            samples &= (occ < 0.15)

    if np.count_nonzero(samples) < minimum_samples:
        # Fallback to general skin region if insufficient shadow-free samples
        samples = (skin > 0.5) & (face > 0.05)
        if np.count_nonzero(samples) < minimum_samples:
            return source.copy()

    basis = _basis(source.shape[0], source.shape[1])
    source_coeff, *_ = np.linalg.lstsq(basis[samples], source_luma[samples], rcond=1e-4)
    target_coeff, *_ = np.linalg.lstsq(basis[samples], target_luma[samples], rcond=1e-4)
    source_shade = np.maximum(basis @ source_coeff, 0.05)
    target_shade = np.maximum(basis @ target_coeff, 0.05)
    gain = np.clip(target_shade / source_shade, low, high)

    active = (skin > 0.05) & (face > 0)
    if not active.any():
        return source.copy()

    # Luminance-chrominance decoupled matching (anti-plastic finish)
    if color_space == "lab":
        lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Modulate L (lightness) exclusively
        lab[active, 0] = np.clip(lab[active, 0] * gain[active], 0.0, 255.0)
        output = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    else:  # default YCbCr
        ycc = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        # Modulate Y (luminance) exclusively
        ycc[active, 0] = np.clip(ycc[active, 0] * gain[active], 0.0, 255.0)
        output = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    return output


# ---------------------------------------------------------------------------
# Submental Under-Chin Contact Shadow Projection
# ---------------------------------------------------------------------------

def project_submental_contact_shadow(
    image_bgr: np.ndarray,
    landmarks_or_contour: np.ndarray | None,
    light_direction: np.ndarray | None = None,
    neck_mask: np.ndarray | None = None,
    *,
    max_opacity: float = 0.12,
    falloff_px: float = 10.0,
    jaw_extension_factor: float = 1.0,
) -> np.ndarray:
    """Project a subtle, soft under-chin contact shadow gradient onto the upper neck.

    Using the 3D head mesh's lower jaw contour and the estimated overhead lighting vector
    SH_tgt, projects a subtle, soft contact shadow gradient (0–15% opacity, 6px–14px falloff)
    immediately below the chin/jawline onto the target's upper neck only when the 3D jaw
    extends slightly past the target's original chin shadow.

    Args:
        image_bgr: (H, W, 3) BGR image to receive shadow.
        landmarks_or_contour: 2D jaw contour or facial landmarks (e.g. 68-pt).
        light_direction: (3,) unit vector from SH_tgt key light direction.
        neck_mask: Optional BiSeNet neck mask (Class 14/15) to restrict shadow.
        max_opacity: Peak shadow darkening, bounded in [0.0, 0.15].
        falloff_px: Spatial shadow falloff, bounded in [6.0, 14.0].
        jaw_extension_factor: Scale factor (0..1) of jaw extension beyond target chin.

    Returns:
        Shadowed image (H, W, 3) uint8 or float32 matching input.
    """
    img = np.asarray(image_bgr)
    h, w = img.shape[:2]
    if landmarks_or_contour is None or len(landmarks_or_contour) < 3 or jaw_extension_factor <= 1e-4:
        return img.copy()

    # Bounded parameter constraints per VFX specification
    eff_opacity = float(np.clip(max_opacity, 0.0, 0.15))
    eff_falloff = float(np.clip(falloff_px, 6.0, 14.0))

    # Parse jaw contour points
    pts = np.asarray(landmarks_or_contour, dtype=np.float32)
    if len(pts) == 68:
        jaw_pts = pts[3:14]  # chin / lower mandible curve
    elif len(pts) == 106:
        jaw_pts = pts[7:26]
    elif len(pts) >= 5:
        jaw_pts = pts
    else:
        return img.copy()

    # Determine overhead lighting vector from SH key light direction
    if light_direction is not None:
        l_vec = np.asarray(light_direction, dtype=np.float32)
        norm = np.linalg.norm(l_vec)
        if norm > 1e-6:
            l_norm = l_vec / norm
        else:
            l_norm = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    else:
        # Default overhead key light (+Y in screen coordinates points downward)
        l_norm = np.array([0.0, 0.85, 0.52], dtype=np.float32)

    # Overhead angle factor: shadow casts down when light comes from above
    overhead_factor = float(np.clip(l_norm[1], 0.1, 1.0))

    # Mask of the submental space immediately below the jaw contour
    p_start = jaw_pts[0]
    p_end = jaw_pts[-1]
    poly_below = np.array([
        *jaw_pts,
        [p_end[0], min(h - 1, p_end[1] + eff_falloff * 2.5)],
        [w - 1, h - 1],
        [0, h - 1],
        [p_start[0], min(h - 1, p_start[1] + eff_falloff * 2.5)],
    ], dtype=np.float32)

    submental_region = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(submental_region, [np.rint(poly_below).astype(np.int32)], 255)

    # Jawline boundary curve
    jaw_line = np.zeros((h, w), dtype=np.uint8)
    int_jaw = np.rint(jaw_pts).astype(np.int32)
    cv2.polylines(jaw_line, [int_jaw], isClosed=False, color=255, thickness=2)

    # Distance transform from jawline downward
    dist = cv2.distanceTransform(255 - jaw_line, cv2.DIST_L2, 5)

    # Smooth Hermite cubic contact shadow gradient
    grad = np.clip(1.0 - dist / eff_falloff, 0.0, 1.0)
    smooth_shadow = (grad * grad * (3.0 - 2.0 * grad)) * (submental_region > 0)

    # If neck mask provided, gate shadow to the upper neck
    if neck_mask is not None:
        nmask = np.clip(np.asarray(neck_mask, dtype=np.float32), 0.0, 1.0)
        if nmask.shape != (h, w):
            nmask = cv2.resize(nmask, (w, h), interpolation=cv2.INTER_LINEAR)
        # Dilate neck mask slightly so submental throat fold is covered
        nmask = cv2.dilate(nmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        smooth_shadow *= nmask

    # Project subtle contact shadow (0-15% opacity)
    total_opacity = eff_opacity * overhead_factor * float(np.clip(jaw_extension_factor, 0.0, 1.0))
    shadow_mult = 1.0 - (total_opacity * smooth_shadow)

    output = img.astype(np.float32) * shadow_mult[..., None]
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(output, 0, 255).round().astype(img.dtype)
    return output.astype(img.dtype)
