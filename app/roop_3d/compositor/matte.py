"""Edge-aware matte refinement and edge decontamination (pure NumPy/OpenCV, no models).

Both operations are deterministic image-processing primitives:

* ``guided_filter`` is the standard He et al. guided filter; it snaps a soft, low-resolution
  occluder matte onto the real edges of the full-resolution frame, so hand, hair and prop
  boundaries are not blocky or haloed.
* ``decontaminate_edges`` replaces the colour of partially transparent source pixels with
  colour propagated from the opaque interior (normalised convolution), so background or border
  colour baked into the swapped crop cannot bleed back out through the feathered edge.
"""
from __future__ import annotations

import cv2
import numpy as np

from roop_3d.core.errors import DomainError


def _box(image: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.boxFilter(image, cv2.CV_32F, (size, size), borderType=cv2.BORDER_REPLICATE)


def guided_filter(guide: np.ndarray, source: np.ndarray, *, radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    """Edge-preserving smoothing of ``source`` (HxW, 0..1) steered by ``guide`` (HxW, 0..1)."""
    g = np.asarray(guide, dtype=np.float32)
    p = np.asarray(source, dtype=np.float32)
    if g.ndim != 2 or p.shape != g.shape or not np.isfinite(g).all() or not np.isfinite(p).all():
        raise DomainError("MATTE_INPUT_INVALID", "Guide and matte must be matching finite 2-D arrays", 422)
    if radius < 1 or eps <= 0:
        raise ValueError("radius must be >= 1 and eps positive")
    mean_g, mean_p = _box(g, radius), _box(p, radius)
    var_g = _box(g * g, radius) - mean_g * mean_g
    cov_gp = _box(g * p, radius) - mean_g * mean_p
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    return np.clip(_box(a, radius) * g + _box(b, radius), 0.0, 1.0).astype(np.float32)


def fast_guided_filter(
    guide: np.ndarray,
    source: np.ndarray,
    *,
    radius: int = 8,
    eps: float = 1e-4,
    s: int = 2,
) -> np.ndarray:
    """Fast Guided Filter (He & Sun 2015) for high-performance edge-preserving alpha matting.

    Subsamples inputs by factor s (s=1 is exact guided filter), computes linear coefficients a and b
    via box filtering, and upsamples back to reconstruct edge-preserving filtered output q = a * I + b.
    Runs on OpenCL/GPU where available through cv2.UMat.
    """
    g = np.asarray(guide, dtype=np.float32)
    p = np.asarray(source, dtype=np.float32)
    if g.ndim != 2 or p.shape != g.shape or not np.isfinite(g).all() or not np.isfinite(p).all():
        raise DomainError("MATTE_INPUT_INVALID", "Guide and matte must be matching finite 2-D arrays", 422)
    if radius < 1 or eps <= 0:
        raise ValueError("radius must be >= 1 and eps positive")
    if s < 1:
        raise ValueError("subsample factor s must be >= 1")

    h, w = g.shape
    r = max(1, radius // s) if s > 1 else radius

    if s > 1 and min(h, w) >= 16:
        small_w, small_h = max(1, w // s), max(1, h // s)
        small_g = cv2.resize(g, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        small_p = cv2.resize(p, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        mean_g = _box(small_g, r)
        mean_p = _box(small_p, r)
        var_g = _box(small_g * small_g, r) - mean_g * mean_g
        cov_gp = _box(small_g * small_p, r) - mean_g * mean_p

        a = cov_gp / (var_g + eps)
        b = mean_p - a * mean_g

        mean_a = cv2.resize(_box(a, r), (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(_box(b, r), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mean_g, mean_p = _box(g, r), _box(p, r)
        var_g = _box(g * g, r) - mean_g * mean_g
        cov_gp = _box(g * p, r) - mean_g * mean_p
        a = cov_gp / (var_g + eps)
        b = mean_p - a * mean_g
        mean_a = _box(a, r)
        mean_b = _box(b, r)

    q = mean_a * g + mean_b
    return np.clip(q, 0.0, 1.0).astype(np.float32)


# Alias per VFX specification
guidedFilter = fast_guided_filter


def refine_occluder(occluder: np.ndarray, guide_bgr: np.ndarray, *, radius: int = 4,
                    eps: float = 4e-3, strength: float = 1.0, contrast: float = 1.5) -> np.ndarray:
    """Snap an occluder matte to image edges. ``strength`` 0 returns the input unchanged.

    The guided filter relocates the transition onto the real edge but keeps the matte's soft plateau
    levels (a 0.15 tail stays 0.15), which would leave a faint halo of the original face beside a hand.
    ``contrast`` > 1 tightens the plateaus around 0.5 after the edge has been located.
    """
    front = np.clip(np.asarray(occluder, dtype=np.float32), 0.0, 1.0)
    image = np.asarray(guide_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != front.shape:
        raise DomainError("MATTE_INPUT_INVALID", "Guide image must match the occluder matte", 422)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be within 0..1")
    if strength == 0.0 or not front.any():
        return front
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    refined = guided_filter(gray, front, radius=radius, eps=eps)
    # The filter must never invent foreground where the models saw none far from any edge.
    support = cv2.dilate((front > 0.02).astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0
    refined = np.where(support, refined, 0.0).astype(np.float32)
    if contrast != 1.0:
        refined = np.clip((refined - 0.5) * contrast + 0.5, 0.0, 1.0)
    return np.clip(front + (refined - front) * strength, 0.0, 1.0).astype(np.float32)


def decontaminate_edges(source: np.ndarray, alpha: np.ndarray, *, core: float = 0.96,
                        sigma: float = 3.0, strength: float = 1.0,
                        band: tuple[float, float] = (0.3, 0.8)) -> np.ndarray:
    """Replace the colour of the outer feather with opaque-interior colour.

    ``core`` (alpha >= core) defines the trusted interior whose colour is propagated outward.
    Pixels with alpha >= ``band[1]`` keep their own colour (they carry real face detail); pixels with
    alpha <= ``band[0]`` take the propagated interior colour fully; in between it is a linear ramp.
    Where no interior colour is reachable the original pixel is kept.
    """
    image = np.asarray(source)
    weight = np.asarray(alpha, dtype=np.float32)
    if (image.ndim != 3 or image.shape[2] != 3 or weight.shape != image.shape[:2]
            or not np.isfinite(weight).all() or not np.isfinite(image).all()):
        raise DomainError("MATTE_INPUT_INVALID", "Source and alpha must have matching finite shapes", 422)
    if not 0.0 < core <= 1.0 or sigma <= 0 or not 0.0 <= strength <= 1.0 or not 0.0 <= band[0] < band[1] <= 1.0:
        raise ValueError("Invalid decontamination parameters")
    weight = np.clip(weight, 0.0, 1.0)
    opaque = (weight >= core).astype(np.float32)
    if strength == 0.0 or not opaque.any() or opaque.all():
        return image.copy()
    colour = image.astype(np.float32)
    interior = np.zeros_like(colour)
    reachable = np.zeros(weight.shape, bool)
    for scale in (sigma, sigma * 3.0):             # multi-scale: a wide feather may be far from the opaque core
        radius = max(1, int(np.ceil(scale * 3)))
        ksize = (2 * radius + 1, 2 * radius + 1)
        numerator = cv2.GaussianBlur(colour * opaque[..., None], ksize, scale, borderType=cv2.BORDER_REPLICATE)
        denominator = cv2.GaussianBlur(opaque, ksize, scale, borderType=cv2.BORDER_REPLICATE)
        found = (denominator > 0.05) & ~reachable
        interior[found] = (numerator / np.maximum(denominator, 1e-6)[..., None])[found]
        reachable |= found
    low, high = band
    ramp = np.clip((high - weight) / max(high - low, 1e-6), 0.0, 1.0)
    pull = np.where(reachable & (weight < core), ramp * strength, 0.0).astype(np.float32)
    mixed = colour * (1.0 - pull[..., None]) + interior * pull[..., None]
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        return np.clip(mixed, limits.min, limits.max).round().astype(image.dtype)
    return mixed.astype(image.dtype)


def border_fade(shape: tuple[int, int], fraction: float = 0.05) -> np.ndarray:
    """Smooth 0 -> 1 ramp over ``fraction`` of the crop size, 0 on the crop border.

    The replacement crop is the only source of swapped content: beyond its border nothing exists. A face alpha
    that is still opaque at the border therefore ends in a hard rectangular seam (seen on real footage when the
    chin or forehead leaves the aligned crop); fading it removes the seam.
    """
    height, width = int(shape[0]), int(shape[1])
    if height < 4 or width < 4 or not 0.0 < fraction <= 0.5:
        raise ValueError("border_fade needs a crop of at least 4 px and a fraction within (0, 0.5]")
    radius = max(2.0, min(height, width) * fraction)
    rows = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    cols = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)
    ramp = np.clip(np.minimum(rows[:, None], cols[None, :]) / radius, 0.0, 1.0)
    return (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)
