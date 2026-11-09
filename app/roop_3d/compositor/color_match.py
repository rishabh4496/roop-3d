"""Masked LAB statistics and histogram color transfer."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.core.errors import DomainError


@dataclass(frozen=True)
class ColorStats:
    source_mean: tuple[float, float, float]
    source_std: tuple[float, float, float]
    target_mean: tuple[float, float, float]
    target_std: tuple[float, float, float]

    def vector(self) -> np.ndarray:
        return np.asarray((*self.source_mean, *self.source_std, *self.target_mean, *self.target_std), np.float32)

    @classmethod
    def from_vector(cls, value: np.ndarray) -> "ColorStats":
        row = np.asarray(value, dtype=np.float32).reshape(-1)
        if row.size != 12 or not np.isfinite(row).all():
            raise DomainError("COLOR_STATS_INVALID", "Color statistics must contain 12 finite values", 422)
        return cls(tuple(row[0:3]), tuple(row[3:6]), tuple(row[6:9]), tuple(row[9:12]))


def _validate_pair(source: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = np.asarray(source), np.asarray(reference)
    weight = np.asarray(mask, dtype=np.float32)
    if (a.ndim != 3 or b.shape != a.shape or a.shape[2] != 3 or weight.shape != a.shape[:2]
            or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(weight).all()):
        raise DomainError("COLOR_MATCH_INPUT_INVALID", "Color inputs and mask have incompatible dimensions", 422)
    return a, b, np.clip(weight, 0.0, 1.0)


def _lab(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    rgb = np.clip(image.astype(np.float32), 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float32)


def _ycbcr(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    rgb = np.clip(image.astype(np.float32), 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2YCrCb).astype(np.float32)


def estimate_lab_stats(source: np.ndarray, reference: np.ndarray, mask: np.ndarray,
                       forbidden_mask: np.ndarray | None = None) -> ColorStats:
    """Estimate LAB statistics exclusively over unoccluded skin pixels, excluding dark hair and neck shadows."""
    a, b, weight = _validate_pair(source, reference, mask)
    selected = weight > 0.15
    if forbidden_mask is not None:
        fmask = np.clip(np.asarray(forbidden_mask, dtype=np.float32), 0.0, 1.0)
        if fmask.shape == selected.shape:
            selected &= (fmask < 0.15)
    # Exclude dark hair and neck shadows (luma <= 0.08) from skin color statistics
    b_u8 = b if b.dtype == np.uint8 else np.clip(b, 0, 255).astype(np.uint8)
    luma = cv2.cvtColor(b_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow_free = selected & (luma > 0.08)
    if np.count_nonzero(shadow_free) >= 8:
        selected = shadow_free
    elif np.count_nonzero(selected) < 8:
        selected = weight > 0
    if np.count_nonzero(selected) < 2:
        raise DomainError("COLOR_MATCH_MASK_EMPTY", "At least two masked pixels are needed for color matching", 422)
    source_lab, target_lab = _lab(a), _lab(b)
    source_mean = source_lab[selected].mean(axis=0)
    source_std = np.maximum(source_lab[selected].std(axis=0), 1.0)
    target_mean = target_lab[selected].mean(axis=0)
    target_std = np.maximum(target_lab[selected].std(axis=0), 1.0)
    return ColorStats(tuple(source_mean), tuple(source_std), tuple(target_mean), tuple(target_std))


def apply_lab_stats(image: np.ndarray, stats: ColorStats, mask: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
    source = np.asarray(image)
    alpha = np.asarray(mask, dtype=np.float32)
    if source.ndim != 3 or source.shape[2] != 3 or alpha.shape != source.shape[:2] or not np.isfinite(alpha).all():
        raise DomainError("COLOR_MATCH_INPUT_INVALID", "Image and mask dimensions are invalid", 422)
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Color transfer strength must be within 0..1")
    if strength == 0:
        return source.copy()
    lab = _lab(source)
    src_mean = np.asarray(stats.source_mean, np.float32)
    src_std = np.maximum(np.asarray(stats.source_std, np.float32), 1.0)
    tgt_mean = np.asarray(stats.target_mean, np.float32)
    tgt_std = np.maximum(np.asarray(stats.target_std, np.float32), 1.0)
    matched = (lab - src_mean) * (tgt_std / src_std) + tgt_mean
    result_lab = lab.copy()
    active = alpha > 0
    result_lab[active] = lab[active] + (matched[active] - lab[active]) * strength
    output = cv2.cvtColor(np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    output[~active] = source[~active]
    return output


def lab_mean_std_match(source: np.ndarray, reference: np.ndarray, mask: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
    return apply_lab_stats(source, estimate_lab_stats(source, reference, mask), mask, strength=strength)


def estimate_ycbcr_stats(source: np.ndarray, reference: np.ndarray, mask: np.ndarray,
                         forbidden_mask: np.ndarray | None = None) -> ColorStats:
    """Estimate YCbCr statistics exclusively over unoccluded skin pixels, excluding dark hair and neck shadows."""
    a, b, weight = _validate_pair(source, reference, mask)
    selected = weight > 0.15
    if forbidden_mask is not None:
        fmask = np.clip(np.asarray(forbidden_mask, dtype=np.float32), 0.0, 1.0)
        if fmask.shape == selected.shape:
            selected &= (fmask < 0.15)
    b_u8 = b if b.dtype == np.uint8 else np.clip(b, 0, 255).astype(np.uint8)
    luma = cv2.cvtColor(b_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow_free = selected & (luma > 0.08)
    if np.count_nonzero(shadow_free) >= 8:
        selected = shadow_free
    elif np.count_nonzero(selected) < 8:
        selected = weight > 0
    if np.count_nonzero(selected) < 2:
        raise DomainError("COLOR_MATCH_MASK_EMPTY", "At least two masked pixels are needed for color matching", 422)
    source_ycc, target_ycc = _ycbcr(a), _ycbcr(b)
    source_mean = source_ycc[selected].mean(axis=0)
    source_std = np.maximum(source_ycc[selected].std(axis=0), 1.0)
    target_mean = target_ycc[selected].mean(axis=0)
    target_std = np.maximum(target_ycc[selected].std(axis=0), 1.0)
    return ColorStats(tuple(source_mean), tuple(source_std), tuple(target_mean), tuple(target_std))


def apply_ycbcr_stats(image: np.ndarray, stats: ColorStats, mask: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
    source = np.asarray(image)
    alpha = np.asarray(mask, dtype=np.float32)
    if source.ndim != 3 or source.shape[2] != 3 or alpha.shape != source.shape[:2] or not np.isfinite(alpha).all():
        raise DomainError("COLOR_MATCH_INPUT_INVALID", "Image and mask dimensions are invalid", 422)
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Color transfer strength must be within 0..1")
    if strength == 0:
        return source.copy()
    ycc = _ycbcr(source)
    src_mean = np.asarray(stats.source_mean, np.float32)
    src_std = np.maximum(np.asarray(stats.source_std, np.float32), 1.0)
    tgt_mean = np.asarray(stats.target_mean, np.float32)
    tgt_std = np.maximum(np.asarray(stats.target_std, np.float32), 1.0)
    matched = (ycc - src_mean) * (tgt_std / src_std) + tgt_mean
    result_ycc = ycc.copy()
    active = alpha > 0
    result_ycc[active] = ycc[active] + (matched[active] - ycc[active]) * strength
    output = cv2.cvtColor(np.clip(result_ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    output[~active] = source[~active]
    return output


def ycbcr_mean_std_match(source: np.ndarray, reference: np.ndarray, mask: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
    return apply_ycbcr_stats(source, estimate_ycbcr_stats(source, reference, mask), mask, strength=strength)


def masked_histogram_match(source: np.ndarray, reference: np.ndarray, mask: np.ndarray, *, smooth_histogram=None) -> np.ndarray:
    """Per-channel masked LAB CDF matching; all pixels outside the mask are copied."""
    a, b, weight = _validate_pair(source, reference, mask)
    active = weight > 0
    if np.count_nonzero(active) < 2:
        return a.copy()
    source_lab, target_lab = _lab(a), _lab(b)
    mapped = source_lab.copy()
    if smooth_histogram is not None:
        histograms = np.array([[np.bincount(lab[...,c][active].astype(np.uint8),minlength=256) for c in range(3)] for lab in (source_lab,target_lab)],dtype=np.float64)
        histograms /= np.maximum(histograms.sum(axis=-1,keepdims=True),1)
        histograms = np.maximum(smooth_histogram(histograms),0)
        cdfs=np.cumsum(histograms,axis=-1)
        cdfs /= np.maximum(cdfs[...,-1:],1e-12)
        for channel in range(3):
            lut=np.interp(cdfs[0,channel],cdfs[1,channel],np.arange(256))
            mapped[...,channel][active]=lut[source_lab[...,channel][active].astype(np.uint8)]
        output=cv2.cvtColor(np.clip(mapped,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)
        output[~active]=a[~active]
        return output
    for channel in range(3):
        x = source_lab[..., channel][active]
        y = target_lab[..., channel][active]
        values, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        quantiles = np.cumsum(counts).astype(np.float64) / max(1, len(x))
        ref_values, ref_counts = np.unique(y, return_counts=True)
        ref_quantiles = np.cumsum(ref_counts).astype(np.float64) / max(1, len(y))
        mapped_values = np.interp(quantiles, ref_quantiles, ref_values)
        mapped[..., channel][active] = mapped_values[inverse]
    output = cv2.cvtColor(np.clip(mapped, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    output[~active] = a[~active]
    return output


# ---------------------------------------------------------------------------
# Perceptual OKLab Color Space Transformations (Björn Ottosson, 2020)
# ---------------------------------------------------------------------------

def srgb_to_oklab(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image [0, 255] to perceptual OKLab space [L, a, b].

    OKLab provides superior perceptual uniformity, hue linearity, and chroma predictability
    compared to CIELAB, eliminating the classic blue-purple hue shift.
    """
    img = np.asarray(image_bgr, dtype=np.float32)
    # BGR -> RGB in [0, 1]
    rgb = img[..., ::-1] / 255.0
    # Linearize sRGB
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    # sRGB to LMS
    l = 0.4122214708 * lin[..., 0] + 0.5363325363 * lin[..., 1] + 0.0514459929 * lin[..., 2]
    m = 0.2119034982 * lin[..., 0] + 0.6806995451 * lin[..., 1] + 0.1073969566 * lin[..., 2]
    s = 0.0883024619 * lin[..., 0] + 0.2817188376 * lin[..., 1] + 0.6299787005 * lin[..., 2]
    # Non-linear cube root with sign preservation
    l_ = np.cbrt(np.maximum(l, 0.0))
    m_ = np.cbrt(np.maximum(m, 0.0))
    s_ = np.cbrt(np.maximum(s, 0.0))
    # LMS to OKLab
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def oklab_to_srgb(oklab: np.ndarray) -> np.ndarray:
    """Convert OKLab [L, a, b] back to BGR uint8 [0, 255]."""
    ok = np.asarray(oklab, dtype=np.float32)
    L, a, b = ok[..., 0], ok[..., 1], ok[..., 2]
    # OKLab to LMS cube root
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    # Cube
    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3
    # LMS to linear RGB
    r_lin = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    rgb_lin = np.clip(np.stack([r_lin, g_lin, b_lin], axis=-1), 0.0, 1.0)
    # Gamma encode to sRGB
    rgb = np.where(rgb_lin <= 0.0031308, 12.92 * rgb_lin, 1.055 * (rgb_lin ** (1.0 / 2.4)) - 0.055)
    bgr = np.clip(rgb[..., ::-1] * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return bgr


# ---------------------------------------------------------------------------
# Skin Diffuse / Specular Highlight Separation
# ---------------------------------------------------------------------------

def separate_skin_specular_diffuse(
    image_bgr: np.ndarray,
    skin_mask: np.ndarray | None = None,
    *,
    threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate diffuse skin reflectance from specular skin highlights (Specular_Map).

    Skin reflectance physically consists of a chromatic diffuse subsurface component and
    an unpolarized neutral Fresnel specular reflection (pore glints, forehead/nasal sheen).
    Separating them before color matching ensures natural skin sheen and pore highlights
    never look chalky or flat when recombined.

    Returns:
        (diffuse_bgr, specular_map)
    """
    img = np.asarray(image_bgr, dtype=np.float32)
    # Luminance channel
    luma = (0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]).astype(np.float32)
    # Diffuse base via morphological opening (removes localized specular glints) + spatial smoothing
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    luma_diffuse = cv2.morphologyEx(luma, cv2.MORPH_OPEN, k)
    luma_diffuse = cv2.GaussianBlur(luma_diffuse, (5, 5), 1.5)

    # Highlight residual
    specular_residual = np.maximum(luma - luma_diffuse, 0.0)
    if skin_mask is not None:
        smask = np.clip(np.asarray(skin_mask, dtype=np.float32), 0.0, 1.0)
        specular_residual *= smask

    # Threshold small noise to preserve genuine specular sheen
    specular_residual = np.where(specular_residual > threshold, specular_residual, 0.0)
    specular_map = np.stack([specular_residual] * 3, axis=-1).astype(np.float32)

    diffuse = np.clip(img - specular_map, 0.0, 255.0)
    return diffuse, specular_map


def recombine_skin_specular(
    diffuse_bgr: np.ndarray,
    specular_map: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Recombine color-matched diffuse skin with preserved specular highlights."""
    diff = np.asarray(diffuse_bgr, dtype=np.float32)
    spec = np.asarray(specular_map, dtype=np.float32) * float(np.clip(strength, 0.0, 2.0))
    recombined = np.clip(diff + spec, 0.0, 255.0)
    if np.issubdtype(diffuse_bgr.dtype, np.integer):
        return recombined.round().astype(np.uint8)
    return recombined.astype(np.float32)


# ---------------------------------------------------------------------------
# Sliced Wasserstein Optimal Transport Color Matcher in OKLab
# ---------------------------------------------------------------------------

def _fibonacci_sphere_directions(num_points: int = 32) -> np.ndarray:
    """Generate quasi-uniform direction vectors on S^2 using Fibonacci spiral."""
    indices = np.arange(0, num_points, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / num_points)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    dirs = np.stack([x, y, z], axis=-1)
    return (dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)).astype(np.float32)


def sliced_wasserstein_color_transfer_oklab(
    source: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 1.0,
    num_slices: int = 32,
    num_iters: int = 2,
    forbidden_mask: np.ndarray | None = None,
    preserve_specular: bool = True,
) -> np.ndarray:
    """Sliced Wasserstein Optimal Transport color matcher in OKLab space.

    Computed exclusively over unoccluded BiSeNet Class 1 (Skin) pixels, excluding dark
    hair and neck shadows. Separates diffuse skin reflectance from specular skin highlights
    before color matching and re-adds the specular highlights at the end.
    """
    a, b, weight = _validate_pair(source, reference, mask)
    if strength <= 0.0:
        return a.copy()

    # Extract unoccluded skin mask
    selected = weight > 0.15
    if forbidden_mask is not None:
        fmask = np.clip(np.asarray(forbidden_mask, dtype=np.float32), 0.0, 1.0)
        if fmask.shape == selected.shape:
            selected &= (fmask < 0.15)

    # Exclude dark hair and neck shadows (luma <= 0.08)
    b_u8 = b if b.dtype == np.uint8 else np.clip(b, 0, 255).astype(np.uint8)
    luma = cv2.cvtColor(b_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow_free = selected & (luma > 0.08)
    if np.count_nonzero(shadow_free) >= 8:
        selected = shadow_free
    elif np.count_nonzero(selected) < 8:
        selected = weight > 0

    if np.count_nonzero(selected) < 2:
        return a.copy()

    # 1. Specular / diffuse separation
    if preserve_specular:
        diffuse_src, spec_map_src = separate_skin_specular_diffuse(a, selected)
        diffuse_ref, _ = separate_skin_specular_diffuse(b, selected)
    else:
        diffuse_src = a.astype(np.float32)
        diffuse_ref = b.astype(np.float32)
        spec_map_src = np.zeros_like(a, dtype=np.float32)

    # 2. Convert to OKLab
    src_oklab = srgb_to_oklab(diffuse_src)
    ref_oklab = srgb_to_oklab(diffuse_ref)

    # Extract skin sample points
    x_pts = src_oklab[selected].copy()  # (N_s, 3)
    y_pts = ref_oklab[selected].copy()  # (N_t, 3)
    n_s, n_t = len(x_pts), len(y_pts)

    if n_s >= 8 and n_t >= 8:
        # 3. Sliced Wasserstein iteration
        directions = _fibonacci_sphere_directions(num_slices)  # (K, 3)
        qs = np.linspace(0.0, 1.0, n_s, dtype=np.float32)

        for _ in range(max(1, num_iters)):
            p_x = x_pts @ directions.T  # (N_s, K)
            p_y = y_pts @ directions.T  # (N_t, K)

            displacements = np.zeros_like(x_pts)
            for k in range(num_slices):
                order = np.argsort(p_x[:, k])
                sorted_px = p_x[order, k]
                target_quantiles = np.quantile(p_y[:, k], qs)
                delta_p = np.empty_like(sorted_px)
                delta_p[order] = target_quantiles - sorted_px
                displacements += np.outer(delta_p, directions[k])

            displacements *= (3.0 / float(num_slices))
            x_pts += displacements

        # Compute affine transport map from original to transported skin
        orig_mean = src_oklab[selected].mean(axis=0)
        orig_std = np.maximum(src_oklab[selected].std(axis=0), 1e-4)
        new_mean = x_pts.mean(axis=0)
        new_std = np.maximum(x_pts.std(axis=0), 1e-4)

        # Scale ratio with safe fallback for uniform or near-uniform patches
        ratio_std = new_std / orig_std if np.all(orig_std > 1e-3) else np.ones(3, np.float32)

        # Apply to entire active face region
        active = weight > 0
        matched_oklab = src_oklab.copy()
        transported = (src_oklab[active] - orig_mean) * ratio_std + new_mean
        matched_oklab[active] = src_oklab[active] + (transported - src_oklab[active]) * strength
    else:
        # Fallback to mean/std if sample size too small
        orig_mean = src_oklab[selected].mean(axis=0)
        orig_std = np.maximum(src_oklab[selected].std(axis=0), 1e-4)
        tgt_mean = ref_oklab[selected].mean(axis=0)
        tgt_std = np.maximum(ref_oklab[selected].std(axis=0), 1e-4)
        active = weight > 0
        matched_oklab = src_oklab.copy()
        matched_oklab[active] = (src_oklab[active] - orig_mean) * (tgt_std / orig_std) + tgt_mean

    # Convert back to sRGB/BGR
    matched_diffuse = oklab_to_srgb(matched_oklab)

    # 4. Re-add specular highlights
    if preserve_specular:
        output = recombine_skin_specular(matched_diffuse, spec_map_src, strength=1.0)
    else:
        output = matched_diffuse

    output[~active] = a[~active]
    return output


# ---------------------------------------------------------------------------
# Biophysically Inspired Subsurface Scattering (SSS) Shadow Warmth
# ---------------------------------------------------------------------------

def apply_sss_shadow_warmth(
    image_bgr: np.ndarray,
    normals: np.ndarray,
    light_direction: np.ndarray,
    skin_mask: np.ndarray | None = None,
    *,
    warmth_amount: float = 0.5,
    falloff: float = 0.15,
) -> np.ndarray:
    """Physically inspired Subsurface Scattering (SSS) shadow-terminator curve.

    Evaluated along the 3D surface normal gradient (N dot L transition zone) so shadow
    edges on the cheek, nose bridge, and brow retain natural dermal warmth instead of turning grey.
    """
    img = np.asarray(image_bgr, dtype=np.float32)
    norm = np.asarray(normals, dtype=np.float32)
    l_vec = np.asarray(light_direction, dtype=np.float32)
    l_norm = l_vec / max(float(np.linalg.norm(l_vec)), 1e-6)

    # N dot L incidence
    n_dot_l = np.sum(norm * l_norm, axis=-1)  # (H, W)

    # Gaussian bell curve centered at the shadow terminator (N dot L ~ 0.04)
    # Direct light ceases, but subsurface red/amber wavelengths scatter through the boundary
    mu_0 = 0.04
    w_sss = np.exp(-((n_dot_l - mu_0) ** 2) / (2.0 * (falloff ** 2)))

    if skin_mask is not None:
        smask = np.clip(np.asarray(skin_mask, dtype=np.float32), 0.0, 1.0)
        w_sss *= smask

    # Shift warmth in OKLab space: +a (red/capillary), +b (warm amber)
    oklab = srgb_to_oklab(img.round().astype(np.uint8))
    d_a = 0.065 * float(warmth_amount) * w_sss
    d_b = 0.035 * float(warmth_amount) * w_sss
    d_L = 0.008 * float(warmth_amount) * w_sss

    oklab[..., 1] += d_a
    oklab[..., 2] += d_b
    oklab[..., 0] += d_L

    return oklab_to_srgb(oklab)


def oklab_wasserstein_match_with_sss(
    source: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    normals: np.ndarray | None = None,
    light_direction: np.ndarray | None = None,
    *,
    strength: float = 1.0,
    forbidden_mask: np.ndarray | None = None,
    warmth_amount: float = 0.5,
    preserve_specular: bool = True,
) -> tuple[np.ndarray, ColorStats]:
    """Integrated OKLab Sliced Wasserstein Color Match with SSS Shadow Warmth."""
    matched = sliced_wasserstein_color_transfer_oklab(
        source, reference, mask,
        strength=strength,
        forbidden_mask=forbidden_mask,
        preserve_specular=preserve_specular,
    )
    if normals is not None and light_direction is not None and warmth_amount > 0:
        matched = apply_sss_shadow_warmth(
            matched, normals, light_direction,
            skin_mask=mask,
            warmth_amount=warmth_amount,
        )
    # Estimate standard ColorStats for temporal smoothing and downstream telemetry
    stats = estimate_lab_stats(source, reference, mask, forbidden_mask=forbidden_mask)
    return matched, stats
