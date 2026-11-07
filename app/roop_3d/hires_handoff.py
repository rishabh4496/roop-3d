"""High-resolution Stage 2 -> Stage 3 handoff.

The swapper emits a small aligned crop (``inswapper_128``: 128 px). The enhancers (GPEN 512, CodeFormer 512,
GFPGAN 512, UltraMax 512/1024) emit a much larger aligned face. Their detail only reaches the output if the
enhanced tensor is warped *directly* from its native resolution into the frame, so the alignment matrix is
re-based onto the enhanced resolution instead of shrinking the enhanced face back to the swap-crop size:

    S = enhancer_res / swap_res
    frame -> hr crop : scale_matrix_output(M_fwd, S)   (basis AND translation scaled by S)
    hr crop -> frame : scale_matrix_input(M_inv, S)    (basis divided by S)

Both include the half-pixel centre correction ``(S - 1) / 2`` so the result is identical to a centre-aligned
``cv2.resize`` (the convention every restorer and ``grid_sample(align_corners=False)`` uses).

Everything that touches pixels in the second half of this module is torch / CUDA resident (NCHW, RGB, float32,
[0, 1]): bicubic ROI warp, weighted LAB / YCbCr mean-std and histogram colour transfer restricted to the skin
class, and a multi-band Laplacian blend at the native frame ROI resolution.
"""
from __future__ import annotations

import math

import numpy as np

from roop_3d.core.errors import DomainError

COLOR_METHODS = ("reinhard_lab", "regional_reinhard", "reinhard_ycbcr", "histogram_lab", "histogram_ycbcr")


# ----------------------------------------------------------------------------------- geometry (host, numpy)
def _affine(matrix) -> np.ndarray:
    value = np.asarray(matrix, np.float64)
    if value.shape != (2, 3) or not np.isfinite(value).all():
        raise DomainError("HIRES_MATRIX_INVALID", "Alignment matrix must be a finite 2x3 affine", 422)
    return value


def invert_affine_np(matrix) -> np.ndarray:
    value = _affine(matrix)
    linear = value[:, :2]
    if abs(float(np.linalg.det(linear))) < 1e-12:
        raise DomainError("HIRES_MATRIX_INVALID", "Alignment matrix is singular", 422)
    inverse = np.linalg.inv(linear)
    return np.concatenate((inverse, (-inverse @ value[:, 2])[:, None]), axis=1)


def hires_scale(enhanced_res: int, swap_res: int) -> float:
    """S = enhancer_res / swap_res."""
    if enhanced_res < 1 or swap_res < 1:
        raise ValueError("Resolutions must be positive")
    return float(enhanced_res) / float(swap_res)


def scale_matrix_output(matrix, scale: float, *, pixel_centers: bool = True) -> np.ndarray:
    """``M`` maps X -> crop. Returns the matrix mapping X -> (crop resampled by ``scale``)."""
    value = _affine(matrix) * float(scale)
    if pixel_centers:
        value[:, 2] += (float(scale) - 1.0) / 2.0
    return value


def scale_matrix_input(matrix, scale: float, *, pixel_centers: bool = True) -> np.ndarray:
    """``M`` maps crop -> Y. Returns the matrix mapping (crop resampled by ``scale``) -> Y."""
    value = _affine(matrix)
    centre = (float(scale) - 1.0) / 2.0 if pixel_centers else 0.0
    result = value.copy()
    result[:, :2] = value[:, :2] / float(scale)
    result[:, 2] = value[:, 2] - value[:, :2] @ np.array((centre / scale, centre / scale))
    return result


def mapped_bounds(matrix, size: tuple[int, int], frame_size: tuple[int, int], margin: int = 4
                  ) -> tuple[int, int, int, int] | None:
    """Frame-space box (x0, y0, x1, y1) covered by a ``size`` (h, w) image mapped through ``matrix``."""
    value = _affine(matrix)
    h, w = size
    corners = np.array(((-0.5, -0.5), (w - 0.5, -0.5), (w - 0.5, h - 0.5), (-0.5, h - 0.5)), np.float64)
    mapped = corners @ value[:, :2].T + value[:, 2]
    x0 = max(0, int(math.floor(mapped[:, 0].min())) - margin)
    y0 = max(0, int(math.floor(mapped[:, 1].min())) - margin)
    x1 = min(int(frame_size[1]), int(math.ceil(mapped[:, 0].max())) + margin)
    y1 = min(int(frame_size[0]), int(math.ceil(mapped[:, 1].max())) + margin)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def destination_face_side(inverse, alpha: np.ndarray, threshold: float = 0.5) -> float | None:
    """Shorter side, in FRAME pixels, of the box the swapped face really occupies in the target frame.

    This is the actual destination bounding box (support of the face alpha mapped through the crop->frame matrix),
    not the size of the 128/256 px working crop. ``None`` when the alpha has no support.
    """
    mask = np.asarray(alpha) > threshold
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    corners = np.array(((xs.min() - 0.5, ys.min() - 0.5), (xs.max() + 0.5, ys.min() - 0.5),
                        (xs.max() + 0.5, ys.max() + 0.5), (xs.min() - 0.5, ys.max() + 0.5)), np.float64)
    value = _affine(inverse)
    mapped = corners @ value[:, :2].T + value[:, 2]
    extent = mapped.max(axis=0) - mapped.min(axis=0)
    return float(extent.min())


def fit_hires_to_frame(image: np.ndarray, alpha: np.ndarray, inverse_hr, *, target_ratio: float = 1.25):
    """Area-filter a hi-res crop that will be *shrunk* into the frame, so warpAffine does not alias.

    ``cv2.warpAffine`` has no mip-mapping: warping a 1024 px face onto a 250 px frame footprint would alias the very
    detail the enhancer produced. When the crop->frame scale is below 0.7 the crop and alpha are first reduced with
    ``INTER_AREA`` to about ``target_ratio`` x the frame footprint and the matrix is re-based accordingly.
    """
    import cv2
    matrix = _affine(inverse_hr)
    side = image.shape[0]
    scale = math.sqrt(abs(float(np.linalg.det(matrix[:, :2]))))
    if scale >= 0.7 or side < 16:
        return image, alpha, matrix
    reduced = max(16, int(round(side * scale * target_ratio)))
    if reduced >= side:
        return image, alpha, matrix
    return (cv2.resize(image, (reduced, reduced), interpolation=cv2.INTER_AREA),
            np.clip(cv2.resize(alpha, (reduced, reduced), interpolation=cv2.INTER_AREA), 0, 1),
            scale_matrix_input(matrix, reduced / side))


# ------------------------------------------------------------------------------------------ GPU (torch)
def _torch():
    import torch
    return torch


def bicubic_warp(image, destination_to_source, size, *, clamp: tuple[float, float] | None = (0.0, 1.0)):
    """Bicubic affine warp of an NCHW float image, ``grid_sample(mode='bicubic', align_corners=False)``.

    ``destination_to_source`` is (N, 2, 3) in pixel-index coordinates (pixel centres at integers, the OpenCV
    convention); zero outside the source. Bicubic overshoot is clamped to ``clamp`` unless ``None``.
    """
    torch = _torch()
    import torch.nn.functional as F
    matrix = destination_to_source
    if image.ndim != 4 or matrix.ndim != 3 or matrix.shape[1:] != (2, 3) or matrix.device != image.device:
        raise ValueError("Bicubic warp requires an NCHW image and a matching (N, 2, 3) matrix")
    height, width = size
    if height < 1 or width < 1:
        raise ValueError("Warp output dimensions must be positive")
    src_h, src_w = image.shape[-2:]
    dtype = torch.float32 if image.dtype not in (torch.float32, torch.float64) else image.dtype
    value = image.to(dtype)
    matrix = matrix.to(dtype)
    y, x = torch.meshgrid(torch.arange(height, device=image.device, dtype=dtype),
                          torch.arange(width, device=image.device, dtype=dtype), indexing="ij")
    sx = x[None] * matrix[:, 0, 0, None, None] + y[None] * matrix[:, 0, 1, None, None] + matrix[:, 0, 2, None, None]
    sy = x[None] * matrix[:, 1, 0, None, None] + y[None] * matrix[:, 1, 1, None, None] + matrix[:, 1, 2, None, None]
    grid = torch.stack(((2 * sx + 1) / src_w - 1, (2 * sy + 1) / src_h - 1), -1)
    if grid.shape[0] != value.shape[0]:
        grid = grid.expand(value.shape[0], -1, -1, -1)
    out = F.grid_sample(value, grid, mode="bicubic", padding_mode="zeros", align_corners=False)
    return out.clamp(*clamp) if clamp is not None else out


def warp_to_frame_roi(image, source_to_frame, frame_size: tuple[int, int], *, margin: int = 3):
    """Warp a hi-res NCHW tensor straight into the native-resolution frame ROI it covers.

    Returns ``(roi_tensor, (x0, y0, x1, y1))`` or ``None`` if it falls outside the frame. When the face is
    drawn much smaller than its enhanced tensor (shrink > ~1.5x) the source is first reduced with an antialiased
    bicubic resize so the final bicubic gather does not alias.
    """
    torch = _torch()
    import torch.nn.functional as F
    matrix = _affine(source_to_frame)
    src_h, src_w = image.shape[-2:]
    shrink = math.sqrt(abs(float(np.linalg.det(matrix[:, :2]))))
    if shrink < 0.66 and min(src_h, src_w) > 16:
        new_h = max(16, int(round(src_h * shrink * 1.5)))
        new_w = max(16, int(round(src_w * shrink * 1.5)))
        if new_h < src_h and new_w < src_w:
            image = F.interpolate(image.float(), size=(new_h, new_w), mode="bicubic", align_corners=False,
                                  antialias=True)
            matrix = scale_matrix_input(matrix, new_h / src_h)
            src_h, src_w = new_h, new_w
    box = mapped_bounds(matrix, (src_h, src_w), frame_size, margin)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    to_source = invert_affine_np(matrix)
    to_source[:, 2] += to_source[:, :2] @ np.array((x0, y0), np.float64)
    tensor = torch.as_tensor(to_source, device=image.device, dtype=torch.float32)[None]
    return bicubic_warp(image, tensor, (y1 - y0, x1 - x0)), box


def rgb_to_ycbcr(rgb):
    """Full-range BT.601, chroma centred on zero. NCHW."""
    torch = _torch()
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return torch.stack((0.299 * r + 0.587 * g + 0.114 * b,
                        -0.168736 * r - 0.331264 * g + 0.5 * b,
                        0.5 * r - 0.418688 * g - 0.081312 * b), 1)


def ycbcr_to_rgb(ycc):
    torch = _torch()
    y, cb, cr = ycc[:, 0], ycc[:, 1], ycc[:, 2]
    return torch.stack((y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb), 1).clamp(0, 1)


def _space(name: str):
    from roop_3d.gpu_compositor import lab_to_rgb, rgb_to_lab
    if name == "lab":
        return rgb_to_lab, lab_to_rgb, ((0.0, 100.0), (-128.0, 128.0), (-128.0, 128.0))
    if name == "ycbcr":
        return rgb_to_ycbcr, ycbcr_to_rgb, ((0.0, 1.0), (-0.5, 0.5), (-0.5, 0.5))
    raise ValueError("Colour space must be 'lab' or 'ycbcr'")


def weighted_moment_transfer(source, target, weight, *, space: str = "lab", strength: float = 1.0,
                             apply_weight=None, gain_limits: tuple[float, float] = (0.5, 2.0)):
    """Weighted mean/std (Reinhard) transfer. Statistics come only from ``weight`` (the skin class).

    Writes are modulated by ``apply_weight`` (default: ``weight``); an empty ``weight`` is an exact identity and no
    host synchronisation happens. The per-channel std ratio is limited to ``gain_limits`` so a tiny skin area
    cannot blow the contrast up.
    """
    torch = _torch()
    if not 0 <= strength <= 1:
        raise ValueError("Colour strength must be in 0..1")
    forward, inverse, _ = _space(space)
    w = weight.clamp(0, 1)
    a, b = forward(source), forward(target)
    count = w.sum((-2, -1), keepdim=True)
    safe = count.clamp_min(1e-6)

    def moments(value):
        mean = (value * w).sum((-2, -1), keepdim=True) / safe
        var = ((value - mean).square() * w).sum((-2, -1), keepdim=True) / safe
        return mean, var.clamp_min(1e-8).sqrt()

    (am, ast), (bm, bst) = moments(a), moments(b)
    gain = (bst / ast.clamp_min(1e-4)).clamp(*gain_limits)
    matched = inverse((a - am) * gain + bm)
    write = (w if apply_weight is None else apply_weight.clamp(0, 1)) * strength
    return torch.where(count > 1e-6, source + (matched - source) * write, source)


def weighted_histogram_match(source, target, weight, *, space: str = "lab", strength: float = 1.0,
                             apply_weight=None, bins: int = 256):
    """Per-channel weighted CDF matching of ``source`` to ``target`` inside the skin class (GPU, sync-free)."""
    torch = _torch()
    if not 0 <= strength <= 1:
        raise ValueError("Colour strength must be in 0..1")
    forward, inverse, ranges = _space(space)
    w = weight.clamp(0, 1)
    a, b = forward(source), forward(target)
    flat_w = w.reshape(-1)
    total = flat_w.sum()
    channels = []
    for index, (low, high) in enumerate(ranges):
        sn = ((a[:, index] - low) / (high - low)).clamp(0, 1)
        tn = ((b[:, index] - low) / (high - low)).clamp(0, 1)
        weights = flat_w.new_zeros(bins)
        reference = flat_w.new_zeros(bins)
        weights.index_add_(0, (sn.reshape(-1) * (bins - 1)).round().long(), flat_w)
        reference.index_add_(0, (tn.reshape(-1) * (bins - 1)).round().long(), flat_w)
        cdf_s = weights.cumsum(0) / total.clamp_min(1e-6)
        cdf_t = (reference.cumsum(0) / total.clamp_min(1e-6)).contiguous()
        lut = torch.searchsorted(cdf_t, cdf_s.contiguous()).clamp(max=bins - 1).to(sn.dtype) / (bins - 1)
        position = sn * (bins - 1)
        lower = position.floor().long().clamp(0, bins - 1)
        upper = (lower + 1).clamp(max=bins - 1)
        fraction = position - lower.to(position.dtype)
        mapped = lut[lower] * (1 - fraction) + lut[upper] * fraction
        channels.append(mapped * (high - low) + low)
    matched = inverse(torch.stack(channels, 1))
    write = (w if apply_weight is None else apply_weight.clamp(0, 1)) * strength
    return torch.where(total > 1e-6, source + (matched - source) * write, source)


def match_skin_colour(source, target, skin, *, method: str = "reinhard_lab", strength: float = 1.0, apply_weight=None):
    """Dispatch a ``COLOR_METHODS`` entry. ``regional_reinhard`` uses the existing spatial-moment RCT."""
    if method not in COLOR_METHODS:
        raise ValueError(f"colour method must be one of {COLOR_METHODS}")
    if method == "regional_reinhard":
        from roop_3d.gpu_compositor import color_transfer
        return color_transfer(source, target, skin, strength=strength, regional=True)
    kind, _, space = method.partition("_")
    if kind == "histogram":
        return weighted_histogram_match(source, target, skin, space=space, strength=strength, apply_weight=apply_weight)
    return weighted_moment_transfer(source, target, skin, space=space, strength=strength, apply_weight=apply_weight)


def _blur5(value):
    import torch.nn.functional as F
    torch = _torch()
    kernel = value.new_tensor((1.0, 4.0, 6.0, 4.0, 1.0)) / 16.0
    channels = value.shape[1]
    horizontal = kernel.view(1, 1, 1, 5).expand(channels, 1, 1, 5)
    vertical = kernel.view(1, 1, 5, 1).expand(channels, 1, 5, 1)
    value = F.conv2d(F.pad(value, (2, 2, 0, 0), mode="replicate"), horizontal, groups=channels)
    return F.conv2d(F.pad(value, (0, 0, 2, 2), mode="replicate"), vertical, groups=channels)


def _down(value):
    return _blur5(value)[..., ::2, ::2]


def _up(value, like):
    import torch.nn.functional as F
    return F.interpolate(value, size=like.shape[-2:], mode="bilinear", align_corners=False)


def multiband_blend(source, target, alpha, *, levels: int = 4, weight: float = 1.0):
    """Multi-band Laplacian blend at the resolution of the inputs.

    Every band of ``source`` is mixed with the matching band of ``target`` under the Gaussian pyramid of ``alpha``.
    Where ``alpha`` is 1 the finest band is taken from ``source`` untouched, so pores, eyelashes and hairlines keep
    full contrast; only the low bands cross-fade over the feather. Exactly zero alpha stays the target.
    """
    torch = _torch()
    if not 0 <= weight <= 1:
        raise ValueError("Blend weight must be in 0..1")
    limit = max(1, int(math.floor(math.log2(max(2, min(source.shape[-2:]))))) - 2)
    levels = max(1, min(int(levels), limit))
    mask = alpha.clamp(0, 1) * weight
    gs, gt, gm = [source], [target], [mask]
    for _ in range(levels - 1):
        gs.append(_down(gs[-1]))
        gt.append(_down(gt[-1]))
        gm.append(_down(gm[-1]))
    result = gs[-1] * gm[-1] + gt[-1] * (1 - gm[-1])
    for index in range(levels - 2, -1, -1):
        ls = gs[index] - _up(gs[index + 1], gs[index])
        lt = gt[index] - _up(gt[index + 1], gt[index])
        result = _up(result, gs[index]) + ls * gm[index] + lt * (1 - gm[index])
    return torch.where(alpha > 0, result.clamp(0, 1), target)


def composite_hires_face(result, enhanced, alpha, hr_to_frame, *, skin=None, color_method: str = "reinhard_lab",
                         color_strength: float = 0.0, apply_weight=None, blend_weight: float = 0.0, levels: int = 4):
    """Warp ``enhanced`` (hi-res, aligned) into ``result`` (NCHW frame) in place and return the frame ROI box.

    1. enhanced + alpha (+ skin) are warped together, bicubic, straight to the native frame ROI via ``hr_to_frame``.
    2. Colour is matched against the *native* frame pixels under the skin class (LAB / YCbCr, mean-std or histogram).
    3. The face is blended with a multi-band Laplacian blend at ROI resolution (``blend_weight`` > 0) or a plain
       alpha mix.
    """
    torch = _torch()
    planes = [enhanced.float(), alpha.float()]
    if skin is not None:
        planes.append(skin.float())
        if apply_weight is not None:
            planes.append(apply_weight.float())
    warped = warp_to_frame_roi(torch.cat(planes, 1), hr_to_frame, result.shape[-2:])
    if warped is None:
        return None
    packed, (x0, y0, x1, y1) = warped
    face = packed[:, :3]
    matte = packed[:, 3:4].clamp(0, 1)
    target = result[..., y0:y1, x0:x1]
    if color_strength and skin is not None:
        face = match_skin_colour(face, target, packed[:, 4:5].clamp(0, 1), method=color_method, strength=color_strength,
                                 apply_weight=packed[:, 5:6].clamp(0, 1) if apply_weight is not None else None)
    if blend_weight > 0:
        mixed = multiband_blend(face, target, matte, levels=levels, weight=blend_weight)
    else:
        mixed = face * matte + target * (1 - matte)
    result[..., y0:y1, x0:x1] = mixed
    return x0, y0, x1, y1


# -------------------------------------------------------------------------------------- measurement helper
def laplacian_variance(image: np.ndarray) -> float:
    """Variance of the Laplacian of the luma plane: a standard sharpness / high-frequency-energy measure."""
    import cv2
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
