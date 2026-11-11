"""Pure image / geometry primitives behind :mod:`roop_3d.pipeline.photorealistic_swap_core`.

Everything here is deterministic numpy/OpenCV on frame-sized (or ROI-sized) arrays. No model is loaded and no state is kept,
so each function has a numeric contract that the unit tests pin (identity when lights match, exact reconstruction at
alpha 0/1, fold-free fields, ...).

Conventions: images are ``uint8`` BGR unless a name says ``linear``/``float``; points are ``(N, 2)`` ``(x, y)`` pixels; masks are
``float32`` in ``0..1``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.reconstruction.expression_tps import BLOCKS, group
from roop_3d.reconstruction.tps_fallback import ThinPlateSplineWarper

LUMA_BGR = np.array([0.114, 0.587, 0.299], np.float32)


# --------------------------------------------------------------------------------------------- 106-point head geometry
@dataclass(frozen=True)
class HeadFrame:
    """The face's own axes, derived from the 106 landmarks (works for a rolled head)."""
    across: np.ndarray        # unit vector, subject-left to subject-right in image terms
    down: np.ndarray          # unit vector, brow -> chin
    brow_mid: np.ndarray
    chin: np.ndarray
    height: float             # brow-to-chin distance, px
    width: float              # contour extent along ``across``, px


def head_frame(points106: np.ndarray) -> HeadFrame:
    """Axes of a 106-point face. Raises ``ValueError`` for a layout the geometry cannot support."""
    points = np.asarray(points106, np.float64)
    if points.shape != (106, 2) or not np.isfinite(points).all():
        raise ValueError("106 finite landmarks required")
    eye_a, eye_b = group(points, "eye_a").mean(axis=0), group(points, "eye_b").mean(axis=0)
    across = eye_b - eye_a
    spread = float(np.linalg.norm(across))
    if spread < 1e-6:
        raise ValueError("degenerate eye line")
    across = across / spread
    down = np.array((-across[1], across[0]))
    mouth = group(points, "mouth").mean(axis=0)
    if np.dot(mouth - (eye_a + eye_b) / 2, down) <= 0:
        down, across = -down, -across
    contour = group(points, "contour")
    brow_mid = np.vstack((group(points, "brow_a"), group(points, "brow_b"))).mean(axis=0)
    chin = contour[int(np.argmax(contour @ down))]
    height = float(np.dot(chin - brow_mid, down))
    width = float(np.ptp(contour @ across))
    if height <= 1e-6 or width <= 1e-6:
        raise ValueError("degenerate face extent")
    return HeadFrame(across, down, brow_mid, chin, height, width)


def forehead_arc(points106: np.ndarray, lift: float, samples: int = 15) -> np.ndarray:
    """Quadratic arc over the brows from one jaw end to the other, peaking ``lift * brow-to-chin`` above the brow line.

    The 106 layout stops at the eyebrows; this heuristic stands in for the missing forehead and is applied identically to the
    target and to the (aligned) source, so its bias cancels in the *difference* the swap acts on.
    """
    frame = head_frame(points106)
    contour = group(points106, "contour")
    along = contour @ frame.across
    a, b = contour[int(np.argmin(along))], contour[int(np.argmax(along))]
    peak = frame.brow_mid - frame.down * lift * frame.height
    control = 2.0 * peak - 0.5 * (a + b)
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return ((1 - t) ** 2) * a + 2 * (1 - t) * t * control + (t ** 2) * b


def head_polygon(points106: np.ndarray, lift: float) -> np.ndarray:
    """Convex head outline: jaw contour + forehead arc."""
    cloud = np.vstack((group(points106, "contour"), forehead_arc(points106, lift))).astype(np.float32)
    return cv2.convexHull(cloud)[:, 0, :].astype(np.float64)


def polygon_mask(polygon: np.ndarray, shape: tuple[int, int], feather: float = 0.0) -> np.ndarray:
    """Anti-aliased polygon fill, optionally feathered with a Gaussian."""
    scale = 4  # supersample the edge
    h, w = shape
    canvas = np.zeros((h * scale, w * scale), np.uint8)
    cv2.fillPoly(canvas, [np.round(np.asarray(polygon, np.float64) * scale).astype(np.int32)], 255, lineType=cv2.LINE_8)
    mask = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return np.clip(mask, 0.0, 1.0)


def similarity_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 2x3 similarity (rotation + uniform scale + translation) taking ``src`` onto ``dst`` (Umeyama)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[0] < 2:
        raise ValueError("similarity needs matching (N>=2, 2) point sets")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    var = (a ** 2).sum() / len(src)
    if var < 1e-12:
        raise ValueError("degenerate source points")
    u, s, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1, 1] = -1
    rot = u @ d @ vt
    scale = float(np.trace(np.diag(s) @ d) / var)
    shift = mu_d - scale * rot @ mu_s
    return np.hstack((scale * rot, shift[:, None]))


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64)
    return points @ np.asarray(matrix, np.float64)[:, :2].T + np.asarray(matrix, np.float64)[:, 2]


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(np.asarray(matrix, np.float64))


def compose_affine(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """``outer ∘ inner`` as a 2x3 matrix."""
    return (np.vstack((outer, (0, 0, 1))) @ np.vstack((inner, (0, 0, 1))))[:2]


def inner_anchor_indices() -> np.ndarray:
    """Landmarks that carry pose/scale but not jaw shape: eyes, nose and mouth (no contour, no brows)."""
    return np.array([i for name in ("eye_a", "eye_b", "nose", "mouth") for i in BLOCKS[name]])


# ------------------------------------------------------------------------------------------------ dense TPS displacement
def tps_flow(from_points: np.ndarray, to_points: np.ndarray, shape: tuple[int, int], *, grid_max_side: int = 96,
             min_jacobian: float = 0.25) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Backward sampling maps ``(map_x, map_y)`` that carry content at ``from_points`` onto ``to_points``.

    The thin-plate spline is solved on a coarse grid (cost is points x grid cells, not points x pixels) and upsampled. The ROI
    border is pinned so the warp cannot leak outward. If the field would fold (determinant below ``min_jacobian``) the
    displacement is scaled down by bisection. Returns ``(map_x, map_y, kept_fraction, min_jacobian)``.
    """
    h, w = int(shape[0]), int(shape[1])
    src, dst = np.asarray(from_points, np.float64), np.asarray(to_points, np.float64)
    border = _border(w, h)
    sources = np.vstack((src, border))
    targets = np.vstack((dst, border))
    cell = max(1.0, max(h, w) / grid_max_side)
    gh, gw = max(4, int(np.ceil(h / cell))), max(4, int(np.ceil(w / cell)))
    # Corner-aligned sample grid in pixel-INDEX coordinates (what cv2.remap consumes): the first and last samples sit on
    # pixel 0 and pixel w-1, so the upsample below is exact at the borders and an identity field stays an identity.
    gx = np.linspace(0.0, w - 1.0, gw)
    gy = np.linspace(0.0, h - 1.0, gh)
    grid = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
    norm = np.array((max(w, 1), max(h, 1)), np.float64)
    warper = ThinPlateSplineWarper(targets / norm, sources / norm, regularization=1e-9)
    mapped = (warper.transform_points(grid / norm) * norm).reshape(gh, gw, 2)
    identity = grid.reshape(gh, gw, 2)

    step_x, step_y = (w - 1.0) / (gw - 1), (h - 1.0) / (gh - 1)

    def min_det(fraction: float) -> float:
        field = identity + fraction * (mapped - identity)
        dxdx = np.gradient(field[..., 0], step_x, axis=1)
        dxdy = np.gradient(field[..., 0], step_y, axis=0)
        dydx = np.gradient(field[..., 1], step_x, axis=1)
        dydy = np.gradient(field[..., 1], step_y, axis=0)
        return float((dxdx * dydy - dxdy * dydx).min())

    kept = 1.0
    determinant = min_det(1.0)
    if determinant < min_jacobian:
        lo, hi = 0.0, 1.0
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if min_det(mid) >= min_jacobian else (lo, mid)
        kept, determinant = lo, min_det(lo)
    field = identity + kept * (mapped - identity)
    # corner-aligned bilinear upsample: dense pixel (x, y) reads coarse sample (x * (gw-1)/(w-1), y * (gh-1)/(h-1))
    u = np.tile((np.arange(w, dtype=np.float32) * ((gw - 1) / max(w - 1, 1)))[None, :], (h, 1))
    v = np.tile((np.arange(h, dtype=np.float32) * ((gh - 1) / max(h - 1, 1)))[:, None], (1, w))
    field32 = field.astype(np.float32)
    map_x = cv2.remap(np.ascontiguousarray(field32[..., 0]), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    map_y = cv2.remap(np.ascontiguousarray(field32[..., 1]), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return map_x, map_y, float(kept), float(determinant)


def _border(width: int, height: int, per_side: int = 12) -> np.ndarray:
    xs, ys = np.linspace(0.0, width - 1.0, per_side + 1), np.linspace(0.0, height - 1.0, per_side + 1)
    return np.array([(x, 0.0) for x in xs] + [(x, height - 1.0) for x in xs]
                    + [(0.0, y) for y in ys[1:-1]] + [(width - 1.0, y) for y in ys[1:-1]], np.float64)


# ------------------------------------------------------------------------------------------------- SH / quotient image
def silhouette_normals(mask: np.ndarray) -> np.ndarray:
    """Unit normals ``(H, W, 3)`` of an ellipsoid-like bump whose footprint is ``mask`` (``z`` toward the camera).

    Height is ``sqrt(1 - (1 - d/d_max)^2)`` of the interior distance ``d``, so the rim is steep and the middle flat. It is a
    stand-in for the real surface, not a measurement; the caller reports it as such. Outside the mask the normal faces the camera.
    """
    inside = np.asarray(mask, np.float32) > 0.5
    normals = np.zeros((*inside.shape, 3), np.float32)
    normals[..., 2] = 1.0
    if int(inside.sum()) < 16:
        return normals
    ys, xs = np.nonzero(inside)
    points = np.stack((xs, ys), 1).astype(np.float64)
    centre = points.mean(0)
    covariance = np.cov((points - centre).T) + np.eye(2) * 1e-6
    # A solid ellipse has variance a^2 / 4 along each axis; the 1.04 keeps the whole mask inside the ellipsoid.
    inverse = np.linalg.inv(covariance * 4.0 * 1.04 ** 2)
    eigenvalues = np.linalg.eigvalsh(covariance)
    depth = 0.9 * float(np.sqrt(eigenvalues.max() * 4.0))        # half the head's depth ~ its larger half-extent
    yy, xx = np.mgrid[:inside.shape[0], :inside.shape[1]].astype(np.float64)
    dx, dy = xx - centre[0], yy - centre[1]
    q = inverse[0, 0] * dx * dx + 2 * inverse[0, 1] * dx * dy + inverse[1, 1] * dy * dy
    height = (depth * np.sqrt(np.clip(1.0 - np.minimum(q, 0.985), 0.0, 1.0))).astype(np.float32)
    dz_dy, dz_dx = np.gradient(height)
    n = np.stack((-dz_dx, -dz_dy, np.ones_like(dz_dx)), -1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    normals[inside] = n[inside]
    return normals


def inside_ramp(mask: np.ndarray, width_px: float) -> np.ndarray:
    """0 on and outside the mask boundary, rising to 1 ``width_px`` inside it (smoothstep): a feather that never leaks outward."""
    distance = cv2.distanceTransform((np.asarray(mask, np.float32) > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    t = np.clip(distance / max(float(width_px), 1e-3), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def quotient_ratio(shading_target: np.ndarray, shading_source: np.ndarray, *, epsilon: float = 1e-4,
                   clip: tuple[float, float] = (0.4, 2.2), sigma: float = 3.0) -> np.ndarray:
    """``R_sh = Render_SH(L_target) / (Render_SH(L_source) + eps)`` per channel, smoothed so it can never carry pore detail.

    Both inputs are ``(H, W, 3)`` irradiance maps of the SAME geometry; the albedo cancels, which is what makes this a
    quotient image. The result is clipped so a mis-estimated light cannot blow out a shadow or a highlight.
    """
    ratio = np.asarray(shading_target, np.float32) / (np.asarray(shading_source, np.float32) + np.float32(epsilon))
    if sigma > 0:
        ratio = cv2.GaussianBlur(ratio, (0, 0), sigma)
    return np.clip(ratio, clip[0], clip[1]).astype(np.float32)


def apply_ratio(linear_bgr: np.ndarray, ratio: np.ndarray, strength: float) -> np.ndarray:
    """Multiply linear-light pixels by ``1 + strength * (R - 1)``."""
    return np.clip(np.asarray(linear_bgr, np.float32) * (1.0 + float(strength) * (ratio - 1.0)), 0.0, 1.0)


# ------------------------------------------------------------------------------------------------------ pyramid blend
def laplacian_pyramid_blend(foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray, levels: int = 4) -> np.ndarray:
    """Blend ``foreground`` over ``background`` with ``levels`` Laplacian bands (+ the Gaussian residual).

    Each band is mixed with the mask at the same scale, so low frequencies (colour, shading) cross the seam over a wide
    distance and high frequencies (pores, edges) over a narrow one. ``alpha = 0`` returns the background and ``alpha = 1`` the
    foreground, up to float rounding.
    """
    fg = np.asarray(foreground, np.float32)
    bg = np.asarray(background, np.float32)
    if fg.shape != bg.shape or fg.ndim != 3:
        raise ValueError("foreground and background must be matching H x W x C images")
    mask = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    if mask.ndim == 2:
        mask = mask[:, :, None]
    if mask.shape[:2] != fg.shape[:2]:
        raise ValueError("alpha must match the image size")
    levels = int(max(1, min(levels, int(np.log2(min(fg.shape[:2]))) - 1)))

    def pyramid(image: np.ndarray) -> list[np.ndarray]:
        out = [image]
        for _ in range(levels):
            out.append(_keep_channels(cv2.pyrDown(out[-1]), image))
        return out

    gp_fg, gp_bg, gp_mask = pyramid(fg), pyramid(bg), pyramid(mask)

    def laplacian(gp: list[np.ndarray]) -> list[np.ndarray]:
        bands = [gp[i] - _keep_channels(cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0])), gp[i]) for i in range(levels)]
        return bands + [gp[levels]]                      # finest ... coarsest, then the Gaussian residual

    lp_fg, lp_bg = laplacian(gp_fg), laplacian(gp_bg)
    mixed = [lf * gm + lb * (1.0 - gm) for lf, lb, gm in zip(lp_fg, lp_bg, gp_mask)]
    out = mixed[levels]
    for i in range(levels - 1, -1, -1):
        out = _keep_channels(cv2.pyrUp(out, dstsize=(mixed[i].shape[1], mixed[i].shape[0])), mixed[i]) + mixed[i]
    return np.clip(out, 0, 255).astype(np.uint8) if foreground.dtype == np.uint8 else out


def _keep_channels(result: np.ndarray, like: np.ndarray) -> np.ndarray:
    """OpenCV drops the channel axis of single-channel images; put it back."""
    return result[:, :, None] if like.ndim == 3 and result.ndim == 2 else result


# ------------------------------------------------------------------------------------------ optics: focus and grain
def laplacian_variance(luma: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian (a standard sharpness proxy); 0 when no pixels qualify."""
    response = cv2.Laplacian(np.asarray(luma, np.float32), cv2.CV_32F, ksize=3)
    if mask is not None:
        selected = np.asarray(mask) > 0.5
        if int(selected.sum()) < 64:
            return 0.0
        response = response[selected]
    return float(np.var(response))


def luma(image_bgr: np.ndarray) -> np.ndarray:
    return np.asarray(image_bgr, np.float32) @ LUMA_BGR


def noise_sigma(luma_plane: np.ndarray, mask: np.ndarray | None = None, *, blur: float = 1.2) -> float:
    """Robust per-pixel noise std (8-bit levels): MAD of the high-pass residual. 0 when too few pixels qualify."""
    residual = np.asarray(luma_plane, np.float32) - cv2.GaussianBlur(np.asarray(luma_plane, np.float32), (0, 0), blur)
    if mask is not None:
        selected = np.asarray(mask) > 0.5
        if int(selected.sum()) < 128:
            return 0.0
        residual = residual[selected]
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))))


def match_focus(layer: np.ndarray, alpha: np.ndarray, target_variance: float, *, max_sigma: float = 1.4,
                tolerance: float = 0.1) -> tuple[np.ndarray, float]:
    """Soften ``layer`` just enough that its Laplacian variance over ``alpha`` matches the target camera's; never sharpens.

    Returns the layer and the Gaussian sigma applied (0 when none was needed).
    """
    plane = luma(layer)
    current = laplacian_variance(plane, alpha)
    if target_variance <= 0 or current <= target_variance * (1.0 + tolerance):
        return layer, 0.0
    best_sigma, best_gap = 0.0, abs(current - target_variance)
    for sigma in np.arange(0.25, max_sigma + 1e-6, 0.15):
        softened = cv2.GaussianBlur(plane, (0, 0), float(sigma))
        gap = abs(laplacian_variance(softened, alpha) - target_variance)
        if gap < best_gap:
            best_sigma, best_gap = float(sigma), gap
    if best_sigma <= 0:
        return layer, 0.0
    softened = cv2.GaussianBlur(np.asarray(layer, np.float32), (0, 0), best_sigma)
    return np.clip(softened, 0, 255).astype(layer.dtype), best_sigma


def add_grain(layer: np.ndarray, alpha: np.ndarray, extra_sigma: float, seed: int) -> np.ndarray:
    """Add zero-mean Gaussian luminance grain of std ``extra_sigma`` levels where ``alpha`` is set (seeded: reproducible)."""
    if extra_sigma <= 0.05:
        return layer
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    noise = rng.normal(0.0, extra_sigma, layer.shape[:2]).astype(np.float32)
    weight = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    return np.clip(np.asarray(layer, np.float32) + (noise * weight)[:, :, None], 0, 255).astype(layer.dtype)


def high_pass(image: np.ndarray, sigma: float) -> np.ndarray:
    value = np.asarray(image, np.float32)
    return value - cv2.GaussianBlur(value, (0, 0), sigma)
