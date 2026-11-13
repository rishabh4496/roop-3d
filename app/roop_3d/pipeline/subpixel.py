"""Sub-pixel crop and warp: float matrices, Lanczos4/cubic resampling, no integer bounding-box rounding (G6).

A crop whose box is rounded to integers moves the face by up to half a pixel per frame (visible as shimmer once the
crop is later warped back), and bilinear resampling low-passes every warp. Here the source-to-destination matrix stays
float64 end to end: the destination-to-source sampling grid is built from its inverse in float64 and handed to
``cv2.remap`` with ``INTER_LANCZOS4`` (default) or ``INTER_CUBIC``. Conventions match ``cv2.warpAffine`` without
``WARP_INVERSE_MAP``: ``matrix`` maps *source* coordinates to *destination* coordinates, pixel centres at integers.

Properties: an identity matrix reproduces the input exactly (a Lanczos/cubic kernel sampled at integer offsets is a
delta); strongly down-scaling transforms (more than 1.25x reduction) are Gaussian pre-filtered so ``remap`` does not
alias; Lanczos overshoot on float input is left to the caller (uint8 output saturates). Sampling grids are float32
(the ``remap`` map type), exact to ~1e-3 px up to 8k frames.
"""
from __future__ import annotations

import cv2
import numpy as np

from roop_3d.core.errors import DomainError

_INTERPOLATION = {"lanczos4": cv2.INTER_LANCZOS4, "cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR}
_BORDER = {"replicate": cv2.BORDER_REPLICATE, "constant": cv2.BORDER_CONSTANT, "reflect101": cv2.BORDER_REFLECT_101}
_ANTIALIAS_REDUCTION = 1.25


def _validate(image: np.ndarray, size: tuple[int, int], interpolation: str, border: str) -> tuple[int, int, int, int]:
    array = np.asarray(image)
    width, height = int(size[0]), int(size[1])
    if (array.ndim not in (2, 3) or array.size == 0 or width <= 0 or height <= 0
            or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all()):
        raise DomainError("SUBPIXEL_WARP_INVALID", "Image or output size is invalid", 422)
    if interpolation not in _INTERPOLATION or border not in _BORDER:
        raise DomainError("SUBPIXEL_WARP_INVALID", f"interpolation must be one of {tuple(_INTERPOLATION)} and border one of {tuple(_BORDER)}", 422)
    return width, height, interpolation, border


def _prefilter(image: np.ndarray, linear: np.ndarray) -> np.ndarray:
    """Gaussian low-pass matched to the strongest reduction of ``linear`` (source -> destination 2x2 part)."""
    smallest = float(np.linalg.svd(linear, compute_uv=False)[-1])
    reduction = 1.0 / max(smallest, 1e-6)
    if reduction <= _ANTIALIAS_REDUCTION:
        return image
    return cv2.GaussianBlur(image, (0, 0), 0.4 * reduction, borderType=cv2.BORDER_REPLICATE)


def _remap(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, interpolation: str, border: str,
           border_value) -> np.ndarray:
    return cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), _INTERPOLATION[interpolation],
                     borderMode=_BORDER[border], borderValue=border_value)


def warp_affine(image: np.ndarray, matrix: np.ndarray, size: tuple[int, int], *, interpolation: str = "lanczos4",
                border: str = "replicate", border_value: float | tuple = 0) -> np.ndarray:
    """Warp ``image`` by the 2x3 source->destination ``matrix`` into a ``size = (width, height)`` canvas."""
    width, height, interpolation, border = _validate(image, size, interpolation, border)
    m = np.asarray(matrix, np.float64)
    if m.shape != (2, 3) or not np.isfinite(m).all() or abs(np.linalg.det(m[:, :2])) < 1e-10:
        raise DomainError("SUBPIXEL_WARP_INVALID", "Affine matrix must be finite, 2x3 and non-singular", 422)
    inverse = cv2.invertAffineTransform(m)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    map_x = inverse[0, 0] * grid_x + inverse[0, 1] * grid_y + inverse[0, 2]
    map_y = inverse[1, 0] * grid_x + inverse[1, 1] * grid_y + inverse[1, 2]
    return _remap(_prefilter(np.asarray(image), m[:, :2]), map_x, map_y, interpolation, border, border_value)


def warp_perspective(image: np.ndarray, homography: np.ndarray, size: tuple[int, int], *,
                     interpolation: str = "lanczos4", border: str = "replicate",
                     border_value: float | tuple = 0) -> np.ndarray:
    """Warp ``image`` by the 3x3 source->destination ``homography``; destination pixels that map behind the camera
    plane (non-positive homogeneous ``w``) are filled with ``border_value``."""
    width, height, interpolation, border = _validate(image, size, interpolation, border)
    h = np.asarray(homography, np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise DomainError("SUBPIXEL_WARP_INVALID", "Homography must be finite, 3x3 and non-singular", 422)
    inverse = np.linalg.inv(h)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    denominator = inverse[2, 0] * grid_x + inverse[2, 1] * grid_y + inverse[2, 2]
    behind = denominator <= 1e-9
    safe = np.where(behind, 1.0, denominator)
    map_x = np.where(behind, -1e6, (inverse[0, 0] * grid_x + inverse[0, 1] * grid_y + inverse[0, 2]) / safe)
    map_y = np.where(behind, -1e6, (inverse[1, 0] * grid_x + inverse[1, 1] * grid_y + inverse[1, 2]) / safe)
    # Local reduction at the centre drives the pre-filter (a homography is locally affine).
    centre = np.array([width / 2.0, height / 2.0, 1.0])
    p = inverse @ centre
    jacobian = (inverse[:2, :2] - np.outer(p[:2] / p[2], inverse[2, :2])) / p[2]
    destination_to_source = jacobian if np.isfinite(jacobian).all() else inverse[:2, :2]
    source_to_destination = np.linalg.inv(destination_to_source) if abs(np.linalg.det(destination_to_source)) > 1e-12 \
        else np.eye(2)
    result = _remap(_prefilter(np.asarray(image), source_to_destination), map_x, map_y, interpolation,
                    "constant" if behind.any() else border, border_value)
    return result


def crop_box(image: np.ndarray, box: tuple[float, float, float, float], size: tuple[int, int], *,
             interpolation: str = "lanczos4", border: str = "replicate") -> np.ndarray:
    """Resample the float box ``(x0, y0, x1, y1)`` (pixel-edge coordinates, so the box ``(0, 0, w, h)`` is the whole
    image) to ``size = (width, height)`` without rounding the box to integers."""
    x0, y0, x1, y1 = (float(v) for v in box)
    width, height = int(size[0]), int(size[1])
    if not np.isfinite((x0, y0, x1, y1)).all() or x1 - x0 <= 0 or y1 - y0 <= 0 or width <= 0 or height <= 0:
        raise DomainError("SUBPIXEL_WARP_INVALID", "Crop box must be finite with positive extent", 422)
    sx, sy = width / (x1 - x0), height / (y1 - y0)
    # Pixel centres sit at integer coordinates: edge coordinate e maps to centre coordinate e - 0.5.
    matrix = np.array([[sx, 0.0, (0.5 - x0) * sx - 0.5], [0.0, sy, (0.5 - y0) * sy - 0.5]])
    return warp_affine(image, matrix, (width, height), interpolation=interpolation, border=border)
