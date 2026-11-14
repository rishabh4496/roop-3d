"""106-point landmark block layout and a geometric plausibility check (pure geometry, no image warping).

Layout: contour 0-32, eye 33-42, brow 43-51, mouth 52-71, nose 72-86, eye 87-96, brow 97-105 (standard 2d106; block sizes
sum to exactly 106). The project does not trust a detector's ordering blindly, so ``layout_is_plausible`` checks the
geometry (brows above eyes above nose above mouth, in the face's own rotated frame). Landmark measurement code
(expression measurement, identity lock, scene adaptation) uses it; the former 2D thin-plate-spline warp was removed.
"""
from __future__ import annotations

import numpy as np

BLOCKS: dict[str, range] = {"contour": range(0, 33), "eye_a": range(33, 43), "brow_a": range(43, 52), "mouth": range(52, 72),
                            "nose": range(72, 87), "eye_b": range(87, 97), "brow_b": range(97, 106)}
EXPRESSION_BLOCKS = ("mouth", "eye_a", "eye_b", "brow_a", "brow_b")


def group(points: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(points, np.float64)[list(BLOCKS[name])]


def layout_is_plausible(points106: np.ndarray) -> bool:
    """Geometric check that the points follow the assumed block layout (in the face's own rotated frame)."""
    points = np.asarray(points106, np.float64)
    if points.shape != (106, 2) or not np.isfinite(points).all():
        return False
    centroids = {name: group(points, name).mean(axis=0) for name in BLOCKS}
    eye_mid = (centroids["eye_a"] + centroids["eye_b"]) / 2
    across = centroids["eye_b"] - centroids["eye_a"]
    spread = float(np.linalg.norm(across))
    if spread < 1e-6:
        return False
    across = across / spread
    down = np.array((-across[1], across[0]))
    if np.dot(centroids["mouth"] - eye_mid, down) <= 0:
        down, across = -down, -across  # eye_a/eye_b may be either side of the face
    height = {name: float(np.dot(c - eye_mid, down)) for name, c in centroids.items()}
    brows = (height["brow_a"] + height["brow_b"]) / 2
    contour = group(points, "contour")
    width = float(np.ptp(contour @ across))
    mouth_width = float(np.ptp(group(points, "mouth") @ across))
    return bool(brows < -0.05 * spread < 0.05 * spread < height["nose"] < height["mouth"]
                and 0.2 * width < spread < 0.9 * width and 0.1 * width < mouth_width < 0.85 * width
                and float(np.ptp(contour @ down)) > 0.3 * width)
