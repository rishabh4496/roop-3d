"""The spherical pose-bin grid and great-circle reference weighting (blueprint 03 §3.2).

Frame: *standard* pose frame of ``core.pose`` — yaw positive when the nose moves toward the image right, pitch positive
when the head looks **up**, roll irrelevant (it spins about the view direction). Callers holding legacy tuples
(``FaceView.pitch``: positive = down) must convert with ``core.pose.as_pose3d`` first.

Grid (``spherical_bin``): yaw lattice ``0, ±15, ..., ±90`` (13 columns) x pitch lattice ``-30, -15, 0, 15, 30`` (5 rows);
65 bins. A pose belongs to the bin whose lattice centre is nearest per axis (ties away from zero, inputs clamped to
the lattice range, so |yaw| > 90 and |pitch| > 30 fall into the outermost bin). Bin name: ``"yaw+45/pitch0"``,
``"yaw0/pitch-15"``.

Reference weights: ``w_k ∝ exp(-θ_k² / 2σ²)`` with θ_k the great-circle angle between the view direction of the query
pose and the centre of available bin ``k``, ``σ = 18°``; only the top-3 bins are kept and renormalised to sum 1.

Legacy grids that remain, deliberately (they encode *different* semantics or persisted formats, not this grid):

* 12 x 8 sphere-coverage grid (``COVERAGE_YAW_BINS`` x ``COVERAGE_PITCH_BINS``, :func:`coverage_bin`) — the ``.fsz``/viewer
  coverage percentage and ``bin_N`` keys already stored in FaceSet archives and consumed by the UI. It is now defined
  here once and re-exported by ``reconstruction.faceset_3d`` and ``pose_matrix`` (they previously duplicated it); its
  bin indices are unchanged. FaceSet coverage output additionally carries ``spherical_bin``.
* 3 x 3 yaw x pitch identity cells (``core.pose.pose_cell``, ``faceset.pose_matrix``) — Stage 0 frozen identity vectors,
  persisted in archives with their own edges.
* 5 x 3 target-lock banks (``identity.target_lock.pose_bucket``, 30° buckets over ±75°/±45°) — runtime identity banks;
  out-of-grid poses intentionally claim no coverage there.
* ``faceset.pose_selector.classify_pose_bin`` — left/center/right x up/center/down labels stored in ``FaceView.pose_bin``
  (legacy pitch frame).
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from roop_3d.core.errors import DomainError

YAW_CENTRES_DEG: tuple[int, ...] = tuple(range(-90, 91, 15))
PITCH_CENTRES_DEG: tuple[int, ...] = (-30, -15, 0, 15, 30)
STEP_DEG = 15
SIGMA_DEG = 18.0
TOP_K = 3
TOTAL_BINS = len(YAW_CENTRES_DEG) * len(PITCH_CENTRES_DEG)

# Legacy 12 x 8 coverage grid (yaw -90..90, pitch -60..60), shared by faceset_3d and pose_matrix.
COVERAGE_YAW_BINS, COVERAGE_PITCH_BINS = 12, 8
COVERAGE_TOTAL_BINS = COVERAGE_YAW_BINS * COVERAGE_PITCH_BINS


@dataclass(frozen=True)
class SphericalBin:
    index: int            # row-major: pitch_index * 13 + yaw_index
    yaw_deg: int
    pitch_deg: int
    name: str


def _label(value: int) -> str:
    return f"{value:+d}" if value else "0"


def bin_name(yaw_deg: int, pitch_deg: int) -> str:
    return f"yaw{_label(yaw_deg)}/pitch{_label(pitch_deg)}"


ALL_BINS: tuple[SphericalBin, ...] = tuple(
    SphericalBin(pi * len(YAW_CENTRES_DEG) + yi, yaw, pitch, bin_name(yaw, pitch))
    for pi, pitch in enumerate(PITCH_CENTRES_DEG) for yi, yaw in enumerate(YAW_CENTRES_DEG))
_BY_NAME: dict[str, SphericalBin] = {b.name: b for b in ALL_BINS}


def _finite_pair(yaw: float, pitch: float) -> tuple[float, float]:
    yaw, pitch = float(yaw), float(pitch)
    if not (math.isfinite(yaw) and math.isfinite(pitch)):
        raise DomainError("POSE_INVALID", "Pose bins need finite yaw and pitch", 422)
    return yaw, pitch


def _nearest(value: float, limit: int) -> int:
    clamped = max(-float(limit), min(float(limit), value))
    return int(math.copysign(math.floor(abs(clamped) / STEP_DEG + 0.5), clamped)) * STEP_DEG


def classify(yaw: float, pitch: float) -> SphericalBin:
    """Deterministic bin of a standard-frame (yaw, pitch) in degrees."""
    yaw, pitch = _finite_pair(yaw, pitch)
    yaw_c = _nearest(yaw, YAW_CENTRES_DEG[-1])
    pitch_c = _nearest(pitch, PITCH_CENTRES_DEG[-1])
    return ALL_BINS[PITCH_CENTRES_DEG.index(pitch_c) * len(YAW_CENTRES_DEG) + YAW_CENTRES_DEG.index(yaw_c)]


def view_direction(yaw: float, pitch: float) -> np.ndarray:
    """Unit direction of the nose in the standard frame (x right, y down, z toward the viewer): ``Rx(pitch) Ry(yaw) ẑ``."""
    yaw, pitch = _finite_pair(yaw, pitch)
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array([math.sin(y), -math.sin(p) * math.cos(y), math.cos(p) * math.cos(y)], np.float64)


def great_circle_deg(yaw_a: float, pitch_a: float, yaw_b: float, pitch_b: float) -> float:
    """Angle in degrees between the view directions of two poses."""
    cosine = float(np.clip(view_direction(yaw_a, pitch_a) @ view_direction(yaw_b, pitch_b), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _resolve(bin_like: SphericalBin | str) -> SphericalBin:
    if isinstance(bin_like, SphericalBin):
        return bin_like
    try:
        return _BY_NAME[bin_like]
    except (KeyError, TypeError):
        raise DomainError("POSE_BIN_UNKNOWN", f"{bin_like!r} is not a spherical bin name", 422) from None


def reference_weights(yaw: float, pitch: float, available_bins: Iterable[SphericalBin | str], *,
                      sigma_deg: float = SIGMA_DEG, top_k: int = TOP_K) -> dict[str, float]:
    """Great-circle Gaussian weights over the ``top_k`` nearest *available* bins; the values sum to 1.

    Returns ``{bin_name: weight}`` ordered by decreasing weight (ties broken by bin index). Raises ``POSE_BINS_EMPTY``
    when no bin is available — there is nothing to weight and no weight is invented."""
    if sigma_deg <= 0 or top_k < 1:
        raise DomainError("POSE_WEIGHTS_PARAMS", "sigma_deg must be > 0 and top_k >= 1", 422)
    unique = {b.index: b for b in (_resolve(item) for item in available_bins)}
    if not unique:
        raise DomainError("POSE_BINS_EMPTY", "No spherical bin is available to weight", 422)
    query = view_direction(yaw, pitch)
    scored = []
    for b in unique.values():
        cosine = float(np.clip(query @ view_direction(b.yaw_deg, b.pitch_deg), -1.0, 1.0))
        scored.append((math.degrees(math.acos(cosine)), b))
    scored.sort(key=lambda item: (item[0], item[1].index))
    kept = scored[:top_k]
    logits = np.array([-(theta ** 2) / (2.0 * sigma_deg ** 2) for theta, _ in kept])
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return {b.name: float(w) for (_, b), w in zip(kept, weights)}


def coverage_bin(yaw: float, pitch: float) -> int:
    """Index in the legacy 12 x 8 coverage grid (yaw clamped to ±90, pitch to ±60); unchanged from the original
    ``faceset_3d.pose_bin``. The caller's pitch frame is preserved (the grid is symmetric)."""
    yi = min(COVERAGE_YAW_BINS - 1, max(0, int((min(90.0, max(-90.0, yaw)) + 90.0) / 180.0 * COVERAGE_YAW_BINS)))
    pi = min(COVERAGE_PITCH_BINS - 1, max(0, int((min(60.0, max(-60.0, pitch)) + 60.0) / 120.0 * COVERAGE_PITCH_BINS)))
    return pi * COVERAGE_YAW_BINS + yi
