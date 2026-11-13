"""Metric 6-DoF head pose from dense landmarks by RANSAC PnP + LM refinement (blueprint 03 §3.1).

Model: ``x_i ≃ K (R X_i + t)`` with ``X_i`` the repo's canonical head landmarks (``tps_fallback.get_canonical_3d_landmarks``
or any caller-supplied table such as the MakeHuman 68-point correspondence) in metres. Those tables use the repo
frame (x right, y down, **z toward the viewer**, nose at +z); this solver reflects depth (``z -> -z``) to obtain the
OpenCV head frame (z away from the camera, nose at -z) so that ``R = I`` is a frontal face. ``HeadPose6DoF.R`` and ``t``
are in that OpenCV frame. ``yaw/pitch/roll`` are reported in the *standard* frame of ``core.pose`` (``R_std = F R F``,
``F = diag(1, 1, -1)``; ``R_std = Rz(roll) Rx(pitch) Ry(yaw)``; yaw + = nose to image right, pitch + = looking up), which is
the repo's single Euler convention (the blueprint's ``R_y R_x R_z`` spelling is for the OpenCV frame; angles agree to
numerical precision for roll-free poses and are defined by ``core.pose`` for everything else).

Pipeline: assumed 35 mm-equivalent focal (``f = 35/36 · max(W, H)``) unless intrinsics are supplied ->
``solvePnPRansac`` (SQPNP, threshold ``max(3 px, 2 % of the landmark extent)``, confidence 0.99) -> ``solvePnPRefineLM`` on
the inliers -> ``t_z > 0`` and in-front checks -> flip resolution: the mirrored-depth hypothesis (seed ``F R F``: the
image-plane block of ``R`` is kept, depth couplings are negated; exactly the second solution of a flat model) is refined by
LM and the pose with the lower ``reproj_rms + λ·angle(R, R_prev)`` wins, among hypotheses whose nose points toward the camera
(a detected face is by construction front-facing, within a 15° margin past exact profile; a back-facing solution is only used when no
front-facing hypothesis exists).

Limits: the canonical model is an anthropometric prior, not the subject's own face, so absolute angles carry that model
error on real photographs; synthetic accuracy bounds are in ``tests/test_w2_3d_extreme_pose.py``. The solver never fakes
a pose: failure raises ``DomainError('POSE_PNP_FAILED' | 'POSE_LANDMARK_COUNT' | 'POSE_LANDMARKS_INVALID')``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.pose import euler_from_rotation
from roop_3d.reconstruction import pose_bins
from roop_3d.reconstruction.geometry_types import CameraIntrinsics, rotation_matrix_to_quaternion
from roop_3d.reconstruction.tps_fallback import get_canonical_3d_landmarks

SOURCE_DENSE = "pnp_dense"
SOURCE_5PT = "pnp_5pt"
EQUIV_FOCAL_MM, EQUIV_SENSOR_MM = 35.0, 36.0
RANSAC_THRESHOLD_PX = 3.0
RANSAC_CONFIDENCE = 0.99
FLIP_PRIOR_PX_PER_DEG = 0.1          # lambda of the temporal prior (reprojection px per degree of rotation change)
MIN_RANSAC_POINTS = 10
MIN_INLIER_FRACTION = 0.5
MIN_INLIERS = 5
_FLIP = np.diag([1.0, 1.0, -1.0])
_DISTINCT_HYPOTHESIS_DEG = 5.0
_FACING_MARGIN = math.sin(math.radians(15.0))


@dataclass(frozen=True)
class HeadPose6DoF:
    """Rigid head pose; ``R``/``t`` map OpenCV-frame head-model points to camera space (metres)."""
    R: np.ndarray                 # (3, 3) proper rotation, OpenCV head frame
    t: np.ndarray                 # (3,) metres, t_z > 0
    quat: tuple[float, float, float, float]   # (w, x, y, z), w >= 0
    yaw_deg: float                # standard frame, see module docstring
    pitch_deg: float
    roll_deg: float
    reproj_rms_px: float          # RMS over the inlier set after refinement
    inliers: int
    source: str                   # "pnp_dense" (>= 68 points) | "pnp_5pt"
    flip_resolved: bool           # a distinct mirrored-depth hypothesis converged and was weighed against the primary

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", np.array(self.R, np.float64))
        object.__setattr__(self, "t", np.array(self.t, np.float64))
        self.R.setflags(write=False)
        self.t.setflags(write=False)

    @property
    def spherical_bin(self) -> str:
        return pose_bins.classify(self.yaw_deg, self.pitch_deg).name

    def to_telemetry(self) -> dict[str, Any]:
        """The ``pose_6dof`` object of the pipeline.status frame (blueprint 03 §4)."""
        return {"yaw_deg": round(self.yaw_deg, 3), "pitch_deg": round(self.pitch_deg, 3),
                "roll_deg": round(self.roll_deg, 3),
                "tx_m": round(float(self.t[0]), 5), "ty_m": round(float(self.t[1]), 5), "tz_m": round(float(self.t[2]), 5),
                "quat": [round(float(v), 6) for v in self.quat],
                "reproj_rms_px": round(self.reproj_rms_px, 4), "inliers": int(self.inliers),
                "source": self.source, "spherical_bin": self.spherical_bin}


def default_intrinsics_matrix(width: int, height: int) -> np.ndarray:
    """K for an assumed 35 mm-equivalent focal length, principal point at the image centre."""
    focal = EQUIV_FOCAL_MM / EQUIV_SENSOR_MM * float(max(width, height))
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], np.float64)


def rotation_to_standard(rotation_cv: np.ndarray) -> np.ndarray:
    return _FLIP @ np.asarray(rotation_cv, np.float64) @ _FLIP


def rotation_from_standard(rotation_std: np.ndarray) -> np.ndarray:
    return _FLIP @ np.asarray(rotation_std, np.float64) @ _FLIP


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two rotations."""
    cosine = (float(np.trace(a @ b.T)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _canonical_cv_model(count: int, model_points: np.ndarray | None) -> np.ndarray:
    if model_points is None:
        if count not in (5, 68):
            raise DomainError("POSE_LANDMARK_COUNT",
                              f"No canonical head model for {count} landmarks (5 or 68, or pass model_points)", 422)
        model_points = get_canonical_3d_landmarks(count)
    model = np.asarray(model_points, np.float64)
    if model.shape != (count, 3) or not np.isfinite(model).all():
        raise DomainError("POSE_LANDMARK_COUNT", f"model_points {model.shape} do not match {count} landmarks", 422)
    return np.ascontiguousarray(model @ _FLIP)


def _project(model: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray) -> np.ndarray:
    return cv2.projectPoints(model, rvec, tvec, k, None)[0].reshape(-1, 2)


def _rms(model: np.ndarray, points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((_project(model, rvec, tvec, k) - points) ** 2, axis=1))))


def _valid(model: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> bool:
    if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()) or tvec[2, 0] <= 0:
        return False
    depth = (model @ cv2.Rodrigues(rvec)[0].T)[:, 2] + tvec[2, 0]
    return bool((depth > 1e-3).all())


def _refine(model: np.ndarray, points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray,
            *, seeded: bool) -> tuple[np.ndarray, np.ndarray] | None:
    """LM refinement; ``seeded`` first re-solves iteratively from the given extrinsic guess."""
    try:
        if seeded:
            ok, rvec, tvec = cv2.solvePnP(model, points, k, None, rvec.copy(), tvec.copy(), True, cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
        rvec, tvec = cv2.solvePnPRefineLM(model, points, k, None, rvec, tvec)
    except cv2.error:
        return None
    return (rvec, tvec) if _valid(model, rvec, tvec) else None


def solve_head_pose_6dof(landmarks_2d: Any, *, image_size: tuple[int, int] | None = None,
                         intrinsics: CameraIntrinsics | np.ndarray | None = None,
                         model_points: np.ndarray | None = None,
                         prev_rotation: np.ndarray | None = None,
                         prior_weight: float = FLIP_PRIOR_PX_PER_DEG,
                         ransac_threshold_px: float = RANSAC_THRESHOLD_PX) -> HeadPose6DoF:
    """Solve the head pose from ``N`` 2-D landmarks (pixels), ``N == 5`` or ``N >= 68`` (iBUG order for the default table).

    ``image_size`` is ``(width, height)`` and selects the assumed-focal K; ``intrinsics`` (CameraIntrinsics or a 3x3 K) takes
    precedence. ``model_points`` is an ``(N, 3)`` metric table in the repo frame (z toward the viewer). ``prev_rotation`` is the
    previous frame's ``HeadPose6DoF.R`` and enables the temporal term of flip resolution."""
    points = np.asarray(landmarks_2d, np.float64)
    if points.ndim == 2 and points.shape[1] == 3:
        points = points[:, :2]
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise DomainError("POSE_LANDMARKS_INVALID", f"Pose needs finite (N, 2) landmarks, got shape {points.shape}", 422)
    count = len(points)
    if count != 5 and count < 68:
        raise DomainError("POSE_LANDMARK_COUNT", f"PnP head pose needs 5 or >= 68 landmarks, got {count}", 422)
    points = np.ascontiguousarray(points)
    model = _canonical_cv_model(count, model_points)
    if isinstance(intrinsics, CameraIntrinsics):
        k = intrinsics.matrix()
    elif intrinsics is not None:
        k = np.asarray(intrinsics, np.float64)
        if k.shape != (3, 3):
            raise DomainError("POSE_INTRINSICS_INVALID", "intrinsics must be CameraIntrinsics or a 3x3 matrix", 422)
    elif image_size is not None:
        k = default_intrinsics_matrix(int(image_size[0]), int(image_size[1]))
    else:
        raise DomainError("POSE_INTRINSICS_INVALID", "Pass image_size=(width, height) or intrinsics", 422)

    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    threshold = max(float(ransac_threshold_px), 0.02 * extent)
    inlier_ids = np.arange(count)
    try:
        if count >= MIN_RANSAC_POINTS:
            ok, rvec, tvec, found = cv2.solvePnPRansac(
                model, points, k, None, iterationsCount=200, reprojectionError=threshold,
                confidence=RANSAC_CONFIDENCE, flags=cv2.SOLVEPNP_SQPNP)
            if not ok or found is None or len(found) < max(MIN_INLIERS, int(MIN_INLIER_FRACTION * count)):
                raise DomainError("POSE_PNP_FAILED", "RANSAC PnP found no consistent landmark consensus", 422)
            inlier_ids = np.sort(found.reshape(-1))
        else:
            ok, rvec, tvec = cv2.solvePnP(model, points, k, None, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                raise DomainError("POSE_PNP_FAILED", "PnP did not converge", 422)
    except cv2.error as error:
        raise DomainError("POSE_PNP_FAILED", f"PnP failed: {error.msg[:120]}", 422) from error

    m_in, p_in = np.ascontiguousarray(model[inlier_ids]), np.ascontiguousarray(points[inlier_ids])
    primary = _refine(m_in, p_in, rvec, tvec, k, seeded=False)
    if primary is None:
        raise DomainError("POSE_PNP_FAILED", "PnP solution is behind the camera or non-finite", 422)

    candidates = [primary]
    # Mirrored-depth seed: ``F R F`` keeps the image-plane block of R and negates the depth couplings (r13, r23, r31, r32).
    # For a flat head model this is exactly the second PnP solution; for relief it is the start of a competing minimum.
    mirrored_seed = _FLIP @ cv2.Rodrigues(primary[0])[0] @ _FLIP
    mirrored = _refine(m_in, p_in, cv2.Rodrigues(mirrored_seed)[0], primary[1], k, seeded=True)
    flip_resolved = False
    if mirrored is not None and rotation_angle_deg(cv2.Rodrigues(mirrored[0])[0], cv2.Rodrigues(primary[0])[0]) > _DISTINCT_HYPOTHESIS_DEG:
        candidates.append(mirrored)
        flip_resolved = True

    prev = None if prev_rotation is None else np.asarray(prev_rotation, np.float64)
    if prev is not None and prev.shape != (3, 3):
        raise DomainError("POSE_PREV_INVALID", "prev_rotation must be a 3x3 rotation", 422)

    def facing(candidate: tuple[np.ndarray, np.ndarray]) -> bool:
        # nose direction in camera space is R·(0,0,-1); facing the camera means its z is negative. A 15° margin beyond
        # exact profile keeps |yaw| ≈ 90° poses (where the sign of that z is noise) from being discarded.
        return float(cv2.Rodrigues(candidate[0])[0][2, 2]) > -_FACING_MARGIN

    def cost(candidate: tuple[np.ndarray, np.ndarray]) -> float:
        value = _rms(m_in, p_in, candidate[0], candidate[1], k)
        if prev is not None:
            value += prior_weight * rotation_angle_deg(cv2.Rodrigues(candidate[0])[0], prev)
        return value

    pool = [c for c in candidates if facing(c)] or candidates
    rvec, tvec = min(pool, key=cost)

    rotation = cv2.Rodrigues(rvec)[0]
    w, x, y, z = rotation_matrix_to_quaternion(rotation)
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    yaw, pitch, roll = euler_from_rotation(rotation_to_standard(rotation))
    return HeadPose6DoF(
        R=rotation, t=tvec.reshape(3), quat=(w, x, y, z), yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
        reproj_rms_px=_rms(m_in, p_in, rvec, tvec, k), inliers=int(len(inlier_ids)),
        source=SOURCE_DENSE if count >= 68 else SOURCE_5PT, flip_resolved=flip_resolved)
