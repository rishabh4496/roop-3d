"""Unified head-pose structure, tuple sanitizer and the single 5-point pose estimator.

Two frames exist in this code base and are kept apart on purpose:

* **Public / standard** (`Pose3D`): yaw, pitch, roll in degrees. Yaw is positive when the nose
  moves toward the image right, **pitch is positive when the head looks up**, roll is the
  in-plane tilt of the eye line (positive = clockwise in image coordinates).
* **Legacy** raw tuples (`FaceView.pitch`, `FaceObservation.pose`, archive `pose` lists, the
  `classify_pose_bin` bins): identical except **pitch is positive when the head looks down**.
  Stored FaceSets and an existing test pin this, so it is not changed in place.

`as_pose3d` is the only place the two frames meet. It also accepts the shapes that used to
crash mixed callers: 2-tuples `(yaw, pitch)`, 3-tuples, numpy arrays, dicts and objects with
`yaw/pitch/roll` attributes. (`reconstruction.geometry_types.Pose3D` is a different, calibrated
rigid-body pose with rvec/tvec; it is accepted by the sanitizer but is not this class.)
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple

import numpy as np
from pydantic import Field

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract

PitchFrame = Literal["legacy", "standard"]

# Stage 0 3x3 pose matrix edges (degrees). Centre bands are inclusive: [-15, 15] and [-10, 10].
YAW_EDGE = 15.0
PITCH_EDGE = 10.0
YAW_NAMES = ("left", "center", "right")
PITCH_NAMES = ("down", "level", "up")


class Pose3D(Contract):
    yaw: float = Field(ge=-90.0, le=90.0)
    pitch: float = Field(ge=-90.0, le=90.0, description="Positive = looking up")
    roll: float = Field(default=0.0, ge=-180.0, le=180.0)

    def legacy_tuple(self) -> tuple[float, float, float]:
        """(yaw, pitch, roll) in the legacy frame where positive pitch means looking down."""
        return (self.yaw, -self.pitch, self.roll)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.yaw, self.pitch, self.roll)

    @property
    def cell(self) -> "PoseCell":
        return pose_cell(self)


class PoseCell(NamedTuple):
    """One of the nine yaw x pitch cells of the Stage 0 pose matrix."""
    yaw_index: int    # 0 left, 1 center, 2 right
    pitch_index: int  # 0 down, 1 level, 2 up

    @property
    def index(self) -> int:
        return self.pitch_index * 3 + self.yaw_index

    @property
    def name(self) -> str:
        return f"{YAW_NAMES[self.yaw_index]}_{PITCH_NAMES[self.pitch_index]}"

    @property
    def center(self) -> tuple[float, float]:
        """Representative (yaw, pitch) of the cell, used to rank neighbours when filling empty cells."""
        return ((-2 * YAW_EDGE, 0.0, 2 * YAW_EDGE)[self.yaw_index],
                (-2 * PITCH_EDGE, 0.0, 2 * PITCH_EDGE)[self.pitch_index])


ALL_CELLS: tuple[PoseCell, ...] = tuple(PoseCell(y, p) for p in range(3) for y in range(3))


def pose_cell(pose: Any, frame: PitchFrame = "legacy") -> PoseCell:
    """Cell of `pose`. Raw tuples/dicts follow `as_pose3d`'s default (legacy frame); `Pose3D` is standard."""
    value = as_pose3d(pose, frame=frame)
    yaw_index = 0 if value.yaw < -YAW_EDGE else 2 if value.yaw > YAW_EDGE else 1
    pitch_index = 0 if value.pitch < -PITCH_EDGE else 2 if value.pitch > PITCH_EDGE else 1
    return PoseCell(yaw_index, pitch_index)


def _finite(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise DomainError("POSE_INVALID", f"Pose {what} must be a number", 422) from None
    if not math.isfinite(number):
        raise DomainError("POSE_INVALID", f"Pose {what} must be finite", 422)
    return number


def as_pose3d(value: Any, *, frame: PitchFrame = "legacy", default: Pose3D | None = None) -> Pose3D:
    """Normalise any supported pose spelling into a `Pose3D` (standard frame).

    `frame` states which pitch sign the *input* uses. Raw tuples/dicts in this code base are legacy
    (positive pitch = down) and are converted; `Pose3D` instances are already standard and pass through
    unchanged. Out-of-range angles are clamped (a noisy estimate must never crash a render); malformed
    shapes and non-finite numbers raise `DomainError(POSE_INVALID)`.
    """
    if isinstance(value, Pose3D):
        return value
    if value is None:
        if default is not None:
            return default
        raise DomainError("POSE_INVALID", "A head pose is required", 422)
    if isinstance(value, Mapping):
        if "yaw" not in value or "pitch" not in value:
            raise DomainError("POSE_INVALID", "A pose mapping needs yaw and pitch", 422)
        yaw, pitch, roll = value["yaw"], value["pitch"], value.get("roll", 0.0)
    elif hasattr(value, "yaw") and hasattr(value, "pitch"):
        yaw, pitch, roll = value.yaw, value.pitch, getattr(value, "roll", 0.0)
        frame = "standard"  # attribute-style poses (rigid-body solver, Pose3D look-alikes) are camera-frame standard
    else:
        try:
            items = list(np.asarray(value, dtype=np.float64).reshape(-1)) if not isinstance(value, (tuple, list)) else list(value)
        except (TypeError, ValueError):
            raise DomainError("POSE_INVALID", "Pose must be a (yaw, pitch[, roll]) sequence, mapping or Pose3D", 422) from None
        if len(items) not in (2, 3):
            raise DomainError("POSE_INVALID", f"Pose needs 2 (yaw, pitch) or 3 (yaw, pitch, roll) values, got {len(items)}", 422)
        yaw, pitch, roll = items[0], items[1], items[2] if len(items) == 3 else 0.0
    yaw, pitch, roll = _finite(yaw, "yaw"), _finite(pitch, "pitch"), _finite(roll, "roll")
    if frame == "legacy":
        pitch = -pitch
    wrapped_roll = (roll + 180.0) % 360.0 - 180.0 if abs(roll) > 180.0 else roll
    return Pose3D(yaw=max(-90.0, min(90.0, yaw)), pitch=max(-90.0, min(90.0, pitch)), roll=max(-180.0, min(180.0, wrapped_roll)))


def as_legacy_triplet(value: Any) -> tuple[float, float, float]:
    """Sanitise `value` and return it in the legacy frame, for code that compares against stored FaceViews."""
    return as_pose3d(value).legacy_tuple()


# --------------------------------------------------------------------------- 5-point estimator
# Canonical 3D layout in metres: x right, y down, z toward the viewer (nose tip nearest).
# Order: image-left eye, image-right eye, nose tip, image-left mouth corner, image-right mouth corner.
_MODEL5 = np.array([
    [-0.032, -0.016, 0.022],
    [0.032, -0.016, 0.022],
    [0.000, 0.022, 0.062],
    [-0.024, 0.048, 0.035],
    [0.024, 0.048, 0.035],
], np.float64)
_MODEL5_CENTERED = _MODEL5 - _MODEL5.mean(axis=0)


def rotation_from_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """R = Rz(roll) @ Rx(pitch) @ Ry(yaw) in the standard frame (image y points down, z toward the viewer)."""
    y, p, r = (math.radians(v) for v in (yaw, pitch, roll))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def euler_from_rotation(matrix: np.ndarray) -> tuple[float, float, float]:
    """Inverse of `rotation_from_euler` (standard frame), degrees."""
    pitch = math.asin(max(-1.0, min(1.0, float(matrix[2, 1]))))
    yaw = math.atan2(-matrix[2, 0], matrix[2, 2])
    roll = math.atan2(-matrix[0, 1], matrix[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def project_model5(pose: Pose3D, *, scale: float, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Scaled-orthographic image positions of the canonical 5 points for `pose` (used by tests and frontalization)."""
    rotated = _MODEL5_CENTERED @ rotation_from_euler(pose.yaw, pose.pitch, pose.roll).T
    return rotated[:, :2] * scale + np.asarray(center, np.float64)


def estimate_pose_5pt(landmarks5: Any) -> Pose3D:
    """Continuous yaw/pitch/roll from five ArcFace landmarks.

    A scaled-orthographic fit to the canonical 5-point head model: the 2x3 projection that best maps the
    model onto the observed points is found by least squares, orthonormalised into a rotation, and
    decomposed. Because the model is non-planar the solution is unique (no Necker flip). This is an
    uncalibrated estimate (no camera intrinsics), good to a few degrees for frontal-to-three-quarter views.
    Degenerate input (collinear / non-finite points) returns the neutral pose rather than raising, so a
    noisy detection cannot abort a render.
    """
    points = np.asarray(landmarks5, np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return Pose3D(yaw=0.0, pitch=0.0, roll=0.0)
    observed = points - points.mean(axis=0)
    solution, _, rank, _ = np.linalg.lstsq(_MODEL5_CENTERED, observed, rcond=None)  # (3, 2): model -> image
    if rank < 3:
        return Pose3D(yaw=0.0, pitch=0.0, roll=0.0)
    rows = solution.T  # (2, 3): first two rows of s * R
    scale = float((np.linalg.norm(rows[0]) + np.linalg.norm(rows[1])) / 2)
    if scale < 1e-9:
        return Pose3D(yaw=0.0, pitch=0.0, roll=0.0)
    u, _, vt = np.linalg.svd(rows / scale, full_matrices=False)
    r12 = u @ vt  # nearest orthonormal pair of rows
    rotation = np.vstack((r12, np.cross(r12[0], r12[1])))
    yaw, pitch, roll = euler_from_rotation(rotation)
    return as_pose3d((yaw, pitch, roll), frame="standard")


def legacy_pose_5pt(landmarks5: Any) -> tuple[float, float, float]:
    """`estimate_pose_5pt` expressed in the legacy frame (pitch positive = down) for existing callers."""
    return estimate_pose_5pt(landmarks5).legacy_tuple()


def estimate_pose_6dof(landmarks5: Any, width: int, height: int) -> dict | None:
    """Perspective PnP with an estimated camera and canonical metric head model.

    Translation is relative to the canonical head size, not calibrated subject depth.
    Reject unreliable fits rather than reporting a spurious pose.
    """
    import cv2
    points = np.asarray(landmarks5, np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return None
    focal = float(max(width, height))
    camera = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
    reflection = np.diag([1., 1., -1.])
    model = _MODEL5 @ reflection
    seed = estimate_pose_5pt(points)
    rotation = reflection @ rotation_from_euler(seed.yaw, seed.pitch, seed.roll) @ reflection
    rvec = cv2.Rodrigues(rotation)[0]
    spread = max(float(np.linalg.norm(points[1]-points[0])), 1.)
    depth = focal * .064 / spread
    center = points.mean(axis=0)
    tvec = np.array([[(center[0]-width/2)*depth/focal],
                     [(center[1]-height/2)*depth/focal], [depth]])
    try:
        ok, rvec, tvec = cv2.solvePnP(model, points, camera, None, rvec, tvec,
                                     useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        projected = cv2.projectPoints(model, rvec, tvec, camera, None)[0].reshape(-1,2)
        error = float(np.sqrt(np.mean(np.sum((projected-points)**2, axis=1))))
        angles = euler_from_rotation(reflection @ cv2.Rodrigues(rvec)[0] @ reflection)
        if not ok or tvec[2,0] <= 0 or error > max(3., spread*.15) or max(abs(v) for v in angles[:2]) > 90:
            return None
        return {"yaw": angles[0], "pitch": angles[1], "roll": angles[2],
                "translation": tvec.ravel().tolist(), "camera_estimated": True,
                "reprojection_error_px": error}
    except cv2.error:
        return None
