"""Eyeball gaze for the tracker: an uncalibrated iris-geometry estimate, and vergence from two gaze directions.

``flame_3d_engine`` is explicit that "eyelids are not gaze" and only accepts a gaze model that carries calibration
evidence. This module adds one honest geometric source for the dense 478-point mesh and labels it as such:

  * ``iris_gaze_from_mesh478`` measures each iris centre against the eye's own corner axis. With the eyeball radius taken as
    ``EYEBALL_TO_HALF_WIDTH`` of the palpebral half-width, ``sin(angle) = offset / radius``. It is an angle estimate in the
    head frame (both landmarks move with the head), it ignores eyelid coupling and eyeball torsion (torsion is not
    observable from the iris centre), and it has NOT been calibrated against measured gaze. The tracker therefore reports
    its source as ``iris_geometry_uncalibrated``; an ``EyeGaze`` supplied by a calibrated estimator is reported as
    ``calibrated_estimator`` and is always preferred.
  * ``vergence`` intersects the two gaze rays in the horizontal plane to give the convergence angle and distance. The
    interpupillary distance is a population constant (``DEFAULT_IPD_M``); the distance scales with it.

Angle convention (the one ``EyeGaze.vectors`` uses): yaw positive toward image-right (+x), pitch positive toward image-down
(+y), the gaze vector (0, 0, 1) looks at the viewer. ``left_eye`` is the subject's left eye, which is the image-right eye of
an unmirrored frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from roop_3d.flame_3d_engine import EyeGaze

DEFAULT_IPD_M = 0.063
EYEBALL_TO_HALF_WIDTH = 0.8        # eyeball radius (~12 mm) / palpebral half-width (~15 mm)
VERGENCE_NOISE_FLOOR_DEG = 1.0     # below this the rays are parallel for every practical purpose
# MediaPipe FaceMesh indices: (outer corner, inner corner, iris centre)
RIGHT_EYE = (33, 133, 468)         # subject's right = image-left
LEFT_EYE = (263, 362, 473)         # subject's left = image-right
SOURCE_ESTIMATOR, SOURCE_IRIS = "calibrated_estimator", "iris_geometry_uncalibrated"


def _eye_angles(mesh: np.ndarray, corners: tuple[int, int, int], aspect: float) -> tuple[float, float] | None:
    xy = mesh[:, :2] * np.array((aspect, 1.0))       # normalised crop coordinates -> isotropic units
    a, b, iris = xy[corners[0]], xy[corners[1]], xy[corners[2]]
    axis = b - a
    width = float(np.linalg.norm(axis))
    if width < 1e-6:
        return None
    axis = axis / width
    down = np.array((-axis[1], axis[0]))
    if down[1] < 0:                                   # make `down` point toward image-down whatever the corner order
        axis, down = -axis, -down
    if axis[0] < 0:
        axis = -axis                                  # and `axis` toward image-right
    offset = iris - 0.5 * (a + b)
    radius = EYEBALL_TO_HALF_WIDTH * 0.5 * width
    sin_yaw, sin_pitch = float(offset @ axis) / radius, float(offset @ down) / radius
    return math.asin(max(-1.0, min(1.0, sin_yaw))), math.asin(max(-1.0, min(1.0, sin_pitch)))


def iris_gaze_from_mesh478(mesh478, *, aspect: float = 1.0) -> EyeGaze | None:
    """Head-frame gaze from a (478, 2|3) mesh, or None.

    Only the x/y geometry is used, and only relative to the eye itself, so frame pixels work as well as normalised crop
    coordinates; for the latter pass ``aspect`` = crop width / height so x and y are isotropic."""
    mesh = np.asarray(mesh478, np.float64)
    if mesh.ndim != 2 or mesh.shape[0] != 478 or mesh.shape[1] not in (2, 3) or not np.isfinite(mesh).all() \
            or not np.isfinite(aspect) or aspect <= 0:
        return None
    left, right = _eye_angles(mesh, LEFT_EYE, aspect), _eye_angles(mesh, RIGHT_EYE, aspect)
    if left is None or right is None:
        return None
    return EyeGaze(left[0], left[1], right[0], right[1])


@dataclass(frozen=True)
class Vergence:
    angle_deg: float                  # signed: positive = the rays converge (eyes turned toward each other)
    distance_m: float | None          # None when the rays are parallel / diverging within the noise floor


def vergence(angles, ipd_m: float = DEFAULT_IPD_M) -> Vergence:
    """Convergence of the two rays. ``angles`` = (left yaw, left pitch, right yaw, right pitch) in radians."""
    left_yaw, _, right_yaw, _ = (float(a) for a in angles)
    # The image-left eye (subject's right) turned toward +x and the image-right eye toward -x converge.
    angle = math.degrees(right_yaw - left_yaw)
    if angle <= VERGENCE_NOISE_FLOOR_DEG:
        return Vergence(angle, None)
    denominator = math.tan(right_yaw) - math.tan(left_yaw)     # rays x = -a + z tan(right), x = +a + z tan(left)
    return Vergence(angle, ipd_m / denominator if denominator > 1e-6 else None)


def gaze_vectors(angles) -> np.ndarray:
    """(2, 3) unit vectors, left then right, in the head frame."""
    return EyeGaze(*(float(a) for a in angles)).vectors()
