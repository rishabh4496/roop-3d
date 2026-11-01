"""3D pose frontalization of an aligned face crop before identity-swap inference.

A profile or three-quarter target crop is a poor input for the 128-px swapper. Using the pose estimated from the
five ArcFace landmarks and the canonical 3D head model, the crop is warped (thin-plate spline) toward the
landmark layout the same head would have with its yaw/pitch reduced by `weight` (0 = untouched, 1 = fully
frontal). The swapper output is warped back with the inverse correspondence so it lines up with the original
alignment and the compositor needs no change. Roll is left alone (the aligned crop is already roll-normalised);
the image border is pinned so the crop edge never moves.

This is a landmark-driven 2D warp on an estimated pose, not a reconstruction of unseen face surface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roop_3d.core.pose import Pose3D, estimate_pose_5pt, project_model5
from roop_3d.reconstruction.tps_warp import border_anchors, warp_points_to

# Don't drag any control point further than this fraction of the crop (keeps the spline fold-free).
MAX_SHIFT_RATIO = 0.22
# Poses closer to frontal than this are left untouched (below the estimator's resolution).
NEGLIGIBLE_DEGREES = 2.0


@dataclass(frozen=True)
class Frontalized:
    image: np.ndarray
    observed_points: np.ndarray   # control points in the observed (original) crop
    frontal_points: np.ndarray    # where those points moved to
    pose: Pose3D
    target_pose: Pose3D
    applied: bool

    def restore(self, generated: np.ndarray) -> np.ndarray:
        """Warp a crop produced from the frontalized input back to the observed geometry."""
        if not self.applied:
            return generated
        h, w = generated.shape[:2]
        scale = np.array((w / self.image.shape[1], h / self.image.shape[0]))
        return warp_points_to(generated, self.frontal_points * scale, self.observed_points * scale)


def frontalize_crop(crop: np.ndarray, landmarks5: np.ndarray, weight: float) -> Frontalized:
    """`landmarks5` are in `crop` pixel coordinates. `weight` is `pose_frontalization` in 0..1."""
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("pose_frontalization must be within 0..1")
    points = np.asarray(landmarks5, np.float64)
    pose = estimate_pose_5pt(points)
    target = Pose3D(yaw=pose.yaw * (1.0 - weight), pitch=pose.pitch * (1.0 - weight), roll=pose.roll)
    untouched = Frontalized(crop, points, points, pose, target, False)
    if weight <= 0.0 or max(abs(pose.yaw), abs(pose.pitch)) < NEGLIGIBLE_DEGREES:
        return untouched
    # Weak-perspective scale of the observed head, then the same head (same scale/centre) at the reduced pose.
    unit = project_model5(pose, scale=1.0)
    centre = points.mean(axis=0)
    observed = points - centre
    denominator = float((unit * unit).sum())
    if denominator < 1e-12:
        return untouched
    scale = float((unit * observed).sum() / denominator)
    if scale <= 0:
        return untouched
    moved = project_model5(target, scale=scale, center=tuple(centre))
    height, width = crop.shape[:2]
    limit = MAX_SHIFT_RATIO * max(height, width)
    longest = float(np.linalg.norm(moved - points, axis=1).max())
    if longest > limit:  # shrink the whole field uniformly so relative motion is preserved
        moved = points + (moved - points) * (limit / longest)
    anchors = border_anchors(width, height)
    source, destination = np.vstack((points, anchors)), np.vstack((moved, anchors))
    image = warp_points_to(crop, source, destination)
    return Frontalized(image, source, destination, pose, target, True)
