"""Nine acquisition buckets and two-view interpolation on the viewing sphere.

Roll rotates the image, not the surface being observed. Atlases already live in
canonical UV space, so it must not change their angular selection weights.
"""
import numpy as np

BUCKETS = ("frontal", "left_quarter", "right_quarter", "left_profile", "right_profile",
           "pitch_up", "pitch_down", "left_oblique", "right_oblique")


def pose_bucket(yaw, pitch):
    if abs(pitch) >= 20 and abs(yaw) >= 20:
        return "left_oblique" if yaw < 0 else "right_oblique"
    if abs(pitch) >= 20:
        return "pitch_up" if pitch > 0 else "pitch_down"
    if abs(yaw) >= 50:
        return "left_profile" if yaw < 0 else "right_profile"
    if abs(yaw) >= 15:
        return "left_quarter" if yaw < 0 else "right_quarter"
    return "frontal"


def view_directions(poses):
    angles = np.deg2rad(np.asarray(poses, np.float64)[..., :2])
    yaw, pitch = angles[..., 0], angles[..., 1]
    return np.stack((np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)), -1)


def nearest_views(poses, target):
    """Top two angular neighbours and normalized spherical arc weights.

    Two samples define an arc, rather than a spherical triangle. Off-arc targets
    use normalized opposite angular distances; exact samples have unit weight.
    """
    directions = view_directions(poses)
    distance = np.arccos(np.clip(directions @ view_directions(target), -1, 1))
    ids = np.argsort(distance, kind="stable")[:2]
    if not len(ids):
        return ids, np.empty(0)
    d = distance[ids]
    if len(ids) == 1 or d[0] < 1e-7:
        return ids[:1], np.ones(1)
    return ids, d[::-1] / d.sum()


def select_bucket_views(views, indices, limit):
    """Preserve rare profiles/pitch views before filling remaining quality slots."""
    from roop_3d.faceset.pose_matrix import view_pose
    ranked = sorted(indices, key=lambda i: (-views[i].quality.score, i))
    first, seen = [], set()
    for i in ranked:
        pose = view_pose(views[i])
        bucket = pose_bucket(pose.yaw, pose.pitch)
        if bucket not in seen:
            first.append(i)
            seen.add(bucket)
    return (first + [i for i in ranked if i not in first])[:limit]
