import cv2
import numpy as np
import pytest

from roop_3d.core.pose import rotation_from_euler
from roop_3d.engine.detection.dense_landmarks import DenseLandmarks, observation_pose
from roop_3d.engine.detection.face_tracker import FaceTracker
from roop_3d.reconstruction.pose_solver import default_intrinsics_matrix, rotation_from_standard
from roop_3d.reconstruction.tps_fallback import get_canonical_3d_landmarks


class _Detector:
    def __init__(self, box, points5):
        self.box, self.points5 = box, points5

    def detect(self, image, threshold):
        return ([(self.box, self.points5)], None)


class _Dense:
    def __init__(self, points68):
        self.points68 = points68

    def infer106(self, image, box):
        return np.zeros((106, 2), np.float32)

    def infer68(self, image, box):
        return self.points68

    def estimate_pose(self, image, points5, *, landmarks68=None, prev_rotation=None):
        height, width = image.shape[:2]
        return observation_pose(points5, landmarks68=landmarks68, image_size=(width, height),
                                prev_rotation=prev_rotation)


class _Recognizer:
    def embed(self, image, points5):
        result = np.zeros(512, np.float32)
        result[0] = 1
        return result


def _tracker(points68, points5):
    tracker = object.__new__(FaceTracker)
    tracker.detector = _Detector(np.array([100, 70, 240, 230], np.float32), points5)
    tracker.dense = _Dense(points68)
    tracker.recognizer = _Recognizer()
    tracker.threshold = 0.5
    tracker.low_threshold = None
    tracker.max_tracks = 4
    return tracker


def _projected_landmarks(width=320, height=240):
    repo_points = get_canonical_3d_landmarks(68)
    flip = np.diag([1.0, 1.0, -1.0])
    standard_rotation = rotation_from_euler(18.0, -7.0, 3.0)
    rotation = rotation_from_standard(standard_rotation)
    rvec = cv2.Rodrigues(rotation)[0]
    tvec = np.array([[0.015], [-0.01], [0.82]], np.float64)
    camera = default_intrinsics_matrix(width, height)
    pixels = cv2.projectPoints(repo_points @ flip, rvec, tvec, camera, None)[0].reshape(-1, 2)
    return pixels.astype(np.float32)


def test_tracker_observation_uses_dense_pnp_and_exposes_evidence_backed_telemetry():
    image = np.zeros((240, 320, 3), np.uint8)
    points5 = np.array([[140, 110], [200, 110], [170, 145], [145, 175], [195, 175]], np.float32)
    observed = _tracker(_projected_landmarks(), points5).observe(image)[0]

    assert observed.pose_source == "pnp_dense"
    assert observed.pose_fallback_reason is None
    assert observed.landmarks68.shape == (68, 2)
    assert observed.pose6dof is not None
    telemetry = observed.pose_6dof
    assert telemetry["source"] == "pnp_dense"
    assert telemetry["inliers"] >= 60
    assert telemetry["reproj_rms_px"] < 1.0
    assert telemetry["tz_m"] == pytest.approx(0.82, abs=0.03)


def test_tracker_reports_geometric_fallback_when_dense_landmarks_are_unavailable():
    image = np.zeros((240, 320, 3), np.uint8)
    points5 = np.array([[140, 110], [200, 110], [170, 145], [145, 175], [195, 175]], np.float32)
    observed = _tracker(None, points5).observe(image)[0]

    assert observed.pose_source == "geometric_5pt"
    assert observed.pose6dof is None
    assert observed.pose_6dof is None
    assert observed.pose_fallback_reason == "no_dense_landmarks"
    assert np.isfinite(observed.pose).all()
