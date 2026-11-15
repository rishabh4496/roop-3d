"""Staged scale / tile / roll face-detection recovery (engine/detection/robust_detector.py) and the relaxed validity rules.

The detector double reproduces the one property of SCRFD that matters here: it only sees faces whose size, after the
letterbox into its 640 input, lies in a fixed range, and only roughly upright ones.  Everything else (coordinate maps
back to native pixels, stage order, bounded cost, NMS, thresholds) is the real code under test.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from roop_3d.engine.detection.robust_detector import RobustDetector, dense_validator, geometry_ok
from roop_3d.media.face_analysis import FaceAnalyzer, five_point_plausible

SKIN = (150, 180, 220)
COLORS = {"eye_l": (0, 0, 255), "eye_r": (0, 255, 0), "nose": (255, 0, 0), "mouth_l": (0, 255, 255), "mouth_r": (255, 0, 255)}
OFFSETS = {"eye_l": (-0.18, -0.12), "eye_r": (0.18, -0.12), "nose": (0.0, 0.05), "mouth_l": (-0.14, 0.27), "mouth_r": (0.14, 0.27)}


def _rot(vector, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]


def draw_face(image: np.ndarray, centre, size: float, roll: float = 0.0) -> np.ndarray:
    """Ground-truth landmarks (5, 2) of a synthetic face drawn into ``image`` (in place)."""
    cv2.ellipse(image, (int(centre[0]), int(centre[1])), (int(0.42 * size), int(0.5 * size)), roll, 0, 360, SKIN, -1)
    points = []
    for name, offset in OFFSETS.items():
        dx, dy = _rot((offset[0] * size, offset[1] * size), roll)
        point = (centre[0] + dx, centre[1] + dy)
        cv2.circle(image, (int(round(point[0])), int(round(point[1]))), max(3, int(0.06 * size)), COLORS[name], -1)
        points.append(point)
    return np.array(points)


class FakeDetector:
    """Finds the synthetic face only when it is upright (<= 35 deg) and 30-300 px after the 640 letterbox."""

    def __init__(self, score: float = 0.9):
        self.score = score
        self.calls: list[tuple[int, int]] = []

    def detect_scored(self, image, threshold):
        self.calls.append(image.shape[:2])
        record = {"model_id": "fake"}
        if self.score < threshold:
            return [], record
        centres = {}
        for name, colour in COLORS.items():
            mask = cv2.inRange(image, np.array(colour) - 70, np.array(colour) + 70)
            count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
            if count < 2:
                return [], record
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[best, cv2.CC_STAT_AREA] < 4:
                return [], record
            centres[name] = centroids[best]
        points = np.array([centres[n] for n in COLORS], np.float32)
        eye_vector = points[1] - points[0]
        roll = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))
        size = float(np.linalg.norm(eye_vector) / 0.36)
        effective = size * 640.0 / max(image.shape[:2])
        if abs(roll) > 35 or not 30 <= effective <= 300 or points[3:, 1].mean() <= points[:2, 1].mean():
            return [], record
        centre = points.mean(axis=0)
        half = np.array([0.42 * size, 0.5 * size])
        box = np.concatenate([centre - half, centre + half]).astype(np.float32)
        return [(box, points, self.score)], record


def canvas(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), 40, np.uint8)


def test_oversized_face_is_found_by_zoom_out_and_mapped_back_exactly():
    frame = canvas(480, 360)
    truth = draw_face(frame, (240, 180), 400)
    base = FakeDetector()
    assert base.detect_scored(frame, 0.3)[0] == []                         # the primary pass really misses it
    detector = RobustDetector(base)
    found, record = detector.detect_scored(frame, 0.3)
    assert len(found) == 1
    box, points, score = found[0]
    assert record["recovery"]["stage"] == "zoom_out"
    assert np.abs(points - truth).max() < 4.0
    assert np.abs((box[:2] + box[2:]) / 2 - truth.mean(axis=0)).max() < 6      # the double centres its box on the landmarks
    assert score == pytest.approx(0.9)


def test_small_face_in_a_4k_frame_is_found_by_tiling():
    frame = canvas(4096, 2160)
    truth = draw_face(frame, (1500, 900), 100)
    detector = RobustDetector(FakeDetector())
    assert FakeDetector().detect_scored(frame, 0.3)[0] == []
    found, record = detector.detect_scored(frame, 0.3)
    assert len(found) == 1 and record["recovery"]["stage"] == "tile"
    assert np.abs(found[0][1] - truth).max() < 3.0                         # tile offsets are un-mapped
    assert detector.stats["passes"] <= 3 + 12                              # zoom-out passes + at most one tile level


@pytest.mark.parametrize("roll", [60.0, -60.0, 90.0])
def test_rolled_face_is_found_by_rotation_and_keeps_landmark_identity(roll):
    frame = canvas(640, 480)
    truth = draw_face(frame, (320, 240), 220, roll=roll)
    detector = RobustDetector(FakeDetector())
    assert FakeDetector().detect_scored(frame, 0.3)[0] == []
    found, record = detector.detect_scored(frame, 0.3)
    assert len(found) == 1 and record["recovery"]["stage"] == "rotate"
    box, points, _ = found[0]
    assert np.abs(points - truth).max() < 5.0                              # eye/eye/nose/mouth/mouth order preserved
    assert np.linalg.norm((box[:2] + box[2:]) / 2 - truth.mean(axis=0)) < 12
    side = max(box[2] - box[0], box[3] - box[1])
    assert 0.6 * 220 < side < 1.4 * 220                                    # area-preserving, not the inflated rotated extent


def test_frames_the_primary_pass_solves_cost_no_extra_passes():
    frame = canvas(640, 480)
    draw_face(frame, (320, 240), 200)
    base = FakeDetector()
    detector = RobustDetector(base)
    found, record = detector.detect_scored(frame, 0.3)
    assert len(found) == 1 and "recovery" not in record
    assert len(base.calls) == 1 and detector.stats["passes"] == 0


def test_empty_frame_has_bounded_recovery_cost_and_budget_is_honoured():
    base = FakeDetector()
    detector = RobustDetector(base)
    found, _ = detector.detect_scored(canvas(640, 480), 0.3)
    # primary + zoom-out (3) + magnified windows (1.5x: 2x2, 2x: 3x2) + rotations (6); no coarse tiles on a small frame
    assert found == [] and len(base.calls) == 1 + 3 + 10 + 6
    base.calls.clear()
    detector.detect_scored(canvas(640, 480), 0.3, max_passes=2)
    assert len(base.calls) == 1 + 2
    base.calls.clear()
    detector.detect_scored(canvas(640, 480), 0.3, recover=False)
    assert len(base.calls) == 1


def test_a_caller_asking_for_half_never_receives_lower_scores_but_a_tolerant_caller_does():
    frame = canvas(480, 360)
    draw_face(frame, (240, 180), 400)
    detector = RobustDetector(FakeDetector(score=0.4))
    assert detector.detect(frame, 0.5)[0] == []
    found, _ = detector.detect_scored(frame, 0.30)
    assert len(found) == 1 and found[0][2] == pytest.approx(0.4)           # honest score, never inflated


def test_validator_can_veto_recovered_faces():
    frame = canvas(480, 360)
    draw_face(frame, (240, 180), 400)
    assert RobustDetector(FakeDetector(), validator=lambda *_: False).detect_scored(frame, 0.3)[0] == []
    assert len(RobustDetector(FakeDetector(), validator=lambda *_: True).detect_scored(frame, 0.3)[0]) == 1


def test_recovered_duplicates_are_merged_by_nms():
    frame = canvas(480, 360)
    draw_face(frame, (240, 180), 120)                                      # primary pass already sees this one
    draw_face(frame, (240, 180), 120)
    found, _ = RobustDetector(FakeDetector(score=0.4)).detect_scored(frame, 0.3)
    assert len(found) == 1


def test_attribute_access_is_forwarded_to_the_wrapped_detector():
    base = FakeDetector()
    base.catalog = "catalog"
    detector = RobustDetector(base)
    assert detector.catalog == "catalog" and detector.base is base


# ---------------------------------------------------------------------------------------- geometry / validity
GOOD = np.array([[125, 140], [175, 140], [150, 165], [130, 190], [170, 190]], float)
BOX = np.array([100.0, 100.0, 200.0, 220.0])


def rolled(points, degrees, about=(150, 165)):
    out = []
    for p in points:
        dx, dy = _rot((p[0] - about[0], p[1] - about[1]), degrees)
        out.append((about[0] + dx, about[1] + dy))
    return np.array(out)


def test_geometry_ok_is_roll_invariant_but_rejects_mislabelled_and_degenerate_points():
    assert geometry_ok(BOX, GOOD)
    assert geometry_ok(np.array([80.0, 80.0, 220.0, 250.0]), rolled(GOOD, 70))        # lying head
    assert not geometry_ok(BOX, GOOD[::-1])                                           # eyes below mouth
    assert not geometry_ok(BOX, GOOD + np.array([300, 0]))                            # landmarks far outside the box
    assert not geometry_ok(np.array([0.0, 0.0, 10.0, 10.0]), GOOD)
    collapsed = GOOD.copy()
    collapsed[3:, 1] = collapsed[:2, 1].mean()
    assert not geometry_ok(BOX, collapsed)
    assert not geometry_ok(np.array([100.0, 100.0, 120.0, 220.0]), GOOD)               # aspect < 0.25


def test_five_point_plausible_accepts_rolled_heads_and_narrow_profiles_but_not_mislabelled_points():
    assert five_point_plausible(BOX, GOOD) == (True, "ok")
    assert five_point_plausible(BOX, GOOD[::-1])[0] is False
    tilted = rolled(GOOD, 45)
    assert five_point_plausible(np.array([70.0, 80.0, 230.0, 250.0]), tilted)[0] is True   # was rejected by the image-y order test
    profile_box = np.array([120.0, 100.0, 156.0, 220.0])                              # aspect 0.30 (steep profile: kept)
    profile = np.array([[140, 140], [146, 140], [152, 165], [138, 190], [145, 190]], float)
    assert five_point_plausible(profile_box, profile) == (True, "narrow_profile")
    assert five_point_plausible(np.array([120.0, 100.0, 144.0, 220.0]), profile)[1] == "box_aspect"   # 0.20: below 0.25


class DenseEngine:
    has_106 = True
    has_68 = False

    def __init__(self, dense):
        self.dense = dense

    def landmarks106(self, image, box):
        return self.dense

    def signature(self):
        return "dense-double"


class ScriptedDetector:
    def __init__(self, box, points, score):
        self.item = (np.asarray(box, np.float64), np.asarray(points, np.float64), score)

    def detect_scored(self, image, threshold):
        return ([self.item] if self.item[2] >= threshold else []), {}


def test_narrow_profile_is_kept_only_when_the_dense_landmarks_agree():
    profile_box = np.array([120.0, 100.0, 160.0, 220.0])
    profile = np.array([[140, 140], [148, 140], [152, 165], [138, 190], [146, 190]], float)
    dense = np.vstack([profile + 0.5, np.random.default_rng(0).uniform(120, 160, (101, 2))])
    image = np.zeros((300, 300, 3), np.uint8)
    # 0.33 sits in the tolerant band [0.30, 0.35): too low to stand on its score, so the dense landmarks must vouch for it.
    agree = FaceAnalyzer(ScriptedDetector(profile_box, profile, 0.33), None, DenseEngine(dense))
    faces = agree.analyze(image, thumbnails=False)
    assert len(faces) == 1 and faces[0].tier == "profile_tolerant" and faces[0].validity["geometry"] == "narrow_profile"
    disagree = FaceAnalyzer(ScriptedDetector(profile_box, profile, 0.33), None, DenseEngine(np.full((106, 2), 900.0)))
    assert disagree.analyze(image, thumbnails=False) == []
    without_model = FaceAnalyzer(ScriptedDetector(profile_box, profile, 0.33), None, None)
    assert without_model.analyze(image, thumbnails=False) == []                       # no dense model: nothing is invented


def test_a_steep_profile_is_not_vetoed_by_the_unreliable_dense_model_when_its_score_is_calibrated_high_enough():
    profile_box = np.array([120.0, 100.0, 160.0, 220.0])
    profile = np.array([[140, 140], [148, 140], [152, 165], [138, 190], [146, 190]], float)
    image = np.zeros((300, 300, 3), np.uint8)
    far_dense = DenseEngine(np.full((106, 2), 900.0))             # the 106 model fails on a steep profile
    tolerant = FaceAnalyzer(ScriptedDetector(profile_box, profile, 0.37), None, far_dense).analyze(image, thumbnails=False)
    assert len(tolerant) == 1 and tolerant[0].tier == "profile_tolerant" and tolerant[0].validity["valid"] is True
    primary = FaceAnalyzer(ScriptedDetector(profile_box, profile, 0.45), None, None).analyze(image, thumbnails=False)
    assert len(primary) == 1 and primary[0].tier == "primary"      # a primary-score profile needs no dense model at all


def test_dense_validator_uses_agreement_and_accepts_when_the_model_is_missing():
    dense = np.vstack([GOOD + 1.0, np.random.default_rng(1).uniform(100, 200, (101, 2))])
    image = np.zeros((300, 300, 3), np.uint8)
    assert dense_validator(DenseEngine(dense))(image, BOX, GOOD) is True
    assert dense_validator(DenseEngine(np.full((106, 2), 900.0)))(image, BOX, GOOD) is False
    assert dense_validator(None)(image, BOX, GOOD) is True

    def missing(_image, _box):
        raise RuntimeError("MODEL_NOT_INSTALLED")

    assert dense_validator(missing)(image, BOX, GOOD) is True


# ------------------------------------------------------------------------------------- real-model letterbox check
CLIP = Path("G:/pinokio/roop-keep/single/s11.mp4")          # 360x640 portrait, one clear frontal face


def _real_detector():
    from roop_3d.media.detector import Detector
    from roop_3d.models.catalog import ModelCatalog
    root = Path(__file__).resolve().parents[2]
    try:
        catalog = ModelCatalog(cache_root=root)
        detector = Detector(catalog)
        catalog.detector()
        return detector
    except Exception as error:                                                         # weights not installed here
        pytest.skip(f"detector weights unavailable: {error}")


def test_letterbox_preserves_aspect_ratio_and_boxes_unmap_to_native_pixels():
    """The same face, placed at a known offset inside a tall canvas, must come back shifted by exactly that offset."""
    if not CLIP.is_file():
        pytest.skip("reference clip not present")
    capture = cv2.VideoCapture(str(CLIP))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ok, frame = capture.read()
    capture.release()
    assert ok
    detector = _real_detector()
    base, _ = detector.detect_scored(frame, 0.5)
    assert base, "reference frame must contain a detectable face"
    ratio = 2.0
    scaled = cv2.resize(frame, None, fx=ratio, fy=ratio)
    wide = np.zeros((1500, 3000, 3), np.uint8)                                          # 2:1 canvas -> a different letterbox
    x0, y0 = 150, 100
    wide[y0:y0 + scaled.shape[0], x0:x0 + scaled.shape[1]] = scaled
    moved, _ = detector.detect_scored(wide, 0.5)
    assert moved
    expected = base[0][1] * ratio + np.array([x0, y0])
    best = min(moved, key=lambda item: float(np.abs(item[1] - expected).max()))
    side = float(best[0][2] - best[0][0])
    # Boxes and landmarks must land on the same native pixels; a squashed or mis-anchored letterbox is off by >> this.
    assert np.abs(best[1] - expected).max() < 0.08 * side
    expected_box = base[0][0] * ratio + np.array([x0, y0, x0, y0])
    assert np.abs(best[0] - expected_box).max() < 0.08 * side
