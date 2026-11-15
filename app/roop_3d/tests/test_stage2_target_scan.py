"""Stage 2: whole-timeline adaptive sampling, honest per-sample observations, pose-tolerant recovery, FaceSet reference bank."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.engine.detection.robust_detector import RobustDetector, _nms
from roop_3d.faceset.locator import resolve_faceset
from roop_3d.faceset.reference_bank import build_reference_bank
from roop_3d.faceset.schemas import FaceSetDocument, FaceView, QualityMetrics
from roop_3d.faceset_loader import FaceSample
from roop_3d.media.adaptive_sampling import AdaptiveSampler, FaceSnapshot
from roop_3d.media.schema import BoundingBox, FaceTrack, PoseRange, TrackObservation
from roop_3d.media.service import MAX_TRACK_OBSERVATIONS, merge_observation, resolve_inspection_track, thin_observations


# ----------------------------------------------------------------------------------------------- adaptive sampler
def _walk(sampler: AdaptiveSampler, total: int, show) -> list[int]:
    sampled = []
    for index in range(total):
        if sampler.wants(index):
            sampled.append(index)
            sampler.update(index, show(index))
    return sampled


def _face(yaw=0.0, track=1, cx=0.5) -> tuple[FaceSnapshot, ...]:
    return (FaceSnapshot(track, yaw, 0.0, 0.0, cx, 0.5, 0.3),)


def test_stable_scene_is_sampled_sparsely_but_across_the_whole_timeline():
    sampled = _walk(AdaptiveSampler(30.0, 3000), 3000, lambda i: _face())
    assert sampled[0] == 0 and sampled[-1] > 2900          # reaches the end, not just the head of the file
    assert len(sampled) < 3000 / 4                          # and costs a fraction of every-frame inference
    assert max(b - a for a, b in zip(sampled, sampled[1:])) <= round(30 * 0.25)   # never wider than face_interval


def test_a_head_turn_is_sampled_densely_then_relaxes():
    turn = range(1000, 1090)
    sampled = _walk(AdaptiveSampler(30.0, 3000), 3000, lambda i: _face(yaw=(i - 1000) * 1.0 if i in turn else 0.0))
    inside = [i for i in sampled if 1000 <= i < 1090]
    assert len(inside) >= 40                                # the turn (frontal -> 3/4 -> profile) is not skipped over
    steep = [i for i in sampled if 1030 <= i < 1090]        # |yaw| >= 30: at most 2 frames apart
    assert max(b - a for a, b in zip(steep, steep[1:])) <= 2
    assert len([i for i in sampled if 2000 <= i < 3000]) < 300   # relaxed again afterwards


def test_a_track_dropout_and_a_cut_collapse_the_stride():
    sampler = AdaptiveSampler(30.0, 1000)
    for index in range(0, 200):
        if sampler.wants(index):
            sampler.update(index, _face())
    assert sampler.stride == round(30 * 0.25)
    sampler.update(sampler.next_index, ())                  # the face vanished
    assert sampler.stride == 1 and sampler.reasons["track_set_changed"] == 1
    sampler.update(sampler.next_index, (), scene_cut=True)
    assert sampler.reasons["scene_cut"] == 1


def test_time_budget_stretches_the_stride_so_the_sweep_still_ends():
    sampler = AdaptiveSampler(30.0, 30000)
    # 0.5 s per sample against a 120 s budget: 240 samples afford a stride of ~125 frames, not 7.
    for index in range(0, 30000):
        if sampler.wants(index):
            sampler.update(index, _face(yaw=40.0 if index % 2 else -40.0, cx=0.5 + (index % 7) * 0.001),
                           elapsed=sampler.samples * 0.5, budget=120.0)
        if sampler.samples * 0.5 > 120.0:
            break
    assert index > 25000


# ----------------------------------------------------------------------------------------------- observations
def _obs(frame: int, box=(0.1, 0.1, 0.4, 0.5), yaw=0.0) -> TrackObservation:
    return TrackObservation(frame=frame, timestamp_seconds=frame / 30, bbox=box, yaw_degrees=yaw, pitch_degrees=0.0,
                            roll_degrees=0.0, confidence=0.9)


def test_thinning_is_bounded_ordered_and_keeps_the_pose_extremes():
    items = [_obs(i, yaw=float((i * 37) % 11)) for i in range(2000)]
    items[777] = _obs(777, yaw=88.0)
    items[1234] = _obs(1234, yaw=-90.0)
    kept = thin_observations(items)
    assert len(kept) <= MAX_TRACK_OBSERVATIONS + 4
    assert [o.frame for o in kept] == sorted(o.frame for o in kept)
    assert {777, 1234} <= {o.frame for o in kept}


def _track(track_id: str, observations) -> FaceTrack:
    return FaceTrack(track_id=track_id, first_frame=0, last_frame=100, first_timestamp_seconds=0, last_timestamp_seconds=3,
                     bounding_box=BoundingBox(x1=1, y1=1, x2=2, y2=2), pose_range=PoseRange(), thumbnail_base64="x",
                     sampled_detections=len(observations) or 1, observations=tuple(observations))


def test_a_frame_face_takes_the_scan_track_it_overlaps_at_that_moment():
    left = _track("track_0007", [_obs(f, (0.05, 0.1, 0.25, 0.5)) for f in range(0, 90, 5)])
    right = _track("track_0000", [_obs(f, (0.6, 0.1, 0.8, 0.5)) for f in range(0, 90, 5)])
    # The registry would call this face track_0000 by order of appearance; the scan knows it is track_0007.
    assert resolve_inspection_track([right, left], 42, (0.06, 0.1, 0.26, 0.5), 30.0) == "track_0007"
    assert resolve_inspection_track([right, left], 42, (0.3, 0.1, 0.45, 0.5), 30.0) is None      # nobody there
    assert resolve_inspection_track([left], 400, (0.06, 0.1, 0.26, 0.5), 30.0) is None            # nearest sample is far in time


def test_merge_replaces_the_same_frame_and_stays_sorted():
    merged = merge_observation((_obs(10), _obs(20)), _obs(10, yaw=30.0))
    assert [o.frame for o in merged] == [10, 20] and merged[0].yaw_degrees == 30.0


# ----------------------------------------------------------------------------------------------- detection recall
class _Base:
    """Detector double that only sees a face when it is magnified: the 640 px primary pass finds nothing."""
    def __init__(self):
        self.calls = []

    def detect_scored(self, image, threshold):
        self.calls.append(image.shape)
        h, w = image.shape[:2]
        if max(h, w) <= 640 and image.sum() > 0 and (image[..., 0] == 7).any():
            ys, xs = np.nonzero(image[..., 0] == 7)
            box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
            side = box[2] - box[0]
            if side < 36:                                # below the detector's reach at 1:1
                return [], {}
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            points = np.array([[cx - side * .2, cy - side * .1], [cx + side * .2, cy - side * .1], [cx, cy],
                               [cx - side * .15, cy + side * .2], [cx + side * .15, cy + side * .2]], np.float32)
            return [(box, points, 0.37)], {}
        return [], {}


def test_small_face_is_recovered_by_magnified_window_and_mapped_to_native_pixels():
    image = np.zeros((480, 640, 3), np.uint8)
    image[200:226, 300:326, 0] = 7                       # a 26 px face: below the primary pass's reach in the double
    detector = RobustDetector(_Base(), validator=None)
    found, record = detector.detect_scored(image, 0.3)
    assert found, "magnified windows must recover the face"
    box, points, score = found[0]
    assert record["recovery"]["stage"] == "magnify"       # zoom_out cannot help a small face; the magnified window does
    assert score == pytest.approx(0.37)                  # a 0.35-0.45 band score is kept, never inflated
    assert abs(box[0] - 300) < 6 and abs(box[2] - 326) < 6 and abs(box[1] - 200) < 6      # back in native pixels


def test_expected_face_count_triggers_recovery_even_when_another_face_passed():
    class Two:
        def __init__(self): self.calls = 0
        def detect_scored(self, image, threshold):
            self.calls += 1
            box = np.array([10, 10, 110, 130], np.float32)
            points = np.array([[40, 50], [80, 50], [60, 75], [45, 100], [75, 100]], np.float32)
            return [(box, points, 0.9)], {}
    base = Two()
    detector = RobustDetector(base)
    detector.detect_scored(np.zeros((480, 640, 3), np.uint8), 0.3, expected=1)
    assert base.calls == 1                               # one face expected, one found: no extra passes
    detector.detect_scored(np.zeros((480, 640, 3), np.uint8), 0.3, expected=2)
    assert base.calls > 1                                # a second face is searched for


def test_nms_collapses_a_nested_partial_box_but_keeps_two_adjacent_faces():
    big = (np.array([100, 100, 300, 340], np.float32), np.zeros((5, 2), np.float32), 0.9)
    nested = (np.array([130, 120, 230, 250], np.float32), np.zeros((5, 2), np.float32), 0.5)
    neighbour = (np.array([310, 110, 500, 340], np.float32), np.zeros((5, 2), np.float32), 0.8)
    kept = _nms([nested, neighbour, big])
    assert len(kept) == 2 and nested[0] is not kept[1][0]


# ----------------------------------------------------------------------------------------------- reference bank
def _unit(rng, base, noise):
    v = base + noise * rng.standard_normal(512).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _view(index, embedding, *, yaw, quality=None, trusted=True) -> FaceView:
    return FaceView(index=index, image_bytes=b"x", embedding=tuple(float(x) for x in embedding),
                    landmarks=((1, 1), (2, 1), (1.5, 1.5), (1.2, 2), (1.8, 2)), bbox=(0, 0, 200, 200),
                    yaw=yaw, pitch=0.0, roll=0.0, quality=quality or QualityMetrics(sharpness=0.8, detector_confidence=0.9,
                    score=0.8, face_pixels=200 * 200), pose_bin="center_level", landmarks_trusted=trusted)


def _sample(view: FaceView, *, yaw, pitch=0.0, dense=True, trusted=True) -> FaceSample:
    crop = np.full((512, 512, 3), 128, np.uint8)
    return FaceSample(view_index=view.index, normed_embedding=np.asarray(view.embedding, np.float32),
                      landmarks_5=np.zeros((5, 2)), landmarks_68=np.zeros((68, 3)) if dense else None, landmarks_106=None,
                      landmark_source="1k3d68" if dense else "archive_5pt", yaw=yaw, pitch=pitch, roll=0.0, reprojection_px=1.0,
                      crop_rgb_512=crop, sphere_bin=0, matrix_cell="center_level", matrix_index=(1, 1, 1), quality=0.8,
                      trusted=trusted)


def _document(views) -> FaceSetDocument:
    mean = np.mean([np.asarray(v.embedding) for v in views], axis=0)
    mean = tuple(float(x) for x in mean / np.linalg.norm(mean))
    return FaceSetDocument(faceset_id="t", name="t", source_sha256="0" * 64, archive_format="legacy_v1",
                           canonical_embedding=mean, medoid_embedding=mean, views=tuple(views), pose_bins={})


def test_reference_bank_centroid_rejections_and_pose_matched_matching():
    rng = np.random.default_rng(3)
    person = rng.standard_normal(512).astype(np.float32)
    stranger = rng.standard_normal(512).astype(np.float32)
    poses = [(0, 0.0), (1, 32.0), (2, -35.0), (3, 80.0)]
    views, samples = [], []
    for index, yaw in poses:
        view = _view(index, _unit(rng, person, 0.3 + 0.002 * abs(yaw)), yaw=yaw)
        views.append(view)
        samples.append(_sample(view, yaw=yaw, dense=abs(yaw) < 60))
    intruder = _view(4, _unit(rng, stranger, 0.05), yaw=0.0)
    fabricated = _view(5, _unit(rng, person, 0.2), yaw=0.0, trusted=False)
    views += [intruder, fabricated]
    samples += [_sample(intruder, yaw=0.0), _sample(fabricated, yaw=0.0, trusted=False)]
    bank = build_reference_bank(_document(views), samples)
    reasons = {r["view_index"]: r["reason"] for r in bank.rejected}
    assert reasons == {4: "identity_outlier", 5: "untrusted_landmarks"}
    assert np.linalg.norm(bank.centroid) == pytest.approx(1.0, abs=1e-5)
    assert float(bank.centroid @ (person / np.linalg.norm(person))) > 0.9        # the intruder did not drag it away
    assert {e.view_index for e in bank.entries} == {0, 1, 2, 3}
    order = bank.uv_views()
    assert order[0] in (0, 1, 2) and set(order) == {0, 1, 2, 3}
    assert order[1] in (3, 2)                                                    # second pick widens pose coverage first
    target = _unit(rng, person, 0.35)
    match = bank.similarity(target, yaw=78.0, pitch=0.0)
    assert match["score"] >= match["centroid_cosine"] and match["pose_matched_cosine"] is not None
    assert bank.nearest(78.0, 0.0, 1)[0].view_index == 3
    assert bank.similarity(stranger, yaw=0.0, pitch=0.0)["score"] < 0.3
    applied = bank.apply(_document(views))
    scores = {v.index: v.quality.score for v in applied.views}
    assert scores[4] == 0.0 and scores[5] == 0.0 and all(scores[i] > 0.2 for i in (0, 1, 2, 3))


def test_bank_refuses_a_set_with_no_trusted_view():
    rng = np.random.default_rng(1)
    view = _view(0, _unit(rng, rng.standard_normal(512).astype(np.float32), 0.0), yaw=0.0, trusted=False)
    with pytest.raises(DomainError) as error:
        build_reference_bank(_document([view]), [_sample(view, yaw=0.0, trusted=False)])
    assert error.value.code == "FACESET_NO_USABLE_VIEW"


# ----------------------------------------------------------------------------------------------- faceset locator
def test_library_faceset_is_found_by_its_stem_without_a_catalog_entry(tmp_path: Path):
    (tmp_path / "facesets").mkdir()
    (tmp_path / "facesets/harjot.fsz").write_bytes(b"PK")
    catalog = tmp_path / "roop_3d/.runtime/facesets"
    catalog.mkdir(parents=True)
    (tmp_path / "roop_3d/outputs/facesets").mkdir(parents=True)
    (catalog / "797577eca6ab4cf88c7b3e075b590979.json").write_text(json.dumps({"name": "harjot"}))
    (tmp_path / "roop_3d/outputs/facesets/797577eca6ab4cf88c7b3e075b590979.fsz").write_bytes(b"PK")
    library = resolve_faceset(tmp_path, "harjot")
    assert library.source == "library" and library.archive.name == "harjot.fsz" and library.faceset_id == "harjot"
    catalogued = resolve_faceset(tmp_path, "797577eca6ab4cf88c7b3e075b590979")
    assert catalogued.source == "catalog"
    with pytest.raises(DomainError) as error:
        resolve_faceset(tmp_path, "nobody")
    assert error.value.code == "FACESET_NOT_FOUND"
    with pytest.raises(DomainError) as error:
        resolve_faceset(tmp_path, "../harjot")
    assert error.value.code == "FACESET_ID_INVALID"


# ----------------------------------------------------------------------------------------------- calibrated threshold band
class _Scripted:
    def __init__(self, items): self.items = items
    def detect_scored(self, image, threshold):
        return [item for item in self.items if item[2] >= threshold], {}


def test_band_between_calibrated_primary_and_certain_must_be_vouched_for_by_landmarks():
    image = np.zeros((480, 640, 3), np.uint8)
    good_points = np.array([[160, 150], [240, 150], [200, 200], [170, 250], [230, 250]], np.float32)
    good_box = np.array([120, 100, 280, 300], np.float32)
    # Same score, same box: the second one's landmarks are mislabelled (mouth above the eyes), i.e. clutter.
    bad_points = good_points[::-1].copy()
    items = [(good_box, good_points, 0.44), (np.array([320, 100, 480, 300], np.float32), bad_points, 0.44),
             (np.array([10, 320, 100, 440], np.float32), bad_points - np.array([310, -60], np.float32), 0.62)]
    kept = RobustDetector(_Scripted(items)).detect_scored(image, 0.3, recover=False)[0]
    scores = sorted(round(float(s), 2) for _, _, s in kept)
    assert scores == [0.44, 0.62]          # the 0.44 clutter is gone; the 0.44 real face and the >= 0.5 detection stand
    assert any(np.array_equal(b, good_box) for b, _, _ in kept)


def test_an_ear_sized_box_mostly_inside_a_head_is_part_of_it_but_a_small_face_beside_it_is_not():
    head = (np.array([1174, 588, 2812, 1771], np.float32), np.zeros((5, 2), np.float32), 0.75)
    ear = (np.array([2457, 1376, 2788, 1938], np.float32), np.zeros((5, 2), np.float32), 0.59)       # 70 % inside, 10x smaller
    child = (np.array([2900, 700, 3300, 1200], np.float32), np.zeros((5, 2), np.float32), 0.7)        # beside it
    kept = _nms([head, ear, child])
    assert [round(float(s), 2) for _, _, s in kept] == [0.75, 0.7]
