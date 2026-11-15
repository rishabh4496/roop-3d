"""Stage 3: Kalman/GIoU motion, head-turn bridging, global tracklet consolidation and locked target binding.

Analytic scenarios with exact ground truth; no models are loaded.
"""
from __future__ import annotations

import numpy as np
import pytest

from roop_3d.engine.detection import track_policy as tp
from roop_3d.engine.detection.face_tracker import FaceObservation, FaceTracker, TrackedFace, optimal_assignment
from roop_3d.engine.detection.motion import BoxMotion, ConstantVelocityKalman, PoseTrajectory, giou
from roop_3d.engine.detection.track_policy import TrackPolicy
from roop_3d.identity.consolidation import TrackletRecorder, consolidate
from roop_3d.identity.locked_target import UNBOUND_BASE, LockedTargetBinder


def unit(vector):
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def basis(index: int) -> np.ndarray:
    vector = np.zeros(512, np.float32)
    vector[index] = 1
    return vector


def blend(base, other, cosine):
    return unit(cosine * base + np.sqrt(1 - cosine ** 2) * other)


LAYOUT = np.array(((0.3, 0.35), (0.7, 0.35), (0.5, 0.55), (0.35, 0.75), (0.65, 0.75)), np.float32)


def face(vector, left, top=0.0, size=40.0, yaw=0.0, score=1.0) -> FaceObservation:
    box = np.array((left, top, left + size, top + size), np.float32)
    return FaceObservation(box, (box[:2] + LAYOUT * size).astype(np.float32), np.zeros((106, 2), np.float32),
                           unit(vector), (yaw, 0.0, 0.0), score=score)


# ---------------------------------------------------------------------------------------------------- motion
def test_giou_ranks_disjoint_boxes_by_distance_and_matches_iou_when_identical():
    a = np.array((0, 0, 10, 10), np.float32)
    assert giou(a, a) == pytest.approx(1.0)
    near, far = np.array((12, 0, 22, 10), np.float32), np.array((60, 0, 70, 10), np.float32)
    assert -1.0 < giou(a, far) < giou(a, near) < 0.0


def test_kalman_box_extrapolates_constant_velocity_across_a_sparse_gap_and_coasts_with_decay():
    motion = BoxMotion(np.array((0, 0, 40, 40), np.float32), 0)
    for frame in range(1, 8):                                    # moves 5 px per frame
        motion.update(np.array((5 * frame, 0, 5 * frame + 40, 40), np.float32), frame)
    ahead = motion.predict_box(11)                                # no measurement for 4 frames
    assert abs(float(ahead[0]) - 55.0) < 6.0 and abs(float(ahead[2] - ahead[0]) - 40.0) < 2.0
    motion.coast(60)
    assert float(motion.predict_box(80)[0]) - float(motion.predict_box(60)[0]) < 5.0     # velocity has decayed, no run-away


def test_kalman_rejects_bad_configuration():
    with pytest.raises(ValueError):
        ConstantVelocityKalman([0.0, 0.0], accel_std=[1.0], measurement_std=[1.0, 1.0])


def test_pose_trajectory_flags_a_turn_before_the_embedding_collapses():
    trajectory = PoseTrajectory()
    for frame, yaw in enumerate((0, 8, 17, 27, 36)):
        trajectory.add(frame, yaw, 0, 0)
    assert trajectory.turning(5, 44.0, steep=40.0, step=15.0)           # steep now
    assert trajectory.turning(5, 36.0, steep=40.0, step=15.0)           # still below 40, but the trend reaches it
    still = PoseTrajectory()
    for frame in range(5):
        still.add(frame, 3.0, 0, 0)
    assert not still.turning(5, 4.0, steep=40.0, step=15.0)


def test_assignment_is_hungarian_and_only_admits_positive_scores():
    scores = np.array([[0.9, 0.8], [0.85, -1.0]])
    assert sorted(optimal_assignment(scores)) == [(0, 1), (1, 0)]       # globally best, not greedy
    assert optimal_assignment(np.full((2, 2), -1.0)) == []


# ---------------------------------------------------------------------------------------------------- tracker
def test_head_turning_into_profile_keeps_one_id_and_stays_swappable():
    a = basis(0)
    steps = [(0, 1.0), (10, 0.9), (22, 0.7), (35, 0.45), (48, 0.2), (62, 0.12), (75, 0.15), (80, 0.1)]
    tracker = FaceTracker(policy=TrackPolicy())
    history = []
    for number, (yaw, cosine) in enumerate(steps):
        history.append(tracker.assign([face(blend(a, basis(3), cosine), 50 + 2 * number, yaw=yaw)])[0])
    assert {item.track_id for item in history} == {0}
    assert all(item.swap_allowed for item in history[:5])
    assert all(item.swap_allowed and item.state == tp.TRACKED for item in history[5:]), [item.reason for item in history]


def test_an_embedding_collapse_without_a_turn_is_still_withheld():
    a = basis(0)
    tracker = FaceTracker(policy=TrackPolicy())
    for _ in range(3):
        tracker.assign([face(a, 50)])
    degraded = tracker.assign([face(blend(a, basis(3), 0.05), 52, yaw=2.0)])[0]
    assert degraded.track_id == 0 and not degraded.swap_allowed


def test_sparse_sampling_uses_real_frame_numbers_for_the_motion_prediction():
    a = basis(0)
    tracker = FaceTracker(policy=TrackPolicy())
    ids = []
    for frame in range(0, 40, 4):                        # every 4th frame, the face moves 8 px per frame
        tracker.frame_number = frame
        ids.append(tracker.assign([face(a, 10 + 8 * frame, size=60.0)])[0].track_id)
    assert set(ids) == {0}


def test_two_people_never_share_an_id_while_one_turns_away():
    a, b = basis(0), basis(1)
    tracker = FaceTracker(policy=TrackPolicy())
    ids = []
    for number in range(12):
        yaw = min(85.0, 9.0 * number)
        cosine = max(0.1, 1.0 - 0.1 * number)
        result = tracker.assign([face(blend(a, basis(5), cosine), 20 + number, yaw=yaw), face(b, 300 - number)])
        ids.append((result[0].track_id, result[1].track_id))
    assert {item[0] for item in ids} == {0} and {item[1] for item in ids} == {1}


# ---------------------------------------------------------------------------------------------------- consolidation
def feed(recorder, frame, track_id, vector, left, *, yaw=0.0, size=60.0, top=0.0, swap=True, reason=None):
    observation = face(vector, left, top=top, size=size, yaw=yaw)
    recorder.observe(frame, [TrackedFace(track_id, observation, 1, 0, tp.TRACKED, 1.0, swap, reason)])


def feed_many(recorder, frames, track_id, vector, left, **kwargs):
    for frame in frames:
        feed(recorder, frame, track_id, vector, left, **kwargs)


def test_one_person_split_at_a_profile_turn_is_one_canonical_id():
    a = basis(0)
    recorder = TrackletRecorder()
    feed_many(recorder, range(0, 40), 0, a, 100)
    feed_many(recorder, range(50, 100), 7, blend(a, basis(9), 0.12), 102, yaw=75.0)     # profile: embedding unrelated
    result = consolidate(recorder.tracklets())
    assert result.canonical(0) == result.canonical(7) == 1 and result.unique_ids == 1
    assert any(d.merged and d.reason == "continuity (turn)" for d in result.decisions)


def test_the_same_box_on_the_next_frame_is_one_object_even_if_the_embedding_changes_mode():
    recorder = TrackletRecorder()                         # a rolled / rendered head: ArcFace jumps to an unrelated mode
    feed_many(recorder, range(0, 60), 0, basis(0), 100)
    feed_many(recorder, range(61, 120), 1, basis(5), 101)
    result = consolidate(recorder.tracklets())
    assert result.unique_ids == 1 and any(d.merged and d.reason == "seamless" for d in result.decisions)
    identity = result.identities[0]
    assert max(float(identity.anchors[0] @ basis(k)) for k in (0, 5)) > 0.99 or len(identity.anchors) >= 2   # both modes kept


def test_a_shot_change_is_never_seamless():
    recorder = TrackletRecorder()
    feed_many(recorder, range(0, 60), 0, basis(0), 100)
    for frame in range(61, 120):
        observation = face(basis(5), 101, size=60.0)
        recorder.observe(frame, [TrackedFace(1, observation, 1, 1, tp.TRACKED, 1.0, True, None)])   # scene_id 1
    assert consolidate(recorder.tracklets()).unique_ids == 2


def test_person_leaving_and_returning_far_away_is_merged_by_appearance():
    a = basis(0)
    recorder = TrackletRecorder()
    feed_many(recorder, range(0, 30), 0, a, 100)
    feed_many(recorder, range(400, 440), 3, unit(a + 0.05 * basis(4)), 900)
    result = consolidate(recorder.tracklets())
    assert result.unique_ids == 1 and result.canonical(3) == 1
    assert any(d.reason == "appearance" for d in result.decisions if d.merged)


def test_a_stranger_in_the_same_place_is_a_new_person_not_a_merge():
    recorder = TrackletRecorder()
    feed_many(recorder, range(0, 30), 0, basis(0), 100)
    feed_many(recorder, range(35, 70), 1, basis(5), 100)
    result = consolidate(recorder.tracklets())
    assert result.unique_ids == 2 and result.canonical(0) != result.canonical(1)


def test_people_seen_together_are_never_merged_even_when_they_look_alike():
    a = basis(0)
    recorder = TrackletRecorder()
    for frame in range(30):
        feed(recorder, frame, 0, a, 100)
        feed(recorder, frame, 1, blend(a, basis(2), 0.9), 600)
    result = consolidate(recorder.tracklets())
    assert result.unique_ids == 2


def test_a_third_person_nearby_blocks_continuity_merging():
    a = basis(0)
    recorder = TrackletRecorder()
    feed_many(recorder, range(0, 30), 0, a, 100)
    feed_many(recorder, range(0, 60), 2, basis(6), 220)             # someone else stands beside the hand-over
    feed_many(recorder, range(35, 70), 1, blend(a, basis(3), 0.2), 104)
    result = consolidate(recorder.tracklets())
    assert result.canonical(0) != result.canonical(1)


def test_parts_of_a_larger_face_and_stray_detections_get_no_id():
    recorder = TrackletRecorder()
    for frame in range(40):
        feed(recorder, frame, 0, basis(0), 100, size=400.0)
        feed(recorder, frame, 5, basis(7), 150, size=60.0, top=80.0)        # an ear-sized box inside the head
    feed_many(recorder, range(10, 13), 9, basis(8), 700)                    # three stray hits
    result = consolidate(recorder.tracklets())
    assert result.unique_ids == 1 and result.canonical(5) is None and result.canonical(9) is None
    assert "part of that face" in result.suppressed[5] and "fewer than" in result.suppressed[9]


def test_canonical_ids_are_by_observation_count_and_deterministic():
    recorder = TrackletRecorder()
    feed_many(recorder, range(0, 20), 4, basis(1), 100)                     # appears first, seen less
    feed_many(recorder, range(30, 90), 2, basis(2), 400)
    first = consolidate(recorder.tracklets())
    again = consolidate(recorder.tracklets())
    assert first.canonical(2) == 1 and first.canonical(4) == 2
    assert first.canonical_of == again.canonical_of


def test_a_profile_prototype_does_not_vouch_for_a_frontal_face():
    a, other = basis(0), basis(1)
    track = tp.TrackMemory(0, None, tp.unit(a), [], np.zeros(4, np.float32), bins={3: tp.unit(other)})
    assert tp.appearance(track, other, yaw=80.0) == pytest.approx(1.0)
    assert tp.appearance(track, other, yaw=0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------------------- locked binding
def tracked(track_id, vector, left, yaw=0.0):
    return TrackedFace(track_id, face(vector, left, yaw=yaw), 1, 0)


def test_binder_relabels_with_canonical_ids_and_keeps_them_when_the_tracker_renumbers():
    a, b = basis(0), basis(1)
    binder = LockedTargetBinder({1: np.stack([a]), 2: np.stack([b])})
    first = binder.bind((tracked(0, a, 100), tracked(1, b, 400)), 0)
    assert [item.track_id for item in first] == [1, 2]
    # The tracker lost both and minted new IDs; the people are the same.
    second = binder.bind((tracked(5, b, 410), tracked(6, a, 110)), 40)
    assert [item.track_id for item in second] == [2, 1]
    third = binder.bind((tracked(5, b, 420),), 41)                         # sticky, and only one of them is visible
    assert third[0].track_id == 2


def test_binder_never_assigns_a_stranger_to_a_locked_id():
    binder = LockedTargetBinder({1: np.stack([basis(0)])})
    result = binder.bind((tracked(3, basis(9), 100),), 0)
    assert result[0].track_id == UNBOUND_BASE + 3
    # two faces, one canonical person: only the matching one is bound
    pair = binder.bind((tracked(3, basis(9), 100), tracked(4, basis(0), 400)), 1)
    assert sorted(item.track_id for item in pair) == [1, UNBOUND_BASE + 3]


def test_binder_follows_a_turn_by_spatial_continuity_when_the_embedding_collapses():
    a = basis(0)
    binder = LockedTargetBinder({1: np.stack([a])})
    binder.bind((tracked(0, a, 100, yaw=0.0),), 10)
    profile = blend(a, basis(5), 0.12)
    result = binder.bind((tracked(8, profile, 102, yaw=80.0),), 12)      # a new tracker ID at a steep yaw, same place
    assert result[0].track_id == 1


def test_binder_prepass_binds_a_split_track_by_the_global_decision_and_replays_it():
    a, b = basis(0), basis(1)
    profile = blend(a, basis(5), 0.1)                 # embedding of the turned head: no local evidence for the locked person
    binder = LockedTargetBinder({1: np.stack([a]), 2: np.stack([b])})
    steps = ([(frame, tracked(0, a, 100 + frame), tracked(9, b, 600)) for frame in range(0, 12)]
             + [(frame, tracked(4, profile, 100 + frame, yaw=80.0), tracked(9, b, 600)) for frame in range(14, 30)])
    binder.start_prepass()
    for frame, subject, other in steps:
        binder.bind((subject, other), frame)
    binder.finish_prepass()
    assert binder.prepass_summary["matched_to_lock"] == 2
    # Replay with a tracker that numbered everything differently and sees a mixed embedding on one face.
    for frame, subject, other in steps:
        replayed = binder.bind((tracked(70, profile if subject.observation.pose[0] else a, subject.observation.box[0], yaw=subject.observation.pose[0]),
                                tracked(71, b, 600)), frame)
        assert [item.track_id for item in replayed] == [1, 2], frame
    assert binder.stats["prepass"] >= 2 * len(steps)


# ---------------------------------------------------------------------------------------------------- inspection plumbing
def test_track_records_and_report_fold_into_canonical_ids():
    from roop_3d.identity.report import IdentityReportCollector
    from roop_3d.media.service import merge_track_records
    base = dict(first=0, last=9, first_at=0.0, last_at=0.3, box=np.zeros(4), pose_min=[-5, 0, 0], pose_max=[5, 0, 0], count=10,
                scene_id=0, thumbnail="a", observations=[])
    later = dict(base, first=20, last=29, first_at=0.7, last_at=1.0, pose_min=[-80, 0, 0], pose_max=[2, 1, 0], count=10, thumbnail="b")
    merged, sources = merge_track_records({0: base, 3: later, 4: dict(base)}, {0: 1, 3: 1, 4: None})
    assert list(merged) == [1] and sources == {1: (0, 3)}
    assert merged[1]["first"] == 0 and merged[1]["last"] == 29 and merged[1]["count"] == 20
    assert merged[1]["pose_min"][0] == -80 and merged[1]["thumbnail"] == "a"

    image = np.zeros((64, 64, 3), np.uint8)
    collector = IdentityReportCollector(TrackPolicy(), detection_threshold=0.4)
    for frame in range(0, 10):
        collector.observe(frame, frame / 30, image, [TrackedFace(0, face(basis(0), 10), 1, 0)])
    for frame in range(20, 30):
        collector.observe(frame, frame / 30, image, [TrackedFace(3, face(basis(2), 12, yaw=70.0), 1, 0)])
    collector.observe(40, 1.3, image, [TrackedFace(4, face(basis(5), 12), 1, 0)])
    collector.consolidate({0: 1, 3: 1, 4: None})
    document = collector.finalize(inspection_id="a" * 32, target_sha256="b" * 64, events=[], events_dropped=0,
                                  scene_cut_frames=[], decoded_frames=41, truncated=False)
    assert list(document["tracks"]) == ["track_0001"]
    track = document["tracks"]["track_0001"]
    assert track["visible_frames"] == 20 and [segment[:2] for segment in track["segments"]] == [[0, 9], [20, 29]]
    assert track["missing_ranges"] == [[10, 19]]


def test_guard_lowers_the_embedding_floor_only_for_steep_heads_and_rivals_still_veto():
    from types import SimpleNamespace
    from roop_3d.identity.guard import IdentityGuard, SIGNATURE_MISMATCH
    a, b = basis(0), basis(1)
    guard = IdentityGuard({1: np.stack([a]), 2: np.stack([b])}, {1: "f1", 2: "f2"}, frozenset({1}))
    weak = blend(a, basis(7), 0.15)                      # below the 0.25 floor, above the steep floor of 0.10
    def tracked_face(vector, yaw):
        return SimpleNamespace(track_id=1, observation=SimpleNamespace(embedding=vector, pose=(yaw, 0.0, 0.0)))
    assert guard.check(tracked_face(weak, 0.0)).reason == SIGNATURE_MISMATCH
    assert guard.check(tracked_face(weak, 80.0)).allowed
    rival = unit(0.2 * a + 0.5 * b)                      # steep, weak for 1 but clearly person 2
    assert not guard.check(tracked_face(rival, 80.0)).allowed
