"""Identity-lock tracking policy: analytic scenarios with exact ground truth.

The invariant exercised everywhere is the product guarantee: whenever the tracker
says ``swap_allowed`` for a face, the track ID it reports is the ID that person was
given. Uncertainty may withhold swaps; it may never relabel a person silently.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from roop_3d.engine.detection import track_policy as tp
from roop_3d.engine.detection.face_tracker import FaceObservation, FaceTracker
from roop_3d.engine.detection.track_policy import TrackPolicy, propagate_box


def unit(vector):
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def basis(index: int) -> np.ndarray:
    vector = np.zeros(512, np.float32)
    vector[index] = 1
    return vector


def blend(base: np.ndarray, other: np.ndarray, cosine: float) -> np.ndarray:
    """Unit vector whose cosine with ``base`` is exactly ``cosine`` (``other`` must be orthogonal)."""
    return unit(cosine * base + np.sqrt(1 - cosine ** 2) * other)


LAYOUT = np.array(((0.3, 0.35), (0.7, 0.35), (0.5, 0.55), (0.35, 0.75), (0.65, 0.75)), np.float32)


def face(vector: np.ndarray, left: float, top: float = 0.0, size: float = 40.0) -> FaceObservation:
    box = np.array((left, top, left + size, top + size), np.float32)
    return FaceObservation(box, (box[:2] + LAYOUT * size).astype(np.float32), np.zeros((106, 2), np.float32),
                           unit(vector), (0, 0, 0))


def make(**kwargs) -> FaceTracker:
    return FaceTracker(policy=TrackPolicy(), **kwargs)


def run(tracker, frames):
    """frames: list of list[(label, FaceObservation)] -> list of {label: TrackedFace}."""
    history = []
    for frame in frames:
        tracked = tracker.assign([item[1] for item in frame])
        history.append({frame[index][0]: tracked[index] for index in range(len(frame))})
    return history


def assert_no_silent_switch(history, truth: dict[str, int]):
    for number, frame in enumerate(history):
        for label, tracked in frame.items():
            if tracked.swap_allowed:
                assert tracked.track_id == truth[label], f"frame {number}: {label} swapped on track {tracked.track_id}"


def test_policy_validation_and_fingerprint():
    base = TrackPolicy()
    assert base.fingerprint() == TrackPolicy().fingerprint()
    assert TrackPolicy(match_cosine=0.4).fingerprint() != base.fingerprint()
    assert base.as_dict()["version"] == tp.POLICY_VERSION
    for bad in (dict(match_cosine=0.1), dict(strong_cosine=1.0), dict(reacquire_frames=0), dict(gallery_rate=1.0),
                dict(ambiguity_margin=0.0), dict(lost_limit=0)):
        with pytest.raises(ValueError):
            TrackPolicy(**bad)


def test_legacy_tracker_is_unchanged_and_always_swappable():
    tracker = FaceTracker(max_missed=2)
    result = tracker.assign([face(basis(0), 0)])
    assert tracker.policy is None and result[0].swap_allowed and result[0].state == tp.TRACKED
    tracker.mark_scene_cut()
    assert not tracker.active_ids and tracker.scene_id == 1
    assert tracker.assign([face(basis(0), 0)])[0].track_id == 1      # legacy: a cut forgets identity


def test_distinct_people_crossing_keep_ids_and_keep_swapping():
    a, b = basis(0), basis(1)
    frames = [[("A", face(a, 20 * step)), ("B", face(b, 200 - 20 * step))] for step in range(11)]
    history = run(make(), frames)
    assert_no_silent_switch(history, {"A": 0, "B": 1})
    assert all(frame["A"].track_id == 0 and frame["B"].track_id == 1 for frame in history)
    assert all(frame["A"].swap_allowed and frame["B"].swap_allowed for frame in history[1:])


def test_similar_looking_people_crossing_never_switch_silently():
    a = basis(0)
    b = blend(a, basis(1), 0.96)         # gap to A (0.04) is below the ambiguity margin
    rng = np.random.default_rng(7)
    def noisy(vector):
        return unit(vector + rng.normal(0, 0.002, 512).astype(np.float32))
    frames = [[("A", face(noisy(a), 20 * step)), ("B", face(noisy(b), 200 - 20 * step))] for step in range(11)]
    history = run(make(), frames)
    assert_no_silent_switch(history, {"A": 0, "B": 1})
    assert history[0]["A"].swap_allowed and history[0]["B"].swap_allowed
    reasons = {tracked.reason for frame in history for tracked in frame.values()}
    assert tp.AMBIGUOUS in reasons                       # the crossing was noticed, not guessed
    assert not history[-1]["A"].swap_allowed or history[-1]["A"].track_id == 0


def test_similar_people_far_apart_keep_working_by_spatial_continuity():
    a = basis(0)
    b = blend(a, basis(1), 0.9)
    frames = [[("A", face(a, 0 + step)), ("B", face(b, 400 + step))] for step in range(20)]
    history = run(make(), frames)
    assert_no_silent_switch(history, {"A": 0, "B": 1})
    assert all(frame["A"].swap_allowed and frame["B"].swap_allowed for frame in history)


def test_ambiguity_clears_when_appearance_separates_the_people():
    a, b = basis(0), basis(1)
    mixed = unit(a + b + 1.5 * basis(7))  # overlapped face: ~0.49 to both people, below the gallery-adaptation floor
    frames = []
    for step in range(21):               # A walks right, B walks left, they overlap around step 10
        overlapped = 7 <= step <= 13
        frames.append([("A", face(a, 10 * step)), ("B", face(mixed if overlapped else b, 200 - 10 * step))])
    history = run(make(), frames)
    held = [number for number, frame in enumerate(history) if not frame["B"].swap_allowed]
    assert held and min(held) >= 7 and tp.AMBIGUOUS in {frame["B"].reason for frame in history}
    assert_no_silent_switch(history, {"A": 0, "B": 1})
    assert history[-1]["A"].swap_allowed and history[-1]["B"].swap_allowed and history[-1]["B"].track_id == 1


def test_person_leaves_and_reenters_is_reacquired_after_probation():
    a = basis(0)
    tracker = make(max_missed=5)
    run(tracker, [[("A", face(a, 10))] for _ in range(4)])
    for _ in range(20):                                 # absent far longer than max_missed
        assert tracker.assign([]) == ()
    assert tracker.track_states() == {0: tp.LOST}
    returned = [tracker.assign([face(unit(a + 0.01 * basis(3)), 300)])[0] for _ in range(6)]
    assert [item.track_id for item in returned] == [0] * 6
    assert returned[0].state == tp.REACQUIRED and returned[0].reason == tp.REACQUIRING
    assert [item.swap_allowed for item in returned] == [False, False, False, True, True, True]
    assert returned[3].state == tp.TRACKED
    assert [(e.track_id, e.state) for e in tracker.events if e.state in (tp.LOST, tp.REACQUIRED, tp.TRACKED)][-3:] == [
        (0, tp.LOST), (0, tp.REACQUIRED), (0, tp.TRACKED)]


def test_a_stranger_is_never_linked_to_a_lost_track():
    tracker = make(max_missed=2)
    run(tracker, [[("A", face(basis(0), 10))] for _ in range(3)])
    for _ in range(5):
        tracker.assign([])
    stranger = tracker.assign([face(basis(5), 10)])[0]       # same place, different person
    assert stranger.track_id != 0 and stranger.state == tp.TRACKED
    assert tracker.track_states()[0] == tp.LOST


def test_reacquisition_refuses_when_two_lost_people_look_alike():
    a = basis(0)
    b = blend(a, basis(1), 0.95)
    tracker = make(max_missed=1)
    run(tracker, [[("A", face(a, 10)), ("B", face(b, 200))] for _ in range(3)])
    for _ in range(4):
        tracker.assign([])
    comeback = tracker.assign([face(unit(a + b), 100)])[0]       # equally close to both lost tracks
    assert comeback.track_id not in (0, 1) and comeback.state == tp.TRACKED


def test_hand_over_face_with_box_kept_is_uncertain_not_a_new_person():
    a = basis(0)
    degraded = blend(a, basis(9), 0.05)                     # embedding destroyed, box continuous
    frames = [[("A", face(a, 50))] for _ in range(4)]
    frames += [[("A", face(degraded, 52))] for _ in range(3)]
    frames += [[("A", face(a, 54))] for _ in range(5)]
    history = run(make(), frames)
    assert {frame["A"].track_id for frame in history} == {0}
    assert all(not frame["A"].swap_allowed and frame["A"].state == tp.UNCERTAIN for frame in history[4:7])
    assert [frame["A"].swap_allowed for frame in history[7:]] == [False, False, True, True, True]
    assert history[4]["A"].reason == tp.APPEARANCE_LOW and history[7]["A"].reason == tp.RECOVERING


def test_hand_over_face_with_detection_dropout_returns_through_a_short_recovery():
    a = basis(0)
    tracker = make()
    run(tracker, [[("A", face(a, 50))] for _ in range(4)])
    for _ in range(3):
        assert tracker.assign([]) == ()
    assert tracker.track_states() == {0: tp.LOST}
    back = [tracker.assign([face(a, 52)])[0] for _ in range(4)]
    assert [item.track_id for item in back] == [0, 0, 0, 0]
    assert [item.swap_allowed for item in back] == [False, False, True, True]
    assert back[0].state == tp.REACQUIRED and back[2].state == tp.TRACKED


def test_partial_occlusion_with_degraded_but_consistent_appearance_keeps_the_lock():
    a = basis(0)
    partial = blend(a, basis(4), 0.5)                       # half-covered face: lower but still plausible
    frames = [[("A", face(a, 50))] for _ in range(3)] + [[("A", face(partial, 51))] for _ in range(4)]
    history = run(make(), frames)
    assert all(frame["A"].swap_allowed and frame["A"].track_id == 0 for frame in history)


def test_camera_angle_change_uses_spatial_continuity_but_not_below_the_floor():
    a = basis(0)
    profile = blend(a, basis(6), 0.3)                       # between continuity (.25) and match (.35)
    history = run(make(), [[("A", face(a, 50))]] * 3 + [[("A", face(profile, 52))]] * 3)
    assert all(frame["A"].swap_allowed for frame in history)
    far = blend(a, basis(6), 0.2)                           # below the continuity floor
    history = run(make(), [[("A", face(a, 50))]] * 3 + [[("A", face(far, 52))]] * 3)
    assert all(not frame["A"].swap_allowed and frame["A"].track_id == 0 for frame in history[3:])


def test_scene_cut_parks_tracks_and_only_reacquires_by_strict_appearance():
    a, b = basis(0), basis(1)
    tracker = make()
    run(tracker, [[("A", face(a, 10)), ("B", face(b, 200))] for _ in range(3)])
    scene = tracker.scene_id
    tracker.mark_scene_cut()
    assert tracker.scene_id == scene + 1 and not tracker.active_ids
    assert set(tracker.track_states().values()) == {tp.LOST}
    after = tracker.assign([face(b, 20), face(basis(8), 300)])       # B moved; a stranger appears
    by_vector = {item.observation.embedding.argmax(): item for item in after}
    assert by_vector[1].track_id == 1 and by_vector[1].state == tp.REACQUIRED and not by_vector[1].swap_allowed
    assert by_vector[8].track_id not in (0, 1)
    assert by_vector[1].scene_id == scene + 1
    assert any(event.reason == tp.SCENE_CUT for event in tracker.events)


def test_missing_faces_are_reported_as_lost_not_filled_in():
    tracker = make()
    run(tracker, [[("A", face(basis(0), 10))]])
    assert tracker.assign([]) == ()
    assert tracker.track_states() == {0: tp.LOST}
    assert [(event.track_id, event.state, event.reason) for event in tracker.events] == [(0, tp.LOST, tp.NOT_DETECTED)]


def test_uncertain_track_times_out_to_lost_instead_of_lingering():
    policy = TrackPolicy(uncertain_limit=3)
    tracker = FaceTracker(policy=policy)
    run(tracker, [[("A", face(basis(0), 50))]])
    degraded = blend(basis(0), basis(9), 0.05)
    run(tracker, [[("A", face(degraded, 50))] for _ in range(5)])
    assert tracker.track_states().get(0) == tp.LOST
    assert any(event.reason == tp.UNCERTAIN_TIMEOUT for event in tracker.events)


def test_new_people_get_new_ids_and_budgets_hold():
    tracker = make(max_tracks=2)
    first = tracker.assign([face(basis(0), 0), face(basis(1), 100)])
    assert [item.track_id for item in first] == [0, 1]
    with pytest.raises(Exception):
        tracker.assign([face(basis(i), 50 * i) for i in range(3)])


def test_released_tracker_refuses_work():
    tracker = make()
    tracker.release()
    with pytest.raises(Exception):
        tracker.assign([])


def test_optical_flow_propagates_a_box_with_the_scene_motion():
    rng = np.random.default_rng(3)
    canvas = cv2.GaussianBlur((rng.random((240, 320)) * 255).astype(np.uint8), (5, 5), 1.2)
    shifted = np.roll(canvas, (3, 7), axis=(0, 1))
    box = np.array((100, 80, 180, 160), np.float32)
    moved, trust = propagate_box(canvas, shifted, box)
    assert trust > 0.5 and np.allclose(moved, box + np.array((7, 3, 7, 3)), atol=1.5)
    flat = np.full((240, 320), 120, np.uint8)
    assert propagate_box(flat, flat, box) is None          # no features: caller keeps its velocity prediction


def test_process_updates_flow_frames_and_cut_resets_them():
    tracker = make()
    image = np.zeros((120, 160, 3), np.uint8)
    tracker._update_flow_frames(image, False)
    assert tracker._flow_frames is None                    # first frame has no predecessor
    tracker._update_flow_frames(image, False)
    assert tracker._flow_frames is not None
    tracker._update_flow_frames(image, True)
    assert tracker._flow_frames is None
