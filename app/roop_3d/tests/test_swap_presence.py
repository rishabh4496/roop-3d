"""A tracker gate that closes for a frame or two fades the swap instead of popping it."""
from dataclasses import dataclass, replace

import numpy as np
import pytest

from roop_3d.engine.volumetric.presence import FORGET_FRAMES, SwapPresence


@dataclass(frozen=True)
class Rep:
    track_id: int
    marker: int = 0
    presence: float = 1.0


def levels(plan, holdable=frozenset({0})):
    """plan: per frame, True when the replacement exists; returns the presence composited each frame (None = nothing)."""
    presence, out = SwapPresence(), []
    for frame, here in enumerate(plan):
        result = presence.apply(frame, [Rep(0, frame)] if here else [], holdable)
        out.append(result[0].presence if result else None)
    return out


def test_one_frame_gate_is_bridged_and_faded_back_in():
    assert levels([1, 1, 0, 1, 1]) == [1.0, 1.0, pytest.approx(2 / 3), 1.0, 1.0]


def test_steps_are_never_larger_than_a_third():
    for plan in ([1, 1, 0, 0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 1]):
        seen = [v or 0.0 for v in levels(plan)]
        assert max(abs(np.diff(seen))) <= 1 / 3 + 1e-9, (plan, seen)


def test_hold_lasts_two_frames_then_nothing_and_returns_gradually():
    assert levels([1, 0, 0, 0, 0, 1, 1, 1, 1]) == [1.0, pytest.approx(2 / 3), pytest.approx(1 / 3), None, None,
                                                   pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0, 1.0]


def test_held_replacement_is_the_last_fresh_one_and_only_for_holdable_tracks():
    presence = SwapPresence()
    presence.apply(0, [Rep(0, 41), Rep(1, 7)], {0, 1})
    held = presence.apply(1, [], {0})
    assert [(r.track_id, r.marker) for r in held] == [(0, 41)]      # track 1 was withheld for identity reasons: no hold
    assert presence.apply(2, [Rep(1, 9)], set())[0].presence == pytest.approx(1 / 3)


def test_new_tracks_and_long_absences_appear_at_full_strength():
    presence = SwapPresence()
    assert presence.apply(0, [Rep(3)], set())[0].presence == 1.0
    presence.apply(1, [], set())
    for frame in range(2, FORGET_FRAMES + 3):
        presence.apply(frame, [], set())
    assert presence.apply(FORGET_FRAMES + 3, [Rep(3)], set())[0].presence == 1.0


def test_session_roll_latch_only_releases_after_the_hysteresis_margin(monkeypatch):
    from types import SimpleNamespace
    from roop_3d.engine.volumetric.session import ReferencePipeline
    monkeypatch.delenv("ROOP3D_3D_MAX_ROLL", raising=False)
    stub = SimpleNamespace(_roll_latched={}, pose_hysteresis_degrees=2.5)
    decide = lambda roll: ReferencePipeline._roll_exceeded(stub, 0, roll)
    assert [decide(r) for r in (39.9, 40.5, 39.0, 38.0, 37.4, 39.9)] == [False, True, True, True, False, False]
    assert decide(-41.0) is True                       # sign-independent


def test_profile_trim_fades_in_with_yaw_instead_of_switching_on_at_a_threshold():
    from roop_3d.engine.volumetric.session import profile_aware_mask
    five = np.array(((40, 50), (88, 50), (64, 70), (45, 95), (83, 95)), np.float64)
    far_cheek = lambda yaw: profile_aware_mask((128, 128), five, yaw)[64, 122 if yaw < 0 else 5]
    assert profile_aware_mask((128, 128), five, 30.0).min() == 1.0                 # nothing below the start
    trims = [1.0 - far_cheek(yaw) for yaw in (31, 33, 35, 37, 39, 41, 60)]
    assert trims == sorted(trims) and trims[0] < 0.2 and trims[-1] > 0.95
    assert max(np.diff([0.0, *trims])) < 0.35                                      # no single-degree jump
    np.testing.assert_allclose(profile_aware_mask((128, 128), five, 40.0), profile_aware_mask((128, 128), five, 80.0))
