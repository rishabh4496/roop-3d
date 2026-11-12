"""Source-view selection: pose cost, hysteresis, and weights that cross-fade instead of popping."""
from types import SimpleNamespace

import numpy as np
import pytest

from roop_3d.reconstruction.view_selection import (
    MARGIN, SWITCH_FRAMES, ViewSelector, neutral_pitch, view_cost, wrap_degrees)

VIEWS = {0: (-60.0, 0.0, 0.0), 1: (-30.0, 0.0, 0.0), 2: (0.0, 0.0, 0.0), 3: (30.0, 0.0, 0.0), 4: (60.0, 0.0, 0.0)}


def costs_for(pose, quality=1.0):
    return {i: view_cost(pose, v, quality) for i, v in VIEWS.items()}


def run_sweep(poses, limit=2, **kwargs):
    selector, shares, sets = ViewSelector(), [], []
    for frame, pose in enumerate(poses):
        weights = selector.select(7, frame, costs_for(pose), limit, timestamp=frame / 30, **kwargs)
        total = sum(weights.values())
        shares.append({i: w / total for i, w in weights.items()})
        sets.append(frozenset(weights))
    return shares, sets


def max_share_step(shares):
    return max(max(abs(b.get(i, 0.0) - a.get(i, 0.0)) for i in set(a) | set(b)) for a, b in zip(shares, shares[1:]))


def test_pose_cost_matches_the_documented_formula():
    expected = (0.52 * (30 / 55) + 0.28 * (10 / 42) + 0.20 * (20 / 55)) * (0.50 + 0.18 * 0.5) + 0.18 * 0.25
    assert view_cost((30, 10, 20), (0, 0, 0), 0.75, 0.5) == pytest.approx(expected)
    assert view_cost((0, 0, 179), (0, 0, -179), 1.0) == pytest.approx(0.20 * (2 / 55) * 0.68)   # roll wraps at 180
    assert view_cost((180, 0, 0), (0, 0, 0), 1.0) == pytest.approx(0.52 * 2 * 0.68)              # capped at 2x


def test_gimbal_lock_hands_the_distance_to_the_rotation_matrices():
    near, far = view_cost((0, 80, 0), (0, 80, 0), 1.0, rotation_distance=0.1), view_cost((0, 80, 0), (0, 80, 0), 1.0, rotation_distance=1.5)
    assert near < far and far - near > 0.3


def test_neutral_pitch_is_the_median_of_near_frontal_views_only():
    views = [SimpleNamespace(yaw=y, pitch=p) for y, p in ((0, 19), (10, 21), (-20, 20), (80, -50))]
    assert neutral_pitch(views) == 20.0
    assert neutral_pitch([SimpleNamespace(yaw=70, pitch=5)]) == 0.0


def test_pose_noise_around_a_tie_does_not_switch_views():
    rng = np.random.default_rng(2)
    poses = [(15.0 + rng.normal(0, 1.2), 0.0, 0.0) for _ in range(150)]    # exactly between views 2 (0 deg) and 3 (30 deg)
    _, sets = run_sweep(poses)
    switches = sum(a != b for a, b in zip(sets, sets[1:]))
    assert switches == 0, "views 2 and 3 are both selected; jitter between 3rd/4th ranks must not churn the set"
    _, sets = run_sweep([(45.0 + rng.normal(0, 3.0), 0.0, 0.0) for _ in range(300)])
    assert sum(a != b for a, b in zip(sets, sets[1:])) <= 3        # per 300 frames, on a deliberate tie between 3 and 4


def test_a_clear_winner_needs_the_margin_for_several_frames():
    selector = ViewSelector()
    start = {0: 0.20, 1: 0.30, 2: 0.50}
    selector.select(1, 0, start, 2, timestamp=0.0)                 # views 0, 1 active
    rival = {0: 0.20, 1: 0.30, 2: 0.30 - MARGIN - 0.001}
    seen = []
    for frame in range(1, 6):
        seen.append(frozenset(selector.select(1, frame, rival, 2, timestamp=frame / 30)))
    # the rival beats the worst active view by the margin from frame 1; it enters on the SWITCH_FRAMES-th frame
    assert 2 not in seen[SWITCH_FRAMES - 2] and 2 in seen[SWITCH_FRAMES - 1]
    short = {0: 0.20, 1: 0.30, 2: 0.30 - MARGIN + 0.002}            # inside the margin: never switches
    selector.select(2, 0, start, 2, timestamp=0.0)
    assert all(2 not in selector.select(2, f, short, 2, timestamp=f / 30) for f in range(1, 12))


def test_weights_crossfade_in_quarter_steps_over_a_yaw_sweep():
    poses = [(yaw, 0.0, 0.0) for yaw in np.linspace(0, 75, 160)]
    shares, sets = run_sweep(poses)
    switches = sum(a != b for a, b in zip(sets, sets[1:]))
    assert switches <= 3 * 160 / 100 + 3                         # a continuous 0 -> 75 deg sweep: a handful of hand-offs
    assert max_share_step(shares) <= 0.25 + 1e-9
    assert all(len(s) >= 2 for s in sets)


def test_with_noise_on_the_sweep_weights_stay_continuous():
    rng = np.random.default_rng(4)
    poses = [(yaw + rng.normal(0, 1.5), rng.normal(0, 1.0), 0.0) for yaw in np.linspace(0, 75, 200)]
    shares, sets = run_sweep(poses, angular={i: 1.0 for i in VIEWS})
    assert max_share_step(shares) <= 0.25 + 1e-9
    assert sum(a != b for a, b in zip(sets, sets[1:])) <= 8


def test_gap_scene_cut_and_one_off_preview_choose_the_best_views_at_once():
    selector = ViewSelector()
    selector.select(0, 0, costs_for((0, 0, 0)), 2, timestamp=0.0)
    far = costs_for((60, 0, 0))
    assert set(selector.select(0, 5, far, 2, timestamp=5 / 30)) == {3, 4}            # frame gap
    selector.select(0, 6, costs_for((0, 0, 0)), 2, timestamp=6 / 30)
    assert set(selector.select(0, 7, far, 2, timestamp=7 / 30, scene_cut=True)) == {3, 4}
    assert set(selector.select(0, None, far, 2)) == {3, 4}                           # frame_index=None: stateless


def test_quality_breaks_a_pose_tie():
    assert min(costs_for((15, 0, 0), 1.0), key=costs_for((15, 0, 0), 1.0).get) in (2, 3)
    mixed = {i: view_cost((0, 0, 0), (0, 0, 0), q) for i, q in ((0, 0.4), (1, 0.9))}
    assert mixed[1] < mixed[0]


def test_wrap_degrees():
    assert wrap_degrees(190) == pytest.approx(-170) and wrap_degrees(-190) == pytest.approx(170)
