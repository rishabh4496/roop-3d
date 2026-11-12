"""Which source photographs condition a frame, with hysteresis and weights that fade instead of switching.

A hard top-k by distance to the target pose flips a view in or out whenever two rank next to each other; the pixel blend of
the photographs then changes colour and texture in one frame. Roop Ultimate's selector (``pose_source_selector``) keeps the
previous view unless a rival is better by a margin; the same rule is applied here to each slot of the top-k, with the
weight of a view entering or leaving the set ramping over a few frames.

    pose_cost = .52 min(2, |dyaw| / 55) + .28 min(2, |dpitch| / 42) + .20 min(2, wrap |droll| / 55)
    total     = pose_cost (.50 + .18 confidence) + .18 (1 - quality)

A view enters only if ``total + MARGIN`` beats the slot it replaces for ``SWITCH_FRAMES`` consecutive frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MARGIN = 0.035
SWITCH_FRAMES = 3
RAMP_STEP = 0.25               # a view's weight moves by this per frame: 4 frames from absent to full
SMOOTHING = 0.35               # EMA on ramp * angular weight
FADE_FLOOR = 0.05              # a fading view is dropped once its smoothed weight falls below this
GIMBAL_PITCH = 75.0            # beyond it Euler angles stop being a reliable distance; the rotation matrices decide
MAX_GAP_SECONDS = 0.25


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def neutral_pitch(views) -> float:
    """Median pitch of a FaceSet's near-frontal views (|yaw| <= 25 degrees): its anatomical zero for the 5-point pitch."""
    pitches = [float(v.pitch) for v in views if abs(float(v.yaw)) <= 25.0]
    return float(np.median(pitches)) if pitches else 0.0


def view_cost(target_pose, view_pose, quality: float, confidence: float = 1.0, *, rotation_distance: float = 0.0) -> float:
    """Selection cost of one source view for a target pose; both poses are legacy ``(yaw, pitch, roll)`` degrees.

    ``rotation_distance`` (Frobenius distance of the two fitted rotations) replaces the Euler terms on a near-vertical
    head, where yaw and roll are not independent, and breaks exact ties elsewhere.
    """
    if abs(target_pose[1]) > GIMBAL_PITCH:
        pose = min(2.0, rotation_distance / 1.6)
    else:
        pose = (0.52 * min(2.0, abs(target_pose[0] - view_pose[0]) / 55.0)
                + 0.28 * min(2.0, abs(target_pose[1] - view_pose[1]) / 42.0)
                + 0.20 * min(2.0, abs(wrap_degrees(target_pose[2] - view_pose[2])) / 55.0))
    return pose * (0.50 + 0.18 * float(np.clip(confidence, 0.0, 1.0))) + 0.18 * (1.0 - float(np.clip(quality, 0.0, 1.0))) \
        + 1e-3 * rotation_distance


@dataclass
class _Track:
    frame: int
    time: float | None
    active: list
    ramps: dict = field(default_factory=dict)
    smoothed: dict = field(default_factory=dict)
    streak: dict = field(default_factory=dict)


class ViewSelector:
    """Per-track hysteretic selection of the ``limit`` best source views and their fading weights."""

    def __init__(self):
        self.tracks: dict[int, _Track] = {}

    def reset(self) -> None:
        self.tracks.clear()

    def retain(self, active_ids) -> None:
        keep = set(active_ids)
        self.tracks = {key: value for key, value in self.tracks.items() if key in keep}

    def select(self, track_id: int, frame: int | None, costs: dict, limit: int, *, angular: dict | None = None,
               timestamp: float | None = None, scene_cut: bool = False) -> dict:
        """``{view index: weight factor in (0, 1]}`` for this frame.

        ``costs`` maps every candidate view index to its :func:`view_cost`; ``angular`` (index -> weight in 0..1) is the
        view's rotation proximity and is smoothed together with its ramp. ``frame=None`` is a one-off preview: the plain
        best ``limit``, no state.
        """
        angular = angular or {}
        ranked = sorted(costs, key=lambda index: (costs[index], index))
        best = ranked[:limit]
        state = self.tracks.get(track_id)
        if frame is None:
            return {index: float(angular.get(index, 1.0)) for index in best}
        contiguous = (state is not None and not scene_cut and frame == state.frame + 1
                      and (timestamp is None or state.time is None or 0 < timestamp - state.time <= MAX_GAP_SECONDS)
                      and set(state.active) <= set(costs))
        if not contiguous:
            state = _Track(frame, timestamp, list(best), {i: 1.0 for i in best},
                           {i: float(angular.get(i, 1.0)) for i in best})
            self.tracks[track_id] = state
            return dict(state.smoothed)
        state.frame, state.time = frame, timestamp
        self._rotate(state, costs, best, limit)
        weights = {}
        for index in set(state.ramps) | set(state.active):
            target = 1.0 if index in state.active else 0.0
            ramp = state.ramps.get(index, 0.0)
            ramp = min(target, ramp + RAMP_STEP) if target > ramp else max(target, ramp - RAMP_STEP)
            factor = ramp * float(angular.get(index, 1.0))
            previous = state.smoothed.get(index)
            smoothed = factor if previous is None else SMOOTHING * factor + (1.0 - SMOOTHING) * previous
            if index not in state.active and smoothed < FADE_FLOOR:
                state.ramps.pop(index, None)
                state.smoothed.pop(index, None)
                continue
            state.ramps[index], state.smoothed[index] = ramp, smoothed
            weights[index] = smoothed
        return weights

    @staticmethod
    def _rotate(state: _Track, costs: dict, best: list, limit: int) -> None:
        for index in best:                        # free slots (fewer than ``limit`` views so far) fill at once
            if len(state.active) < limit and index not in state.active:
                state.active.append(index)
        worst = max(state.active, key=lambda index: (costs[index], index))
        rivals = [index for index in best if index not in state.active and costs[index] + MARGIN < costs[worst]]
        state.streak = {index: state.streak.get(index, 0) + 1 for index in rivals}
        ready = [index for index in rivals if state.streak[index] >= SWITCH_FRAMES]
        if ready and len(state.active) >= limit:
            winner = min(ready, key=lambda index: (costs[index], index))
            state.active.remove(worst)
            state.active.append(winner)
            state.streak = {}
