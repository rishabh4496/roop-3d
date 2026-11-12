"""When a dense swap is drawn at all: a pose fade towards the camera limit, and a short temporal bridge over tracker gates.

* :class:`PoseFade` is the gate that matters. As a head turns away, the swap fades out smoothly with the angle off the camera
  axis and is gone before the tracker can lose the face (an extreme profile and a head turned away give the tracker no
  appearance to match); on the way back it fades in with the same curve. A latch keeps a face that reached the limit off
  until it is clearly back inside it. Nothing is drawn on the back of a head.
* :class:`SwapPresence` bridges a one/two-frame tracker-policy gate at a drawable pose with a short fade and fades a
  returning replacement in, so that gate does not pop the face either. It never raises a replacement above the pose fade.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from roop_3d.temporal import smoothstep

POSE_FADE_START = 65.0   # degrees off the camera axis where the swap starts to fade
POSE_FADE_END = 82.0     # ... and where it is gone (the tracker gives up on the face a few degrees later)

HOLD_FRAMES = 2          # frames a gated face keeps its last replacement, at 2/3 and 1/3 presence
RAMP_FRAMES = 3          # frames a returning replacement takes to reach full presence
FORGET_FRAMES = 30       # an absence longer than this is a new appearance, not a dropout


def off_axis_degrees(yaw: float, pitch: float) -> float:
    """Angle between the head's facing direction and the camera axis: ``acos(cos yaw cos pitch)``."""
    return math.degrees(math.acos(min(1.0, max(-1.0, math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))))))


class PoseFade:
    """Per-track swap presence from the filtered head pose (legacy ``(yaw, pitch, roll)`` tuple, degrees)."""

    def __init__(self):
        self._off: set[int] = set()

    def reset(self) -> None:
        self._off.clear()

    def retain(self, active_ids) -> None:
        self._off &= set(active_ids)

    def level(self, track_id: int, pose, hysteresis: float = 2.5) -> float:
        """0 (not drawn) .. 1 (full). Reaching the limit latches the face off until it is ``hysteresis`` degrees inside."""
        angle = off_axis_degrees(float(pose[0]), float(pose[1]))
        if track_id in self._off:
            if angle >= POSE_FADE_END - hysteresis:
                return 0.0
            self._off.discard(track_id)
        if angle >= POSE_FADE_END:
            self._off.add(track_id)
            return 0.0
        return 1.0 - smoothstep((angle - POSE_FADE_START) / (POSE_FADE_END - POSE_FADE_START))


@dataclass
class _State:
    level: float
    last_frame: int
    replacement: object


class SwapPresence:
    def __init__(self):
        self._states: dict[int, _State] = {}

    def reset(self) -> None:
        self._states.clear()

    def apply(self, frame: int, replacements, holdable) -> list:
        """Call once per frame, with ``replacements`` of this frame and ``holdable``: ids whose gate may be bridged.

        Returns what to composite: fresh replacements (their ``presence``, the pose fade, scaled by the ramp while a face
        fades back in) plus the last replacement of any holdable track that is gated this frame (its pose fade, scaled
        2/3 then 1/3).
        """
        result, present = [], set()
        for item in replacements:
            present.add(item.track_id)
            state = self._states.get(item.track_id)
            level = 1.0 if state is None else min(1.0, state.level + 1.0 / RAMP_FRAMES) if state.level < 1.0 else 1.0
            self._states[item.track_id] = _State(level, frame, item)
            result.append(item if level >= 1.0 else replace(item, presence=item.presence * level))
        for track_id, state in list(self._states.items()):
            if track_id in present:
                continue
            gap = frame - state.last_frame
            if gap > FORGET_FRAMES:
                del self._states[track_id]
            elif track_id in holdable and gap <= HOLD_FRAMES:
                state.level = 1.0 - gap / (HOLD_FRAMES + 1.0)
                result.append(replace(state.replacement, presence=state.replacement.presence * state.level))
            else:
                state.level = 0.0
        return result
