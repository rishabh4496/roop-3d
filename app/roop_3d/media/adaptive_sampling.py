"""Adaptive frame sampling for target-face inspection: the whole timeline, dense where the picture changes.

Running the detector + ArcFace + landmark stack on every frame costs ~0.1-0.5 s per face-frame, so a long clip cannot be
scanned frame by frame inside an interactive budget, and a fixed head-of-file limit sees only the first seconds.  The
sampler walks the decoded timeline once and decides, from what the previous *sampled* frame showed, how far to jump:

* a stable scene relaxes the stride geometrically (1, 2, 4, ...) up to ``max_stride`` (``face_interval`` seconds while a
  face is present, ``empty_interval`` while none is, so a late appearance is at most that late);
* anything that can hide a pose excursion collapses the stride back to 1 for a short burst: a scene cut, a track that
  appears or disappears (including a detection dropout), a yaw / pitch / roll jump between samples, a box that moved or
  rescaled by more than the stride can bridge, or a head already past ``steep_yaw`` / ``steep_pitch`` (3/4 and profile
  frames are sampled at most ``steep_stride`` apart because that is where recall is lost);
* when the measured cost per sample would overrun the time budget, the stride ceiling is raised so the sweep still reaches
  the end of the video instead of stopping early.

Pure logic (no model, no decode), so it is unit-testable with synthetic snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FaceSnapshot:
    """What one sampled frame showed for one tracked face (pose in degrees; box normalised to the frame)."""
    track_id: int
    yaw: float
    pitch: float
    roll: float
    cx: float
    cy: float
    size: float


@dataclass(frozen=True)
class SamplerPolicy:
    face_interval: float = 0.25            # seconds: widest gap while a stable face is on screen
    empty_interval: float = 0.5            # seconds: widest gap while no face is on screen
    burst: int = 6                         # samples at stride 1 after a change
    pose_step: float = 8.0                 # degrees of yaw / pitch / roll change between samples that counts as a change
    steep_yaw: float = 30.0
    steep_pitch: float = 22.0
    steep_stride: int = 2
    move_fraction: float = 0.6             # centre shift as a fraction of the face size
    scale_ratio: float = 1.35
    budget_reserve: float = 0.1            # share of the time budget kept back for finishing work
    max_gap_seconds: float = 8.0           # the widest jump a tight time budget may force between two samples


class AdaptiveSampler:
    def __init__(self, fps: float, total_frames: int | None, policy: SamplerPolicy = SamplerPolicy()):
        self.fps = fps if fps and fps > 0 else 30.0
        self.total = total_frames if total_frames and total_frames > 0 else None
        self.policy = policy
        self.face_cap = max(1, round(self.fps * policy.face_interval))
        self.empty_cap = max(1, round(self.fps * policy.empty_interval))
        self.stride = 1
        self.burst_left = 0
        self.next_index = 0
        self.budget_floor = 1                 # minimum stride the time budget currently demands
        self.budget_cap = max(1, round(self.fps * policy.max_gap_seconds))
        self.samples = 0
        self._previous: dict[int, FaceSnapshot] | None = None
        self.reasons: dict[str, int] = {}

    def wants(self, index: int) -> bool:
        return index >= self.next_index

    def _changed(self, previous: dict[int, FaceSnapshot], current: dict[int, FaceSnapshot], cut: bool) -> str | None:
        p = self.policy
        if cut:
            return "scene_cut"
        if set(previous) != set(current):
            return "track_set_changed"
        for key, now in current.items():
            before = previous[key]
            if max(abs(now.yaw - before.yaw), abs(now.pitch - before.pitch), abs(now.roll - before.roll)) >= p.pose_step:
                return "pose_change"
            reach = max(now.size, before.size, 1e-6)
            if math.hypot(now.cx - before.cx, now.cy - before.cy) > p.move_fraction * reach:
                return "motion"
            if max(now.size, before.size) / max(min(now.size, before.size), 1e-6) > p.scale_ratio:
                return "rescale"
        return None

    def update(self, index: int, faces: tuple[FaceSnapshot, ...], *, scene_cut: bool = False,
               elapsed: float = 0.0, budget: float | None = None) -> None:
        """Record the sample taken at ``index`` and schedule the next one."""
        p = self.policy
        self.samples += 1
        current = {face.track_id: face for face in faces}
        reason = None
        if self._previous is None:
            reason = "start" if current else None
        else:
            reason = self._changed(self._previous, current, scene_cut)
        self._previous = current
        if reason is not None:
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
            self.stride, self.burst_left = 1, p.burst
        elif self.burst_left > 0:
            self.burst_left -= 1
        else:
            self.stride = min(self.stride * 2, self.face_cap if current else self.empty_cap)
        step = self.stride
        if any(abs(f.yaw) >= p.steep_yaw or abs(f.pitch) >= p.steep_pitch for f in faces):
            step = min(step, p.steep_stride)
        self.next_index = index + max(1, step, min(self.budget_floor, self.budget_cap))
        if budget is not None and self.total is not None and self.samples >= 4 and elapsed > 0:
            self._fit_budget(index, elapsed, budget)

    def _fit_budget(self, index: int, elapsed: float, budget: float) -> None:
        """Raise the stride ceiling when the remaining frames cannot all be sampled at the current cost per sample."""
        remaining_time = budget * (1.0 - self.policy.budget_reserve) - elapsed
        remaining_frames = max(0, self.total - index - 1)
        if remaining_frames == 0:
            return
        cost = elapsed / self.samples
        affordable = max(1.0, remaining_time / cost) if remaining_time > 0 else 1.0
        # Never lower than the policy's own ceiling; only stretch it when the budget demands.
        self.budget_floor = max(1, math.ceil(remaining_frames / affordable))
        if self.budget_floor > 1:
            self.reasons["budget_stretch"] = self.reasons.get("budget_stretch", 0) + 1
