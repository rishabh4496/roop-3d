"""SceneTrackerAdaptation: an optional, time-boxed self-supervised calibration of the tracker to one target actor.

SPARK (SIGGRAPH Asia 2024) personalises a face-capture network from a few frames of the actor with photometric and
landmark losses. No such network is installed here, so this pass personalises the parts of the tracker that ARE parametric,
from up to ``frames`` (16) frames sampled across the target video and within ``budget_s`` (3 s) of wall clock:

  * facial proportions: the five 3-D anchors the camera is solved against are re-estimated for this actor. Pose and shape
    are alternated (the shape step is a closed-form Gaussian MAP solve, so a round costs milliseconds and the deadline is
    honoured between steps). The result is accepted only if the held-out frames' landmark residual really drops;
  * neutral expression: the track's own resting mouth/eye/brow ratios, so the tracker needs no warm-up;
  * lighting environment: a robust median of per-frame SH fits and a measured stability verdict (``lighting_static``);
  * identity prior: the pooled shape descriptor, against which each shot's lock is compared.

Everything is self-supervised: it needs the video's own landmarks and pixels, no labels. The ``AdaptationReport`` carries the
measured before/after numbers, the time spent, and whether the budget cut the pass short. Nothing here changes a model's
weights; depth (z) of the anchors is only as observable as the yaw spread of the sampled frames, and the report says how
much of it was.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Sequence

import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.pose import rotation_from_euler
from roop_3d.reconstruction.expression_tps import layout_is_plausible
from roop_3d.reconstruction.geometry_types import WeakPerspectiveCamera
from roop_3d.reconstruction.multiview_fit import (
    SIGMA_OBS_M,
    SIGMA_PRIOR_XY_M,
    SIGMA_PRIOR_Z_M,
    estimate_pose_from_landmarks,
    solve_view_pose,
)
from roop_3d.reconstruction.reenactment import NeutralCalibrator, measure_expression
from roop_3d.tracking.identity_lock import KIND_FLAME, landmark_proportions, pose_weight, sigma_floor, weighted_median
from roop_3d.tracking.lighting import MeshSHLightingEstimator, angle_between_deg
from roop_3d.tracking.video_tracker import POSE_PRIOR_DEG, FrameEvidence, HeadAnchors, corner_lift_asymmetry

DEFAULT_FRAMES, DEFAULT_BUDGET_S = 16, 3.0
MAX_ADAPT_YAW_DEG = 60.0            # beyond this the 5 points are too foreshortened / occluded to personalise from
MAX_SAMPLE_RMS_MM = 12.0            # a sample the generic head explains this badly is a bad detection, not an actor
IMPROVEMENT_REQUIRED = 0.98         # held-out residual must fall to <= 98 % of the generic head's
# Measured on a truly static light seen at +/-20 degrees of yaw: key direction within 9 degrees, ambient within 28 % (the generic
# head's normals make the ambient term the less reliable one). A real exposure change is >= 100 %, a moved light tens of degrees.
LIGHTING_STATIC_DEG, LIGHTING_STATIC_AMBIENT = 25.0, 0.40


def sample_frame_indices(total_frames: int, count: int = DEFAULT_FRAMES) -> list[int]:
    """Frame indices at the centres of ``count`` equal segments of the video (all frames of a short video)."""
    if total_frames <= 0 or count <= 0:
        return []
    if total_frames <= count:
        return list(range(total_frames))
    return sorted({min(total_frames - 1, int((k + 0.5) * total_frames / count)) for k in range(count)})


@dataclass(frozen=True)
class AdaptationReport:
    frames_requested: int
    frames_used: int
    elapsed_s: float
    budget_s: float
    budget_exhausted: bool
    rounds: int = 0
    anchors_applied: bool = False
    train_residual_mm: tuple[float, float] | None = None      # generic head, personalised head
    holdout_residual_mm: tuple[float, float] | None = None
    anchor_shift_mm: float | None = None
    depth_observed: float | None = None                       # 0..1: how much of the z prior the data replaced
    yaw_span_deg: float | None = None
    expression_samples: int = 0
    identity_kind: str | None = None
    identity_samples: int = 0
    lighting_samples: int = 0
    lighting_spread_deg: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneAdaptation:
    anchors: HeadAnchors | None
    expression_baseline: np.ndarray | None            # ExpressionMeasure.vector() of the actor's neutral
    asymmetry_baseline: float
    identity: tuple[str, np.ndarray, np.ndarray] | None      # (kind, beta, sigma)
    lighting: np.ndarray | None                       # (9, 3) camera-frame SH, BGR
    lighting_static: bool
    report: AdaptationReport

    def to_dict(self) -> dict:
        """JSON-friendly form, e.g. to keep with a job."""
        listed = lambda a: None if a is None else np.asarray(a, float).tolist()
        return {
            "anchors": None if self.anchors is None else listed(self.anchors.points),
            "expression_baseline": listed(self.expression_baseline), "asymmetry_baseline": float(self.asymmetry_baseline),
            "identity": None if self.identity is None else [self.identity[0], listed(self.identity[1]), listed(self.identity[2])],
            "lighting": listed(self.lighting), "lighting_static": bool(self.lighting_static),
            "report": {**self.report.__dict__, "warnings": list(self.report.warnings)}}

    @classmethod
    def from_dict(cls, data: dict) -> "SceneAdaptation":
        array = lambda v: None if v is None else np.asarray(v, np.float64)
        report = dict(data["report"])
        for key in ("train_residual_mm", "holdout_residual_mm"):
            report[key] = None if report.get(key) is None else tuple(report[key])
        report["warnings"] = tuple(report.get("warnings", ()))
        identity = data.get("identity")
        return cls(anchors=None if data["anchors"] is None else HeadAnchors.canonical().with_points(array(data["anchors"])),
                   expression_baseline=array(data["expression_baseline"]), asymmetry_baseline=float(data["asymmetry_baseline"]),
                   identity=None if identity is None else (identity[0], array(identity[1]), array(identity[2])),
                   lighting=None if data["lighting"] is None else np.asarray(data["lighting"], np.float32),
                   lighting_static=bool(data["lighting_static"]), report=AdaptationReport(**report))


@dataclass
class _Sample:
    evidence: FrameEvidence
    five: np.ndarray
    fit: object = None              # ViewPoseFit against the anchors currently in use


def _camera(fit) -> WeakPerspectiveCamera:
    return fit.camera()


class SceneTrackerAdaptation:
    def __init__(self, *, frames: int = DEFAULT_FRAMES, budget_s: float = DEFAULT_BUDGET_S, rounds: int = 8,
                 min_frames: int = 6, lighting: MeshSHLightingEstimator | None = None,
                 clock: Callable[[], float] = time.perf_counter):
        if frames < 2 or budget_s <= 0 or rounds < 1 or not 2 <= min_frames <= frames:
            raise ValueError("Invalid adaptation configuration")
        self.frames, self.budget_s, self.rounds, self.min_frames = frames, float(budget_s), rounds, min_frames
        self.lighting, self.clock = lighting, clock

    def frame_indices(self, total_frames: int) -> list[int]:
        return sample_frame_indices(total_frames, self.frames)

    # ------------------------------------------------------------------ the pass
    def calibrate(self, samples: Sequence[FrameEvidence], *, started: float | None = None,
                  base: HeadAnchors | None = None, requested: int | None = None) -> SceneAdaptation:
        """Adapt to the actor seen in ``samples``. ``started`` is when the whole pass began (so frame extraction counts
        against the same budget) and ``requested`` how many frames were asked for; the pass never raises on poor input,
        it reports why it adapted nothing."""
        started = self.clock() if started is None else started
        deadline = started + self.budget_s
        base = base or HeadAnchors.canonical()
        warnings: list[str] = []
        pool = self._usable(samples, base)
        requested = len(samples) if requested is None else requested
        if len(pool) < self.min_frames:
            warnings.append(f"Only {len(pool)} of {requested} frames had a usable face; at least {self.min_frames} are needed")
            return self._result(None, None, 0.0, None, None, False, self._report(requested, len(pool), started, deadline, warnings=warnings))

        anchors, rounds, stats = self._personalise(pool, base, deadline, warnings)
        expression, asymmetry, expression_count = self._neutral(pool)
        identity, identity_count = self._identity(pool)
        lighting, static, spread, lighting_count = self._lighting(pool, deadline, warnings)
        report = self._report(requested, len(pool), started, deadline, rounds=rounds, anchors=anchors, base=base, stats=stats,
                              expression_samples=expression_count, identity=identity, identity_samples=identity_count,
                              lighting_samples=lighting_count, lighting_spread=spread, warnings=warnings,
                              yaw_span=float(max(s.fit.yaw for s in pool) - min(s.fit.yaw for s in pool)))
        return self._result(anchors, expression, asymmetry, identity, lighting, static, report)

    # ------------------------------------------------------------------ 1. facial proportions
    def _usable(self, samples: Sequence[FrameEvidence], base: HeadAnchors) -> list[_Sample]:
        pool = []
        for item in samples:
            five = None if item.landmarks5 is None else np.asarray(item.landmarks5, np.float64)
            if five is None or five.shape != (5, 2) or not np.isfinite(five).all():
                continue
            fit = self._solve(five, base)
            if fit is None or not np.isfinite([fit.yaw, fit.pitch, fit.roll, fit.scale, fit.tx, fit.ty, fit.rms_mm]).all() \
                    or abs(fit.yaw) > MAX_ADAPT_YAW_DEG or fit.rms_mm > MAX_SAMPLE_RMS_MM:
                continue
            pool.append(_Sample(item, five, fit))
        return pool

    @staticmethod
    def _solve(five: np.ndarray, anchors: HeadAnchors):
        try:
            with np.errstate(all="ignore"):         # degenerate detections (all points equal) overflow inside the solver
                return solve_view_pose(five, anchors.points, anchors.normals, estimate_pose_from_landmarks(five),
                                       anchors.weights, pose_sigma_deg=POSE_PRIOR_DEG)
        except DomainError:
            return None

    def _personalise(self, pool: list[_Sample], base: HeadAnchors, deadline: float, warnings: list[str]):
        holdout = {i for i in range(len(pool)) if i % 4 == 3} if len(pool) >= 8 else set()
        train = [i for i in range(len(pool)) if i not in holdout]
        generic = [s.fit for s in pool]
        before = self._residuals(generic, train, holdout)
        best, best_anchors, best_sigma, rounds = before, None, None, 0
        fits = generic
        for _ in range(self.rounds):
            if self.clock() >= deadline:
                warnings.append("The time budget ended the personalisation early")
                break
            points, sigma_z = self._shape_step(pool, fits, train, base)
            candidate = base.with_points(points)
            refit = [self._solve(s.five, candidate) for s in pool]
            if any(f is None for f in refit):
                warnings.append("A pose solve failed against the personalised anchors; keeping the best round so far")
                break
            rounds += 1
            score = self._residuals(refit, train, holdout)
            fits = refit
            if score[1] >= best[1] * 0.995:             # converged: another round would not change the held-out residual
                break
            best, best_anchors, best_sigma = score, candidate, float(sigma_z.mean())
        stats = {"before": before, "after": best, "holdout": bool(holdout), "sigma_z": best_sigma}
        if best_anchors is None:
            warnings.append("Personalised anchors did not improve the residual; the generic head is kept")
            return None, rounds, stats
        if best[1] > before[1] * IMPROVEMENT_REQUIRED:
            warnings.append("The held-out residual did not improve enough to accept personalised anchors; the generic head is kept")
            return None, rounds, stats
        return best_anchors, rounds, stats

    @staticmethod
    def _residuals(fits, train, holdout) -> tuple[float, float]:
        """(train, holdout-or-train) weighted RMS landmark residual in millimetres."""
        rms = lambda ids: float(np.sqrt(np.mean([fits[i].rms_mm ** 2 for i in ids]))) if ids else 0.0
        t = rms(train)
        return t, rms(sorted(holdout)) if holdout else t

    def _shape_step(self, pool: list[_Sample], fits, train: list[int], base: HeadAnchors):
        """Closed-form MAP of the five 3-D anchors given the current per-frame cameras (prior = the generic head)."""
        prior_inv = np.diag([1 / SIGMA_PRIOR_XY_M ** 2, 1 / SIGMA_PRIOR_XY_M ** 2, 1 / SIGMA_PRIOR_Z_M ** 2])
        a = np.stack([prior_inv] * 5)
        b = np.stack([prior_inv @ base.points[j] for j in range(5)])
        for i in train:
            fit = fits[i]
            rotation = rotation_from_euler(fit.yaw, fit.pitch, fit.roll)
            m = rotation[:2, :]
            offset = np.array((fit.tx, fit.ty)) - fit.scale * (rotation @ fit.center)[:2]
            quality = math.exp(-0.5 * (fit.rms_mm / 4.0) ** 2)
            for j in range(5):
                y = (pool[i].five[j] - offset) / fit.scale
                weight = quality * float(base.weights[j]) / SIGMA_OBS_M ** 2
                a[j] += weight * m.T @ m
                b[j] += weight * m.T @ y
        points = np.stack([np.linalg.solve(a[j], b[j]) for j in range(5)])
        sigma_z = np.array([math.sqrt(np.linalg.inv(a[j])[2, 2]) for j in range(5)])
        return points, sigma_z

    # ------------------------------------------------------------------ 2. neutral expression
    def _neutral(self, pool: list[_Sample]):
        calibrator = NeutralCalibrator(max(len(pool), 8), 4)
        lifts = []
        for sample in pool:
            lm106 = sample.evidence.landmarks106
            if lm106 is None or abs(sample.fit.yaw) > 60.0:
                continue
            points = np.asarray(lm106, np.float64)
            if points.shape != (106, 2) or not np.isfinite(points).all() or not layout_is_plausible(points):
                continue
            try:
                measure = measure_expression(sample.five, points, sample.fit.yaw)
                lifts.append(corner_lift_asymmetry(sample.five, points, sample.fit.yaw))
            except DomainError:
                continue
            calibrator.push(measure.vector())
        if not calibrator.calibrated:
            return None, 0.0, len(calibrator.history)
        return calibrator.baseline(), float(np.median(lifts)), len(calibrator.history)

    # ------------------------------------------------------------------ 3. identity prior
    def _identity(self, pool: list[_Sample]):
        rows: dict[str, list[tuple[np.ndarray, float]]] = {}
        for sample in pool:
            ev, fit = sample.evidence, sample.fit
            if ev.flame is not None:
                kind, vector = KIND_FLAME, np.asarray(ev.flame.beta, np.float64)
            else:
                lm106 = ev.landmarks106
                if lm106 is not None and not (np.asarray(lm106).shape == (106, 2) and layout_is_plausible(np.asarray(lm106, np.float64))):
                    lm106 = None
                described = landmark_proportions(ev.landmarks5, lm106, fit.yaw, fit.pitch)
                if described is None:
                    continue
                kind, vector = described
            rows.setdefault(kind, []).append((vector, pose_weight(fit.yaw, fit.pitch)))
        if not rows:
            return None, 0
        kind = max(rows, key=lambda k: len(rows[k]))
        usable = [(v, w) for v, w in rows[kind] if w >= 0.35]
        if len(usable) < 4:
            return None, len(usable)
        values, weights = np.stack([v for v, _ in usable]), np.array([w for _, w in usable])
        beta = weighted_median(values, weights)
        sigma = np.maximum(1.4826 * weighted_median(np.abs(values - beta), weights), sigma_floor(kind, beta))
        return (kind, beta, sigma), len(usable)

    # ------------------------------------------------------------------ 4. lighting environment
    def _lighting(self, pool: list[_Sample], deadline: float, warnings: list[str]):
        if self.lighting is None:
            return None, False, None, 0
        estimates = []
        for sample in pool:
            if sample.evidence.image_bgr is None:
                continue
            if self.clock() >= deadline:
                warnings.append("The time budget ended the lighting estimate early")
                break
            result = self.lighting.estimate(sample.evidence.image_bgr, _camera(sample.fit))
            if result is not None:
                estimates.append(result)
        if len(estimates) < 3:
            return None, False, None, len(estimates)
        coefficients = np.stack([e.coefficients for e in estimates])               # (N, 9, 3)
        median = np.median(coefficients, axis=0).astype(np.float32)
        keys = np.stack([e.key_direction for e in estimates])
        key = np.median(keys, axis=0)
        spread = max(angle_between_deg(k, key) for k in keys)
        ambient = coefficients[:, 0, :].mean(axis=1)
        ambient_spread = float(np.max(np.abs(ambient - np.median(ambient))) / max(float(np.median(ambient)), 1e-6))
        static = spread <= LIGHTING_STATIC_DEG and ambient_spread <= LIGHTING_STATIC_AMBIENT
        return median, bool(static), float(spread), len(estimates)

    # ------------------------------------------------------------------ results
    def _report(self, requested, used, started, deadline, *, rounds=0, anchors=None, base=None, stats=None,
                expression_samples=0, identity=None, identity_samples=0, lighting_samples=0, lighting_spread=None,
                yaw_span=None, warnings=()) -> AdaptationReport:
        now = self.clock()
        applied = anchors is not None
        return AdaptationReport(
            frames_requested=requested, frames_used=used, elapsed_s=float(now - started), budget_s=self.budget_s,
            budget_exhausted=now >= deadline, rounds=rounds, anchors_applied=applied,
            train_residual_mm=None if not stats else tuple(round(x, 4) for x in (stats["before"][0], stats["after"][0])),
            holdout_residual_mm=None if not stats or not stats["holdout"] else tuple(round(x, 4) for x in (stats["before"][1], stats["after"][1])),
            anchor_shift_mm=float(np.mean(np.linalg.norm(anchors.points - base.points, axis=1)) * 1000) if applied else None,
            depth_observed=float(np.clip(1.0 - stats["sigma_z"] / SIGMA_PRIOR_Z_M, 0.0, 1.0)) if applied and stats["sigma_z"] is not None else None,
            yaw_span_deg=yaw_span,
            expression_samples=expression_samples, identity_kind=None if identity is None else identity[0],
            identity_samples=identity_samples, lighting_samples=lighting_samples, lighting_spread_deg=lighting_spread,
            warnings=tuple(warnings))

    @staticmethod
    def _result(anchors, expression, asymmetry, identity, lighting, static, report) -> SceneAdaptation:
        return SceneAdaptation(anchors, expression, float(asymmetry or 0.0), identity, lighting, bool(static), report)
