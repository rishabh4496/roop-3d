"""VideoHeadTracker3D: monocular video head and gaze tracker (VHAP / GAGAvatar_track / flame-head-tracker lineage).

What one frame yields, and where each quantity honestly comes from (nothing is filled in when it was not measured):

  camera pose (6 DoF)  yaw/pitch/roll + scale + 2-D translation of a weak-perspective camera, solved from the 5-point
                       landmarks against the canonical head (or the actor-personalised anchors from SceneTrackerAdaptation).
                       Depth is the scale, not metric. A supplied FLAME rotation is deliberately NOT used: no FLAME weights
                       are installed, so its axis convention cannot be verified against this project's frame.
  shape (beta_target)  locked per shot by robust weighted-median pooling (``identity_lock``): the supplied FLAME beta when an
                       encoder provides one, otherwise facial proportions measured from the landmarks (reported as such).
  jaw (theta_jaw)      the supplied FLAME jaw opening, otherwise a landmark proxy: lip drop divided by the jaw lever arm.
  expression (psi)     eight asymmetric channels (mouth open/width, left and right smile, blink, brow) measured against the
                       track's own neutral; the FLAME psi vector is passed through when an encoder supplies one.
  gaze (theta_eyes)    per-eye yaw/pitch from a calibrated estimator when given, else the uncalibrated iris-geometry estimate
                       of the 478-point mesh, else None. Eyeball torsion is not observable and is not reported.
  lighting (L_t)       second-order SH on the generic head under the tracked camera, in the camera frame (``lighting.py``).

Temporal behaviour: every continuous channel runs through the no-overshoot One-Euro + Kalman stage of ``filters`` (the head
rotation through an SO(3) One-Euro); frames that cannot be measured are bridged by the Kalman prediction for a bounded time.
A shot boundary (explicit flag, image change, or a physically impossible jump of the face) resets all filters, the identity
lock and the expression baseline; a sustained identity change inside a shot is reported as an ``identity_break`` cut.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Callable, Iterable, Iterator, Literal

import cv2
import numpy as np
from pydantic import Field

from roop_3d.core.errors import DomainError
from roop_3d.core.pose import euler_from_rotation, rotation_from_euler
from roop_3d.core.settings import Contract
from roop_3d.flame_3d_engine import EyeGaze, FlameParameters
from roop_3d.reconstruction.expression_tps import group, layout_is_plausible
from roop_3d.reconstruction.geometry_types import WeakPerspectiveCamera
from roop_3d.reconstruction.multiview_fit import (
    estimate_pose_from_landmarks,
    identity_template,
    landmark_normals,
    solve_view_pose,
    template_landmarks,
)
from roop_3d.reconstruction.reenactment import NeutralCalibrator, measure_expression, params_from_measure
from roop_3d.tracking.filters import TrackFilterBank
from roop_3d.tracking.gaze import SOURCE_ESTIMATOR, SOURCE_IRIS, iris_gaze_from_mesh478, gaze_vectors, vergence
from roop_3d.tracking.identity_lock import IdentitySnapshot, KIND_FLAME, ShotIdentityLock, landmark_proportions, pose_weight
from roop_3d.tracking.lighting import LightingEstimate, MeshSHLightingEstimator

log = logging.getLogger(__name__)

EXPRESSION_CHANNELS = ("mouth_open", "mouth_width", "smile_a", "smile_b", "blink_a", "blink_b", "brow_a", "brow_b")
EXPRESSION_FULL_SCALE = np.array((0.55, 0.30, 0.22, 0.22, 1.0, 1.0, 0.16, 0.16))     # the clamp limits of ExpressionParams
PSI_FULL_SCALE = 2.0                  # a FLAME psi coefficient of 2 sigma counts as "fully expressive"
POSE_PRIOR_DEG = 45.0                 # solve_view_pose's default (8) pulls the pose toward the coarse 5-point estimate: measured, on a
                                      # head that matches the model exactly, 0.43 vs 0.12 degrees of pitch error. 45 lets the data decide.
JAW_LEVER_M = 0.085                   # hinge to lower lip (the reenactment rig's measured lever arm)
JAW_MAX_RAD = 0.6
FLAME_PSI_DIMS = 50
Status = Literal["tracked", "coasting", "lost"]
CutReason = Literal["explicit", "image_discontinuity", "landmark_discontinuity", "identity_break"]


# =========================================================================== settings
class TrackerSettings(Contract):
    # shot boundaries
    cut_image_difference: float = Field(default=0.45, gt=0, le=1, description="Mean 32x32 thumbnail difference that is a cut by itself")
    cut_image_difference_with_histogram: float = Field(default=0.18, gt=0, le=1)
    cut_histogram_distance: float = Field(default=0.7, gt=0, le=1)
    cut_jump_iod: float = Field(default=2.5, gt=0, description="Face-centre jump, in inter-ocular distances, that no real motion makes in one frame")
    cut_rotation_deg_s: float = Field(default=2700.0, gt=0, description="Head rotation speed that is a cut rather than a measurement error")
    cut_max_dt_s: float = Field(default=0.2, gt=0, description="Continuity cuts are only judged across gaps this short")
    # measurement gates
    max_pose_rate_deg_s: float = Field(default=1400.0, gt=0, description="Physical limit; faster is a measurement error")
    max_abs_yaw_deg: float = Field(default=85.0, gt=10, le=90)
    expression_max_yaw_deg: float = Field(default=60.0, gt=10, le=85)
    max_pose_rms_iod: float = Field(default=0.12, gt=0)
    coast_s: float = Field(default=0.25, ge=0, le=0.5, description="Longest bridged run without a measurement")
    reacquire_s: float = Field(default=1.0, gt=0, description="After this long without a face the filters restart (the shot is kept)")
    # expression
    calibration_frames: int = Field(default=8, ge=2, le=120)
    calibration_window: int = Field(default=120, ge=8, le=1000)
    # identity
    lock_min_frames: int = Field(default=10, ge=2, le=200)
    lock_frames: int = Field(default=30, ge=2, le=400)
    identity_break_run: int = Field(default=12, ge=3, le=200)
    # optional channels
    gaze_enabled: bool = True
    iris_gaze_fallback: bool = True
    lighting_enabled: bool = True
    lighting_stride: int = Field(default=3, ge=1, le=60)
    ipd_m: float = Field(default=0.063, gt=0.03, lt=0.1, description="Population interpupillary distance; scales the vergence distance")


# =========================================================================== data
@dataclass(frozen=True)
class HeadAnchors:
    """The 3-D geometry the 5-point landmarks are solved against (image-left eye, image-right eye, nose tip, mouth corners)."""
    points: np.ndarray                  # (5, 3) metres, template frame
    normals: np.ndarray                 # (5, 3)
    weights: np.ndarray                 # (5,) down-weights the mouth corners, which move with expression
    iod_m: float
    personalised: bool = False

    @classmethod
    def canonical(cls) -> "HeadAnchors":
        mesh = identity_template(2)
        points = template_landmarks("5pt", mesh)
        return cls(points, landmark_normals("5pt", mesh), np.array((1.0, 1.0, 1.5, 0.35, 0.35)),
                   float(np.linalg.norm(points[0] - points[1])))

    def with_points(self, points: np.ndarray) -> "HeadAnchors":
        points = np.asarray(points, np.float64)
        return HeadAnchors(points, self.normals, self.weights, float(np.linalg.norm(points[0] - points[1])), True)


@dataclass(frozen=True)
class FrameEvidence:
    """Everything the extraction stage measured for one frame of the tracked face (canvas pixel coordinates)."""
    timestamp: float
    landmarks5: np.ndarray | None
    landmarks106: np.ndarray | None = None
    flame: FlameParameters | None = None          # from an encoder when one is installed
    mesh478: np.ndarray | None = None             # normalised crop coordinates
    mesh_aspect: float = 1.0                      # crop width / height of mesh478
    gaze: EyeGaze | None = None                   # from a calibrated estimator
    image_bgr: np.ndarray | None = None
    scene_cut: bool = False
    frame_index: int | None = None


@dataclass(frozen=True)
class CameraPose:
    """Weak-perspective camera of the head: ``pixel = scale * (R @ X)[:2] + t`` for the model centre at (tx, ty)."""
    yaw_deg: float                      # standard frame: pitch positive = looking up
    pitch_deg: float
    roll_deg: float
    scale: float                        # pixels per metre (depth is encoded here, it is not metric)
    tx: float
    ty: float
    rotation: np.ndarray                # (3, 3)


@dataclass(frozen=True)
class GazeState:
    angles: np.ndarray                  # (4,) left yaw, left pitch, right yaw, right pitch (radians, head frame)
    head_vectors: np.ndarray            # (2, 3) left, right
    camera_vectors: np.ndarray          # (2, 3) rotated by the head rotation
    vergence_deg: float
    convergence_distance_m: float | None
    source: str


@dataclass(frozen=True)
class TrackedHeadFrame:
    frame_index: int
    timestamp: float
    shot_index: int
    shot_cut: bool
    cut_reason: str | None
    status: Status
    reason: str
    confidence: float
    components: dict[str, float | None]
    camera: CameraPose | None = None
    pose_residual_iod: float | None = None
    identity: IdentitySnapshot | None = None
    expression: dict[str, float] | None = None
    expression_held: bool = False
    expression_energy: float | None = None
    expression_calibrated: bool = False
    psi: np.ndarray | None = None
    jaw_open_rad: float | None = None
    jaw_source: str | None = None
    gaze: GazeState | None = None
    lighting: LightingEstimate | None = None
    lighting_source: str | None = None
    lighting_coefficients: np.ndarray | None = None        # filtered (9, 3), what L_t is


# =========================================================================== shot boundaries
class ShotBoundaryDetector:
    """Image-based cut detection: the same 32x32 difference + colour-histogram rule the face tracker uses."""

    def __init__(self, settings: TrackerSettings):
        self.s = settings
        self._thumbnail: np.ndarray | None = None

    def reset(self) -> None:
        self._thumbnail = None

    def observe(self, image_bgr: np.ndarray) -> bool:
        thumb = cv2.resize(image_bgr, (32, 32), interpolation=cv2.INTER_AREA)
        previous, self._thumbnail = self._thumbnail, thumb
        if previous is None:
            return False
        difference = float(np.abs(previous.astype(np.float32) - thumb.astype(np.float32)).mean() / 255)
        if difference <= self.s.cut_image_difference_with_histogram:
            return False
        if difference > self.s.cut_image_difference:
            return True
        histograms = [cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 2], None, [16, 16], [0, 180, 0, 256])
                      for img in (previous, thumb)]
        return bool(cv2.compareHist(*histograms, cv2.HISTCMP_BHATTACHARYYA) > self.s.cut_histogram_distance)


# =========================================================================== measurement helpers
def corner_lift_asymmetry(lm5, lm106, yaw_deg: float) -> float:
    """Raw (image-left minus image-right) mouth-corner lift in foreshortening-corrected inter-ocular units."""
    five = np.asarray(lm5, np.float64)
    across = five[1] - five[0]
    iod = float(np.linalg.norm(across))
    if iod < 2.0:
        raise DomainError("EXPRESSION_UNMEASURABLE", "Inter-ocular distance too small to measure expression", 422)
    across = across / iod
    down = np.array((-across[1], across[0]))
    mouth = (group(lm106, "mouth") - five[:2].mean(axis=0))
    u, v = mouth @ across, mouth @ down
    centre = float(np.median(v))
    lift = lambda corner_v: (centre - corner_v) / (iod / max(math.cos(math.radians(yaw_deg)), 0.5))
    return float(lift(v[int(np.argmin(u))]) - lift(v[int(np.argmax(u))]))


def expression_energy(values: np.ndarray, full_scale: np.ndarray) -> float:
    """RMS of the channels relative to their full-scale limits, 0..1."""
    return float(np.clip(np.sqrt(np.mean((np.abs(values) / full_scale) ** 2)), 0.0, 1.0))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))))


# =========================================================================== tracker
@dataclass
class _Raw:
    time: float
    rotation: np.ndarray
    centre: np.ndarray
    iod_px: float


class VideoHeadTracker3D:
    """Stateful tracker of ONE face through a video. Feed frames in order; ``process`` never raises on bad measurements."""

    def __init__(self, settings: TrackerSettings | None = None, *, anchors: HeadAnchors | None = None,
                 adaptation=None, lighting: MeshSHLightingEstimator | None = None,
                 on_frame: Callable[[TrackedHeadFrame], None] | None = None):
        self.s = settings or TrackerSettings()
        self.adaptation = adaptation
        self.anchors = anchors or (adaptation.anchors if adaptation is not None and adaptation.anchors is not None else HeadAnchors.canonical())
        self.lighting = lighting
        self.on_frame = on_frame
        self.filters = TrackFilterBank()
        self.lock = ShotIdentityLock(min_frames=self.s.lock_min_frames, lock_frames=self.s.lock_frames,
                                     break_run=self.s.identity_break_run)
        self.shots = ShotBoundaryDetector(self.s)
        if adaptation is not None and adaptation.identity is not None:
            self.lock.set_prior(*adaptation.identity)
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Forget everything: a new video."""
        self.shot_index = 0
        self.shot_cuts = 0
        self.frames = 0
        self.status_counts: dict[str, int] = {"tracked": 0, "coasting": 0, "lost": 0}
        self.last_time: float | None = None
        self.shots.reset()
        self._pending_cut: str | None = None
        self._begin_shot_state()

    def _begin_shot_state(self) -> None:
        self.filters.reset()
        self.lock.reset()
        self.calibrator = NeutralCalibrator(self.s.calibration_window, self.s.calibration_frames)
        self.asymmetry_history: list[float] = []
        if self.adaptation is not None and self.adaptation.expression_baseline is not None:
            for _ in range(self.s.calibration_frames):          # a known actor needs no warm-up
                self.calibrator.push(self.adaptation.expression_baseline)
            self.asymmetry_history = [self.adaptation.asymmetry_baseline] * self.s.calibration_frames
        self._last: _Raw | None = None
        self._iod_ref: float | None = None
        self._lost_since: float | None = None
        self._last_expression: np.ndarray | None = None
        self._lighting_countdown = 0
        self._last_lighting: LightingEstimate | None = None
        self._last_camera: CameraPose | None = None
        self._last_jaw: tuple[float, str] | None = None
        self._gaze_source: str | None = None
        self._model_centre: np.ndarray = self.anchors.points.mean(axis=0)

    def _start_shot(self, reason: str) -> None:
        self.shot_index += 1
        self.shot_cuts += 1
        self._begin_shot_state()

    def track(self, evidence: Iterable[FrameEvidence]) -> Iterator[TrackedHeadFrame]:
        for item in evidence:
            yield self.process(item)

    def summary(self) -> dict:
        return {"frames": self.frames, "shots": self.shot_index + 1, "shot_cuts": self.shot_cuts,
                "status": dict(self.status_counts), "identity": self.lock.snapshot().state}

    # ------------------------------------------------------------------ one frame
    def process(self, ev: FrameEvidence) -> TrackedHeadFrame:
        index = ev.frame_index if ev.frame_index is not None else self.frames
        self.frames += 1
        t = float(ev.timestamp) if ev.timestamp is not None else float("nan")
        if not np.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            return self._emit(self._frame(index, t, "lost", "TIMESTAMP_INVALID", {"timestamp": 0.0}))
        cut = "explicit" if ev.scene_cut else self._pending_cut
        image_cut = self.shots.observe(ev.image_bgr) if ev.image_bgr is not None else False
        cut = cut or ("image_discontinuity" if image_cut else None)
        self._pending_cut = None
        if cut and self.last_time is not None:
            self._start_shot(cut)
        else:
            cut = None
        self.last_time = t
        measured = self._measure_pose(ev, t)
        if isinstance(measured, str):                    # a rejection reason: bridge the frame from the filters
            return self._emit(self._coast(index, t, measured, cut))
        fit, raw_rotation, comps = measured
        if self._last is not None:                       # continuity: a real cut that no flag or image change announced
            dt = t - self._last.time
            speed = _angle_deg(self._last.rotation, raw_rotation) / dt
            centre_jump = float(np.linalg.norm(np.array((fit.tx, fit.ty)) - self._last.centre) / self._last.iod_px)
            if dt <= self.s.cut_max_dt_s and (speed > self.s.cut_rotation_deg_s or centre_jump > self.s.cut_jump_iod):
                self._start_shot("landmark_discontinuity")
                cut = "landmark_discontinuity"
            elif speed > self.s.max_pose_rate_deg_s:     # an impossible speed that is not a cut: a measurement error
                comps["temporal"] = 0.0
                return self._emit(self._coast(index, t, "POSE_JUMP_EXCEEDS_PHYSICAL_RATE", cut, comps))
            else:
                comps["temporal"] = 1.0
        iod_px = fit.scale * self.anchors.iod_m
        self._last = _Raw(t, raw_rotation, np.array((fit.tx, fit.ty)), iod_px)
        self._model_centre = np.asarray(fit.center, np.float64)     # the point (tx, ty) is the image position of
        self._lost_since = None
        camera = self._filter_camera(fit, raw_rotation, iod_px, t)
        self._last_camera = camera

        expression, held, calibrated = self._expression(ev, fit, t)
        jaw, jaw_source = self._jaw(ev, expression, t)
        psi = self._psi(ev, t)
        energy = self._energy(expression, psi)
        gaze = self._gaze(ev, camera, t) if self.s.gaze_enabled else None
        identity = self._identity(ev, fit, camera, energy, comps)
        if identity is not None and identity.break_detected:
            self._pending_cut = "identity_break"
        lighting, lighting_source, lighting_coefficients = self._lighting(ev, camera, t)

        measured_components = [v for v in comps.values() if v is not None]
        confidence = float(min(measured_components)) if measured_components else 0.0
        return self._emit(self._frame(
            index, t, "tracked", "OK", comps, confidence=confidence, camera=camera, residual=fit.rms_iod, identity=identity,
            expression=expression, held=held, energy=energy, calibrated=calibrated, psi=psi, jaw=jaw, jaw_source=jaw_source,
            gaze=gaze, lighting=lighting, lighting_source=lighting_source, lighting_coefficients=lighting_coefficients,
            cut=cut))

    # ------------------------------------------------------------------ pose
    def _valid5(self, ev: FrameEvidence) -> bool:
        points = ev.landmarks5
        if points is None:
            return False
        points = np.asarray(points, np.float64)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            return False
        canvas = ev.image_bgr.shape[:2] if ev.image_bgr is not None else None
        if canvas is not None:
            h, w = canvas
            return bool(np.all((points[:, 0] > -0.25 * w) & (points[:, 0] < 1.25 * w) & (points[:, 1] > -0.25 * h) & (points[:, 1] < 1.25 * h)))
        return True

    def _measure_pose(self, ev: FrameEvidence, t: float):
        comps: dict[str, float | None] = {"landmarks": None, "pose_fit": None, "temporal": None, "identity": None}
        if not self._valid5(ev):
            comps["landmarks"] = 0.0
            return "LANDMARKS_MISSING_OR_INVALID"
        comps["landmarks"] = 1.0
        five = np.asarray(ev.landmarks5, np.float64)
        if abs(estimate_pose_from_landmarks(five)[0]) > self.s.max_abs_yaw_deg + 15.0:
            return "POSE_OUT_OF_SUPPORTED_RANGE"
        try:
            with np.errstate(all="ignore"):         # degenerate detections (all points equal) overflow inside the solver
                fit = solve_view_pose(five, self.anchors.points, self.anchors.normals, estimate_pose_from_landmarks(five),
                                      self.anchors.weights, pose_sigma_deg=POSE_PRIOR_DEG)
        except DomainError:
            return "POSE_SOLVE_FAILED"
        if not np.isfinite([fit.yaw, fit.pitch, fit.roll, fit.scale, fit.tx, fit.ty, fit.rms_iod]).all() or fit.scale <= 0:
            return "POSE_SOLVE_FAILED"
        if abs(fit.yaw) > self.s.max_abs_yaw_deg:
            return "POSE_OUT_OF_SUPPORTED_RANGE"
        comps["pose_fit"] = float(math.exp(-0.5 * (fit.rms_iod / self.s.max_pose_rms_iod) ** 2))
        return fit, rotation_from_euler(fit.yaw, fit.pitch, fit.roll), comps

    def _filter_camera(self, fit, raw_rotation: np.ndarray, iod_px: float, t: float) -> CameraPose:
        if self._iod_ref is None:
            self._iod_ref = iod_px                    # translation is filtered in units of the shot's first face size
        rotation = self.filters.rotation(raw_rotation, t)
        translation = self.filters.update("translation", np.array((fit.tx, fit.ty)) / self._iod_ref, t) * self._iod_ref
        scale = float(math.exp(self.filters.update("log_scale", [math.log(fit.scale)], t)[0]))
        yaw, pitch, roll = euler_from_rotation(rotation)
        return CameraPose(float(yaw), float(pitch), float(roll), scale, float(translation[0]), float(translation[1]), rotation)

    # ------------------------------------------------------------------ expression, jaw, psi
    def _expression(self, ev: FrameEvidence, fit, t: float):
        """-> (named channels | None, held, calibrated)."""
        measured = None
        if ev.landmarks106 is not None and abs(fit.yaw) <= self.s.expression_max_yaw_deg:
            points = np.asarray(ev.landmarks106, np.float64)
            if points.shape == (106, 2) and np.isfinite(points).all() and layout_is_plausible(points):
                try:
                    measured = (measure_expression(ev.landmarks5, points, fit.yaw),
                                corner_lift_asymmetry(ev.landmarks5, points, fit.yaw))
                except DomainError:
                    measured = None
        if measured is not None:
            self.calibrator.push(measured[0].vector())
            self.asymmetry_history = (self.asymmetry_history + [measured[1]])[-self.s.calibration_window:]
        calibrated = self.calibrator.calibrated
        if measured is not None and calibrated:
            params = params_from_measure(measured[0], self.calibrator.baseline()).clamped()
            asym = measured[1] - float(np.median(self.asymmetry_history))
            smile = params.smile
            clip_smile = lambda x: float(np.clip(x, -0.12, 0.22))
            vector = np.array((params.mouth_open, params.mouth_width, clip_smile(smile + 0.5 * asym), clip_smile(smile - 0.5 * asym),
                               params.blink_a, params.blink_b, params.brow_a, params.brow_b))
            smoothed = self.filters.update("expression", vector, t)
            self._last_expression = smoothed
            return dict(zip(EXPRESSION_CHANNELS, map(float, smoothed))), False, True
        held = self.filters.coast("expression", t)       # too turned / not calibrated: bridge or hold, never invent
        if held is not None:
            return dict(zip(EXPRESSION_CHANNELS, map(float, held))), True, calibrated
        return None, False, calibrated

    def _jaw(self, ev: FrameEvidence, expression: dict | None, t: float):
        if ev.flame is not None:
            value, source = float(np.clip(ev.flame.jaw_rotation[0], 0.0, JAW_MAX_RAD)), "flame_encoder"
        elif expression is not None:
            value = float(np.clip(expression["mouth_open"] * self.s.ipd_m / JAW_LEVER_M, 0.0, JAW_MAX_RAD))
            source = "landmark_proxy"
        else:
            held = self.filters.coast("jaw", t)
            return (float(held[0]), self._last_jaw[1]) if held is not None and self._last_jaw else (None, None)
        smoothed = float(self.filters.update("jaw", [value], t)[0])
        self._last_jaw = (smoothed, source)
        return smoothed, source

    def _psi(self, ev: FrameEvidence, t: float) -> np.ndarray | None:
        if ev.flame is None:
            return None
        return self.filters.update("psi", np.asarray(ev.flame.psi, np.float64), t).astype(np.float32)

    @staticmethod
    def _energy(expression: dict | None, psi: np.ndarray | None) -> float | None:
        if expression is not None:
            return expression_energy(np.array([expression[k] for k in EXPRESSION_CHANNELS]), EXPRESSION_FULL_SCALE)
        if psi is not None:
            return expression_energy(psi, np.full(len(psi), PSI_FULL_SCALE))
        return None

    # ------------------------------------------------------------------ gaze
    def _gaze(self, ev: FrameEvidence, camera: CameraPose, t: float) -> GazeState | None:
        measurement, source = None, None
        if ev.gaze is not None:
            measurement, source = ev.gaze, SOURCE_ESTIMATOR
        elif ev.flame is not None and ev.flame.gaze is not None:
            measurement, source = ev.flame.gaze, SOURCE_ESTIMATOR
        elif self.s.iris_gaze_fallback and ev.mesh478 is not None:
            measurement, source = iris_gaze_from_mesh478(ev.mesh478, aspect=ev.mesh_aspect), SOURCE_IRIS
        if measurement is not None:
            angles = self.filters.update("gaze", measurement.angles().astype(np.float64), t)
        else:
            angles = self.filters.coast("gaze", t)
            source = self._gaze_source
            if angles is None:
                return None
        self._gaze_source = source
        head = gaze_vectors(angles)
        converge = vergence(angles, self.s.ipd_m)
        return GazeState(angles.astype(np.float32), head.astype(np.float32), (head @ camera.rotation.T).astype(np.float32),
                         float(converge.angle_deg), converge.distance_m, source)

    # ------------------------------------------------------------------ identity
    def _identity(self, ev: FrameEvidence, fit, camera: CameraPose, energy: float | None, comps: dict) -> IdentitySnapshot | None:
        if ev.flame is not None:
            kind, vector = KIND_FLAME, np.asarray(ev.flame.beta, np.float64)
        else:
            lm106 = None
            if ev.landmarks106 is not None and np.asarray(ev.landmarks106).shape == (106, 2) \
                    and layout_is_plausible(np.asarray(ev.landmarks106, np.float64)):
                lm106 = ev.landmarks106
            described = landmark_proportions(ev.landmarks5, lm106, camera.yaw_deg, camera.pitch_deg)
            if described is None:
                return self.lock.snapshot()
            kind, vector = described
        neutral = 1.0 - min(1.0, 1.5 * (energy or 0.0))
        weight = pose_weight(camera.yaw_deg, camera.pitch_deg) * neutral * float(comps["pose_fit"] or 0.0)
        snapshot = self.lock.observe(kind, vector, weight)
        if snapshot.state != "collecting":
            comps["identity"] = 1.0 if snapshot.prior_distance is None or snapshot.prior_distance < 3.0 else 0.5
        return snapshot

    # ------------------------------------------------------------------ lighting
    def _lighting(self, ev: FrameEvidence, camera: CameraPose, t: float):
        """-> (this frame's raw estimate | None, source, filtered (9, 3) | None)."""
        if not self.s.lighting_enabled:
            return None, None, None
        estimate = None
        if self.lighting is not None and ev.image_bgr is not None and self._lighting_countdown <= 0:
            self._lighting_countdown = self.s.lighting_stride
            shift = camera.scale * (camera.rotation @ self._model_centre)[:2]
            estimate = self.lighting.estimate(ev.image_bgr, WeakPerspectiveCamera(
                rotation=tuple(tuple(float(x) for x in row) for row in camera.rotation), scale=camera.scale,
                tx=camera.tx - float(shift[0]), ty=camera.ty - float(shift[1])))
        self._lighting_countdown -= 1
        if estimate is not None:
            filtered = self.filters.update("lighting", estimate.coefficients.reshape(-1).astype(np.float64), t)
            self._last_lighting = estimate
            return estimate, "frame_fit", filtered.reshape(9, 3).astype(np.float32)
        held = self.filters.coast("lighting", t)
        if held is not None:
            return None, "held", held.reshape(9, 3).astype(np.float32)
        prior = getattr(self.adaptation, "lighting", None) if self.adaptation is not None else None
        if prior is not None and self.adaptation.lighting_static:
            return None, "adaptation_prior", np.asarray(prior, np.float32)
        return None, None, None

    # ------------------------------------------------------------------ output
    def _coast(self, index: int, t: float, reason: str, cut, comps: dict | None = None) -> TrackedHeadFrame:
        """No usable measurement: bridge from the filters for a bounded time, then report the track as lost."""
        comps = comps or {"landmarks": 0.0 if "LANDMARKS" in reason else None}
        if self._lost_since is None:
            self._lost_since = t
        rotation = self.filters.coast_rotation(t, self.s.coast_s)
        translation, log_scale = self.filters.coast("translation", t), self.filters.coast("log_scale", t)
        if rotation is None or translation is None or log_scale is None or self._last_camera is None:
            if t - self._lost_since > self.s.reacquire_s:
                self.filters.reset()                      # the shot, its identity lock and baseline are kept
                self._last = None
            return self._frame(index, t, "lost", reason, comps, cut=cut)
        yaw, pitch, roll = euler_from_rotation(rotation)
        scale = float(math.exp(log_scale[0]))
        translation = translation * self._iod_ref
        camera = CameraPose(float(yaw), float(pitch), float(roll), scale, float(translation[0]), float(translation[1]), rotation)
        expression = None
        held = self.filters.coast("expression", t)
        if held is not None:
            expression = dict(zip(EXPRESSION_CHANNELS, map(float, held)))
        gaze_angles = self.filters.coast("gaze", t)
        gaze = None
        if gaze_angles is not None and self.s.gaze_enabled:
            head = gaze_vectors(gaze_angles)
            converge = vergence(gaze_angles, self.s.ipd_m)
            gaze = GazeState(gaze_angles.astype(np.float32), head.astype(np.float32), (head @ rotation.T).astype(np.float32),
                             float(converge.angle_deg), converge.distance_m, self._gaze_source or "held")
        return self._frame(index, t, "coasting", reason, comps, camera=camera, expression=expression, held=True,
                           energy=self._energy(expression, None), gaze=gaze, identity=self.lock.snapshot(), cut=cut,
                           calibrated=self.calibrator.calibrated)

    def _frame(self, index: int, t: float, status: Status, reason: str, comps: dict, *, confidence: float = 0.0,
               camera: CameraPose | None = None, residual: float | None = None, identity=None, expression=None,
               held: bool = False, energy: float | None = None, calibrated: bool = False, psi=None, jaw=None,
               jaw_source=None, gaze=None, lighting=None, lighting_source=None, lighting_coefficients=None,
               cut: str | None = None) -> TrackedHeadFrame:
        self.status_counts[status] += 1
        if status == "coasting":
            confidence = 0.0
        return TrackedHeadFrame(
            frame_index=index, timestamp=t, shot_index=self.shot_index, shot_cut=cut is not None, cut_reason=cut,
            status=status, reason=reason, confidence=confidence, components=comps, camera=camera, pose_residual_iod=residual,
            identity=identity, expression=expression, expression_held=held, expression_energy=energy,
            expression_calibrated=calibrated, psi=psi, jaw_open_rad=jaw, jaw_source=jaw_source, gaze=gaze, lighting=lighting,
            lighting_source=lighting_source, lighting_coefficients=lighting_coefficients)

    def _emit(self, frame: TrackedHeadFrame) -> TrackedHeadFrame:
        if self.on_frame is not None:
            try:
                self.on_frame(frame)
            except Exception:       # telemetry must never take tracking down
                log.exception("Head-tracker frame observer failed")
        return frame
