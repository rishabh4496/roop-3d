"""Temporal 3D face reenactment (Stage 4): source identity geometry + target motion.

Division of labour (the architectural principle of the stage):

    SOURCE  -> shape, identity, texture: the Stage 3 ``IdentityAsset`` (mesh, atlas, per-texel observation states).
    TARGET  -> head pose, expression, jaw/eye/brow motion, timing: measured per frame from the tracked landmarks.

Per frame, per track, ``ReenactmentEngine.process`` (pure numpy; no model is loaded here)

  1. validates the observation (finite, inside the canvas, 106-point layout plausible),
  2. solves the weak-perspective camera of the target head AGAINST THE SOURCE'S 3D landmarks,
  3. measures expression as inter-ocular-normalised deltas relative to the track's own calibrated neutral,
  4. smooths pose and expression with One-Euro filters (separate bandwidths; scene cuts reset),
  5. deforms the identity mesh (jaw hinge, lips, smile, cheeks, eyelids, brows) - source anatomy, target motion,
  6. rasterises it with the atlas texture (z-buffer; unobserved texels carry zero coverage),
  7. snaps eye / mouth anchors onto the target's measured positions (bounded) and measures the residual,
  8. computes a MEASURED confidence (minimum of its components, never an average) and applies hysteresis.

When the geometry is not reliable the engine returns ``engaged=False`` with the reason and the caller keeps the
existing proven path: a 3D model is never forced onto a frame that cannot support it.

What is NOT claimed: this is a parametric, hand-built deformation model (not FLAME, not learned). The source atlas
has no mouth interior or eyelid skin, so those regions carry zero coverage when the mouth is open / the eye is
closed and the target's own pixels are preserved by the compositor.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import Field

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.reconstruction.expression_tps import BLOCKS, group, layout_is_plausible
from roop_3d.reconstruction.geometry_types import ConfidenceState
from roop_3d.reconstruction.multiview_fit import (
    ViewPoseFit,
    estimate_pose_from_landmarks,
    identity_template,
    landmark_normals,
    solve_view_pose,
)
from roop_3d.reconstruction.pose_blend import rasterize
from roop_3d.temporal import TrackTemporalStabilizer

RenderMode = Literal["standard", "pose_3d", "reenactment_3d"]
RENDER_MODES: tuple[str, ...] = ("standard", "pose_3d", "reenactment_3d")
FAR_CAMERA_M = 50.0     # weak-perspective camera expressed as a far pinhole so `pose_blend.rasterize` can be reused

# 68-landmark vertex groups of the template (indices 0..67 are preserved by every subdivision level)
EYE_A, EYE_B = range(36, 42), range(42, 48)       # image-left / image-right eye of a frontal head
BROW_A, BROW_B = range(17, 22), range(22, 27)
NOSE_TIP, MOUTH_A, MOUTH_B = 30, 48, 54
INNER_LIP = range(60, 68)


# =========================================================================== settings / data
class ReenactmentSettings(Contract):
    # gate
    enter_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    exit_confidence: float = Field(default=0.40, ge=0.0, le=1.0)
    enter_frames: int = Field(default=3, ge=1, le=30)
    exit_frames: int = Field(default=2, ge=1, le=30)
    ramp_frames: int = Field(default=4, ge=1, le=30, description="Frames over which the engine cross-fades in / out")
    max_abs_yaw_deg: float = Field(default=80.0, ge=10.0, le=90.0)
    expression_max_yaw_deg: float = Field(default=60.0, ge=10.0, le=85.0)
    max_pose_rate_deg_s: float = Field(default=1400.0, gt=0.0, description="Physical limit; faster is a measurement error")
    min_pose_support: float = Field(default=0.35, ge=0.0, le=1.0, description="Observed-texture share of the rendered footprint")
    coverage_margin_deg: float = Field(default=10.0, ge=0.0, le=45.0, description="Extrapolation allowed beyond the source views' pose range at full confidence")
    coverage_ramp_deg: float = Field(default=10.0, gt=0.0, le=45.0, description="...then confidence falls to 0 over this many further degrees")
    max_pose_rms_iod: float = Field(default=0.12, gt=0.0)
    max_alignment_iod: float = Field(default=0.25, gt=0.0)
    min_sharpness_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    calibration_frames: int = Field(default=8, ge=2, le=120)
    calibration_window: int = Field(default=120, ge=8, le=1000)
    # temporal filters (One-Euro): pose is smooth, expression keeps blink-speed events
    pose_min_cutoff: float = Field(default=1.5, gt=0.0)
    pose_beta: float = Field(default=0.03, ge=0.0)
    pose_d_cutoff: float = Field(default=8.0, gt=0.0)
    expression_min_cutoff: float = Field(default=6.0, gt=0.0)
    expression_beta: float = Field(default=1.0, ge=0.0)
    expression_d_cutoff: float = Field(default=15.0, gt=0.0)
    anchor_min_cutoff: float = Field(default=3.0, gt=0.0)
    anchor_beta: float = Field(default=0.05, ge=0.0)
    anchor_d_cutoff: float = Field(default=8.0, gt=0.0)
    # expression transfer
    expression_gain: float = Field(default=1.0, ge=0.0, le=1.5)
    mouth_hole_open_iod: float = Field(default=0.10, ge=0.0, description="Mouth interior is released to the target above this opening")
    eye_hole_blink: float = Field(default=0.45, ge=0.0, le=1.0)
    max_snap_iod: float = Field(default=0.15, ge=0.0, le=0.5)
    snap_enabled: bool = True


@dataclass(frozen=True)
class TargetObservation:
    """One tracked target face in canvas (crop) coordinates."""
    landmarks5: np.ndarray | None                 # (5, 2) image-left eye, image-right eye, nose, image-left mouth corner, image-right mouth corner
    landmarks106: np.ndarray | None               # (106, 2) 2d106 layout (block structure of expression_tps.BLOCKS)
    timestamp: float
    scene_cut: bool = False
    sharpness: float | None = None                # focus measure of the face crop; None = not measured


@dataclass
class ExpressionParams:
    """Target motion as inter-ocular-normalised deltas from the track's own neutral. All zero = neutral."""
    mouth_open: float = 0.0     # >= 0, lower-lip / jaw drop
    mouth_width: float = 0.0    # + wider, - pursed
    smile: float = 0.0          # + mouth corners up, - down
    blink_a: float = 0.0        # 0 open .. 1 closed, image-left eye
    blink_b: float = 0.0
    brow_a: float = 0.0         # + raised
    brow_b: float = 0.0

    NAMES = ("mouth_open", "mouth_width", "smile", "blink_a", "blink_b", "brow_a", "brow_b")

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.NAMES], np.float64)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "ExpressionParams":
        return cls(**{n: float(x) for n, x in zip(cls.NAMES, v)})

    def clamped(self) -> "ExpressionParams":
        return ExpressionParams(
            mouth_open=float(np.clip(self.mouth_open, 0.0, 0.55)), mouth_width=float(np.clip(self.mouth_width, -0.25, 0.30)),
            smile=float(np.clip(self.smile, -0.12, 0.22)), blink_a=float(np.clip(self.blink_a, 0.0, 1.0)),
            blink_b=float(np.clip(self.blink_b, 0.0, 1.0)), brow_a=float(np.clip(self.brow_a, -0.12, 0.16)),
            brow_b=float(np.clip(self.brow_b, -0.12, 0.16)))


@dataclass
class ReenactmentFrame:
    engaged: bool                                   # the caller should use `image`/`coverage` (blended by `weight`)
    weight: float                                   # 0..1 cross-fade weight of the engine output
    reason: str                                     # "OK" or the reason the engine is not used
    confidence: float                               # measured: min of the measured components
    components: dict[str, float | None]             # None = could not be measured (never faked)
    image: np.ndarray | None = None
    coverage: np.ndarray | None = None
    pose: ViewPoseFit | None = None                 # raw (unsmoothed) solve
    smoothed_pose_deg: tuple[float, float, float] | None = None
    params: ExpressionParams | None = None          # smoothed, clamped
    vertices: np.ndarray | None = None
    projected: np.ndarray | None = None             # (V, 2) canvas pixels
    alignment_error_iod: float | None = None        # eye/mouth anchor residual AFTER snapping, in inter-ocular units
    expression_held: bool = False
    deformation_strength: float = 1.0
    frame_index: int = 0
    camera: Any = None                              # WeakPerspectiveCamera of the SMOOTHED pose, acting on mesh coordinates


# =========================================================================== rig
def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gauss(d2: np.ndarray, rho: float) -> np.ndarray:
    return np.exp(-d2 / (2.0 * rho * rho))


class IdentityRig:
    """The Stage 3 asset prepared for animation: rest geometry, anatomical weight fields, texture and coverage."""

    def __init__(self, asset) -> None:
        manifest = asset.manifest
        if manifest.schema_id != "roop.identity_asset.v1":
            raise DomainError("RIG_ASSET_INVALID", "Unsupported identity asset schema", 422)
        if manifest.settings.get("template", "canonical") != "canonical":
            # The rig's eyelid / brow / mouth-ring / jaw kernels are defined on the canonical 68-point topology.
            raise DomainError("IDENTITY_TEMPLATE_UNSUPPORTED",
                              f"Expression reenactment needs a canonical-template identity asset, not "
                              f"{manifest.settings.get('template')!r}. Build the asset with template='canonical'.", 422)
        self.asset = asset
        self.build_key = manifest.build_key
        subdiv = int(manifest.settings.get("subdivisions", 2))
        template = identity_template(subdiv)
        if len(template.vertices) != len(asset.vertices):
            raise DomainError("RIG_ASSET_INVALID", "Identity asset does not match its template topology", 422)
        self.T = template.vertices.astype(np.float64)
        self.V0 = asset.vertices.astype(np.float64)
        self.faces = asset.faces.astype(np.int32)
        self.normals0 = asset.normals.astype(np.float64)
        self.rest_normals_face = self._face_normals(self.V0)
        self.front_faces = self.rest_normals_face[:, 2] > 0.3
        self.texture = asset.texture
        size = asset.texture.shape[0]
        self.uv_pixels = np.column_stack([asset.uvs[:, 0] * (size - 1), (1.0 - asset.uvs[:, 1]) * (size - 1)])
        # Coverage follows the atlas's own per-texel confidence: a weakly observed texel (a seam between loosely registered
        # views, a grazing sample) is released to the target instead of being painted at full strength.
        confidence = np.clip(asset.texel_confidence / 0.5, 0.0, 1.0)
        cov = np.where(asset.texel_state == int(ConfidenceState.OBSERVED), confidence,
                       np.where(asset.texel_state == int(ConfidenceState.INTERPOLATED), 0.4 * confidence, 0.0)).astype(np.float32)
        self.texture_coverage = cv2.GaussianBlur(cov, (0, 0), 1.2)
        self.observed_share = float(np.mean(asset.texel_state == int(ConfidenceState.OBSERVED)))
        v0 = self.V0
        self.eye_a, self.eye_b = v0[list(EYE_A)].mean(axis=0), v0[list(EYE_B)].mean(axis=0)
        self.iod = float(np.linalg.norm(self.eye_a - self.eye_b))
        self.center = np.vstack([self.eye_a, self.eye_b, v0[NOSE_TIP], v0[MOUTH_A], v0[MOUTH_B]]).mean(axis=0)
        # the five anchors the target camera is solved against (layout independent: asset mesh vertices)
        self.anchors5 = np.vstack([self.eye_a, self.eye_b, v0[NOSE_TIP], v0[MOUTH_A], v0[MOUTH_B]])
        near = [int(np.argmin(np.linalg.norm(v0 - p, axis=1))) for p in self.anchors5]
        self.anchor_normals = self.normals0[near]
        self.pose_weights = np.array([1.0, 1.0, 1.5, 0.35, 0.35])           # mouth corners move with expression
        # Pose range the identity was actually OBSERVED over (refined yaw/pitch of the views that built the asset).
        used = [src for src in manifest.sources if src.used and src.refined_yaw is not None]
        self.yaw_range = (min(u.refined_yaw for u in used), max(u.refined_yaw for u in used)) if used else (0.0, 0.0)
        self.pitch_range = (min(u.refined_pitch for u in used), max(u.refined_pitch for u in used)) if used else (0.0, 0.0)
        self.depth_triangulated = bool(manifest.claims.get("depth_triangulated", False))
        self._build_weights()

    @staticmethod
    def _face_normals_of(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        n = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def _face_normals(self, vertices: np.ndarray) -> np.ndarray:
        return self._face_normals_of(vertices, self.faces)

    def _build_weights(self) -> None:
        T, v0 = self.T, self.V0
        x, y = T[:, 0], T[:, 1]
        # jaw: lower lip + chin follow the jaw, the upper lip and mouth corners (nearly) do not; fixed near the ears
        self.w_jaw = _smoothstep(y, 0.044, 0.056) * (1.0 - _smoothstep(np.abs(x), 0.050, 0.078))
        # Triangles that span the lip line (jaw weight 0 -> 1) ARE the mouth gap: they shear by design and are released to the
        # target by `_release_holes`. Every other triangle moves (nearly) rigidly and is held to the no-fold guarantee.
        jaw_span = self.w_jaw[self.faces].max(axis=1) - self.w_jaw[self.faces].min(axis=1)
        # (Rim slivers - edge-on to the viewer even at rest - are not part of the visible face surface.)
        self.fold_faces = (jaw_span < 0.5) & self.front_faces
        ears = (v0[0] + v0[16]) / 2.0
        self.hinge = ears + np.array([0.0, 0.0, -0.04])
        # Rotating about the x-axis hinge moves the lower lip DOWN by theta * (its depth in front of the hinge) - the
        # component perpendicular to the lever arm - so that depth, not the arm length, converts an opening to an angle.
        self.lip_rel = (v0[57] - self.hinge)[1:3]                 # (dy, dz) of the lower lip relative to the hinge
        self.lever = float(max(self.lip_rel[1], 1e-3))
        xy = T[:, :2]
        d2 = lambda p: ((xy - np.asarray(p)[:2]) ** 2).sum(axis=1)
        self.k_corner_a, self.k_corner_b = _gauss(d2(T[MOUTH_A]), 0.014), _gauss(d2(T[MOUTH_B]), 0.014)
        self.k_cheek_a = _gauss(d2((-0.048, 0.015)), 0.024)
        self.k_cheek_b = _gauss(d2((0.048, 0.015)), 0.024)
        self.eye_kernels = []
        for eye in (EYE_A, EYE_B):
            center = T[list(eye)].mean(axis=0)
            dx, dy = xy[:, 0] - center[0], xy[:, 1] - center[1]
            kernel = np.exp(-(dx ** 2 / (2 * 0.013 ** 2) + dy ** 2 / (2 * 0.009 ** 2)))
            sign = np.tanh(dy / 0.0025)
            idx = list(eye)
            half_height = float(np.linalg.norm(v0[idx[1]] - v0[idx[5]]) + np.linalg.norm(v0[idx[2]] - v0[idx[4]])) / 4.0
            self.eye_kernels.append((kernel, sign, max(half_height, 1e-4)))
        self.k_brow_a = np.max([_gauss(d2(T[i]), 0.016) for i in BROW_A], axis=0)
        self.k_brow_b = np.max([_gauss(d2(T[i]), 0.016) for i in BROW_B], axis=0)

    def _jaw_angle(self, drop: float, max_angle: float = 0.70) -> float:
        """Jaw angle (rad) whose rotation moves the lower lip DOWN by exactly `drop` metres: the exact inverse of
        drop(theta) = dz sin(theta) - dy (1 - cos(theta)), monotonic on [0, max_angle] (bisection)."""
        if drop <= 0:
            return 0.0
        dy, dz = float(self.lip_rel[0]), float(self.lip_rel[1])
        f = lambda t: dz * math.sin(t) - dy * (1.0 - math.cos(t))
        if drop >= f(max_angle):
            return max_angle
        lo, hi = 0.0, max_angle
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) < drop else (lo, mid)
        return 0.5 * (lo + hi)

    def pose_coverage(self, yaw: float, pitch: float, margin: float, ramp: float) -> float:
        """1 inside the observed pose range (+margin), falling linearly to 0 over `ramp` degrees: beyond it the identity
        geometry would be extrapolated from its prior, which this engine refuses to present as measured."""
        def axis(value: float, low: float, high: float) -> float:
            excess = max(low - margin - value, value - high - margin, 0.0)
            return float(np.clip(1.0 - excess / ramp, 0.0, 1.0))
        return min(axis(yaw, *self.yaw_range), axis(pitch, *self.pitch_range))

    # ------------------------------------------------------------------ deformation
    def deform(self, params: ExpressionParams, gain: float = 1.0, max_disp: float = 0.045) -> tuple[np.ndarray, float]:
        """Source anatomy + target motion. Returns (vertices, applied strength). Strength < 1: scaled back to avoid folds."""
        p = params.clamped()
        iod = self.iod
        v = self.V0.copy()
        disp = np.zeros_like(v)
        # jaw rotation about the (identity-specific) hinge; opening also drops the chin backwards
        theta = self._jaw_angle(gain * p.mouth_open * iod)
        if theta > 0:
            rel = v - self.hinge
            ang = -theta * self.w_jaw
            c, s = np.cos(ang), np.sin(ang)
            y = rel[:, 1] * c - rel[:, 2] * s
            z = rel[:, 1] * s + rel[:, 2] * c
            disp += np.column_stack([np.zeros(len(v)), y - rel[:, 1], z - rel[:, 2]])
        # mouth width / smile at the corners, cheeks follow the smile
        raise_ = gain * p.smile * iod
        widen = gain * p.mouth_width * iod / 2.0
        disp[:, 1] += -raise_ * (self.k_corner_a + self.k_corner_b) - 0.45 * raise_ * (self.k_cheek_a + self.k_cheek_b)
        disp[:, 0] += -widen * self.k_corner_a + widen * self.k_corner_b - 0.10 * 2 * widen * (self.k_cheek_a - self.k_cheek_b)
        # eyelids
        for blink, (kernel, sign, half) in zip((p.blink_a, p.blink_b), self.eye_kernels):
            if blink > 0:
                depth = np.abs(sign) * kernel
                disp[:, 1] += blink * half * depth * np.where(sign < 0, 0.8, -0.5)
        # brows
        disp[:, 1] += -gain * iod * 0.8 * (p.brow_a * self.k_brow_a + p.brow_b * self.k_brow_b)
        norm = np.linalg.norm(disp, axis=1, keepdims=True)
        disp = disp * np.minimum(1.0, max_disp / np.maximum(norm, 1e-12))
        strength = 1.0
        for _ in range(8):
            trial = v + strength * disp
            # A fold is a triangle that has flipped over relative to ITS OWN rest orientation. (Tilting edge-on to the
            # camera, e.g. the chin underside of an open jaw, is foreshortening, not a fold.)
            kept = np.einsum("ij,ij->i", self._face_normals(trial), self.rest_normals_face)
            if not np.any(kept[self.fold_faces] < 0.15):
                break
            strength *= 0.7
        else:
            strength = 0.0
        return v + strength * disp, float(strength)


# =========================================================================== measurement
@dataclass
class ExpressionMeasure:
    mouth_open: float
    mouth_width: float
    smile: float
    eye_h_a: float          # vertical eye opening / foreshortening-corrected inter-ocular distance
    eye_h_b: float
    brow_a: float
    brow_b: float
    iod_px: float

    NAMES = ("mouth_open", "mouth_width", "smile", "eye_h_a", "eye_h_b", "brow_a", "brow_b")

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.NAMES], np.float64)


def measure_expression(lm5: np.ndarray, lm106: np.ndarray, yaw_deg: float) -> ExpressionMeasure:
    """Block-level expression ratios in the face's own (rolled) frame, normalised by the foreshortened inter-ocular distance.

    Uses only the block STRUCTURE of the 106-point layout (which points belong to mouth / eyes / brows), never a
    point-by-point correspondence, because a detector's ordering is not trusted beyond what ``layout_is_plausible`` checks.
    """
    lm5 = np.asarray(lm5, np.float64)
    lm106 = np.asarray(lm106, np.float64)
    across = lm5[1] - lm5[0]
    iod_img = float(np.linalg.norm(across))
    if iod_img < 2.0:
        raise DomainError("EXPRESSION_UNMEASURABLE", "Inter-ocular distance too small to measure expression", 422)
    across = across / iod_img
    down = np.array([-across[1], across[0]])
    iod_eff = iod_img / max(math.cos(math.radians(yaw_deg)), 0.5)
    origin = lm5[:2].mean(axis=0)
    coords = lambda pts: np.column_stack([(pts - origin) @ across, (pts - origin) @ down])
    mouth = coords(group(lm106, "mouth"))
    eyes = sorted((coords(group(lm106, "eye_a")), coords(group(lm106, "eye_b"))), key=lambda c: c[:, 0].mean())
    brows = sorted((coords(group(lm106, "brow_a")), coords(group(lm106, "brow_b"))), key=lambda c: c[:, 0].mean())
    u, v = mouth[:, 0], mouth[:, 1]
    corner_v = 0.5 * (v[np.argmin(u)] + v[np.argmax(u)])
    ar = [float(np.ptp(e[:, 1]) / iod_eff) for e in eyes]
    brow = [float((e[:, 1].mean() - b[:, 1].mean()) / iod_eff) for e, b in zip(eyes, brows)]
    return ExpressionMeasure(
        mouth_open=float(np.ptp(v) / iod_eff), mouth_width=float(np.ptp(u) / iod_img),
        smile=float((np.median(v) - corner_v) / iod_eff), eye_h_a=ar[0], eye_h_b=ar[1], brow_a=brow[0], brow_b=brow[1],
        iod_px=iod_img)


class NeutralCalibrator:
    """Per-track baselines: the track's own neutral, taken as robust percentiles over a sliding window."""

    def __init__(self, window: int, min_frames: int):
        self.min_frames = min_frames
        self.history: deque[np.ndarray] = deque(maxlen=window)

    def push(self, vector: np.ndarray) -> None:
        self.history.append(np.asarray(vector, np.float64))

    @property
    def calibrated(self) -> bool:
        return len(self.history) >= self.min_frames

    def baseline(self) -> np.ndarray:
        data = np.stack(self.history)
        base = np.median(data, axis=0)
        base[0] = np.percentile(data[:, 0], 10)       # mouth_open: neutral = nearly closed
        base[3] = np.percentile(data[:, 3], 90)       # eye openings: neutral = open
        base[4] = np.percentile(data[:, 4], 90)
        return base


def params_from_measure(m: ExpressionMeasure, base: np.ndarray) -> ExpressionParams:
    b = dict(zip(ExpressionMeasure.NAMES, base))
    def blink(opening: float, ref: float) -> float:
        return float(np.clip((1.0 - opening / max(ref, 1e-6)) / 0.6, 0.0, 1.0)) if ref > 0.01 else 0.0
    return ExpressionParams(
        mouth_open=max(0.0, m.mouth_open - b["mouth_open"]), mouth_width=m.mouth_width - b["mouth_width"],
        smile=m.smile - b["smile"], blink_a=blink(m.eye_h_a, b["eye_h_a"]), blink_b=blink(m.eye_h_b, b["eye_h_b"]),
        brow_a=m.brow_a - b["brow_a"], brow_b=m.brow_b - b["brow_b"])


# =========================================================================== engine
class ReenactmentEngine:
    """Per-track state machine. Deterministic: the output is a pure function of the observation sequence."""

    def __init__(self, rig: IdentityRig, settings: ReenactmentSettings | None = None):
        self.rig = rig
        self.s = settings or ReenactmentSettings()
        self.sampling_device = "cpu"
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        s = self.s
        self.frame_index = 0
        self.calibrator = NeutralCalibrator(s.calibration_window, s.calibration_frames)
        def temporal(min_cutoff, beta, derivative_cutoff):
            return TrackTemporalStabilizer(min_cutoff=min_cutoff, beta=beta, derivative_cutoff=derivative_cutoff,
                                           process_noise=8.0, measurement_noise=1.0)
        self.pose_temporal = temporal(s.pose_min_cutoff, s.pose_beta, s.pose_d_cutoff)
        self.expression_temporal = temporal(s.expression_min_cutoff, s.expression_beta, s.expression_d_cutoff)
        self.anchor_temporal = temporal(s.anchor_min_cutoff, s.anchor_beta, s.anchor_d_cutoff)
        self.last_time: float | None = None
        self.last_pose_deg: np.ndarray | None = None
        self.last_params = ExpressionParams()
        self.last_expression_vec: np.ndarray | None = None
        self.sharpness_ref: float | None = None
        self.good_run = 0
        self.bad_run = 0
        self.engaged = False
        self.ramp = 0.0

    # ------------------------------------------------------------------ gate helpers
    def _fallback(self, reason: str, hard: bool, comps: dict, index: int) -> ReenactmentFrame:
        # No render exists for this frame, so the engine output is withdrawn at once (the caller's proven path runs).
        self.good_run = 0
        self.bad_run += 1
        self.engaged = False
        self.ramp = 0.0
        conf = min([c for c in comps.values() if c is not None], default=0.0)
        return ReenactmentFrame(engaged=False, weight=0.0, reason=reason, confidence=float(conf), components=comps,
                                frame_index=index)

    @staticmethod
    def _valid_points(points, count: int, canvas: tuple[int, int], margin: float = 0.25) -> bool:
        if points is None:
            return False
        p = np.asarray(points, np.float64)
        if p.shape != (count, 2) or not np.isfinite(p).all():
            return False
        h, w = canvas
        return bool(np.all((p[:, 0] > -margin * w) & (p[:, 0] < (1 + margin) * w) & (p[:, 1] > -margin * h) & (p[:, 1] < (1 + margin) * h)))

    # ------------------------------------------------------------------ main
    def process(self, obs: TargetObservation, canvas: tuple[int, int]) -> ReenactmentFrame:
        s, rig = self.s, self.rig
        if obs.scene_cut:
            self.reset()
        index = self.frame_index
        self.frame_index += 1
        comps: dict[str, float | None] = {"landmarks": None, "layout": None, "pose_fit": None, "pose_support": None,
                                          "pose_coverage": None, "alignment": None, "temporal": None, "sharpness": None,
                                          "calibration": None}
        if not np.isfinite(obs.timestamp) or (self.last_time is not None and obs.timestamp <= self.last_time):
            return self._fallback("TIMESTAMP_INVALID", True, comps, index)
        if not (self._valid_points(obs.landmarks5, 5, canvas) and self._valid_points(obs.landmarks106, 106, canvas)):
            comps["landmarks"] = 0.0
            return self._fallback("LANDMARKS_MISSING_OR_INVALID", True, comps, index)
        comps["landmarks"] = 1.0
        lm5, lm106 = np.asarray(obs.landmarks5, np.float64), np.asarray(obs.landmarks106, np.float64)
        if not layout_is_plausible(lm106):
            comps["layout"] = 0.0
            return self._fallback("LANDMARK_LAYOUT_IMPLAUSIBLE", True, comps, index)
        comps["layout"] = 1.0

        # ---- 2. camera of the target head against the SOURCE's 3D anchors
        pose0 = estimate_pose_from_landmarks(lm5)
        if abs(pose0[0]) > s.max_abs_yaw_deg + 15.0:
            return self._fallback("POSE_OUT_OF_SUPPORTED_RANGE", True, comps, index)
        try:
            pose = solve_view_pose(lm5, rig.anchors5, rig.anchor_normals, pose0, rig.pose_weights)
        except DomainError:
            return self._fallback("POSE_SOLVE_FAILED", True, comps, index)
        comps["pose_fit"] = float(math.exp(-((pose.rms_iod / max(s.max_pose_rms_iod, 1e-9)) ** 2) * 0.5))
        comps["pose_coverage"] = rig.pose_coverage(pose.yaw, pose.pitch, s.coverage_margin_deg, s.coverage_ramp_deg)
        raw_pose = np.array([pose.yaw, pose.pitch, pose.roll], np.float64)
        if abs(pose.yaw) > s.max_abs_yaw_deg:
            return self._fallback("POSE_OUT_OF_SUPPORTED_RANGE", True, comps, index)
        dt = None if self.last_time is None else obs.timestamp - self.last_time
        temporal_ok = True
        if dt is not None and self.last_pose_deg is not None and dt > 0:
            rate = float(np.max(np.abs(raw_pose - self.last_pose_deg)) / dt)
            temporal_ok = rate <= s.max_pose_rate_deg_s
        comps["temporal"] = 1.0 if temporal_ok else 0.0
        if not temporal_ok:
            # an impossible jump is a measurement error: do not feed it to the filters
            return self._fallback("POSE_JUMP_EXCEEDS_PHYSICAL_RATE", False, comps, index)

        # ---- 3. expression relative to the track's own neutral
        held = False
        measured = None
        if abs(pose.yaw) <= s.expression_max_yaw_deg:
            try:
                measured = measure_expression(lm5, lm106, pose.yaw)
            except DomainError:
                measured = None
        if measured is not None:
            self.calibrator.push(measured.vector())
        calibrated = self.calibrator.calibrated
        comps["calibration"] = 1.0 if calibrated else 0.0
        if measured is not None and calibrated:
            raw_params = params_from_measure(measured, self.calibrator.baseline())
            vec = raw_params.clamped().vector() * s.expression_gain
        elif self.last_expression_vec is not None:
            vec, held = self.last_expression_vec, True          # too turned to measure: hold, never invent
        else:
            vec, held = np.zeros(7), True

        # ---- 4. temporal stabilisation (state only advances on accepted frames)
        t = obs.timestamp
        pose_measurement = [raw_pose[0], raw_pose[1], raw_pose[2], math.log(pose.scale), pose.tx, pose.ty]
        smoothed = self.pose_temporal.filter(0, "head_pose", pose_measurement, t)
        # Expression coefficients and facial anchors use the same track-local One-Euro + Kalman predictor.
        vec_s = self.expression_temporal.filter(0, "expression_coefficients", vec, t)
        anchors_raw = np.concatenate([lm5.ravel(), group(lm106, "mouth").mean(axis=0)])
        anchors_s = self.anchor_temporal.filter(0, "expression_anchors", anchors_raw, t)
        self.last_time, self.last_pose_deg, self.last_expression_vec = t, raw_pose.copy(), vec.copy()
        params = ExpressionParams.from_vector(vec_s).clamped()
        self.last_params = params

        # ---- 5./6. deform + render
        vertices, strength = rig.deform(params, gain=1.0)
        yaw, pitch, roll, log_scale, tx, ty = smoothed
        smooth_fit = ViewPoseFit(float(yaw), float(pitch), float(roll), float(math.exp(log_scale)), float(tx), float(ty),
                                 rig.center, pose.rms_mm, pose.rms_iod, pose.converged)
        camera = smooth_fit.camera()
        rot = camera.rotation_matrix()
        projected = camera.project(vertices)
        camera_vertices = self._camera_vertices(vertices, rot)

        # ---- 7. anchor correspondence: eyes / nose / mouth corners / mouth centre onto the (filtered) target
        anchor_idx = (list(EYE_A), list(EYE_B), [NOSE_TIP], [MOUTH_A], [MOUTH_B], list(range(48, 60)))
        rendered_anchor = np.vstack([projected[idx].mean(axis=0) for idx in anchor_idx])
        target_anchor = np.vstack([anchors_s[:10].reshape(5, 2), anchors_s[10:12]])
        error = target_anchor - rendered_anchor
        iod_px = float(np.linalg.norm(target_anchor[1] - target_anchor[0]))
        raw_alignment = float(np.max(np.linalg.norm(error, axis=1)) / max(iod_px, 1e-6))
        comps["alignment"] = float(math.exp(-((raw_alignment / max(s.max_alignment_iod, 1e-9)) ** 2) * 0.5))
        shown = projected
        if s.snap_enabled and s.max_snap_iod > 0:
            cap = s.max_snap_iod * iod_px
            capped = error * np.minimum(1.0, cap / np.maximum(np.linalg.norm(error, axis=1, keepdims=True), 1e-9))
            rho = 0.45 * iod_px
            k = np.exp(-(np.linalg.norm(projected[:, None] - rendered_anchor[None], axis=2) ** 2) / (2 * rho * rho))
            shown = projected + (k @ capped) / (k.sum(axis=1, keepdims=True) + 0.05)
        post = np.vstack([shown[idx].mean(axis=0) for idx in anchor_idx])
        alignment_after = float(np.max(np.linalg.norm(target_anchor - post, axis=1)) / max(iod_px, 1e-6))

        h, w = canvas
        image, coverage, support = self._render(shown, camera_vertices, (h, w))
        coverage = self._release_holes(coverage, shown, params, iod_px)
        # below the minimum the observed texture cannot support this pose at all; above it the score rises to 1 at 75 %
        comps["pose_support"] = 0.0 if support < s.min_pose_support else float(
            np.clip(0.4 + 0.6 * (support - s.min_pose_support) / max(0.75 - s.min_pose_support, 1e-6), 0.0, 1.0))
        # ---- sharpness (motion blur): ratio to the track's own sharp level
        if obs.sharpness is not None and np.isfinite(obs.sharpness):
            self.sharpness_ref = obs.sharpness if self.sharpness_ref is None else max(self.sharpness_ref * 0.995, obs.sharpness)
            ratio = obs.sharpness / max(self.sharpness_ref, 1e-9)
            comps["sharpness"] = float(np.clip((ratio - s.min_sharpness_ratio * 0.5) / (s.min_sharpness_ratio), 0.0, 1.0))
        measured_vals = [v for v in comps.values() if v is not None]
        confidence = float(min(measured_vals)) if measured_vals else 0.0

        # ---- 8. hysteresis
        reason = "OK"
        if not calibrated:
            reason = "EXPRESSION_NOT_CALIBRATED"
        elif comps["pose_coverage"] == 0.0:
            reason = "POSE_OUTSIDE_OBSERVED_COVERAGE"
        elif confidence < s.exit_confidence or comps["pose_support"] == 0.0:
            reason = "LOW_GEOMETRY_CONFIDENCE:" + min((k for k, v in comps.items() if v is not None), key=lambda k: comps[k])
        usable = reason == "OK" and confidence >= (s.exit_confidence if self.engaged else s.enter_confidence)
        if usable:
            self.good_run += 1
            self.bad_run = 0
            if not self.engaged and self.good_run >= s.enter_frames:
                self.engaged = True
        else:
            self.good_run = 0
            self.bad_run += 1
            if self.engaged and self.bad_run >= s.exit_frames:
                self.engaged = False
            if reason == "OK":
                reason = "CONFIDENCE_BELOW_THRESHOLD"
        step = 1.0 / s.ramp_frames
        self.ramp = min(1.0, self.ramp + step) if (self.engaged and usable) else max(0.0, self.ramp - step) if not self.engaged else self.ramp
        engaged = self.ramp > 0.0
        return ReenactmentFrame(
            engaged=engaged, weight=float(self.ramp), reason="OK" if (engaged and reason == "OK") else (reason if reason != "OK" else "ENTERING"),
            confidence=confidence, components=comps, image=image if engaged else None, coverage=coverage if engaged else None,
            pose=pose, smoothed_pose_deg=(float(yaw), float(pitch), float(roll)), params=params, vertices=vertices, projected=shown,
            alignment_error_iod=alignment_after, expression_held=held, deformation_strength=strength, frame_index=index,
            camera=camera)

    # ------------------------------------------------------------------ rendering
    def _camera_vertices(self, vertices: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """z toward the viewer -> depth along a far pinhole camera (smaller = closer), as `rasterize` expects."""
        rotated = vertices @ rot.T
        return np.column_stack([rotated[:, 0], rotated[:, 1], FAR_CAMERA_M - rotated[:, 2]])

    def _render(self, pixels: np.ndarray, camera_vertices: np.ndarray, shape: tuple[int, int]):
        """Painter's-order rasteriser (far -> near by triangle depth) + vectorised barycentric UV lookup.

        The identity mesh is a smooth facial height field under a weak-perspective camera, so depth-sorted triangles are
        exact for it and barycentrics are affine. (``pose_blend.rasterize`` is the independent reference this is tested
        against.)"""
        rig = self.rig
        faces = rig.faces[:, [0, 2, 1]]                 # the z reflection reverses winding (same fix as PoseBlendProjector)
        normals = np.cross(camera_vertices[faces[:, 1]] - camera_vertices[faces[:, 0]],
                           camera_vertices[faces[:, 2]] - camera_vertices[faces[:, 0]])
        keep = np.where(normals[:, 2] < 0)[0]            # front-facing (toward the camera at the origin)
        owner = np.full(shape, -1.0, np.float32)
        order = keep[np.argsort(-camera_vertices[faces[keep]][:, :, 2].mean(axis=1))]   # farthest first
        fixed = np.round(pixels[faces[order]] * 16.0).astype(np.int32)       # 4 fractional bits: sub-pixel ownership
        for face_id, tri in zip(order, fixed):
            cv2.fillConvexPoly(owner, tri, float(face_id), cv2.LINE_8, 4)
        footprint = owner >= 0
        image = np.zeros(shape + (3,), np.uint8)
        coverage = np.zeros(shape, np.float32)
        if footprint.any():
            ys, xs = np.nonzero(footprint)
            tri_id = owner[ys, xs].astype(np.int64)
            tri = faces[tri_id]
            a, b, c = pixels[tri[:, 0]], pixels[tri[:, 1]], pixels[tri[:, 2]]
            point = np.column_stack([xs, ys]).astype(np.float64)
            det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
            ok = np.abs(det) > 1e-9
            det = np.where(ok, det, 1.0)
            l0 = ((b[:, 1] - c[:, 1]) * (point[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (point[:, 1] - c[:, 1])) / det
            l1 = ((c[:, 1] - a[:, 1]) * (point[:, 0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (point[:, 1] - c[:, 1])) / det
            # a pixel a hair outside its triangle is projected onto it (clamp THEN renormalise, so the weights still sum to 1)
            bary = np.clip(np.column_stack([l0, l1, 1 - l0 - l1]), 0.0, 1.0)
            bary = bary / np.maximum(bary.sum(axis=1, keepdims=True), 1e-9)
            uv = (bary[:, :1] * rig.uv_pixels[tri[:, 0]] + bary[:, 1:2] * rig.uv_pixels[tri[:, 1]]
                  + bary[:, 2:3] * rig.uv_pixels[tri[:, 2]])
            from roop_3d.reconstruction.uv_synthesizer import remap_points
            image[ys, xs] = remap_points(rig.texture, uv[:, 0], uv[:, 1], cv2.INTER_LINEAR,
                                         device=self.sampling_device).reshape(-1, 3)
            coverage[ys, xs] = remap_points(rig.texture_coverage, uv[:, 0], uv[:, 1], cv2.INTER_LINEAR,
                                             device=self.sampling_device).reshape(-1) * ok
        inside = footprint
        support = float(coverage[inside].mean()) if inside.any() else 0.0
        coverage = coverage * cv2.GaussianBlur(footprint.astype(np.float32), (0, 0), 1.0)
        image[coverage <= 0] = 0
        return image, np.clip(coverage, 0.0, 1.0).astype(np.float32), support

    def _release_holes(self, coverage: np.ndarray, pixels: np.ndarray, p: ExpressionParams, iod_px: float) -> np.ndarray:
        """The atlas has no mouth interior and no eyelid skin: release those regions to the target instead of stretching texture."""
        s = self.s
        out = coverage.copy()
        shape = coverage.shape
        if p.mouth_open >= s.mouth_hole_open_iod:
            hole = np.zeros(shape, np.float32)
            cv2.fillConvexPoly(hole, cv2.convexHull(np.round(pixels[list(INNER_LIP)]).astype(np.int32)), 1.0)
            grow = max(1, int(round(0.05 * iod_px)))
            hole = cv2.GaussianBlur(cv2.dilate(hole, np.ones((grow, grow), np.uint8)), (0, 0), max(1.0, 0.02 * iod_px))
            strength = float(np.clip((p.mouth_open - s.mouth_hole_open_iod) / 0.06, 0.0, 1.0))
            out *= 1.0 - strength * hole
        for blink, eye in ((p.blink_a, EYE_A), (p.blink_b, EYE_B)):
            if blink >= s.eye_hole_blink:
                hole = np.zeros(shape, np.float32)
                cv2.fillConvexPoly(hole, cv2.convexHull(np.round(pixels[list(eye)]).astype(np.int32)), 1.0)
                grow = max(1, int(round(0.06 * iod_px)))
                hole = cv2.GaussianBlur(cv2.dilate(hole, np.ones((grow, grow), np.uint8)), (0, 0), max(1.0, 0.02 * iod_px))
                out *= 1.0 - float(np.clip((blink - s.eye_hole_blink) / 0.25, 0.0, 1.0)) * hole
        return out
