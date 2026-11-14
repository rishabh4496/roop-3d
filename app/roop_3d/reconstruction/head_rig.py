"""Full-head rig built from an identity asset (MakeHuman CC0 head + neck): rest geometry, weight fields and an expression deformer.

Moved out of the former 2.5D reenactment module. The canonical frontal-strip template is not supported.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.geometry_types import ConfidenceState
from roop_3d.reconstruction.reenactment import (BROW_A, BROW_B, EYE_A, EYE_B, MOUTH_A, MOUTH_B, NOSE_TIP, ExpressionParams)


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gauss(d2: np.ndarray, rho: float) -> np.ndarray:
    return np.exp(-d2 / (2.0 * rho * rho))


class HeadRig:
    """The identity asset (full-head MakeHuman template) prepared for animation: rest geometry, anatomical weight fields, texture and
    coverage. This is the rig contract ``FullHeadEngine`` consumes (``V0``, ``faces``, ``landmark_indices``, ``texture``...)."""

    def __init__(self, asset) -> None:
        manifest = asset.manifest
        if manifest.schema_id != "roop.identity_asset.v1":
            raise DomainError("RIG_ASSET_INVALID", "Unsupported identity asset schema", 422)
        self.asset = asset
        self.build_key = manifest.build_key
        if manifest.settings.get("template", "makehuman") != "makehuman":
            raise DomainError("RIG_ASSET_INVALID", "Only full-head (MakeHuman) identity assets can be rigged; the canonical strip template was removed", 422)
        from roop_3d.reconstruction.makehuman_template import makehuman_template
        template = makehuman_template()
        self.landmark_indices = template.landmark_vertices["68pt"]
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
        self._view_appearance_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        self._view_appearance_aggregator = None
        self._view_uv_normals = None
        view_textures = np.asarray(getattr(asset, "view_textures", ()))
        if view_textures.ndim == 4 and len(view_textures) and view_textures.shape[-1] == 3:
            from roop_3d.reconstruction.geometry_types import Mesh3D
            from roop_3d.reconstruction.uv_synthesizer import UVSynthesizer
            uv_for_image = np.column_stack((asset.uvs[:, 0], 1.0 - asset.uvs[:, 1]))
            uv_mesh = Mesh3D(self.V0.astype(np.float32), self.faces, uvs=uv_for_image,
                             normals=self.normals0.astype(np.float32))
            uv_surface = UVSynthesizer(uv_mesh, uv_size=view_textures.shape[1],
                                       max_interpolation_fraction=0)
            self._view_uv_normals = np.asarray(uv_surface.norm_map, np.float32)
        self.observed_share = float(np.mean(asset.texel_state == int(ConfidenceState.OBSERVED)))
        v0 = self.V0[self.landmark_indices]
        self.eye_a, self.eye_b = v0[list(EYE_A)].mean(axis=0), v0[list(EYE_B)].mean(axis=0)
        self.iod = float(np.linalg.norm(self.eye_a - self.eye_b))
        self.center = np.vstack([self.eye_a, self.eye_b, v0[NOSE_TIP], v0[MOUTH_A], v0[MOUTH_B]]).mean(axis=0)
        # the five anchors the target camera is solved against (layout independent: asset mesh vertices)
        self.anchors5 = np.vstack([self.eye_a, self.eye_b, v0[NOSE_TIP], v0[MOUTH_A], v0[MOUTH_B]])
        near = [int(np.argmin(np.linalg.norm(self.V0 - p, axis=1))) for p in self.anchors5]
        self.anchor_normals = self.normals0[near]
        self.pose_weights = np.array([1.0, 1.0, 1.5, 0.35, 0.35])           # mouth corners move with expression
        # Pose range the identity was actually OBSERVED over (refined yaw/pitch of the views that built the asset).
        used = [src for src in manifest.sources if src.used and src.refined_yaw is not None]
        self.yaw_range = (min(u.refined_yaw for u in used), max(u.refined_yaw for u in used)) if used else (0.0, 0.0)
        self.pitch_range = (min(u.refined_pitch for u in used), max(u.refined_pitch for u in used)) if used else (0.0, 0.0)
        self.depth_triangulated = bool(manifest.claims.get("depth_triangulated", False))
        self._build_weights()

    def texture_for_pose(self, yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
        """Return a view-attended UV atlas, preferring measured side views at profile yaw."""
        textures = np.asarray(getattr(self.asset, "view_textures", ()))
        supports = np.asarray(getattr(self.asset, "view_support", ()))
        poses = np.asarray(getattr(self.asset, "view_poses", ()))
        if (textures.ndim != 4 or not len(textures) or supports.shape != textures.shape[:3]
                or poses.shape[0] != len(textures)):
            return self.texture, self.texture_coverage
        key = (round(float(yaw)), round(float(pitch)))
        cached = self._view_appearance_cache.get(key)
        if cached is not None:
            return cached
        if self._view_appearance_aggregator is None:
            from roop_3d.rendering.hybrid_renderer import MultiViewCanonicalAggregator
            self._view_appearance_aggregator = MultiViewCanonicalAggregator()
        bank_h, bank_w = textures.shape[1:3]
        base = cv2.resize(self.texture, (bank_w, bank_h), interpolation=cv2.INTER_AREA)
        base_confidence = cv2.resize(self.texture_coverage, (bank_w, bank_h), interpolation=cv2.INTER_AREA)
        appearance = self._view_appearance_aggregator.aggregate(
            textures, supports, poses, (yaw, pitch, 0.0), base_albedo=base,
            base_confidence=base_confidence, uv_normals=self._view_uv_normals)
        bank_texture = np.rint(np.clip(appearance.albedo_bgr, 0, 1) * 255).astype(np.uint8)
        texture = cv2.resize(bank_texture, (self.texture.shape[1], self.texture.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
        coverage = cv2.resize(appearance.confidence, (self.texture.shape[1], self.texture.shape[0]),
                              interpolation=cv2.INTER_LINEAR).astype(np.float32)
        result = (texture, np.clip(coverage, 0, 1))
        if len(self._view_appearance_cache) >= 8:
            self._view_appearance_cache.pop(next(iter(self._view_appearance_cache)))
        self._view_appearance_cache[key] = result
        return result

    @staticmethod
    def _face_normals_of(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        n = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def _face_normals(self, vertices: np.ndarray) -> np.ndarray:
        return self._face_normals_of(vertices, self.faces)

    def _build_weights(self) -> None:
        T, v0 = self.T[self.landmark_indices], self.V0[self.landmark_indices]
        x, y = self.T[:, 0], self.T[:, 1]
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
        xy = self.T[:, :2]
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
