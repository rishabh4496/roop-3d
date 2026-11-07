"""GPU full-head rendering. No ArcFace canvas and no software raster fallback.

Camera intrinsics are estimated unless supplied; metric size and distance cannot
both be inferred from monocular video. Scale is fixed by the source asset (metres).
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
import threading

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.reenactment import (
    ExpressionParams, NeutralCalibrator, measure_expression, params_from_measure,
)
from roop_3d.temporal import TrackTemporalStabilizer


@dataclass(frozen=True)
class HeadPose:
    rotation: np.ndarray
    translation: np.ndarray
    camera: np.ndarray
    scale: float = 1.0

    def camera_vertices(self, vertices):
        # Stored mesh has z toward the viewer; OpenCV has z away.
        return (vertices * (1, 1, -1)) @ self.rotation.T * self.scale + self.translation

    def project(self, vertices):
        camera = self.camera_vertices(vertices)
        pixels = camera @ self.camera.T
        if not np.isfinite(camera).all() or np.any(camera[:, 2] <= .001):
            raise DomainError("HEAD_BEHIND_CAMERA", "Full head intersects the camera near plane", 422)
        return pixels[:, :2] / pixels[:, 2:3]


def head_roi(pixels, side=512, padding=.08):
    """Square ROI from ALL projected head vertices, including off-frame vertices."""
    points = np.asarray(pixels, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Full-head projection must be finite Nx2")
    lo, hi = points.min(0), points.max(0)
    extent = max(float(np.max(hi - lo)) * (1 + 2 * padding), 2.0)
    origin = (lo + hi - extent) / 2
    inverse = np.array([[extent / side, 0, origin[0]], [0, extent / side, origin[1]]], np.float32)
    return inverse


class GPUHeadRasterizer:
    """Headless OpenGL depth rasterization + perspective UV interpolation.

    Contexts are thread-local because Pinokio job workers own their GL context.
    Host readback is explicit at the existing numpy compositor boundary.
    """
    def __init__(self):
        self.local = threading.local()

    def _context(self):
        if getattr(self.local, "context", None) is None:
            try:
                import moderngl
                context = moderngl.create_standalone_context(require=330,
                    **({"backend": "egl"} if sys.platform != "win32" else {}))
                renderer = context.info.get("GL_RENDERER", "").lower()
                if any(word in renderer for word in ("llvmpipe", "softpipe", "software", "swiftshader", "gdi generic")):
                    context.release()
                    raise RuntimeError("Software OpenGL is not supported")
                program = context.program(vertex_shader='''#version 330
                    in vec4 clip_position; in vec2 atlas_uv;
                    out vec2 uv;
                    void main() { gl_Position = clip_position; uv = atlas_uv; }
                    ''', fragment_shader='''#version 330
                    uniform sampler2D albedo; uniform sampler2D confidence;
                    in vec2 uv; out vec4 color;
                    void main() { color = vec4(texture(albedo, uv).rgb,
                                               texture(confidence, uv).r); }
                    ''')
                program["albedo"].value = 0
                program["confidence"].value = 1
                self.local.context, self.local.program = context, program
            except Exception as error:
                raise DomainError("GPU_HEAD_RENDERER_UNAVAILABLE",
                    f"Hardware ModernGL context required: {error}", 503) from error
        return self.local.context, self.local.program

    def render(self, vertices, faces, uvs, texture, coverage, pose, inverse, side=512):
        ctx, program = self._context()
        import moderngl
        camera = pose.camera_vertices(vertices)
        pixels = pose.project(vertices)
        local = cv2.transform(pixels[None].astype(np.float32), cv2.invertAffineTransform(inverse))[0]
        z = camera[:, 2]
        near, far = .001, max(10., float(z.max()) * 2)
        clip = np.column_stack(((2 * (local[:, 0] + .5) / side - 1) * z,
                                (1 - 2 * (local[:, 1] + .5) / side) * z,
                                (far + near) / (far - near) * z - 2 * far * near / (far - near), z))
        # UV data is uploaded in image row order, so flip OBJ v for sampling.
        data = np.column_stack((clip, uvs[:, 0], 1 - uvs[:, 1])).astype("f4")
        resources = []
        try:
            def keep(value):
                resources.append(value)
                return value
            vbo = keep(ctx.buffer(data.tobytes()))
            ibo = keep(ctx.buffer(np.asarray(faces, "i4").tobytes()))
            vao = keep(ctx.vertex_array(program, [(vbo, "4f 2f", "clip_position", "atlas_uv")], ibo))
            tex = keep(ctx.texture((texture.shape[1], texture.shape[0]), 3,
                                   np.ascontiguousarray(texture).tobytes()))
            cov = keep(ctx.texture((coverage.shape[1], coverage.shape[0]), 1,
                                   np.asarray(coverage, "f4").tobytes(), dtype="f4"))
            for unit, item in enumerate((tex, cov)):
                item.repeat_x = item.repeat_y = False
                item.filter = (moderngl.LINEAR, moderngl.LINEAR)
                item.use(unit)
            color = keep(ctx.texture((side, side), 4, dtype="f4"))
            depth = keep(ctx.depth_renderbuffer((side, side)))
            fbo = keep(ctx.framebuffer([color], depth))
            fbo.use()
            ctx.viewport = (0, 0, side, side)
            ctx.enable_only(moderngl.DEPTH_TEST)
            fbo.clear(0, 0, 0, 0, depth=1.)
            vao.render(moderngl.TRIANGLES)
            rgba = np.frombuffer(fbo.read(components=4, dtype="f4"), np.float32).reshape(side, side, 4)[::-1].copy()
            return np.clip(rgba[..., :3] * 255, 0, 255).round().astype(np.uint8), np.clip(rgba[..., 3], 0, 1)
        finally:
            for resource in reversed(resources):
                resource.release()

    def release(self):
        if getattr(self.local, "context", None) is not None:
            self.local.program.release()
            self.local.context.release()
            self.local.context = None


class FullHeadEngine:
    def __init__(self, rig, *, side=512, refiner=None):
        self.rig, self.side, self.refiner = rig, side, refiner
        self.rasterizer = GPUHeadRasterizer()
        self.calibrator = NeutralCalibrator(120, 8)
        self.temporal = TrackTemporalStabilizer(min_cutoff=1.5, beta=.03, derivative_cutoff=8.)

    def process(self, image, face, landmarks68, timestamp, *, scene_cut=False, animate=True):
        points = np.asarray(landmarks68, np.float64)
        if points.shape not in ((68, 2), (68, 3)) or not np.isfinite(points).all():
            raise DomainError("DENSE_HEAD_LANDMARKS_REQUIRED", "Full-head synthesis requires measured 68-point landmarks", 422)
        points = points[:, :2]
        if scene_cut:
            self.temporal.forget_track(0)
            self.calibrator = NeutralCalibrator(120, 8)
        h, w = image.shape[:2]
        focal = max(h, w) * 1.5
        k = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], np.float64)
        model = self.rig.V0[self.rig.landmark_indices] * (1, 1, -1)
        # Stable interior anatomy drives camera; contour landmarks constrain the
        # identity mesh, but the target's different jaw must not overwrite it.
        stable = np.arange(17, 48)
        ok, rv, tv = cv2.solvePnP(model[stable], points[stable], k, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            raise DomainError("HEAD_POSE_FAILED", "Dense head pose solve failed", 422)
        rv, tv = cv2.solvePnPRefineLM(model[stable], points[stable], k, None, rv, tv)
        fitted, _ = cv2.projectPoints(model[stable], rv, tv, k, None)
        iod = max(1., float(np.linalg.norm(face.landmarks5[1] - face.landmarks5[0])))
        if np.sqrt(np.mean((fitted.reshape(-1, 2) - points[stable]) ** 2)) > .2 * iod:
            raise DomainError("HEAD_POSE_REJECTED", "Dense head pose reprojection exceeds 20% interocular distance", 422)
        motion = self.temporal.filter(0, "pose", np.r_[rv.ravel(), tv.ravel()], timestamp)
        pose = HeadPose(cv2.Rodrigues(motion[:3])[0], motion[3:], k)
        from roop_3d.core.pose import euler_from_rotation
        reflection = np.diag([1., 1., -1.])
        yaw, pitch, roll = euler_from_rotation(reflection @ pose.rotation @ reflection)
        params = ExpressionParams()
        if animate and abs(yaw) <= 60:
            measured = measure_expression(face.landmarks5, face.landmarks106, yaw)
            self.calibrator.push(measured.vector())
            if self.calibrator.calibrated:
                vec = params_from_measure(measured, self.calibrator.baseline()).clamped().vector()
                params = ExpressionParams.from_vector(self.temporal.filter(0, "expression", vec, timestamp)).clamped()
        vertices, _ = self.rig.deform(params)
        inverse = head_roi(pose.project(vertices), self.side)
        texture, confidence = view_texture(self.rig.asset, (yaw, pitch, roll))
        rendered, coverage = self.rasterizer.render(vertices, self.rig.faces, self.rig.asset.uvs,
            texture, confidence, pose, inverse, self.side)
        if self.refiner is not None:
            target = cv2.warpAffine(image, cv2.invertAffineTransform(inverse), (self.side, self.side))
            refined = self.refiner(rendered, target, coverage, self.rig.asset)
            if refined.shape != rendered.shape or refined.dtype != np.uint8:
                raise ValueError("Full-head refiner must preserve ROI shape and uint8 BGR")
            rendered = refined
        return rendered, coverage, inverse


def view_texture(asset, angles):
    """Blend nearby pose-bank atlases, filling their unseen texels from the stitched atlas."""
    base = asset.texture.astype(np.float32)
    confidence = np.where(asset.texel_state > 0, np.clip(asset.texel_confidence / .5, 0, 1), 0).astype(np.float32)
    if not len(getattr(asset, "view_poses", ())):
        return asset.texture, confidence
    delta = (asset.view_poses - np.asarray(angles) + 180) % 360 - 180
    distance = np.linalg.norm(delta, axis=1)
    ids = np.argsort(distance)[:3]
    weights = np.exp(-(distance[ids] - distance[ids].min()) / 12.)
    colors, total = np.zeros_like(base), np.zeros(base.shape[:2], np.float32)
    for index, weight in zip(ids, weights):
        support = cv2.resize(asset.view_support[index], (base.shape[1], base.shape[0])) * weight
        colors += cv2.resize(asset.view_textures[index], (base.shape[1], base.shape[0])) * support[..., None]
        total += support
    blend = np.clip(total, 0, 1)
    result = colors / np.maximum(total[..., None], 1e-6)
    return np.clip(result * blend[..., None] + base * (1 - blend[..., None]), 0, 255).astype(np.uint8), confidence


def bridge_neural_detail(rendered, neural, coverage):
    """Multi-scale full-ROI residual; coverage remains owned by 3D geometry."""
    if neural.shape != rendered.shape:
        raise ValueError("Neural refinement must span the complete head ROI")
    sigma = max(1., rendered.shape[0] / 128.)
    detail = neural.astype(np.float32) - cv2.GaussianBlur(neural.astype(np.float32), (0, 0), sigma)
    base_detail = rendered.astype(np.float32) - cv2.GaussianBlur(rendered.astype(np.float32), (0, 0), sigma)
    # Avoid importing network border hallucinations or spreading into occluders.
    weight = np.clip(coverage, 0, 1)[..., None]
    return np.clip(rendered.astype(np.float32) + .5 * weight * (detail - base_detail), 0, 255).astype(np.uint8)
