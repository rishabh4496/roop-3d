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

    @property
    def projection_matrix(self):
        """K[R|t] in OpenCV's object frame (mesh z is reflected separately)."""
        return self.camera @ np.column_stack((self.rotation * self.scale, self.translation))

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


@dataclass(frozen=True)
class HeadRenderBuffers:
    image: np.ndarray
    coverage: np.ndarray
    silhouette: np.ndarray
    depth: np.ndarray
    uv: np.ndarray


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
                    in vec4 clip_position; in vec2 atlas_uv; in float camera_depth;
                    out vec2 uv; out float metric_depth;
                    void main() { gl_Position = clip_position; uv = atlas_uv; metric_depth = camera_depth; }
                    ''', fragment_shader='''#version 330
                    uniform sampler2D albedo; uniform sampler2D confidence;
                    in vec2 uv; in float metric_depth;
                    layout(location=0) out vec4 color;
                    layout(location=1) out vec4 geometry;
                    void main() { color = vec4(texture(albedo, uv).rgb,
                                               texture(confidence, uv).r);
                                  geometry = vec4(uv, metric_depth, 1); }
                    ''')
                program["albedo"].value = 0
                program["confidence"].value = 1
                self.local.context, self.local.program = context, program
            except Exception as error:
                raise DomainError("GPU_HEAD_RENDERER_UNAVAILABLE",
                    f"Hardware ModernGL context required: {error}", 503) from error
        return self.local.context, self.local.program

    def render(self, vertices, faces, uvs, texture, coverage, pose, inverse, side=512):
        buffers = self.render_buffers(vertices, faces, uvs, texture, coverage, pose, inverse, side)
        return buffers.image, buffers.coverage

    @property
    def last_buffers(self):
        return self.local.last_buffers

    def render_buffers(self, vertices, faces, uvs, texture, coverage, pose, inverse, side=512):
        ctx, program = self._context()
        import moderngl
        camera = pose.camera_vertices(vertices)
        pixels = pose.project(vertices)
        local = cv2.transform(pixels[None].astype(np.float32), cv2.invertAffineTransform(inverse))[0]
        z = camera[:, 2]
        near, far = .001, max(10., float(z.max()) * 2)
        perspective = getattr(pose, "perspective", True)
        weight = z if perspective else np.ones_like(z)
        depth_clip = ((far + near) / (far - near) * z - 2 * far * near / (far - near)
                      if perspective else 2 * (z - near) / (far - near) - 1)
        clip = np.column_stack(((2 * (local[:, 0] + .5) / side - 1) * weight,
                                (1 - 2 * (local[:, 1] + .5) / side) * weight, depth_clip, weight))
        # UV data is uploaded in image row order, so flip OBJ v for sampling.
        data = np.column_stack((clip, uvs[:, 0], 1 - uvs[:, 1], z)).astype("f4")
        resources = []
        try:
            def keep(value):
                resources.append(value)
                return value
            vbo = keep(ctx.buffer(data.tobytes()))
            ibo = keep(ctx.buffer(np.asarray(faces, "i4").tobytes()))
            vao = keep(ctx.vertex_array(program, [(vbo, "4f 2f 1f", "clip_position", "atlas_uv", "camera_depth")], ibo))
            tex = keep(ctx.texture((texture.shape[1], texture.shape[0]), 3,
                                   np.ascontiguousarray(texture).tobytes()))
            cov = keep(ctx.texture((coverage.shape[1], coverage.shape[0]), 1,
                                   np.asarray(coverage, "f4").tobytes(), dtype="f4"))
            for unit, item in enumerate((tex, cov)):
                item.repeat_x = item.repeat_y = False
                item.filter = (moderngl.LINEAR, moderngl.LINEAR)
                item.use(unit)
            color = keep(ctx.texture((side, side), 4, dtype="f4"))
            geometry = keep(ctx.texture((side, side), 4, dtype="f4"))
            depth = keep(ctx.depth_renderbuffer((side, side)))
            fbo = keep(ctx.framebuffer([color, geometry], depth))
            fbo.use()
            ctx.viewport = (0, 0, side, side)
            ctx.enable_only(moderngl.DEPTH_TEST)
            fbo.clear(0, 0, 0, 0, depth=1.)
            vao.render(moderngl.TRIANGLES)
            rgba = np.frombuffer(fbo.read(components=4, dtype="f4"), np.float32).reshape(side, side, 4)[::-1].copy()
            surface = np.frombuffer(fbo.read(attachment=1, components=4, dtype="f4"), np.float32).reshape(side, side, 4)[::-1].copy()
            buffers = HeadRenderBuffers(np.clip(rgba[..., :3] * 255, 0, 255).round().astype(np.uint8),
                np.clip(rgba[..., 3], 0, 1), surface[..., 3],
                np.where(surface[..., 3] > 0, surface[..., 2], np.inf), surface[..., :2])
            self.local.last_buffers = buffers
            return buffers
        finally:
            for resource in reversed(resources):
                resource.release()

    def visibility(self, points, pose, inverse, buffers=None, tolerance=.0015):
        """GPU depth-test arbitrary surface samples (vertices OR canonical UV texels).

        Hidden far-side samples stay invisible even when a visible near-side
        sample projects onto the same pixel. Nothing is erased in screen space.
        """
        import moderngl
        buffers = buffers or self.last_buffers
        points = np.asarray(points, np.float64)
        if not len(points):
            return np.empty(0, bool)
        ctx, _ = self._context()
        h, w = buffers.depth.shape
        xy = cv2.transform(pose.project(points)[None].astype(np.float32), cv2.invertAffineTransform(inverse))[0]
        query = np.column_stack((xy, pose.camera_vertices(points)[:, 2])).astype("f4")
        width = min(1024, len(query))
        height = (len(query) + width - 1) // width
        resources = []
        def keep(item):
            resources.append(item)
            return item
        try:
            program = keep(ctx.program(vertex_shader='''#version 330
                in vec3 query; uniform sampler2D surface_depth;
                uniform ivec2 output_size; uniform float tolerance;
                flat out float visible;
                void main() {
                    ivec2 size = textureSize(surface_depth, 0);
                    ivec2 p = ivec2(floor(query.xy + .5));
                    visible = 0;
                    if (all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, size)) && query.z > 0) {
                        float d = texelFetch(surface_depth, p, 0).r;
                        visible = (d > 0 && abs(query.z-d) <= tolerance) ? 1 : 0;
                    }
                    ivec2 outp = ivec2(gl_VertexID % output_size.x, gl_VertexID / output_size.x);
                    gl_Position = vec4((vec2(outp)+.5)/vec2(output_size)*2-1, 0, 1);
                    gl_PointSize = 1;
                }''', fragment_shader='''#version 330
                flat in float visible; out float result;
                void main() { result = visible; }'''))
            program["surface_depth"].value = 0
            program["output_size"].value = (width, height)
            program["tolerance"].value = tolerance
            data = keep(ctx.buffer(query.tobytes()))
            vao = keep(ctx.vertex_array(program, [(data, "3f", "query")]))
            depth = keep(ctx.texture((w, h), 1, np.where(np.isfinite(buffers.depth), buffers.depth, 0).astype("f4").tobytes(), dtype="f4"))
            depth.use(0)
            output = keep(ctx.texture((width, height), 1, dtype="f4"))
            fbo = keep(ctx.framebuffer([output]))
            fbo.use()
            ctx.viewport = (0, 0, width, height)
            ctx.enable_only(0)
            fbo.clear()
            vao.render(moderngl.POINTS, vertices=len(query))
            return np.frombuffer(fbo.read(components=1, dtype="f4"), np.float32)[:len(query)] > .5
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

    def process(self, image, face, landmarks68, timestamp, *, scene_cut=False, animate=True,
                camera_matrix=None, landmark_indices=None, landmark_confidence=None):
        points = np.asarray(landmarks68, np.float64)
        if points.ndim != 2 or len(points) < 6 or points.shape[1] not in (2, 3) or not np.isfinite(points).all():
            raise DomainError("DENSE_HEAD_LANDMARKS_REQUIRED", "Full-head synthesis requires measured dense landmarks", 422)
        if landmark_indices is None:
            if len(points) != 68:
                raise DomainError("HEAD_CORRESPONDENCE_REQUIRED", "Dense layouts other than 68 require mesh vertex correspondences", 422)
            landmark_indices = self.rig.landmark_indices
        indices = np.asarray(landmark_indices)
        if (indices.shape != (len(points),) or not np.issubdtype(indices.dtype, np.integer)
                or indices.min() < 0 or indices.max() >= len(self.rig.V0) or len(np.unique(indices)) < 6):
            raise DomainError("HEAD_CORRESPONDENCE_INVALID", "Dense landmarks need at least six distinct mesh correspondences", 422)
        points = points[:, :2]
        if scene_cut:
            self.temporal.forget_track(0)
            self.calibrator = NeutralCalibrator(120, 8)
        h, w = image.shape[:2]
        focal = max(h, w) * 1.5
        k = np.asarray(camera_matrix, np.float64) if camera_matrix is not None else np.array(
            [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], np.float64)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], (0, 0, 1))):
            raise DomainError("HEAD_CAMERA_INVALID", "Supply a finite pinhole intrinsic matrix", 422)
        model = self.rig.V0[indices] * (1, 1, -1)
        # Stable interior anatomy drives camera; contour landmarks constrain the
        # identity mesh, but the target's different jaw must not overwrite it.
        stable = np.arange(17, 48) if len(points) == 68 else np.arange(len(points))
        if landmark_confidence is not None:
            confidence = np.asarray(landmark_confidence, np.float64)
            if confidence.shape != (len(points),) or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
                raise DomainError("HEAD_CONFIDENCE_INVALID", "Landmark confidence must be finite values in 0..1", 422)
            stable = stable[confidence[stable] >= .5]
        rv, tv = solve_dense_pose(model[stable], points[stable], k)
        # Refit using anatomy that survives actual GPU depth testing. The far
        # eye/brow estimates from a profile detector cannot steer the camera.
        seed_pose = HeadPose(cv2.Rodrigues(rv)[0], tv.ravel(), k)
        seed_inverse = head_roi(seed_pose.project(self.rig.V0), 512)
        seed_buffers = self.rasterizer.render_buffers(self.rig.V0, self.rig.faces, self.rig.asset.uvs,
            np.zeros((64, 64, 3), np.uint8), np.ones((64, 64), np.float32), seed_pose, seed_inverse, 512)
        visible = self.rasterizer.visibility(self.rig.V0[indices[stable]], seed_pose, seed_inverse, seed_buffers)
        if visible.sum() >= 6:
            stable = stable[visible]
            rv, tv = solve_dense_pose(model[stable], points[stable], k)
        else:
            raise DomainError("HEAD_VISIBLE_LANDMARKS_INSUFFICIENT", "Too few depth-visible dense landmarks for head pose", 422)
        motion = self.temporal.filter(0, "pose", np.r_[rv.ravel(), tv.ravel()], timestamp)
        pose = HeadPose(cv2.Rodrigues(motion[:3])[0], motion[3:], k)
        self.last_pose = pose
        from roop_3d.core.pose import euler_from_rotation
        reflection = np.diag([1., 1., -1.])
        yaw, pitch, roll = euler_from_rotation(reflection @ pose.rotation @ reflection)
        self.last_angles = (yaw, pitch, roll)
        params = ExpressionParams()
        if animate and abs(yaw) <= 60:
            measured = measure_expression(face.landmarks5, face.landmarks106, yaw)
            self.calibrator.push(measured.vector())
            if self.calibrator.calibrated:
                vec = params_from_measure(measured, self.calibrator.baseline()).clamped().vector()
                params = ExpressionParams.from_vector(self.temporal.filter(0, "expression", vec, timestamp)).clamped()
        vertices, _ = self.rig.deform(params)
        # The ROI must include the old jaw as well as the new head, otherwise a
        # narrower source crops away the very ribbon that needs to be removed.
        target68 = points if len(points) == 68 else getattr(face, "landmarks68", None)
        target_jaw = np.asarray(target68)[:17, :2] if target68 is not None else np.empty((0, 2))
        inverse = head_roi(np.vstack((pose.project(vertices), target_jaw)), self.side)
        self.target_silhouette = None
        if len(target_jaw):
            jaw = cv2.transform(target_jaw[None].astype(np.float32), cv2.invertAffineTransform(inverse))[0]
            self.target_silhouette = np.zeros((self.side, self.side), np.float32)
            cv2.fillPoly(self.target_silhouette, [np.round(jaw).astype(np.int32)], 1.)
        texture, confidence = view_texture(self.rig.asset, (yaw, pitch, roll))
        texture = complete_diffuse_albedo(texture, confidence)
        rendered, coverage = self.rasterizer.render(vertices, self.rig.faces, self.rig.asset.uvs,
            texture, confidence, pose, inverse, self.side)
        self.source_silhouette = self.rasterizer.last_buffers.silhouette
        self.observed_fraction = float(coverage.sum() / max(self.source_silhouette.sum(), 1))
        # Source geometry owns the matte. Missing observations use a diffuse
        # color prior, never the target's original facial features or stretched eyes.
        coverage = self.source_silhouette
        if self.refiner is not None:
            target = cv2.warpAffine(image, cv2.invertAffineTransform(inverse), (self.side, self.side))
            refined = self.refiner(rendered, target, coverage, self.rig.asset)
            if refined.shape != rendered.shape or refined.dtype != np.uint8:
                raise ValueError("Full-head refiner must preserve ROI shape and uint8 BGR")
            rendered = refined
        return rendered, coverage, inverse


def solve_dense_pose(model, points, camera):
    """Robust six-DoF solve; reject uncertainty instead of falling back to 5pt."""
    if len(model) < 6 or np.linalg.matrix_rank(points - points.mean(0)) < 2:
        raise DomainError("HEAD_POSE_FAILED", "At least six non-degenerate dense correspondences are required", 422)
    span = max(float(np.ptp(points, axis=0).max()), 1.)
    try:
        ok, rv, tv, inliers = cv2.solvePnPRansac(model, points, camera, None,
            flags=cv2.SOLVEPNP_EPNP, iterationsCount=100, reprojectionError=max(2., .04 * span), confidence=.999)
    except cv2.error as error:
        raise DomainError("HEAD_POSE_FAILED", "Dense head pose solver rejected the geometry", 422) from error
    if not ok or inliers is None or len(inliers) < max(6, len(model) // 2):
        raise DomainError("HEAD_POSE_FAILED", "Dense head pose has insufficient consistent correspondences", 422)
    ids = inliers.ravel()
    try:
        rv, tv = cv2.solvePnPRefineLM(model[ids], points[ids], camera, None, rv, tv)
    except cv2.error as error:
        raise DomainError("HEAD_POSE_FAILED", "Dense head pose refinement failed", 422) from error
    fitted, _ = cv2.projectPoints(model[ids], rv, tv, camera, None)
    error = np.sqrt(np.mean((fitted.reshape(-1, 2) - points[ids]) ** 2))
    if not np.isfinite(rv).all() or not np.isfinite(tv).all() or tv[2, 0] <= 0 or error > .1 * span:
        raise DomainError("HEAD_POSE_REJECTED", "Dense head pose reprojection failed", 422)
    return rv, tv


def complete_diffuse_albedo(texture, confidence):
    """Fill missing appearance with diffuse source color, without copying features.

    This is an explicit appearance prior. It does not change asset observation
    states or claim to recover source features missing from the FaceSet.
    """
    known = confidence > .05
    if not known.any():
        raise DomainError("HEAD_TEXTURE_UNAVAILABLE", "Full-head synthesis needs observed source albedo", 422)
    if np.all(confidence >= 1):
        return texture
    sigma = max(1., texture.shape[0] / 64.)
    support = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
    low = cv2.GaussianBlur(texture.astype(np.float32) * known[..., None], (0, 0), sigma)
    low /= np.maximum(support[..., None], 1e-6)
    low[support < .01] = np.median(texture[known], axis=0)
    weight = np.clip(confidence, 0, 1)[..., None]
    return np.clip(texture * weight + low * (1 - weight), 0, 255).round().astype(np.uint8)


def view_texture(asset, angles):
    """Blend nearby pose-bank atlases, filling their unseen texels from the stitched atlas."""
    base = asset.texture.astype(np.float32)
    confidence = np.where(asset.texel_state > 0, np.clip(asset.texel_confidence / .5, 0, 1), 0).astype(np.float32)
    if not len(getattr(asset, "view_poses", ())):
        return asset.texture, confidence
    from roop_3d.reconstruction.view_texture_bank import nearest_views
    ids, weights = nearest_views(asset.view_poses, angles)
    colors, total = np.zeros_like(base), np.zeros(base.shape[:2], np.float32)
    for index, weight in zip(ids, weights):
        raw_support = np.clip(asset.view_support[index], 0, 1).astype(np.float32)
        # Resize premultiplied color, not black unobserved texels: bilinear taps
        # across a UV visibility boundary otherwise introduce dark streaks.
        support = cv2.resize(raw_support, (base.shape[1], base.shape[0])) * weight
        colors += cv2.resize(asset.view_textures[index].astype(np.float32) * raw_support[..., None],
                             (base.shape[1], base.shape[0])) * weight
        total += support
    blend = np.clip(total, 0, 1)
    result = colors / np.maximum(total[..., None], 1e-6)
    confidence = np.maximum(confidence, np.clip(total / .5, 0, 1)).astype(np.float32)
    return np.clip(result * blend[..., None] + base * (1 - blend[..., None]), 0, 255).round().astype(np.uint8), confidence


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
