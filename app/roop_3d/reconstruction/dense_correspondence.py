"""Depth-tested 2D -> corresponding face surface -> 2D sampling.

Rasterization produces geometry attributes only. No lit mesh image is returned.
Coordinates use OpenCV cameras (+z away), pixels at integer centres, metres.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import sys

import cv2
import numpy as np

from roop_3d.core.errors import DomainError


@dataclass(frozen=True)
class SurfaceBuffer:
    attributes: np.ndarray
    depth: np.ndarray
    valid: np.ndarray


def _size(size):
    if (not isinstance(size, (tuple, list, np.ndarray)) or len(size) != 2
            or any(isinstance(v, (bool, np.bool_)) or
            not isinstance(v, (int, np.integer)) or v <= 0 or v >= 32767 for v in size)):
        raise ValueError("Image size must contain two positive integers below 32767")
    if int(size[0]) * int(size[1]) > 16_777_216:
        raise ValueError("Image size exceeds the 16-megapixel raster budget")
    return tuple(size)


def _mesh(vertices, faces, intrinsics, attributes, size):
    size = _size(size)
    vertices = np.asarray(vertices, np.float32)
    pixels = project(vertices, intrinsics)
    faces = np.asarray(faces)
    attributes = np.asarray(attributes, np.float32)
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or faces.dtype.kind not in "iu"
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("Faces must be integer triangle indices in vertex bounds")
    if attributes.shape != vertices.shape or not np.isfinite(attributes).all():
        raise ValueError("Attributes must be finite Nx3 values matching vertices")
    return vertices, faces, attributes, size, pixels


def project(camera_vertices, intrinsics):
    points = np.asarray(camera_vertices, np.float32)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
            or not np.isfinite(points).all() or np.any(np.abs(points) > 1e10)
            or np.any(points[:, 2] <= .001)):
        raise ValueError("Face vertices must be finite and in front of the camera")
    k = np.asarray(intrinsics, np.float32)
    # Affine crop cameras may rotate, reflect or skew the top two rows. The
    # homogeneous row must retain camera z, which both rasterizers interpolate.
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or abs(np.linalg.det(k.astype(np.float64))) < 1e-12
            or not np.array_equal(k[2], [0, 0, 1])):
        raise ValueError("Invalid camera intrinsics")
    q = points.astype(np.float64) @ k.T
    if not np.isfinite(q).all() or np.any(q[:, 2] <= 1e-8):
        raise ValueError("Invalid homogeneous projection")
    pixels = q[:, :2] / q[:, 2:3]
    if not np.isfinite(pixels).all() or np.any(np.abs(pixels) > 1e7):
        raise ValueError("Projection exceeds safe raster bounds")
    return pixels.astype(np.float32)


class ReferenceSurfaceRasterizer:
    """CPU correctness reference; deliberately makes no realtime performance claim."""
    def rasterize(self, vertices, faces, intrinsics, attributes, size):
        vertices, faces, attributes, size, pixels = _mesh(vertices, faces, intrinsics, attributes, size)
        width, height = size
        depth = np.full((height, width), np.inf, np.float32)
        output = np.zeros((height, width, 3), np.float32)
        for triangle in faces:
            p = pixels[triangle]
            lo = np.maximum(np.ceil(p.min(0)).astype(int), 0)
            hi = np.minimum(np.floor(p.max(0)).astype(int), (width - 1, height - 1))
            if np.any(lo > hi):
                continue
            basis = np.column_stack((p[1] - p[0], p[2] - p[0]))
            if abs(np.linalg.det(basis)) < 1e-8:
                continue
            y, x = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
            uv = (np.stack((x, y), -1) - p[0]) @ np.linalg.inv(basis).T
            bary = np.concatenate((1 - uv.sum(-1, keepdims=True), uv), -1)
            weights = bary / vertices[triangle, 2]
            reciprocal = weights.sum(-1)
            z = np.divide(1, reciprocal, out=np.full_like(reciprocal, np.inf), where=reciprocal > 0)
            ys, xs = slice(lo[1], hi[1]+1), slice(lo[0], hi[0]+1)
            keep = (bary.min(-1) >= -1e-6) & (z < depth[ys, xs])
            value = (weights @ attributes[triangle]) * z[..., None]
            depth[ys, xs][keep] = z[keep]
            output[ys, xs][keep] = value[keep]
        return SurfaceBuffer(output, depth, np.isfinite(depth))

    def release(self):
        pass


class GPUSurfaceRasterizer:
    """Hardware OpenGL perspective interpolation and z-buffer, no color shader."""
    def __init__(self):
        self.local = threading.local()

    def _context(self):
        if getattr(self.local, "context", None) is None:
            try:
                import moderngl
                ctx = moderngl.create_standalone_context(require=330,
                    **({"backend": "egl"} if sys.platform != "win32" else {}))
                if any(s in ctx.info.get("GL_RENDERER", "").lower() for s in
                       ("llvmpipe", "softpipe", "software", "swiftshader", "gdi generic")):
                    ctx.release()
                    raise RuntimeError("Hardware OpenGL required")
                program = ctx.program(vertex_shader='''#version 330
                    in vec4 clip; in vec3 attribute_value; in float camera_z;
                    out vec3 value; out float z;
                    void main() { gl_Position=clip; value=attribute_value; z=camera_z; }
                    ''', fragment_shader='''#version 330
                    in vec3 value; in float z; out vec4 geometry;
                    void main() { geometry=vec4(value,z); }
                    ''')
                self.local.context, self.local.program = ctx, program
            except Exception as error:
                raise DomainError("DENSE_WARP_GPU_UNAVAILABLE", str(error), 503) from error
        return self.local.context, self.local.program

    def rasterize(self, vertices, faces, intrinsics, attributes, size):
        vertices, faces, attributes, size, pixels = _mesh(vertices, faces, intrinsics, attributes, size)
        import moderngl
        width, height = size
        z = vertices[:, 2]
        near, far = .001, max(10., float(z.max()) * 2)
        clip = np.column_stack(((2 * (pixels[:, 0]+.5)/width-1)*z,
            (1-2*(pixels[:, 1]+.5)/height)*z,
            (far+near)/(far-near)*z-2*far*near/(far-near), z))
        data = np.column_stack((clip, attributes, z)).astype("f4")
        ctx, program = self._context()
        resources = []
        ctx.__enter__()
        try:
            def keep(item):
                resources.append(item)
                return item
            vbo = keep(ctx.buffer(data.tobytes()))
            ibo = keep(ctx.buffer(np.asarray(faces, "i4").tobytes()))
            vao = keep(ctx.vertex_array(program, [(vbo, "4f 3f 1f", "clip", "attribute_value", "camera_z")], ibo))
            color = keep(ctx.texture(size, 4, dtype="f4"))
            depth = keep(ctx.depth_renderbuffer(size))
            fbo = keep(ctx.framebuffer([color], depth))
            fbo.use()
            ctx.viewport = (0, 0, width, height)
            ctx.enable_only(moderngl.DEPTH_TEST)
            fbo.clear(0, 0, 0, 0, depth=1)
            vao.render(moderngl.TRIANGLES)
            data = np.frombuffer(fbo.read(components=4, dtype="f4"), np.float32).reshape(height, width, 4)[::-1].copy()
            valid = data[..., 3] > 0
            return SurfaceBuffer(data[..., :3], np.where(valid, data[..., 3], np.inf), valid)
        finally:
            for resource in reversed(resources):
                resource.release()
            ctx.__exit__(None, None, None)

    def release(self):
        if getattr(self.local, "context", None) is not None:
            self.local.program.release()
            self.local.context.release()
            self.local.context = None


class ViewWorkers:
    """Runs independent per-source-view jobs on a small persistent thread pool and returns the results in job order.

    Each view's ``correspondence_field`` is whole-mesh numpy plus an OpenGL pass, and a frame needs up to four of them in
    a row on the one swap thread (the measured bottleneck of a dense-3D render).  The views do not depend on each other,
    so they overlap; the caller still accumulates them in its own fixed order, which keeps the output bit-identical to the
    sequential loop.  ``GPUSurfaceRasterizer`` keeps one GL context per thread, so every worker lazily owns one;
    ``close`` releases them on the threads that created them.  One worker (or one job) runs inline, exactly as before.
    """

    def __init__(self, workers: int, rasterizer):
        self.workers = max(1, int(workers))
        self.rasterizer = rasterizer
        self._pool = None
        self._lock = threading.Lock()

    def run(self, jobs):
        jobs = list(jobs)
        if len(jobs) <= 1 or self.workers <= 1:
            return [job() for job in jobs]
        with self._lock:
            if self._pool is None:
                from concurrent.futures import ThreadPoolExecutor
                self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix="r3d-view")
            pool = self._pool
        from concurrent.futures import wait
        futures = [pool.submit(job) for job in jobs]
        wait(futures)                       # never leave a view running while the caller unwinds or tears down
        return [future.result() for future in futures]   # the first failure in view order, like the sequential loop

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is None:
            return
        barrier = threading.Barrier(self.workers)

        def drop():
            try:
                barrier.wait(timeout=10)    # forces the tasks onto distinct threads, one per context
            except threading.BrokenBarrierError:
                pass
            self.rasterizer.release()       # releases the calling worker thread's GL context

        try:
            for future in [pool.submit(drop) for _ in range(self.workers)]:
                future.result(timeout=30)
        except Exception:
            pass
        finally:
            pool.shutdown(wait=True)


@dataclass(frozen=True)
class WarpField:
    source_xy: np.ndarray  # target pixel -> supplied source-image pixel; -1 outside support
    confidence: np.ndarray

    def sample(self, source_bgr):
        image = np.asarray(source_bgr)
        xy = np.asarray(self.source_xy, np.float32)
        confidence = np.asarray(self.confidence)
        if image.ndim not in (2, 3) or (image.ndim == 3 and not 0 < image.shape[2] <= 512):
            raise ValueError("Source image must be HxW or HxWxC")
        _size((image.shape[1], image.shape[0]))
        if (image.dtype not in (np.uint8, np.uint16, np.int16, np.float32, np.float64)
                or not np.isfinite(image).all()):
            raise ValueError("Source image must have a finite OpenCV remap-compatible dtype")
        if (xy.ndim != 3 or xy.shape[2] != 2 or confidence.shape != xy.shape[:2]
                or not np.isfinite(xy).all() or np.any(np.abs(xy) >= 32767)
                or not np.isfinite(confidence).all()):
            raise ValueError("Warp coordinates and confidence must be finite matching HxWx2/HxW arrays")
        _size((xy.shape[1], xy.shape[0]))
        return cv2.remap(image, xy, None, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


def correspondence_field(rasterizer, faces, source_vertices, source_k, target_vertices,
                         target_k, target_size, source_depth, source_size, tolerance_m=.001,
                         source_support=None):
    """Perspective-correct correspondence with source AND target visibility.

    The source camera position is interpolated at the target surface hit before
    projection; interpolating projected source pixels would be incorrect.
    """
    source_vertices, faces, _, source_size, pixels = _mesh(
        source_vertices, faces, source_k, source_vertices, source_size)
    target_vertices, _, _, target_size, _ = _mesh(
        target_vertices, faces, target_k, source_vertices, target_size)
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("Depth tolerance must be finite and nonnegative")
    width, height = source_size
    source_depth = np.asarray(source_depth, np.float32)
    if (source_depth.shape != (height, width) or np.isnan(source_depth).any()
            or np.isneginf(source_depth).any() or np.any(source_depth <= 0)):
        raise ValueError("Source depth must be positive HxW depths or +inf background")
    if source_support is not None:
        source_support = np.asarray(source_support, np.float32)
        if (source_support.shape != (height, width) or not np.isfinite(source_support).all()
                or np.any((source_support < 0) | (source_support > 1))):
            raise ValueError("Source support must be finite HxW values in [0, 1]")
    hit = rasterizer.rasterize(target_vertices, faces, target_k, source_vertices, target_size)
    q = hit.attributes @ np.asarray(source_k, np.float32).T
    z = hit.attributes[..., 2]
    xy = np.zeros(q.shape[:-1] + (2,), np.float32)
    np.divide(q[..., :2], q[..., 2:3], out=xy, where=q[..., 2:3] > .001)
    valid = hit.valid & np.isfinite(xy).all(-1) & (z > .001)
    valid &= (xy[..., 0] >= 0) & (xy[..., 0] <= width-1)
    valid &= (xy[..., 1] >= 0) & (xy[..., 1] <= height-1)

    # A second geometry-only pass carries a constant source inverse-depth plane
    # for each target triangle. At a source tap, 1/z = a*u+b*v+c. Comparing
    # every tap to the centre depth instead incorrectly rejects slanted faces.
    # Duplicating vertices prevents interpolation of planes across shared edges.
    p = np.column_stack((pixels, 1 / source_vertices[:, 2]))[faces].astype(np.float64)
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    usable = np.abs(normal[:, 2]) > 1e-12
    planes = np.zeros_like(normal)
    planes[usable, :2] = -normal[usable, :2] / normal[usable, 2:3]
    planes[usable, 2] = np.sum(normal[usable] * p[usable, 0], axis=1) / normal[usable, 2]
    plane_hit = rasterizer.rasterize(target_vertices[faces].reshape(-1, 3),
        np.arange(faces.size, dtype=np.int32).reshape(-1, 3), target_k,
        np.repeat(planes.astype(np.float32), 3, axis=0), target_size)
    valid &= plane_hit.valid
    # OpenCV INTER_LINEAR rounds coordinates to a 1/32-pixel table. Validate
    # precisely its nonzero taps, not an unrelated nearest-depth sample. Return
    # the original map so public projection coordinates retain their precision.
    safe_xy = np.where(valid[..., None], xy, 0)
    quantized = np.rint(safe_xy * 32).astype(np.int64)
    base = quantized // 32
    fraction = quantized % 32
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        active = ((fraction[..., 0] > 0) if dx else np.ones(valid.shape, bool))
        active &= ((fraction[..., 1] > 0) if dy else True)
        x, y = base[..., 0] + dx, base[..., 1] + dy
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        ix, iy = np.clip(x, 0, width-1), np.clip(y, 0, height-1)
        inverse_z = (plane_hit.attributes[..., 0] * x +
                     plane_hit.attributes[..., 1] * y + plane_hit.attributes[..., 2])
        expected = np.full(valid.shape, np.inf, np.float64)
        np.divide(1, inverse_z, out=expected, where=inverse_z > 0)
        observed = source_depth[iy, ix]
        consistent = inside & np.isfinite(observed) & np.isfinite(expected)
        # Small float32 raster/interpolation allowance, separate from caller's
        # metric tolerance. This is not a licence to cross depth discontinuities.
        residual = np.full(valid.shape, np.inf, np.float64)
        np.subtract(observed, expected, out=residual, where=consistent)
        consistent &= np.abs(residual) <= tolerance_m + 2e-6 * np.abs(expected)
        if source_support is not None:
            consistent &= source_support[iy, ix] >= .999
        valid &= ~active | consistent
    return WarpField(np.where(valid[..., None], xy, -1).astype(np.float32), valid.astype(np.float32))
