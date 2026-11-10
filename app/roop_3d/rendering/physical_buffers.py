# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""GPU rasterization of pixel-aligned face geometry and physical appearance buffers.

The mesh z-buffer/UV pass delegates to Roop 3D's hardware ModernGL rasterizer;
this module adds a linear-light second-order SH diffuse/specular pass. SH
coefficients follow the application's BGR irradiance convention and contain
nine scalar basis coefficients per color channel.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np


_LUMA_BGR = np.array([0.114, 0.587, 0.299], np.float32)
_FRESNEL_DIELECTRIC = 0.04


def _validate_sh(sh_coefficients: np.ndarray) -> np.ndarray:
    value = np.asarray(sh_coefficients, np.float32)
    if value.shape != (9, 3) or not np.isfinite(value).all():
        raise ValueError("sh_coefficients must be finite with shape (9, 3) in BGR order")
    return value


def evaluate_sh_basis(normals: np.ndarray) -> np.ndarray:
    """Evaluate the real orthonormal 0th, 1st and 2nd order SH basis."""
    n = np.asarray(normals, np.float32)
    if n.ndim < 1 or n.shape[-1] != 3 or not np.isfinite(n).all():
        raise ValueError("normals must be finite with final dimension three")
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(length, 1e-8)
    x, y, z = np.moveaxis(n, -1, 0)
    return np.stack((
        np.full_like(x, 0.886227),
        1.023328 * x,
        1.023328 * y,
        1.023328 * z,
        0.858086 * x * y,
        0.858086 * x * z,
        0.858086 * y * z,
        0.429043 * (x * x - y * y),
        0.743125 * (3 * z * z - 1),
    ), axis=-1).astype(np.float32)


def evaluate_sh_irradiance(normals: np.ndarray, sh_coefficients: np.ndarray) -> np.ndarray:
    """Return un-clamped linear BGR irradiance from the nine SH RGB coefficients."""
    coefficients = _validate_sh(sh_coefficients)
    basis = evaluate_sh_basis(normals)
    return (basis @ coefficients).astype(np.float32)


def shade_lambertian_sh(
    albedo_linear_bgr: np.ndarray,
    normals: np.ndarray,
    sh_coefficients: np.ndarray,
    *,
    specular_strength: float = 0.0,
    shininess: float = 32.0,
    view_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Energy-bounded Lambertian + dielectric specular render in linear BGR.

    The SH field is treated as incoming diffuse irradiance. Diffuse reflection
    spends ``1 - F0`` of the energy budget; the specular lobe spends at most
    ``F0``. Thus for albedo in [0, 1] the outgoing value cannot exceed the
    incoming irradiance per channel, even when specular is enabled.
    """
    albedo = np.asarray(albedo_linear_bgr, np.float32)
    n = np.asarray(normals, np.float32)
    if albedo.shape != n.shape or albedo.ndim < 1 or albedo.shape[-1] != 3:
        raise ValueError("albedo and normals must have matching (..., 3) shapes")
    if not np.isfinite(albedo).all() or np.any(albedo < 0) or np.any(albedo > 1):
        raise ValueError("linear albedo must be finite and within [0, 1]")
    strength, exponent = float(specular_strength), float(shininess)
    if not np.isfinite(strength) or not 0 <= strength <= 1 or not np.isfinite(exponent) or exponent < 1:
        raise ValueError("specular_strength must be in [0, 1] and shininess must be >= 1")
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    irradiance = np.maximum(evaluate_sh_irradiance(n, sh_coefficients), 0.0)
    diffuse = albedo * irradiance * (1.0 - _FRESNEL_DIELECTRIC)
    if strength == 0:
        return diffuse.astype(np.float32)

    weighted = np.asarray(sh_coefficients, np.float32)[1:4] @ _LUMA_BGR
    light_direction = weighted / max(float(np.linalg.norm(weighted)), 1e-8)
    if float(np.linalg.norm(weighted)) < 1e-8:
        light_direction = np.array([0, 0, 1], np.float32)
    view = np.asarray(view_direction, np.float32)
    if view.shape != (3,) or not np.isfinite(view).all() or np.linalg.norm(view) < 1e-8:
        raise ValueError("view_direction must be a finite nonzero 3-vector")
    view /= np.linalg.norm(view)
    half_vector = light_direction + view
    half_vector /= max(float(np.linalg.norm(half_vector)), 1e-8)
    lobe = np.maximum(n @ half_vector, 0) ** exponent
    lobe *= np.maximum(n @ light_direction, 0)
    specular = np.minimum(irradiance * (_FRESNEL_DIELECTRIC * strength * lobe[..., None]),
                          irradiance * _FRESNEL_DIELECTRIC)
    return (diffuse + specular).astype(np.float32)


@dataclass(frozen=True)
class PhysicalBuffers:
    """Four pixel-aligned GPU conditioning buffers and a visibility mask."""

    normals: np.ndarray             # HxWx3 float32, camera-space unit normals
    z_depth: np.ndarray             # HxW float32 metres; +inf outside geometry
    albedo: np.ndarray              # HxWx3 float32 linear BGR reflectance
    sh_shaded: np.ndarray           # HxWx3 float32 linear BGR radiance
    coverage: np.ndarray            # HxW float32 visible geometry x UV confidence


_VERTEX_SHADER = """#version 330
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_FRAGMENT_SHADER = """#version 330
uniform sampler2D albedo_srgb;
uniform sampler2D normal_buffer;
uniform sampler2D coverage_buffer;
uniform vec3 sh_coefficients[9];
uniform vec3 key_direction;
uniform float specular_strength;
uniform float shininess;
in vec2 uv;
layout(location=0) out vec4 albedo_linear_out;
layout(location=1) out vec4 shaded_out;
vec3 decode_srgb(vec3 c) {
    return mix(c/12.92, pow((c+0.055)/1.055, vec3(2.4)), step(vec3(0.04045), c));
}
vec3 evaluate_irradiance(vec3 n) {
    float b[9] = float[9](0.886227, 1.023328*n.x, 1.023328*n.y, 1.023328*n.z,
        0.858086*n.x*n.y, 0.858086*n.x*n.z, 0.858086*n.y*n.z,
        0.429043*(n.x*n.x-n.y*n.y), 0.743125*(3.0*n.z*n.z-1.0));
    vec3 energy = vec3(0.0);
    for (int i=0; i<9; ++i) energy += sh_coefficients[i] * b[i];
    return max(energy, vec3(0.0));
}
void main() {
    float coverage = texture(coverage_buffer, uv).r;
    vec3 albedo = decode_srgb(texture(albedo_srgb, uv).rgb);
    vec3 normal = normalize(texture(normal_buffer, uv).xyz);
    if (coverage <= 0.0 || dot(normal, normal) < 0.1) {
        albedo_linear_out = vec4(0.0);
        shaded_out = vec4(0.0);
        return;
    }
    vec3 irradiance = evaluate_irradiance(normal);
    const float F0 = 0.04;
    vec3 diffuse = albedo * irradiance * (1.0 - F0);
    vec3 view_direction = vec3(0.0, 0.0, 1.0);
    vec3 half_vector = normalize(key_direction + view_direction);
    float lobe = pow(max(dot(normal, half_vector), 0.0), shininess)
               * max(dot(normal, key_direction), 0.0);
    vec3 specular = min(irradiance * (F0 * specular_strength * lobe), irradiance * F0);
    albedo_linear_out = vec4(albedo, coverage);
    shaded_out = vec4(diffuse + specular, coverage);
}
"""


class PhysicalBufferRenderer:
    """Rasterize a mesh with the hardware renderer, then shade aligned buffers on GPU."""

    def __init__(self, rasterizer=None) -> None:
        if rasterizer is None:
            from roop_3d.reconstruction.full_head import GPUHeadRasterizer
            rasterizer = GPUHeadRasterizer()
        self.rasterizer = rasterizer
        self._local = threading.local()

    def _program(self, context):
        program = getattr(self._local, "program", None)
        if program is None or getattr(self._local, "context", None) is not context:
            if program is not None:
                program.release()
            self._local.context = context
            self._local.program = context.program(vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER)
        return self._local.program

    def render(
        self,
        vertices,
        faces,
        uvs,
        albedo_uv,
        uv_confidence,
        pose,
        inverse,
        sh_coefficients,
        *,
        side: int = 512,
        specular_strength: float = 0.25,
        shininess: float = 32.0,
    ) -> PhysicalBuffers:
        """Render one frame; image arrays and depth are returned in top-left row order."""
        coefficients = _validate_sh(sh_coefficients)
        if side <= 0:
            raise ValueError("side must be positive")
        strength, exponent = float(specular_strength), float(shininess)
        if not 0 <= strength <= 1 or not np.isfinite(exponent) or exponent < 1:
            raise ValueError("invalid specular parameters")
        texture = np.asarray(albedo_uv)
        confidence = np.asarray(uv_confidence, np.float32)
        if texture.ndim != 3 or texture.shape[2] != 3 or texture.dtype != np.uint8:
            raise ValueError("albedo_uv must be a uint8 BGR UV texture")
        if confidence.shape != texture.shape[:2] or not np.isfinite(confidence).all():
            raise ValueError("uv_confidence must be finite and match the UV texture")
        confidence = np.clip(confidence, 0, 1)

        # Existing depth-tested rasterization supplies aligned albedo, depth,
        # normals, UV coverage and silhouette from the same mesh/camera pass.
        surface = self.rasterizer.render_buffers(
            vertices, faces, uvs, texture, confidence, pose, inverse, side)
        coverage = np.clip(surface.silhouette * surface.coverage, 0, 1).astype(np.float32)
        normals = np.asarray(surface.normals, np.float32).copy()
        normals[coverage <= 0] = 0
        depth = np.asarray(surface.depth, np.float32).copy()
        depth[coverage <= 0] = np.inf

        context, _ = self.rasterizer._context()
        import moderngl
        resources = []
        def keep(item):
            resources.append(item)
            return item
        context.__enter__()
        try:
            program = self._program(context)
            height, width = coverage.shape
            inputs = (
                keep(context.texture((width, height), 3, np.ascontiguousarray(surface.image).tobytes(), dtype="f1")),
                keep(context.texture((width, height), 3, np.ascontiguousarray(normals).tobytes(), dtype="f4")),
                keep(context.texture((width, height), 1, np.ascontiguousarray(coverage).tobytes(), dtype="f4")),
            )
            for unit, item in enumerate(inputs):
                item.filter = (moderngl.NEAREST, moderngl.NEAREST)
                item.repeat_x = item.repeat_y = False
                item.use(unit)
            program["albedo_srgb"].value = 0
            program["normal_buffer"].value = 1
            program["coverage_buffer"].value = 2
            program["sh_coefficients"].write(coefficients.tobytes())
            weighted = coefficients[1:4] @ _LUMA_BGR
            direction = weighted / max(float(np.linalg.norm(weighted)), 1e-8)
            if float(np.linalg.norm(weighted)) < 1e-8:
                direction = np.array([0, 0, 1], np.float32)
            program["key_direction"].value = tuple(float(v) for v in direction)
            program["specular_strength"].value = strength
            program["shininess"].value = exponent

            albedo_out = keep(context.texture((width, height), 4, dtype="f4"))
            shaded_out = keep(context.texture((width, height), 4, dtype="f4"))
            framebuffer = keep(context.framebuffer([albedo_out, shaded_out]))
            framebuffer.use()
            context.viewport = (0, 0, width, height)
            context.enable_only(0)
            framebuffer.clear(0, 0, 0, 0)
            vao = keep(context.vertex_array(program, []))
            vao.render(moderngl.TRIANGLES, vertices=3)
            albedo_linear = np.frombuffer(framebuffer.read(attachment=0, components=4, dtype="f4"), np.float32)
            shaded = np.frombuffer(framebuffer.read(attachment=1, components=4, dtype="f4"), np.float32)
            albedo_linear = albedo_linear.reshape(height, width, 4)[..., :3].copy()
            shaded = shaded.reshape(height, width, 4)[..., :3].copy()
            return PhysicalBuffers(normals, depth, albedo_linear, shaded, coverage)
        finally:
            for resource in reversed(resources):
                resource.release()
            context.__exit__(None, None, None)

    def release(self) -> None:
        program = getattr(self._local, "program", None)
        if program is not None:
            program.release()
            self._local.program = None
            self._local.context = None
        self.rasterizer.release()
