"""GPU jump-flood propagation of surrounding background into a silhouette ribbon.

This is spatial edge completion, not recovery of an unseen background. All old
head pixels and foreground occluders are excluded as donors, including pixels
inside the new head; only the requested ribbon is written by the compositor.
"""
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.full_head import GPUHeadRasterizer


class GPUBackgroundPropagator:
    def __init__(self):
        self.renderer = GPUHeadRasterizer()

    def fill(self, image, excluded):
        import moderngl
        ctx, _ = self.renderer._context()
        h, w = image.shape[:2]
        excluded = np.asarray(excluded, bool)
        if excluded.shape != (h, w) or excluded.all():
            raise DomainError("BACKGROUND_DONORS_UNAVAILABLE", "Jaw removal requires surrounding background pixels", 422)
        yy, xx = np.indices((h, w))
        seeds = np.stack((xx, yy), -1).astype("f4")
        seeds[excluded] = -1
        resources = []
        def keep(item):
            resources.append(item)
            return item
        vertex = '''#version 330
            void main() {
                vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
                gl_Position = vec4(p * 2.0 - 1.0, 0, 1);
            }'''
        try:
            program = keep(ctx.program(vertex_shader=vertex, fragment_shader='''#version 330
                uniform sampler2D seeds; uniform int jump;
                out vec2 nearest;
                void main() {
                    ivec2 p = ivec2(gl_FragCoord.xy), size = textureSize(seeds, 0);
                    float best = 1e30; nearest = vec2(-1);
                    for (int y=-1; y<=1; ++y) for (int x=-1; x<=1; ++x) {
                        ivec2 q = p + ivec2(x,y) * jump;
                        if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size))) continue;
                        vec2 candidate = texelFetch(seeds, q, 0).rg;
                        if (candidate.x < 0) continue;
                        float d = dot(candidate-vec2(p), candidate-vec2(p));
                        if (d < best) { best = d; nearest = candidate; }
                    }
                }'''))
            program["seeds"].value = 0
            vao = keep(ctx.vertex_array(program, []))
            textures = [keep(ctx.texture((w, h), 2, seeds.tobytes(), dtype="f4")),
                        keep(ctx.texture((w, h), 2, dtype="f4"))]
            fbos = [keep(ctx.framebuffer([t])) for t in textures]
            for tex in textures:
                tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                tex.repeat_x = tex.repeat_y = False
            ctx.enable_only(0)
            ctx.viewport = (0, 0, w, h)
            current = 0
            step = 1 << (max(w, h) - 1).bit_length()
            while step >= 1:
                textures[current].use(0)
                fbos[1-current].use()
                program["jump"].value = step
                vao.render(moderngl.TRIANGLES, vertices=3)
                current = 1-current
                step //= 2
            color_program = keep(ctx.program(vertex_shader=vertex, fragment_shader='''#version 330
                uniform sampler2D source_image; uniform sampler2D nearest_map;
                out vec4 color;
                void main() {
                    ivec2 p = ivec2(gl_FragCoord.xy);
                    ivec2 q = ivec2(texelFetch(nearest_map, p, 0).rg);
                    color = vec4(texelFetch(source_image, q, 0).rgb, 1);
                }'''))
            color_program["source_image"].value = 0
            color_program["nearest_map"].value = 1
            color_vao = keep(ctx.vertex_array(color_program, []))
            source = keep(ctx.texture((w, h), 3, np.ascontiguousarray(image).tobytes()))
            output = keep(ctx.texture((w, h), 3))
            fbo = keep(ctx.framebuffer([output]))
            source.use(0)
            textures[current].use(1)
            fbo.use()
            color_vao.render(moderngl.TRIANGLES, vertices=3)
            return np.frombuffer(fbo.read(components=3, alignment=1), np.uint8).reshape(h, w, 3).copy()
        finally:
            for resource in reversed(resources):
                resource.release()

    def release(self):
        self.renderer.release()
