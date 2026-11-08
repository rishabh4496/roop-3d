"""Hardware-OpenGL photometric pass at the enhancer's native ROI resolution.

Diffuse ratio, colored Blinn-Phong, screen-space 3D ambient occlusion and key
light visibility run in VRAM. Calibration and readback use the existing numpy
compositor boundary. Screen-space shadows only know visible geometry; hidden
back-of-head occluders require a light-space shadow atlas in a future renderer.
"""
import threading

import cv2
import numpy as np

from roop_3d.reconstruction.photometry import fit_sh, linearize
from roop_3d.relighting_3d import SphericalHarmonicsLighting

_local = threading.local()

_VERTEX = """#version 330
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p; gl_Position = vec4(p * 2 - 1, 0, 1);
}
"""

_FRAGMENT = """#version 330
uniform sampler2D enhanced_image, normal_depth, canonical_normals, target_image, write_mask;
uniform vec3 sh_source[9], sh_target[9];
uniform vec3 key_direction, key_color;
uniform vec4 intrinsics;
uniform vec3 inverse_x, inverse_y;
uniform float strength, specular, shadow_strength;
in vec2 uv;
layout(location=0) out vec4 result;
layout(location=1) out vec4 lighting;
vec3 irradiance(vec3 n, bool source) {
    float b[9] = float[9](.886227, 1.023328*n.x, 1.023328*n.y, 1.023328*n.z,
        .858086*n.x*n.y, .858086*n.x*n.z, .858086*n.y*n.z,
        .429043*(n.x*n.x-n.y*n.y), .743125*(3*n.z*n.z-1));
    vec3 value = vec3(0);
    for (int i=0; i<9; ++i) value += b[i] * (source ? sh_source[i] : sh_target[i]);
    return max(value, vec3(.01));
}
vec3 decode(vec3 c) {
    return mix(c / 12.92, pow((c+.055)/1.055, vec3(2.4)), step(vec3(.04045), c));
}
vec3 encode(vec3 c) {
    c = max(c, vec3(0));
    return mix(c*12.92, 1.055*pow(c, vec3(1./2.4))-.055, step(vec3(.0031308), c));
}
vec3 position(vec2 q, float d) {
    vec2 pixel = q * vec2(textureSize(enhanced_image, 0)) - .5;
    vec3 p = vec3(pixel, 1);
    vec2 frame = vec2(dot(inverse_x,p), dot(inverse_y,p));
    return vec3((frame-intrinsics.zw)/intrinsics.xy*d, -d);
}
bool inside(vec2 p) { return all(greaterThanEqual(p,vec2(0))) && all(lessThan(p,vec2(1))); }
void main() {
    vec3 original = texture(enhanced_image,uv).rgb;
    vec4 nd = texture(normal_depth,uv);
    float mask = texture(write_mask,uv).r;
    if (nd.w <= 0 || dot(nd.xyz,nd.xyz)<.1 || mask<=0) {
        result=vec4(original,1); lighting=vec4(0); return;
    }
    vec3 n=normalize(nd.xyz), rest=normalize(texture(canonical_normals,uv).xyz);
    vec3 p=position(uv,nd.w);
    // Short-range horizon AO. Differences are metric camera-space positions,
    // not image brightness: nose/brow/lip concavities occlude nearby skin.
    float ao=0;
    vec2 texel=1./vec2(textureSize(normal_depth,0));
    float radius=clamp(.012*intrinsics.x/max(nd.w*abs(inverse_x.x),.0001),2.,32.);
    for (int j=0;j<8;++j) {
        float angle=6.2831853*(float(j)+.5)/8.;
        vec2 direction=vec2(cos(angle),sin(angle));
        float horizon=0;
        for (int k=1;k<=3;++k) {
            vec2 q=uv+direction*texel*radius*float(k)/3.;
            if (!inside(q)) continue;
            float d=texture(normal_depth,q).w;
            if (d<=0) continue;
            vec3 delta=position(q,d)-p;
            float distance=length(delta);
            if (distance>.0002 && distance<.025)
                horizon=max(horizon,max(dot(n,delta/distance)-.08,0.)*(1.-distance/.025));
        }
        ao+=horizon/8.;
    }
    float visibility=1.;
    // Trace towards the key light; depth increases away from the viewer, so
    // q.z > ray.z is an occluder in our +z-towards-viewer shading coordinates.
    for (int k=1;k<=12;++k) {
        vec3 ray=p+key_direction*(.002*float(k));
        vec2 frame=ray.xy/(-ray.z)*intrinsics.xy+intrinsics.zw;
        mat2 affine=mat2(inverse_x.x,inverse_y.x,inverse_x.y,inverse_y.y);
        vec2 q=(inverse(affine)*(frame-vec2(inverse_x.z,inverse_y.z))+.5)
            /vec2(textureSize(enhanced_image,0));
        if (!inside(q)) continue;
        float d=texture(normal_depth,q).w;
        if(d>0 && -d>ray.z+.001 && abs(d-nd.w)<.04) visibility=min(visibility,.35);
    }
    vec3 view=normalize(-p), h=key_direction+view;
    vec3 half_vector=h/max(length(h),.00001);
    float sheen=pow(max(dot(n,half_vector),0.),32.)*max(dot(n,key_direction),0.)*specular;
    vec3 diffuse=irradiance(n,false);
    // Hard target shadows are the residual beyond the smooth SH prediction.
    vec3 observed=decode(texture(target_image,uv).rgb);
    float residual=clamp(dot(observed,vec3(.114,.587,.299)) /
        max(dot(diffuse*.5,vec3(.114,.587,.299)),.02), .25,1.);
    float shadow=mix(1.,(1.-.65*ao)*visibility*residual,shadow_strength);
    vec3 numerator=diffuse*shadow+sheen*key_color*visibility;
    vec3 ratio=clamp(numerator/max(irradiance(rest,true),vec3(.04)),vec3(.08),vec3(6));
    vec3 relit=encode(decode(original)*ratio);
    result=vec4(mix(original,relit,clamp(strength*mask,0.,1.)),1);
    lighting=vec4(encode(numerator*.5),1);
}
"""


class PhotometricRenderer:
    def __init__(self):
        from roop_3d.reconstruction.full_head import GPUHeadRasterizer
        self.owner = GPUHeadRasterizer()
        self.context, _ = self.owner._context()
        with self.context:
            self.program = self.context.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)

    def uv_normals(self, vertices, faces, uvs, side):
        """Rasterize true rest-mesh normals to the canonical UV atlas once."""
        import moderngl
        from roop_3d.reconstruction.rbf_deform import welded_vertex_normals
        normals = welded_vertex_normals(vertices, faces)
        ctx = self.context
        resources = []
        def keep(x):
            resources.append(x)
            return x
        with ctx:
            try:
                program = keep(ctx.program(vertex_shader='''#version 330
                    in vec2 atlas_uv; in vec3 normal; out vec3 n;
                    void main(){ gl_Position=vec4(atlas_uv*2-1,0,1); n=normal; }
                    ''', fragment_shader='''#version 330
                    in vec3 n; out vec4 value;
                    void main(){value=vec4(normalize(n),1);}
                    '''))
                # Read in row order; GL bottom row represents image row zero.
                data = np.column_stack((uvs[:, 0], 1-uvs[:, 1], normals)).astype('f4')
                vbo = keep(ctx.buffer(data.tobytes()))
                ibo = keep(ctx.buffer(np.asarray(faces, 'i4').tobytes()))
                vao = keep(ctx.vertex_array(program, [(vbo, '2f 3f', 'atlas_uv', 'normal')], ibo))
                tex = keep(ctx.texture((side, side), 4, dtype='f4'))
                fbo = keep(ctx.framebuffer([tex]))
                fbo.use()
                ctx.viewport = (0, 0, side, side)
                ctx.enable_only(0)
                fbo.clear()
                vao.render(moderngl.TRIANGLES)
                return np.frombuffer(fbo.read(components=4, dtype='f4'), np.float32).reshape(side, side, 4).copy()
            finally:
                for item in reversed(resources):
                    item.release()

    def render(self, photo, image, target, alpha, sh_target, *, strength, specular, shadows):
        import moderngl
        from roop_3d.hires_handoff import scale_matrix_input
        ctx, program = self.context, self.program
        side = image.shape[0]
        if image.shape != (side, side, 3) or target.shape != image.shape or side > 2048:
            raise ValueError('Photometric pass requires matching square BGR ROIs up to 2048')
        resources = []
        def keep(x):
            resources.append(x)
            return x
        with ctx:
            try:
                normals = cv2.resize(photo.normals, (side, side))
                d = cv2.resize(np.where(np.isfinite(photo.depth), photo.depth, 0).astype('f4'),
                               (side, side), interpolation=cv2.INTER_NEAREST)
                nd = np.dstack((normals, d)).astype('f4')
                rest = cv2.resize(photo.rest_normals, (side, side)).astype('f4')
                for unit, (name, array) in enumerate((('enhanced_image', image), ('normal_depth', nd),
                    ('canonical_normals', rest), ('target_image', target), ('write_mask', np.asarray(alpha, 'f4')))):
                    components = 1 if array.ndim == 2 else array.shape[2]
                    texture = keep(ctx.texture((side, side), components, np.ascontiguousarray(array).tobytes(),
                                               dtype='f1' if array.dtype == np.uint8 else 'f4'))
                    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    texture.repeat_x = texture.repeat_y = False
                    texture.use(unit)
                    program[name].value = unit
                program['sh_source'].write(np.asarray(photo.sh_source, 'f4').tobytes())
                program['sh_target'].write(np.asarray(sh_target, 'f4').tobytes())
                key = SphericalHarmonicsLighting().extract_key_light_direction(sh_target)
                program['key_direction'].value = tuple(key)
                # Project the RGB first-order field onto the dominant key.
                # Opposing blue/red rims must not become a magenta highlight.
                tint = np.maximum(key @ sh_target[1:4], 0)
                if float(tint.max()) < 1e-6:
                    tint = np.maximum(sh_target[0], 0)
                program['key_color'].value = tuple(tint / max(float(tint.max()), 1e-6))
                inverse = scale_matrix_input(photo.inverse, side / photo.normals.shape[0])
                program['inverse_x'].value = tuple(inverse[0])
                program['inverse_y'].value = tuple(inverse[1])
                k = photo.camera
                program['intrinsics'].value = (float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))
                for name, value in (('strength', strength), ('specular', specular), ('shadow_strength', shadows)):
                    program[name].value = float(value)
                output = keep(ctx.texture((side, side), 4, dtype='f1'))
                lighting = keep(ctx.texture((side, side), 4, dtype='f1'))
                fbo = keep(ctx.framebuffer([output, lighting]))
                fbo.use()
                ctx.viewport = (0, 0, side, side)
                ctx.enable_only(0)
                vao = keep(ctx.vertex_array(program, []))
                vao.render(moderngl.TRIANGLES, vertices=3)
                def read(attachment):
                    return np.frombuffer(fbo.read(attachment=attachment, components=3, alignment=1), np.uint8).reshape(side, side, 3).copy()
                return read(0), read(1)
            finally:
                for item in reversed(resources):
                    item.release()

    def release(self):
        with self.context:
            self.program.release()
        self.owner.release()


def renderer():
    if getattr(_local, 'renderer', None) is None:
        _local.renderer = PhotometricRenderer()
    return _local.renderer


def release_renderer():
    if getattr(_local, 'renderer', None) is not None:
        _local.renderer.release()
        _local.renderer = None


def relight_head(photo, image, target, skin, alpha, options):
    side = photo.target_normals.shape[0]
    mask = cv2.resize(np.asarray(skin if skin is not None else alpha, np.float32), (side, side))
    mask *= cv2.resize(np.asarray(alpha, np.float32), (side, side)) * photo.target_support
    if photo.detail is not None:
        mask *= 1 - photo.detail.masks(side)[2]
    target_small = cv2.resize(target, (side, side), interpolation=cv2.INTER_AREA)
    sh = fit_sh(photo.target_normals, linearize(target_small), mask)
    if options.lighting_mode == 'manual':
        yaw, pitch = np.radians([options.manual_light_yaw, options.manual_light_pitch])
        direction = np.array([np.sin(yaw)*np.cos(pitch), -np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
        sh = SphericalHarmonicsLighting().create_directional_lighting(direction, ambient=.3, intensity=.7)
    result, lighting = renderer().render(photo, image, target, photo.lighting_mask(alpha), sh,
        strength=options.relight_strength, specular=options.specular_sheen, shadows=options.shadow_preservation)
    normal_vis = cv2.resize(((photo.normals[..., ::-1] + 1) * 127.5).clip(0, 255).astype(np.uint8),
                           (image.shape[0], image.shape[0]))
    return result, normal_vis, lighting
