/**
 * Pure model + shader source for the WebGL2 comparison viewport (Stage 11). Everything that can be checked without a GPU
 * lives here: the GLSL strings, the colormap LUTs, CPU reference implementations of the shader maths (so the diff / bicubic
 * formulas are unit-tested against the same constants the shader is built from), and the pose-gizmo geometry.
 */
import { facePose } from './poseHud';

export const MODE_IDS = { wipe: 0, crossfade: 1, blend: 2, diff: 3, loupe: 4, result: 5, matte: 6 };
export const LOUPE_ZOOM = 3.5;
export const LOUPE_RADIUS_PX = 90;
/** Per-channel difference (0-255) below which a pixel counts as unchanged (JPEG preview noise); matches the 2D diff map. */
export const DIFF_NOISE_FLOOR = 3;
/** `raw255 * gain / DIFF_FULL_SCALE` reaches the top of the colormap; matches the 2D diff map's scale. */
export const DIFF_FULL_SCALE = 128;

/** Smooth 0 → 1 → 0 cosine used by the persistent crossfade (shared by the DOM fallback and the shader path). */
export const crossfadeValue = (elapsedMs, periodSeconds) => {
  const phase = ((elapsedMs / 1000) % periodSeconds) / periodSeconds;
  return 0.5 - 0.5 * Math.cos(2 * Math.PI * phase);
};

/* ───────────────────────────── colormaps ───────────────────────────── */

// 10-stop samples of matplotlib's perceptually uniform maps, interpolated piecewise-linearly into a 256-entry LUT texture.
export const COLORMAP_STOPS = {
  inferno: ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60', '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4'],
  viridis: ['#440154', '#482878', '#3e4a89', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],
};
export const COLORMAPS = Object.keys(COLORMAP_STOPS);
const hex = value => [1, 3, 5].map(i => parseInt(value.slice(i, i + 2), 16));

/** RGB (0-255, fractional) of colormap `name` at `t` in [0, 1]. */
export function colormapColor(name, t) {
  const stops = COLORMAP_STOPS[name] ?? COLORMAP_STOPS.inferno;
  const x = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0)) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  const a = hex(stops[i]), b = hex(stops[i + 1]);
  return a.map((channel, k) => channel + (b[k] - channel) * f);
}
/** RGBA8 256×1 lookup texture data for `name`. */
export function colormapLut(name) {
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) { const [r, g, b] = colormapColor(name, i / 255); lut.set([Math.round(r), Math.round(g), Math.round(b), 255], i * 4); }
  return lut;
}

/* ──────────────────────── CPU reference of the shader maths ──────────────────────── */

/** Heat value in [0,1] for one pixel pair (channels 0-255): `|swapped - original|`, noise-floored, scaled by `gain`. */
export function diffValue(a, b, gain) {
  const raw = (Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2])) / 3;
  return raw < DIFF_NOISE_FLOOR ? 0 : Math.min(1, (raw * gain) / DIFF_FULL_SCALE);
}

/** Catmull-Rom weights for the four taps around a fractional position `f` in [0,1) (what the 9-tap shader evaluates). */
export function catmullRomWeights(f) {
  return [f * (-0.5 + f * (1 - 0.5 * f)), 1 + f * f * (-2.5 + 1.5 * f), f * (0.5 + f * (2 - 1.5 * f)), f * f * (-0.5 + 0.5 * f)];
}

/** Maps a pointer in stage px to (uv, loupe sample uv): the lens shows the image at LOUPE_ZOOM× around the cursor. */
export function loupeSample(pointer, stage, uv, zoom = LOUPE_ZOOM) {
  const cx = pointer.x / stage.width, cy = pointer.y / stage.height;
  return [cx + (uv[0] - cx) / zoom, cy + (uv[1] - cy) / zoom];
}

/* ───────────────────────────── pose gizmo ───────────────────────────── */

const rad = degrees => (degrees * Math.PI) / 180;
/**
 * Screen-space basis of the head's local axes for a STANDARD-frame pose (yaw + = nose toward image right, pitch + = looking
 * up, roll + = clockwise tilt on screen). Returns, for the local X (ear→ear, "pitch axis"), Y (up, "yaw axis") and Z
 * (out of the nose, "roll axis"), the screen direction `[sx, sy]` (y grows downward) and `depth` (+ = toward the camera).
 */
export function poseAxes({ yaw = 0, pitch = 0, roll = 0 }) {
  const cy = Math.cos(rad(yaw)), sy = Math.sin(rad(yaw)), cp = Math.cos(rad(pitch)), sp = Math.sin(rad(pitch)), cr = Math.cos(rad(roll)), sr = Math.sin(rad(roll));
  // v' = Rz(-roll) · Rx(pitch-up) · Ry(yaw) · v, with x right, y up, z toward the camera.
  const rotate = ([x, y, z]) => {
    const x1 = cy * x + sy * z, y1 = y, z1 = -sy * x + cy * z;          // yaw about Y: +z → +x
    const x2 = x1, y2 = cp * y1 + sp * z1, z2 = -sp * y1 + cp * z1;      // pitch about X: +z → +y (looking up)
    const x3 = cr * x2 + sr * y2, y3 = -sr * x2 + cr * y2, z3 = z2;      // roll about Z: clockwise on screen
    return { sx: x3, sy: -y3, depth: z3 };
  };
  return { x: rotate([1, 0, 0]), y: rotate([0, 1, 0]), z: rotate([0, 0, 1]) };
}

/** The nose-bridge anchor in normalised media coordinates, derived from the 5-point ArcFace landmarks (eye midpoint ↔ nose tip). */
export function noseBridge(face) {
  const p = face.landmarks5;
  if (Array.isArray(p) && p.length === 5 && p.every(point => Array.isArray(point) && point.every(Number.isFinite))) {
    const eyes = [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2];
    return [(eyes[0] + p[2][0]) / 2, (eyes[1] + p[2][1]) / 2];
  }
  const [x1, y1, x2, y2] = face.bbox;
  return [(x1 + x2) / 2, y1 + (y2 - y1) * 0.48];
}

/** Gizmo inputs (anchor, size, standard pose) for every face that has a usable pose. */
export function gizmoFaces(faces) {
  const out = [];
  for (const face of faces ?? []) {
    if (!face?.bbox || !(face.pose || face.pose3d)) continue;
    const pose = facePose(face);
    if (!pose) continue;
    out.push({ id: face.id, anchor: noseBridge(face), height: Math.max(0.02, face.bbox[3] - face.bbox[1]), pose });
  }
  return out;
}

export const GIZMO_COLORS = { x: [0.94, 0.27, 0.27, 1], y: [0.13, 0.77, 0.37, 1], z: [0.23, 0.51, 0.96, 1] };

/**
 * Triangle list (x, y in clip space; r, g, b, a) for every face's three arrows, far axes first. Arrow length scales with the
 * face's bbox height, so the gizmo reads at any zoom; `thickness` is in canvas pixels.
 */
export function buildGizmoGeometry(faces, width, height, { thickness = 2.5, scale = 0.55 } = {}) {
  const data = [];
  const px = (x, y) => [(x / width) * 2 - 1, 1 - (y / height) * 2];
  const quad = (a, b, color, w) => {
    const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1, nx = (-dy / len) * (w / 2), ny = (dx / len) * (w / 2);
    const corners = [[a[0] + nx, a[1] + ny], [a[0] - nx, a[1] - ny], [b[0] + nx, b[1] + ny], [b[0] - nx, b[1] - ny]];
    for (const index of [0, 1, 2, 2, 1, 3]) data.push(...px(...corners[index]), ...color);
  };
  for (const face of faces) {
    const origin = [face.anchor[0] * width, face.anchor[1] * height], length = face.height * height * scale, axes = poseAxes(face.pose);
    const ordered = Object.entries(axes).sort((a, b) => a[1].depth - b[1].depth);       // painter's order: far → near
    for (const [name, axis] of ordered) {
      const color = GIZMO_COLORS[name];
      // Axes pointing at the camera foreshorten to a dot; keep a minimum so the arrow never disappears.
      const reach = Math.max(0.12, Math.hypot(axis.sx, axis.sy)), dir = Math.hypot(axis.sx, axis.sy) > 1e-3 ? [axis.sx / Math.hypot(axis.sx, axis.sy), axis.sy / Math.hypot(axis.sx, axis.sy)] : [0, -1];
      const tip = [origin[0] + dir[0] * length * reach, origin[1] + dir[1] * length * reach];
      quad(origin, tip, color, thickness);
      const head = Math.max(6, length * 0.16), back = [tip[0] - dir[0] * head, tip[1] - dir[1] * head], side = [-dir[1] * head * 0.55, dir[0] * head * 0.55];
      data.push(...px(...tip), ...color, ...px(back[0] + side[0], back[1] + side[1]), ...color, ...px(back[0] - side[0], back[1] - side[1]), ...color);
    }
  }
  return new Float32Array(data);
}

/* ───────────────────────────── GLSL ───────────────────────────── */

export const VERTEX_SHADER = `#version 300 es
in vec2 aPosition;
out vec2 vUv;
void main() {
  vUv = vec2(aPosition.x * 0.5 + 0.5, 0.5 - aPosition.y * 0.5);
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

export const FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uOriginal;   // A: the source frame
uniform sampler2D uSwapped;    // B: the swapped frame
uniform sampler2D uColormap;   // 256x1 inferno / viridis LUT
uniform vec2 uOriginalSize;
uniform vec2 uSwappedSize;
uniform vec2 uStage;           // stage size in device px
uniform int uMode;             // 0 wipe, 1 crossfade, 2 blend, 3 diff, 4 loupe
uniform float uSplit;          // wipe divider position, 0..1
uniform int uVertical;         // wipe orientation: 0 left/right, 1 top/bottom
uniform float uMix;            // crossfade phase / blend opacity, 0..1
uniform float uGain;           // diff gain
uniform vec2 uPointer;         // loupe centre, device px
uniform float uLoupeRadius;    // device px
uniform float uLoupeZoom;
uniform int uBicubic;          // 0 bilinear, 1 bicubic (Catmull-Rom)
uniform int uHasSwapped;

const float NOISE_FLOOR = ${DIFF_NOISE_FLOOR}.0 / 255.0;
const float FULL_SCALE = ${DIFF_FULL_SCALE}.0 / 255.0;

// Catmull-Rom in 9 bilinear taps (the 16-tap kernel folded onto the texture hardware's linear filter).
vec4 sampleCatmullRom(sampler2D tex, vec2 uv, vec2 texSize) {
  vec2 samplePos = uv * texSize;
  vec2 texPos1 = floor(samplePos - 0.5) + 0.5;
  vec2 f = samplePos - texPos1;
  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  vec2 w3 = f * f * (-0.5 + 0.5 * f);
  vec2 w12 = w1 + w2;
  vec2 offset12 = w2 / w12;
  vec2 p0 = (texPos1 - 1.0) / texSize;
  vec2 p3 = (texPos1 + 2.0) / texSize;
  vec2 p12 = (texPos1 + offset12) / texSize;
  vec4 c = vec4(0.0);
  c += texture(tex, vec2(p0.x, p0.y)) * w0.x * w0.y;
  c += texture(tex, vec2(p12.x, p0.y)) * w12.x * w0.y;
  c += texture(tex, vec2(p3.x, p0.y)) * w3.x * w0.y;
  c += texture(tex, vec2(p0.x, p12.y)) * w0.x * w12.y;
  c += texture(tex, vec2(p12.x, p12.y)) * w12.x * w12.y;
  c += texture(tex, vec2(p3.x, p12.y)) * w3.x * w12.y;
  c += texture(tex, vec2(p0.x, p3.y)) * w0.x * w3.y;
  c += texture(tex, vec2(p12.x, p3.y)) * w12.x * w3.y;
  c += texture(tex, vec2(p3.x, p3.y)) * w3.x * w3.y;
  return clamp(c, 0.0, 1.0);
}

vec4 fetchA(vec2 uv) { return uBicubic == 1 ? sampleCatmullRom(uOriginal, uv, uOriginalSize) : texture(uOriginal, uv); }
vec4 fetchB(vec2 uv) { return uHasSwapped == 1 ? (uBicubic == 1 ? sampleCatmullRom(uSwapped, uv, uSwappedSize) : texture(uSwapped, uv)) : fetchA(uv); }

void main() {
  vec2 uv = vUv;
  vec3 a = texture(uOriginal, uv).rgb;
  vec3 b = uHasSwapped == 1 ? texture(uSwapped, uv).rgb : a;
  vec3 color = b;

  if (uMode == 0) {                                   // split wipe
    float coord = uVertical == 1 ? uv.y : uv.x;
    color = coord < uSplit ? a : b;
  } else if (uMode == 1 || uMode == 2) {              // alpha crossfade / fixed blend
    color = mix(a, b, uMix);
  } else if (uMode == 3) {                            // diff heat map: |swapped - original| * gain -> colormap
    float raw = dot(abs(b - a), vec3(1.0 / 3.0));
    float t = raw < NOISE_FLOOR ? 0.0 : clamp(raw * uGain / FULL_SCALE, 0.0, 1.0);
    color = texture(uColormap, vec2(t * (255.0 / 256.0) + 0.5 / 256.0, 0.5)).rgb;
  } else if (uMode == 4) {                            // 3.5x magnifier loupe over the swapped frame
    vec2 px = uv * uStage;
    float d = distance(px, uPointer);
    if (d < uLoupeRadius) {
      vec2 centre = uPointer / uStage;
      vec2 sampleUv = centre + (uv - centre) / uLoupeZoom;
      color = fetchB(sampleUv).rgb;
      float ring = smoothstep(uLoupeRadius - 3.0, uLoupeRadius - 1.5, d);
      color = mix(color, vec3(0.0, 0.961, 0.831), ring);
    }
  }
  outColor = vec4(color, 1.0);
}`;

export const GIZMO_VERTEX_SHADER = `#version 300 es
in vec2 aPosition;
in vec4 aColor;
out vec4 vColor;
void main() { vColor = aColor; gl_Position = vec4(aPosition, 0.0, 1.0); }`;
export const GIZMO_FRAGMENT_SHADER = `#version 300 es
precision mediump float;
in vec4 vColor;
out vec4 outColor;
void main() { outColor = vColor; }`;

/** Feature test that never touches a context in environments without WebGL2 (jsdom, SSR). */
export function webgl2Available() {
  if (typeof document === 'undefined' || typeof WebGL2RenderingContext === 'undefined') return false;
  try { return !!document.createElement('canvas').getContext('webgl2'); } catch { return false; }
}
