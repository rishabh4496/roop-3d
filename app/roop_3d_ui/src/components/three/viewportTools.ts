/**
 * Pure math and shaders behind the 3D viewport's buffer switcher and interaction gizmos (no React, no WebGL context).
 *
 * Frame: the viewer's own (+x image-right, +y up, +z toward the viewer), head normalised to a bounding radius of ~0.95.
 * Everything here is computed in the browser from the mesh the server already sent; nothing is fetched.
 */
import type { WorkstationParameters, FaceEditor } from '../workstation/pipeline';

// ------------------------------------------------------------------------------------------------ buffer views
export type BufferView = 'lit' | 'wireframe' | 'albedo' | 'normals' | 'depth' | 'cavity' | 'sh';

export const BUFFER_VIEWS: readonly { id: BufferView; label: string; hint: string }[] = [
  { id: 'lit', label: 'Shaded', hint: 'The default shaded head with the scene lights.' },
  { id: 'wireframe', label: 'Wireframe Mesh', hint: 'Topology only: every triangle edge of the mesh being shown.' },
  { id: 'albedo', label: 'Albedo', hint: 'The identity atlas texture, unlit. It is intrinsic-estimated by the local pipeline (not output of the TRUST network, which is not installed).' },
  { id: 'normals', label: 'Surface Normals', hint: 'Camera-space surface normals as RGB (x, y, z), computed from the mesh in the browser.' },
  { id: 'depth', label: 'Z-Depth', hint: 'Distance from the camera, near = bright, normalised to the head.' },
  { id: 'cavity', label: 'Cavity AO', hint: 'Concave areas (eye sockets, nostrils, creases) darkened from the mesh curvature, computed in the browser. It is a curvature map, not ray-traced ambient occlusion.' },
  { id: 'sh', label: 'SH Relit', hint: 'Albedo lit by a 9-coefficient (3-band) Spherical Harmonics model of the key light and ambient fill.' },
];
export const isBufferView = (value: unknown): value is BufferView => BUFFER_VIEWS.some(view => view.id === value);

// ------------------------------------------------------------------------------------------------ spherical harmonics
export const SH_COEFFS = 9;
/** Lambertian convolution weight per SH index (band 0: pi, band 1: 2pi/3, band 2: pi/4). */
const BAND_WEIGHT = [Math.PI, 2 * Math.PI / 3, 2 * Math.PI / 3, 2 * Math.PI / 3, Math.PI / 4, Math.PI / 4, Math.PI / 4, Math.PI / 4, Math.PI / 4];
export const SH_Y00 = 0.282095;

/** Real spherical-harmonic basis, bands 0-2, for a unit direction. Order: Y00, Y1-1 (y), Y10 (z), Y11 (x), Y2-2 (xy), Y2-1 (yz), Y20, Y21 (xz), Y22. */
export function shBasis(x: number, y: number, z: number): number[] {
  return [
    SH_Y00,
    0.488603 * y, 0.488603 * z, 0.488603 * x,
    1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y),
  ];
}

export type SHLight = { yaw: number; pitch: number; /** key light strength, 0..1 */ strength: number; /** ambient fill, 0..1 */ ambient: number };
export const DEFAULT_SH_LIGHT: SHLight = { yaw: 35, pitch: 25, strength: 0.8, ambient: 0.3 };

/** Unit vector toward a light at yaw (+ = image right) and pitch (+ = above); same convention as PoseGeometryViewer.lightDirection. */
export function lightVector(yawDeg: number, pitchDeg: number): [number, number, number] {
  const yaw = yawDeg * Math.PI / 180, pitch = pitchDeg * Math.PI / 180;
  return [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
}
/** Inverse of {@link lightVector}: the yaw / pitch (degrees) of a direction. */
export function lightAngles(direction: ArrayLike<number>): { yaw: number; pitch: number } {
  const length = Math.hypot(direction[0], direction[1], direction[2]) || 1;
  const x = direction[0] / length, y = direction[1] / length, z = direction[2] / length;
  return { yaw: Math.atan2(x, z) * 180 / Math.PI, pitch: Math.asin(Math.min(1, Math.max(-1, y))) * 180 / Math.PI };
}

/**
 * Irradiance-convolved SH coefficients of one directional key light plus a uniform ambient: the shader evaluates
 * `sum_i c_i * Y_i(n)` for a world normal n. A delta light of unit strength reproduces max(n.l, 0) to within the usual
 * 3-band error (1.06 at the light, 0.06 opposite it).
 */
export function shLightCoefficients(light: SHLight): number[] {
  const [lx, ly, lz] = lightVector(light.yaw, light.pitch);
  const basis = shBasis(lx, ly, lz);
  const strength = Math.min(1, Math.max(0, light.strength));
  const ambient = Math.min(1, Math.max(0, light.ambient));
  const coefficients = basis.map((value, index) => strength * BAND_WEIGHT[index] * value);
  coefficients[0] += ambient / SH_Y00;
  return coefficients;
}
/** CPU evaluation of the same expression the shader runs (clamped at 0). Used by the tests and the legend. */
export function shRadiance(coefficients: readonly number[], normal: ArrayLike<number>): number {
  const length = Math.hypot(normal[0], normal[1], normal[2]) || 1;
  const basis = shBasis(normal[0] / length, normal[1] / length, normal[2] / length);
  return Math.max(0, basis.reduce((sum, value, index) => sum + value * coefficients[index], 0));
}

export const SH_VERTEX_SHADER = /* glsl */`
varying vec3 vWorldNormal;
varying vec2 vUv;
void main() {
  vUv = uv;
  vWorldNormal = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
export const SH_FRAGMENT_SHADER = /* glsl */`
uniform vec3 uSH[9];
uniform sampler2D uMap;
uniform float uHasMap;
uniform vec3 uFlatColor;
varying vec3 vWorldNormal;
varying vec2 vUv;
void main() {
  vec3 n = normalize(vWorldNormal);
  if (!gl_FrontFacing) n = -n;
  float x = n.x, y = n.y, z = n.z;
  vec3 e = uSH[0] * 0.282095
    + uSH[1] * 0.488603 * y + uSH[2] * 0.488603 * z + uSH[3] * 0.488603 * x
    + uSH[4] * 1.092548 * x * y + uSH[5] * 1.092548 * y * z + uSH[6] * 0.315392 * (3.0 * z * z - 1.0)
    + uSH[7] * 1.092548 * x * z + uSH[8] * 0.546274 * (x * x - y * y);
  vec3 albedo = uFlatColor;
  if (uHasMap > 0.5) albedo = texture2D(uMap, vUv).rgb;
  gl_FragColor = vec4(albedo * max(e, vec3(0.0)), 1.0);
  #include <colorspace_fragment>
}`;

/** The scalar coefficients of a manual key light as the RGB triples the shader takes (white light). */
export const shLightCoefficientsRGB = (light: SHLight): [number, number, number][] => shLightCoefficients(light).map(value => [value, value, value] as [number, number, number]);

export const DEPTH_VERTEX_SHADER = /* glsl */`
varying float vViewDepth;
void main() {
  vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
  vViewDepth = -viewPosition.z;
  gl_Position = projectionMatrix * viewPosition;
}`;
export const DEPTH_FRAGMENT_SHADER = /* glsl */`
uniform float uCenter;
uniform float uRadius;
varying float vViewDepth;
void main() {
  float d = clamp(0.5 - (vViewDepth - uCenter) / (2.0 * uRadius), 0.0, 1.0);
  gl_FragColor = vec4(vec3(d), 1.0);
}`;

// ------------------------------------------------------------------------------------------------ jaw hinge (theta_jaw)
/** Largest opening the gizmo allows; `mouth_open` 1.0 corresponds to this angle. */
export const JAW_MAX_DEG = 28;
export const jawDegToMouthOpen = (deg: number): number => Math.min(1, Math.max(0, deg / JAW_MAX_DEG));
export const mouthOpenToJawDeg = (value: number): number => Math.min(1, Math.max(0, value)) * JAW_MAX_DEG;

export type JawRig = {
  /** Hinge axis point in the (y, z) plane; the axis runs along x. */
  hinge: [number, number];
  /** Rest position of the chin tip in the (y, z) plane. */
  chin: [number, number];
  /** Vertices below `full` height get the whole rotation; the weight falls to 0 at `zero`. */
  full: number; zero: number;
  /** Depth (z) band over which the weight fades in toward the front of the hinge. */
  zFadeStart: number; zFadeEnd: number;
};

const smoothstep = (edge0: number, edge1: number, value: number): number => {
  if (edge0 === edge1) return value < edge0 ? 0 : 1;
  const t = Math.min(1, Math.max(0, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
};

/**
 * A viewer-side jaw hinge estimated from the mesh extent: chin = lowest centre-line vertex, hinge a fixed fraction up and back
 * from it. This is a visualisation of the opening the render uses (`mouth_open`), not a skinned FLAME jaw: the asset carries no
 * jaw weights. Returns null for a degenerate mesh.
 */
export function buildJawRig(positions: ArrayLike<number>): JawRig | null {
  const count = Math.floor(positions.length / 3);
  if (count < 8) return null;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < count; i++) {
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
    if (!Number.isFinite(x + y + z)) return null;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const width = maxX - minX, height = maxY - minY, depth = maxZ - minZ;
  if (!(width > 0 && height > 0 && depth > 0)) return null;
  const centre = (minX + maxX) / 2;
  let chinY = Infinity, chinZ = 0;
  for (let i = 0; i < count; i++) {
    if (Math.abs(positions[i * 3] - centre) > 0.1 * width) continue;
    if (positions[i * 3 + 1] < chinY) { chinY = positions[i * 3 + 1]; chinZ = positions[i * 3 + 2]; }
  }
  if (!Number.isFinite(chinY)) return null;
  const hingeY = minY + 0.38 * height, hingeZ = minZ + 0.3 * depth;
  return {
    hinge: [hingeY, hingeZ], chin: [chinY, chinZ],
    full: minY + 0.2 * height, zero: minY + 0.34 * height,
    zFadeStart: hingeZ - 0.15 * depth, zFadeEnd: hingeZ + 0.1 * depth,
  };
}

/** Rotates the lower face of `base` about the hinge by `thetaDeg` (positive = mouth opening: chin down and back) into `out`. */
export function openJaw(base: ArrayLike<number>, out: Float32Array | number[], rig: JawRig, thetaDeg: number): void {
  const theta = thetaDeg * Math.PI / 180, cos = Math.cos(theta), sin = Math.sin(theta);
  const [hy, hz] = rig.hinge;
  for (let i = 0; i + 2 < base.length; i += 3) {
    const x = base[i], y = base[i + 1], z = base[i + 2];
    const weight = theta === 0 ? 0 : (1 - smoothstep(rig.full, rig.zero, y)) * smoothstep(rig.zFadeStart, rig.zFadeEnd, z);
    if (weight <= 0) { out[i] = x; out[i + 1] = y; out[i + 2] = z; continue; }
    const dy = y - hy, dz = z - hz;
    const ry = hy + dy * cos - dz * sin, rz = hz + dy * sin + dz * cos;
    out[i] = x; out[i + 1] = y + (ry - y) * weight; out[i + 2] = z + (rz - z) * weight;
  }
}

/** Opening angle (degrees, clamped 0..JAW_MAX_DEG) that puts the chin handle on the ray hit (y, z) in the head's centre-line plane. */
export function jawAngleFromPoint(rig: JawRig, point: readonly [number, number]): number {
  const [hy, hz] = rig.hinge;
  const rest = Math.atan2(rig.chin[1] - hz, rig.chin[0] - hy);
  const now = Math.atan2(point[1] - hz, point[0] - hy);
  let delta = now - rest;
  while (delta > Math.PI) delta -= 2 * Math.PI;
  while (delta < -Math.PI) delta += 2 * Math.PI;
  return Math.min(JAW_MAX_DEG, Math.max(0, delta * 180 / Math.PI));
}
/** Where the chin handle sits for an opening angle. */
export function chinAt(rig: JawRig, thetaDeg: number): [number, number] {
  const theta = thetaDeg * Math.PI / 180, dy = rig.chin[0] - rig.hinge[0], dz = rig.chin[1] - rig.hinge[1];
  return [rig.hinge[0] + dy * Math.cos(theta) - dz * Math.sin(theta), rig.hinge[1] + dy * Math.sin(theta) + dz * Math.cos(theta)];
}

// ------------------------------------------------------------------------------------------------ gaze target (theta_eyes)
/** Head-local distance of the gaze target plane in front of the eyes, and the half-extent that maps to gaze -1..1. */
export const GAZE_DEPTH = 1.4;
export const GAZE_HALF_EXTENT = 0.9;
const clampUnit = (value: number): number => Math.min(1, Math.max(-1, value));
/** Engine gaze (x: + = image right, y: + = up, both -1..1) to a target point in head-local space, from the eye-centre `origin`. */
export function gazeToPoint(gaze: { x: number; y: number }, origin: readonly [number, number, number]): [number, number, number] {
  return [origin[0] + clampUnit(gaze.x) * GAZE_HALF_EXTENT, origin[1] + clampUnit(gaze.y) * GAZE_HALF_EXTENT, origin[2] + GAZE_DEPTH];
}
export function pointToGaze(point: readonly [number, number, number], origin: readonly [number, number, number]): { x: number; y: number } {
  return { x: clampUnit((point[0] - origin[0]) / GAZE_HALF_EXTENT), y: clampUnit((point[1] - origin[1]) / GAZE_HALF_EXTENT) };
}

/** Eye centres for the gizmo: iBUG landmarks 36-47 when the mesh has them, otherwise a fixed fraction of the bounding box. */
export function eyeAnchors(positions: ArrayLike<number>, landmarks: ArrayLike<number> | null | undefined): { left: [number, number, number]; right: [number, number, number] } | null {
  const mean = (source: ArrayLike<number>, from: number, to: number): [number, number, number] => {
    let x = 0, y = 0, z = 0; const n = to - from;
    for (let i = from; i < to; i++) { x += source[i * 3]; y += source[i * 3 + 1]; z += source[i * 3 + 2]; }
    return [x / n, y / n, z / n];
  };
  if (landmarks && landmarks.length >= 48 * 3) {
    // iBUG: 36-41 is the subject's right eye (image left), 42-47 the subject's left eye (image right).
    return { left: mean(landmarks, 36, 42), right: mean(landmarks, 42, 48) };
  }
  const count = Math.floor(positions.length / 3);
  if (count < 8) return null;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < count; i++) {
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
  }
  const centre = (minX + maxX) / 2, width = maxX - minX, y = minY + 0.62 * (maxY - minY), z = maxZ - 0.22 * Math.max(width, maxY - minY) * 0.5;
  return { left: [centre - 0.17 * width, y, z], right: [centre + 0.17 * width, y, z] };
}

// ------------------------------------------------------------------------------------------------ wiring to the real parameters
/** Sets (or, at 0, clears) fields of one target face's `face_expression_overrides`, dropping the entry when it is empty again. */
export function withFaceEdit(parameters: WorkstationParameters, trackId: string, patch: FaceEditor): WorkstationParameters {
  const all = { ...(parameters.face_expression_overrides ?? {}) };
  const next: FaceEditor = { ...(all[trackId] ?? {}) };
  for (const [key, value] of Object.entries(patch) as [keyof FaceEditor, number][]) {
    if (!Number.isFinite(value) || value === 0) delete next[key]; else next[key] = value;
  }
  if (Object.keys(next).length) all[trackId] = next; else delete all[trackId];
  return { ...parameters, face_expression_overrides: all };
}
export const faceEditOf = (parameters: WorkstationParameters, trackId: string | null): FaceEditor => (trackId ? parameters.face_expression_overrides?.[trackId] ?? {} : {});

/** Manual key light from a gizmo drag. Relighting itself stays the user's choice (it is a render-cost toggle), so `relight` is untouched. */
export function withManualLight(parameters: WorkstationParameters, yaw: number, pitch: number): WorkstationParameters {
  return { ...parameters, lighting_mode: 'manual', manual_light_yaw: Math.round(Math.min(180, Math.max(-180, yaw))), manual_light_pitch: Math.round(Math.min(90, Math.max(-90, pitch))) };
}
/** The light the gizmo / SH view should show: the manual override when set, otherwise the viewer's neutral three-quarter key. */
export function shLightOf(parameters: WorkstationParameters | undefined): SHLight {
  if (!parameters) return DEFAULT_SH_LIGHT;
  const manual = parameters.lighting_mode === 'manual';
  return {
    yaw: manual ? parameters.manual_light_yaw ?? DEFAULT_SH_LIGHT.yaw : DEFAULT_SH_LIGHT.yaw,
    pitch: manual ? parameters.manual_light_pitch ?? DEFAULT_SH_LIGHT.pitch : DEFAULT_SH_LIGHT.pitch,
    strength: parameters.relight_strength ?? DEFAULT_SH_LIGHT.strength,
    ambient: parameters.ambient_fill ?? DEFAULT_SH_LIGHT.ambient,
  };
}

/** What the viewport needs to drive the three gizmos; built by the inspector from the real workstation parameters. */
export interface ViewportTools {
  /** Selected target track, or null: jaw and gaze edit that face's `face_expression_overrides`, so they are disabled without one. */
  trackId: string | null;
  jawDeg: number; onJaw: (deg: number) => void;
  gaze: { x: number; y: number }; onGaze: (x: number, y: number) => void;
  light: SHLight & { manual: boolean }; onLight: (yaw: number, pitch: number) => void;
  relightOn: boolean; onRelight?: (on: boolean) => void;
}
