/**
 * Pure maths behind the viewport's studio relighting, physical-camera preview and cavity buffer (no React, no WebGL).
 *
 * Everything here is a PREVIEW in the browser. The CC0 lighting probes and the physical-camera matcher live in
 * `rendering/physical_camera.py`; they are not part of the render pipeline yet, so nothing below changes a render.
 */

// ------------------------------------------------------------------------------------------------ physical camera
/** Full-frame sensor: the reference the 18-135 mm range of `PhysicalCameraMatcher` is expressed against. */
export const SENSOR_WIDTH_MM = 36;
export const SENSOR_HEIGHT_MM = 24;
export const FOCAL_RANGE_MM: readonly [number, number] = [18, 135];
export const DEFAULT_FOCAL_MM = 50;

export const clampFocal = (mm: number): number => {
  const value = Number.isFinite(mm) ? mm : DEFAULT_FOCAL_MM;
  return Math.min(FOCAL_RANGE_MM[1], Math.max(FOCAL_RANGE_MM[0], value));
};
const toDeg = (radians: number): number => (radians * 180) / Math.PI;
/** Vertical field of view of a landscape full-frame sensor behind a lens of `mm`. */
export const verticalFovDeg = (mm: number): number => toDeg(2 * Math.atan(SENSOR_HEIGHT_MM / 2 / clampFocal(mm)));
export const horizontalFovDeg = (mm: number): number => toDeg(2 * Math.atan(SENSOR_WIDTH_MM / 2 / clampFocal(mm)));
/**
 * Distance multiplier that keeps the head the same size in frame when the field of view changes from `baseFovDeg` to `fovDeg`
 * (what a photographer does when changing lens: step back for a longer lens). The head then shows the lens's perspective
 * compression instead of just getting bigger or smaller.
 */
export const dollyFactor = (baseFovDeg: number, fovDeg: number): number => Math.tan((baseFovDeg * Math.PI) / 360) / Math.tan((fovDeg * Math.PI) / 360);

// ------------------------------------------------------------------------------------------------ studio lighting environments
export interface StudioEnvironment {
  id: string; name: string; kelvin: number; license: string; sourceUrl: string;
  /** Nine RGB coefficients in the server's basis order (1, x, y, z, xy, xz, yz, x2-y2, 3z2-1), irradiance-convolved. */
  sh: number[][];
  /** Equirectangular preview as a data URL (tone-mapped), when the server sent one. */
  thumbnail: string | null;
}

const finiteRow = (row: unknown): row is number[] => Array.isArray(row) && row.length === 3 && row.every(value => typeof value === 'number' && Number.isFinite(value));

/** Validates `GET /rendering/studio-environments`; entries with malformed coefficients are dropped, never half-used. */
export function parseEnvironments(raw: unknown): StudioEnvironment[] {
  const list = (raw as { environments?: unknown } | null)?.environments;
  if (!Array.isArray(list)) return [];
  return list.flatMap((item): StudioEnvironment[] => {
    const row = item as Record<string, unknown> | null;
    if (!row || typeof row.id !== 'string' || !Array.isArray(row.sh_coefficients) || row.sh_coefficients.length !== 9 || !row.sh_coefficients.every(finiteRow)) return [];
    return [{
      id: row.id, name: typeof row.name === 'string' ? row.name : row.id,
      kelvin: typeof row.kelvin === 'number' && Number.isFinite(row.kelvin) ? row.kelvin : 5500,
      license: typeof row.license === 'string' ? row.license : 'CC0-1.0', sourceUrl: typeof row.page_url === 'string' ? row.page_url : '',
      sh: row.sh_coefficients as number[][], thumbnail: typeof row.thumbnail === 'string' && row.thumbnail.startsWith('data:image/') ? row.thumbnail : null,
    }];
  });
}

/**
 * The server's basis (`rendering/physical_buffers.evaluate_sh_basis`) already carries the Lambertian convolution weights, e.g.
 * 0.886227 = pi * 0.282095. The viewer's shader multiplies plain real SH constants by its coefficients, so each coefficient is
 * re-ordered and multiplied by the weight of its band (pi, 2pi/3, pi/4; the 3z2-1 term carries 3pi/4 because the server folds
 * the factor 3 of 3z2-1 into its constant). Verified numerically in `studioTools.test.ts` against the server's formula.
 */
const VIEWER_FROM_SERVER: readonly { source: number; weight: number }[] = [
  { source: 0, weight: Math.PI },
  { source: 2, weight: (2 * Math.PI) / 3 },     // y
  { source: 3, weight: (2 * Math.PI) / 3 },     // z
  { source: 1, weight: (2 * Math.PI) / 3 },     // x
  { source: 4, weight: Math.PI / 4 },           // xy
  { source: 6, weight: Math.PI / 4 },           // yz
  { source: 8, weight: (3 * Math.PI) / 4 },     // 3z2 - 1
  { source: 5, weight: Math.PI / 4 },           // xz
  { source: 7, weight: Math.PI / 4 },           // x2 - y2
];
export function environmentToViewerSH(sh: readonly (readonly number[])[]): [number, number, number][] {
  return VIEWER_FROM_SERVER.map(({ source, weight }) => {
    const row = sh[source] ?? [0, 0, 0];
    return [row[0] * weight, row[1] * weight, row[2] * weight] as [number, number, number];
  });
}

/** The viewer's SH shader expression for one colour, as the fragment shader computes it. */
export function viewerRadiance(coefficients: readonly (readonly number[])[], normal: readonly [number, number, number]): [number, number, number] {
  const length = Math.hypot(normal[0], normal[1], normal[2]) || 1;
  const x = normal[0] / length, y = normal[1] / length, z = normal[2] / length;
  const basis = [0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)];
  const out: [number, number, number] = [0, 0, 0];
  basis.forEach((value, index) => { for (let c = 0; c < 3; c++) out[c] += value * (coefficients[index]?.[c] ?? 0); });
  return [Math.max(0, out[0]), Math.max(0, out[1]), Math.max(0, out[2])];
}

/**
 * Auto-exposure: scales the coefficients so a surface facing the camera comes out at `target` luminance. The probes are linear
 * radiance of very different brightness (a sunrise versus a softbox), so without this the five choices would not be comparable.
 */
export function exposeEnvironment(coefficients: readonly (readonly number[])[], target = 0.85): [number, number, number][] {
  const lit = viewerRadiance(coefficients, [0, 0.25, 1]);
  const luminance = 0.2126 * lit[0] + 0.7152 * lit[1] + 0.0722 * lit[2];
  const scale = luminance > 1e-6 ? target / luminance : 1;
  return coefficients.map(row => [row[0] * scale, row[1] * scale, row[2] * scale] as [number, number, number]);
}

// ------------------------------------------------------------------------------------------------ cavity
/**
 * Per-vertex cavity in [0, 1] (1 = deepest concavity). A vertex whose neighbours sit ABOVE its tangent plane is in a hollow
 * (eye socket, nostril, the crease beside the nose); one whose neighbours fall below it is on a ridge. This is a curvature
 * proxy computed from the mesh alone, not a ray-traced ambient occlusion. Positions are welded first, so a mesh with split
 * (per-corner) vertices still reads as one surface. The result is normalised by the mesh's own curvature scale, so the same
 * contrast comes out whatever the units or resolution.
 */
export function computeVertexCavity(positions: ArrayLike<number>, index: ArrayLike<number> | null, options: { smoothPasses?: number; strength?: number } = {}): Float32Array {
  const count = Math.floor(positions.length / 3);
  const out = new Float32Array(count);
  const triangles = index ? Array.from(index) : Array.from({ length: count }, (_, i) => i);
  if (count < 4 || triangles.length < 3) return out;

  // Weld coincident vertices.
  let extent = 0;
  for (let i = 0; i < positions.length; i++) extent = Math.max(extent, Math.abs(positions[i]));
  const quantum = Math.max(extent, 1e-9) * 1e-5;
  const welded = new Int32Array(count);
  const keys = new Map<string, number>();
  let unique = 0;
  const representative: number[] = [];
  for (let i = 0; i < count; i++) {
    const key = `${Math.round(positions[i * 3] / quantum)},${Math.round(positions[i * 3 + 1] / quantum)},${Math.round(positions[i * 3 + 2] / quantum)}`;
    let id = keys.get(key);
    if (id === undefined) { id = unique++; keys.set(key, id); representative.push(i); }
    welded[i] = id;
  }

  const normal = new Float64Array(unique * 3);
  const neighbours: Set<number>[] = Array.from({ length: unique }, () => new Set<number>());
  const at = (id: number, axis: number): number => positions[representative[id] * 3 + axis];
  for (let t = 0; t + 2 < triangles.length; t += 3) {
    const a = welded[triangles[t]], b = welded[triangles[t + 1]], c = welded[triangles[t + 2]];
    if (a === b || b === c || a === c) continue;
    const ux = at(b, 0) - at(a, 0), uy = at(b, 1) - at(a, 1), uz = at(b, 2) - at(a, 2);
    const vx = at(c, 0) - at(a, 0), vy = at(c, 1) - at(a, 1), vz = at(c, 2) - at(a, 2);
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;     // area-weighted
    for (const id of [a, b, c]) { normal[id * 3] += nx; normal[id * 3 + 1] += ny; normal[id * 3 + 2] += nz; }
    neighbours[a].add(b).add(c); neighbours[b].add(a).add(c); neighbours[c].add(a).add(b);
  }

  const raw = new Float64Array(unique);
  for (let id = 0; id < unique; id++) {
    const length = Math.hypot(normal[id * 3], normal[id * 3 + 1], normal[id * 3 + 2]);
    const around = neighbours[id];
    if (length < 1e-12 || !around.size) continue;
    const nx = normal[id * 3] / length, ny = normal[id * 3 + 1] / length, nz = normal[id * 3 + 2] / length;
    let above = 0, distance = 0;
    for (const other of around) {
      const dx = at(other, 0) - at(id, 0), dy = at(other, 1) - at(id, 1), dz = at(other, 2) - at(id, 2);
      above += dx * nx + dy * ny + dz * nz;                    // > 0: the neighbour is in front of the tangent plane
      distance += Math.hypot(dx, dy, dz);
    }
    raw[id] = distance > 0 ? above / distance : 0;           // dimensionless, positive in a hollow
  }

  // Smooth over the one-ring, then scale by the typical magnitude so contrast does not depend on resolution.
  let field: Float64Array = raw;
  for (let pass = 0; pass < (options.smoothPasses ?? 2); pass++) {
    const next = new Float64Array(unique);
    for (let id = 0; id < unique; id++) {
      let sum = field[id], n = 1;
      for (const other of neighbours[id]) { sum += field[other]; n += 1; }
      next[id] = sum / n;
    }
    field = next;
  }
  const hollow = Array.from(field).filter(value => value > 0).sort((a, b) => a - b);
  const scale = hollow.length ? hollow[Math.min(hollow.length - 1, Math.floor(hollow.length * 0.9))] : 0;
  const strength = options.strength ?? 1;
  for (let i = 0; i < count; i++) {
    const value = scale > 1e-12 ? field[welded[i]] / scale : 0;
    out[i] = Math.min(1, Math.max(0, value * strength));
  }
  return out;
}

export const CAVITY_VERTEX_SHADER = /* glsl */`
attribute float aCavity;
varying float vCavity;
varying vec3 vViewNormal;
void main() {
  vCavity = aCavity;
  vViewNormal = normalize(normalMatrix * normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
export const CAVITY_FRAGMENT_SHADER = /* glsl */`
varying float vCavity;
varying vec3 vViewNormal;
void main() {
  vec3 n = normalize(vViewNormal);
  if (!gl_FrontFacing) n = -n;
  float headlight = 0.78 + 0.22 * max(n.z, 0.0);
  float occlusion = 1.0 - 0.88 * vCavity;
  gl_FragColor = vec4(vec3(0.94 * headlight * occlusion), 1.0);
  #include <colorspace_fragment>
}`;
