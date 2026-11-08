/**
 * 6DoF head-mesh overlay for the media canvas: a dense wireframe of the head topology rotated by each face's measured
 * yaw / pitch / roll, a gimbal (three rings), a numeric readout and head-gaze arrows. Pure geometry + canvas drawing, no React.
 *
 * Honesty notes (shown in the UI as well):
 *  - the wireframe is registered to the face by the detected NOSE TIP and sized from the detector box, not fitted per vertex, so it
 *    shows the head's orientation and topology, not a photometric alignment;
 *  - the gaze arrows are the head's forward direction from each eye: there is no iris / eye-direction tracking in the pipeline.
 *
 * Frame: the mesh is first converted to the viewer frame (+x image-right, +y up, +z toward the viewer), the same frame
 * `poseQuaternion` uses, so a pose reads identically in the 3D inspector and on the canvas. Poses are STANDARD (pitch + = up).
 */

/** Per source coordinate frame: the sign applied to (x, y, z) to reach the viewer frame (see PoseGeometryViewer FRAME_SIGNS). */
const FRAME_SIGNS = { mesh: [1, -1, 1], opencv: [1, -1, -1], opengl: [1, 1, 1] };
/** Past this many edges the wireframe is thinned evenly: the overlay is redrawn on every frame and must stay interactive. */
export const MAX_DRAWN_EDGES = 24000;

const rad = degrees => (degrees * Math.PI) / 180;
const finite = value => typeof value === 'number' && Number.isFinite(value);

/**
 * Prepares a reconstruction ({ vertices, faces, coordinate_system }) for the overlay: centred positions in the viewer frame, a unique edge
 * list with the (up to two) triangles each edge borders (for back-face culling) and the nose-tip vertex used to register it to a face.
 * @returns {null | { positions: Float32Array, tris: Uint32Array, edges: Uint32Array, edgeStride: number, width: number, height: number, nose: [number, number, number], vertexCount: number, triCount: number }}
 */
export function prepareHeadMesh(mesh) {
  if (!mesh?.vertices?.length || !mesh?.faces?.length) return null;
  const signs = FRAME_SIGNS[mesh.coordinate_system] ?? FRAME_SIGNS.opengl;
  const mirrored = signs[0] * signs[1] * signs[2] < 0;
  const count = mesh.vertices.length;
  const triangles = mesh.faces.filter(face => face?.length === 3 && face.every(index => Number.isInteger(index) && index >= 0 && index < count));
  if (!triangles.length) return null;

  const positions = new Float32Array(count * 3);
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < count; i++) {
    const vertex = mesh.vertices[i];
    for (let axis = 0; axis < 3; axis++) {
      const value = (finite(vertex?.[axis]) ? vertex[axis] : 0) * signs[axis];
      positions[i * 3 + axis] = value;
      if (value < min[axis]) min[axis] = value;
      if (value > max[axis]) max[axis] = value;
    }
  }
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  for (let i = 0; i < count; i++) for (let axis = 0; axis < 3; axis++) positions[i * 3 + axis] -= center[axis];
  const width = max[0] - min[0], height = max[1] - min[1];
  if (!(width > 0) || !(height > 0)) return null;

  // Triangles (winding fixed when an odd number of axes were mirrored, so normals stay outward) and unique edges with their triangles.
  const tris = new Uint32Array(triangles.length * 3);
  const edgeMap = new Map();
  triangles.forEach((face, t) => {
    const [a, b, c] = mirrored ? [face[0], face[2], face[1]] : face;
    tris[t * 3] = a; tris[t * 3 + 1] = b; tris[t * 3 + 2] = c;
    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      const lo = Math.min(u, v), hi = Math.max(u, v), key = lo * count + hi;
      const entry = edgeMap.get(key);
      if (entry) entry[3] = t; else edgeMap.set(key, [lo, hi, t, -1]);
    }
  });
  const edgeStride = Math.max(1, Math.ceil(edgeMap.size / MAX_DRAWN_EDGES));
  const kept = [];
  let n = 0;
  for (const entry of edgeMap.values()) if (n++ % edgeStride === 0) kept.push(entry);
  const edges = new Int32Array(kept.length * 4);
  kept.forEach((entry, i) => { edges.set(entry, i * 4); });

  // Nose tip: the most forward vertex near the mid-line, in the middle band of the head's height.
  let nose = -1, best = -Infinity;
  for (let i = 0; i < count; i++) {
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
    if (Math.abs(x) <= 0.08 * width && y >= -0.2 * height && y <= 0.3 * height && z > best) { best = z; nose = i; }
  }
  if (nose < 0) return null;
  return { positions, tris, edges, edgeStride, width, height, vertexCount: count, triCount: triangles.length,
    nose: [positions[nose * 3], positions[nose * 3 + 1], positions[nose * 3 + 2]] };
}

/** 3x3 rotation (row-major) for a standard-frame pose, `Ry(yaw) * Rx(-pitch) * Rz(-roll)`: the viewer's `poseQuaternion` convention. */
export function poseRotation(pose) {
  const yaw = rad(pose?.yaw ?? 0), pitch = -rad(pose?.pitch ?? 0), roll = -rad(pose?.roll ?? 0);
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cx = Math.cos(pitch), sx = Math.sin(pitch), cz = Math.cos(roll), sz = Math.sin(roll);
  // Ry * Rx
  const yx = [cy, sy * sx, sy * cx, 0, cx, -sx, -sy, cy * sx, cy * cx];
  // (Ry * Rx) * Rz
  return [yx[0] * cz + yx[1] * sz, -yx[0] * sz + yx[1] * cz, yx[2],
    yx[3] * cz + yx[4] * sz, -yx[3] * sz + yx[4] * cz, yx[5],
    yx[6] * cz + yx[7] * sz, -yx[6] * sz + yx[7] * cz, yx[8]];
}
export const rotate = (m, x, y, z) => [m[0] * x + m[1] * y + m[2] * z, m[3] * x + m[4] * y + m[5] * z, m[6] * x + m[7] * y + m[8] * z];

/**
 * Where the head sits on screen for one face: pixels per mesh unit from the box width (head ~ 1.4 x the detector box), registered so the rotated
 * nose tip lands on the detected nose (or the box centre when the detector gave no landmarks).
 * @param {{ nose: number[], width: number }} head prepared mesh
 * @param {{ x: number, y: number, w: number, h: number }} box face box in pixels
 * @param {{ x: number, y: number } | null} noseScreen detected nose tip in pixels
 */
export function headPlacement(head, box, noseScreen, rotation) {
  const scale = (box.w * 1.4) / head.width;
  const [nx, ny] = rotate(rotation, head.nose[0], head.nose[1], head.nose[2]);
  const anchor = noseScreen ?? { x: box.x + box.w / 2, y: box.y + box.h * 0.58 };
  return { scale, cx: anchor.x - nx * scale, cy: anchor.y + ny * scale };
}

/** Screen position of a model-space point for a placement (y is flipped: model +y is up, screen +y is down). */
export const toScreen = (placement, rotation, x, y, z) => { const [rx, ry, rz] = rotate(rotation, x, y, z); return { x: placement.cx + rx * placement.scale, y: placement.cy - ry * placement.scale, z: rz }; };

/** The unit direction the head faces, in the viewer frame (+z = toward the viewer). */
export const forwardVector = rotation => rotate(rotation, 0, 0, 1);

/**
 * Wireframe segments of the visible (front-facing) edges as a flat [x1,y1,x2,y2,...] array in screen pixels. An edge is drawn when
 * either triangle it borders faces the camera; edges of a single-triangle boundary follow that triangle.
 */
export function wireframeSegments(head, placement, rotation) {
  const { positions, tris, edges, vertexCount } = head;
  const projected = new Float32Array(vertexCount * 3);
  for (let i = 0; i < vertexCount; i++) {
    const p = toScreen(placement, rotation, positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    projected[i * 3] = p.x; projected[i * 3 + 1] = p.y; projected[i * 3 + 2] = p.z;
  }
  const front = new Uint8Array(head.triCount);
  for (let t = 0; t < head.triCount; t++) {
    const a = tris[t * 3] * 3, b = tris[t * 3 + 1] * 3, c = tris[t * 3 + 2] * 3;
    // Screen y grows downward, so a counter-clockwise (outward) triangle in the viewer frame has a NEGATIVE signed screen area here.
    const area = (projected[b] - projected[a]) * (projected[c + 1] - projected[a + 1]) - (projected[c] - projected[a]) * (projected[b + 1] - projected[a + 1]);
    front[t] = area < 0 ? 1 : 0;
  }
  const out = [];
  for (let e = 0; e < edges.length; e += 4) {
    const t0 = edges[e + 2], t1 = edges[e + 3];
    if (!(front[t0] || (t1 >= 0 && front[t1]))) continue;
    const u = edges[e] * 3, v = edges[e + 1] * 3;
    out.push(projected[u], projected[u + 1], projected[v], projected[v + 1]);
  }
  return out;
}

/** A gimbal ring as screen points: `axis` 'yaw' (around +y), 'pitch' (around +x) or 'roll' (around +z), model radius `radius`. */
export function gimbalRing(axis, radius, placement, rotation, segments = 48) {
  const points = [];
  for (let i = 0; i <= segments; i++) {
    const angle = (i / segments) * Math.PI * 2, c = Math.cos(angle) * radius, s = Math.sin(angle) * radius;
    const [x, y, z] = axis === 'yaw' ? [c, 0, s] : axis === 'pitch' ? [0, c, s] : [c, s, 0];
    points.push(toScreen(placement, rotation, x, y, z));
  }
  return points;
}

/** Head-gaze arrows from each eye: the forward vector projected on screen, length scaled by the face box. Null without both eye landmarks. */
export function gazeArrows(eyes, rotation, box) {
  if (!eyes?.[0] || !eyes?.[1]) return null;
  const [fx, fy, fz] = forwardVector(rotation), length = box.w * 0.9;
  return eyes.map(eye => ({ from: eye, to: { x: eye.x + fx * length, y: eye.y - fy * length }, towardViewer: fz > 0 }));
}

export const AXIS_COLORS = { yaw: '#22c55e', pitch: '#ef4444', roll: '#3b82f6' };
const GAZE_COLOR = '#fbbf24';
const fixed1 = value => { const v = Math.round(value * 10) / 10; return `${v < 0 ? '-' : '+'}${Math.abs(v).toFixed(1)}`; };

/**
 * Pose filter: exponential smoothing toward `target`, or a frozen pose when locked. `state` is a mutable per-face memory.
 * @param {{ pose?: { yaw: number, pitch: number, roll: number } }} state
 * @param {{ yaw: number, pitch: number, roll: number }} target
 * @param {{ locked?: { yaw: number, pitch: number, roll: number } | null, alpha?: number }} [options]
 */
export function filterPose(state, target, { locked = null, alpha = 0.35 } = {}) {
  if (locked) { state.pose = { ...locked }; return state.pose; }
  const previous = state.pose;
  state.pose = previous && finite(previous.yaw) ? {
    yaw: previous.yaw + (target.yaw - previous.yaw) * alpha, pitch: previous.pitch + (target.pitch - previous.pitch) * alpha, roll: previous.roll + (target.roll - previous.roll) * alpha,
  } : { ...target };
  return state.pose;
}

/**
 * Draws the overlay for every face.
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array<{ id: string, key?: string, bbox: number[], landmarks5?: number[][] }>} faces
 * @param {{ head: any, mapBox: Function, mapPoint: Function, poseOf: (face: any) => ({yaw:number,pitch:number,roll:number}|null),
 *   wireframe?: boolean, gimbal?: boolean, gaze?: boolean, locked?: Record<string, any>, soloKey?: string | null, keyOf?: (face: any) => string,
 *   filters?: Map<string, any>, bounds?: { left: number, right: number } | null }} options
 * @returns {Array<{ key: string, segments: number, locked: boolean, pose: any }>} what was drawn (for tests / the readout)
 */
export function drawHeadOverlay(ctx, faces, { head, mapBox, mapPoint, poseOf, wireframe = true, gimbal = true, gaze = true, locked = {}, soloKey = null, keyOf = face => face.id, filters = new Map(), bounds = null }) {
  const drawn = [];
  for (const face of faces) {
    const key = keyOf(face), measured = poseOf(face);
    if (!measured) continue;
    const dimmed = soloKey !== null && soloKey !== key;
    const state = filters.get(key) ?? {}; filters.set(key, state);
    const pose = filterPose(state, measured, { locked: locked[key] ?? null });
    const box = mapBox(face.bbox), rotation = poseRotation(pose);
    const nose = face.landmarks5?.[2] ? mapPoint(face.landmarks5[2]) : null;
    const placement = head ? headPlacement(head, box, nose, rotation) : null;
    ctx.save();
    ctx.globalAlpha = dimmed ? 0.18 : 1;
    let segments = 0;
    if (placement && wireframe) {
      const lines = wireframeSegments(head, placement, rotation);
      segments = lines.length / 4;
      ctx.beginPath();
      for (let i = 0; i < lines.length; i += 4) { ctx.moveTo(lines[i], lines[i + 1]); ctx.lineTo(lines[i + 2], lines[i + 3]); }
      ctx.lineWidth = 0.6; ctx.strokeStyle = 'rgba(34, 211, 238, 0.55)'; ctx.stroke();
    }
    if (placement && gimbal) {
      const radius = head.width * 0.62;
      ctx.lineWidth = 1.4;
      for (const axis of ['yaw', 'pitch', 'roll']) {
        const ring = gimbalRing(axis, radius, placement, rotation);
        ctx.beginPath(); ring.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
        ctx.strokeStyle = AXIS_COLORS[axis]; ctx.globalAlpha = dimmed ? 0.12 : 0.7; ctx.stroke();
      }
      ctx.globalAlpha = dimmed ? 0.18 : 1;
    }
    if (gaze) {
      const eyes = face.landmarks5 ? [face.landmarks5[0], face.landmarks5[1]].map(point => (point ? mapPoint(point) : null)) : null;
      const arrows = gazeArrows(eyes, rotation, box);
      if (arrows) for (const arrow of arrows) {
        const dx = arrow.to.x - arrow.from.x, dy = arrow.to.y - arrow.from.y, angle = Math.atan2(dy, dx), head1 = Math.max(5, box.w * 0.07);
        ctx.beginPath(); ctx.moveTo(arrow.from.x, arrow.from.y); ctx.lineTo(arrow.to.x, arrow.to.y);
        ctx.moveTo(arrow.to.x, arrow.to.y); ctx.lineTo(arrow.to.x - head1 * Math.cos(angle - 0.45), arrow.to.y - head1 * Math.sin(angle - 0.45));
        ctx.moveTo(arrow.to.x, arrow.to.y); ctx.lineTo(arrow.to.x - head1 * Math.cos(angle + 0.45), arrow.to.y - head1 * Math.sin(angle + 0.45));
        ctx.lineWidth = 2; ctx.lineCap = 'round'; ctx.strokeStyle = GAZE_COLOR; ctx.setLineDash(arrow.towardViewer ? [] : [4, 3]); ctx.stroke(); ctx.setLineDash([]);
      }
    }
    // Gimbal readout under the box: coloured Y / P / R and the pose-lock state.
    if (gimbal) {
      const parts = [['Y', pose.yaw, AXIS_COLORS.yaw], ['P', pose.pitch, AXIS_COLORS.pitch], ['R', pose.roll, AXIS_COLORS.roll]];
      ctx.font = '600 10px ui-monospace, Menlo, Consolas, monospace';
      const label = locked[key] ? '🔒 6DoF ' : '6DoF ';
      const texts = parts.map(([name, value]) => `${name} ${fixed1(value)}°`);
      const width = ctx.measureText(label + texts.join('  ')).width + 12, top = box.y - 38 >= 0 ? box.y - 38 : box.y + box.h + 18;
      // Keep the readout inside the picture: slide it left rather than let it run off the right edge.
      const left = bounds ? Math.max(bounds.left, Math.min(box.x, bounds.right - width)) : box.x;
      ctx.fillStyle = 'rgba(8, 14, 16, 0.9)'; ctx.fillRect(left, top, width, 16);
      let x = left + 6;
      ctx.fillStyle = locked[key] ? GAZE_COLOR : '#e8f1f2'; ctx.fillText(label, x, top + 11); x += ctx.measureText(label).width;
      parts.forEach(([, , color], i) => { ctx.fillStyle = color; ctx.fillText(texts[i], x, top + 11); x += ctx.measureText(`${texts[i]}  `).width; });
    }
    ctx.restore();
    drawn.push({ key, segments, locked: !!locked[key], pose });
  }
  return drawn;
}

/* ---- canonical fallback head ---------------------------------------------------------------------------------------- */
/** A prepared head for the low-poly canonical topology when no MakeHuman identity asset is available. */
export function prepareFromBuffers(positions, indices) {
  const vertices = [];
  for (let i = 0; i < positions.length; i += 3) vertices.push([positions[i], positions[i + 1], positions[i + 2]]);
  const faces = [];
  for (let i = 0; i < indices.length; i += 3) faces.push([indices[i], indices[i + 1], indices[i + 2]]);
  return prepareHeadMesh({ vertices, faces, coordinate_system: 'opengl' });
}
