import * as THREE from 'three';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  MAX_DRAWN_EDGES, drawHeadOverlay, filterPose, forwardVector, gazeArrows, gimbalRing, headPlacement, poseRotation, prepareHeadMesh, rotate, toScreen, wireframeSegments,
} from './components/canvas/headMesh';
import { faceFlags, faceKey, togglePoseLock, toggleSoloFace } from './components/canvas/faceFlags';
import { poseDirection, poseQuaternion } from './components/three/PoseGeometryViewer';

/** A UV sphere with a "nose" bump at +z, as a reconstruction in the glTF / viewer frame. */
function sphereMesh(rings = 12, sectors = 16, frame: 'opengl' | 'mesh' | 'opencv' = 'opengl') {
  const vertices: number[][] = [];
  for (let r = 0; r <= rings; r++) {
    const phi = (r / rings) * Math.PI;
    for (let s = 0; s < sectors; s++) {
      const theta = (s / sectors) * Math.PI * 2;
      const radius = 1, x = Math.sin(phi) * Math.sin(theta) * radius, y = Math.cos(phi) * radius, z = Math.sin(phi) * Math.cos(theta) * radius;
      vertices.push([x, y, z]);
    }
  }
  vertices.push([0, 0.0, 1.35]);                                    // nose tip: the most forward vertex, on the mid-line
  const nose = vertices.length - 1;
  const faces: number[][] = [];
  for (let r = 0; r < rings; r++) for (let s = 0; s < sectors; s++) {
    const a = r * sectors + s, b = r * sectors + ((s + 1) % sectors), c = (r + 1) * sectors + s, d = (r + 1) * sectors + ((s + 1) % sectors);
    faces.push([a, c, b], [b, c, d]);                               // counter-clockwise seen from outside
  }
  faces.push([0, 1, nose]);                                         // degenerate-ish tri keeping the nose vertex referenced
  const flip = frame === 'mesh' ? [1, -1, 1] : frame === 'opencv' ? [1, -1, -1] : [1, 1, 1];
  return { vertices: vertices.map(v => v.map((value, i) => value * flip[i])), faces, coordinate_system: frame };
}

describe('prepareHeadMesh', () => {
  it('centres the mesh in the viewer frame, finds the nose tip and keeps unique edges with their triangles', () => {
    const head = prepareHeadMesh(sphereMesh())!;
    expect(head).toBeTruthy();
    expect(head.nose[2]).toBeGreaterThan(0.5);                      // the forward-most mid-line vertex
    expect(Math.abs(head.nose[0])).toBeLessThan(0.01);
    const seen = new Set<string>();
    for (let e = 0; e < head.edges.length; e += 4) {
      const key = `${head.edges[e]}:${head.edges[e + 1]}`;
      expect(seen.has(key)).toBe(false);                           // every edge once
      seen.add(key);
      expect(head.edges[e]).toBeLessThan(head.edges[e + 1]);
      expect(head.edges[e + 2]).toBeGreaterThanOrEqual(0);
    }
    expect(seen.size).toBeGreaterThan(head.triCount);
  });

  it('converts the backend mesh and OpenCV frames to the same upright, forward-facing head', () => {
    const opengl = prepareHeadMesh(sphereMesh(12, 16, 'opengl'))!, mesh = prepareHeadMesh(sphereMesh(12, 16, 'mesh'))!, opencv = prepareHeadMesh(sphereMesh(12, 16, 'opencv'))!;
    for (const head of [mesh, opencv]) {
      expect(head.nose[2]).toBeCloseTo(opengl.nose[2], 4);
      expect(head.nose[1]).toBeCloseTo(opengl.nose[1], 4);
      expect(head.height).toBeCloseTo(opengl.height, 4);
    }
    // After the mirrored frame's winding fix the front-facing triangle count matches the glTF head at the identity pose.
    const placement = { cx: 0, cy: 0, scale: 100 }, rotation = poseRotation({ yaw: 0, pitch: 0, roll: 0 });
    const counts = [opengl, mesh, opencv].map(head => wireframeSegments(head, placement, rotation).length);
    expect(counts[1]).toBe(counts[0]); expect(counts[2]).toBe(counts[0]);
  });

  it('rejects empty or degenerate meshes and thins very dense ones evenly', () => {
    expect(prepareHeadMesh(null)).toBeNull();
    expect(prepareHeadMesh({ vertices: [[0, 0, 0]], faces: [] })).toBeNull();
    expect(prepareHeadMesh({ vertices: [[0, 0, 0], [0, 0, 0], [0, 0, 0]], faces: [[0, 1, 2]] })).toBeNull();     // zero-size
    expect(prepareHeadMesh({ vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces: [[0, 1, 9]] })).toBeNull();      // index out of range
    const dense = prepareHeadMesh(sphereMesh(200, 220))!;
    expect(dense.edges.length / 4).toBeLessThanOrEqual(MAX_DRAWN_EDGES);
    expect(dense.edgeStride).toBeGreaterThan(1);
  });
});

describe('pose conventions', () => {
  it('rotates exactly like the 3D inspector (poseQuaternion) for yaw, pitch and roll, so a pose reads the same on the canvas', () => {
    for (const pose of [{ yaw: 0, pitch: 0, roll: 0 }, { yaw: 40, pitch: 0, roll: 0 }, { yaw: 0, pitch: 25, roll: 0 }, { yaw: 0, pitch: 0, roll: 30 }, { yaw: -55, pitch: 18, roll: -12 }]) {
      const q = poseQuaternion(pose), m = poseRotation(pose);
      for (const v of [[1, 0.3, 0.2], [0, 0, 1], [0.2, 1, -0.4]]) {
        const expected = new THREE.Vector3(...(v as [number, number, number])).applyQuaternion(q), got = rotate(m, v[0], v[1], v[2]);
        expect(got[0]).toBeCloseTo(expected.x, 6); expect(got[1]).toBeCloseTo(expected.y, 6); expect(got[2]).toBeCloseTo(expected.z, 6);
      }
    }
  });

  it('points the head forward along the orientation-sphere direction: +yaw turns to image right, +pitch looks up', () => {
    for (const [yaw, pitch] of [[0, 0], [60, 0], [-30, 20], [0, -40]]) {
      const forward = forwardVector(poseRotation({ yaw, pitch, roll: 0 })), expected = poseDirection(yaw, pitch);
      expect(forward[0]).toBeCloseTo(expected[0], 6); expect(forward[1]).toBeCloseTo(expected[1], 6); expect(forward[2]).toBeCloseTo(expected[2], 6);
    }
  });
});

describe('placement and wireframe', () => {
  const box = { x: 300, y: 200, w: 160, h: 200 };

  it('registers the rotated nose tip on the detected nose at any pose, and sizes the head from the box', () => {
    const head = prepareHeadMesh(sphereMesh())!, nose = { x: 372, y: 310 };
    for (const pose of [{ yaw: 0, pitch: 0, roll: 0 }, { yaw: 50, pitch: -10, roll: 8 }]) {
      const rotation = poseRotation(pose), placement = headPlacement(head, box, nose, rotation);
      const screen = toScreen(placement, rotation, head.nose[0], head.nose[1], head.nose[2]);
      expect(screen.x).toBeCloseTo(nose.x, 4); expect(screen.y).toBeCloseTo(nose.y, 4);
      expect(placement.scale * head.width).toBeCloseTo(box.w * 1.4, 4);
    }
    const noLandmarks = headPlacement(head, box, null, poseRotation({ yaw: 0, pitch: 0, roll: 0 }));
    expect(Number.isFinite(noLandmarks.cx) && Number.isFinite(noLandmarks.cy)).toBe(true);
  });

  it('draws only the front-facing side: a camera-facing triangle shows, the same triangle turned away does not', () => {
    const triangle = prepareHeadMesh({ vertices: [[-1, -1, 0], [1, -1, 0], [0, 1, 0.1], [0, 0, 1]], faces: [[0, 1, 2], [0, 1, 3]], coordinate_system: 'opengl' });
    // Counter-clockwise seen from +z (toward the viewer), so it faces the camera at the identity pose.
    const facing = prepareHeadMesh({ vertices: [[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, 0, 0.5]], faces: [[0, 1, 2]], coordinate_system: 'opengl' })!;
    expect(triangle).toBeTruthy();
    const identity = poseRotation({ yaw: 0, pitch: 0, roll: 0 }), away = poseRotation({ yaw: 180, pitch: 0, roll: 0 }), placement = { cx: 0, cy: 0, scale: 50 };
    expect(wireframeSegments(facing, placement, identity).length / 4).toBe(3);
    expect(wireframeSegments(facing, placement, away).length).toBe(0);
  });

  it('shows about half of a closed head at any orientation (back faces culled)', () => {
    const head = prepareHeadMesh(sphereMesh(16, 24))!, placement = { cx: 0, cy: 0, scale: 100 };
    const total = head.edges.length / 4;
    for (const yaw of [0, 70, 150]) {
      const shown = wireframeSegments(head, placement, poseRotation({ yaw, pitch: 10, roll: 0 })).length / 4;
      expect(shown).toBeGreaterThan(total * 0.35); expect(shown).toBeLessThan(total * 0.75);
    }
  });
});

describe('gimbal, gaze and the pose filter', () => {
  it('draws closed rings around the three axes at the head radius', () => {
    const rotation = poseRotation({ yaw: 0, pitch: 0, roll: 0 }), placement = { cx: 100, cy: 100, scale: 10 };
    const yawRing = gimbalRing('yaw', 2, placement, rotation, 8), rollRing = gimbalRing('roll', 2, placement, rotation, 8);
    expect(yawRing).toHaveLength(9);
    expect(yawRing[0].x).toBeCloseTo(yawRing[8].x); expect(yawRing[0].y).toBeCloseTo(yawRing[8].y);
    expect(Math.max(...yawRing.map(p => p.y)) - Math.min(...yawRing.map(p => p.y))).toBeCloseTo(0, 6);           // a yaw ring is edge-on at identity
    expect(Math.max(...rollRing.map(p => p.y)) - Math.min(...rollRing.map(p => p.y))).toBeCloseTo(40, 4);        // a roll ring faces the camera
  });

  it('points the gaze arrows along the head direction in screen space, and is absent without both eyes', () => {
    const eyes = [{ x: 100, y: 100 }, { x: 160, y: 100 }], box = { x: 80, y: 60, w: 100, h: 120 };
    const straight = gazeArrows(eyes, poseRotation({ yaw: 0, pitch: 0, roll: 0 }), box)!;
    expect(straight[0].to.x).toBeCloseTo(100); expect(straight[0].to.y).toBeCloseTo(100); expect(straight[0].towardViewer).toBe(true);
    const right = gazeArrows(eyes, poseRotation({ yaw: 45, pitch: 0, roll: 0 }), box)!;
    expect(right[0].to.x).toBeGreaterThan(100 + 50); expect(right[0].to.y).toBeCloseTo(100);
    const up = gazeArrows(eyes, poseRotation({ yaw: 0, pitch: 30, roll: 0 }), box)!;
    expect(up[0].to.y).toBeLessThan(100);                                                                        // looking up moves the arrow up the screen
    expect(gazeArrows([eyes[0], undefined as never], poseRotation({ yaw: 0, pitch: 0, roll: 0 }), box)).toBeNull();
    expect(gazeArrows(null, poseRotation({ yaw: 0, pitch: 0, roll: 0 }), box)).toBeNull();
  });

  it('smooths toward the measured pose and freezes at the locked pose', () => {
    const state: { pose?: { yaw: number; pitch: number; roll: number } } = {};
    expect(filterPose(state, { yaw: 10, pitch: 0, roll: 0 }).yaw).toBe(10);                                      // first sample passes through
    expect(filterPose(state, { yaw: 20, pitch: 0, roll: 0 }, { alpha: 0.5 }).yaw).toBeCloseTo(15);
    const locked = { yaw: 3, pitch: -4, roll: 1 };
    expect(filterPose(state, { yaw: 90, pitch: 90, roll: 90 }, { locked })).toEqual(locked);
    expect(filterPose(state, { yaw: 50, pitch: 0, roll: 0 }, { locked })).toEqual(locked);                       // still frozen on later frames
  });
});

/** A canvas context that records what is drawn. */
function recordingContext() {
  const calls: Record<string, number> = {};
  const texts: string[] = [];
  const rects: number[][] = [];
  const count = (name: string) => () => { calls[name] = (calls[name] ?? 0) + 1; };
  const ctx = new Proxy({} as Record<string, unknown>, {
    get: (target, prop: string) => {
      if (prop === 'measureText') return (text: string) => ({ width: text.length * 6 });
      if (prop === 'fillText') return (text: string) => { texts.push(text); };
      if (prop === 'fillRect') return (...args: number[]) => { rects.push(args); };
      if (prop in target) return target[prop];
      return count(prop);
    },
    set: (target, prop: string, value) => { target[prop] = value; return true; },
  });
  return { ctx: ctx as unknown as CanvasRenderingContext2D, calls, texts, rects };
}

describe('drawHeadOverlay', () => {
  beforeEach(() => faceFlags.reset());
  const face = (id: string, extra = {}) => ({ id, clusterId: id, bbox: [0.3, 0.2, 0.5, 0.6], landmarks5: [[0.34, 0.32], [0.46, 0.32], [0.4, 0.4], [0.35, 0.5], [0.45, 0.5]], pose: { yaw: 20, pitch: -5, roll: 2 }, ...extra });
  const options = (head: ReturnType<typeof prepareHeadMesh>, more = {}) => ({
    head, mapBox: (b: number[]) => ({ x: b[0] * 1000, y: b[1] * 1000, w: (b[2] - b[0]) * 1000, h: (b[3] - b[1]) * 1000 }),
    mapPoint: (p: number[]) => ({ x: p[0] * 1000, y: p[1] * 1000 }), poseOf: (f: { pose: { yaw: number; pitch: number; roll: number } | null }) => (f.pose ? { yaw: f.pose.yaw, pitch: -f.pose.pitch, roll: f.pose.roll } : null),
    keyOf: faceKey, ...more,
  });

  it('draws the wireframe, rings, gaze arrows and a 6DoF readout for each face', () => {
    const { ctx, calls, texts } = recordingContext();
    const drawn = drawHeadOverlay(ctx, [face('a'), face('b')] as never, options(prepareHeadMesh(sphereMesh())) as never);
    expect(drawn.map(item => item.key)).toEqual(['a', 'b']);
    expect(drawn.every(item => item.segments > 20)).toBe(true);
    expect((calls.stroke ?? 0)).toBeGreaterThanOrEqual(2 * (1 + 3 + 2));                       // wireframe + 3 rings + 2 gaze arrows per face
    expect(texts.some(text => text.includes('Y +20.0°'))).toBe(true);
    expect(texts.some(text => text.includes('P +5.0°'))).toBe(true);                          // stored pitch -5 (legacy) is +5 standard
  });

  it('draws gaze only (no mesh) when the head topology is missing, and nothing for a face without a pose', () => {
    const { ctx, calls } = recordingContext();
    const drawn = drawHeadOverlay(ctx, [face('a'), face('b', { pose: null })] as never, options(null, { wireframe: false, gimbal: false }) as never);
    expect(drawn.map(item => item.key)).toEqual(['a']);
    expect(drawn[0].segments).toBe(0);
    expect(calls.stroke).toBe(2);                                                              // two gaze arrows
  });

  it('holds a locked face at its locked pose and shows the lock; solo dims the other faces', () => {
    const { ctx, texts } = recordingContext();
    togglePoseLock('a', { yaw: -33, pitch: 4, roll: 0 });
    toggleSoloFace('b');
    const drawn = drawHeadOverlay(ctx, [face('a'), face('b')] as never, options(prepareHeadMesh(sphereMesh()), { locked: faceFlags.get().locks, soloKey: faceFlags.get().soloKey }) as never);
    expect(drawn[0]).toMatchObject({ key: 'a', locked: true, pose: { yaw: -33, pitch: 4, roll: 0 } });
    expect(drawn[1]).toMatchObject({ key: 'b', locked: false });
    expect(texts.some(text => text.startsWith('🔒'))).toBe(true);
  });
});

describe('per-face canvas flags', () => {
  beforeEach(() => faceFlags.reset());

  it('keys a face by its track, or by its left-to-right rank while it is only a candidate', () => {
    expect(faceKey({ clusterId: 'track_0002', rank: 5 })).toBe('track_0002');
    expect(faceKey({ clusterId: null, rank: 1 })).toBe('rank:1');
    expect(faceKey({ faceRank: 3 })).toBe('rank:3');
    expect(faceKey(null)).toBe('rank:0');
  });

  it('toggles the pose lock and the solo face', () => {
    togglePoseLock('track_0001', { yaw: 1, pitch: 2, roll: 3 });
    expect(faceFlags.get().locks.track_0001).toEqual({ yaw: 1, pitch: 2, roll: 3 });
    togglePoseLock('track_0001', { yaw: 9, pitch: 9, roll: 9 });
    expect(faceFlags.get().locks).toEqual({});
    togglePoseLock('track_0001', null);                          // nothing to lock without a pose
    expect(faceFlags.get().locks).toEqual({});
    toggleSoloFace('track_0001'); expect(faceFlags.get().soloKey).toBe('track_0001');
    toggleSoloFace('track_0001'); expect(faceFlags.get().soloKey).toBeNull();
  });
});

describe('readout placement', () => {
  it('slides the 6DoF readout left so it never runs off the right edge of the picture', () => {
    const { ctx, rects } = recordingContext();
    const face = { id: 'edge', clusterId: 'edge', bbox: [0.82, 0.2, 0.98, 0.6], pose: { yaw: 10, pitch: 0, roll: 0 } };
    const base = { head: null, mapBox: (b: number[]) => ({ x: b[0] * 1000, y: b[1] * 1000, w: (b[2] - b[0]) * 1000, h: (b[3] - b[1]) * 1000 }), mapPoint: (p: number[]) => ({ x: p[0] * 1000, y: p[1] * 1000 }),
      poseOf: (f: { pose: { yaw: number; pitch: number; roll: number } }) => ({ yaw: f.pose.yaw, pitch: f.pose.pitch, roll: f.pose.roll }), keyOf: faceKey, wireframe: false, gaze: false };
    drawHeadOverlay(ctx, [face] as never, { ...base, bounds: { left: 0, right: 1000 } } as never);
    expect(rects.at(-1)![0] + rects.at(-1)![2]).toBeLessThanOrEqual(1000);              // clamped inside the picture
    const unclamped = recordingContext();
    drawHeadOverlay(unclamped.ctx, [face] as never, base as never);
    expect(unclamped.rects.at(-1)![0]).toBeCloseTo(820);                               // without bounds it starts at the box edge, as before
  });
});
