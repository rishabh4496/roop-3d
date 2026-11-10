import { describe, expect, it } from 'vitest';
import {
  BUFFER_VIEWS, DEFAULT_SH_LIGHT, JAW_MAX_DEG, buildJawRig, chinAt, eyeAnchors, faceEditOf, gazeToPoint, jawAngleFromPoint, jawDegToMouthOpen,
  lightAngles, lightVector, mouthOpenToJawDeg, openJaw, pointToGaze, shBasis, shLightCoefficients, shLightOf, shRadiance, withFaceEdit, withManualLight,
} from './components/three/viewportTools';
import { DEFAULT_PARAMETERS } from './components/workstation/pipeline';
import { lightDirection } from './components/three/PoseGeometryViewer';
import { cutoffToSmoothing, smoothingToCutoff, DEFAULT_MIN_CUTOFF, meshQuery } from './components/studio/WorkstationControls3D';

/** A unit-ish head: points on an ellipsoid, +y up, +z toward the viewer. */
function ellipsoid(): Float32Array {
  const points: number[] = [];
  for (let i = 0; i <= 40; i++) for (let j = 0; j < 60; j++) {
    const phi = (i / 40) * Math.PI, theta = (j / 60) * 2 * Math.PI;
    points.push(0.8 * Math.sin(phi) * Math.sin(theta), 1.0 * Math.cos(phi), 0.9 * Math.sin(phi) * Math.cos(theta));
  }
  return Float32Array.from(points);
}

describe('buffer views', () => {
  it('lists the five requested buffers plus the default shaded view, each with an honest hint', () => {
    expect(BUFFER_VIEWS.map(view => view.id)).toEqual(['lit', 'wireframe', 'albedo', 'normals', 'depth', 'sh']);
    expect(BUFFER_VIEWS.find(view => view.id === 'albedo')?.hint).toMatch(/not output of the TRUST network/i);
  });
});

describe('9-coefficient spherical harmonics relighting', () => {
  it('reproduces a clamped-cosine key light to 3-band accuracy', () => {
    const coefficients = shLightCoefficients({ yaw: 0, pitch: 0, strength: 1, ambient: 0 });
    expect(coefficients).toHaveLength(9);
    expect(shRadiance(coefficients, [0, 0, 1])).toBeCloseTo(1.0625, 3);   // toward the light
    expect(shRadiance(coefficients, [0, 0, -1])).toBeCloseTo(0.0625, 3); // away from it
    expect(shRadiance(coefficients, [1, 0, 0])).toBeCloseTo(0.09375, 3); // perpendicular (true cosine is 0)
  });

  it('orders the faces by their angle to the light and follows the light as it moves', () => {
    const right = shLightCoefficients({ yaw: 90, pitch: 0, strength: 1, ambient: 0 });
    expect(shRadiance(right, [1, 0, 0])).toBeGreaterThan(shRadiance(right, [0, 0, 1]));
    expect(shRadiance(right, [0, 0, 1])).toBeGreaterThan(shRadiance(right, [-1, 0, 0]));
    const above = shLightCoefficients({ yaw: 0, pitch: 60, strength: 1, ambient: 0 });
    expect(shRadiance(above, [0, 1, 0])).toBeGreaterThan(shRadiance(above, [0, -1, 0]));
  });

  it('ambient fill is a uniform term and strength scales only the key', () => {
    const ambientOnly = shLightCoefficients({ ...DEFAULT_SH_LIGHT, strength: 0, ambient: 0.4 });
    for (const normal of [[0, 0, 1], [1, 0, 0], [0, -1, 0], [0.3, 0.5, -0.8]]) expect(shRadiance(ambientOnly, normal)).toBeCloseTo(0.4, 4);
    const half = shLightCoefficients({ yaw: 0, pitch: 0, strength: 0.5, ambient: 0 });
    expect(shRadiance(half, [0, 0, 1])).toBeCloseTo(0.5 * 1.0625, 3);
  });

  it('the basis is orthonormal-ish at the poles and unit-length input is not required for radiance', () => {
    expect(shBasis(0, 0, 1)[0]).toBeCloseTo(0.282095, 5);
    const coefficients = shLightCoefficients(DEFAULT_SH_LIGHT);
    expect(shRadiance(coefficients, [0, 0, 5])).toBeCloseTo(shRadiance(coefficients, [0, 0, 1]), 6);
  });
});

describe('light direction', () => {
  it('matches the viewer convention and round-trips through yaw / pitch', () => {
    for (const [yaw, pitch] of [[0, 0], [35, 25], [-120, -40], [90, 0], [10, 80]]) {
      expect(lightVector(yaw, pitch)).toEqual(lightDirection(yaw, pitch));
      const back = lightAngles(lightVector(yaw, pitch));
      expect(back.yaw).toBeCloseTo(yaw, 4);
      expect(back.pitch).toBeCloseTo(pitch, 4);
    }
  });
  it('a manual light drag sets the manual mode, rounds and clamps, and leaves relighting alone', () => {
    const next = withManualLight(DEFAULT_PARAMETERS, 200.4, -95);
    expect(next).toMatchObject({ lighting_mode: 'manual', manual_light_yaw: 180, manual_light_pitch: -90 });
    expect(next.relight).toBeUndefined();
    expect(shLightOf(next)).toMatchObject({ yaw: 180, pitch: -90 });
    expect(shLightOf(DEFAULT_PARAMETERS)).toMatchObject({ yaw: 35, pitch: 25 });
  });
});

describe('jaw hinge', () => {
  const base = ellipsoid();
  const rig = buildJawRig(base)!;

  it('finds a chin at the lowest centre-line vertex and a hinge above and behind it', () => {
    expect(rig).not.toBeNull();
    expect(rig.chin[0]).toBeCloseTo(-1.0, 1);
    expect(rig.hinge[0]).toBeGreaterThan(rig.chin[0]);
    expect(rig.hinge[1]).toBeLessThan(rig.chin[1]);
    expect(buildJawRig(new Float32Array(9))).toBeNull();
  });

  it('is the identity at 0 degrees and restores exactly after opening and closing', () => {
    const out = new Float32Array(base.length);
    openJaw(base, out, rig, 0);
    expect(Array.from(out)).toEqual(Array.from(base));
    openJaw(base, out, rig, 22);
    expect(Array.from(out)).not.toEqual(Array.from(base));
    openJaw(base, out, rig, 0);
    expect(Array.from(out)).toEqual(Array.from(base));
  });

  it('moves the chin down and back, keeps the upper face and the back of the head still, and keeps distances to the hinge', () => {
    const out = new Float32Array(base.length);
    openJaw(base, out, rig, 20);
    let chinIndex = 0;
    for (let i = 0; i < base.length; i += 3) if (Math.abs(base[i]) < 0.05 && base[i + 1] < base[chinIndex * 3 + 1]) chinIndex = i / 3;
    expect(out[chinIndex * 3 + 1]).toBeLessThan(base[chinIndex * 3 + 1] - 0.05);   // down
    expect(out[chinIndex * 3 + 2]).toBeLessThan(base[chinIndex * 3 + 2] - 0.05);   // back
    expect(Math.hypot(out[chinIndex * 3 + 1] - rig.hinge[0], out[chinIndex * 3 + 2] - rig.hinge[1]))
      .toBeCloseTo(Math.hypot(base[chinIndex * 3 + 1] - rig.hinge[0], base[chinIndex * 3 + 2] - rig.hinge[1]), 3);
    for (let i = 0; i < base.length; i += 3) {
      if (base[i + 1] > 0.5 || base[i + 2] < -0.8) expect([out[i], out[i + 1], out[i + 2]]).toEqual([base[i], base[i + 1], base[i + 2]]);
    }
  });

  it('the chin handle round-trips through the drag angle and the angle is clamped to the physical range', () => {
    for (const degrees of [0, 5, 14, 27]) expect(jawAngleFromPoint(rig, chinAt(rig, degrees))).toBeCloseTo(degrees, 3);
    expect(jawAngleFromPoint(rig, chinAt(rig, 80))).toBe(JAW_MAX_DEG);
    expect(jawAngleFromPoint(rig, chinAt(rig, -30))).toBe(0);
  });

  it('maps the angle onto the engine mouth_open value both ways', () => {
    expect(jawDegToMouthOpen(JAW_MAX_DEG)).toBe(1);
    expect(jawDegToMouthOpen(-4)).toBe(0);
    expect(mouthOpenToJawDeg(jawDegToMouthOpen(14))).toBeCloseTo(14, 6);
  });
});

describe('eye gaze target', () => {
  it('maps the engine gaze (+x right, +y up) to a point in front of the eyes and back', () => {
    const origin = [0.1, 0.3, 0.7] as const;
    const point = gazeToPoint({ x: 0.5, y: -0.25 }, origin);
    expect(point[2]).toBeGreaterThan(origin[2]);
    expect(point[0]).toBeGreaterThan(origin[0]);
    expect(point[1]).toBeLessThan(origin[1]);
    const back = pointToGaze(point, origin);
    expect(back.x).toBeCloseTo(0.5, 6);
    expect(back.y).toBeCloseTo(-0.25, 6);
    expect(pointToGaze([10, -10, 0], origin)).toEqual({ x: 1, y: -1 });
  });
  it('finds eye centres from iBUG landmarks, else from the bounding box', () => {
    const landmarks = new Float32Array(68 * 3);
    for (let i = 36; i < 42; i++) landmarks.set([-0.3, 0.2, 0.8], i * 3);
    for (let i = 42; i < 48; i++) landmarks.set([0.3, 0.2, 0.8], i * 3);
    expect(eyeAnchors(new Float32Array(30), landmarks)).toEqual({ left: [expect.closeTo(-0.3, 5), expect.closeTo(0.2, 5), expect.closeTo(0.8, 5)], right: [expect.closeTo(0.3, 5), expect.closeTo(0.2, 5), expect.closeTo(0.8, 5)] });
    const fallback = eyeAnchors(ellipsoid(), null)!;
    expect(fallback.left[0]).toBeLessThan(0);
    expect(fallback.right[0]).toBeGreaterThan(0);
    expect(fallback.left[1]).toBeGreaterThan(0);
  });
});

describe('face edits written by the gizmos and panels', () => {
  it('sets a value on one track without touching the input, and clears empty entries', () => {
    const start = { ...DEFAULT_PARAMETERS, face_expression_overrides: { track_0001: { mouth_smile: 0.4 } } };
    const opened = withFaceEdit(start, 'track_0001', { mouth_open: 0.5 });
    expect(opened.face_expression_overrides).toEqual({ track_0001: { mouth_smile: 0.4, mouth_open: 0.5 } });
    expect(start.face_expression_overrides).toEqual({ track_0001: { mouth_smile: 0.4 } });
    const cleared = withFaceEdit(withFaceEdit(opened, 'track_0001', { mouth_smile: 0 }), 'track_0001', { mouth_open: 0 });
    expect(cleared.face_expression_overrides).toEqual({});
    expect(faceEditOf(opened, 'track_0001').mouth_open).toBe(0.5);
    expect(faceEditOf(opened, null)).toEqual({});
  });
});

describe('temporal smoothing mapping', () => {
  it('is monotonic (more smoothing = lower cutoff) and round-trips', () => {
    let previous = Infinity;
    for (let t = 0; t <= 1.0001; t += 0.1) { const cutoff = smoothingToCutoff(t); expect(cutoff).toBeLessThan(previous); previous = cutoff; }
    for (const t of [0, 0.25, 0.6, 1]) expect(cutoffToSmoothing(smoothingToCutoff(t))).toBeCloseTo(t, 6);
    expect(cutoffToSmoothing(DEFAULT_MIN_CUTOFF)).toBeGreaterThan(0.4);
    expect(cutoffToSmoothing(DEFAULT_MIN_CUTOFF)).toBeLessThan(0.7);
    expect(cutoffToSmoothing(Number.NaN)).toBeCloseTo(cutoffToSmoothing(DEFAULT_MIN_CUTOFF), 9);
  });
});

describe('semantic mesh query', () => {
  it('requests the MakeHuman asset on the chosen topology and sends only non-zero sliders', () => {
    expect(meshQuery('flame2023_open', {})).toBe('template=makehuman&topology=flame2023_open');
    expect(meshQuery('makehuman_cc0', { jaw_width: 1.5, nose_bridge: -2, chin_projection: 0 })).toBe('template=makehuman&topology=makehuman_cc0&jaw_width=1.5&nose_bridge=-2');
  });
});
