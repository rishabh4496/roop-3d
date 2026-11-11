import { describe, expect, it } from 'vitest';
import {
  FOCAL_RANGE_MM, clampFocal, computeVertexCavity, dollyFactor, environmentToViewerSH, exposeEnvironment, horizontalFovDeg, parseEnvironments, verticalFovDeg, viewerRadiance,
} from './components/three/studioTools';

/** The server's formula, copied from `rendering/physical_buffers.evaluate_sh_basis`: irradiance = basis(n) . coefficients (per channel). */
function serverIrradiance(sh: number[][], [x, y, z]: [number, number, number]): number[] {
  const basis = [0.886227, 1.023328 * x, 1.023328 * y, 1.023328 * z, 0.858086 * x * y, 0.858086 * x * z, 0.858086 * y * z, 0.429043 * (x * x - y * y), 0.743125 * (3 * z * z - 1)];
  return [0, 1, 2].map(c => basis.reduce((sum, value, i) => sum + value * sh[i][c], 0));
}
const unit = (a: number, b: number, c: number): [number, number, number] => { const n = Math.hypot(a, b, c); return [a / n, b / n, c / n]; };

describe('physical camera preview maths', () => {
  it('is the field of view of a 36 x 24 mm sensor', () => {
    expect(verticalFovDeg(50)).toBeCloseTo(26.99, 1);
    expect(verticalFovDeg(18)).toBeCloseTo(67.38, 1);
    expect(verticalFovDeg(135)).toBeCloseTo(10.17, 1);
    expect(horizontalFovDeg(50)).toBeCloseTo(39.6, 1);
  });

  it('clamps to the 18-135 mm range the matcher supports and survives NaN', () => {
    expect(FOCAL_RANGE_MM).toEqual([18, 135]);
    expect(clampFocal(5)).toBe(18);
    expect(clampFocal(400)).toBe(135);
    expect(clampFocal(Number.NaN)).toBe(50);
    expect(verticalFovDeg(1)).toBeCloseTo(verticalFovDeg(18), 6);
  });

  it('dollies so the head keeps its size: a longer lens steps back, a wider one comes closer', () => {
    expect(dollyFactor(42, 42)).toBeCloseTo(1, 9);
    expect(dollyFactor(42, verticalFovDeg(135))).toBeGreaterThan(4);
    expect(dollyFactor(42, verticalFovDeg(18))).toBeLessThan(0.6);
    // Projected size of a unit sphere is ~ 1 / (distance * tan(fov / 2)): unchanged by construction.
    const size = (distance: number, fov: number) => 1 / (distance * Math.tan((fov * Math.PI) / 360));
    const fov = verticalFovDeg(85);
    expect(size(3.8 * dollyFactor(42, fov), fov)).toBeCloseTo(size(3.8, 42), 9);
  });
});

describe('studio lighting environments', () => {
  const sh = Array.from({ length: 9 }, (_, i) => [0.3 + 0.1 * Math.sin(i + 1), 0.2 + 0.1 * Math.cos(2 * i), 0.5 + 0.05 * i]);
  sh[0] = [1.5, 1.2, 0.9];

  it('re-expresses the server SH in the viewer shader basis: identical irradiance for every direction', () => {
    const converted = environmentToViewerSH(sh);
    expect(converted).toHaveLength(9);
    for (let i = 0; i < 200; i++) {
      const n = unit(Math.sin(i * 1.7), Math.cos(i * 0.9) - 0.2, Math.sin(i * 0.31 + 1));
      const expected = serverIrradiance(sh, n).map(value => Math.max(0, value));
      const actual = viewerRadiance(converted, n);
      expected.forEach((value, c) => expect(actual[c]).toBeCloseTo(value, 4));
    }
  });

  it('exposes every environment to the same frontal brightness so the five are comparable', () => {
    const bright = sh.map(row => row.map(value => value * 7));
    const dim = sh.map(row => row.map(value => value * 0.05));
    const luminance = (rows: number[][]) => { const [r, g, b] = viewerRadiance(rows, [0, 0.25, 1]); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
    expect(luminance(exposeEnvironment(environmentToViewerSH(bright)))).toBeCloseTo(0.85, 4);
    expect(luminance(exposeEnvironment(environmentToViewerSH(dim)))).toBeCloseTo(0.85, 4);
  });

  it('keeps the colour ratio when exposing (a golden-hour probe stays red)', () => {
    const golden = [[1.5, 0.97, 0.28], ...Array.from({ length: 8 }, () => [0, 0, 0])];
    const exposed = exposeEnvironment(environmentToViewerSH(golden));
    const [r, , b] = viewerRadiance(exposed, [0, 0, 1]);
    expect(r / b).toBeCloseTo(1.5 / 0.28, 3);
  });

  it('parses the server response and drops malformed environments instead of half-using them', () => {
    const good = { id: 'a', name: 'Alpha', kelvin: 5600, license: 'CC0-1.0', page_url: 'https://polyhaven.com/a/a', sh_coefficients: sh, thumbnail: 'data:image/png;base64,AAAA' };
    const list = parseEnvironments({ environments: [good, { ...good, id: 'short', sh_coefficients: sh.slice(0, 8) }, { ...good, id: 'nan', sh_coefficients: sh.map((row, i) => (i === 3 ? [1, Number.NaN, 1] : row)) }, null, { ...good, id: undefined }] });
    expect(list.map(item => item.id)).toEqual(['a']);
    expect(list[0]).toMatchObject({ name: 'Alpha', kelvin: 5600, license: 'CC0-1.0', thumbnail: 'data:image/png;base64,AAAA' });
    expect(parseEnvironments(null)).toEqual([]);
    expect(parseEnvironments({ environments: 'nope' })).toEqual([]);
    expect(parseEnvironments({ environments: [{ ...good, thumbnail: 'javascript:alert(1)' }] })[0].thumbnail).toBeNull();
  });
});

/** A flat n x n grid in the xy plane (normals +z) with a gaussian height `bump` at the centre. */
function grid(n: number, height: (r2: number) => number) {
  const positions: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 - 1, y = (j / (n - 1)) * 2 - 1;
    positions.push(x, y, height(x * x + y * y));
  }
  const index: number[] = [];
  for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
    const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
    index.push(a, b, c, b, d, c);                 // counter-clockwise seen from +z
  }
  return { positions: Float32Array.from(positions), index: Uint32Array.from(index), centre: Math.floor(n / 2) * n + Math.floor(n / 2), corner: 0 };
}

describe('cavity buffer', () => {
  it('darkens a dent, leaves a bump and a flat plane alone', () => {
    const dent = grid(21, r2 => -0.2 * Math.exp(-r2 / 0.15));
    const bump = grid(21, r2 => 0.2 * Math.exp(-r2 / 0.15));
    const flat = grid(21, () => 0);
    const dentCavity = computeVertexCavity(dent.positions, dent.index);
    expect(dentCavity[dent.centre]).toBeGreaterThan(0.6);
    expect(dentCavity[dent.corner]).toBeLessThan(0.05);
    expect(computeVertexCavity(bump.positions, bump.index)[bump.centre]).toBeLessThan(0.05);    // a bump has no hollow at its summit (its foot does)
    expect(Math.max(...computeVertexCavity(flat.positions, flat.index))).toBe(0);
  });

  it('is independent of units and of mesh resolution scale', () => {
    const small = grid(21, r2 => -0.2 * Math.exp(-r2 / 0.15));
    const scaled = Float32Array.from(small.positions, value => value * 1000);
    const a = computeVertexCavity(small.positions, small.index)[small.centre];
    const b = computeVertexCavity(scaled, small.index)[small.centre];
    expect(b).toBeCloseTo(a, 3);
  });

  it('welds split vertices: a non-indexed mesh gives the same answer as the indexed one', () => {
    const indexed = grid(15, r2 => -0.2 * Math.exp(-r2 / 0.15));
    const soup = new Float32Array(indexed.index.length * 3);
    indexed.index.forEach((vertex, k) => soup.set(indexed.positions.subarray(vertex * 3, vertex * 3 + 3), k * 3));
    const withIndex = computeVertexCavity(indexed.positions, indexed.index)[indexed.centre];
    // Find the centre vertex in the soup (any copy of it must come out identical).
    const cx = indexed.positions[indexed.centre * 3], cy = indexed.positions[indexed.centre * 3 + 1];
    let copy = -1;
    for (let k = 0; k < soup.length / 3; k++) if (soup[k * 3] === cx && soup[k * 3 + 1] === cy) { copy = k; break; }
    expect(copy).toBeGreaterThanOrEqual(0);
    expect(computeVertexCavity(soup, null)[copy]).toBeCloseTo(withIndex, 5);
  });

  it('returns zeros for degenerate input instead of throwing', () => {
    expect(Array.from(computeVertexCavity([], null))).toEqual([]);
    expect(Array.from(computeVertexCavity([0, 0, 0, 1, 0, 0], [0, 1, 0]))).toEqual([0, 0]);
    const collapsed = Float32Array.from([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(Array.from(computeVertexCavity(collapsed, [0, 1, 2, 1, 2, 3]))).toEqual([0, 0, 0, 0]);
  });
});
