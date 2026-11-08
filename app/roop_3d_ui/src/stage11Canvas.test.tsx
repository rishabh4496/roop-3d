// @vitest-environment jsdom
/** Stage 11 WebGL2 comparison canvas: shader maths, colormaps, pose gizmo, the GL component (against a recording fake context), and its fallback. */
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  COLORMAPS, COLORMAP_STOPS, DIFF_FULL_SCALE, DIFF_NOISE_FLOOR, FRAGMENT_SHADER, GIZMO_COLORS, LOUPE_ZOOM, MODE_IDS, VERTEX_SHADER,
  buildGizmoGeometry, catmullRomWeights, colormapColor, colormapLut, crossfadeValue, diffValue, gizmoFaces, loupeSample, noseBridge, poseAxes, webgl2Available,
} from './components/canvas/webglModel';
import { diffHeatmap } from './components/canvas/WorkstationCanvas';
import WebGLViewport from './components/canvas/WebGLViewport';
import WorkstationCanvas from './components/canvas/WorkstationCanvas';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

describe('colormaps', () => {
  it('Inferno and Viridis hit their published endpoints and the LUT is 256 opaque RGBA entries', () => {
    expect(colormapColor('inferno', 0).map(Math.round)).toEqual([0, 0, 4]);
    expect(colormapColor('inferno', 1).map(Math.round)).toEqual([252, 255, 164]);
    expect(colormapColor('viridis', 0).map(Math.round)).toEqual([68, 1, 84]);
    expect(colormapColor('viridis', 1).map(Math.round)).toEqual([253, 231, 37]);
    for (const name of COLORMAPS) {
      const lut = colormapLut(name);
      expect(lut).toHaveLength(256 * 4);
      expect([...lut].filter((_, i) => i % 4 === 3).every(alpha => alpha === 255)).toBe(true);
      expect(COLORMAP_STOPS[name as keyof typeof COLORMAP_STOPS]).toHaveLength(10);
    }
  });
  it('Inferno brightens monotonically (so a larger difference always reads hotter) and clamps out-of-range input', () => {
    const luma = (t: number) => { const [r, g, b] = colormapColor('inferno', t); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
    for (let i = 1; i <= 20; i++) expect(luma(i / 20)).toBeGreaterThan(luma((i - 1) / 20));
    expect(colormapColor('inferno', 5)).toEqual(colormapColor('inferno', 1));
    expect(colormapColor('inferno', -1)).toEqual(colormapColor('inferno', 0));
    expect(colormapColor('nope', 0.5)).toEqual(colormapColor('inferno', 0.5));
  });
});

describe('shader maths (CPU reference)', () => {
  it('diff value is |swapped - original| * gain through the noise floor, matching the 2D heat-map scale', () => {
    expect(diffValue([10, 20, 30], [10, 20, 30], 3)).toBe(0);
    expect(diffValue([100, 100, 100], [100 + DIFF_NOISE_FLOOR - 1, 100, 100], 8)).toBe(0);            // JPEG noise stays dark
    expect(diffValue([0, 0, 0], [64, 64, 64], 1)).toBeCloseTo(64 / DIFF_FULL_SCALE, 6);
    expect(diffValue([0, 0, 0], [255, 255, 255], 8)).toBe(1);                                          // saturates at the hot end
    expect(diffValue([0, 0, 0], [30, 60, 90], 2)).toBeCloseTo((60 * 2) / DIFF_FULL_SCALE, 6);
    // the existing 2D path uses the same constants: a pixel the shader floors is the one the CPU floors
    const out = new Uint8ClampedArray(4); diffHeatmap(new Uint8ClampedArray([100, 100, 100, 255]), new Uint8ClampedArray([102, 100, 100, 255]), 3, out);
    expect(diffValue([100, 100, 100], [102, 100, 100], 3)).toBe(0); expect(out[0]).toBe(15);
  });
  it('Catmull-Rom weights sum to 1, interpolate the sample at f=0/1 and are symmetric at f=0.5', () => {
    for (const f of [0, 0.1, 0.25, 0.5, 0.8, 0.999]) expect(catmullRomWeights(f).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    catmullRomWeights(0).forEach((weight, tap) => expect(weight).toBeCloseTo([0, 1, 0, 0][tap], 10));
    const half = catmullRomWeights(0.5); expect(half[0]).toBeCloseTo(half[3], 10); expect(half[1]).toBeCloseTo(half[2], 10); expect(half[0]).toBeLessThan(0);   // negative lobes = sharper than bilinear
  });
  it('the loupe magnifies 3.5x about the pointer', () => {
    expect(LOUPE_ZOOM).toBe(3.5);
    expect(loupeSample({ x: 500, y: 300 }, { width: 1000, height: 600 }, [0.5, 0.5])).toEqual([0.5, 0.5]);        // centre maps to itself
    const [u] = loupeSample({ x: 500, y: 300 }, { width: 1000, height: 600 }, [0.5 + 0.35, 0.5]);
    expect(u).toBeCloseTo(0.5 + 0.35 / 3.5, 10);
  });
  it('crossfade is a 0→1→0 cosine', () => {
    expect(crossfadeValue(0, 2)).toBeCloseTo(0, 10); expect(crossfadeValue(1000, 2)).toBeCloseTo(1, 10); expect(crossfadeValue(2000, 2)).toBeCloseTo(0, 10);
  });
  it('GLSL is built from the same constants and contains every mode and both samplers', () => {
    expect(VERTEX_SHADER).toContain('#version 300 es'); expect(FRAGMENT_SHADER).toContain('#version 300 es');
    expect(FRAGMENT_SHADER).toContain(`const float NOISE_FLOOR = ${DIFF_NOISE_FLOOR}.0 / 255.0;`);
    expect(FRAGMENT_SHADER).toContain(`const float FULL_SCALE = ${DIFF_FULL_SCALE}.0 / 255.0;`);
    for (const needle of ['abs(b - a)', 'sampleCatmullRom', 'uBicubic', 'uLoupeZoom', 'uMode == 3', 'uMode == 4', 'mix(a, b, uMix)', 'uColormap']) expect(FRAGMENT_SHADER).toContain(needle);
    expect(Object.keys(MODE_IDS)).toEqual(['wipe', 'crossfade', 'blend', 'diff', 'loupe', 'result', 'matte']);
  });
});

describe('3D pose axis gizmo', () => {
  const near = (value: number, expected: number) => expect(value).toBeCloseTo(expected, 6);
  it('frontal pose: X points right, Y points up, Z comes straight at the camera', () => {
    const axes = poseAxes({ yaw: 0, pitch: 0, roll: 0 });
    near(axes.x.sx, 1); near(axes.x.sy, 0); near(axes.y.sx, 0); near(axes.y.sy, -1); near(axes.z.depth, 1); near(axes.z.sx, 0);
  });
  it('follows the standard frame: yaw + turns the nose toward image right, pitch + looks up, roll + tilts clockwise', () => {
    const yaw = poseAxes({ yaw: 90, pitch: 0, roll: 0 }); near(yaw.z.sx, 1); near(yaw.z.depth, 0);
    const pitch = poseAxes({ yaw: 0, pitch: 90, roll: 0 }); near(pitch.z.sy, -1); near(pitch.z.sx, 0);
    const roll = poseAxes({ yaw: 0, pitch: 0, roll: 90 }); near(roll.y.sx, 1); near(roll.x.sy, 1);
    const turned = poseAxes({ yaw: 30, pitch: 0, roll: 0 }); expect(turned.z.sx).toBeGreaterThan(0);
  });
  it('anchors at the nose bridge (midway between the eyes and the nose tip) from the 5-point landmarks', () => {
    const face = { landmarks5: [[0.4, 0.4], [0.6, 0.4], [0.5, 0.6], [0.43, 0.75], [0.57, 0.75]], bbox: [0.3, 0.2, 0.7, 0.9] };
    const [x, y] = noseBridge(face); near(x, 0.5); near(y, 0.5);
    const bare = noseBridge({ bbox: [0.2, 0.2, 0.6, 0.8] }); near(bare[0], 0.4); expect(bare[1]).toBeGreaterThan(0.2);
  });
  it('only faces with a pose get a gizmo; the legacy pitch sign is converted to the standard frame', () => {
    const faces = gizmoFaces([{ id: 'a', bbox: [0.4, 0.3, 0.6, 0.7], landmarks5: [], pose: { yaw: 10, pitch: 20, roll: 0 } }, { id: 'b', bbox: [0, 0, 1, 1] }, null]);
    expect(faces).toHaveLength(1); near(faces[0].pose.pitch, -20); near(faces[0].height, 0.4);
    expect(gizmoFaces([{ id: 'c', bbox: [0, 0, 1, 1], pose3d: { yaw: 0, pitch: 20, roll: 0 } }])[0].pose.pitch).toBe(20);
  });
  it('builds three coloured arrows per face as clip-space triangles that stay on screen, far axes first', () => {
    const faces = gizmoFaces([{ id: 'a', bbox: [0.4, 0.3, 0.6, 0.7], landmarks5: [[0.45, 0.4], [0.55, 0.4], [0.5, 0.5], [0.46, 0.6], [0.54, 0.6]], pose3d: { yaw: 25, pitch: 10, roll: -5 } }]);
    const data = buildGizmoGeometry(faces, 1920, 1080);
    expect(data.length).toBe(3 * 9 * 6);                                          // 3 axes × (6 shaft + 3 head vertices) × (x,y,r,g,b,a)
    for (let i = 0; i < data.length; i += 6) { expect(Math.abs(data[i])).toBeLessThanOrEqual(1); expect(Math.abs(data[i + 1])).toBeLessThanOrEqual(1); expect(Number.isFinite(data[i + 2])).toBe(true); }
    const colours = new Set<string>(); for (let i = 2; i < data.length; i += 6) colours.add([...data.slice(i, i + 4)].map(v => v.toFixed(2)).join());
    expect(colours).toEqual(new Set(Object.values(GIZMO_COLORS).map(c => c.map(v => v.toFixed(2)).join())));
    expect(buildGizmoGeometry([], 100, 100)).toHaveLength(0);
  });
});

/* ───────────────────────── recording fake WebGL2 context ───────────────────────── */
type Call = [string, unknown[]];
function fakeGL(options: { failCompile?: boolean } = {}) {
  const calls: Call[] = [], uniforms = new Map<string, unknown[]>(), enums = new Map<string, number>();
  let next = 1;
  const handlers: Record<string, (...args: any[]) => unknown> = {
    createShader: () => ({ id: next++ }), createProgram: () => ({ id: next++ }), createTexture: () => ({ id: next++ }), createBuffer: () => ({ id: next++ }), createVertexArray: () => ({ id: next++ }),
    getShaderParameter: () => !options.failCompile, getShaderInfoLog: () => 'syntax error', getProgramParameter: () => true, getProgramInfoLog: () => '',
    getUniformLocation: (_program, name) => ({ name }), getAttribLocation: () => 0,
    getParameter: (what: number) => (what === enums.get('MAX_VIEWPORT_DIMS') ? [16384, 16384] : 16384), isContextLost: () => false, getExtension: () => null,
  };
  const gl = new Proxy({} as Record<string, unknown>, {
    get(_target, name: string) {
      if (/^[A-Z][A-Z0-9_]+$/.test(name)) { if (!enums.has(name)) enums.set(name, 0x1000 + enums.size); return enums.get(name); }
      return (...args: unknown[]) => {
        calls.push([name, args]);
        if (/^uniform/.test(name)) uniforms.set((args[0] as { name: string }).name, args.slice(1));
        return handlers[name]?.(...args);
      };
    },
  });
  const count = (name: string) => calls.filter(([call]) => call === name).length;
  return { gl, calls, uniforms, count, drawn: () => calls.filter(([call]) => call === 'drawArrays').map(([, args]) => args) };
}

let host: HTMLDivElement, root: Root, rafQueue: FrameRequestCallback[], context: ReturnType<typeof fakeGL> | null;
const flushFrames = async (times = 3) => { for (let i = 0; i < times; i++) { const run = rafQueue.splice(0); await act(async () => { run.forEach(callback => callback(0)); }); } };
const image = (width: number, height: number) => ({ naturalWidth: width, naturalHeight: height, width, height }) as unknown as HTMLImageElement;
beforeEach(() => {
  rafQueue = []; context = fakeGL();
  vi.stubGlobal('WebGL2RenderingContext', class {});
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => rafQueue.push(callback));
  vi.stubGlobal('cancelAnimationFrame', () => {});
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement, type: string) { return type === 'webgl2' && context ? (context.gl as never) : null; } as never);
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('WebGLViewport', () => {
  const base = { imageA: image(3840, 2160), imageB: image(3840, 2160), width: 1920, height: 1080, pixelRatio: 2 };
  const mount = async (props: Record<string, unknown> = {}) => { await act(async () => root.render(<WebGLViewport mode="wipe" {...base} {...props}/>)); await flushFrames(); };

  it('feature-detects WebGL2 without touching a context where it does not exist', () => {
    vi.unstubAllGlobals(); expect(webgl2Available()).toBe(false);                      // jsdom defines no WebGL2RenderingContext
    vi.stubGlobal('WebGL2RenderingContext', class {}); expect(webgl2Available()).toBe(true);
    context = null; expect(webgl2Available()).toBe(false);
  });

  it('compiles both programs, uploads native-resolution textures, and renders at a 4K backing store', async () => {
    const ready = vi.fn();
    await mount({ onReady: ready });
    expect(context!.count('createShader')).toBe(4); expect(context!.count('linkProgram')).toBe(2); expect(ready).toHaveBeenCalledWith({ maxTexture: 16384 });
    const uploads = context!.calls.filter(([name, args]) => name === 'texImage2D' && args.length === 6);
    expect(uploads).toHaveLength(2);                                                   // original + swapped, passed straight from the 3840×2160 images (no downscale)
    expect(uploads.every(([, args]) => (args[5] as HTMLImageElement).naturalWidth === 3840)).toBe(true);
    expect(context!.count('texImage2D') - uploads.length).toBe(COLORMAPS.length);       // one 256×1 LUT per colormap
    const canvas = host.querySelector<HTMLCanvasElement>('[data-testid="webgl-viewport"]')!;
    expect([canvas.width, canvas.height]).toEqual([3840, 2160]);                       // 1920×1080 CSS px × devicePixelRatio 2
    expect(context!.uniforms.get('uMode')).toEqual([MODE_IDS.wipe]); expect(context!.uniforms.get('uSplit')).toEqual([0.5]);
    expect(context!.drawn()[0]).toEqual([context!.gl.TRIANGLE_STRIP, 0, 4]);
  });

  it('draws only when something changed, and coalesces bursts into one frame', async () => {
    await mount();
    const draws = () => context!.drawn().length;
    const idle = draws(); await flushFrames(); expect(draws()).toBe(idle);
    await act(async () => root.render(<WebGLViewport mode="wipe" {...base} split={0.3}/>)); await act(async () => root.render(<WebGLViewport mode="wipe" {...base} split={0.7}/>));
    await flushFrames(); expect(draws()).toBe(idle + 1); expect(context!.uniforms.get('uSplit')![0]).toBeCloseTo(0.7);
  });

  it('diff mode feeds the gain and selected colormap; loupe mode turns on the lens and the bicubic toggle', async () => {
    await mount({ mode: 'diff', gain: 5, colormap: 'viridis' });
    expect(context!.uniforms.get('uMode')).toEqual([MODE_IDS.diff]); expect(context!.uniforms.get('uGain')).toEqual([5]);
    await mount({ mode: 'loupe', bicubic: false, pointer: { x: 400, y: 300, active: true } });
    expect(context!.uniforms.get('uMode')).toEqual([MODE_IDS.loupe]); expect(context!.uniforms.get('uBicubic')).toEqual([0]);
    expect(context!.uniforms.get('uLoupeZoom')).toEqual([3.5]); expect(context!.uniforms.get('uPointer')).toEqual([800, 600]);   // CSS px → device px
    expect((context!.uniforms.get('uLoupeRadius')![0] as number)).toBeGreaterThan(0);
    await mount({ mode: 'loupe', bicubic: true, pointer: { x: 400, y: 300, active: true } });
    expect(context!.uniforms.get('uBicubic')).toEqual([1]);
    await mount({ mode: 'loupe', pointer: { x: 400, y: 300, active: false } });
    expect(context!.uniforms.get('uLoupeRadius')).toEqual([0]);
  });

  it('draws the pose gizmo as a second pass only when enabled and a face has a pose', async () => {
    const face = { id: 'f', bbox: [0.4, 0.3, 0.6, 0.7], landmarks5: [[0.45, 0.4], [0.55, 0.4], [0.5, 0.5], [0.46, 0.6], [0.54, 0.6]], pose3d: { yaw: 20, pitch: 5, roll: 0 } };
    await mount({ faces: [face], showGizmo: true });
    const passes = context!.drawn();
    expect(passes.at(-1)).toEqual([context!.gl.TRIANGLES, 0, 27]); expect(passes.length).toBeGreaterThanOrEqual(2);
    expect(context!.count('blendFunc')).toBeGreaterThan(0);
    context = fakeGL(); rafQueue = []; await act(async () => root.unmount()); root = createRoot(host);
    await mount({ faces: [face], showGizmo: false });
    expect(context!.drawn().every(args => args[0] === context!.gl.TRIANGLE_STRIP)).toBe(true);
  });

  it('runs the persistent crossfade on the GPU uniform without re-rendering React, reporting at most ~10 Hz', async () => {
    let clock = 0; const onFade = vi.fn();
    await mount({ mode: 'crossfade', fadePlaying: true, fadeSeconds: 2, now: () => clock, onFade });
    for (const t of [0, 250, 500, 1000]) { clock = t; await flushFrames(1); }
    await flushFrames(2);                                                              // the tick at t=1000 schedules one more draw
    expect(context!.uniforms.get('uMix')![0]).toBeCloseTo(crossfadeValue(1000, 2), 6);
    expect(onFade.mock.calls.length).toBeLessThanOrEqual(5); expect(onFade).toHaveBeenCalledWith(expect.any(Number));
  });

  it('reports unsupported when there is no WebGL2 context or the shaders fail, so the caller can fall back', async () => {
    const unsupported = vi.fn(); context = null;
    await mount({ onUnsupported: unsupported }); expect(unsupported).toHaveBeenCalledWith('WebGL2 is not available in this browser.');
    await act(async () => root.unmount()); root = createRoot(host); context = fakeGL({ failCompile: true }); unsupported.mockClear();
    await mount({ onUnsupported: unsupported }); expect(unsupported.mock.calls[0][0]).toContain('Shader compile failed');
  });
});

describe('WorkstationCanvas GPU path', () => {
  class LoadedImage { naturalWidth = 1920; naturalHeight = 1080; onload: (() => void) | null = null; onerror: (() => void) | null = null; set src(_value: string) { queueMicrotask(() => this.onload?.()); } }
  const mount = async (props: Record<string, unknown> = {}) => {
    vi.stubGlobal('Image', LoadedImage);
    await act(async () => root.render(<WorkstationCanvas fullBleed output="out.mp4" originalUrl="blob:a" swappedUrl="blob:b" containerWidth={1280} containerHeight={720} mediaDimensions={{ width: 1920, height: 1080 }} {...props}/>));
    await act(async () => { await Promise.resolve(); }); await flushFrames(1);
  };

  it('renders every inspection mode through the shader canvas instead of stacked <img>s and a 2D pixel loop', async () => {
    await mount({ mode: 'diff' });
    expect(host.querySelector('[data-testid="webgl-viewport"]')).toBeTruthy();
    expect(host.querySelector('.compare-a')).toBeNull(); expect(host.querySelector('.diff-canvas')).toBeNull();
    expect(host.querySelector('select[aria-label="Diff colormap"]')).toBeTruthy();
    expect(context!.uniforms.get('uMode')).toEqual([MODE_IDS.diff]);
    expect(host.querySelector('.gizmo-legend')?.textContent).toContain('Yaw');
    await mount({ mode: 'loupe' });
    expect([...host.querySelectorAll('.loupe-sampling button')].map(b => b.textContent)).toEqual(['Bilinear', 'Bicubic']);
  });

  it('switches the loupe between bilinear and bicubic and the diff between Inferno and Viridis from the toolbar', async () => {
    await mount({ mode: 'loupe' });
    await act(async () => ([...host.querySelectorAll('.loupe-sampling button')].find(b => b.textContent === 'Bicubic') as HTMLElement).click()); await flushFrames(1);
    expect(context!.uniforms.get('uBicubic')).toEqual([1]);
    await mount({ mode: 'diff' });
    const select = host.querySelector<HTMLSelectElement>('select[aria-label="Diff colormap"]')!;
    expect([...select.options].map(option => option.value)).toEqual(['inferno', 'viridis']);
  });

  it('falls back to the DOM + 2D path when WebGL2 is unavailable', async () => {
    vi.unstubAllGlobals(); vi.stubGlobal('requestAnimationFrame', () => 1); vi.stubGlobal('cancelAnimationFrame', () => {});
    await mount({ mode: 'wipe' });
    expect(host.querySelector('[data-testid="webgl-viewport"]')).toBeNull();
    expect(host.querySelector('.compare-a')).toBeTruthy(); expect(host.querySelector('.compare-b')).toBeTruthy();
  });

  it('falls back when the GPU rejects the shaders after mounting', async () => {
    context = fakeGL({ failCompile: true });
    await mount({ mode: 'wipe' }); await flushFrames(1);
    expect(host.querySelector('[data-testid="webgl-viewport"]')).toBeNull();
    expect(host.querySelector('.compare-a')).toBeTruthy();
  });
});
