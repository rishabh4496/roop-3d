/**
 * What kind of WebGL this browser really gives us. Used to decide between the 3D viewport and the lightweight inspector.
 *
 * `available` is a real context creation attempt (a throw-away canvas that is released again), never a feature guess.
 * `software` means the context is rendered on the CPU (SwiftShader / llvmpipe / "Microsoft Basic Render Driver"): it works,
 * but a head mesh with a few thousand triangles is slow, so the viewport shows a notice and offers the lightweight inspector.
 */
export interface WebGLReport {
  available: boolean;
  webgl2: boolean;
  software: boolean;
  renderer: string;
  vendor: string;
  /** Why WebGL is unavailable (empty when it is). */
  reason: string;
}

const SOFTWARE_RENDERER = /swiftshader|llvmpipe|softpipe|software|basic render|microsoft basic|mesa offscreen|angle \(.*(?:swiftshader|software)/i;
export const isSoftwareRenderer = (renderer: string): boolean => SOFTWARE_RENDERER.test(renderer);

export const LITE_INSPECTOR_KEY = 'roop3d-lite-inspector';
export const LITE_INSPECTOR_QUERY = 'r3d-lite';

type ContextFactory = () => { context: WebGLRenderingContext | WebGL2RenderingContext | null; webgl2: boolean };

const defaultFactory: ContextFactory = () => {
  const canvas = document.createElement('canvas');
  let context: WebGLRenderingContext | WebGL2RenderingContext | null = null;
  let webgl2 = false;
  try { context = canvas.getContext('webgl2'); webgl2 = !!context; } catch { context = null; }
  if (!context) { try { context = (canvas.getContext('webgl') ?? canvas.getContext('experimental-webgl')) as WebGLRenderingContext | null; } catch { context = null; } }
  return { context, webgl2 };
};

let cached: WebGLReport | null = null;

/** Probe once per page load (results are cached); `factory` and `force` exist for tests. */
export function detectWebGL(factory: ContextFactory = defaultFactory, force = false): WebGLReport {
  if (cached && !force) return cached;
  let report: WebGLReport;
  try {
    const { context, webgl2 } = factory();
    if (!context) {
      report = { available: false, webgl2: false, software: false, renderer: '', vendor: '', reason: 'This browser cannot create a WebGL context (hardware acceleration is off, or the GPU driver is blocked).' };
    } else {
      const info = context.getExtension('WEBGL_debug_renderer_info');
      const renderer = String(info ? context.getParameter(info.UNMASKED_RENDERER_WEBGL) : context.getParameter(context.RENDERER) ?? '');
      const vendor = String(info ? context.getParameter(info.UNMASKED_VENDOR_WEBGL) : context.getParameter(context.VENDOR) ?? '');
      report = { available: true, webgl2, software: isSoftwareRenderer(`${renderer} ${vendor}`), renderer, vendor, reason: '' };
      context.getExtension('WEBGL_lose_context')?.loseContext();     // do not hold one of the browser's ~16 contexts for the probe
    }
  } catch (error) {
    report = { available: false, webgl2: false, software: false, renderer: '', vendor: '', reason: error instanceof Error ? error.message : 'WebGL detection failed.' };
  }
  cached = report;
  return report;
}

export const resetWebGLDetection = (): void => { cached = null; };

/** True when the user (or a `?r3d-lite=1` link) asked for the lightweight inspector instead of the 3D viewport. */
export function liteInspectorRequested(): boolean {
  try {
    if (typeof location !== 'undefined' && new URLSearchParams(location.search).get(LITE_INSPECTOR_QUERY) === '1') return true;
    return localStorage.getItem(LITE_INSPECTOR_KEY) === '1';
  } catch { return false; }
}

export function setLiteInspector(on: boolean): void {
  try { if (on) localStorage.setItem(LITE_INSPECTOR_KEY, '1'); else localStorage.removeItem(LITE_INSPECTOR_KEY); } catch { /* not remembered */ }
}
