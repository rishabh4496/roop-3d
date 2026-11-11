// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// One shared <canvas> stands in for the WebGL canvas react-three-fiber would create.
const gl = vi.hoisted(() => ({ canvas: null as HTMLCanvasElement | null, renderer: null as { domElement: HTMLCanvasElement } | null, created: 0, invalidate: vi.fn(), frameloop: [] as string[], failCreate: false }));
vi.mock('@react-three/fiber', async () => {
  const { useEffect } = await import('react');
  return {
    // "created" counts mounts (a new context), not re-renders.
    Canvas: ({ children, frameloop }: { children: unknown; frameloop?: string }) => {
      if (gl.failCreate) throw new Error('Error creating WebGL context.');
      useEffect(() => { gl.created += 1; }, []);
      gl.frameloop.push(frameloop ?? 'always');
      return <div data-testid="r3f-canvas">{children as never}</div>;
    },
    useThree: (selector: (state: unknown) => unknown) => selector({ gl: gl.renderer, invalidate: gl.invalidate }),
  };
});
const support = vi.hoisted(() => ({ report: { available: true, webgl2: true, software: false, renderer: 'NVIDIA', vendor: 'NVIDIA', reason: '' } }));
vi.mock('./utils/webglSupport', async importOriginal => ({ ...(await importOriginal<typeof import('./utils/webglSupport')>()), detectWebGL: () => support.report }));

import GuardedCanvas, { MAX_REBUILDS, RESTORE_TIMEOUT_MS } from './components/three/GuardedCanvas';

let host: HTMLDivElement, root: Root;
const fallback = vi.fn(({ reason, retry, use3D }: { reason: string; retry?: () => void; use3D?: () => void }) => (
  <div data-testid="fallback">{reason}{retry && <button id="retry" onClick={retry}>retry</button>}{use3D && <button id="use3d" onClick={use3D}>3d</button>}</div>));

const mount = async (props: Record<string, unknown> = {}) => { await act(async () => { root.render(<GuardedCanvas fallback={fallback} {...props}><mesh /></GuardedCanvas>); }); };
const fire = (name: string) => { const event = new Event(name, { cancelable: true }); gl.canvas!.dispatchEvent(event); return event; };

beforeEach(() => {
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  gl.canvas = document.createElement('canvas'); gl.renderer = { domElement: gl.canvas }; gl.created = 0; gl.frameloop = []; gl.failCreate = false; gl.invalidate.mockClear();
  support.report = { available: true, webgl2: true, software: false, renderer: 'NVIDIA', vendor: 'NVIDIA', reason: '' };
  fallback.mockClear(); localStorage.clear();
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.useRealTimers(); vi.restoreAllMocks(); });

describe('GuardedCanvas', () => {
  it('draws the canvas when WebGL works', async () => {
    await mount();
    expect(host.querySelector('[data-testid=r3f-canvas]')).not.toBeNull();
    expect(host.querySelector('[data-testid=fallback]')).toBeNull();
    expect(gl.created).toBe(1);
    expect(host.querySelector('.guarded-canvas')!.getAttribute('data-webgl-phase')).toBe('ok');
  });

  it('shows the fallback with the browser\'s reason when WebGL is unavailable, and never creates a canvas', async () => {
    support.report = { available: false, webgl2: false, software: false, renderer: '', vendor: '', reason: 'Hardware acceleration is off.' };
    await mount();
    expect(host.querySelector('[data-testid=fallback]')!.textContent).toContain('Hardware acceleration is off.');
    expect(gl.created).toBe(0);
  });

  it('shows the fallback when the canvas itself throws on creation, and Restart rebuilds it', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    gl.failCreate = true;
    await mount({ label: '3D viewport' });
    expect(host.querySelector('[data-testid=fallback]')!.textContent).toContain('3D viewport could not start: Error creating WebGL context.');
    gl.failCreate = false;
    await act(async () => { (host.querySelector('#retry') as HTMLButtonElement).click(); });
    expect(host.querySelector('[data-testid=r3f-canvas]')).not.toBeNull();
    expect(host.querySelector('[data-testid=fallback]')).toBeNull();
  });

  it('keeps the context restorable on loss (preventDefault) and repaints when it comes back', async () => {
    vi.useFakeTimers();
    await mount();
    const lost = fire('webglcontextlost');
    expect(lost.defaultPrevented).toBe(true);                       // without this the browser may never restore the context
    await act(async () => { await Promise.resolve(); });
    expect(host.querySelector('.guarded-canvas')!.getAttribute('data-webgl-phase')).toBe('lost');
    expect(host.querySelector('[data-testid=webgl-lost]')!.textContent).toContain('Recovering');
    expect(gl.frameloop.at(-1)).toBe('never');                      // no drawing into a dead context
    vi.advanceTimersByTime(RESTORE_TIMEOUT_MS - 100);
    await act(async () => { fire('webglcontextrestored'); });
    expect(host.querySelector('.guarded-canvas')!.getAttribute('data-webgl-phase')).toBe('ok');
    expect(host.querySelector('[data-testid=webgl-lost]')).toBeNull();
    expect(gl.invalidate).toHaveBeenCalled();
    expect(gl.frameloop.at(-1)).toBe('always');
    await act(async () => { vi.advanceTimersByTime(RESTORE_TIMEOUT_MS * 2); });
    expect(gl.created).toBe(1);                                     // a restored context is never rebuilt
  });

  it('rebuilds a canvas whose context never comes back, then gives up with the fallback after MAX_REBUILDS', async () => {
    vi.useFakeTimers();
    await mount();
    for (let attempt = 1; attempt <= MAX_REBUILDS; attempt++) {
      await act(async () => { fire('webglcontextlost'); });
      await act(async () => { vi.advanceTimersByTime(RESTORE_TIMEOUT_MS + 1); });
      expect(gl.created).toBe(1 + attempt);                         // a fresh canvas (new context) each time
      expect(host.querySelector('[data-testid=fallback]')).toBeNull();
    }
    await act(async () => { fire('webglcontextlost'); });
    await act(async () => { vi.advanceTimersByTime(RESTORE_TIMEOUT_MS + 1); });
    expect(host.querySelector('[data-testid=fallback]')!.textContent).toContain('could not be restored');
    await act(async () => { (host.querySelector('#retry') as HTMLButtonElement).click(); });
    expect(host.querySelector('[data-testid=r3f-canvas]')).not.toBeNull();
  });

  it('flags a software renderer and lets the user switch to the lightweight inspector (remembered)', async () => {
    support.report = { available: true, webgl2: true, software: true, renderer: 'SwiftShader', vendor: 'Google', reason: '' };
    await mount();
    expect(host.querySelector('[data-testid=r3f-canvas]')).not.toBeNull();                       // software GL still draws: it is slow, not broken
    expect(host.querySelector('.guarded-canvas')!.getAttribute('data-webgl-software')).toBe('true');
    expect(host.querySelector('[data-testid=webgl-software]')!.textContent).toContain('SwiftShader');
    await act(async () => { [...host.querySelectorAll('button')].find(b => b.textContent === 'Use lightweight inspector')!.click(); });
    expect(host.querySelector('[data-testid=fallback]')!.textContent).toContain('Lightweight inspector selected');
    expect(localStorage.getItem('roop3d-lite-inspector')).toBe('1');
    await act(async () => { (host.querySelector('#use3d') as HTMLButtonElement).click(); });
    expect(host.querySelector('[data-testid=r3f-canvas]')).not.toBeNull();
    expect(localStorage.getItem('roop3d-lite-inspector')).toBeNull();
  });

  it('starts in the lightweight inspector when the user chose it earlier', async () => {
    localStorage.setItem('roop3d-lite-inspector', '1');
    await mount();
    expect(host.querySelector('[data-testid=fallback]')).not.toBeNull();
    expect(gl.created).toBe(0);
  });

  it('rebuilds from scratch when the resetKey (the FaceSet) changes, but not on a plain re-render', async () => {
    await mount({ resetKey: 'a' });
    await mount({ resetKey: 'a' });
    expect(gl.created).toBe(1);
    await mount({ resetKey: 'b' });
    expect(gl.created).toBeGreaterThanOrEqual(2);
  });

  it('removes its listeners when it unmounts', async () => {
    const remove = vi.spyOn(gl.canvas!, 'removeEventListener');
    await mount();
    await act(async () => { root.render(<div />); });
    expect(remove.mock.calls.map(call => call[0]).sort()).toEqual(['webglcontextlost', 'webglcontextrestored']);
  });
});
