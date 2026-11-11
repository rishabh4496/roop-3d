// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import WorkstationErrorBoundary from './components/common/WorkstationErrorBoundary';
import WebGLFallbackInspector, { frontProjection } from './components/three/WebGLFallbackInspector';
import { detectWebGL, isSoftwareRenderer, resetWebGLDetection } from './utils/webglSupport';
import { ResilientSocket, backoffDelay, type SocketState } from './utils/resilientSocket';
import { acquirePipelineStream, ingestFrame, parseProgressFrame, pipelineStream, resetPipelineStream } from './stores/pipelineStream';
import { installGlobalErrorHandlers, isBenignRejection } from './utils/globalErrors';
import { clearToasts, toastStore } from './stores/toasts';
import type { Reconstruction } from './contracts';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement, root: Root;
beforeEach(() => { host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.useRealTimers(); vi.restoreAllMocks(); resetWebGLDetection(); resetPipelineStream(); clearToasts(); });

// ---------------------------------------------------------------------------------------------------------- error boundary
describe('WorkstationErrorBoundary', () => {
  function Flaky({ fail }: { fail: { current: boolean } }) {
    if (fail.current) throw new Error('mesh buffer is corrupt');
    return <p>viewer ok</p>;
  }

  it('shows a recoverable message instead of a blank page and retries on demand', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const fail = { current: true };
    await act(async () => { root.render(<WorkstationErrorBoundary name="The 3D viewport"><Flaky fail={fail} /></WorkstationErrorBoundary>); });
    const alert = host.querySelector('[role=alert]')!;
    expect(alert.textContent).toContain('The 3D viewport stopped unexpectedly.');
    expect(alert.textContent).toContain('mesh buffer is corrupt');
    expect(alert.textContent).toContain('stored on disk');
    expect(warn).toHaveBeenCalledWith('[workstation] The 3D viewport failed', expect.any(Error), expect.any(String));
    fail.current = false;
    await act(async () => { [...host.querySelectorAll('button')].find(b => b.textContent === 'Try again')!.click(); });
    expect(host.textContent).toContain('viewer ok');
    expect(host.querySelector('[role=alert]')).toBeNull();
  });

  it('clears itself when a reset key changes (a different FaceSet gets a fresh try)', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const fail = { current: true };
    function Harness() {
      const [id, setId] = useState('a');
      return <><button id="next" onClick={() => { fail.current = false; setId('b'); }}>next</button>
        <WorkstationErrorBoundary name="Viewer" resetKeys={[id]}><Flaky fail={fail} /></WorkstationErrorBoundary></>;
    }
    await act(async () => { root.render(<Harness />); });
    expect(host.querySelector('[role=alert]')).not.toBeNull();
    await act(async () => { (host.querySelector('#next') as HTMLButtonElement).click(); });
    expect(host.textContent).toContain('viewer ok');
  });

  it('the app variant offers a reload and a custom fallback receives the error', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    await act(async () => { root.render(<WorkstationErrorBoundary name="The workstation" variant="app"><Flaky fail={{ current: true }} /></WorkstationErrorBoundary>); });
    expect([...host.querySelectorAll('button')].map(b => b.textContent)).toEqual(['Try again', 'Reload workstation']);
    await act(async () => { root.render(<WorkstationErrorBoundary name="x" fallback={({ error }) => <i>custom: {error.message}</i>}><Flaky fail={{ current: true }} /></WorkstationErrorBoundary>); });
    expect(host.textContent).toBe('custom: mesh buffer is corrupt');
  });

  it('a faulty onError reporter cannot mask the original error', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    await act(async () => { root.render(<WorkstationErrorBoundary name="v" onError={() => { throw new Error('reporter down'); }}><Flaky fail={{ current: true }} /></WorkstationErrorBoundary>); });
    expect(host.textContent).toContain('mesh buffer is corrupt');
  });
});

// ---------------------------------------------------------------------------------------------------------- WebGL detection
describe('detectWebGL', () => {
  const context = (renderer: string | null, webgl2 = true) => () => ({
    webgl2,
    context: {
      RENDERER: 1, VENDOR: 2,
      getExtension: (name: string) => (name === 'WEBGL_debug_renderer_info' && renderer !== null ? { UNMASKED_RENDERER_WEBGL: 10, UNMASKED_VENDOR_WEBGL: 11 } : name === 'WEBGL_lose_context' ? { loseContext: lose } : null),
      getParameter: (key: number) => (key === 10 || key === 1 ? renderer ?? 'WebKit WebGL' : key === 11 || key === 2 ? 'Google Inc.' : ''),
    } as unknown as WebGLRenderingContext,
  });
  const lose = vi.fn();

  it('reports no WebGL with a reason a person can act on', () => {
    const report = detectWebGL(() => ({ context: null, webgl2: false }), true);
    expect(report).toMatchObject({ available: false, software: false });
    expect(report.reason).toMatch(/hardware acceleration/);
  });

  it('recognises software rasterisers and releases the probe context', () => {
    const report = detectWebGL(context('ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)), SwiftShader driver)'), true);
    expect(report).toMatchObject({ available: true, software: true, webgl2: true });
    expect(lose).toHaveBeenCalled();
    for (const name of ['llvmpipe (LLVM 15.0.7, 256 bits)', 'Microsoft Basic Render Driver', 'Mesa OffScreen']) expect(isSoftwareRenderer(name)).toBe(true);
    for (const name of ['NVIDIA GeForce RTX 4070', 'Apple M2', 'ANGLE (AMD, Radeon RX 7800 XT Direct3D11)']) expect(isSoftwareRenderer(name)).toBe(false);
  });

  it('treats a throwing driver as "unavailable", and caches the answer', () => {
    const factory = vi.fn(() => { throw new Error('GPU process crashed'); });
    expect(detectWebGL(factory, true)).toMatchObject({ available: false, reason: 'GPU process crashed' });
    detectWebGL(factory);
    expect(factory).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------------------------------------- fallback inspector
describe('lightweight inspector', () => {
  const mesh = (frame: Reconstruction['coordinate_system']): Reconstruction => ({
    method: 'identity_asset', coordinate_system: frame, vertices: [[0, 0, 0], [1, 0, 0.2], [0, 2, 0.1], [1, 2, 0.3]], faces: [[0, 1, 2], [1, 3, 2]], uvs: null, texture: 'data:image/png;base64,AA', provenance: 'three_tier_facial_surface',
  });

  it('projects the mesh into a 0..100 SVG box and reports its extent', () => {
    const { path, edges, bounds } = frontProjection(mesh('mesh'));
    expect(edges).toBe(6);
    expect(path.startsWith('M')).toBe(true);
    expect(bounds).toEqual({ width: 1, height: 2, depth: 0.3 });
    // y-up frames are flipped so the head is not upside down.
    expect(frontProjection(mesh('opengl')).path).not.toBe(path);
  });

  it('survives garbage: missing arrays, NaN vertices, one-vertex faces', () => {
    expect(frontProjection(null).path).toBe('');
    expect(frontProjection({ ...mesh('mesh'), vertices: [[Number.NaN, 0, 0]] }).path).toBe('');
    expect(frontProjection({ ...mesh('mesh'), faces: [[0], [0, 1]] as never }).edges).toBe(0);
    expect(() => frontProjection({ ...mesh('mesh'), vertices: undefined as never })).not.toThrow();
  });

  it('thins very dense meshes to a bounded number of edges', () => {
    const faces = Array.from({ length: 40000 }, (_, i) => [i % 4, (i + 1) % 4, (i + 2) % 4]);
    expect(frontProjection({ ...mesh('mesh'), faces }, 3000).edges).toBeLessThanOrEqual(3 * 3000);
  });

  it('renders stats, the reason, and the recovery buttons', async () => {
    const retry = vi.fn();
    await act(async () => { root.render(<WebGLFallbackInspector mesh={mesh('mesh')} reason="The graphics context was lost." onRetry={retry} />); });
    expect(host.querySelector('[data-testid=webgl-fallback]')).not.toBeNull();
    expect(host.textContent).toContain('The graphics context was lost.');
    expect(host.textContent).toContain('4'); // vertices
    expect(host.querySelector('svg path')).not.toBeNull();
    await act(async () => { [...host.querySelectorAll('button')].find(b => b.textContent === 'Restart 3D view')!.click(); });
    expect(retry).toHaveBeenCalled();
  });

  it('says so when there is no mesh', async () => {
    await act(async () => { root.render(<WebGLFallbackInspector mesh={null} reason="" />); });
    expect(host.textContent).toContain('No mesh is loaded yet.');
    expect(host.textContent).toContain('The 3D viewport is not available.');
  });
});

// ---------------------------------------------------------------------------------------------------------- reconnecting socket
class FakeSocket {
  onopen: (() => void) | null = null; onmessage: ((event: { data: string }) => void) | null = null; onclose: (() => void) | null = null; onerror: (() => void) | null = null;
  closed = false;
  close() { this.closed = true; }
  open() { this.onopen?.(); }
  send(value: unknown) { this.onmessage?.({ data: typeof value === 'string' ? value : JSON.stringify(value) }); }
  drop() { this.onclose?.(); }
}
const factory = () => { const sockets: FakeSocket[] = []; return { sockets, make: (_url: string) => { const s = new FakeSocket(); sockets.push(s); return s as never; } }; };

describe('backoffDelay', () => {
  it('doubles from 500 ms up to a 15 s ceiling', () => {
    expect([0, 1, 2, 3, 4, 5, 6, 12].map(attempt => backoffDelay(attempt, {}, () => 0.5))).toEqual([500, 1000, 2000, 4000, 8000, 15000, 15000, 15000]);
  });
  it('jitters by at most +-25 % and never exceeds the ceiling or goes negative', () => {
    expect(backoffDelay(3, {}, () => 0)).toBe(3000);
    expect(backoffDelay(3, {}, () => 1)).toBe(5000);
    expect(backoffDelay(10, {}, () => 1)).toBe(15000);
    expect(backoffDelay(0, { jitter: 5 }, () => 0)).toBe(0);
  });
});

describe('ResilientSocket', () => {
  const setup = (options: Partial<ConstructorParameters<typeof ResilientSocket>[0]> = {}) => {
    vi.useFakeTimers();
    const { sockets, make } = factory();
    const states: [SocketState, number | null][] = [];
    const opened: boolean[] = [];
    const messages: unknown[] = [];
    const socket = new ResilientSocket({ url: 'ws://x/ws', makeSocket: make, random: () => 0.5, onMessage: data => messages.push(data), onState: (state, info) => states.push([state, info.nextDelayMs]), onOpen: r => opened.push(r), ...options });
    return { socket, sockets, states, opened, messages };
  };

  it('reconnects after every drop with exponentially growing waits, and a successful open resets the ladder', () => {
    const { socket, sockets, states, opened } = setup();
    socket.start();
    expect(sockets).toHaveLength(1);
    sockets[0].open();
    sockets[0].drop();                                     // backend restarted
    expect(states.at(-1)).toEqual(['reconnecting', 500]);
    vi.advanceTimersByTime(499); expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(1); expect(sockets).toHaveLength(2);
    sockets[1].drop();                                     // still down: next wait doubles
    expect(states.at(-1)).toEqual(['reconnecting', 1000]);
    vi.advanceTimersByTime(1000); expect(sockets).toHaveLength(3);
    sockets[2].drop();
    expect(states.at(-1)).toEqual(['reconnecting', 2000]);
    vi.advanceTimersByTime(2000);
    sockets[3].open();                                     // back: ladder resets
    expect(states.at(-1)).toEqual(['online', null]);
    expect(opened).toEqual([false, true]);
    sockets[3].drop();
    expect(states.at(-1)).toEqual(['reconnecting', 500]);
    socket.stop();
  });

  it('never stops trying while started, then goes quiet on stop()', () => {
    const { socket, sockets, states } = setup();
    socket.start();
    for (let i = 0; i < 12; i++) { sockets.at(-1)!.drop(); vi.advanceTimersByTime(20_000); }
    expect(sockets.length).toBeGreaterThan(12);
    const count = sockets.length;
    socket.stop();
    vi.advanceTimersByTime(60_000);
    expect(sockets).toHaveLength(count);
    expect(states.at(-1)).toEqual(['offline', null]);
    expect(sockets.at(-1)!.closed).toBe(true);
  });

  it('delivers parsed frames, ignores malformed ones, and survives a throwing handler', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    let first = true;
    const { socket, sockets, messages } = setup({ onMessage: data => { if (first) { first = false; throw new Error('handler bug'); } messages.push(data); } });
    socket.start(); sockets[0].open();
    sockets[0].send({ type: 'a' });                        // handler throws: logged, socket kept
    sockets[0].send('{not json');                          // dropped
    sockets[0].send({ type: 'b' });
    expect(messages).toEqual([{ type: 'b' }]);
    expect(warn).toHaveBeenCalled();
    expect(sockets).toHaveLength(1);
    socket.stop();
  });

  it('a throwing WebSocket constructor is just another failed attempt', () => {
    vi.useFakeTimers();
    let calls = 0;
    const socket = new ResilientSocket({ url: 'ws://x', random: () => 0.5, onMessage: () => undefined, makeSocket: () => { calls += 1; if (calls < 3) throw new Error('SecurityError'); return new FakeSocket() as never; } });
    socket.start();
    vi.advanceTimersByTime(500 + 1000);
    expect(calls).toBe(3);
    socket.stop();
  });

  it('retries at once when the browser reports the network is back', () => {
    const { socket, sockets } = setup();
    socket.start(); sockets[0].open(); sockets[0].drop();
    vi.advanceTimersByTime(500); sockets[1].drop();       // now waiting 1000 ms
    expect(sockets).toHaveLength(2);
    window.dispatchEvent(new Event('online'));
    expect(sockets).toHaveLength(3);
    socket.stop();
  });

  it('a stale watchdog replaces a silent connection', () => {
    const { socket, sockets } = setup({ staleMs: 5000 });
    socket.start(); sockets[0].open();
    vi.advanceTimersByTime(4000); sockets[0].send({ type: 'tick' });          // traffic re-arms it
    vi.advanceTimersByTime(4000); expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(1001);                                              // silent for 5 s: dropped
    expect(sockets[0].closed).toBe(true);
    vi.advanceTimersByTime(500); expect(sockets).toHaveLength(2);
    socket.stop();
  });

  it('events from a replaced socket are ignored', () => {
    const { socket, sockets, messages } = setup();
    socket.start(); sockets[0].open();
    const stale = sockets[0];
    socket.reconnectNow();
    expect(sockets).toHaveLength(2);
    stale.send({ type: 'ghost' }); stale.drop();
    expect(messages).toEqual([]);
    vi.advanceTimersByTime(60_000);
    expect(sockets).toHaveLength(2);
    socket.stop();
  });
});

// ---------------------------------------------------------------------------------------------------------- /ws/v1/pipeline/status store
describe('pipeline status stream', () => {
  const frame = (patch: Record<string, unknown> = {}) => ({ type: 'model_download_progress', model_id: 'det_10g', downloaded_bytes: 50, total_bytes: 200, percent: 25, speed_mbps: 80, eta_seconds: 12.5, status: 'downloading', ...patch });

  it('validates progress frames into fully-defaulted records', () => {
    expect(parseProgressFrame(frame(), 7)).toEqual({
      modelId: 'det_10g', status: 'downloading', downloadedBytes: 50, totalBytes: 200, percent: 25, speedMbps: 80, etaSeconds: 12.5, mirror: null, batchId: null, detail: null, receivedAt: 7,
    });
    expect(parseProgressFrame({ type: 'model_download_progress', model_id: 'x' })).toMatchObject({ status: 'downloading', downloadedBytes: 0, totalBytes: 0, percent: 0, speedMbps: 0, etaSeconds: null });
  });

  it('never trusts the wire: bad numbers, unknown status, missing id, wrong type', () => {
    expect(parseProgressFrame(frame({ percent: 400, speed_mbps: -3, eta_seconds: -1 }))).toMatchObject({ percent: 100, speedMbps: 0, etaSeconds: null });
    expect(parseProgressFrame(frame({ percent: 'NaN', downloaded_bytes: 50, total_bytes: 200 }))!.percent).toBe(25);          // derived from the byte counts
    expect(parseProgressFrame(frame({ status: 'exploding' }))!.status).toBe('downloading');
    expect(parseProgressFrame(frame({ eta_seconds: null }))!.etaSeconds).toBeNull();
    expect(parseProgressFrame(frame({ model_id: '' }))).toBeNull();
    expect(parseProgressFrame({ ...frame(), type: 'pipeline.status' })).toBeNull();
    expect(parseProgressFrame(null)).toBeNull();
    expect(parseProgressFrame('frame')).toBeNull();
    expect(parseProgressFrame(frame({ bytes_downloaded: 9, downloaded_bytes: undefined }))!.downloadedBytes).toBe(9);          // the older field name
  });

  it('ingests frames per model and ignores everything else without throwing', () => {
    ingestFrame(frame()); ingestFrame(frame({ model_id: 'other', percent: 90 })); ingestFrame(frame({ percent: 60 }));
    ingestFrame(null); ingestFrame(42); ingestFrame({ type: 'pipeline.status' });
    const { transfers } = pipelineStream.get();
    expect(Object.keys(transfers).sort()).toEqual(['det_10g', 'other']);
    expect(transfers.det_10g.percent).toBe(60);
  });

  it('connects once for any number of holders, reconnects after a backend restart and asks readers to resync', () => {
    vi.useFakeTimers();
    const { sockets, make } = factory();
    const releaseA = acquirePipelineStream({ makeSocket: make, random: () => 0.5 });
    const releaseB = acquirePipelineStream({ makeSocket: make });
    expect(sockets).toHaveLength(1);
    sockets[0].open();
    expect(pipelineStream.get()).toMatchObject({ connection: 'online', resyncs: 0 });
    sockets[0].send(frame());
    expect(pipelineStream.get().transfers.det_10g.percent).toBe(25);

    sockets[0].drop();                                    // the backend was restarted
    expect(pipelineStream.get().connection).toBe('reconnecting');
    vi.advanceTimersByTime(500);
    sockets[1].open();
    expect(pipelineStream.get()).toMatchObject({ connection: 'online', resyncs: 1 });
    sockets[1].send(frame({ percent: 70 }));
    expect(pipelineStream.get().transfers.det_10g.percent).toBe(70);

    releaseA(); releaseA();                               // releasing twice must not drop the other holder
    expect(sockets[1].closed).toBe(false);
    releaseB();
    expect(sockets[1].closed).toBe(true);
    expect(pipelineStream.get().connection).toBe('offline');
  });
});

// ---------------------------------------------------------------------------------------------------------- global net
describe('unhandled promise rejections', () => {
  const rejection = (reason: unknown) => { const event = new Event('unhandledrejection', { cancelable: true }) as PromiseRejectionEvent; Object.assign(event, { reason, promise: Promise.resolve() }); return event; };

  it('cancelled requests are benign', () => {
    expect(isBenignRejection(Object.assign(new Error('x'), { name: 'AbortError' }))).toBe(true);
    expect(isBenignRejection(new DOMException('The user aborted a request.', 'AbortError'))).toBe(true);
    expect(isBenignRejection(new Error('Network down'))).toBe(false);
  });

  it('turns a missed rejection into one toast and one console warning, de-duplicated', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    let clock = 1000;
    const uninstall = installGlobalErrorHandlers(window, () => clock);
    const first = rejection(new Error('Local service returned 500'));
    window.dispatchEvent(first);
    expect(first.defaultPrevented).toBe(true);
    window.dispatchEvent(rejection(new Error('Local service returned 500')));
    expect(toastStore.get().items).toHaveLength(1);
    clock += 11_000;
    window.dispatchEvent(rejection(new Error('Local service returned 500')));
    expect(toastStore.get().items).toHaveLength(2);
    window.dispatchEvent(rejection('plain string reason'));
    expect(toastStore.get().items.at(-1)!.text).toBe('plain string reason');
    window.dispatchEvent(rejection(Object.assign(new Error('aborted'), { name: 'AbortError' })));
    expect(toastStore.get().items).toHaveLength(3);
    uninstall();
    const after = rejection(new Error('later'));
    window.dispatchEvent(after);
    expect(after.defaultPrevented).toBe(false);
  });
});
