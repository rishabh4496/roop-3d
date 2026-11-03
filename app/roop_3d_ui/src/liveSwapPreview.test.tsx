// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import {
  JOB_START_EVENT, LIVE_DEBOUNCE_MS, buildLivePayload, downloadBlob, previewFilename, useLiveSwapPreview,
  type LivePayload, type LivePreview,
} from './livePreview';
import { DEFAULT_PARAMETERS } from './components/workstation/pipeline';
import type { JobRequest } from './contracts';
import { VideoViewport } from './VideoViewport';

vi.mock('./components/canvas/FaceOverlayCanvas', () => ({ default: () => null }));
vi.mock('./components/canvas/WorkstationCanvas', () => ({ default: () => null }));
vi.mock('./components/canvas/MaskBrushOverlay', () => ({ default: () => null }));
vi.mock('./components/tray/FaceSelectionTray', () => ({ default: () => null }));
vi.mock('./components/timeline/TimelineWorkstation', () => ({ default: () => null, parseFps: () => 25 }));

const REQUEST: JobRequest = {
  target_path: 'roop_3d/.runtime/media/t.mp4', inspection_id: 'a'.repeat(32), processing: 'local', remote_consent: false,
  face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }], trim_start_seconds: null, trim_end_seconds: null,
};
const png = () => new Blob([new Uint8Array([137, 80, 78, 71])], { type: 'image/png' });
const swapped = (cache = 'MISS') => new Response(png(), { status: 200, headers: {
  'Content-Type': 'image/png', 'X-Preview-Cache': cache,
  'X-Preview-Meta': JSON.stringify({ media_id: 'a'.repeat(32), frame_index: 5, faces: [], applied: [], unsupported: ['tps_expression_weight'], notes: [], latency: { total_ms: 41.2 } }) } });
const busy = (stage = 'STAGE_1_SPARSE_PRESCAN') => new Response(JSON.stringify({ busy: true, stage, retry_after_ms: 1000 }), {
  status: 503, headers: { 'Content-Type': 'application/json', 'Retry-After': '1' } });

let host: HTMLDivElement, root: Root, fetchMock: ReturnType<typeof vi.fn>, urls: number;
beforeEach(() => {
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
  vi.useFakeTimers();
  urls = 0;
  URL.createObjectURL = vi.fn(() => `blob:mock-${++urls}`);
  URL.revokeObjectURL = vi.fn();
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => { root.unmount(); });
  host.remove();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const flush = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

describe('payload + filenames', () => {
  it('builds frame-accurate payloads from the job request and parameters', () => {
    expect(buildLivePayload({ request: null, time: 1, fps: 25, parameters: DEFAULT_PARAMETERS, restorerAvailable: true })).toBeNull();
    const payload = buildLivePayload({ request: REQUEST, time: 0.2, fps: 25, parameters: { ...DEFAULT_PARAMETERS, pose_strength: .8, restoration_blend: .5 }, maskBase64: 'data:image/png;base64,AA', restorerAvailable: true })!;
    expect(payload).toMatchObject({ frame_index: 5, face_mappings: REQUEST.face_mappings, frontalization_weight: .8, restorer: 'gpen', restoration_blend: .5,
      mask_base64: 'data:image/png;base64,AA', occlusion: { erosion_px: 0, feather_px: 5 }, bypass: [] });
    const retarget = buildLivePayload({ request: REQUEST, time: 0, fps: 25, parameters: { ...DEFAULT_PARAMETERS, pose_mode: 'retarget', restoration_blend: .5 }, restorerAvailable: false })!;
    expect(retarget.frontalization_weight).toBeUndefined();
    expect(retarget.restorer).toBe('off'); // GPEN weights unavailable -> never request it
    const bypassed = buildLivePayload({ request: REQUEST, time: 0, fps: 25, parameters: { ...DEFAULT_PARAMETERS, restoration_blend: .5, bypass: ['restorer'] }, restorerAvailable: true })!;
    expect(bypassed.restorer).toBe('off');
  });

  it('names downloads roop3d_preview_<ISO timestamp>.png without filesystem-hostile characters', () => {
    expect(previewFilename(new Date('2026-11-01T14:02:03.456Z'))).toBe('roop3d_preview_2026-11-01T14-02-03Z.png');
    expect(previewFilename()).toMatch(/^roop3d_preview_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z\.png$/);
  });

  it('downloads through a blob URL created in the click handler', () => {
    const clicked: HTMLAnchorElement[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) { clicked.push(this); });
    downloadBlob(png(), 'roop3d_preview_x.png');
    expect(clicked).toHaveLength(1);
    expect(clicked[0].download).toBe('roop3d_preview_x.png');
    expect(clicked[0].href).toBe('blob:mock-1');
    expect(document.body.querySelector('a[download]')).toBeNull(); // temporary anchor removed
    vi.advanceTimersByTime(10_001);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-1');
  });
});

function Probe({ payload, out }: { payload: LivePayload | null; out: { current: LivePreview | null } }) {
  out.current = useLiveSwapPreview(payload);
  return null;
}

describe('useLiveSwapPreview', () => {
  const mount = async (payload: LivePayload | null, out: { current: LivePreview | null }) => act(async () => { root.render(<Probe payload={payload} out={out}/>); });

  it('debounces slider changes into one POST /api/preview carrying the latest values', async () => {
    fetchMock.mockImplementation(async () => swapped());
    const out = { current: null as LivePreview | null };
    await mount({ frame_index: 1, swap_weight: .1 }, out);
    await flush(LIVE_DEBOUNCE_MS - 50);
    await mount({ frame_index: 1, swap_weight: .5 }, out);
    await flush(LIVE_DEBOUNCE_MS - 50);
    await mount({ frame_index: 1, swap_weight: .9 }, out);
    expect(fetchMock).not.toHaveBeenCalled();
    await flush(LIVE_DEBOUNCE_MS);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/preview');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ frame_index: 1, swap_weight: .9 });
    expect(out.current).toMatchObject({ status: 'ready', url: 'blob:mock-1', cached: false, stale: false });
    expect(out.current!.meta?.unsupported).toEqual(['tps_expression_weight']);
  });

  it('treats 503 busy as a status, keeps the last image, and re-sends after retry_after_ms', async () => {
    fetchMock.mockImplementationOnce(async () => swapped())
      .mockImplementationOnce(async () => busy())
      .mockImplementationOnce(async () => busy('TENSORRT_COMPILATION'))
      .mockImplementation(async () => swapped('HIT'));
    const out = { current: null as LivePreview | null };
    await mount({ frame_index: 1 }, out);
    await flush(LIVE_DEBOUNCE_MS);
    expect(out.current!.status).toBe('ready');

    await mount({ frame_index: 2 }, out);
    await flush(LIVE_DEBOUNCE_MS);
    expect(out.current).toMatchObject({ status: 'busy', stage: 'STAGE_1_SPARSE_PRESCAN', url: 'blob:mock-1', stale: true, error: null });
    expect(fetchMock).toHaveBeenCalledTimes(2);

    await flush(999);
    expect(fetchMock).toHaveBeenCalledTimes(2); // not before retry_after_ms
    await flush(1);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(out.current).toMatchObject({ status: 'busy', stage: 'TENSORRT_COMPILATION' });

    await flush(1000);
    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(out.current).toMatchObject({ status: 'ready', cached: true, stale: false, url: 'blob:mock-2' });
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-1'); // replaced image released
  });

  it('aborts a superseded request and never applies its late answer', async () => {
    const signals: AbortSignal[] = [];
    fetchMock.mockImplementation((_url: string, init: RequestInit) => new Promise((resolve, reject) => {
      signals.push(init.signal!);
      init.signal!.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
      setTimeout(() => resolve(swapped()), 500);
    }));
    const out = { current: null as LivePreview | null };
    await mount({ frame_index: 1 }, out);
    await flush(LIVE_DEBOUNCE_MS);
    await mount({ frame_index: 2 }, out);
    expect(signals[0].aborted).toBe(true);
    await flush(LIVE_DEBOUNCE_MS + 600);
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ frame_index: 2 });
    expect(out.current).toMatchObject({ status: 'ready', url: 'blob:mock-1' });
    expect(out.current!.error).toBeNull();
  });

  it('surfaces real errors with a retry, and clears everything when switched off', async () => {
    fetchMock.mockImplementationOnce(async () => new Response(JSON.stringify({ error: { code: 'X', message: 'No mapped face is visible at this playhead frame' } }), { status: 422 }))
      .mockImplementation(async () => swapped());
    const out = { current: null as LivePreview | null };
    await mount({ frame_index: 1 }, out);
    await flush(LIVE_DEBOUNCE_MS);
    expect(out.current).toMatchObject({ status: 'error', error: 'No mapped face is visible at this playhead frame' });
    await act(async () => { out.current!.retry(); });
    await flush(LIVE_DEBOUNCE_MS);
    expect(out.current!.status).toBe('ready');
    await mount(null, out);
    expect(out.current).toMatchObject({ status: 'idle', url: null, blob: null });
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-1');
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('treats HTTP 204 or cancelled response benignly and maintains stale preview without error', async () => {
    fetchMock.mockImplementationOnce(async () => swapped())
      .mockImplementationOnce(async () => new Response(null, { status: 204, headers: { 'X-Preview-Status': 'cancelled' } }));
    const out = { current: null as LivePreview | null };
    await mount({ frame_index: 1 }, out);
    await flush(LIVE_DEBOUNCE_MS);
    expect(out.current).toMatchObject({ status: 'ready', url: 'blob:mock-1', stale: false });

    await mount({ frame_index: 2 }, out);
    await flush(LIVE_DEBOUNCE_MS);
    expect(out.current).toMatchObject({ status: 'ready', url: 'blob:mock-1', stale: true, error: null });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('restarts the request when a render is queued so the server can yield the GPU', async () => {
    fetchMock.mockImplementation(async () => swapped());
    const out = { current: null as LivePreview | null };
    await mount({ frame_index: 1 }, out);
    await flush(LIVE_DEBOUNCE_MS);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await act(async () => { window.dispatchEvent(new Event(JOB_START_EVENT)); });
    await flush(LIVE_DEBOUNCE_MS);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

// Source-frame requests (`/media/preview-frame`) share the mocked fetch; answer them 404 so only /api/preview yields blobs.
const routed = (preview: () => Promise<Response>) => async (url: string) =>
  String(url).startsWith('/api/preview') ? preview() : new Response('', { status: 404 });

describe('VideoViewport live swap', () => {
  const props = { target: 'roop_3d/.runtime/media/t.mp4', output: '', time: 0.2, setTime: () => {}, duration: 4, enabled: true, keys: [] as number[],
    previewRequest: REQUEST, restorerAvailable: false };

  it('shows the swapped frame in the output pane, a busy indicator on 503, and downloads PNG via blob', async () => {
    fetchMock.mockImplementation(routed(async () => swapped()));
    const clicked: HTMLAnchorElement[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) { clicked.push(this); });
    await act(async () => { root.render(<VideoViewport {...props}/>); });
    await flush(LIVE_DEBOUNCE_MS + 50);

    const previewCalls = fetchMock.mock.calls.filter(([url]) => url === '/api/preview');
    expect(previewCalls).toHaveLength(1);
    expect(JSON.parse(previewCalls[0][1].body)).toMatchObject({ timestamp: .2, face_mappings: REQUEST.face_mappings, restorer: 'off' });
    const image = host.querySelector('.processed img.live-swap-preview') as HTMLImageElement;
    expect(image.src).toBe('blob:mock-1');
    expect(host.querySelector('.processed .pane-label')!.textContent).toContain('LIVE SWAP');
    expect(host.querySelector('.processed .viewport-empty')).toBeNull(); // no placeholder behind a real swap
    expect(host.querySelector('.live-swap-status.ready')!.textContent).toContain('Live');

    const download = host.querySelector('.live-download-btn') as HTMLButtonElement;
    expect(download.disabled).toBe(false);
    await act(async () => { download.click(); });
    expect(clicked).toHaveLength(1);
    expect(clicked[0].download).toMatch(/^roop3d_preview_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z\.png$/);
    expect(clicked[0].href).toMatch(/^blob:/);
  });

  it('keeps the previous swap visible and dimmed while the GPU is busy', async () => {
    let calls = 0;
    fetchMock.mockImplementation(routed(async () => (++calls === 1 ? swapped() : busy())));
    await act(async () => { root.render(<VideoViewport {...props}/>); });
    await flush(LIVE_DEBOUNCE_MS + 50);
    await act(async () => { root.render(<VideoViewport {...props} time={0.4}/>); });
    await flush(LIVE_DEBOUNCE_MS + 50);
    const status = host.querySelector('.live-swap-status.busy')!;
    expect(status.textContent).toContain('GPU busy');
    expect(status.textContent).toContain('stage 1');
    expect(host.querySelector('img.live-swap-preview.stale')).not.toBeNull();
  });

  it('requests nothing when live swap is toggled off or there is no job request', async () => {
    fetchMock.mockImplementation(routed(async () => swapped()));
    await act(async () => { root.render(<VideoViewport {...props} previewRequest={null}/>); });
    await flush(LIVE_DEBOUNCE_MS * 3);
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/preview')).toHaveLength(0);
    await act(async () => { root.render(<VideoViewport {...props}/>); });
    await flush(LIVE_DEBOUNCE_MS + 50);
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/preview')).toHaveLength(1);
    await act(async () => { (host.querySelector('.live-swap-toggle') as HTMLButtonElement).click(); });
    await flush(LIVE_DEBOUNCE_MS * 3);
    expect(host.querySelector('img.live-swap-preview')).toBeNull();
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/preview')).toHaveLength(1);
  });
});
