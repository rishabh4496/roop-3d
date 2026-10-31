// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import { VideoViewport } from './VideoViewport';
import { DEFAULT_PARAMETERS } from './components/workstation/pipeline';
import type { JobRequest } from './contracts';

vi.mock('./components/canvas/FaceOverlayCanvas', () => ({ default: () => null }));
vi.mock('./components/canvas/WorkstationCanvas', () => ({ default: () => <div>Comparison canvas</div> }));
vi.mock('./components/canvas/MaskBrushOverlay', () => ({ default: () => null }));
vi.mock('./components/tray/FaceSelectionTray', () => ({ default: () => null }));
vi.mock('./components/timeline/TimelineWorkstation', () => ({ default: () => <div>Timeline</div>, parseFps: () => 25 }));

it('renders real solo buffers in the main viewport, returns to comparison, and discards stale frame results', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  vi.useFakeTimers();
  const replies: Array<{ resolve: (value: unknown) => void; signal: AbortSignal }> = [];
  const fetch = vi.fn((_url, init) => new Promise(resolve => replies.push({ resolve, signal: init.signal })));
  vi.stubGlobal('fetch', fetch);
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  const request: JobRequest = { target_path: 'local.mp4', inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }], processing: 'local', remote_consent: false, trim_start_seconds: null, trim_end_seconds: null };
  const props = { target: '', output: '', setTime: vi.fn(), duration: 3, enabled: false, keys: [], previewRequest: request, parameters: DEFAULT_PARAMETERS };
  try {
    await act(async () => root.render(<VideoViewport {...props} time={1}/>));
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Solo XSeg / Occluder Matte"]')!.click());
    await act(async () => vi.advanceTimersByTime(200)); expect(fetch).toHaveBeenCalledOnce();
    await act(async () => replies[0].resolve({ ok: true, json: async () => ({ buffers: { matte: 'data:image/jpeg;base64,AQ==' }, frame_index: 25, timestamp: 1, parameters: DEFAULT_PARAMETERS, notes: [] }) }));
    expect(host.querySelector('.solo-viewport img')?.getAttribute('src')).toBe('data:image/jpeg;base64,AQ==');
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Solo XSeg / Occluder Matte"]')!.click());
    expect(host.querySelector('.solo-viewport')).toBeNull(); expect(host.textContent).toContain('Comparison canvas');
    await act(async () => root.render(<VideoViewport {...props} time={2}/>));
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Solo Face Swapper"]')!.click());
    await act(async () => vi.advanceTimersByTime(200));
    await act(async () => root.render(<VideoViewport {...props} time={3}/>)); expect(replies[1].signal.aborted).toBe(true);
    await act(async () => replies[1].resolve({ ok: true, json: async () => ({ buffers: { swapper: 'stale' }, notes: [] }) }));
    expect(host.querySelector('.solo-viewport img')).toBeNull();
  } finally { await act(async () => root.unmount()); host.remove(); vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); }
});
