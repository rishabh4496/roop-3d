// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { VideoViewport } from './VideoViewport';
import { extractFaceFromFrame } from './api';

vi.mock('./components/canvas/FaceOverlayCanvas', () => ({ default: () => null }));
vi.mock('./components/canvas/WorkstationCanvas', () => ({ default: () => null }));
vi.mock('./components/canvas/MaskBrushOverlay', () => ({ default: () => null }));
vi.mock('./components/tray/FaceSelectionTray', () => ({ default: () => null }));
vi.mock('./components/timeline/TimelineWorkstation', () => ({ default: () => null, parseFps: () => 25 }));

const TARGET = 'roop_3d/.runtime/media/target.mp4';
const created = {
  destination: 'source_gallery',
  face: { face_id: 'face_abc', frame_index: 5, face_index: 1, pose: { yaw: 28, pitch: -16, roll: 0 }, pose_cell: 'right_down', embedding_dim: 512, thumbnail_base64: '' },
  active_source: { faceset_id: 'Frame_1a2b3c4d', name: 'Frame 5 · face 1' },
};
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });

let host: HTMLDivElement, root: Root, fetchMock: ReturnType<typeof vi.fn>;
beforeEach(() => {
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
  vi.useFakeTimers();
  fetchMock = vi.fn(async (url: string) => String(url).includes('/faces/extract_from_frame') ? json(created, 201) : new Response('', { status: 404 }));
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

const props = { target: TARGET, output: '', time: 0.2, setTime: () => {}, duration: 4, enabled: true, keys: [] as number[] };
const calls = () => fetchMock.mock.calls.filter(([url]) => String(url).includes('/faces/extract_from_frame'));
const flush = (ms = 0) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

describe('Use Face From Frame', () => {
  it('posts the current frame and face number, then hands the new FaceSet to the workspace', async () => {
    const onSourceExtracted = vi.fn();
    await act(async () => { root.render(<VideoViewport {...props} onSourceExtracted={onSourceExtracted}/>); });
    const number = host.querySelector('.use-as-source-group input') as HTMLInputElement;
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
      setter.call(number, '2');
      number.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => { (host.querySelector('.use-as-source-btn') as HTMLButtonElement).click(); });
    await flush();
    expect(calls()).toHaveLength(1);
    const [url, init] = calls()[0];
    expect(url).toBe('/api/v1/faces/extract_from_frame');
    expect(JSON.parse(init.body)).toEqual({ target_media_path: TARGET, frame_index: 5, face_index: 1, destination: 'source_gallery' }); // 0.2 s at 25 fps; "face 2" is index 1
    expect(onSourceExtracted).toHaveBeenCalledWith({ faceset_id: 'Frame_1a2b3c4d', name: 'Frame 5 · face 1', item_count: 1 });
    expect(host.querySelector('.face-toast')!.textContent).toContain('right / down');
  });

  it('reports a failed extraction without selecting anything', async () => {
    fetchMock.mockImplementation(async () => json({ error: { code: 'FACE_NOT_FOUND', message: 'Frame has 0 detected face(s); face_index 0 is out of range' } }, 422));
    const onSourceExtracted = vi.fn();
    await act(async () => { root.render(<VideoViewport {...props} onSourceExtracted={onSourceExtracted}/>); });
    await act(async () => { (host.querySelector('.use-as-source-btn') as HTMLButtonElement).click(); });
    await flush();
    expect(onSourceExtracted).not.toHaveBeenCalled();
    expect(host.querySelector('.face-toast.error')!.textContent).toContain('0 detected face');
    await flush(4100);
    expect(host.querySelector('.face-toast')).toBeNull();
    expect((host.querySelector('.use-as-source-btn') as HTMLButtonElement).disabled).toBe(false);
  });

  it('is disabled without a target and the helper sends any destination', async () => {
    await act(async () => { root.render(<VideoViewport {...props} target=""/>); });
    expect((host.querySelector('.use-as-source-btn') as HTMLButtonElement).disabled).toBe(true);
    await extractFaceFromFrame({ target_media_path: TARGET, frame_index: 3, face_index: 0, destination: 'active_faceset', faceset_name: 'Person' });
    expect(JSON.parse(calls()[0][1].body)).toEqual({ target_media_path: TARGET, frame_index: 3, face_index: 0, destination: 'active_faceset', faceset_name: 'Person' });
  });
});
