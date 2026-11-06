// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { FrameScrubber, filmstripCount, fetchFrame, type FrameFace, type FrameMeta } from './frameScrub';
import { bestOverlap, cropRect, iou, matchFrameFaces } from './utils/frameFaces';
import FrameFaceStrip from './components/tray/FrameFaceStrip';
import type { DetectedFace } from './contracts';

const meta = (index: number, faces?: FrameFace[]): FrameMeta => ({
  frame_index: index, requested_index: index, timestamp_sec: index / 25, width: 1000, height: 500, image_width: 1000, image_height: 500,
  total_frames: 1000, fps: 25, cache: 'miss', decode_ms: 3, ...(faces ? { faces } : {}),
});
const face = (rank: number, bbox: [number, number, number, number], yaw = 0): FrameFace => ({
  face_index: rank, track_id: null, bbox, bbox_norm: [bbox[0] / 1000, bbox[1] / 500, bbox[2] / 1000, bbox[3] / 500],
  kps_5: [[bbox[0] + 20, bbox[1] + 20], [bbox[2] - 20, bbox[1] + 20], [(bbox[0] + bbox[2]) / 2, bbox[1] + 40], [bbox[0] + 25, bbox[3] - 20], [bbox[2] - 25, bbox[3] - 20]],
  pose: { yaw, pitch: 5, roll: 0 }, pose3d: { yaw, pitch: -5, roll: 0 }, confidence: 0.9,
});

/** A fetcher whose responses the test resolves by hand, in the order it likes. */
function manualFetcher() {
  const pending: { index: number; detect: boolean; resolve: (faces?: FrameFace[]) => void; reject: (error: Error) => void; signal?: AbortSignal }[] = [];
  const fetcher = vi.fn((_path: string, index: number, options: { detect?: boolean; signal?: AbortSignal } = {}) =>
    new Promise<{ blob: Blob; meta: FrameMeta }>((resolve, reject) => {
      pending.push({ index, detect: !!options.detect, signal: options.signal, reject,
        resolve: faces => resolve({ blob: new Blob([`frame${index}`]), meta: meta(index, options.detect ? faces ?? [] : undefined) }) });
    }));
  return { fetcher: fetcher as unknown as typeof fetchFrame, pending };
}
const tick = () => act(async () => { await vi.advanceTimersByTimeAsync(0); });

describe('FrameScrubber: latest frame wins, one request in flight', () => {
  let urls = 0;
  const make = (fetcher: typeof fetchFrame, extra = {}) => new FrameScrubber(fetcher, {
    schedule: run => queueMicrotask(run), makeUrl: () => `blob:f${++urls}`, revokeUrl: vi.fn(), settleMs: 100, ...extra });
  beforeEach(() => { urls = 0; vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it('never has two requests in flight and always ends on the newest wanted frame', async () => {
    const { fetcher, pending } = manualFetcher();
    const scrubber = make(fetcher);
    scrubber.setTarget('a.mp4', 10, true, true); await tick();
    for (const index of [11, 12, 13, 14, 15]) { scrubber.setTarget('a.mp4', index, true, true); await tick(); }   // a drag: five moves while frame 10 is out
    expect(pending.map(item => item.index)).toEqual([10]);
    pending[0].resolve(); await tick();
    expect(scrubber.getState().shown?.index).toBe(10);                    // the early frame is shown at once ...
    expect(pending.map(item => item.index)).toEqual([10, 15]);            // ... and the next request is the newest one: 11-14 are skipped
    pending[1].resolve(); await tick();
    expect(scrubber.getState().shown?.index).toBe(15);
    expect(scrubber.requests.every(request => !request.detect)).toBe(true);   // no detection while the playhead is moving
  });

  it('asks for faces only once the playhead rests, for the frame already on screen', async () => {
    const { fetcher, pending } = manualFetcher();
    const scrubber = make(fetcher);
    scrubber.setTarget('a.mp4', 40, true, true); await tick();
    pending[0].resolve(); await tick();
    expect(scrubber.getState().shown?.faces).toBeNull();
    expect(pending).toHaveLength(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(110); });
    expect(pending.map(item => [item.index, item.detect])).toEqual([[40, false], [40, true]]);
    pending[1].resolve([face(0, [100, 100, 200, 220])]); await tick();
    expect(scrubber.getState().shown?.faces).toHaveLength(1);
    expect(scrubber.getState().shown?.meta.faces?.[0].bbox).toEqual([100, 100, 200, 220]);
  });

  it('serves frames it already has from memory and does not ask the server again', async () => {
    const { fetcher, pending } = manualFetcher();
    const scrubber = make(fetcher);
    scrubber.setTarget('a.mp4', 1, true, false); await tick(); pending[0].resolve(); await tick();
    scrubber.setTarget('a.mp4', 2, true, false); await tick(); pending[1].resolve(); await tick();
    scrubber.setTarget('a.mp4', 1, true, false); await tick();
    expect(scrubber.getState().shown?.index).toBe(1);
    expect(pending).toHaveLength(2);
    expect(scrubber.requests.length).toBe(2);
  });

  it('does not request detection when it is off, nor anything while playback is running', async () => {
    const { fetcher, pending } = manualFetcher();
    const scrubber = make(fetcher);
    scrubber.setTarget('a.mp4', 5, true, false); await tick(); pending[0].resolve(); await tick();
    await act(async () => { await vi.advanceTimersByTimeAsync(300); });
    expect(pending).toHaveLength(1);
    scrubber.setTarget('a.mp4', 6, false, false); await tick();
    expect(pending).toHaveLength(1);
  });

  it('forgets everything when the clip changes, and ignores a response that arrives late', async () => {
    const { fetcher, pending } = manualFetcher();
    const revoke = vi.fn();
    const scrubber = make(fetcher, { revokeUrl: revoke });
    scrubber.setTarget('a.mp4', 1, true, false); await tick();
    scrubber.setTarget('b.mp4', 1, true, false); await tick();
    expect(pending[0].signal?.aborted).toBe(true);
    pending[0].resolve(); await tick();
    expect(scrubber.getState().shown).toBeNull();                      // the old clip's frame never appears
    pending[1].resolve(); await tick();
    expect(scrubber.getState().shown?.index).toBe(1);
  });

  it('keeps the last good frame on an error and retries on request', async () => {
    const { fetcher, pending } = manualFetcher();
    const scrubber = make(fetcher);
    scrubber.setTarget('a.mp4', 1, true, false); await tick(); pending[0].resolve(); await tick();
    scrubber.setTarget('a.mp4', 2, true, false); await tick();
    pending[1].reject(new Error('Requested frame is beyond the video')); await tick();
    expect(scrubber.getState().shown?.index).toBe(1);
    expect(scrubber.getState().error).toBe('Requested frame is beyond the video');
    scrubber.retry(); await tick();
    expect(pending).toHaveLength(3);
  });

  it('bounds its frame cache and frees the evicted frames', async () => {
    const { fetcher, pending } = manualFetcher();
    const revoke = vi.fn();
    const scrubber = make(fetcher, { cacheSize: 3, revokeUrl: revoke });
    for (let index = 0; index < 6; index++) { scrubber.setTarget('a.mp4', index, true, false); await tick(); pending[index].resolve(); await tick(); }
    expect(revoke).toHaveBeenCalledTimes(3);
    scrubber.reset();
    expect(revoke).toHaveBeenCalledTimes(6);
  });
});

describe('fetchFrame', () => {
  afterEach(() => { vi.unstubAllGlobals(); });
  it('reads the frame metadata from the X-Frame-Meta header and sends the selectors', async () => {
    const fetchMock = vi.fn(async (_url: string) => new Response(new Blob(['jpeg']), { status: 200, headers: { 'X-Frame-Meta': JSON.stringify(meta(7, [face(0, [1, 2, 3, 4])])) } }));
    vi.stubGlobal('fetch', fetchMock);
    const result = await fetchFrame('roop_3d/.runtime/media/a.mp4', 7, { detect: true });
    expect(result.meta.faces).toHaveLength(1);
    const url = new URL(String(fetchMock.mock.calls[0][0]), 'http://x');
    expect(url.pathname).toBe('/api/v1/media/frame');
    expect(url.searchParams.get('frame_index')).toBe('7');
    expect(url.searchParams.get('detect_faces')).toBe('true');
  });
  it('turns an API error into its message', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ error: { message: 'Local video file does not exist' } }), { status: 404 })));
    await expect(fetchFrame('x.mp4', 0)).rejects.toThrow('Local video file does not exist');
  });
});

describe('filmstripCount', () => {
  it('is at least 16, one per ~120 px in steps of 8, and at most 64', () => {
    expect(filmstripCount(300)).toBe(16);
    expect(filmstripCount(1920)).toBe(16);
    expect(filmstripCount(2400)).toBe(24);
    expect(filmstripCount(100000)).toBe(64);
  });
});

describe('matching detected faces to stored target faces', () => {
  const stored = (id: string, clusterId: string, bbox: [number, number, number, number]): DetectedFace =>
    ({ id, clusterId, frameIndex: 10, bbox, pose: { yaw: 0, pitch: 0, roll: 0 } });
  const frame = { frame_index: 10, width: 1000, height: 500 };

  it('computes overlap', () => {
    expect(iou([0, 0, 1, 1], [0, 0, 1, 1])).toBe(1);
    expect(iou([0, 0, 1, 1], [2, 2, 3, 3])).toBe(0);
    expect(iou([0, 0, 2, 2], [1, 1, 3, 3])).toBeCloseTo(1 / 7, 5);
  });

  it('a detection over a stored face is that face; the rest are candidates with their own ids', () => {
    const faces = [face(0, [100, 100, 200, 220]), face(1, [600, 100, 700, 220])];
    const matches = matchFrameFaces(faces, frame, [stored('track_0000_10', 'track_0000', [0.1, 0.2, 0.2, 0.44])], null);
    expect(matches[0]).toMatchObject({ id: 'track_0000_10', clusterId: 'track_0000', locked: false });
    expect(matches[1]).toMatchObject({ id: 'frame:10:1', clusterId: null, stored: null, locked: false });
    expect(matches[1].landmarks5[2]).toEqual([0.65, 0.28]);       // native pixels -> fractions of the frame
  });

  it('the active face, and any face of the active track, is the locked target', () => {
    const faces = [face(0, [100, 100, 200, 220]), face(1, [600, 100, 700, 220])];
    const store = [stored('t0_10', 't0', [0.1, 0.2, 0.2, 0.44]), stored('t1_10', 't1', [0.6, 0.2, 0.7, 0.44])];
    const matches = matchFrameFaces(faces, frame, store, 't1_10');
    expect(matches.map(match => match.locked)).toEqual([false, true]);
  });

  it('one stored face is never matched to two detections', () => {
    const faces = [face(0, [100, 100, 200, 220]), face(1, [105, 100, 205, 220])];
    const matches = matchFrameFaces(faces, frame, [stored('t0_10', 't0', [0.1, 0.2, 0.2, 0.44])], null);
    expect(matches.filter(match => match.stored).length).toBe(1);
  });

  it('crops around a face with a margin and clamps to the frame; finds the same face in a server response', () => {
    expect(cropRect([100, 100, 200, 200], 1000, 500)).toEqual({ x: 70, y: 70, w: 160, h: 160 });
    expect(cropRect([0, 0, 100, 100], 1000, 500).x).toBe(0);
    expect(bestOverlap(face(0, [100, 100, 200, 220]), [{ bbox: [0.6, 0.2, 0.7, 0.44] }, { bbox: [0.1, 0.2, 0.21, 0.44] }])).toEqual({ bbox: [0.1, 0.2, 0.21, 0.44] });
    expect(bestOverlap(face(0, [100, 100, 200, 220]), [{ bbox: [0.8, 0.8, 0.9, 0.9] }])).toBeNull();
  });
});

describe('FrameFaceStrip', () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true; host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); });
  const render = (props: Partial<React.ComponentProps<typeof FrameFaceStrip>>) => act(async () => {
    root.render(<FrameFaceStrip imageUrl="" meta={meta(10)} faces={null} detecting={false} lockingId={null} onLock={() => {}} {...props} />);
  });
  const matches = (locked = false) => matchFrameFaces([face(0, [100, 100, 200, 220], -12), face(1, [600, 100, 700, 220], 30)], { frame_index: 10, width: 1000, height: 500 },
    locked ? [{ id: 't1_10', clusterId: 't1', frameIndex: 10, bbox: [0.6, 0.2, 0.7, 0.44], pose: { yaw: 0, pitch: 0, roll: 0 } }] : [], locked ? 't1_10' : null);

  it('says what it is waiting for, and shows nothing before detection has run', async () => {
    await render({});
    expect(host.querySelector('.frame-face-empty')!.textContent).toMatch(/when the playhead rests/);
    await render({ detecting: true });
    expect(host.querySelector('[role=status]')!.textContent).toBe('Detecting faces…');
    await render({ faces: [] });
    expect(host.querySelector('.frame-face-empty')!.textContent).toMatch(/Scrub to a frame/);
  });

  it('shows one card per face with its rank, pose and confidence; clicking a card locks that face', async () => {
    const onLock = vi.fn();
    await render({ faces: matches(), onLock });
    const cards = [...host.querySelectorAll<HTMLButtonElement>('.frame-face-card')];
    expect(cards).toHaveLength(2);
    expect(cards[0].textContent).toContain('#1');
    expect(cards[0].textContent).toContain('Y -12° P -5°');          // standard frame: the server's pose3d
    expect(cards[1].textContent).toContain('90%');
    await act(async () => { cards[1].click(); });
    expect(onLock).toHaveBeenCalledTimes(1);
    expect(onLock.mock.calls[0][0].face.face_index).toBe(1);
  });

  it('marks the target face TARGET LOCKED and disables a card while it is being locked', async () => {
    await render({ faces: matches(true) });
    const cards = [...host.querySelectorAll<HTMLButtonElement>('.frame-face-card')];
    expect(cards[0].getAttribute('aria-pressed')).toBe('false');
    expect(cards[1].getAttribute('aria-pressed')).toBe('true');
    expect(cards[1].textContent).toContain('TARGET LOCKED');
    await render({ faces: matches(), lockingId: 'frame:10:0' });
    expect(host.querySelector<HTMLButtonElement>('[data-face-rank="0"]')!.disabled).toBe(true);
    expect(host.querySelector('.frame-face-badge.pending')!.textContent).toBe('Locking…');
  });
});
