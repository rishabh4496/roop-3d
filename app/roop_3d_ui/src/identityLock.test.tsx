// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { IdentityLockBar, IdentityTrackStats, clock, lockProblem, rangeSummary, withheldSummary } from './components/facemgr/IdentityLockPanel';
import type { FaceSet, IdentityLock, IdentitySheetTrack, Inspection } from './contracts';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const track: IdentitySheetTrack = {
  track_id: 'track_0001', persistent_id: 1, first_frame: 0, last_frame: 99,
  segments: [[0, 24, 0, 0.96], [50, 99, 2, 3.96], [120, 130, 5, 5.4], [150, 160, 6, 6.4]], missing_ranges: [[25, 49], [100, 100]],
  visible_frames: 90, swappable_frames: 80, withheld_frames: 10, withheld_reasons: { AMBIGUOUS_IDENTITY: 8, APPEARANCE_LOW: 2 },
  state_frames: { TRACKED: 80, REACQUIRED: 3, UNCERTAIN: 7 }, confidence_mean: 0.912, confidence_min: 0.204, scene_ids: [0, 1],
  thumbnails: [{ frame: 3, timestamp_seconds: 0.1, jpeg_base64: 'AAAA' }, { frame: 60, timestamp_seconds: 2.4, jpeg_base64: 'BBBB' }],
};

const inspection = (method = 'identity_lock_v1'): Inspection => ({
  inspection_id: 'a'.repeat(32), target_sha256: 'b'.repeat(64), tracks: [], warnings: [], truncated: false, tracking_method: method });

const sources: FaceSet[] = [{ faceset_id: 'alice', name: 'Alice' } as FaceSet, { faceset_id: 'bob', name: 'Bob' } as FaceSet];

const lock = (overrides: Partial<IdentityLock> = {}): IdentityLock => ({
  lock_id: 'c'.repeat(32), revision: 2, digest: 'd'.repeat(64), target_path: 'media/t.mp4', target_sha256: 'e'.repeat(64),
  inspection_id: 'a'.repeat(32), assignments: [{ track_id: 'track_0001', faceset_id: 'alice', faceset_sha256: 'f'.repeat(64) }],
  excluded_tracks: [], status: { valid: true, reasons: [] }, ...overrides });

describe('contact-sheet formatting', () => {
  it('formats clock, visible ranges and withheld reasons', () => {
    expect(clock(0)).toBe('0:00');
    expect(clock(125.9)).toBe('2:05');
    expect(rangeSummary(track)).toBe('0:00–0:00, 0:02–0:03, 0:05–0:05 +1 more');
    expect(withheldSummary(track)).toBe('ambiguous identity ×8, appearance low ×2');
  });

  it('shows thumbnails, track number, confidence, gaps and withheld frames', () => {
    const html = renderToStaticMarkup(<IdentityTrackStats track={track} />);
    expect(html).toContain('data:image/jpeg;base64,AAAA');
    expect(html).toContain('Confidence 0.91 (min 0.20) · reacquired 3×');
    expect(html).toContain('Not detected: frames 25–49, 100');
    expect(html).toContain('10 frames kept original: ambiguous identity ×8');
    expect(html).not.toContain('cannot be swapped');
  });

  it('tells the user when a track has no verified frame and renders nothing without data', () => {
    expect(renderToStaticMarkup(<IdentityTrackStats track={{ ...track, swappable_frames: 0 }} />)).toContain('cannot be swapped');
    expect(renderToStaticMarkup(<IdentityTrackStats track={undefined} />)).toBe('');
  });
});

describe('lockProblem', () => {
  it('is quiet only for a valid lock that equals the on-screen mapping', () => {
    expect(lockProblem(null, { track_0001: 'alice' })).toBeNull();
    expect(lockProblem(lock(), { track_0001: 'alice' })).toBeNull();
    expect(lockProblem(lock(), { track_0001: 'alice', track_0002: '' })).toBeNull();
    expect(lockProblem(lock(), { track_0001: 'bob' })).toContain('differs');
    expect(lockProblem(lock(), { track_0001: 'alice', track_0002: 'bob' })).toContain('differs');
    expect(lockProblem(lock(), {})).toContain('differs');
    const stale = lock({ status: { valid: false, reasons: [{ code: 'FACESET_CHANGED', detail: 'x', track_id: 'track_0001' }] } });
    expect(lockProblem(stale, { track_0001: 'alice' })).toContain('FACESET_CHANGED');
  });
});

describe('IdentityLockBar', () => {
  let host: HTMLDivElement, root: Root, fetchMock: ReturnType<typeof vi.fn>;
  const onLock = vi.fn(), onError = vi.fn();

  const mount = async (props: Partial<Parameters<typeof IdentityLockBar>[0]> = {}) => {
    await act(async () => {
      root.render(<IdentityLockBar target="media/t.mp4" inspection={inspection()} mappings={{ track_0001: 'alice' }} sources={sources}
        lock={null} onLock={onLock} busy={false} onError={onError} {...props} />);
    });
  };

  beforeEach(() => {
    host = document.createElement('div'); document.body.appendChild(host); root = createRoot(host);
    onLock.mockReset(); onError.mockReset();
    fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method === 'POST') return new Response(JSON.stringify(lock()), { status: 201 });
      if (init?.method === 'DELETE') return new Response(null, { status: 204 });
      return new Response(JSON.stringify(lock()), { status: 200 });
    });
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

  it('explains that a legacy inspection cannot be locked and renders nothing without an inspection', async () => {
    await mount({ inspection: inspection('iou_arcface_hungarian') });
    expect(host.querySelector('[data-testid="identity-legacy"]')?.textContent).toContain('Re-inspect');
    await mount({ inspection: null });
    expect(host.innerHTML).toBe('');
  });

  it('locks exactly the chosen mapping on an explicit click', async () => {
    await mount({ mappings: { track_0001: 'alice', track_0002: '' } });
    const button = [...host.querySelectorAll('button')].find(item => item.textContent === 'Lock mapping')!;
    expect(button.disabled).toBe(false);
    await act(async () => { button.click(); });
    const [url, init] = fetchMock.mock.calls.find(call => call[1]?.method === 'POST')!;
    expect(url).toBe('/api/v1/identity/locks');
    expect(JSON.parse(init.body)).toEqual({ target_path: 'media/t.mp4', inspection_id: 'a'.repeat(32),
      assignments: [{ track_id: 'track_0001', faceset_id: 'alice' }] });
    expect(onLock).toHaveBeenCalledWith(lock());
  });

  it('cannot lock an empty mapping (nothing is swapped without an explicit choice)', async () => {
    await mount({ mappings: {} });
    expect([...host.querySelectorAll('button')].find(item => item.textContent === 'Lock mapping')!.disabled).toBe(true);
  });

  it('shows the lock, its digest and assignments, and offers release', async () => {
    await mount({ lock: lock() });
    expect(host.textContent).toContain('Identity locked');
    expect(host.textContent).toContain('Face 0001 → Alice');
    expect(host.textContent).toContain('d'.repeat(12));
    const release = [...host.querySelectorAll('button')].find(item => item.textContent === 'Release lock')!;
    await act(async () => { release.click(); });
    expect(fetchMock.mock.calls.some(call => call[1]?.method === 'DELETE')).toBe(true);
    expect(onLock).toHaveBeenLastCalledWith(null);
  });

  it('flags a stale lock with the exact reasons and a mapping that drifted from it', async () => {
    const stale = lock({ status: { valid: false, reasons: [{ code: 'TARGET_CHANGED', detail: 'The target video changed after it was inspected' }] } });
    fetchMock.mockImplementation(async () => new Response(JSON.stringify(stale), { status: 200 }));
    await mount({ lock: stale });
    expect(host.textContent).toContain('needs attention');
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('TARGET_CHANGED');
    expect(host.textContent).toContain('The target video changed after it was inspected');
    const relock = [...host.querySelectorAll('button')].find(item => item.textContent === 'Re-lock mapping')!;
    expect(relock.disabled).toBe(false);
    await mount({ lock: lock(), mappings: { track_0001: 'bob' } });
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('differs from the identity lock');
  });

  it('re-verifies the lock against the files on disk and adopts a newly stale status', async () => {
    const stale = lock({ status: { valid: false, reasons: [{ code: 'FACESET_CHANGED', detail: 'changed', track_id: 'track_0001' }] } });
    fetchMock.mockImplementation(async () => new Response(JSON.stringify(stale), { status: 200 }));
    await mount({ lock: lock() });
    expect(fetchMock.mock.calls.some(call => String(call[0]).includes('?verify=true'))).toBe(true);
    expect(onLock).toHaveBeenCalledWith(stale);
  });

  it('reports server refusals instead of pretending the lock was made', async () => {
    fetchMock.mockImplementation(async () => new Response(JSON.stringify({ error: { message: 'Local service returned 422' } }), { status: 422 }));
    await mount();
    await act(async () => { [...host.querySelectorAll('button')].find(item => item.textContent === 'Lock mapping')!.click(); });
    expect(onError).toHaveBeenCalled();
    expect(onLock).not.toHaveBeenCalled();
  });
});
