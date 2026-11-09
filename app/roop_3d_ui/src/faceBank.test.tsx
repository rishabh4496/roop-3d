// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AngleCaptureHud, { hudSummary } from './components/tray/AngleCaptureHud';
import FrameFaceStrip from './components/tray/FrameFaceStrip';
import { useTargetLock } from './components/canvas/useTargetLock';
import { matchFrameFaces } from './utils/frameFaces';
import { FACE_COLORS, faceColor, faceTag, angleLabel } from './utils/faceIdentity';
import type { AngleBucket, TargetLock } from './api';
import type { FaceSet } from './contracts';
import type { FrameFace } from './frameScrub';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const bucket = (yaw: number, pitch: number, patch: Partial<AngleBucket> = {}): AngleBucket => ({
  bucket: [yaw, pitch], quality: 250, quality_locked: true, has_crop: true, yaw: (yaw - 2) * 30, pitch: (pitch - 1) * 30, roll: 0,
  frame_index: yaw * 10 + pitch, source: 'auto', match_score: 0.7, ...patch,
});
const allCells = () => [0, 1, 2, 3, 4].flatMap(yaw => [0, 1, 2].map(pitch => bucket(yaw, pitch)));
const lockOf = (faceId: number, patch: Partial<TargetLock> = {}): TargetLock => ({
  lock_id: String(faceId).repeat(32), target_number: faceId + 1, label: `face ${faceId}`, face_id: faceId, media_path: 'm.mp4', cluster_id: faceId + 1,
  anchor_frame_index: 5, bank_size: 3, observed_frames: [5], reference_bank: [], track_id: `track_000${faceId}`,
  coverage: { captured: 3, quality_locked: 3, total: 15, buckets: [bucket(2, 1, { source: 'anchor', match_score: 1 }), bucket(1, 1), bucket(3, 2, { match_score: 0.55 })] },
  scan: { state: 'running', sampled: 40, requested: 200 }, ...patch,
});
const faceOf = (rank: number, persistent: number, x: number): FrameFace => ({
  face_index: rank, track_id: `track_000${persistent}`, persistent_face_id: persistent, bbox: [x, 100, x + 100, 220], bbox_norm: [x / 1000, 0.2, (x + 100) / 1000, 0.44],
  kps_5: [[x + 20, 120], [x + 80, 120], [x + 50, 150], [x + 25, 200], [x + 75, 200]], pose: { yaw: 0, pitch: 0, roll: 0 }, pose3d: { yaw: 0, pitch: 0, roll: 0 }, confidence: 0.9,
});
const frame = { frame_index: 10, width: 1000, height: 500 };
const sources = [{ faceset_id: 'fs_a', name: 'Identity A' }, { faceset_id: 'fs_b', name: 'Identity B' }] as FaceSet[];

describe('faceIdentity', () => {
  it('gives each persistent id its own colour and an upper-case tag; unidentified faces fall back to the rank', () => {
    expect(new Set([0, 1, 2].map(faceColor)).size).toBe(3);
    expect(faceColor(FACE_COLORS.length)).toBe(faceColor(0));                 // wraps instead of failing
    expect(faceColor(null)).toBe('#9aa7ab');
    expect(faceTag(0)).toBe('FACE 0'); expect(faceTag(null, 1)).toBe('#2'); expect(faceTag(undefined)).toBe('FACE ?');
    expect(angleLabel(0, 2)).toBe('L Profile · Pitch Up');
  });
});

describe('hudSummary', () => {
  it('reports each scan state only from the server fields', () => {
    expect(hudSummary(lockOf(0)).title).toBe('Capturing Target Angles…');
    expect(hudSummary(lockOf(0)).detail).toContain('scanned 40 / 200 frames');
    const full = lockOf(0, { scan: { state: 'done' }, coverage: { captured: 15, quality_locked: 15, total: 15, buckets: allCells() } });
    expect(hudSummary(full)).toMatchObject({ state: 'complete', title: 'Angles Locked (100% Coverage)', fraction: 1 });
    const lowQuality = lockOf(0, { scan: { state: 'done' }, coverage: { captured: 15, quality_locked: 12, total: 15 } });
    expect(hudSummary(lowQuality).detail).toContain('3 still low quality');
    const partial = lockOf(0, { scan: { state: 'done', exhaustive: true }, coverage: { captured: 9, quality_locked: 9, total: 15 } });
    expect(hudSummary(partial)).toMatchObject({ state: 'partial', title: '9/15 angles locked (60% coverage)' });          // never claims 100%
    expect(hudSummary(partial).detail).toContain('do not occur in it');
    expect(hudSummary(lockOf(0, { scan: { state: 'done', skipped: 4 }, coverage: { captured: 9, quality_locked: 9, total: 15 } })).detail).toContain('4 frames could not be read');
    expect(hudSummary(lockOf(0, { scan: { state: 'error', message: 'decoder died' } }))).toMatchObject({ state: 'failed', detail: 'decoder died' });
  });
});

describe('Target Face Bank UI', () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); });
  const render = (node: React.ReactNode) => act(async () => root.render(node));

  it('fills the 5 x 3 grid with the crops of captured angles only, and Re-Scan is disabled while scanning', async () => {
    const onRescan = vi.fn();
    await render(<AngleCaptureHud lock={lockOf(0)} onRescan={onRescan} />);
    expect(host.querySelectorAll('[role=gridcell]')).toHaveLength(15);
    const filled = [...host.querySelectorAll<HTMLElement>('[role=gridcell][data-captured=true]')];
    expect(filled.map(cell => cell.dataset.bucket).sort()).toEqual(['1,1', '2,1', '3,2']);
    expect(filled[0].querySelector('img')!.getAttribute('src')).toMatch(/^\/api\/v1\/target\/0{32}\/angle\/\d\/\d\/crop\.jpg\?v=/);
    expect(host.querySelector('[data-testid=angle-hud]')!.getAttribute('data-state')).toBe('capturing');
    expect(host.querySelector('svg.angle-ring')!.getAttribute('aria-label')).toBe('3 of 15 angles captured');
    const rescan = [...host.querySelectorAll('button')].find(item => item.textContent!.includes('Scanning'))!;
    expect(rescan.disabled).toBe(true);

    await render(<AngleCaptureHud lock={lockOf(0, { scan: { state: 'done', exhaustive: true } })} onRescan={onRescan} />);
    await act(async () => [...host.querySelectorAll('button')].find(item => item.textContent === 'Re-Scan All Angles')!.click());
    expect(onRescan).toHaveBeenCalledWith('0'.repeat(32));
  });

  it('shows the green 100% badge only for a complete grid', async () => {
    await render(<AngleCaptureHud lock={lockOf(0, { scan: { state: 'done' }, coverage: { captured: 15, quality_locked: 15, total: 15, buckets: allCells() } })} />);
    expect(host.querySelector('.angle-hud-title.complete')!.textContent).toContain('Angles Locked (100% Coverage)');
    expect(host.querySelectorAll('[data-captured=true]')).toHaveLength(15);
  });

  it('puts a colour-coded FACE pill on every card, with a padlock and the HUD inside the locked face 0 card only', async () => {
    const faces = [faceOf(0, 0, 100), faceOf(1, 1, 400), faceOf(2, 2, 700)];
    const matched = matchFrameFaces(faces, frame, [], null).map(match => ({ ...match, locked: match.face.persistent_face_id === 0 }));
    await render(<FrameFaceStrip imageUrl="" meta={{ ...frame, frame_index: 10 } as never} faces={matched} detecting={false} lockingId={null} onLock={() => {}} locks={[lockOf(0)]} sources={sources} />);
    const items = [...host.querySelectorAll<HTMLElement>('.face-bank-item')];
    expect(items).toHaveLength(3);
    expect(items.map(item => item.querySelector('.face-id-pill')!.textContent)).toEqual(['face 0', 'face 1', 'face 2']);        // CSS upper-cases it
    expect(new Set(items.map(item => item.style.getPropertyValue('--face-color'))).size).toBe(3);
    expect(items[0].querySelector('.face-lock-icon.locked')).not.toBeNull();
    expect(items[1].querySelector('.face-lock-icon.locked')).toBeNull();
    expect(items[0].querySelector('[data-testid=angle-hud]')).not.toBeNull();
    expect(items[1].querySelector('[data-testid=angle-hud]')).toBeNull();
    expect(host.querySelector('.eyebrow')!.textContent).toBe('TARGET FACE BANK');
  });

  it('binds a source identity to the locked face (by its track), and only once the face is locked', async () => {
    const onAssign = vi.fn(), onLock = vi.fn();
    const faces = [faceOf(0, 0, 100), faceOf(1, 1, 400)];
    const matched = matchFrameFaces(faces, frame, [], null).map(match => ({ ...match, locked: match.face.persistent_face_id === 0 }));
    await render(<FrameFaceStrip imageUrl="" meta={null} faces={matched} detecting={false} lockingId={null} onLock={onLock} locks={[lockOf(0)]} sources={sources}
      assignments={{ track_0001: 'fs_b' }} onAssign={onAssign} />);
    const select = (id: number) => host.querySelector<HTMLSelectElement>(`select[aria-label="Source identity for face ${id}"]`)!;
    expect(select(1).disabled).toBe(true);                                            // not locked: lock it first
    expect(select(1).value).toBe('fs_b');                                             // but an existing binding is still shown
    expect(select(0).disabled).toBe(false);
    await act(async () => { select(0).value = 'fs_a'; select(0).dispatchEvent(new Event('change', { bubbles: true })); });
    expect(onAssign).toHaveBeenCalledWith('track_0000', 'fs_a', 0);
    expect(onLock).not.toHaveBeenCalled();                                            // choosing a source is not a lock toggle
    await act(async () => host.querySelector<HTMLButtonElement>('.frame-face-card[data-face-rank="1"]')!.click());
    expect(onLock.mock.calls[0][0].face.persistent_face_id).toBe(1);
  });

  it('keeps a locked face that is not in the frame in the bank, so its capture progress and unlock stay reachable', async () => {
    const onUnlock = vi.fn(), onRescan = vi.fn();
    await render(<FrameFaceStrip imageUrl="" meta={null} faces={[]} detecting={false} lockingId={null} onLock={() => {}} locks={[lockOf(2, { scan: { state: 'done' } })]}
      onUnlock={onUnlock} onRescan={onRescan} />);
    const item = host.querySelector<HTMLElement>('.face-bank-item.off-frame')!;
    expect(item.textContent).toContain('Not in this frame');
    expect(item.querySelector('[data-testid=angle-hud]')).not.toBeNull();
    await act(async () => item.querySelector<HTMLButtonElement>('button[aria-label="Unlock face 2"]')!.click());
    expect(onUnlock).toHaveBeenCalledWith('2'.repeat(32));
    await act(async () => [...item.querySelectorAll('button')].find(b => b.textContent === 'Re-Scan All Angles')!.click());
    expect(onRescan).toHaveBeenCalledWith('2'.repeat(32));
  });

  it('lists every captured crop with its measured similarity when a locked face is hovered, and closes on leave', async () => {
    const matched = matchFrameFaces([faceOf(0, 0, 100)], frame, [], null).map(match => ({ ...match, locked: true }));
    await render(<FrameFaceStrip imageUrl="" meta={null} faces={matched} detecting={false} lockingId={null} onLock={() => {}} locks={[lockOf(0, { scan: { state: 'done' } })]} />);
    expect(document.querySelector('[data-testid=angle-inspector]')).toBeNull();
    const item = host.querySelector<HTMLElement>('.face-bank-item')!;
    await act(async () => { item.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, relatedTarget: document.body })); });
    const inspector = document.querySelector('[data-testid=angle-inspector]')!;
    expect(inspector).not.toBeNull();
    expect(inspector.querySelectorAll('figure')).toHaveLength(3);
    expect(inspector.textContent).toContain('Every crop was assigned to face 0');
    expect(inspector.textContent).toContain('0.55 – 0.70');                           // auto crops only: the anchor's 1.0 is not a measurement
    expect(inspector.textContent).toContain('anchor (your click)');
    await act(async () => { item.dispatchEvent(new MouseEvent('mouseout', { bubbles: true, relatedTarget: document.body })); });
    expect(document.querySelector('[data-testid=angle-inspector]')).toBeNull();
  });
});

describe('useTargetLock with several locked faces', () => {
  let host: HTMLDivElement;
  beforeEach(() => { vi.useFakeTimers(); host = document.createElement('div'); document.body.append(host); });
  afterEach(() => { host.remove(); vi.useRealTimers(); });
  function mount(options: Parameters<typeof useTargetLock>[0]) {
    const box: { current: ReturnType<typeof useTargetLock> | null } = { current: null };
    function Probe() { box.current = useTargetLock(options); return null; }
    act(() => createRoot(host).render(<Probe />));
    return box;
  }

  it('holds face 0 and face 1 at once, unlocks one without touching the other, and keeps polling the one that is scanning', async () => {
    const results = [lockOf(0), lockOf(1)];
    const lock = vi.fn(async () => results.shift()!);
    const unlock = vi.fn(async (id: string) => ({ lock_id: id, face_id: 0, unlocked: true }));
    const refresh = vi.fn(async (id: string) => lockOf(Number(id[0]), { scan: { state: 'running', sampled: 90, requested: 200 } }));
    const box = mount({ target: 'm.mp4', frameIndex: 0, lock, unlock, refresh, pollMs: 100 });
    await act(async () => { await box.current!.lockFace(0); });
    await act(async () => { await box.current!.lockFace(1); });
    expect(box.current!.locks.map(item => item.face_id)).toEqual([0, 1]);
    expect(box.current!.lock?.face_id).toBe(1);                                       // the most recently locked is the primary

    await act(async () => { await vi.advanceTimersByTimeAsync(250); });
    expect(refresh.mock.calls.map(call => call[0][0])).toContain('0');                // the first face's scan is still being followed
    expect(box.current!.locks[0].scan.sampled).toBe(90);

    await act(async () => { await box.current!.unlockFace('1'.repeat(32)); });
    expect(unlock).toHaveBeenCalledWith('1'.repeat(32));
    expect(box.current!.locks.map(item => item.face_id)).toEqual([0]);
    expect(box.current!.lock?.face_id).toBe(0);
    const seen = refresh.mock.calls.length;
    await act(async () => { await vi.advanceTimersByTimeAsync(300); });
    expect(refresh.mock.calls.slice(seen).every(call => call[0][0] === '0')).toBe(true);   // no polling, and no resurrection, of the unlocked face
    expect(box.current!.locks.map(item => item.face_id)).toEqual([0]);
  });

  it('re-scans through the server, then follows the new scan; a failed unlock keeps the lock and says why', async () => {
    const rescan = vi.fn(async () => lockOf(0, { scan: { state: 'running', sampled: 0, requested: 200 }, coverage: { captured: 1, quality_locked: 1, total: 15, buckets: [] } }));
    const refresh = vi.fn(async () => lockOf(0, { scan: { state: 'done' } }));
    const unlock = vi.fn(async () => { throw new Error('Unknown target lock'); });
    const box = mount({ target: 'm.mp4', frameIndex: 0, lock: vi.fn(async () => lockOf(0, { scan: { state: 'done' } })), rescan, refresh, unlock, pollMs: 100 });
    await act(async () => { await box.current!.lockFace(0); });
    await act(async () => { await box.current!.rescanAngles('0'.repeat(32)); });
    expect(rescan).toHaveBeenCalledWith('0'.repeat(32));
    expect(box.current!.locks[0].coverage!.captured).toBe(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(150); });
    expect(box.current!.locks[0].scan.state).toBe('done');
    await act(async () => { await box.current!.unlockFace('0'.repeat(32)); });
    expect(box.current!.locks).toHaveLength(1);
    expect(box.current!.toast).toEqual({ kind: 'error', text: 'Unknown target lock' });
  });
});
