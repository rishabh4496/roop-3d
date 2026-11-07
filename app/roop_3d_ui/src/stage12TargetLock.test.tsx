// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useTargetLock } from './components/canvas/useTargetLock';
import { poseAxes, signed1 } from './components/canvas/FaceOverlayCanvas';
import type { TargetLock } from './api';

const lockOf = (patch: Partial<TargetLock> = {}): TargetLock => ({
  lock_id: 'a'.repeat(32), target_number: 1, label: 'Target #1', media_path: 'm.mp4', cluster_id: 1, anchor_frame_index: 5, bank_size: 1,
  observed_frames: [5], reference_bank: [], scan: { state: 'done' }, ...patch });

describe('Stage 12 pose pill and axes', () => {
  it('formats one decimal with an explicit sign and never -0.0', () => {
    expect(signed1(-34.24)).toBe('-34.2');
    expect(signed1(6.15)).toBe('+6.2');
    expect(signed1(-0.04)).toBe('+0.0');
    expect(`#1 · Y: ${signed1(-34.2)}° P: ${signed1(6.1)}° R: ${signed1(-2)}°`).toBe('#1 · Y: -34.2° P: +6.1° R: -2.0°');
  });
  it('points the facing axis toward image right for + yaw, up for + pitch, and rotates the roll axes', () => {
    const turned = poseAxes({ yaw: 60, pitch: 0, roll: 0 });
    expect(turned.z[0]).toBeGreaterThan(0.8); expect(turned.z[1]).toBeCloseTo(0, 9);
    expect(poseAxes({ yaw: 0, pitch: 30, roll: 0 }).z[1]).toBeLessThan(0);                 // screen y is down: looking up is negative
    const rolled = poseAxes({ yaw: 0, pitch: 0, roll: 90 });
    expect(rolled.x[0]).toBeCloseTo(0, 9); expect(rolled.x[1]).toBeCloseTo(1, 9);
  });
});

describe('useTargetLock', () => {
  let host: HTMLDivElement;
  beforeEach(() => { (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true; vi.useFakeTimers(); host = document.createElement('div'); document.body.append(host); });
  afterEach(() => { host.remove(); vi.useRealTimers(); });

  function mount(options: Parameters<typeof useTargetLock>[0]) {
    const box: { current: ReturnType<typeof useTargetLock> | null } = { current: null };
    function Probe() { box.current = useTargetLock(options); return null; }
    const root = createRoot(host);
    act(() => root.render(<Probe />));
    return { box, root };
  }

  it('locks the clicked face of the native frame on screen and announces TARGET LOCKED', async () => {
    const lock = vi.fn(async () => lockOf({ bank_size: 4, scan: { state: 'done' } }));
    const { box } = mount({ target: 'm.mp4', frameIndex: 17, lock });
    await act(async () => { await box.current!.lockFace(1); });
    expect(lock).toHaveBeenCalledWith({ path: 'm.mp4', frame_index: 17, face_index: 1, sample_count: 16, wait: false });
    expect(box.current!.lock?.label).toBe('Target #1');
    expect(box.current!.toast?.text).toBe('TARGET LOCKED ✓ Target #1 · 4 angles');
  });

  it('adds an angle to the existing lock, and refuses (without calling the API) when nothing is locked', async () => {
    const add = vi.fn(async () => lockOf({ bank_size: 5 }));
    const { box } = mount({ target: 'm.mp4', frameIndex: 30, lock: vi.fn(async () => lockOf()), addAngle: add });
    await act(async () => { await box.current!.addAngle(0); });
    expect(add).not.toHaveBeenCalled();
    expect(box.current!.toast?.kind).toBe('error');
    await act(async () => { await box.current!.lockFace(0); });
    await act(async () => { await box.current!.addAngle(0); });
    expect(add).toHaveBeenCalledWith({ lock_id: 'a'.repeat(32), frame_index: 30, face_index: 0 });
    expect(box.current!.toast?.text).toBe('Angle added to Target #1 · 5 angles');
  });

  it('polls a running background scan until the bank is complete', async () => {
    const refresh = vi.fn()
      .mockResolvedValueOnce(lockOf({ bank_size: 2, scan: { state: 'running' } }))
      .mockResolvedValueOnce(lockOf({ bank_size: 5, scan: { state: 'done' } }));
    const { box } = mount({ target: 'm.mp4', frameIndex: 0, lock: vi.fn(async () => lockOf({ scan: { state: 'running' } })), refresh, pollMs: 100 });
    await act(async () => { await box.current!.lockFace(0); });
    await act(async () => { await vi.advanceTimersByTimeAsync(250); });
    expect(refresh).toHaveBeenCalledTimes(2);
    expect(box.current!.lock?.bank_size).toBe(5);
  });

  it('reports a failed lock and drops the lock when the video changes', async () => {
    const lock = vi.fn().mockRejectedValueOnce(new Error('No face was detected in that frame')).mockResolvedValue(lockOf());
    const { box, root } = mount({ target: 'a.mp4', frameIndex: 0, lock });
    await act(async () => { await box.current!.lockFace(0); });
    expect(box.current!.toast).toEqual({ kind: 'error', text: 'No face was detected in that frame' });
    await act(async () => { await box.current!.lockFace(0); });
    expect(box.current!.lock).not.toBeNull();
    function Probe() { box.current = useTargetLock({ target: 'b.mp4', frameIndex: 0, lock }); return null; }
    act(() => root.render(<Probe />));
    expect(box.current!.lock).toBeNull();
  });
});
