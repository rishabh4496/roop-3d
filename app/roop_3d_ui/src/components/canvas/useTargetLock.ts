import { useCallback, useEffect, useRef, useState } from 'react';
import { addTargetAngleSample, api, lockTargetFromFrame, type TargetLock } from '../../api';
import type { Toast } from './useFaceExtraction';

type Options = {
  /** Local media path of the target video; changing it drops the lock (locks belong to one video). */
  target: string;
  /** Native frame index on screen (the frame whose face is clicked). */
  frameIndex: number;
  /** Test seams. */
  lock?: typeof lockTargetFromFrame;
  addAngle?: typeof addTargetAngleSample;
  refresh?: (id: string) => Promise<TargetLock>;
  toastMs?: number;
  pollMs?: number;
};

const label = (lock: TargetLock) => `${lock.label} · ${lock.bank_size} angle${lock.bank_size === 1 ? '' : 's'}`;

/**
 * Stage 12: lock a face of the frame on screen as the target (`POST /target/lock-from-frame`) and add further angles of the
 * same person to its reference bank (`POST /target/add-angle-sample`). The server keeps one persistent cluster id per person,
 * so the lock survives crossing paths, left/right order swaps and head turns to steep profiles.
 */
export function useTargetLock({ target, frameIndex, lock = lockTargetFromFrame, addAngle = addTargetAngleSample,
  refresh = id => api<TargetLock>(`/target/${id}`), toastMs = 4000, pollMs = 1500 }: Options) {
  const [current, setCurrent] = useState<TargetLock | null>(null);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<Toast | null>(null);
  const timers = useRef<{ toast?: ReturnType<typeof setTimeout>; poll?: ReturnType<typeof setTimeout> }>({});
  useEffect(() => () => { generation.current += 1; clearTimeout(timers.current.toast); clearTimeout(timers.current.poll); }, []);
  const generation = useRef(0);
  useEffect(() => { generation.current += 1; clearTimeout(timers.current.poll); setCurrent(null); }, [target]);
  const show = useCallback((next: Toast) => {
    setToast(next);
    clearTimeout(timers.current.toast);
    timers.current.toast = setTimeout(() => setToast(null), toastMs);
  }, [toastMs]);

  const watchScan = useCallback((first: TargetLock) => {
    clearTimeout(timers.current.poll);
    const version = generation.current;
    const tick = async (value: TargetLock) => {
      if (value.scan?.state !== 'running' || generation.current !== version) return;
      timers.current.poll = setTimeout(async () => {
        try { const next = await refresh(value.lock_id); if (generation.current !== version) return; setCurrent(prev => (prev?.lock_id === next.lock_id ? next : prev)); if (next.scan?.state === 'done') window.dispatchEvent(new Event('target-angle-scan-completed')); void tick(next); } catch { void tick(value); }
      }, pollMs);
    };
    void tick(first);
  }, [refresh, pollMs]);

  const lockFace = useCallback(async (faceIndex: number, trackId?: string) => {
    if (!target || busy) return null;
    setBusy(true);
    try {
      const version = ++generation.current;
      const result = await lock({ path: target, frame_index: frameIndex, face_index: faceIndex, sample_count: 192, wait: false, ...(trackId ? { track_id: trackId } : {}) });
      if (version !== generation.current) return null;
      setCurrent(result);
      show({ kind: 'ok', text: `TARGET LOCKED ✓ ${label(result)}` });
      watchScan(result);
      return result;
    } catch (error) {
      show({ kind: 'error', text: (error as Error).message || 'Could not lock that face.' });
      return null;
    } finally { setBusy(false); }
  }, [target, frameIndex, busy, lock, show, watchScan]);

  const addAngleOf = useCallback(async (faceIndex: number) => {
    if (!current || busy) { if (!current) show({ kind: 'error', text: 'Lock a target first (right-click a face → Lock as Primary Target).' }); return null; }
    setBusy(true);
    try {
      const result = await addAngle({ lock_id: current.lock_id, frame_index: frameIndex, face_index: faceIndex });
      setCurrent(result);
      show({ kind: 'ok', text: `Angle added to ${label(result)}` });
      return result;
    } catch (error) {
      show({ kind: 'error', text: (error as Error).message || 'Could not add that angle.' });
      return null;
    } finally { setBusy(false); }
  }, [current, frameIndex, busy, addAngle, show]);

  return { lock: current, lockFace, addAngle: addAngleOf, busy, toast, dismiss: () => setToast(null) };
}
