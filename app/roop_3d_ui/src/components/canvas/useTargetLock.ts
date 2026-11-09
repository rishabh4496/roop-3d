import { useCallback, useEffect, useRef, useState } from 'react';
import { addTargetAngleSample, api, lockTargetFromFrame, rescanTargetAngles, unlockTarget, type TargetLock } from '../../api';
import type { Toast } from './useFaceExtraction';

type Options = {
  /** Local media path of the target video; changing it drops the locks (locks belong to one video). */
  target: string;
  /** Native frame index on screen (the frame whose face is clicked). */
  frameIndex: number;
  playing?: boolean;
  getPlaybackFrame?: () => number;
  /** Test seams. */
  lock?: typeof lockTargetFromFrame;
  addAngle?: typeof addTargetAngleSample;
  unlock?: typeof unlockTarget;
  rescan?: typeof rescanTargetAngles;
  refresh?: (id: string) => Promise<TargetLock>;
  toastMs?: number;
  pollMs?: number;
};

const refreshTarget = (id: string) => api<TargetLock>(`/target/${id}`);

const label = (lock: TargetLock) => lock.coverage
  ? `${lock.label} · ${lock.coverage.captured === 15 ? 'All Angles Achieved (15/15)' : `${lock.coverage.captured}/15 angles captured`}`
  : `${lock.label} · ${lock.bank_size} angle${lock.bank_size === 1 ? '' : 's'}`;

/** Insert or replace a lock by id, keeping the order faces were locked in. */
const upsert = (list: TargetLock[], next: TargetLock) => list.some(item => item.lock_id === next.lock_id)
  ? list.map(item => item.lock_id === next.lock_id ? next : item) : [...list, next];
/** Replace only a lock the UI still holds (a late refresh must not resurrect an unlocked target). */
const replaceIfHeld = (list: TargetLock[], next: TargetLock) => list.some(item => item.lock_id === next.lock_id)
  ? list.map(item => item.lock_id === next.lock_id ? next : item) : list;

/**
 * Stage 12: lock faces of the frame on screen as targets (`POST /target/lock-from-frame`), add further angles of the same
 * person to a reference bank (`POST /target/add-angle-sample`), unlock a target (`DELETE /target/{id}`) and re-harvest all of
 * its angles (`POST /target/{id}/rescan`). The server keeps one persistent `face N` slot per person, so each lock survives
 * crossing paths, left/right order swaps and head turns to steep profiles.
 *
 * Several faces can be locked at once (`locks`, in locking order); `lock` is the primary one: the most recently locked or
 * touched target, which the "add angle" menu item and the status chip refer to. Every lock whose scan is running is polled
 * so its angle grid fills in live.
 */
export function useTargetLock({ target, frameIndex, playing = false, getPlaybackFrame, lock = lockTargetFromFrame, addAngle = addTargetAngleSample,
  unlock = unlockTarget, rescan = rescanTargetAngles, refresh = refreshTarget, toastMs = 4000, pollMs = 1500 }: Options) {
  const [locks, setLocks] = useState<TargetLock[]>([]);
  const [primaryId, setPrimaryId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<Toast | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const polls = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const generation = useRef(0);
  const stopPolls = useCallback(() => { polls.current.forEach(timer => clearTimeout(timer)); polls.current.clear(); }, []);
  useEffect(() => () => { generation.current += 1; clearTimeout(toastTimer.current); stopPolls(); }, [stopPolls]);
  useEffect(() => { generation.current += 1; stopPolls(); setLocks([]); setPrimaryId(null); }, [target, stopPolls]);
  const show = useCallback((next: Toast) => {
    setToast(next);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), toastMs);
  }, [toastMs]);

  const watchScan = useCallback((first: TargetLock) => {
    clearTimeout(polls.current.get(first.lock_id));
    polls.current.delete(first.lock_id);
    const version = generation.current;
    const tick = (value: TargetLock) => {
      if (value.scan?.state !== 'running' || generation.current !== version) return;
      polls.current.set(value.lock_id, setTimeout(async () => {
        try {
          const next = await refresh(value.lock_id);
          if (generation.current !== version) return;
          setLocks(prev => replaceIfHeld(prev, next));
          if (next.scan?.state === 'done') window.dispatchEvent(new Event('target-angle-scan-completed'));
          tick(next);
        } catch { tick(value); }
      }, pollMs));
    };
    tick(first);
  }, [refresh, pollMs]);

  // Scrubbing/playback can improve buckets after the background scan has finished.
  const heldIds = locks.map(item => item.lock_id).join(',');
  const primary = locks.find(item => item.lock_id === primaryId) ?? locks[locks.length - 1] ?? null;
  const currentId = primary?.lock_id;
  const lastHarvestFrame = useRef(frameIndex);
  useEffect(() => {
    if (lastHarvestFrame.current === frameIndex) return;
    lastHarvestFrame.current = frameIndex;
    if (!heldIds) return;
    const version = generation.current;
    const timer = setTimeout(() => {
      for (const id of heldIds.split(',')) {
        void refresh(id).then(next => { if (version === generation.current) setLocks(prev => replaceIfHeld(prev, next)); }).catch(() => {});
      }
    }, 250);
    return () => clearTimeout(timer);
  }, [heldIds, frameIndex, refresh]);

  useEffect(() => {
    if (!playing || !currentId || !getPlaybackFrame) return;
    const controller = new AbortController();
    const version = generation.current;
    let timer: ReturnType<typeof setTimeout>;
    const harvest = async () => {
      try {
        // One match call associates every face of the frame with its cluster, so it feeds all locks' banks.
        await api(`/target/${currentId}/match?frame_index=${getPlaybackFrame()}`, { signal: controller.signal });
        for (const id of heldIds.split(',')) {
          const next = await refresh(id);
          if (!controller.signal.aborted && version === generation.current) setLocks(prev => replaceIfHeld(prev, next));
        }
      } catch { /* Retry on the next playback sample without piling up requests. */ }
      if (!controller.signal.aborted) timer = setTimeout(() => void harvest(), 500);
    };
    void harvest();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [playing, currentId, heldIds, getPlaybackFrame, refresh]);

  const lockFace = useCallback(async (faceIndex: number, trackId?: string) => {
    if (!target || busy) return null;
    setBusy(true);
    try {
      const version = generation.current;
      const result = await lock({ path: target, frame_index: frameIndex, face_index: faceIndex, sample_count: 192, wait: false, ...(trackId ? { track_id: trackId } : {}) });
      if (version !== generation.current) return null;
      if (!result || typeof result.lock_id !== 'string') throw new Error('The local service returned an unexpected target-lock response.');
      setLocks(prev => upsert(prev, result));
      setPrimaryId(result.lock_id);
      show({ kind: 'ok', text: `TARGET LOCKED ✓ ${label(result)}` });
      watchScan(result);
      return result;
    } catch (error) {
      show({ kind: 'error', text: (error as Error).message || 'Could not lock that face.' });
      return null;
    } finally { setBusy(false); }
  }, [target, frameIndex, busy, lock, show, watchScan]);

  /** Unlock a target. The `face N` slot stays on the server; this only drops the lock and its capture polling. */
  const unlockFace = useCallback(async (lockId: string) => {
    const held = locks.find(item => item.lock_id === lockId);
    if (!held || busy) return false;
    setBusy(true);
    try {
      await unlock(lockId);
      clearTimeout(polls.current.get(lockId)); polls.current.delete(lockId);
      setLocks(prev => prev.filter(item => item.lock_id !== lockId));
      setPrimaryId(prev => (prev === lockId ? null : prev));
      show({ kind: 'ok', text: `${held.label} unlocked` });
      return true;
    } catch (error) {
      show({ kind: 'error', text: (error as Error).message || `Could not unlock ${held.label}.` });
      return false;
    } finally { setBusy(false); }
  }, [locks, busy, unlock, show]);

  /** "Re-Scan All Angles": the server clears the auto-harvested references and harvests the whole video again. */
  const rescanAngles = useCallback(async (lockId: string) => {
    if (!locks.some(item => item.lock_id === lockId)) return null;
    try {
      const result = await rescan(lockId);
      setLocks(prev => replaceIfHeld(prev, result));
      watchScan(result);
      show({ kind: 'ok', text: `Re-scanning every angle of ${result.label}…` });
      return result;
    } catch (error) {
      show({ kind: 'error', text: (error as Error).message || 'Could not start the angle scan.' });
      return null;
    }
  }, [locks, rescan, show, watchScan]);

  const addAngleOf = useCallback(async (faceIndex: number, lockId?: string) => {
    const into = (lockId ? locks.find(item => item.lock_id === lockId) : primary) ?? null;
    if (!into || busy) { if (!into) show({ kind: 'error', text: 'Lock a target first (right-click a face → Lock as Primary Target).' }); return null; }
    setBusy(true);
    try {
      const result = await addAngle({ lock_id: into.lock_id, frame_index: frameIndex, face_index: faceIndex });
      setLocks(prev => upsert(prev, result));
      show({ kind: 'ok', text: `Angle added to ${label(result)}` });
      return result;
    } catch (error) {
      show({ kind: 'error', text: (error as Error).message || 'Could not add that angle.' });
      return null;
    } finally { setBusy(false); }
  }, [locks, primary, frameIndex, busy, addAngle, show]);

  return { lock: primary, locks, lockFace, unlockFace, rescanAngles, addAngle: addAngleOf, busy, toast, dismiss: () => setToast(null) };
}
