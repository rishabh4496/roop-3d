import { useCallback, useEffect, useState } from 'react';
import { api, post } from '../../api';
import type { IdentityLock, IdentitySheet, IdentitySheetTrack, Inspection, FaceSet } from '../../contracts';

/** Seconds -> m:ss without importing the viewport (keeps this panel dependency-free). */
export function clock(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

export function rangeSummary(track: IdentitySheetTrack, limit = 3): string {
  const shown = track.segments.slice(0, limit).map(([, , start, end]) => `${clock(start)}–${clock(end)}`);
  const extra = track.segments.length - shown.length;
  return shown.join(', ') + (extra > 0 ? ` +${extra} more` : '');
}

export function withheldSummary(track: IdentitySheetTrack): string {
  const reasons = Object.entries(track.withheld_reasons).map(([reason, count]) => `${reason.replace(/_/g, ' ').toLowerCase()} ×${count}`);
  return reasons.join(', ');
}

/** The contact sheet exists only for identity-lock inspections; older inspections show nothing extra. */
export function useContactSheet(inspection: Inspection | null, target: string) {
  const [sheet, setSheet] = useState<IdentitySheet | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    setSheet(null); setError('');
    if (!inspection || inspection.tracking_method !== 'identity_lock_v1' || !target) return;
    const controller = new AbortController();
    api<IdentitySheet>(`/identity/inspections/${inspection.inspection_id}?${new URLSearchParams({ path: target })}`, { signal: controller.signal })
      .then(setSheet).catch(failure => { if (failure.name !== 'AbortError') setError(failure.message); });
    return () => controller.abort();
  }, [inspection, target]);
  return { sheet, error };
}

/** Per-track evidence shown on each target card: representative frames, visible time, confidence, withheld frames. */
export function IdentityTrackStats({ track }: { track: IdentitySheetTrack | undefined }) {
  if (!track) return null;
  const withheld = withheldSummary(track);
  return <span className="identity-stats" data-testid={`identity-stats-${track.track_id}`}>
    <span className="identity-thumbs">{track.thumbnails.map(item => <img key={item.frame} alt={`${track.track_id} frame ${item.frame}`} title={`Frame ${item.frame}`} src={`data:image/jpeg;base64,${item.jpeg_base64}`} />)}</span>
    <small>Visible {rangeSummary(track)} · {track.visible_frames} frames</small>
    <small>Confidence {track.confidence_mean.toFixed(2)} (min {track.confidence_min.toFixed(2)}){track.state_frames.REACQUIRED ? ` · reacquired ${track.state_frames.REACQUIRED}×` : ''}</small>
    {track.missing_ranges.length > 0 && <small className="identity-warn">Not detected: frames {track.missing_ranges.slice(0, 3).map(([a, b]) => (a === b ? `${a}` : `${a}–${b}`)).join(', ')}{track.missing_ranges.length > 3 ? '…' : ''}</small>}
    {track.withheld_frames > 0 && <small className="identity-warn">{track.withheld_frames} frames kept original: {withheld}</small>}
    {track.swappable_frames === 0 && <small className="identity-warn">No frame has verified identity; this track cannot be swapped.</small>}
  </span>;
}

function sameMapping(lock: IdentityLock, mappings: Record<string, string>): boolean {
  const locked = Object.fromEntries(lock.assignments.map(item => [item.track_id, item.faceset_id]));
  const current = Object.fromEntries(Object.entries(mappings).filter(([, value]) => value));
  const keys = new Set([...Object.keys(locked), ...Object.keys(current)]);
  return [...keys].every(key => locked[key] === current[key]);
}

export function lockProblem(lock: IdentityLock | null, mappings: Record<string, string>): string | null {
  if (!lock) return null;
  if (!lock.status.valid) return `Identity lock is stale (${[...new Set(lock.status.reasons.map(item => item.code))].join(', ')}).`;
  if (!sameMapping(lock, mappings)) return 'The mapping on screen differs from the identity lock.';
  return null;
}

interface BarProps {
  target: string; inspection: Inspection | null; mappings: Record<string, string>; sources: FaceSet[];
  lock: IdentityLock | null; onLock: (lock: IdentityLock | null) => void; busy: boolean; onError: (message: string) => void;
}

/** Explicit, auditable binding of each selected target track to one authorised source FaceSet. */
export function IdentityLockBar({ target, inspection, mappings, sources, lock, onLock, busy, onError }: BarProps) {
  const [working, setWorking] = useState(false);
  const eligible = !!inspection && inspection.tracking_method === 'identity_lock_v1';
  const chosen = Object.entries(mappings).filter(([, value]) => value);
  const name = useCallback((id: string) => sources.find(item => item.faceset_id === id)?.name ?? id, [sources]);
  const problem = lockProblem(lock, mappings);

  // Re-verify against the files on disk whenever the lock identity changes (the server cache is size/mtime based).
  useEffect(() => {
    if (!lock) return;
    let cancelled = false;
    api<IdentityLock>(`/identity/locks/${lock.lock_id}?verify=true`)
      .then(fresh => { if (!cancelled && JSON.stringify(fresh.status) !== JSON.stringify(lock.status)) onLock(fresh); })
      .catch(() => { /* a vanished lock is reported by the next action */ });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lock?.lock_id]);

  if (!eligible) return inspection ? <p className="inline-warning" data-testid="identity-legacy">This inspection predates the identity lock. Re-inspect the target to lock a mapping.</p> : null;

  const run = async (work: () => Promise<void>) => {
    setWorking(true);
    try { await work(); } catch (failure) { onError((failure as Error).message); } finally { setWorking(false); }
  };
  const create = () => run(async () => {
    const created = await post<IdentityLock>('/identity/locks', {
      target_path: target, inspection_id: inspection!.inspection_id,
      assignments: chosen.map(([track_id, faceset_id]) => ({ track_id, faceset_id })) });
    if (lock) await fetch(`/api/v1/identity/locks/${lock.lock_id}`, { method: 'DELETE' }).catch(() => undefined);
    onLock(created);
  });
  const release = () => run(async () => {
    if (lock) await fetch(`/api/v1/identity/locks/${lock.lock_id}`, { method: 'DELETE' });
    onLock(null);
  });

  return <div className={`identity-lock ${lock ? (problem ? 'stale' : 'locked') : ''}`} data-testid="identity-lock">
    <div>
      <strong>{lock ? (problem ? '⚠ Identity lock needs attention' : '🔒 Identity locked') : 'Identity not locked'}</strong>
      {lock && !problem && <small> rev {lock.revision} · {lock.digest.slice(0, 12)} · video {lock.target_sha256.slice(0, 8)}</small>}
    </div>
    {lock && <ul className="identity-assignments">{lock.assignments.map(item => <li key={item.track_id}>{item.track_id.replace('track_', 'Face ')} → {name(item.faceset_id)}</li>)}</ul>}
    {problem && <p className="inline-warning" role="alert">{problem}</p>}
    {lock && !lock.status.valid && <ul className="identity-reasons">{lock.status.reasons.map((item, index) => <li key={index}>{item.code}{item.track_id ? ` (${item.track_id})` : ''}: {item.detail}</li>)}</ul>}
    <div className="identity-actions">
      <button className="button" disabled={busy || working || !chosen.length || (!!lock && !problem)} onClick={create}
        title={chosen.length ? 'Bind these tracks to their source FaceSets for this exact video' : 'Assign at least one source first'}>
        {lock ? 'Re-lock mapping' : 'Lock mapping'}
      </button>
      {lock && <button className="button" disabled={busy || working} onClick={release}>Release lock</button>}
    </div>
    <small className="muted">Unlocked faces are never swapped. Faces whose identity is uncertain keep the original video.</small>
  </div>;
}
