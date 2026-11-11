import { useEffect, useMemo, useState } from 'react';
import { api } from '../../api';
import type { FaceSet, Inspection } from '../../contracts';

/** `GET /actors/{inspection_id}` (roop_3d/server/actor_routes.py): distinct people in the target, most prominent first. */
export interface Actor {
  actor_id: string; label: string; rank: number; track_ids: string[]; persistent_ids: number[];
  grouping: 'persistent_id' | 'embedding' | 'single_track';
  screen_time_seconds: number; prominence: number; thumbnail_base64: string; thumbnail_mime: string;
}
export interface ActorReport { inspection_id: string; actor_count: number; actors: Actor[]; warnings: string[] }

export const MIXED = '__mixed__';

/** The FaceSet every track of `actor` is mapped to, '' when none, or MIXED when its tracks disagree. */
export function actorAssignment(actor: Pick<Actor, 'track_ids'>, assigned: Record<string, string>): string {
  const values = new Set(actor.track_ids.map(trackId => assigned[trackId] ?? ''));
  return values.size === 1 ? [...values][0] : MIXED;
}

export function formatScreenTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return '0s';
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)}s`;
  return `${Math.floor(seconds / 60)}m ${String(Math.round(seconds % 60)).padStart(2, '0')}s`;
}

const GROUPING_NOTE: Record<Actor['grouping'], string> = {
  persistent_id: 'tracks joined by the tracker’s identity',
  embedding: 'tracks joined by face-identity similarity',
  single_track: 'one continuous appearance',
};

export interface ActorPanelProps {
  inspection: Inspection | null;
  sources: FaceSet[];
  /** track id -> FaceSet id (the face-selection store's assignments). */
  assigned: Record<string, string>;
  onAssign: (trackId: string, facesetId: string | null) => void;
  /** Test seam: skip the network and use this report. */
  report?: ActorReport | null;
}

/**
 * Multi-person routing: "Person #1 … Person #N" ranked by screen time and prominence. Picking a FaceSet for a person maps every
 * one of their tracks at once, so several actors can be swapped in the same job, each with its own `.fsz` (its own 3D skull lock).
 */
export default function ActorPanel({ inspection, sources, assigned, onAssign, report: injected = null }: ActorPanelProps) {
  const [fetched, setFetched] = useState<ActorReport | null>(null);
  const [error, setError] = useState('');
  const inspectionId = inspection?.inspection_id ?? '';

  useEffect(() => {
    setFetched(null); setError('');
    if (injected || !inspectionId) return undefined;
    const controller = new AbortController();
    api<ActorReport>(`/actors/${inspectionId}`, { signal: controller.signal })
      .then(setFetched)
      .catch((reason: Error) => { if (reason.name !== 'AbortError') setError(reason.message); });
    return () => controller.abort();
  }, [inspectionId, injected]);

  const report = injected ?? fetched;
  const actors = useMemo(() => report?.actors ?? [], [report]);
  const mapped = useMemo(() => actors.filter(actor => actorAssignment(actor, assigned) !== '').length, [actors, assigned]);
  if (!inspection) return null;
  if (error) return <p className="muted studio-hint" role="status">People grouping unavailable: {error}</p>;
  if (!actors.length) return null;

  const facesetIds = new Set(actors.map(actor => actorAssignment(actor, assigned)).filter(value => value && value !== MIXED));
  return (
    <section className="actor-panel" data-testid="actor-panel" aria-label="People in this clip">
      <header className="actor-panel-head">
        <strong>People in this clip</strong>
        <span className="muted">{actors.length} {actors.length === 1 ? 'person' : 'people'} · {mapped} mapped{facesetIds.size > 1 ? ` · ${facesetIds.size} different FaceSets` : ''}</span>
      </header>
      {report?.warnings?.map(note => <p key={note} className="muted studio-hint">{note}</p>)}
      <ol className="actor-list">
        {actors.map(actor => {
          const value = actorAssignment(actor, assigned);
          return (
            <li key={actor.actor_id} className="actor-row" data-testid={`actor-${actor.actor_id}`}>
              {actor.thumbnail_base64
                ? <img src={`data:${actor.thumbnail_mime};base64,${actor.thumbnail_base64}`} alt={`${actor.label} in the target`} />
                : <span className="thumb-empty" aria-hidden />}
              <div>
                <b>{actor.label}</b>
                <small className="muted" title={GROUPING_NOTE[actor.grouping]}>
                  {formatScreenTime(actor.screen_time_seconds)} on screen · {actor.track_ids.length} {actor.track_ids.length === 1 ? 'track' : 'tracks'}
                </small>
                <select aria-label={`${actor.label} source FaceSet`} value={value} disabled={!sources.length}
                  onChange={event => { if (event.target.value === MIXED) return; for (const trackId of actor.track_ids) onAssign(trackId, event.target.value || null); }}>
                  <option value="">Keep original</option>
                  {value === MIXED && <option value={MIXED} disabled>Mixed (set per track below)</option>}
                  {sources.map(item => <option key={item.faceset_id} value={item.faceset_id}>{item.name} · {item.item_count} angle{item.item_count === 1 ? '' : 's'}</option>)}
                </select>
              </div>
            </li>
          );
        })}
      </ol>
      <p className="muted studio-hint">Each person can use a different FaceSet; every FaceSet keeps its own 3D skull lock and atlas.</p>
    </section>
  );
}
