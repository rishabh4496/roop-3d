import type { FaceSet, Track } from '../../contracts';
import { trackLabel } from './TargetPanel';

export interface FaceMappingProps {
  tracks: Track[];
  sources: FaceSet[];
  /** track id -> FaceSet id (the face-selection store's assignments). */
  assigned: Record<string, string>;
  selectedTrack: string | null;
  onSelectTrack: (trackId: string) => void;
  onAssign: (trackId: string, facesetId: string | null) => void;
}

/** "Source -> Target" rows: which FaceSet will replace which detected target face, with the angle count of each source. */
export default function FaceMapping({ tracks, sources, assigned, selectedTrack, onSelectTrack, onAssign }: FaceMappingProps) {
  if (!tracks.length) return <p className="muted studio-hint">Inspect the target (or click a face in the frame): each detected face is listed here, ready to map to a source.</p>;
  return (
    <div className="face-mapping" data-testid="face-mapping">
      {tracks.map(track => {
        const source = sources.find(item => item.faceset_id === assigned[track.track_id]);
        return (
          <div key={track.track_id} className={`face-mapping-row ${selectedTrack === track.track_id ? 'selected' : ''}`} role="group" aria-label={`${trackLabel(track.track_id)} mapping`}>
            {source ? <img src={`/api/v1/facesets/${source.faceset_id}/thumb`} alt={`${source.name} source`} /> : <span className="thumb-empty" aria-hidden />}
            <span className="arrow" aria-hidden>→</span>
            {track.thumbnail_base64
              ? <img src={`data:${track.thumbnail_mime};base64,${track.thumbnail_base64}`} alt={`${trackLabel(track.track_id)} target`} onClick={() => onSelectTrack(track.track_id)} />
              : <span className="thumb-empty" aria-hidden />}
            <div>
              <button type="button" className="link-button" onClick={() => onSelectTrack(track.track_id)}><b>{trackLabel(track.track_id)}</b></button>
              <select aria-label={`${trackLabel(track.track_id)} source`} value={source?.faceset_id ?? ''} onChange={event => onAssign(track.track_id, event.target.value || null)} disabled={!sources.length}>
                <option value="">Keep original</option>
                {sources.map(item => <option key={item.faceset_id} value={item.faceset_id}>{item.name} · {item.item_count} angle{item.item_count === 1 ? '' : 's'}</option>)}
              </select>
              {source && <span className="badge">{source.item_count} angle{source.item_count === 1 ? '' : 's'} loaded</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}
