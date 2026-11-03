import { memo, useState, type ReactNode } from 'react';
import type { FaceSet, IdentitySheet, Inspection, Probe, Track } from '../../contracts';
import { IdentityTrackStats } from '../facemgr/IdentityLockPanel';
import { usePlayhead } from '../../stores/playheadStore';
import { facesetDropProps } from '../../utils/facesetDrag';
import VirtualList from '../common/VirtualList';

export const trackLabel = (trackId: string) => trackId.replace('track_', 'Face ').replace(/ 0+(\d)/, ' $1');
export function formatClock(seconds: number) {
  const safe = Math.max(0, seconds || 0);
  return `${Math.floor(safe / 60).toString().padStart(2, '0')}:${(safe % 60).toFixed(2).padStart(5, '0')}`;
}
const range = (value: [number, number] | null | undefined) => (value ? `${value[0].toFixed(0)}° … ${value[1].toFixed(0)}°` : 'n/a');

/**
 * Frame-accurate navigator. The readout follows the live playhead store, so playback ticks re-render only this
 * leaf; stepping commits through `setTime` exactly like the timeline does.
 */
export const FrameNavigator = memo(function FrameNavigator({ time, setTime, fps, duration, disabled }: { time: number; setTime: (value: number) => void; fps: number; duration: number; disabled: boolean }) {
  const liveFrame = usePlayhead(value => Math.round(value * fps));
  const frame = Number.isFinite(liveFrame) ? liveFrame : Math.round(time * fps);
  const last = Math.max(0, Math.floor(duration * fps) - 1);
  const [draft, setDraft] = useState<string | null>(null);
  const go = (next: number) => setTime(Math.max(0, Math.min(duration, Math.max(0, Math.min(last, Math.round(next))) / fps)));
  const commit = () => { if (draft !== null && draft.trim() !== '' && Number.isFinite(Number(draft))) go(Number(draft)); setDraft(null); };
  return (
    <div className="frame-nav" role="group" aria-label="Frame navigation">
      <button type="button" className="icon-button" disabled={disabled || frame <= 0} aria-label="First frame" title="First frame" onClick={() => go(0)}>⏮</button>
      <button type="button" className="icon-button" disabled={disabled || frame <= 0} aria-label="Previous frame" title="Previous frame (←)" onClick={() => go(frame - 1)}>◀</button>
      <input className="frame-input" aria-label="Frame number" inputMode="numeric" disabled={disabled} value={draft ?? String(frame)}
        onChange={event => setDraft(event.target.value)} onBlur={commit} onKeyDown={event => { if (event.key === 'Enter') commit(); if (event.key === 'Escape') setDraft(null); }} />
      <button type="button" className="icon-button" disabled={disabled || frame >= last} aria-label="Next frame" title="Next frame (→)" onClick={() => go(frame + 1)}>▶</button>
      <button type="button" className="icon-button" disabled={disabled || frame >= last} aria-label="Last frame" title="Last frame" onClick={() => go(last)}>⏭</button>
      <span className="frame-clock" aria-label="Timecode">{formatClock(frame / fps)}<small> / {formatClock(duration)}</small></span>
    </div>
  );
});

function TrackCard({ track, selected, assignedId, sources, sheetTrack, onSelect, onAssign }: {
  track: Track; selected: boolean; assignedId: string; sources: FaceSet[]; sheetTrack: IdentitySheet['tracks'][number] | undefined;
  onSelect: (trackId: string) => void; onAssign: (trackId: string, facesetId: string | null) => void;
}) {
  const [over, setOver] = useState(false);
  const assigned = sources.find(source => source.faceset_id === assignedId);
  return (
    <div className={`track-card studio-track ${selected ? 'active' : ''} ${over ? 'drop-over' : ''} ${assignedId ? 'assigned' : ''}`} data-track-id={track.track_id}
      {...facesetDropProps(id => onAssign(track.track_id, id), setOver)}>
      <button className="track-select" aria-label={`Inspect ${track.track_id}`} aria-pressed={selected} onClick={() => onSelect(track.track_id)}>
        {track.thumbnail_base64 ? <img src={`data:${track.thumbnail_mime};base64,${track.thumbnail_base64}`} alt={`${track.track_id} detected face`} /> : <span className="track-fallback">◎</span>}
        <span className="track-text">
          <strong>{trackLabel(track.track_id)}</strong>
          <small>Frames {track.first_frame}–{track.last_frame} · {formatClock(track.first_timestamp_seconds)} → {formatClock(track.last_timestamp_seconds)}</small>
          <small>Yaw {range(track.pose_range.yaw_degrees)}</small>
          <IdentityTrackStats track={sheetTrack} />
        </span>
      </button>
      <label className="mapping-choice">
        <span>{over ? 'Drop to assign' : assigned ? 'Swap with' : 'Source identity'}</span>
        <select aria-label={`Identity for ${track.track_id}`} value={assignedId} disabled={!sources.length} onChange={event => onAssign(track.track_id, event.target.value || null)}>
          <option value="">Keep original</option>
          {sources.map(source => <option key={source.faceset_id} value={source.faceset_id}>{source.name}</option>)}
        </select>
      </label>
    </div>
  );
}

export interface TargetPanelProps {
  target: string; probe: Probe | null; busy: boolean;
  uploadReason: string | null; onImport: () => void;
  time: number; setTime: (value: number) => void; fps: number; duration: number;
  inspection: Inspection | null; inspectReason: string | null; onInspect: () => void;
  sheet: IdentitySheet | null; selectedTrack: string; onSelectTrack: (trackId: string) => void;
  sources: FaceSet[]; assigned: Record<string, string>; onAssign: (trackId: string, facesetId: string | null) => void;
  lockBar?: ReactNode;
}

export default function TargetPanel(props: TargetPanelProps) {
  const { target, probe, busy, uploadReason, onImport, time, setTime, fps, duration, inspection, inspectReason, onInspect, sheet, selectedTrack, onSelectTrack, sources, assigned, onAssign, lockBar } = props;
  const name = target ? target.split('/').at(-1) : '';
  return (
    <div className="studio-target">
      <div className="studio-section">
        <button className="button full primary-soft" disabled={!!uploadReason || busy} title={uploadReason ?? 'Import a local video; it never leaves this machine'} onClick={onImport}>＋ Import target video</button>
        <div className={`target-asset ${target ? 'loaded' : ''}`}>
          <span className="asset-symbol">▤</span>
          <div><strong title={target}>{name || 'No target loaded'}</strong>
            <small>{probe ? `${probe.width} × ${probe.height} · ${probe.fps_rational ?? '?'} fps · ${formatClock(probe.duration_seconds ?? 0)}` : 'Your files stay on this machine'}</small></div>
        </div>
        <FrameNavigator time={time} setTime={setTime} fps={fps} duration={duration} disabled={!target} />
      </div>
      <div className="studio-section grow">
        <div className="section-title"><span>DETECTED TARGET FACES</span><span>{inspection ? inspection.tracks.length.toString().padStart(2, '0') : '—'}</span></div>
        <button className="button full" disabled={!target || !!inspectReason || busy} title={inspectReason ?? 'Detect and track every face in the target'} onClick={onInspect}>{inspection ? 'Re-inspect faces' : 'Inspect target faces'}</button>
        {busy && !inspection && target && <div className="skeleton-stack" aria-hidden="true"><span className="skeleton-line" /><span className="skeleton-line" style={{ width: '70%' }} /></div>}
        {!inspection ? <p className="muted studio-hint">{target ? 'Inspect the target to find persistent face tracks, then drag a FaceSet onto a track (or click a face in the video).' : 'Import a target video to begin.'}</p> : <>
          {inspection.truncated && <p className="inline-warning">Bounded sample: inspection was truncated.</p>}
          {inspection.warnings.map(warning => <p key={warning} className="inline-warning">{warning}</p>)}
          {!inspection.tracks.length && <p className="muted studio-hint">No faces were detected. Try another clip or a clearer frame.</p>}
          {/* Virtualized against the panel's own scroll box: long clips can carry many tracks. */}
          <VirtualList items={inspection.tracks} itemKey={track => track.track_id} estimateHeight={132} gap={8} className="studio-track-list" aria-label="Target face tracks"
            renderItem={track => <TrackCard track={track} selected={selectedTrack === track.track_id} assignedId={assigned[track.track_id] ?? ''} sources={sources}
              sheetTrack={sheet?.tracks.find(item => item.track_id === track.track_id)} onSelect={onSelectTrack} onAssign={onAssign} />} />
          {lockBar}
        </>}
      </div>
    </div>
  );
}
