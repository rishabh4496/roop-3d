import { memo, useState, type MouseEvent } from 'react';
import type { FaceSet, IdentitySheet, Job, Track } from '../../contracts';
import { usePlayhead } from '../../stores/playheadStore';
import { facesetDropProps } from '../../utils/facesetDrag';
import { formatClock, trackLabel } from './TargetPanel';

const COLORS = ['#2dd4bf', '#a78bfa', '#f59e0b', '#38bdf8', '#f472b6', '#84cc16', '#fb7185', '#facc15'];
export const laneColor = (index: number) => COLORS[index % COLORS.length];
const pct = (value: number, duration: number) => `${Math.max(0, Math.min(100, (value / Math.max(duration, 1e-6)) * 100))}%`;

/** Visible frame spans of a track: the identity sheet's segments when available, else first → last frame. */
export function trackSpans(track: Track, sheetTrack?: IdentitySheet['tracks'][number]): [number, number][] {
  if (sheetTrack?.segments.length) return sheetTrack.segments.map(([start, end]) => [start, end]);
  return [[track.first_frame, track.last_frame]];
}

/** Rendered fraction of the job's range, or null when the job has no frame total yet. */
export function renderFraction(job: Job | null): number | null {
  if (!job) return null;
  if (job.state === 'succeeded') return 1;
  const total = job.progress.total_frames;
  return total && total > 0 ? Math.max(0, Math.min(1, job.progress.frame / total)) : null;
}

const Playhead = memo(function Playhead({ duration }: { duration: number }) {
  const time = usePlayhead();
  return <span className="lane-playhead" style={{ left: pct(time, duration) }} aria-hidden="true" />;
});

function TrackLane({ track, index, sheetTrack, fps, duration, selected, assignedName, onSelect, onSeek, onAssign }: {
  track: Track; index: number; sheetTrack?: IdentitySheet['tracks'][number]; fps: number; duration: number; selected: boolean; assignedName: string | null;
  onSelect: (trackId: string) => void; onSeek: (time: number) => void; onAssign: (trackId: string, facesetId: string) => void;
}) {
  const [over, setOver] = useState(false);
  const seekTo = (event: MouseEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    onSelect(track.track_id);
    if (box.width > 0) onSeek(((event.clientX - box.left) / box.width) * duration);
  };
  return (
    <div className={`lane ${selected ? 'selected' : ''} ${over ? 'drop-over' : ''}`} data-lane-track={track.track_id} {...facesetDropProps(id => onAssign(track.track_id, id), setOver)}>
      <button type="button" className="lane-label" aria-pressed={selected} onClick={() => onSelect(track.track_id)} title={`Select ${trackLabel(track.track_id)}`}>
        <i style={{ background: laneColor(index) }} />{trackLabel(track.track_id)}<small>{over ? 'drop FaceSet' : assignedName ?? 'original'}</small>
      </button>
      <div className="lane-body" role="presentation" onClick={seekTo}>
        {trackSpans(track, sheetTrack).map(([start, end]) => (
          <span key={`${start}-${end}`} className="lane-span" title={`${trackLabel(track.track_id)}: frames ${start}–${end}`}
            style={{ left: pct(start / fps, duration), width: `calc(${pct((end - start + 1) / fps, duration)} + 1px)`, background: laneColor(index), opacity: assignedName ? 0.95 : 0.55 }} />
        ))}
        {sheetTrack?.missing_ranges.map(([start, end]) => <span key={`m${start}`} className="lane-gap" title={`Not detected: frames ${start}–${end}`} style={{ left: pct(start / fps, duration), width: pct((end - start + 1) / fps, duration) }} />)}
      </div>
    </div>
  );
}

export interface TrackLanesProps {
  tracks: Track[]; sheet: IdentitySheet | null; sceneCuts: number[]; fps: number; duration: number;
  trimStart: number; trimEnd: number | null; job: Job | null; selectedTrack: string;
  assigned: Record<string, string>; sources: FaceSet[];
  onSelectTrack: (trackId: string) => void; onSeek: (time: number) => void; onAssign: (trackId: string, facesetId: string) => void;
}

/**
 * Overview lanes above the timeline: scene cuts, the processing (export) range, one lane per face track (drop a FaceSet
 * on it to assign), and the selected render's progress. Everything shown is backed by inspection / identity-sheet /
 * job data; nothing is drawn when the data does not exist.
 */
export default function TrackLanes({ tracks, sheet, sceneCuts, fps, duration, trimStart, trimEnd, job, selectedTrack, assigned, sources, onSelectTrack, onSeek, onAssign }: TrackLanesProps) {
  if (!(duration > 0)) return null;
  const end = trimEnd ?? duration;
  const fraction = renderFraction(job);
  const jobStart = job?.request.trim_start_seconds ?? 0, jobEnd = job?.request.trim_end_seconds ?? duration;
  const name = (id: string | undefined) => (id ? sources.find(source => source.faceset_id === id)?.name ?? id : null);
  const seekFromBar = (event: MouseEvent<HTMLDivElement>) => { const box = event.currentTarget.getBoundingClientRect(); if (box.width > 0) onSeek(((event.clientX - box.left) / box.width) * duration); };
  return (
    <div className="track-lanes" aria-label="Face track lanes">
      <div className="lane meta">
        <span className="lane-label static">Scene cuts<small>{sceneCuts.length}</small></span>
        <div className="lane-body" role="presentation" onClick={seekFromBar}>
          {sceneCuts.map(frame => <button type="button" key={frame} className="lane-cut" style={{ left: pct(frame / fps, duration) }} aria-label={`Scene cut at frame ${frame}`} title={`Scene cut · frame ${frame} · ${formatClock(frame / fps)}`}
            onClick={event => { event.stopPropagation(); onSeek(frame / fps); }} />)}
        </div>
      </div>
      <div className="lane meta">
        <span className="lane-label static">Processing<small>{formatClock(trimStart)}–{formatClock(end)}</small></span>
        <div className="lane-body" role="presentation" onClick={seekFromBar}>
          <span className="lane-range" style={{ left: pct(trimStart, duration), width: pct(end - trimStart, duration) }} title={`Export range ${formatClock(trimStart)} → ${formatClock(end)}`} />
        </div>
      </div>
      {tracks.map((track, index) => (
        <TrackLane key={track.track_id} track={track} index={index} sheetTrack={sheet?.tracks.find(item => item.track_id === track.track_id)} fps={fps} duration={duration}
          selected={selectedTrack === track.track_id} assignedName={name(assigned[track.track_id])} onSelect={onSelectTrack} onSeek={onSeek} onAssign={onAssign} />
      ))}
      {job && (
        <div className="lane meta">
          <span className="lane-label static">Render<small>{job.state}{fraction !== null ? ` · ${Math.round(fraction * 100)}%` : ''}</small></span>
          <div className="lane-body" role="progressbar" aria-label="Render progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={fraction === null ? undefined : Math.round(fraction * 100)}>
            <span className="lane-range render-scope" style={{ left: pct(jobStart, duration), width: pct(jobEnd - jobStart, duration) }} />
            {fraction !== null && <span className={`lane-range render-done ${job.state === 'failed' ? 'failed' : ''}`} style={{ left: pct(jobStart, duration), width: pct((jobEnd - jobStart) * fraction, duration) }} />}
          </div>
        </div>
      )}
      <div className="lane-playhead-host" aria-hidden="true"><Playhead duration={duration} /></div>
    </div>
  );
}
