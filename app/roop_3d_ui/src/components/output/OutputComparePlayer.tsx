import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react';
import type { Job } from '../../contracts';
import DialogPanel from '../guide/DialogPanel';
import './outputCompare.css';

/** Beyond this many seconds apart the follower is re-seeked to the master (about three frames at 25 fps). */
export const DRIFT_SECONDS = 0.12;
const VIDEO_OUTPUT = /\.(mp4|mov|mkv|webm)$/i;

export const afterUrl = (jobId: string) => `/api/v1/jobs/${encodeURIComponent(jobId)}/download`;
export const beforeUrl = (targetPath: string) => `/api/v1/media/content?path=${encodeURIComponent(targetPath)}`;
/** A finished job whose result is one playable video (image sequences and pass-only exports have no single file to compare). */
export const hasComparableVideo = (job: Job) => job.state === 'succeeded' && !!job.output && job.output.baked_video !== false && VIDEO_OUTPUT.test(job.output.output_path);

const clock = (seconds: number) => {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00.0';
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}.${Math.floor((seconds - whole) * 10)}`;
};
const play = (video: HTMLVideoElement | null) => { try { return Promise.resolve(video?.play()); } catch (error) { return Promise.reject(error); } };

export interface OutputComparePlayerProps {
  beforeSrc: string;
  afterSrc: string;
  downloadHref?: string;
  downloadName?: string;
}

/**
 * Original (left) vs. result (right), one transport. The result video is the master (it carries the audio): play, pause, seek, rate
 * and ended are mirrored onto the original, and any drift beyond `DRIFT_SECONDS` while playing is corrected on `timeupdate`.
 * "Wipe" reveals the result to the right of a draggable divider; "Side by side" shows both in full.
 */
export default function OutputComparePlayer({ beforeSrc, afterSrc, downloadHref = afterSrc, downloadName = 'roop3d_result.mp4' }: OutputComparePlayerProps) {
  const before = useRef<HTMLVideoElement>(null), after = useRef<HTMLVideoElement>(null);
  const [position, setPosition] = useState(50), [mode, setMode] = useState<'wipe' | 'side'>('wipe');
  const [playing, setPlaying] = useState(false), [time, setTime] = useState(0), [duration, setDuration] = useState(0);
  const [muted, setMuted] = useState(false), [aspect, setAspect] = useState('16 / 9');
  const [problem, setProblem] = useState<{ before?: string; after?: string }>({});
  const [loaded, setLoaded] = useState({ before: false, after: false });

  const follow = useCallback((force = false) => {
    const master = after.current, follower = before.current;
    if (!master || !follower || !Number.isFinite(master.currentTime)) return;
    if (force || Math.abs(follower.currentTime - master.currentTime) > DRIFT_SECONDS) follower.currentTime = master.currentTime;
  }, []);

  useEffect(() => { setProblem({}); setLoaded({ before: false, after: false }); setTime(0); setPlaying(false); }, [beforeSrc, afterSrc]);

  const toggle = () => {
    const master = after.current;
    if (!master) return;
    if (master.paused) { follow(true); void Promise.all([play(master), play(before.current)]).catch(() => setPlaying(false)); }
    else { master.pause(); before.current?.pause(); }
  };
  const seek = (seconds: number) => {
    for (const video of [after.current, before.current]) if (video) video.currentTime = seconds;
    setTime(seconds);
  };
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    const tag = (event.target as HTMLElement).tagName;
    if (event.key === ' ' && tag !== 'INPUT' && tag !== 'BUTTON' && tag !== 'A') { event.preventDefault(); toggle(); }
  };
  const fail = (which: 'before' | 'after') => setProblem(current => ({ ...current,
    [which]: which === 'before' ? 'The original video could not be loaded; the result is still available.' : 'The rendered video could not be loaded. Try Download MP4.' }));
  const both = loaded.before && loaded.after;

  return <div className="ocp" role="group" aria-label="Before and after comparison" onKeyDown={onKey}>
    <div className={`ocp-stage ${mode}`} data-mode={mode} style={{ aspectRatio: aspect, ['--ocp-pos' as string]: `${position}%` }}>
      <video ref={before} className="ocp-before" aria-label="Original video" src={beforeSrc} muted playsInline preload="auto"
        onLoadedData={() => setLoaded(value => ({ ...value, before: true }))} onError={() => fail('before')} />
      <video ref={after} className="ocp-after" aria-label="Completed render playback" src={afterSrc} muted={muted} playsInline preload="auto"
        onLoadedMetadata={event => {
          const { videoWidth, videoHeight, duration: length } = event.currentTarget;
          if (videoWidth && videoHeight) setAspect(`${videoWidth} / ${videoHeight}`);
          setDuration(Number.isFinite(length) ? length : 0);
        }}
        onLoadedData={() => setLoaded(value => ({ ...value, after: true }))} onError={() => fail('after')}
        onPlay={() => { setPlaying(true); follow(true); void play(before.current); }}
        onPause={() => { setPlaying(false); before.current?.pause(); follow(true); }}
        onSeeking={() => follow(true)} onSeeked={() => { follow(true); setTime(after.current?.currentTime ?? 0); }}
        onRateChange={event => { if (before.current) before.current.playbackRate = event.currentTarget.playbackRate; }}
        onEnded={() => { setPlaying(false); before.current?.pause(); }}
        onTimeUpdate={event => { setTime(event.currentTarget.currentTime); if (!event.currentTarget.paused) follow(); }} />
      {mode === 'wipe' && <div className="ocp-divider" aria-hidden="true" />}
      <span className="ocp-tag left">Before</span><span className="ocp-tag right">After</span>
      {!both && !problem.after && <div className="ocp-loading" role="status">Loading both videos…</div>}
    </div>
    {(problem.before || problem.after) && <p className="ocp-problem" role="alert">{problem.after ?? problem.before}</p>}
    <div className="ocp-controls">
      <button type="button" className="button" aria-pressed={playing} onClick={toggle} disabled={!!problem.after}>{playing ? 'Pause' : 'Play'}</button>
      <input className="ocp-seek" type="range" aria-label="Seek" min={0} max={duration || 0} step="any" value={Math.min(time, duration || 0)}
        onChange={event => seek(Number(event.target.value))} disabled={!duration} />
      <span className="ocp-time" aria-live="off">{clock(time)} / {clock(duration)}</span>
      <button type="button" className="button small" aria-pressed={muted} onClick={() => setMuted(value => !value)}>{muted ? 'Unmute' : 'Mute'}</button>
    </div>
    <div className="ocp-controls">
      <div role="group" aria-label="Comparison layout" className="ocp-mode">
        <button type="button" className="button small" aria-pressed={mode === 'wipe'} onClick={() => setMode('wipe')}>Wipe</button>
        <button type="button" className="button small" aria-pressed={mode === 'side'} onClick={() => setMode('side')}>Side by side</button>
      </div>
      {mode === 'wipe' && <label className="ocp-wipe">Divider
        <input type="range" aria-label="Before and after divider" min={0} max={100} value={position} onChange={event => setPosition(Number(event.target.value))} />
      </label>}
      <a className="button accent ocp-download" href={downloadHref} download={downloadName}>Download MP4</a>
    </div>
  </div>;
}

/** Opens the moment a job finishes: the original target next to the rendered `/api/v1/jobs/{id}/download`. */
export function RenderCompleteModal({ job, onClose }: { job: Job; onClose: () => void }) {
  return <div className="modal-backdrop" onClick={onClose}>
    <DialogPanel className="project-modal ocp-modal" label="Render complete" onClose={onClose} testId="render-complete-modal">
      <div className="panel-heading"><div><span className="eyebrow">RENDER COMPLETE</span><h2>Before / After</h2></div>
        <button type="button" className="button" onClick={onClose}>Close</button></div>
      <p className="muted">Synthetic media processed locally by Roop 3D · {job.output?.frames ?? 0} verified frames</p>
      <OutputComparePlayer beforeSrc={beforeUrl(job.request.target_path)} afterSrc={afterUrl(job.job_id)}
        downloadName={`roop3d_${job.job_id.slice(0, 8)}.mp4`} />
    </DialogPanel>
  </div>;
}

/** Remembers each job's last state so the player opens for a job that FINISHES in this session, never for history on page load. */
export function newlyCompleted(previous: ReadonlyMap<string, string> | null, jobs: readonly Job[]): Job | null {
  if (!previous) return null;
  let found: Job | null = null;
  for (const job of jobs) if (hasComparableVideo(job) && previous.get(job.job_id) !== 'succeeded') found = job;
  return found;
}
