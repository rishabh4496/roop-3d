import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from 'react';
import { useFilmstrip } from '../../frameScrub';
import { EmblaCarousel } from '../ui-layouts/EmblaCarousel';
import { TimelineAnimation } from '../ui-layouts/TimelineAnimation';
import { cn } from '../../lib/utils';
import { formatTimecode } from './TimelineWorkstation';
import { CHAPTERS_CHANGED, loadChapters } from './chapters';
import { usePlaybackRate, setPlaybackRate } from '../../stores/playbackRateStore';
import {
  SPEEDS, activeSlide, chapterSlides, inTrimRange, moveTrimHandle, percentOf, placeholderFrames, rulerLabel, slideSpan, stepSeconds, type DockFrame,
} from './dockMath';

export interface TimelineDockProps {
  target: string;
  duration: number;
  fps: number;
  currentTime: number;
  onSeek: (seconds: number) => void;
  trimStart?: number;
  trimEnd?: number | null;
  /** Without it the In / Out handles are read-only (the Editor screen only previews). */
  onTrimChange?: (start: number, end: number | null) => void;
  /** Show the 0.25x - 4x speed group. */
  showSpeed?: boolean;
  disabled?: boolean;
  /** Thumbnails to show instead of asking the server (tests, storybook). */
  frames?: DockFrame[];
  className?: string;
}

const THUMBS = 24;
type Chapter = { id: string; time: number; label: string };

/**
 * Stage 24 dock: a measured timecode ruler, an Embla filmstrip, In / Out trim handles, chapter markers and 0.25x - 4x speed.
 * It sits at the bottom of the Studio and the Editor. It never owns the playhead: every control calls the same `onSeek` /
 * `onTrimChange` the main timeline uses, so the two always agree.
 */
export default function TimelineDock({ target, duration, fps, currentTime, onSeek, trimStart = 0, trimEnd = null, onTrimChange, showSpeed = true, disabled = false, frames: injected, className }: TimelineDockProps) {
  const strip = useFilmstrip(injected ? '' : target || '', THUMBS);
  const frames = useMemo<DockFrame[]>(() => {
    if (injected) return injected;
    if (strip.status === 'ready' && Array.isArray(strip.data?.frames) && strip.data.frames.length) {
      return strip.data.frames.map(frame => ({ index: frame.slot, timestamp: frame.timestamp_sec, url: frame.data_url, frameIndex: frame.frame_index }));
    }
    return duration > 0 ? placeholderFrames(duration, 16) : [];
  }, [injected, strip.status, strip.data, duration]);

  const [chapters, setChapters] = useState<Chapter[]>(() => loadChapters(target));
  useEffect(() => {
    setChapters(loadChapters(target));
    const refresh = () => setChapters(loadChapters(target));
    window.addEventListener(CHAPTERS_CHANGED, refresh);
    return () => window.removeEventListener(CHAPTERS_CHANGED, refresh);
  }, [target]);

  const rate = usePlaybackRate();
  const safeFps = Math.max(1, fps || 30);
  const active = activeSlide(frames, currentTime);
  const flags = useMemo(() => chapterSlides(chapters, frames), [chapters, frames]);
  const out = trimEnd ?? duration;
  const ready = !!target && duration > 0 && !disabled;

  const setTrim = (next: { start: number; end: number | null }) => onTrimChange?.(next.start, next.end);
  const markIn = () => setTrim(moveTrimHandle('in', currentTime, { start: trimStart, end: trimEnd }, duration, safeFps));
  const markOut = () => setTrim(moveTrimHandle('out', currentTime, { start: trimStart, end: trimEnd }, duration, safeFps));
  const clearTrim = () => onTrimChange?.(0, null);

  return (
    <section className={cn('timeline-dock', className)} aria-label="Frame navigator" data-testid="timeline-dock" data-active-slide={active}>
      <TimelineAnimation animationNum={0} className="dock-head">
        <span className="dock-timecode telemetry" aria-label="Playhead timecode">{formatTimecode(currentTime, safeFps)}</span>
        <span className="dock-range telemetry" aria-label="Trim range">
          <b>IN</b> {rulerLabel(trimStart)} <b>OUT</b> {trimEnd === null ? 'end' : rulerLabel(trimEnd)} <span className="muted">· {rulerLabel(Math.max(0, out - trimStart))} kept</span>
        </span>
        <span className="dock-actions" role="group" aria-label="Trim at playhead">
          <button type="button" className="dock-btn" disabled={!ready || !onTrimChange} onClick={markIn} title="Set the In point to the playhead (I)">Set In</button>
          <button type="button" className="dock-btn" disabled={!ready || !onTrimChange} onClick={markOut} title="Set the Out point to the playhead (O)">Set Out</button>
          <button type="button" className="dock-btn" disabled={!ready || !onTrimChange || (trimStart === 0 && trimEnd === null)} onClick={clearTrim} title="Use the whole clip">Reset</button>
        </span>
        {showSpeed && (
          <span className="dock-speed" role="radiogroup" aria-label="Playback speed">
            {SPEEDS.map(speed => (
              <button key={speed} type="button" role="radio" aria-checked={rate === speed} className={cn('dock-btn', rate === speed && 'active')} disabled={disabled} onClick={() => setPlaybackRate(speed)} title={`Play at ${speed}x`}>{speed}x</button>
            ))}
          </span>
        )}
      </TimelineAnimation>

      <TimelineAnimation animationNum={1}>
        <EmblaCarousel label="Frame filmstrip" className="dock-embla" followIndex={ready ? active : null} trackClassName="dock-track">
          {frames.map((frame, index) => {
            const span = slideSpan(frames, index, duration);
            const kept = inTrimRange(span, trimStart, trimEnd);
            const labelled = flags.get(index);
            return (
              <div key={`${frame.index}-${index}`} className={cn('dock-slide', index === active && 'is-active', !kept && 'is-trimmed')} role="group" aria-roledescription="slide" aria-label={`Frame ${index + 1} of ${frames.length}`}>
                <button type="button" className="dock-thumb" disabled={!ready} aria-current={index === active ? 'true' : undefined} onClick={() => onSeek(frame.timestamp)}
                  title={`Jump to ${formatTimecode(frame.timestamp, safeFps)}${kept ? '' : ' (outside the In/Out range)'}`}>
                  {frame.url ? <img src={frame.url} alt="" draggable={false} loading="lazy" /> : <span className="dock-thumb-empty" aria-hidden />}
                  {labelled && <span className="dock-flag" title={chapters.filter(chapter => labelled.includes(chapter.id)).map(chapter => chapter.label).join(', ')} aria-hidden>⚑</span>}
                </button>
                <TimelineAnimation as="span" animationNum={Math.min(index, 12)} className="dock-tick telemetry">{rulerLabel(frame.timestamp, false)}</TimelineAnimation>
              </div>
            );
          })}
          {!frames.length && <p className="muted dock-empty">Load a target video to see its frames.</p>}
        </EmblaCarousel>
      </TimelineAnimation>

      <TimelineAnimation animationNum={2}>
        <TrimRail duration={duration} fps={safeFps} currentTime={currentTime} trimStart={trimStart} trimEnd={trimEnd} chapters={chapters} disabled={!ready}
          readOnly={!onTrimChange} onSeek={onSeek} onTrim={setTrim} />
      </TimelineAnimation>
    </section>
  );
}

interface RailProps {
  duration: number; fps: number; currentTime: number; trimStart: number; trimEnd: number | null; chapters: Chapter[]; disabled: boolean; readOnly: boolean;
  onSeek: (seconds: number) => void; onTrim: (trim: { start: number; end: number | null }) => void;
}

/** The measured rail under the filmstrip: In / Out handles (pointer or keyboard), chapter ticks and the playhead. */
function TrimRail({ duration, fps, currentTime, trimStart, trimEnd, chapters, disabled, readOnly, onSeek, onTrim }: RailProps) {
  const rail = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<'in' | 'out' | null>(null);
  const out = trimEnd ?? duration;
  const timeAt = useCallback((clientX: number) => {
    const box = rail.current?.getBoundingClientRect();
    if (!box || box.width <= 0) return 0;
    return Math.min(duration, Math.max(0, ((clientX - box.left) / box.width) * duration));
  }, [duration]);

  const apply = (which: 'in' | 'out', time: number) => onTrim(moveTrimHandle(which, time, { start: trimStart, end: trimEnd }, duration, fps));
  const down = (which: 'in' | 'out') => (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (disabled || readOnly) return;
    event.preventDefault(); event.currentTarget.setPointerCapture?.(event.pointerId); setDrag(which);
  };
  const move = (event: ReactPointerEvent<HTMLButtonElement>) => { if (drag) apply(drag, timeAt(event.clientX)); };
  const up = () => setDrag(null);
  const key = (which: 'in' | 'out') => (event: KeyboardEvent<HTMLButtonElement>) => {
    if (disabled || readOnly) return;
    const sign = event.key === 'ArrowRight' || event.key === 'ArrowUp' ? 1 : event.key === 'ArrowLeft' || event.key === 'ArrowDown' ? -1 : 0;
    const home = event.key === 'Home', end = event.key === 'End';
    if (!sign && !home && !end) return;
    event.preventDefault();
    const base = which === 'in' ? trimStart : out;
    apply(which, home ? 0 : end ? duration : base + sign * stepSeconds(fps, event.shiftKey));
  };

  return (
    <div className="dock-rail-wrap">
      <div ref={rail} className={cn('dock-rail', disabled && 'is-disabled')} role="group" aria-label="Trim range" onPointerDown={event => { if (!disabled && event.target === event.currentTarget) onSeek(timeAt(event.clientX)); }}>
        <span className="dock-rail__kept" style={{ left: `${percentOf(trimStart, duration)}%`, width: `${Math.max(0, percentOf(out, duration) - percentOf(trimStart, duration))}%` }} />
        {chapters.map(chapter => (
          <button key={chapter.id} type="button" className="dock-chapter" style={{ left: `${percentOf(chapter.time, duration)}%` }} disabled={disabled} onClick={() => onSeek(chapter.time)} title={`${chapter.label} · ${rulerLabel(chapter.time)}`} aria-label={`Chapter ${chapter.label} at ${rulerLabel(chapter.time)}`} />
        ))}
        <span className="dock-rail__head" style={{ left: `${percentOf(currentTime, duration)}%` }} aria-hidden />
        <button type="button" role="slider" className={cn('dock-handle dock-handle--in', drag === 'in' && 'is-dragging')} style={{ left: `${percentOf(trimStart, duration)}%` }}
          aria-label="In point" aria-valuemin={0} aria-valuemax={Math.max(0, out)} aria-valuenow={Number(trimStart.toFixed(3))} aria-valuetext={rulerLabel(trimStart)} aria-readonly={readOnly || undefined}
          disabled={disabled} onPointerDown={down('in')} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onKeyDown={key('in')}><span aria-hidden>[</span></button>
        <button type="button" role="slider" className={cn('dock-handle dock-handle--out', drag === 'out' && 'is-dragging')} style={{ left: `${percentOf(out, duration)}%` }}
          aria-label="Out point" aria-valuemin={Math.max(0, trimStart)} aria-valuemax={Math.max(0, duration)} aria-valuenow={Number(out.toFixed(3))} aria-valuetext={trimEnd === null ? 'end of clip' : rulerLabel(out)} aria-readonly={readOnly || undefined}
          disabled={disabled} onPointerDown={down('out')} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onKeyDown={key('out')}><span aria-hidden>]</span></button>
      </div>
      <div className="dock-rail__scale telemetry" aria-hidden><span>0:00</span><span>{rulerLabel(duration / 2, false)}</span><span>{rulerLabel(duration, false)}</span></div>
    </div>
  );
}
