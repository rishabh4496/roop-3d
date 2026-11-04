import CurveEditorTrack from './CurveEditorTrack';
import { renderProbe } from '../../utils/renderProbe';
import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import { useFaceSelectionStore } from '../../useFaceSelectionStore';
import { addChapter, adjacentChapter, chapterAt, loadChapters, removeChapter, renameChapter, saveChapters } from './chapters';
import { getPlayhead, playheadStore, usePlayhead } from '../../stores/playheadStore';

/**
 * Stage 11 live-playhead leaves. `currentTime` (committed by seek / step / pause) drives the timeline's layout and
 * logic; while `playing`, these small subscribers show the per-tick playhead from `playheadStore`, so a playback tick
 * re-renders only the needle, timecode and frame counter, never the filmstrip, ruler, pose track or curve editor.
 */
const liveOr = (playing, committed) => time => (playing ? time : committed);
function LiveTimecode({ currentTime, playing, fps, slice = 0 }) {
  const time = usePlayhead(liveOr(playing, currentTime));
  return formatTimecode(time, fps).slice(slice);
}
function LiveFrameCounter({ currentTime, playing, fps, total }) {
  const frame = usePlayhead(time => Math.floor((playing ? time : currentTime) * fps));
  return <>[F: {frame} / {total}]</>;
}
function LiveNeedle({ currentTime, playing, fps, duration, dragging }) {
  const time = usePlayhead(liveOr(playing, currentTime));
  const percent = Math.max(0, Math.min(100, (time / duration) * 100));
  return (
    <div className={`timeline-playhead-needle ${dragging ? 'dragging' : ''}`} style={{ left: `${percent}%` }} aria-label={`Playhead needle at ${formatTimecode(time, fps)}`}>
      <div className="playhead-handle-head">
        <span className="needle-head-arrow">▼</span>
        <span className="needle-head-tag">{formatTimecode(time, fps).slice(3)}</span>
      </div>
      <div className="playhead-needle-line" />
    </div>
  );
}
function LiveScrubber({ currentTime, playing, duration, fps, disabled, onSeek }) {
  const time = usePlayhead(liveOr(playing, currentTime));
  return (
    <input aria-label="Timeline scrubber" type="range" min="0" max={duration} step={1 / fps} value={Math.min(time, duration)} disabled={disabled}
      className="timeline-sr-scrubber" onChange={event => { if (!disabled && onSeek) onSeek(+event.target.value); }} />
  );
}
function LiveCurveEditor({ currentTime, playing, fps, ...props }) {
  const frame = usePlayhead(time => Math.round((playing ? time : currentTime) * fps));
  return <CurveEditorTrack frame={frame} {...props} />;
}

/**
 * Format seconds into standard SMPTE timecode (HH:MM:SS:FF).
 * @param {number} seconds
 * @param {number} fps
 * @returns {string}
 */
export function formatTimecode(seconds, fps = 30) {
  if (isNaN(seconds) || seconds < 0) seconds = 0;
  const safeFps = Math.max(1, fps);
  const totalFrames = Math.floor(seconds * safeFps);
  const ff = Math.floor(totalFrames % safeFps);
  const totalSeconds = Math.floor(seconds);
  const ss = totalSeconds % 60;
  const mm = Math.floor(totalSeconds / 60) % 60;
  const hh = Math.floor(totalSeconds / 3600);
  return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}:${String(ff).padStart(2, '0')}`;
}

/**
 * Parse rational FPS or return fallback.
 * @param {string | number | null | undefined} rational
 * @param {number} fallback
 * @returns {number}
 */
export function parseFps(rational, fallback = 30) {
  if (!rational) return fallback;
  if (typeof rational === 'number' && rational > 0) return rational;
  const str = String(rational);
  const parts = str.split('/');
  if (parts.length === 2) {
    const num = parseFloat(parts[0]);
    const den = parseFloat(parts[1]);
    if (den > 0 && !isNaN(num) && !isNaN(den)) return Math.round((num / den) * 100) / 100;
  }
  const val = parseFloat(str);
  return !isNaN(val) && val > 0 ? val : fallback;
}

/**
 * Parse timecode or frame string into seconds.
 * Supports:
 * - HH:MM:SS:FF or MM:SS:FF
 * - Frame count: 'f:120', 'f120', '#120'
 * - Seconds: '12.5'
 * @param {string} str
 * @param {number} fps
 * @returns {number}
 */
export function parseTimecodeString(str, fps = 30) {
  if (!str) return 0;
  const safeFps = Math.max(1, fps || 30);
  const cleaned = String(str).trim();
  if (/^[fF#][: ]*(\d+)$/.test(cleaned)) {
    const f = parseInt(cleaned.replace(/^[fF#][: ]*/, ''), 10);
    return f / safeFps;
  }
  const parts = cleaned.split(':');
  if (parts.length === 4) {
    const [hh, mm, ss, ff] = parts.map(p => parseFloat(p) || 0);
    return hh * 3600 + mm * 60 + ss + (ff / safeFps);
  }
  if (parts.length === 3) {
    const [mm, ss, ff] = parts.map(p => parseFloat(p) || 0);
    return mm * 60 + ss + (ff / safeFps);
  }
  if (parts.length === 2) {
    const [mm, ss] = parts.map(p => parseFloat(p) || 0);
    return mm * 60 + ss;
  }
  const val = parseFloat(cleaned);
  return !isNaN(val) && val >= 0 ? val : 0;
}

/**
 * Convert an array of 2D points into a smooth Catmull-Rom cubic Bézier SVG path string.
 * @param {Array<{ x: number; y: number }>} points
 * @returns {string}
 */
export function pointsToSplinePath(points) {
  if (!points || points.length === 0) return '';
  if (points.length === 1) return `M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`;
  if (points.length === 2) {
    return `M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)} L ${points[1].x.toFixed(1)} ${points[1].y.toFixed(1)}`;
  }

  let d = `M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`;
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(points.length - 1, i + 2)];

    // Catmull-Rom tangent tension (1/6 standard)
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;

    d += ` C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)}, ${cp2x.toFixed(1)} ${cp2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  }
  return d;
}

// Preset color palette for face cluster identities
const CLUSTER_COLORS = [
  '#00f5d4', // Mint
  '#f59e0b', // Amber
  '#a855f7', // Purple
  '#38bdf8', // Sky
  '#f43f5e', // Rose
  '#10b981', // Emerald
  '#eab308', // Yellow
  '#ec4899', // Pink
];

/**
 * Get deterministic color for cluster ID.
 * @param {string} clusterId
 * @returns {string}
 */
export function getClusterColor(clusterId) {
  if (!clusterId) return CLUSTER_COLORS[0];
  let hash = 0;
  for (let i = 0; i < clusterId.length; i++) {
    hash = (hash << 5) - hash + clusterId.charCodeAt(i);
    hash |= 0;
  }
  return CLUSTER_COLORS[Math.abs(hash) % CLUSTER_COLORS.length];
}

/**
 * Chip-shaped inline editable range field matching roop-ultimate.
 * Displays SMPTE timecode and frame number, editable on click.
 */
export function RangeField({ label, frame, time, fps, onCommitFrame, disabled = false }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');

  const handleStart = () => {
    if (disabled) return;
    setDraft(String(frame));
    setEditing(true);
  };

  const handleCommit = () => {
    if (!editing) return;
    setEditing(false);
    const cleaned = String(draft).trim();
    if (!cleaned) return;
    let targetFrame;
    if (/^\d+$/.test(cleaned)) {
      targetFrame = parseInt(cleaned, 10);
    } else {
      const parsed = parseTimecodeString(cleaned, fps);
      targetFrame = Math.round(parsed * fps);
    }
    if (!isNaN(targetFrame)) {
      onCommitFrame(targetFrame);
    }
  };

  return (
    <div
      className={`range-field ${editing ? 'editing' : ''}`}
      title={`Click to edit ${label} point (frame number or HH:MM:SS:FF timecode)`}
    >
      <span className="range-field-label">{label}</span>
      {editing ? (
        <input
          type="text"
          className="range-field-input"
          value={draft}
          autoFocus
          onChange={(e) => setDraft(e.target.value)}
          onBlur={handleCommit}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleCommit();
            if (e.key === 'Escape') setEditing(false);
          }}
        />
      ) : (
        <button
          type="button"
          className="range-field-btn"
          disabled={disabled}
          onClick={handleStart}
        >
          <span className="range-field-tc">{formatTimecode(time, fps).slice(3)}</span>
          <span className="range-field-frame">f:{frame}</span>
        </button>
      )}
    </div>
  );
}

/**
 * Video editor-grade timeline workstation matching roop-ultimate.
 * Includes Filmstrip Track, Timecode Ruler, Draggable Trim In/Out, Playback Engine, and Landmark & Pose Sub-Track.
 *
 * @param {{
 *   parameters?: any;
 *   target?: string;
 *   output?: string;
 *   currentTime?: number;
 *   duration?: number;
 *   fps?: number;
 *   playing?: boolean;
 *   disabled?: boolean;
 *   keys?: number[];
 *   trimStart?: number;
 *   trimEnd?: number | null;
 *   targetVideoRef?: React.RefObject<HTMLVideoElement | null>;
 *   processedVideoRef?: React.RefObject<HTMLVideoElement | null>;
 *   onTogglePlay?: () => void;
 *   onSeek?: (time: number) => void;
 *   onTrimChange?: (trimStart: number, trimEnd: number | null) => void;
 * }} props
 */
export default function TimelineWorkstation({
  parameters = {},
  target = '',
  output = '',
  currentTime = 0,
  duration = 0,
  fps = 30,
  playing = false,
  disabled = false,
  keys = [],
  trimStart = 0,
  trimEnd = null,
  targetVideoRef,
  processedVideoRef,
  onTogglePlay,
  onSeek,
  onTrimChange,
}) {
  const containerRef = useRef(null);
  const trackRef = useRef(null);
  renderProbe('TimelineWorkstation');
  const trackInnerRef = useRef(null);
  const [trackWidth, setTrackWidth] = useState(1000);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [chapters, setChapters] = useState(() => loadChapters(target));
  const [renamingChapter, setRenamingChapter] = useState(null);
  const [isPoseTrackExpanded, setIsPoseTrackExpanded] = useState(true);
  const [activeDrag, setActiveDrag] = useState(null); // 'needle' | 'trim-in' | 'trim-out' | null
  const [hoverTime, setHoverTime] = useState(null);
  const [hoverFace, setHoverFace] = useState(null);

  // Timeline zoom & navigation extensions
  const [timelineZoom, setTimelineZoom] = useState(1); // 1x to 16x
  const [snapToFrames, setSnapToFrames] = useState(true);
  const [isLooping, setIsLooping] = useState(false);
  const [isEditingTimecode, setIsEditingTimecode] = useState(false);
  const [timecodeInput, setTimecodeInput] = useState('');

  // Background thumbnail generation
  const [customThumbnails, setCustomThumbnails] = useState(null);
  const [isGeneratingThumbs, setIsGeneratingThumbs] = useState(false);

  // Access detected faces and clusters from store
  const detectedFaces = useFaceSelectionStore(s => s.detectedFaces);
  const targetClusters = useFaceSelectionStore(s => s.targetClusters);
  const activeSelection = useFaceSelectionStore(s => s.activeSelection);

  const effectiveFps = Math.max(1, fps || 30);
  const effectiveDuration = Math.max(0.1, duration || 1);
  // Chapters belong to the target clip: reload when it changes, persist on every edit.
  useEffect(() => { setChapters(loadChapters(target)); setRenamingChapter(null); }, [target]);
  const updateChapters = useCallback(next => { setChapters(next); saveChapters(target, next); }, [target]);
  // The chapter id is a primitive, so the timeline re-renders on a chapter boundary, not on every playback tick.
  const currentChapterId = usePlayhead(time => chapterAt(chapters, playing ? time : currentTime)?.id ?? null);
  const currentChapter = chapters.find(chapter => chapter.id === currentChapterId) ?? null;
  const liveNow = () => (playing ? getPlayhead() : currentTime);
  const effectiveTrimEnd = trimEnd !== null && trimEnd !== undefined ? trimEnd : effectiveDuration;

  // Track container width via ResizeObserver
  useEffect(() => {
    if (!trackRef.current) return;
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        if (entry.contentRect.width > 0) {
          setTrackWidth(entry.contentRect.width);
        }
      }
    });
    observer.observe(trackRef.current);
    return () => observer.disconnect();
  }, []);

  const scaledTrackWidth = Math.max(trackWidth, Math.round(trackWidth * timelineZoom));

  // Compute 120px-wide video thumbnail frames across the timeline
  const defaultThumbnails = useMemo(() => {
    if (!target || effectiveDuration <= 0) return [];
    const thumbWidth = 120;
    const count = Math.max(3, Math.min(128, Math.floor(Math.max(scaledTrackWidth, 360) / thumbWidth)));
    return Array.from({ length: count }, (_, i) => ({
      index: i,
      timestamp: ((i + 0.5) / count) * effectiveDuration,
      url: `/api/v1/media/preview-frame?${new URLSearchParams({ path: target, timestamp: String(((i + 0.5) / count) * effectiveDuration) })}`,
    }));
  }, [target, effectiveDuration, scaledTrackWidth]);

  const thumbnails = customThumbnails || defaultThumbnails;

  // Update video playback rate
  const handleRateChange = useCallback((rate) => {
    setPlaybackRate(rate);
    if (targetVideoRef?.current) targetVideoRef.current.playbackRate = rate;
    if (processedVideoRef?.current) processedVideoRef.current.playbackRate = rate;
  }, [targetVideoRef, processedVideoRef]);

  // Frame stepping controls
  const handleStepFrame = useCallback((step) => {
    if (disabled || !onSeek) return;
    const delta = step / effectiveFps;
    const nextTime = Math.max(0, Math.min(effectiveDuration, (playing ? getPlayhead() : currentTime) + delta));
    onSeek(nextTime);
  }, [disabled, onSeek, effectiveFps, effectiveDuration, currentTime, playing]);

  // Second jumping controls (Jump -1s / +1s)
  const handleJumpSeconds = useCallback((deltaSec) => {
    if (disabled || !onSeek) return;
    const nextTime = Math.max(0, Math.min(effectiveDuration, (playing ? getPlayhead() : currentTime) + deltaSec));
    onSeek(nextTime);
  }, [disabled, onSeek, effectiveDuration, currentTime, playing]);

  // Jump to Start / In Point
  const handleJumpStart = useCallback(() => {
    if (disabled || !onSeek) return;
    onSeek(trimStart > 0 ? trimStart : 0);
  }, [disabled, onSeek, trimStart]);

  // Jump to End / Out Point
  const handleJumpEnd = useCallback(() => {
    if (disabled || !onSeek) return;
    onSeek(effectiveTrimEnd < effectiveDuration ? effectiveTrimEnd : effectiveDuration);
  }, [disabled, onSeek, effectiveTrimEnd, effectiveDuration]);

  // Reset custom thumbnails when target or duration changes
  useEffect(() => {
    setCustomThumbnails(null);
  }, [target, effectiveDuration]);

  // Calculate time from clientX with optional snapping
  const getTimeFromPointer = useCallback((clientX) => {
    const targetElement = trackInnerRef.current || trackRef.current;
    if (!targetElement) return 0;
    const rect = targetElement.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const rawTime = ratio * effectiveDuration;
    if (snapToFrames) {
      const frameIdx = Math.round(rawTime * effectiveFps);
      return Math.max(0, Math.min(effectiveDuration, frameIdx / effectiveFps));
    }
    return rawTime;
  }, [effectiveDuration, snapToFrames, effectiveFps]);

  // Pointer drag scrubbing & trim dragging
  useEffect(() => {
    if (!activeDrag) return;

    const handlePointerMove = (e) => {
      const time = getTimeFromPointer(e.clientX);
      if (activeDrag === 'needle') {
        onSeek?.(time);
      } else if (activeDrag === 'trim-in') {
        const newStart = Math.max(0, Math.min(effectiveTrimEnd - (1 / effectiveFps), time));
        onTrimChange?.(newStart, trimEnd);
      } else if (activeDrag === 'trim-out') {
        const newEnd = Math.max(trimStart + (1 / effectiveFps), Math.min(effectiveDuration, time));
        onTrimChange?.(trimStart, newEnd >= effectiveDuration ? null : newEnd);
      }
    };

    const handlePointerUp = () => {
      setActiveDrag(null);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [activeDrag, getTimeFromPointer, onSeek, onTrimChange, trimStart, trimEnd, effectiveTrimEnd, effectiveFps, effectiveDuration]);

  // Auto-scroll track viewport to keep playhead in view when zoomed in
  // While playing, the playhead is followed through a transient store subscription (no React render per tick).
  useEffect(() => {
    if (timelineZoom <= 1 || !trackRef.current) return undefined;
    const viewport = trackRef.current;
    const follow = time => {
      const needleX = (time / effectiveDuration) * scaledTrackWidth;
      const viewLeft = viewport.scrollLeft;
      const viewRight = viewLeft + viewport.clientWidth;
      if (needleX < viewLeft + 60 || needleX > viewRight - 60) {
        viewport.scrollLeft = Math.max(0, needleX - viewport.clientWidth / 2);
      }
    };
    follow(currentTime);
    return playing ? playheadStore.subscribeSelector(state => state.time, follow) : undefined;
  }, [currentTime, playing, effectiveDuration, scaledTrackWidth, timelineZoom]);

  // Range loop playback
  useEffect(() => {
    if (!playing || !isLooping) return undefined;
    const wrap = time => { if (time >= effectiveTrimEnd - (1 / effectiveFps)) onSeek?.(trimStart); };
    wrap(currentTime);
    return playheadStore.subscribeSelector(state => state.time, wrap);
  }, [currentTime, playing, isLooping, effectiveTrimEnd, trimStart, effectiveFps, onSeek]);

  // Ctrl+Wheel or Meta+Wheel to zoom timeline
  const handleWheelZoom = useCallback((e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 0.5 : -0.5;
      setTimelineZoom(z => Math.max(1, Math.min(16, Math.round((z + delta) * 2) / 2)));
    }
  }, []);

  // Commit text timecode edit
  const commitTimecodeInput = useCallback(() => {
    if (!timecodeInput) return;
    const parsed = parseTimecodeString(timecodeInput, effectiveFps);
    const clamped = Math.max(0, Math.min(effectiveDuration, parsed));
    onSeek?.(clamped);
  }, [timecodeInput, effectiveFps, effectiveDuration, onSeek]);

  // Trim In / Out button shortcuts
  const handleMarkIn = () => {
    if (disabled || !onTrimChange) return;
    const newStart = Math.min(liveNow(), effectiveTrimEnd - (1 / effectiveFps));
    onTrimChange(newStart, trimEnd);
  };

  const handleMarkOut = () => {
    if (disabled || !onTrimChange) return;
    const newEnd = Math.max(liveNow(), trimStart + (1 / effectiveFps));
    onTrimChange(trimStart, newEnd >= effectiveDuration ? null : newEnd);
  };

  const handleResetTrim = () => {
    if (disabled || !onTrimChange) return;
    onTrimChange(0, null);
  };

  const handleJumpIn = useCallback(() => {
    if (disabled || !onSeek) return;
    onSeek(trimStart > 0 ? trimStart : 0);
  }, [disabled, onSeek, trimStart]);

  const handleJumpOut = useCallback(() => {
    if (disabled || !onSeek) return;
    onSeek(effectiveTrimEnd < effectiveDuration ? effectiveTrimEnd : effectiveDuration);
  }, [disabled, onSeek, effectiveTrimEnd, effectiveDuration]);

  // NLE keyboard shortcuts (mark / jump in-out, reset range) are handled by the central, remappable hotkey manager
  // (`markIn`, `markOut`, `jumpIn`, `jumpOut`, `resetRange` in Hotkeys.tsx, bound in VideoViewport), not hardcoded here.

  // Timecode Ruler Major and Minor Ticks
  const rulerTicks = useMemo(() => {
    if (effectiveDuration <= 0) return [];
    // Adapt step: choose tick interval so ticks are spaced ~60px to 100px apart
    const desiredTickCount = Math.max(4, Math.floor(scaledTrackWidth / 80));
    const rawStep = effectiveDuration / desiredTickCount;

    // Standard time intervals: 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60
    const intervals = [0.04, 0.08, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    let chosenInterval = intervals[intervals.length - 1];
    for (const interval of intervals) {
      if (interval >= rawStep) {
        chosenInterval = interval;
        break;
      }
    }

    const ticks = [];
    const count = Math.ceil(effectiveDuration / chosenInterval);
    for (let i = 0; i <= count; i++) {
      const t = Math.min(effectiveDuration, i * chosenInterval);
      const ratio = (t / effectiveDuration) * 100;
      ticks.push({
        time: t,
        ratio,
        label: formatTimecode(t, effectiveFps),
        isMajor: true,
      });

      // Minor ticks (halfway)
      if (i < count) {
        const halfT = t + chosenInterval / 2;
        if (halfT < effectiveDuration) {
          ticks.push({
            time: halfT,
            ratio: (halfT / effectiveDuration) * 100,
            label: '',
            isMajor: false,
          });
        }
      }
    }
    return ticks;
  }, [effectiveDuration, scaledTrackWidth, effectiveFps]);

  // Head Yaw continuous spline data
  const yawSplineData = useMemo(() => {
    if (!detectedFaces || detectedFaces.length === 0 || effectiveDuration <= 0) {
      return { path: '', points: [], hasFaces: false };
    }

    // Sort detected faces by timestamp
    const sorted = [...detectedFaces].sort((a, b) => a.timestamp - b.timestamp);
    const splineHeight = 54;
    const centerY = splineHeight / 2;
    const maxDegree = 70; // Map ±70° to track height

    const points = sorted.map(face => {
      const x = (face.timestamp / effectiveDuration) * Math.max(scaledTrackWidth, 100);
      const yaw = face.pose?.yaw ?? 0;
      // Clamp mapped yaw to visible track
      const clampedYaw = Math.max(-maxDegree, Math.min(maxDegree, yaw));
      const y = centerY - (clampedYaw / maxDegree) * (centerY * 0.78);
      return {
        x,
        y,
        face,
        yaw,
        time: face.timestamp,
      };
    });

    const path = pointsToSplinePath(points);
    return { path, points, hasFaces: true };
  }, [detectedFaces, effectiveDuration, scaledTrackWidth]);

  // Click on a detected face marker: seek playhead and focus 3D viewer
  const handleFaceMarkerClick = (face, e) => {
    e.stopPropagation();
    if (disabled) return;
    onSeek?.(face.timestamp);
    useFaceSelectionStore.getState().setActiveSelection(face.id);
  };

  // Trim handle percentages (the playhead needle is a live leaf: see LiveNeedle)
  const trimInPercent = Math.max(0, Math.min(100, (trimStart / effectiveDuration) * 100));
  const trimOutPercent = Math.max(0, Math.min(100, (effectiveTrimEnd / effectiveDuration) * 100));

  return (
    <div
      className="timeline-workstation"
      ref={containerRef}
      tabIndex={0}
      role="region"
      aria-label="Video Editor Workstation Timeline"
    >
      {/* 1. Playback Engine Transport & Speed Bar */}
      <div className="timeline-engine-bar">
        {/* Play / Pause Toggle */}
        <button
          type="button"
          className={`timeline-play-btn ${playing ? 'playing' : ''}`}
          disabled={disabled || !target}
          onClick={onTogglePlay}
          title={playing ? 'Pause Playback [Space]' : 'Start Playback [Space]'}
          aria-label={playing ? 'Pause playback' : 'Play playback'}
        >
          <span className="btn-icon">{playing ? '❚❚' : '▶'}</span>
        </button>

        {/* Frame Stepping & Jump Transport Controls */}
        <div className="stepping-group">
          <button
            type="button"
            className="stepping-btn jump-btn"
            disabled={disabled || !target || currentTime <= 0}
            onClick={handleJumpStart}
            title="Jump to Start / In Point [Home]"
            aria-label="Jump to Start"
          >
            |◀
          </button>
          <button
            type="button"
            className="stepping-btn jump-btn"
            disabled={disabled || !target || currentTime <= 0}
            onClick={() => handleJumpSeconds(-1)}
            title="Jump -1 Second [Shift+Left]"
            aria-label="Jump -1 Second"
          >
            ◀◀
          </button>
          <button
            type="button"
            className="stepping-btn"
            disabled={disabled || !target || currentTime <= 0}
            onClick={() => handleStepFrame(-1)}
            title="Step -1 Frame [Left Arrow]"
            aria-label="Step -1 Frame"
          >
            ◀| -1f
          </button>
          <button
            type="button"
            className="stepping-btn"
            disabled={disabled || !target || currentTime >= effectiveDuration}
            onClick={() => handleStepFrame(1)}
            title="Step +1 Frame [Right Arrow]"
            aria-label="Step +1 Frame"
          >
            +1f |▶
          </button>
          <button
            type="button"
            className="stepping-btn jump-btn"
            disabled={disabled || !target || currentTime >= effectiveDuration}
            onClick={() => handleJumpSeconds(1)}
            title="Jump +1 Second [Shift+Right]"
            aria-label="Jump +1 Second"
          >
            ▶▶
          </button>
          <button
            type="button"
            className="stepping-btn jump-btn"
            disabled={disabled || !target || currentTime >= effectiveDuration}
            onClick={handleJumpEnd}
            title="Jump to End / Out Point [End]"
            aria-label="Jump to End"
          >
            ▶|
          </button>
        </div>

        {/* Timecode Readout Box (Click-to-Edit) */}
        <div className="timecode-readout-box" title="Click to enter exact timecode or frame number">
          {isEditingTimecode ? (
            <input
              type="text"
              autoFocus
              className="timecode-inline-input"
              value={timecodeInput}
              onChange={(e) => setTimecodeInput(e.target.value)}
              onBlur={() => {
                commitTimecodeInput();
                setIsEditingTimecode(false);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  commitTimecodeInput();
                  setIsEditingTimecode(false);
                } else if (e.key === 'Escape') {
                  setIsEditingTimecode(false);
                }
              }}
            />
          ) : (
            <span
              className="current-tc"
              title="Click to jump to exact Timecode or Frame [HH:MM:SS:FF]"
              onClick={() => {
                if (!disabled) {
                  setTimecodeInput(formatTimecode(liveNow(), effectiveFps));
                  setIsEditingTimecode(true);
                }
              }}
            >
              <LiveTimecode currentTime={currentTime} playing={playing} fps={effectiveFps} />
            </span>
          )}
          <span className="tc-separator">/</span>
          <span className="total-tc" title="Total Clip Duration (HH:MM:SS:FF)">
            {formatTimecode(effectiveDuration, effectiveFps)}
          </span>
          <span className="frame-counter">
            <LiveFrameCounter currentTime={currentTime} playing={playing} fps={effectiveFps} total={Math.floor(effectiveDuration * effectiveFps)} />
          </span>
        </div>

        {/* Snap to Frames Toggle */}
        <button
          type="button"
          className={`timeline-snap-btn ${snapToFrames ? 'active' : ''}`}
          onClick={() => setSnapToFrames(s => !s)}
          title={`Snap to Frame Boundaries: ${snapToFrames ? 'ON' : 'OFF'}`}
        >
          <span>🧲</span> Snap
        </button>

        {/* Loop Playback Toggle */}
        <button
          type="button"
          className={`timeline-loop-btn ${isLooping ? 'active' : ''}`}
          onClick={() => setIsLooping(l => !l)}
          title={`Loop Range Playback: ${isLooping ? 'ON' : 'OFF'}`}
        >
          <span>🔁</span> Loop
        </button>

        {/* Variable Speed Controller */}
        <div className="speed-controller" role="group" aria-label="Playback Speed">
          <span className="speed-label">SPEED:</span>
          <div className="speed-segmented">
            {[0.25, 0.5, 1, 2, 4].map(rate => (
              <button
                key={rate}
                type="button"
                className={`speed-btn ${playbackRate === rate ? 'active' : ''}`}
                onClick={() => handleRateChange(rate)}
                disabled={disabled || !target}
                title={`Set playback speed to ${rate}x`}
              >
                {rate}x
              </button>
            ))}
          </div>
        </div>

        {/* Chapter markers */}
        <div className="chapter-actions-group" role="group" aria-label="Chapters">
          <button type="button" className="chapter-btn add" disabled={disabled || !target} title="Add a chapter marker at the playhead"
            onClick={() => updateChapters(addChapter(chapters, liveNow(), { duration: effectiveDuration, fps: effectiveFps }))}>＋ Chapter</button>
          <button type="button" className="chapter-btn prev" disabled={disabled || !adjacentChapter(chapters, currentTime, -1)} title="Previous chapter"
            onClick={() => { const chapter = adjacentChapter(chapters, currentTime, -1); if (chapter) onSeek?.(chapter.time); }}>⏮</button>
          <button type="button" className="chapter-btn next" disabled={disabled || !adjacentChapter(chapters, currentTime, 1)} title="Next chapter"
            onClick={() => { const chapter = adjacentChapter(chapters, currentTime, 1); if (chapter) onSeek?.(chapter.time); }}>⏭</button>
          {currentChapter && <span className="chapter-current" title="Current chapter">{currentChapter.label}</span>}
        </div>

        {/* Range In / Out Trim Buttons */}
        <div className="trim-actions-group">
          <button
            type="button"
            className="trim-btn jump-in"
            onClick={handleJumpIn}
            disabled={disabled || !target}
            title="Jump to In Point [Shift+I]"
          >
            ⏮ In
          </button>
          <button
            type="button"
            className="trim-btn mark-in"
            onClick={handleMarkIn}
            disabled={disabled || !target}
            title="Set Range In Point to Playhead [I]"
          >
            [ In
          </button>
          <RangeField
            label="In"
            frame={Math.round(trimStart * effectiveFps)}
            time={trimStart}
            fps={effectiveFps}
            disabled={disabled || !target}
            onCommitFrame={(f) => {
              const sec = Math.max(0, f / effectiveFps);
              onTrimChange?.(sec, trimEnd);
            }}
          />
          <RangeField
            label="Out"
            frame={Math.round(effectiveTrimEnd * effectiveFps)}
            time={effectiveTrimEnd}
            fps={effectiveFps}
            disabled={disabled || !target}
            onCommitFrame={(f) => {
              const sec = Math.min(effectiveDuration, Math.max(trimStart + 0.01, f / effectiveFps));
              onTrimChange?.(trimStart, sec);
            }}
          />
          <button
            type="button"
            className="trim-btn mark-out"
            onClick={handleMarkOut}
            disabled={disabled || !target}
            title="Set Range Out Point to Playhead [O]"
          >
            Out ]
          </button>
          <button
            type="button"
            className="trim-btn jump-out"
            onClick={handleJumpOut}
            disabled={disabled || !target}
            title="Jump to Out Point [Shift+O]"
          >
            Out ⏭
          </button>
          {(trimStart > 0 || trimEnd !== null) && (
            <button
              type="button"
              className="trim-btn reset"
              onClick={handleResetTrim}
              disabled={disabled}
              title="Reset trim range to full clip [Alt+R]"
            >
              Reset
            </button>
          )}
          <div className="trim-range-summary" title="Length of selected in-out range">
            <span className="summary-len">{formatTimecode(Math.max(0, effectiveTrimEnd - trimStart), effectiveFps).slice(3)}</span>
            <small className="summary-frames">({Math.max(1, Math.round((effectiveTrimEnd - trimStart) * effectiveFps))} f · {Math.round(((effectiveTrimEnd - trimStart) / effectiveDuration) * 100)}%)</small>
          </div>
        </div>

        {/* Timeline Zoom Controller */}
        <div className="timeline-zoom-controller" role="group" aria-label="Timeline Zoom">
          <span className="zoom-label">ZOOM:</span>
          <button
            type="button"
            className="zoom-step-btn"
            disabled={disabled || timelineZoom <= 1}
            onClick={() => setTimelineZoom(z => Math.max(1, Math.round((z - 0.5) * 2) / 2))}
            title="Zoom Out Timeline [-]"
          >
            -
          </button>
          <input
            type="range"
            min="1"
            max="16"
            step="0.5"
            value={timelineZoom}
            onChange={(e) => setTimelineZoom(parseFloat(e.target.value))}
            className="zoom-slider"
            title={`Timeline Zoom: ${timelineZoom.toFixed(1)}x (Ctrl+Wheel to zoom)`}
          />
          <button
            type="button"
            className="zoom-step-btn"
            disabled={disabled || timelineZoom >= 16}
            onClick={() => setTimelineZoom(z => Math.min(16, Math.round((z + 0.5) * 2) / 2))}
            title="Zoom In Timeline [+]"
          >
            +
          </button>
          {timelineZoom > 1 && (
            <button
              type="button"
              className="zoom-fit-btn"
              onClick={() => setTimelineZoom(1)}
              title="Reset Timeline Zoom to Fit [1x]"
            >
              Fit
            </button>
          )}
          <span className="zoom-tag">{timelineZoom.toFixed(1)}x</span>
        </div>

        <div className="transport-flex-spacer" />

        {/* Sub-Track Collapse / Expand Toggle */}
        <button
          type="button"
          className={`pose-track-toggle ${isPoseTrackExpanded ? 'expanded' : ''}`}
          onClick={() => setIsPoseTrackExpanded(prev => !prev)}
          title="Toggle Landmark & Pose Sub-Track"
        >
          <span className="toggle-chevron">{isPoseTrackExpanded ? '▼' : '▶'}</span>
          <span>LANDMARKS & POSE</span>
          <span className="badge-count">{detectedFaces.length}</span>
        </button>
      </div>

      {/* 2. Interactive Timeline Canvas Tracks Container */}
      <div
        className="timeline-tracks-viewport"
        ref={trackRef}
        onWheel={handleWheelZoom}
        onPointerDown={(e) => {
          if (disabled || !target) return;
          const time = getTimeFromPointer(e.clientX);
          onSeek?.(time);
          setActiveDrag('needle');
        }}
        onPointerMove={(e) => {
          if (disabled || !target) return;
          setHoverTime(getTimeFromPointer(e.clientX));
        }}
        onPointerLeave={() => {
          setHoverTime(null);
          setHoverFace(null);
        }}
      >
        <div
          className="timeline-tracks-inner"
          ref={trackInnerRef}
          style={{ width: `${scaledTrackWidth}px`, minWidth: '100%', position: 'relative' }}
        >
        {/* Timecode Ruler */}
        <div className="timecode-ruler-track" aria-label="Timecode Ruler">
          {rulerTicks.map((tick, idx) => (
            <div
              key={idx}
              className={`ruler-tick ${tick.isMajor ? 'major' : 'minor'}`}
              style={{ left: `${tick.ratio}%` }}
            >
              {tick.isMajor && <span className="tick-label">{tick.label}</span>}
            </div>
          ))}
        </div>

        {/* Chapter markers on the ruler: click to seek, double-click to rename, × to remove */}
        <div className="chapter-markers-layer" role="group" aria-label="Chapter markers">
          {chapters.map(chapter => (
            <div key={chapter.id} className={`chapter-marker ${currentChapter?.id === chapter.id ? 'current' : ''}`} style={{ left: `${Math.min(100, (chapter.time / effectiveDuration) * 100)}%` }}>
              {renamingChapter === chapter.id ? (
                <input className="chapter-rename" aria-label="Chapter name" autoFocus defaultValue={chapter.label} maxLength={48}
                  onBlur={event => { updateChapters(renameChapter(chapters, chapter.id, event.target.value)); setRenamingChapter(null); }}
                  onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur(); if (event.key === 'Escape') setRenamingChapter(null); }} />
              ) : (
                <>
                  <button type="button" className="chapter-flag" title={`${chapter.label} · ${formatTimecode(chapter.time, effectiveFps)} (double-click to rename)`}
                    aria-label={`Chapter ${chapter.label} at ${formatTimecode(chapter.time, effectiveFps)}`}
                    onClick={event => { event.stopPropagation(); if (!disabled) onSeek?.(chapter.time); }} onDoubleClick={event => { event.stopPropagation(); setRenamingChapter(chapter.id); }}>
                    ⚑ <span>{chapter.label}</span>
                  </button>
                  <button type="button" className="chapter-remove" aria-label={`Remove chapter ${chapter.label}`}
                    onClick={event => { event.stopPropagation(); updateChapters(removeChapter(chapters, chapter.id)); }}>×</button>
                </>
              )}
            </div>
          ))}
        </div>

        {/* Filmstrip Track (120px-wide thumbnail frames) */}
        <div className="filmstrip-track" aria-label="Video Filmstrip Track">
          <div className="sprocket-holes top" />
          <div className="filmstrip-frames-row">
            {thumbnails.length > 0 ? (
              thumbnails.map((thumb) => (
                <div
                  key={thumb.index}
                  className="filmstrip-frame-card"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (!disabled) onSeek?.(thumb.timestamp);
                  }}
                  title={`Seek to ${formatTimecode(thumb.timestamp, effectiveFps)}`}
                >
                  <img
                    src={thumb.url}
                    alt={`Frame at ${thumb.timestamp.toFixed(1)}s`}
                    loading="lazy"
                    className="filmstrip-thumb-img"
                    onError={(e) => {
                      // Fallback placeholder styling
                      e.currentTarget.style.display = 'none';
                    }}
                  />
                  <span className="frame-timestamp-pill">
                    {formatTimecode(thumb.timestamp, effectiveFps).slice(3)}
                  </span>
                </div>
              ))
            ) : (
              <div className="filmstrip-empty">
                <span>{target ? 'Generating 120px filmstrip frames…' : 'Load a target video to display filmstrip'}</span>
              </div>
            )}
          </div>
          <div className="sprocket-holes bottom" />
        </div>

        {/* Range In / Out Markers & Shaded Dim Overlays */}
        <div className="trim-range-layer">
          {/* Dimmed Inactive Left Region */}
          <div
            className="trim-dimmed-region left"
            style={{ width: `${trimInPercent}%` }}
          />
          {/* Active Highlighted Range */}
          <div
            className="trim-active-region"
            style={{
              left: `${trimInPercent}%`,
              width: `${Math.max(0, trimOutPercent - trimInPercent)}%`,
            }}
          />
          {/* Dimmed Inactive Right Region */}
          <div
            className="trim-dimmed-region right"
            style={{
              left: `${trimOutPercent}%`,
              width: `${Math.max(0, 100 - trimOutPercent)}%`,
            }}
          />

          {/* Draggable In Trim Handle */}
          <div
            className="trim-handle-marker in"
            style={{ left: `${trimInPercent}%` }}
            onPointerDown={(e) => {
              e.stopPropagation();
              if (!disabled) setActiveDrag('trim-in');
            }}
            title={`Drag Range In Point [${formatTimecode(trimStart, effectiveFps)}]`}
          >
            <div className="trim-handle-tab in">
              <span className="handle-tag">IN</span>
              <span className="handle-tc">{formatTimecode(trimStart, effectiveFps).slice(3)}</span>
            </div>
            <div className="trim-handle-line in" />
          </div>

          {/* Draggable Out Trim Handle */}
          <div
            className="trim-handle-marker out"
            style={{ left: `${trimOutPercent}%` }}
            onPointerDown={(e) => {
              e.stopPropagation();
              if (!disabled) setActiveDrag('trim-out');
            }}
            title={`Drag Range Out Point [${formatTimecode(effectiveTrimEnd, effectiveFps)}]`}
          >
            <div className="trim-handle-tab out">
              <span className="handle-tag">OUT</span>
              <span className="handle-tc">{formatTimecode(effectiveTrimEnd, effectiveFps).slice(3)}</span>
            </div>
            <div className="trim-handle-line out" />
          </div>
        </div>

        {/* Sampled Keyframe Markers */}
        <div className="keyframe-markers-layer">
          {keys
            .filter(at => at >= 0 && at <= effectiveDuration)
            .map((at, index) => (
              <button
                key={index}
                type="button"
                className="keyframe"
                style={{ left: `${(at / effectiveDuration) * 100}%` }}
                title={`Keyframe at ${formatTimecode(at, effectiveFps)}`}
                aria-label={`Seek keyframe ${formatTimecode(at, effectiveFps)}`}
                onClick={(e) => {
                  e.stopPropagation();
                  if (!disabled) onSeek?.(at);
                }}
              >
                ◆
              </button>
            ))}
        </div>

        {/* 3. Landmark & Pose Collapsible Sub-Track */}
        {isPoseTrackExpanded && (
          <div className="landmark-pose-subtrack" aria-label="Landmark & Pose Track">
            <div className="subtrack-header-row">
              <span className="subtrack-title">HEAD YAW & IDENTITY CONTINUITY</span>
              <div className="subtrack-legend">
                <span className="legend-tag frontal"><i className="legend-dot frontal" /> Frontal (±20°)</span>
                <span className="legend-tag profile"><i className="legend-dot profile" /> Profile (&gt;45°)</span>
              </div>
            </div>

            <div className="pose-graph-container">
              {/* Reference Grid Lines */}
              <div className="yaw-guide-line top" title="+45° Profile Right"><span>+45°</span></div>
              <div className="yaw-guide-line center" title="0° Frontal Center"><span>0°</span></div>
              <div className="yaw-guide-line bottom" title="-45° Profile Left"><span>-45°</span></div>

              {/* Continuous Spline Graph of Head Yaw */}
              {yawSplineData.hasFaces && (
                <svg
                  className="yaw-spline-svg"
                  viewBox={`0 0 ${Math.max(trackWidth, 100)} 54`}
                  preserveAspectRatio="none"
                >
                  <defs>
                    <linearGradient id="yawGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stopColor="#f43f5e" />
                      <stop offset="25%" stopColor="#f59e0b" />
                      <stop offset="50%" stopColor="#10b981" />
                      <stop offset="75%" stopColor="#f59e0b" />
                      <stop offset="100%" stopColor="#f43f5e" />
                    </linearGradient>
                  </defs>

                  {/* Shaded Area Beneath Frontal Zone */}
                  <rect
                    x="0"
                    y="18"
                    width={trackWidth}
                    height="18"
                    fill="rgba(16, 185, 129, 0.07)"
                  />

                  {/* Spline Path */}
                  <path
                    d={yawSplineData.path}
                    fill="none"
                    stroke="url(#yawGradient)"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="yaw-spline-path"
                  />
                </svg>
              )}

              {/* Colored Face Markers along the track */}
              <div className="face-markers-overlay">
                {detectedFaces.map(face => {
                  const percent = (face.timestamp / effectiveDuration) * 100;
                  const clusterColor = getClusterColor(face.clusterId || 'cluster_0');
                  const isSelected = activeSelection === face.id;

                  return (
                    <button
                      key={face.id}
                      type="button"
                      className={`face-marker-pin ${isSelected ? 'selected' : ''}`}
                      style={{
                        left: `${percent}%`,
                        backgroundColor: clusterColor,
                        boxShadow: isSelected ? `0 0 10px ${clusterColor}, 0 0 16px ${clusterColor}` : undefined,
                      }}
                      onClick={(e) => handleFaceMarkerClick(face, e)}
                      onMouseEnter={() => setHoverFace(face)}
                      onMouseLeave={() => setHoverFace(null)}
                      title={`Face ${face.id} (${(face.pose?.yaw ?? 0).toFixed(1)}° yaw) - Click to focus 3D Pose`}
                      aria-label={`Face marker at ${formatTimecode(face.timestamp, effectiveFps)}`}
                    >
                      <span className="marker-dot" />
                    </button>
                  );
                })}
              </div>

              {/* Hover Face Tooltip */}
              {hoverFace && (
                <div
                  className="face-hover-popover"
                  style={{ left: `${Math.min(92, Math.max(8, (hoverFace.timestamp / effectiveDuration) * 100))}%` }}
                >
                  {hoverFace.thumbnailUrl && (
                    <img src={hoverFace.thumbnailUrl} alt="Face thumbnail" className="popover-thumb" />
                  )}
                  <div className="popover-info">
                    <strong>{hoverFace.id.split('_').slice(0, 2).join(' ')}</strong>
                    <small>{formatTimecode(hoverFace.timestamp, effectiveFps)} (F: {hoverFace.frameIndex})</small>
                    <div className="popover-angles">
                      <span>Yaw: <b>{(hoverFace.pose?.yaw ?? 0).toFixed(1)}°</b></span>
                      <span>Pitch: <b>{(hoverFace.pose?.pitch ?? 0).toFixed(1)}°</b></span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Hover Needle Indicator */}
        {hoverTime !== null && (
          <div
            className="timeline-hover-needle"
            style={{ left: `${(hoverTime / effectiveDuration) * 100}%` }}
          >
            <span className="hover-timecode-tooltip">
              {formatTimecode(hoverTime, effectiveFps)}
            </span>
          </div>
        )}

        {/* Drag-to-Scrub Playhead Needle */}
        <LiveNeedle currentTime={currentTime} playing={playing} fps={effectiveFps} duration={effectiveDuration} dragging={activeDrag === 'needle'} />

        {/* Fallback range input for full accessibility & test compliance */}
        <LiveScrubber currentTime={currentTime} playing={playing} fps={effectiveFps} duration={effectiveDuration} disabled={disabled || !target} onSeek={onSeek} />
        </div>
      </div>

      <LiveCurveEditor currentTime={currentTime} playing={playing} fps={effectiveFps} totalFrames={Math.max(1, Math.round(duration * effectiveFps))} parameters={parameters} disabled={disabled || !target} onSeekFrame={frame => onSeek?.(frame / effectiveFps)}/>
      {/* Timeline Footer Info */}
      <div className="timeline-workstation-foot">
        <span className="foot-left">
          <span className="accent-tag">◆</span> Frame-accurate SMPTE ruler · Drag playhead or click any thumbnail to scrub
        </span>
        <span className="foot-right">
          Export Range: <b>{formatTimecode(trimStart, effectiveFps)}</b> → <b>{formatTimecode(effectiveTrimEnd, effectiveFps)}</b> ({((effectiveTrimEnd - trimStart)).toFixed(2)}s)
        </span>
      </div>
    </div>
  );
}
