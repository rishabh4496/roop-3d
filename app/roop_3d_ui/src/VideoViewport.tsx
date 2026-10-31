import { useEffect, useRef, useState, useCallback } from 'react';
import { mediaURL } from './api';
import { PreviewURL } from './state';
import type { FaceSet, Probe } from './contracts';
import FaceOverlayCanvas from './components/canvas/FaceOverlayCanvas';
import FaceSelectionTray from './components/tray/FaceSelectionTray';
import WorkstationCanvas from './components/canvas/WorkstationCanvas';
import MaskBrushOverlay from './components/canvas/MaskBrushOverlay';

function usePreview(path: string, time: number, playing: boolean) {
  const [source, setSource] = useState('');
  const [error, setError] = useState('');
  useEffect(() => {
    setSource(''); setError('');
    if (!path || playing) return;
    const controller = new AbortController();
    const lease = new PreviewURL();
    let active = true;
    const timer = setTimeout(() => {
      void fetch(`/api/v1/media/preview-frame?${new URLSearchParams({ path, timestamp: String(time) })}`, { signal: controller.signal, cache: 'no-store' })
        .then(async response => { if (!response.ok) throw new Error('Frame preview unavailable at this position'); const blob = await response.blob(); if (active) setSource(lease.replace(blob)); })
        .catch(error => { if (active && error.name !== 'AbortError') setError(error.message); });
    }, 100);
    return () => { active = false; clearTimeout(timer); controller.abort(); lease.clear(); };
  }, [path, time, playing]);
  return { source, error };
}

export function VideoViewport({
  target,
  output,
  time,
  setTime,
  duration,
  enabled,
  keys,
  probe = null,
  sources = [],
  onAssignSource,
  onMaskChange,
}: {
  target: string;
  output: string;
  time: number;
  setTime: (time: number) => void;
  duration: number;
  enabled: boolean;
  keys: number[];
  probe?: Probe | null;
  sources?: FaceSet[];
  onAssignSource?: (clusterId: string, sourceId: string | null) => void;
  onMaskChange?: (maskBase64: string | null) => void;
}) {
  const original = useRef<HTMLVideoElement>(null), processed = useRef<HTMLVideoElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);

  const [playing, setPlaying] = useState(false);
  const [playbackError, setPlaybackError] = useState('');

  // Comparison & Masking Modes
  const [compMode, setCompMode] = useState<'wipe' | 'diff' | 'blend' | 'loupe'>('wipe');
  const [isMaskEditing, setIsMaskEditing] = useState(false);
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });

  const sourcePreview = usePreview(enabled ? target : '', time, playing);
  const outputPreview = usePreview(enabled ? output : '', time, playing);

  const updateViewportSize = useCallback(() => {
    if (viewportRef.current) {
      const rect = viewportRef.current.getBoundingClientRect();
      setViewportSize({ width: rect.width, height: rect.height });
    }
  }, []);

  useEffect(() => {
    updateViewportSize();
    window.addEventListener('resize', updateViewportSize);
    return () => window.removeEventListener('resize', updateViewportSize);
  }, [updateViewportSize]);

  useEffect(() => { setPlaying(false); original.current?.pause(); processed.current?.pause(); setPlaybackError(''); }, [target, output]);
  useEffect(() => { for (const video of [original.current, processed.current]) if (video && Math.abs(video.currentTime - time) > .2) video.currentTime = time; }, [time]);

  const toggle = async () => {
    if (playing) { original.current?.pause(); processed.current?.pause(); setPlaying(false); return; }
    try { await original.current?.play(); await processed.current?.play(); setPlaying(true); }
    catch { setPlaybackError('Browser playback is unavailable for this codec. Use frame previews to inspect it.'); }
  };

  const mediaDims = probe ? { width: probe.width, height: probe.height } : { width: 1920, height: 1080 };

  const pane = (path: string, result: typeof sourcePreview, isOutput: boolean) => (
    <div
      className={`video-pane ${isOutput ? 'processed' : 'original'}`}
      style={isOutput && output ? undefined : undefined}
    >
      {path ? (
        <>
          <video
            ref={isOutput ? processed : original}
            src={mediaURL(path)}
            preload="metadata"
            playsInline
            muted
            onEnded={() => setPlaying(false)}
            onError={() => setPlaybackError('Browser playback unavailable; frame previews can still be used.')}
            onTimeUpdate={!isOutput ? event => { if (playing) setTime(event.currentTarget.currentTime); } : undefined}
          />
          {!playing && result.source && (
            <img
              className="frame-preview"
              src={result.source}
              alt={`${isOutput ? 'Processed' : 'Source'} frame at ${time.toFixed(2)} seconds`}
            />
          )}
          {!isOutput && !isMaskEditing && (
            <FaceOverlayCanvas
              videoRef={original}
              mediaDimensions={mediaDims}
              currentTime={time}
              sources={sources}
              onAssignSource={onAssignSource}
            />
          )}
        </>
      ) : (
        <div className="viewport-empty">
          <span className="film-icon">{isOutput ? '◇' : '⊞'}</span>
          <h2>{isOutput ? 'Your output starts here' : 'Bring a scene to life'}</h2>
          <p>{isOutput ? 'Completed renders appear alongside your source.' : 'Import a local video, inspect its faces, and assign an identity.'}</p>
        </div>
      )}
      <span className="pane-label">
        {isOutput ? 'OUTPUT' : 'SOURCE'} <i className={`dot ${path ? 'mint' : 'dim'}`} />
      </span>
      {result.error && <span className="preview-error">{result.error}</span>}
    </div>
  );

  return (
    <section className="viewport-section">
      <div className="viewport-toolbar">
        <span className="eyebrow">WORKSTATION MEDIA INSPECTOR</span>

        {/* Mask Editing Mode Toggle */}
        <button
          type="button"
          className={`mask-toggle-btn ${isMaskEditing ? 'active' : ''}`}
          disabled={!target}
          onClick={() => setIsMaskEditing(e => !e)}
          title="Manual & Occlusion Mask Editor (Paint keep-target areas & auto-detect occluders)"
        >
          <span>✎</span> {isMaskEditing ? 'Close Mask Editor' : 'Edit Mask'}
        </button>

        <span className="muted">Source timing preserved</span>
      </div>

      <div className="viewport dual" ref={viewportRef}>
        {pane(target, sourcePreview, false)}
        {pane(output, outputPreview, true)}

        {/* Workstation Comparison Canvas (Wipe, Diff Map, Alpha Blend, Magnifier Loupe) */}
        {!isMaskEditing && (
          <WorkstationCanvas
            target={target}
            output={output}
            targetMediaRef={original}
            outputMediaRef={processed}
            sourcePreviewUrl={sourcePreview.source}
            outputPreviewUrl={outputPreview.source}
            mode={compMode}
            onModeChange={setCompMode}
            containerWidth={viewportSize.width}
            containerHeight={viewportSize.height}
            mediaDimensions={mediaDims}
          />
        )}

        {/* Manual & Occlusion Mask Editor Overlay */}
        {isMaskEditing && (
          <MaskBrushOverlay
            containerWidth={viewportSize.width}
            containerHeight={viewportSize.height}
            mediaDimensions={mediaDims}
            onMaskChange={onMaskChange}
            onClose={() => setIsMaskEditing(false)}
          />
        )}
      </div>

      {playbackError && <p className="inline-warning">{playbackError}</p>}

      <div className="transport">
        <button
          className="play-button"
          disabled={!target || !enabled}
          onClick={() => void toggle()}
          aria-label={playing ? 'Pause playback' : 'Play playback'}
        >
          {playing ? 'Ⅱ' : '▶'}
        </button>
        <time>{formatTime(time)}</time>
        <span className="muted">/ {formatTime(duration)}</span>
        <div className="transport-spacer" />
        <span className="tiny-badge">LOCAL PREVIEW</span>
      </div>

      <div className="timeline">
        <div className="timeline-ruler">
          <span>00:00</span>
          <span>{formatTime(duration / 2)}</span>
          <span>{formatTime(duration)}</span>
        </div>
        <input
          aria-label="Timeline scrubber"
          type="range"
          min="0"
          max={duration || 1}
          step="0.01"
          value={Math.min(time, duration || 1)}
          disabled={!target || !enabled}
          onChange={event => {
            setPlaying(false);
            original.current?.pause();
            processed.current?.pause();
            setTime(+event.target.value);
          }}
        />
        <div className="timeline-clip">
          <span>{target ? target.split('/').at(-1) : 'No target video'}</span>
          {keys
            .filter(at => at >= 0 && at <= duration)
            .map((at, index) => (
              <button
                key={index}
                className="keyframe"
                style={{ left: `${duration ? (at / duration) * 100 : 0}%` }}
                title={`Keyframe ${formatTime(at)}`}
                aria-label={`Seek keyframe ${formatTime(at)}`}
                onClick={() => {
                  setPlaying(false);
                  original.current?.pause();
                  processed.current?.pause();
                  setTime(at);
                }}
              >
                ◆
              </button>
            ))}
        </div>
        <div className="timeline-foot">
          <span>◆ Sampled keyframes</span>
          <span>Scrub for decoded frame preview</span>
        </div>
      </div>

      {/* Target Face Selection Tray */}
      <FaceSelectionTray
        currentTime={time}
        sources={sources}
        onSeekTime={setTime}
        onAssignSource={onAssignSource}
      />
    </section>
  );
}

export function formatTime(value: number) {
  return `${Math.floor(value / 60).toString().padStart(2, '0')}:${(value % 60).toFixed(2).padStart(5, '0')}`;
}
