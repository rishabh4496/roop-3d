import { useEffect, useRef, useState, useCallback } from 'react';
import { mediaURL } from './api';
import { PreviewURL } from './state';
import type { FaceSet, Probe, Inspection, FrameFaceDetectionResponse } from './contracts';
import { useFaceSelectionStore } from './useFaceSelectionStore';
import FaceOverlayCanvas from './components/canvas/FaceOverlayCanvas';
import FaceSelectionTray from './components/tray/FaceSelectionTray';
import WorkstationCanvas from './components/canvas/WorkstationCanvas';
import MaskBrushOverlay from './components/canvas/MaskBrushOverlay';
import CompositionGridOverlay from './components/canvas/CompositionGridOverlay';
import TimelineWorkstation, { parseFps } from './components/timeline/TimelineWorkstation';
import type { ReactNode } from 'react';
import type { JobRequest } from './contracts';
import { post } from './api';
import PipelineNodeFlowBar from './components/workstation/PipelineNodeFlowBar';
import ParameterSweepModal from './components/workstation/ParameterSweepModal';
import { DEFAULT_PARAMETERS, PIPELINE_NODES, type WorkstationParameters, type PreviewResult } from './components/workstation/pipeline';
import { DockArea, DockPanel, LayoutToolbar } from './components/workstation/WorkspaceLayouts';
import { FloatingPreviewMedia, useFloatingPreview } from './components/workstation/FloatingPreview';
import { useShortcutActions } from './components/controls/Hotkeys';
import NumericInput from './components/controls/NumericInput';
import useHistoryStore from './stores/useHistoryStore';
import { CURVE_PARAMETERS, evaluateParameters, upsertKey } from './stores/parameterCurves';
import HistoryDrawer from './components/workstation/HistoryDrawer';
import SnapshotSlots, { SnapshotViewport } from './components/workstation/SnapshotSlots';
import { panAnchoredAt } from './utils/zoomPan';
import { buildLivePayload, downloadBlob, previewFilename, useLiveSwapPreview } from './livePreview';
import LiveSwapStatus from './components/workstation/LiveSwapStatus';

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
  trimStart = 0,
  trimEnd = null,
  onTrimChange,
  active = true, parameters = DEFAULT_PARAMETERS, setParameters = () => {}, previewRequest = null,
  inspector, posePanel, restorerAvailable = false, onPreviewLatency,
  maskBase64 = null, captureWorkspace = () => ({}), restoreWorkspace = () => {},
  onInspectionUpdate,
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
  trimStart?: number;
  trimEnd?: number | null;
  onTrimChange?: (trimStart: number, trimEnd: number | null) => void;
  active?: boolean;
  parameters?: WorkstationParameters;
  setParameters?: (parameters: WorkstationParameters) => void;
  previewRequest?: JobRequest | null;
  inspector?: ReactNode;
  posePanel?: ReactNode;
  restorerAvailable?: boolean;
  onPreviewLatency?: (latency: Record<string, number>) => void;
  maskBase64?: string | null;
  captureWorkspace?: () => any;
  restoreWorkspace?: (value: any) => void;
  onInspectionUpdate?: (inspection: Inspection) => void;
}) {
  const [detectingFaces, setDetectingFaces] = useState(false);
  const [detectionNotice, setDetectionNotice] = useState('');

  const handleUseFaceFromFrame = useCallback(async () => {
    if (!target || detectingFaces) return;
    setDetectingFaces(true);
    setDetectionNotice('');
    try {
      const res = await post<FrameFaceDetectionResponse>('/media/detect-frame-faces', {
        path: target,
        timestamp: time,
      });
      if (res && res.faces && res.faces.length > 0) {
        useFaceSelectionStore.getState().addFrameDetectedFaces(res.faces, true);
        if (onInspectionUpdate && res.inspection) {
          onInspectionUpdate(res.inspection);
        }
        setDetectionNotice(`Selected ${res.faces.length} target face(s) from current frame.`);
        setTimeout(() => setDetectionNotice(''), 4000);
      } else {
        setDetectionNotice('No faces detected in this frame. Scrub to a frame with a clear face.');
        setTimeout(() => setDetectionNotice(''), 4000);
      }
    } catch (err: any) {
      setDetectionNotice(err.message || 'Face detection failed for current frame.');
      setTimeout(() => setDetectionNotice(''), 4000);
    } finally {
      setDetectingFaces(false);
    }
  }, [target, time, detectingFaces, onInspectionUpdate]);
  const original = useRef<HTMLVideoElement>(null), processed = useRef<HTMLVideoElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);

  const [playing, setPlaying] = useState(false);
  const [playbackError, setPlaybackError] = useState('');

  // Comparison & Masking Modes
  const [compMode, setCompMode] = useState<'wipe' | 'diff' | 'blend' | 'loupe'>('wipe');
  const [isMaskEditing, setIsMaskEditing] = useState(false);
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });
  const history = useHistoryStore(), [historyOpen, setHistoryOpen] = useState(false), [captureBusy, setCaptureBusy] = useState(false), [captureError, setCaptureError] = useState('');
  const captureAbort = useRef<AbortController | null>(null);
  const currentFrame = Math.max(0, Math.round(time * parseFps(probe?.fps_rational, 30)));
  const effectiveParameters = evaluateParameters(parameters, history.curves, currentFrame);
  const editParameter = (field: string, value: number) => { const curve = Object.keys(CURVE_PARAMETERS).find(name => CURVE_PARAMETERS[name].field === field); if (curve && history.curves[curve]?.length) history.editCurves({ ...history.curves, [curve]: upsertKey(history.curves[curve], { frame: currentFrame, value, easing: 'linear' }) }, 'Edit animated parameter', `parameter-${field}`); else setParameters({ ...parameters, [field]: value }); };
  const [selectedNode, setSelectedNode] = useState('final'), [solo, setSolo] = useState<string | null>(null);
  const [sweepOpen, setSweepOpen] = useState(false), [previewBusy, setPreviewBusy] = useState(false), [previewError, setPreviewError] = useState('');
  const [preview, setPreview] = useState<{ key: string; result: PreviewResult } | null>(null);
  const previewKey = JSON.stringify({ request: previewRequest, timestamp: time, parameters });
  const result = preview?.key === previewKey ? preview.result : null;
  useEffect(() => { captureAbort.current?.abort(); setCaptureBusy(false); return () => captureAbort.current?.abort(); }, [previewKey]);
  useEffect(() => { if (history.previewSlot || history.comparison) { setPlaying(false); original.current?.pause(); processed.current?.pause(); } }, [history.previewSlot, history.comparison]);
  const captureSlot = async (slot: string) => {
    if (!previewRequest || captureBusy) return;
    const controller = new AbortController(); captureAbort.current = controller; const workspace = captureWorkspace();
    setCaptureBusy(true); setCaptureError(''); setPlaying(false); original.current?.pause(); processed.current?.pause();
    try {
      const rendered = result ?? await post<PreviewResult>('/workstation/preview', { request: previewRequest, timestamp: time, parameters }, { signal: controller.signal });
      if (controller.signal.aborted) return;
      if (!rendered.buffers.final) throw new Error('No rendered composite is available for this frame');
      useHistoryStore.getState().captureSlot(slot, { image: rendered.buffers.final, frame: rendered.frame_index, timestamp: rendered.timestamp, workspace });
    } catch (error) { if ((error as Error).name !== 'AbortError') setCaptureError((error as Error).message); }
    finally { if (captureAbort.current === controller) setCaptureBusy(false); }
  };
  const previewBody = useRef({ request: previewRequest, timestamp: time, parameters });
  previewBody.current = { request: previewRequest, timestamp: time, parameters };
  const editFingerprint = JSON.stringify({ parameters, curves: history.curves, maskBase64, roles: previewRequest?.face_segments, finishing: previewRequest?.export });
  const previousEdit = useRef(editFingerprint);
  useEffect(() => {
    if (previousEdit.current === editFingerprint) return;
    previousEdit.current = editFingerprint;
    if (active && previewRequest) { history.live(); setSolo(previous => previous ?? 'final'); }
  }, [editFingerprint, active, previewRequest]);
  useEffect(() => { if (active && previewRequest && (previewRequest.export?.finishing.cube || previewRequest.export?.finishing.grain.intensity)) setSolo('final'); }, [active, previewRequest?.export?.finishing]);
  const latencyCallback = useRef(onPreviewLatency); latencyCallback.current = onPreviewLatency;
  useEffect(() => {
    if (!solo || !active || !previewBody.current.request || preview?.key === previewKey) { setPreviewBusy(false); return; }
    const controller = new AbortController(); let live = true;
    setPreviewBusy(true); setPreviewError(''); setPlaying(false); original.current?.pause(); processed.current?.pause();
    const timer = setTimeout(() => {
      void post<PreviewResult>('/workstation/preview', previewBody.current, { signal: controller.signal }).then(response => {
        if (live) { setPreview({ key: previewKey, result: response }); if (response.latency) latencyCallback.current?.(response.latency); }
      }).catch(error => { if (live && error.name !== 'AbortError') setPreviewError(error.message); }).finally(() => { if (live) setPreviewBusy(false); });
    }, 180);
    return () => { live = false; controller.abort(); clearTimeout(timer); };
  }, [previewKey, solo, active, preview?.key]);
  useEffect(() => { if (!active) { setPlaying(false); setSweepOpen(false); original.current?.pause(); processed.current?.pause(); } }, [active]);
  const selectSolo = (node: string) => { if (PIPELINE_NODES.find(item => item.id === node)?.unavailable || !previewRequest) return; history.live(); setSolo(previous => previous === node ? null : node); setIsMaskEditing(false); };
  const bypassNode = (node: string) => { if (node !== 'restorer' && node !== 'color') return; setParameters({ ...parameters, bypass: parameters.bypass.includes(node) ? parameters.bypass.filter(item => item !== node) : [...parameters.bypass, node] }); setSolo('final'); setIsMaskEditing(false); };

  const sourcePreview = usePreview(enabled && active ? target : '', time, playing);
  const outputPreview = usePreview(enabled && active ? output : '', time, playing);
  // Real swapped frame for the current playhead + every edit, via POST /api/preview (debounced, 503-aware).
  const [liveSwap, setLiveSwap] = useState(true);
  const live = useLiveSwapPreview(active && enabled && liveSwap && !playing && !solo && !isMaskEditing
    ? buildLivePayload({ request: previewRequest, time, fps: parseFps(probe?.fps_rational, 30), parameters, maskBase64, restorerAvailable })
    : null);
  const floatingImage = history.previewSlot ? history.slots[history.previewSlot]?.image : solo ? result?.buffers[solo] : live.url || outputPreview.source || sourcePreview.source;
  const floating = useFloatingPreview(<FloatingPreviewMedia image={floatingImage} path={solo ? '' : output || target} time={time} playing={playing && !solo} label={solo ? `SOLO · ${solo}` : output ? 'OUTPUT' : 'SOURCE'}/>);

  const updateViewportSize = useCallback(() => {
    if (viewportRef.current) {
      const rect = viewportRef.current.getBoundingClientRect();
      setViewportSize({ width: rect.width, height: rect.height });
    }
  }, []);

  useEffect(() => {
    updateViewportSize();
    window.addEventListener('resize', updateViewportSize);
    const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(updateViewportSize) : null;
    if (observer && viewportRef.current) observer.observe(viewportRef.current);
    return () => { observer?.disconnect(); window.removeEventListener('resize', updateViewportSize); };
  }, [updateViewportSize]);

  useEffect(() => { setPlaying(false); original.current?.pause(); processed.current?.pause(); setPlaybackError(''); }, [target, output]);
  useEffect(() => { for (const video of [original.current, processed.current]) if (video && Math.abs(video.currentTime - time) > .2) video.currentTime = time; }, [time]);

  const toggle = async () => {
    if (solo) { setSolo(null); return; }
    if (playing) { original.current?.pause(); processed.current?.pause(); setPlaying(false); return; }
    try { await original.current?.play(); await processed.current?.play(); setPlaying(true); }
    catch { setPlaybackError('Browser playback is unavailable for this codec. Use frame previews to inspect it.'); }
  };
  const seek = (value: number) => { original.current?.pause(); processed.current?.pause(); setPlaying(false); setTime(Math.max(0, Math.min(duration, value))); };
  useShortcutActions({ play: target ? () => void toggle() : undefined, previous: () => seek(time - 1 / parseFps(probe?.fps_rational, 30)), next: () => seek(time + 1 / parseFps(probe?.fps_rational, 30)), markIn: () => onTrimChange?.(time, trimEnd !== null && trimEnd <= time ? null : trimEnd), markOut: () => { if (time > trimStart) onTrimChange?.(trimStart, time); }, solo: () => selectSolo(selectedNode), bypass: () => bypassNode(selectedNode), sweep: () => { if (previewRequest) setSweepOpen(true); }, floating: floating.open }, active && !sweepOpen);

  const mediaDims = probe ? { width: probe.width, height: probe.height } : { width: 1920, height: 1080 };

  // Preview Box zoom, pan, grid, peek, and display state
  const [previewZoom, setPreviewZoom] = useState(1);
  const [previewPan, setPreviewPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panOrigin, setPanOrigin] = useState({ x: 0, y: 0 });
  const [gridMode, setGridMode] = useState<'off' | 'thirds' | 'crosshair' | 'safe'>('off');
  const [showSourcePeek, setShowSourcePeek] = useState(false);
  const [showFaceOverlays, setShowFaceOverlays] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [snapshotNotice, setSnapshotNotice] = useState('');
  const [isSpaceDown, setIsSpaceDown] = useState(false);
  const isSpaceDownRef = useRef(false);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat && !(e.target as HTMLElement)?.closest('input, textarea, select')) {
        isSpaceDownRef.current = true;
        setIsSpaceDown(true);
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        isSpaceDownRef.current = false;
        setIsSpaceDown(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  // Fullscreen sync
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      viewportRef.current?.requestFullscreen?.().catch(() => {});
    } else {
      document.exitFullscreen?.().catch(() => {});
    }
  }, []);

  useEffect(() => {
    const handleFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  // Swapped-preview download: a click handler builds the blob URL itself, so the saved name and type never
  // depend on an `<a download>` attribute surviving a data/blob URL round trip.
  const handleDownloadLive = useCallback(() => {
    if (!live.blob) {
      setSnapshotNotice('No swapped preview is available to download yet.');
      setTimeout(() => setSnapshotNotice(''), 3000);
      return;
    }
    downloadBlob(live.blob, previewFilename());
    setSnapshotNotice('Swapped preview downloaded');
    setTimeout(() => setSnapshotNotice(''), 3000);
  }, [live.blob]);

  // Frame snapshot download
  const handleDownloadSnapshot = useCallback(() => {
    try {
      if (!solo && !showSourcePeek && live.blob && !live.stale) { handleDownloadLive(); return; }
      let dataUrl = '';
      if (solo && result?.buffers[solo]) {
        dataUrl = result.buffers[solo];
      } else if (showSourcePeek && sourcePreview.source) {
        dataUrl = sourcePreview.source;
      } else if (output && outputPreview.source) {
        dataUrl = outputPreview.source;
      } else if (sourcePreview.source) {
        dataUrl = sourcePreview.source;
      } else {
        const v = showSourcePeek ? original.current : (processed.current || original.current);
        if (v && v.videoWidth > 0 && v.videoHeight > 0) {
          const c = document.createElement('canvas');
          c.width = v.videoWidth;
          c.height = v.videoHeight;
          const ctx = c.getContext('2d');
          if (ctx) {
            ctx.drawImage(v, 0, 0);
            dataUrl = c.toDataURL('image/png');
          }
        }
      }

      if (!dataUrl) {
        setSnapshotNotice('No frame buffer available to capture.');
        setTimeout(() => setSnapshotNotice(''), 3000);
        return;
      }

      const a = document.createElement('a');
      a.href = dataUrl;
      const timeMs = Math.round(time * 1000);
      a.download = `roop3d_snapshot_${timeMs}ms.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      setSnapshotNotice('Snapshot downloaded! 📸');
      setTimeout(() => setSnapshotNotice(''), 3000);
    } catch (err: any) {
      setSnapshotNotice(err.message || 'Snapshot download failed');
      setTimeout(() => setSnapshotNotice(''), 3000);
    }
  }, [solo, result, showSourcePeek, sourcePreview.source, outputPreview.source, output, time, live.blob, live.stale, handleDownloadLive]);

  // Viewport pan & zoom handlers
  const handleViewportPointerDown = (e: React.PointerEvent) => {
    if ((previewZoom <= 1 && !isSpaceDownRef.current) || isMaskEditing) return;
    const targetEl = e.target as HTMLElement;
    if (targetEl.closest('button, input, select, .face-assignment-popover, .preview-hud-toolbar, .preview-face-selector-strip')) return;
    setIsPanning(true);
    setPanOrigin({ x: e.clientX - previewPan.x, y: e.clientY - previewPan.y });
    try {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } catch {}
  };

  const handleViewportPointerMove = (e: React.PointerEvent) => {
    if (!isPanning || (previewZoom <= 1 && !isSpaceDownRef.current)) return;
    setPreviewPan({
      x: e.clientX - panOrigin.x,
      y: e.clientY - panOrigin.y,
    });
  };

  const handleViewportPointerUp = (e: React.PointerEvent) => {
    if (isPanning) {
      setIsPanning(false);
      try {
        (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
      } catch {}
    }
  };

  const handleWheelZoom = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      e.stopPropagation();
      const delta = e.deltaY < 0 ? 0.25 : -0.25;
      const nextZoom = Math.max(1, Math.min(4, Math.round((previewZoom + delta) * 4) / 4));
      if (nextZoom === 1) {
        setPreviewZoom(1);
        setPreviewPan({ x: 0, y: 0 });
      } else {
        const stage = viewportRef.current;
        const client = { x: e.clientX, y: e.clientY };
        const nextPan = panAnchoredAt(client, stage, previewZoom, nextZoom, previewPan);
        setPreviewZoom(nextZoom);
        setPreviewPan(nextPan);
      }
    }
  };

  const handleViewportDoubleClick = (e: React.MouseEvent) => {
    const targetEl = e.target as HTMLElement;
    if (targetEl.closest('button, input, select, .face-assignment-popover, .preview-hud-toolbar, .preview-face-selector-strip')) return;
    if (previewZoom > 1) {
      setPreviewZoom(1);
      setPreviewPan({ x: 0, y: 0 });
    } else {
      setPreviewZoom(2);
    }
  };

  const pane = (path: string, result: typeof sourcePreview, isOutput: boolean) => {
    const isShowingSourcePeek = isOutput && showSourcePeek;
    const activePath = isShowingSourcePeek ? target : path;
    const activeResult = isShowingSourcePeek ? sourcePreview : result;
    const showLive = isOutput && !isShowingSourcePeek && !playing && !!live.url;

    return (
      <div
        className={`video-pane ${isOutput ? 'processed' : 'original'} ${isShowingSourcePeek ? 'source-peek-active' : ''}`}
        style={isOutput && output ? undefined : undefined}
      >
        {activePath ? (
          <>
            <video
              ref={isOutput ? (isShowingSourcePeek ? undefined : processed) : original}
              src={mediaURL(activePath)}
              preload="metadata"
              playsInline
              muted
              onEnded={() => setPlaying(false)}
              onError={() => setPlaybackError('Browser playback unavailable; frame previews can still be used.')}
              onTimeUpdate={!isOutput ? event => { if (playing) setTime(event.currentTarget.currentTime); } : undefined}
            />
            {!playing && !showLive && activeResult.source && (
              <img
                className="frame-preview"
                src={activeResult.source}
                alt={`${isOutput ? (isShowingSourcePeek ? 'Source peek' : 'Processed') : 'Source'} frame at ${time.toFixed(2)} seconds`}
              />
            )}
            {!isOutput && !isMaskEditing && showFaceOverlays && (
              <FaceOverlayCanvas
                videoRef={original}
                mediaDimensions={mediaDims}
                currentTime={time}
                sources={sources}
                onAssignSource={onAssignSource}
                showOverlays={showFaceOverlays}
              />
            )}
            {isShowingSourcePeek && (
              <div className="source-peek-badge">
                <span>👁</span> SOURCE PEEK (A/B)
              </div>
            )}
          </>
        ) : !showLive && (
          <div className="viewport-empty">
            <span className="film-icon">{isOutput ? '◇' : '⊞'}</span>
            <h2>{isOutput ? 'Your output starts here' : 'Bring a scene to life'}</h2>
            <p>{isOutput ? 'Completed renders appear alongside your source.' : 'Import a local video, inspect its faces, and assign an identity.'}</p>
          </div>
        )}
        {showLive && (
          <img
            className={`frame-preview live-swap-preview ${live.stale ? 'stale' : ''}`}
            src={live.url!}
            alt={`Live swapped preview at ${time.toFixed(2)} seconds`}
          />
        )}
        {isOutput && !isShowingSourcePeek && <LiveSwapStatus live={live}/>}
        <span className="pane-label">
          {isOutput ? (isShowingSourcePeek ? 'SOURCE (PEEK)' : showLive ? 'LIVE SWAP' : 'OUTPUT') : 'SOURCE'} <i className={`dot ${activePath || showLive ? 'mint' : 'dim'}`} />
        </span>
        {activeResult.error && <span className="preview-error">{activeResult.error}</span>}
      </div>
    );
  };

  return (
    <section className={`viewport-section ${isFullscreen ? 'is-fullscreen' : ''}`}>
      <LayoutToolbar onFloat={floating.open}/>
      <SnapshotSlots onCapture={(slot: string) => void captureSlot(slot)} onRestore={restoreWorkspace} busy={captureBusy} captureAvailable={!!previewRequest}/>
      {captureError && <p role="alert" className="inline-warning">{captureError}</p>}
      <PipelineNodeFlowBar selected={selectedNode} solo={solo} bypass={parameters.bypass} onSelect={setSelectedNode} onSolo={selectSolo} onBypass={bypassNode} busy={previewBusy} available={!!previewRequest}/>
      <div className="viewport-toolbar">
        <span className="eyebrow">WORKSTATION MEDIA INSPECTOR</span>
        <button className="button small" disabled={!history.past.length} onClick={history.undo}>Undo</button><button className="button small" disabled={!history.future.length} onClick={history.redo}>Redo</button><button className="button small" onClick={() => setHistoryOpen(true)}>Edit history ({history.past.length})</button>
        <button className="button small" disabled={!previewRequest} onClick={() => { setPlaying(false); original.current?.pause(); processed.current?.pause(); setSweepOpen(true); }}>Parameter Sweep</button>
        <button
          type="button"
          className={`button small use-face-frame-btn ${detectingFaces ? 'busy' : ''}`}
          disabled={!target || detectingFaces}
          onClick={handleUseFaceFromFrame}
          title="Detect and select target faces directly from the currently displayed preview frame (like Roop Ultimate)"
        >
          <span>🎯</span> {detectingFaces ? 'Detecting Faces…' : 'Use Face from this Frame'}
        </button>
        {detectionNotice && <span className="detection-inline-notice">{detectionNotice}</span>}
        {solo && <button className="button small" onClick={() => setSolo(null)}>Exit solo · {solo}</button>}

        {/* Mask Editing Mode Toggle */}
        <button
          type="button"
          className={`mask-toggle-btn ${isMaskEditing ? 'active' : ''}`}
          disabled={!target}
          onClick={() => { history.live(); setSolo(null); setIsMaskEditing(e => !e); }}
          title="Manual & Occlusion Mask Editor (Paint keep-target areas & auto-detect occluders)"
        >
          <span>✎</span> {isMaskEditing ? 'Close Mask Editor' : 'Edit Mask'}
        </button>

        <span className="muted">Source timing preserved</span>
      </div>

      <div
        className={`viewport dual ${isMaskEditing ? 'mask-editing' : ''} ${isFullscreen ? 'fullscreen-active' : ''}`}
        ref={viewportRef}
        onPointerDown={handleViewportPointerDown}
        onPointerMove={handleViewportPointerMove}
        onPointerUp={handleViewportPointerUp}
        onPointerCancel={handleViewportPointerUp}
        onDoubleClick={handleViewportDoubleClick}
        onWheel={handleWheelZoom}
      >
        {/* Floating Preview Controls HUD */}
        <div className="preview-hud-toolbar hud-glass" role="toolbar" aria-label="Preview Controls">
          {/* Zoom Controller */}
          <div className="hud-group zoom-group spring-cluster" role="group" aria-label="Preview Zoom">
            <button
              type="button"
              className="hud-btn hud-glass-button zoom-step-btn"
              disabled={previewZoom <= 1}
              onClick={() => {
                const next = Math.max(1, Math.round((previewZoom - 0.25) * 4) / 4);
                setPreviewZoom(next);
                if (next === 1) setPreviewPan({ x: 0, y: 0 });
              }}
              title="Zoom Out Preview [-]"
            >
              -
            </button>
            <button
              type="button"
              className="hud-btn hud-glass-button zoom-level-btn"
              onClick={() => {
                if (previewZoom === 1) {
                  setPreviewZoom(2);
                } else {
                  setPreviewZoom(1);
                  setPreviewPan({ x: 0, y: 0 });
                }
              }}
              title="Click to toggle Fit / 200% Zoom"
            >
              {previewZoom === 1 ? 'Fit' : `${Math.round(previewZoom * 100)}%`}
            </button>
            <button
              type="button"
              className="hud-btn hud-glass-button zoom-step-btn"
              disabled={previewZoom >= 4}
              onClick={() => setPreviewZoom(z => Math.min(4, Math.round((z + 0.25) * 4) / 4))}
              title="Zoom In Preview [+]"
            >
              +
            </button>
            <button
              type="button"
              className={`hud-btn hud-glass-button zoom-preset-btn ${previewZoom === 1 ? 'active' : ''}`}
              onClick={() => {
                setPreviewZoom(1);
                setPreviewPan({ x: 0, y: 0 });
              }}
              title="1:1 Native Resolution (Shift+1)"
            >
              1:1
            </button>
            <button
              type="button"
              className={`hud-btn hud-glass-button zoom-preset-btn ${previewZoom === 2 ? 'active' : ''}`}
              onClick={() => {
                setPreviewZoom(2);
              }}
              title="2:1 200% Zoom (Shift+2)"
            >
              2:1
            </button>
            <button
              type="button"
              className={`hud-btn hud-glass-button zoom-preset-btn ${previewZoom === 4 ? 'active' : ''}`}
              onClick={() => {
                setPreviewZoom(4);
              }}
              title="4:1 400% Zoom (Shift+4)"
            >
              4:1
            </button>
            {previewZoom > 1 && (
              <button
                type="button"
                className="hud-btn hud-glass-button hud-reset-btn"
                onClick={() => {
                  setPreviewZoom(1);
                  setPreviewPan({ x: 0, y: 0 });
                }}
                title="Reset Zoom & Pan to Fit (Double-click viewport also resets)"
              >
                Fit
              </button>
            )}
          </div>

          {/* Composition Framing Guides Cycle */}
          <button
            type="button"
            className={`hud-btn ${gridMode !== 'off' ? 'active' : ''}`}
            onClick={() => {
              const modes: Array<'off' | 'thirds' | 'crosshair' | 'safe'> = ['off', 'thirds', 'crosshair', 'safe'];
              const nextIdx = (modes.indexOf(gridMode) + 1) % modes.length;
              setGridMode(modes[nextIdx]);
            }}
            title={`Framing Guides: ${gridMode.toUpperCase()} (Cycle: Off → Thirds → Crosshair → Safe)`}
          >
            <span>#</span> {gridMode === 'off' ? 'Grid' : gridMode}
          </button>

          {/* Target Face Overlays Toggle */}
          <button
            type="button"
            className={`hud-btn ${showFaceOverlays ? 'active' : ''}`}
            onClick={() => setShowFaceOverlays(o => !o)}
            title={`Target Face Overlays: ${showFaceOverlays ? 'ON' : 'OFF'}`}
          >
            <span>🎯</span> Overlays
          </button>

          {/* Hold-to-Compare Source (A/B Quick Peek) */}
          {output && !solo && (
            <button
              type="button"
              className={`hud-btn hud-peek-btn ${showSourcePeek ? 'peeking' : ''}`}
              onMouseDown={() => setShowSourcePeek(true)}
              onMouseUp={() => setShowSourcePeek(false)}
              onMouseLeave={() => setShowSourcePeek(false)}
              onTouchStart={() => setShowSourcePeek(true)}
              onTouchEnd={() => setShowSourcePeek(false)}
              title="Click & HOLD to temporarily view original source frame (A/B peek)"
            >
              <span>👁</span> Hold: Source
            </button>
          )}

          {/* Live swapped preview: on/off + PNG download (click handler, blob URL) */}
          <button
            type="button"
            className={`hud-btn live-swap-toggle ${liveSwap ? 'active' : ''}`}
            aria-pressed={liveSwap}
            onClick={() => setLiveSwap(value => !value)}
            title="Render the real face swap for the current frame whenever the playhead, faces or parameters change"
          >
            <span>⚡</span> Live swap
          </button>
          <button
            type="button"
            className="hud-btn live-download-btn"
            disabled={!live.blob}
            onClick={handleDownloadLive}
            title="Download the swapped preview frame as a PNG"
          >
            <span>⬇</span> PNG
          </button>

          {/* Instant Frame Snapshot Grab */}
          <button
            type="button"
            className="hud-btn snapshot-btn"
            onClick={handleDownloadSnapshot}
            title="Download current frame preview as high-res PNG (📸)"
          >
            <span>📸</span> Snapshot
          </button>

          {/* Fullscreen Toggle */}
          <button
            type="button"
            className={`hud-btn fullscreen-toggle-btn ${isFullscreen ? 'active' : ''}`}
            onClick={toggleFullscreen}
            title={isFullscreen ? 'Exit Fullscreen [Esc]' : 'Enter Fullscreen Preview [F]'}
          >
            <span>{isFullscreen ? '⤓' : '⛶'}</span>
          </button>
        </div>

        {/* Framing & Composition Grid Overlays */}
        <CompositionGridOverlay mode={gridMode} />

        {/* Snapshot Download Notice */}
        {snapshotNotice && <div className="snapshot-notice-toast">{snapshotNotice}</div>}

        <SnapshotViewport/>
        {solo && <div className="solo-viewport" role="region" aria-label={`${solo} intermediate buffer`}>
          {result?.buffers[solo] ? <img src={result.buffers[solo]} alt={`${solo} execution buffer at frame ${result.frame_index}`}/> : <p role="status">{previewError || (previewBusy ? 'Rendering real pipeline buffers…' : 'No buffer is available at this frame')}</p>}
          <span className="pane-label">SOLO · {solo} · {result ? `frame ${result.frame_index}` : formatTime(time)}</span>
        </div>}

        <div
          className="viewport-stage"
          style={{
            transform: `translate(${previewPan.x}px, ${previewPan.y}px) scale(${previewZoom})`,
            transformOrigin: 'center center',
            width: '100%',
            height: '100%',
            display: 'grid',
            gridTemplateColumns: isMaskEditing ? '1fr' : '1fr 1fr',
            cursor: isPanning ? 'grabbing' : isSpaceDown ? 'grab' : previewZoom > 1 ? 'grab' : 'default',
            transition: isPanning ? 'none' : 'transform 0.12s cubic-bezier(0.2, 0, 0, 1)',
          }}
        >
          {pane(target, sourcePreview, false)}
          {!isMaskEditing && pane(output, outputPreview, true)}

          {/* Workstation Comparison Canvas (Wipe, Diff Map, Alpha Blend, Magnifier Loupe) */}
          {!isMaskEditing && !solo && (
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
              initialMask={maskBase64}
              onClose={() => setIsMaskEditing(false)}
            />
          )}
        </div>
      </div>

      {playbackError && <p className="inline-warning">{playbackError}</p>}
      {previewError && <p className="inline-warning" role="alert">{previewError}<button className="button small" onClick={() => { setPreview(null); setSolo(null); setTimeout(() => setSolo(selectedNode), 0); }}>Retry preview</button></p>}
      {solo && result?.notes.map(note => <p className="muted buffer-note" key={note}>{note}</p>)}
      <div className="preview-parameters">
        <label className="numeric-control">Pose correction method<select aria-label="Pose correction method" value={parameters.pose_mode ?? 'retarget'} onChange={event => setParameters({ ...parameters, pose_mode: event.target.value as 'retarget' | 'mesh_blend' })}><option value="mesh_blend">2.5D / 3D mesh blend (canonical proxy)</option><option value="retarget">Reference LivePortrait retarget</option></select></label>
        <NumericInput label="Swap weight" value={effectiveParameters.swap_weight ?? 1} min={0} max={1} step={.01} onChange={value => editParameter('swap_weight', value ?? 1)}/>
        <NumericInput label="Mask erosion (px)" value={effectiveParameters.erosion_px ?? 0} min={-32} max={32} step={1} onChange={value => editParameter('erosion_px', value ?? 0)}/>
        <NumericInput label="Restorer blend" value={effectiveParameters.restoration_blend} min={0} max={1} step={.01} disabled={!restorerAvailable} title={restorerAvailable ? 'Local GPEN' : 'GPEN weights unavailable'} onChange={value => editParameter('restoration_blend', value ?? 0)}/>
        <NumericInput label="Mask feather (px)" value={effectiveParameters.feather_px} min={0} max={64} step={.1} onChange={value => editParameter('feather_px', value ?? 0)}/>
        <NumericInput label="Color match strength" value={effectiveParameters.color_strength} min={0} max={1} step={.01} onChange={value => editParameter('color_strength', value ?? 0)}/>
        <NumericInput label={parameters.pose_mode === 'mesh_blend' ? '2.5D → 3D pose blend' : 'Reference retarget strength'} value={effectiveParameters.pose_strength} min={0} max={1} step={.01} onChange={value => editParameter('pose_strength', value ?? 0)}/>
      </div>
      {floating.portal}
      {historyOpen && <HistoryDrawer onClose={() => setHistoryOpen(false)}/>}
      {sweepOpen && <ParameterSweepModal request={previewRequest} timestamp={time} parameters={parameters} onApply={setParameters} onClose={() => setSweepOpen(false)} restorerAvailable={restorerAvailable}/>}
      <DockArea>

      {/* Video Editor-Grade Workstation Timeline */}
      <DockPanel id="timeline" title="Timeline"><TimelineWorkstation
        parameters={parameters}
        target={target}
        output={output}
        currentTime={time}
        duration={duration}
        fps={parseFps(probe?.fps_rational, 30)}
        playing={playing}
        disabled={!target || !enabled || !active}
        keys={keys}
        trimStart={trimStart}
        trimEnd={trimEnd}
        targetVideoRef={original}
        processedVideoRef={processed}
        onTogglePlay={() => void toggle()}
        onSeek={(seekTime) => {
          setPlaying(false);
          original.current?.pause();
          processed.current?.pause();
          setTime(seekTime);
        }}
        onTrimChange={onTrimChange}
      /></DockPanel>

      {/* Target Face Selection Tray */}
      <DockPanel id="tray" title="Face Tray"><FaceSelectionTray
        currentTime={time}
        sources={sources}
        onSeekTime={setTime}
        onAssignSource={onAssignSource}
      /></DockPanel>
      <DockPanel id="inspector" title="Inspector">{inspector ?? <p className="muted">Processing inspector</p>}</DockPanel>
      <DockPanel id="pose" title="3D Pose & Geometry">{posePanel ?? <p className="muted">Select a source identity to inspect geometry.</p>}</DockPanel>
      </DockArea>
    </section>
  );
}

export function formatTime(value: number) {
  return `${Math.floor(value / 60).toString().padStart(2, '0')}:${(value % 60).toFixed(2).padStart(5, '0')}`;
}
