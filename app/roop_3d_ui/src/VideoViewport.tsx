import { useEffect, useRef, useState, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { renderProbe } from './utils/renderProbe';
import { mediaURL, postLivePreview } from './api';
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
import ContactSheetModal, { type Patch } from './components/workstation/ContactSheetModal';
import SettingControl from './components/controls/SettingControl';
import { PANELS, settingValue, settingsOf, type PanelId } from './components/controls/settingsRegistry';
import FaceContextMenu, { type FaceMenuTarget } from './components/canvas/FaceContextMenu';
import { useFaceExtraction } from './components/canvas/useFaceExtraction';
import { facePose } from './components/canvas/poseHud';
import { calculateLetterbox, mapBoxToPixels } from './utils/letterbox';
import { leftToRightRank } from './utils/poseCells';
import { DEFAULT_PARAMETERS, PIPELINE_NODES, nodeBuffer, type WorkstationParameters } from './components/workstation/pipeline';
import { DockArea, DockPanel, LayoutToolbar } from './components/workstation/WorkspaceLayouts';
import { FloatingPreviewMedia, pipStore, useFloatingPreview } from './components/workstation/FloatingPreview';
import { LiveTime } from './components/workstation/LiveTime';
import { useStoreSelector } from './stores/atomicStore';
import { getPlayhead, setPlayhead } from './stores/playheadStore';
import { useShortcutActions } from './components/controls/Hotkeys';
import NumericInput from './components/controls/NumericInput';
import useHistoryStore from './stores/useHistoryStore';
import { CURVE_PARAMETERS, evaluateParameters, upsertKey } from './stores/parameterCurves';
import HistoryDrawer from './components/workstation/HistoryDrawer';
import SnapshotSlots, { SnapshotViewport } from './components/workstation/SnapshotSlots';
import { panAnchoredAt } from './utils/zoomPan';
import { buildLivePayload, downloadBlob, previewFilename, useLiveSwapPreview } from './livePreview';
import LiveSwapStatus from './components/workstation/LiveSwapStatus';
import { DEFAULT_VIEWPORT, type ViewportPreferences } from './previewPreferences';
import type { Dispatch, SetStateAction } from 'react';

/** Snapshot slots keep a self-contained image, so convert the preview PNG before its object URL is revoked. */
async function blobToDataUrl(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  for (let at = 0; at < bytes.length; at += 0x8000) binary += String.fromCharCode(...bytes.subarray(at, at + 0x8000));
  return `data:${blob.type.startsWith('image/') ? blob.type : 'image/png'};base64,${btoa(binary)}`;
}

function usePreview(path: string, time: number, duration: number, fps: number, playing: boolean) {
  const [source, setSource] = useState('');
  const [error, setError] = useState('');
  useEffect(() => {
    setSource(''); setError('');
    if (!path || playing) return;
    const controller = new AbortController();
    const lease = new PreviewURL();
    let active = true;
    const timer = setTimeout(() => {
      // Seeking to duration asks the decoder for a frame after the final frame's start time.
      // Back up one nominal frame so the final timeline position remains previewable.
      const previewTime = duration > 0 ? Math.min(time, Math.max(0, duration - 1 / Math.max(fps, 1))) : time;
      void fetch(`/api/v1/media/preview-frame?${new URLSearchParams({ path, timestamp: String(previewTime) })}`, { signal: controller.signal, cache: 'no-store' })
        .then(async response => {
          if (!response.ok) {
            let message = `Frame preview failed (${response.status})`;
            try {
              const body = await response.json();
              if (typeof body?.error?.message === 'string') message = body.error.message;
            } catch { /* Keep the status-based message when the response is not JSON. */ }
            throw new Error(message);
          }
          const blob = await response.blob();
          if (active) setSource(lease.replace(blob));
        })
        .catch(error => { if (active && error.name !== 'AbortError') setError(error.message); });
    }, 100);
    return () => { active = false; clearTimeout(timer); controller.abort(); lease.clear(); };
  }, [path, time, duration, fps, playing]);
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
  onInspectionUpdate, onSourceExtracted, activeSourceId = '', studio = null,
  previewView, onPreviewViewChange,
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
  /** Called with the FaceSet created by "Use as source" so the workspace can make it the active source face. */
  onSourceExtracted?: (faceset: FaceSet) => void;
  /** Source FaceSet currently selected in the workspace: "Add to .fsz Faceset" extends it. */
  activeSourceId?: string;
  /**
   * Stage 7 studio layout: the parameter panels, timeline and face tray render (through portals, so all their state
   * stays here) into the workstation's inspector / bottom dock instead of below the viewport. Null keeps the classic
   * stacked layout unchanged.
   */
  studio?: { panels: HTMLElement | null; timeline: HTMLElement | null; tray: HTMLElement | null } | null;
  previewView?: ViewportPreferences;
  onPreviewViewChange?: Dispatch<SetStateAction<ViewportPreferences>>;
}) {
  const [localView, setLocalView] = useState(DEFAULT_VIEWPORT);
  const view = previewView ?? localView, setView = onPreviewViewChange ?? setLocalView;
  const preference = <K extends keyof ViewportPreferences,>(key: K) => [view[key], (next: SetStateAction<ViewportPreferences[K]>) =>
    setView(previous => ({ ...previous, [key]: typeof next === 'function' ? (next as (value: ViewportPreferences[K]) => ViewportPreferences[K])(previous[key]) : next }))] as const;
  const [detectingFaces, setDetectingFaces] = useState(false);
  renderProbe('VideoViewport');
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
  const [sourceFaceNumber, setSourceFaceNumber] = useState(1);
  const original = useRef<HTMLVideoElement>(null), processed = useRef<HTMLVideoElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);

  const [playing, setPlaying] = useState(false);
  const [playbackError, setPlaybackError] = useState('');
  // `time` is the committed playhead. While playing, ticks go to playheadStore only (see stores/playheadStore.ts);
  // `now()` is the up-to-the-frame value for anything that acts on "the current frame" (marks, steps, captures).
  const now = () => (playing ? getPlayhead() : time);
  const wasPlaying = useRef(false), commitTime = useRef(setTime); commitTime.current = setTime;
  useEffect(() => {
    if (wasPlaying.current && !playing) { const live = getPlayhead(); if (live !== time) commitTime.current(live); }   // playback stopped: commit the final tick
    wasPlaying.current = playing;
  }, [playing, time]);
  // A pinned floating preview keeps rendering the swapped face even while another screen (#/facemgr, #/batch …) is shown.
  const previewPinned = useStoreSelector(pipStore, state => state.mode !== 'closed');
  const previewLive = active || previewPinned;

  // Comparison & Masking Modes
  const [compMode, setCompMode] = preference('compare_mode');
  const [isMaskEditing, setIsMaskEditing] = useState(false);
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });
  const history = useHistoryStore(), [historyOpen, setHistoryOpen] = useState(false), [captureBusy, setCaptureBusy] = useState(false), [captureError, setCaptureError] = useState('');
  const captureAbort = useRef<AbortController | null>(null);
  const currentFrame = Math.max(0, Math.round(time * parseFps(probe?.fps_rational, 30)));
  const effectiveParameters = evaluateParameters(parameters, history.curves, currentFrame);
  const editParameter = (field: string, value: number) => { const curve = Object.keys(CURVE_PARAMETERS).find(name => CURVE_PARAMETERS[name].field === field); if (curve && history.curves[curve]?.length) history.editCurves({ ...history.curves, [curve]: upsertKey(history.curves[curve], { frame: currentFrame, value, easing: 'linear' }) }, 'Edit animated parameter', `parameter-${field}`); else setParameters({ ...parameters, [field]: value }); };
  const [selectedNode, setSelectedNode] = useState('final'), [solo, setSolo] = useState<string | null>(null);
  const [sweepPanel, setSweepPanel] = useState<PanelId | null>(null);
  const fps = parseFps(probe?.fps_rational, 30);
  const previewKey = JSON.stringify({ request: previewRequest, timestamp: time, parameters });
  useEffect(() => { captureAbort.current?.abort(); setCaptureBusy(false); return () => captureAbort.current?.abort(); }, [previewKey]);
  useEffect(() => { if (history.previewSlot || history.comparison) { setPlaying(false); original.current?.pause(); processed.current?.pause(); } }, [history.previewSlot, history.comparison]);
  const captureSlot = async (slot: string) => {
    const at = now();
    const payload = buildLivePayload({ timestampSelection: true, request: previewRequest, time: at, fps, parameters, maskBase64, restorerAvailable, solo: 'final' });
    if (!payload || captureBusy) return;
    const controller = new AbortController(); captureAbort.current = controller; const workspace = captureWorkspace();
    setCaptureBusy(true); setCaptureError(''); setPlaying(false); original.current?.pause(); processed.current?.pause();
    try {
      const rendered = await postLivePreview(payload, { signal: controller.signal });
      if (controller.signal.aborted || rendered.cancelled) return;
      useHistoryStore.getState().captureSlot(slot, { image: await blobToDataUrl(rendered.blob), frame: rendered.meta?.frame_index ?? currentFrame, timestamp: at, workspace });
    } catch (error) { if ((error as Error).name !== 'AbortError') setCaptureError((error as Error).message); }
    finally { if (captureAbort.current === controller) setCaptureBusy(false); }
  };
  const editFingerprint = JSON.stringify({ parameters, curves: history.curves, maskBase64, roles: previewRequest?.face_segments, finishing: previewRequest?.export });
  const previousEdit = useRef(editFingerprint);
  useEffect(() => {
    if (previousEdit.current === editFingerprint) return;
    previousEdit.current = editFingerprint;
    if (active && previewRequest) history.live();   // an edit leaves any snapshot comparison; the live preview re-renders on its own
  }, [editFingerprint, active, previewRequest]);
  const latencyCallback = useRef(onPreviewLatency); latencyCallback.current = onPreviewLatency;
  useEffect(() => { if (!active) { setPlaying(false); setSweepPanel(null); original.current?.pause(); processed.current?.pause(); } }, [active]);
  const selectSolo = (node: string) => { if (PIPELINE_NODES.find(item => item.id === node)?.unavailable || !previewRequest) return; history.live(); setSolo(previous => previous === node ? null : node); setIsMaskEditing(false); };
  const bypassNode = (node: string) => { if (node !== 'restorer') return; setParameters({ ...parameters, bypass: parameters.bypass.includes(node) ? parameters.bypass.filter(item => item !== node) : [...parameters.bypass, node] }); setSolo('final'); setIsMaskEditing(false); };

  const sourcePreview = usePreview(enabled && previewLive ? target : '', time, duration, fps, playing);
  const outputPreview = usePreview(enabled && previewLive ? output : '', time, duration, fps, playing);
  // The real swapped frame (or, when a node is soloed, that stage's buffer) from POST /api/preview: debounced, 503-aware.
  const [liveSwap, setLiveSwap] = preference('live_swap');
  const live = useLiveSwapPreview(previewLive && enabled && liveSwap && !playing && !isMaskEditing
    ? buildLivePayload({ timestampSelection: true, request: previewRequest, time, fps, parameters, maskBase64, restorerAvailable, solo: nodeBuffer(solo) })
    : null);
  useEffect(() => { if (live.meta?.latency) latencyCallback.current?.(live.meta.latency); }, [live.meta]);
  // Only show a live image that is the buffer currently asked for: never the previous node's output while a new one loads.
  const liveFresh = !!live.url && (live.meta?.solo_node ?? 'final') === (nodeBuffer(solo) ?? 'final');
  const liveUrl = liveFresh ? live.url : null, liveBlob = liveFresh ? live.blob : null;
  const [viewMode, setViewMode] = preference('view_mode');
  const [showLandmarks, setShowLandmarks] = preference('show_landmarks'), [showPose, setShowPose] = preference('show_pose');
  const [faceMenu, setFaceMenu] = useState<FaceMenuTarget | null>(null);
  const extraction = useFaceExtraction({ target, frameIndex: currentFrame, activeSourceId: activeSourceId || null, onSourceExtracted, onRefreshPreview: live.retry });
  const activeSourceName = sources.find(item => item.faceset_id === activeSourceId)?.name ?? null;
  const openFaceMenu = useCallback((menu: { faceRank: number; pose: FaceMenuTarget['pose']; x: number; y: number; face?: any; norm_cx?: number; norm_cy?: number }) => {
    const bbox = menu.face?.bbox;
    const norm_cx = menu.norm_cx ?? (bbox ? (bbox[0] + bbox[2]) / 2 : undefined);
    const norm_cy = menu.norm_cy ?? (bbox ? (bbox[1] + bbox[3]) / 2 : undefined);
    setFaceMenu({ x: menu.x, y: menu.y, faceRank: menu.faceRank, pose: menu.pose, norm_cx, norm_cy });
  }, []);
  const openMenuFromBar = useCallback((anchor: { x: number; y: number }) => {
    const store = useFaceSelectionStore.getState(), faces = store.getFacesAtTimestamp(time, fps);
    const face = faces.find(item => item.id === store.activeSelection) ?? faces[0];
    const bbox = face?.bbox;
    const norm_cx = bbox ? (bbox[0] + bbox[2]) / 2 : undefined;
    const norm_cy = bbox ? (bbox[1] + bbox[3]) / 2 : undefined;
    setFaceMenu({ ...anchor, faceRank: face ? leftToRightRank(faces, face) : 0, pose: face ? facePose(face) : null, norm_cx, norm_cy });
  }, [time, fps]);
  const floatingImage = history.previewSlot ? history.slots[history.previewSlot]?.image : liveUrl || outputPreview.source || sourcePreview.source;
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
  const frameStep = 1 / parseFps(probe?.fps_rational, 30);
  const flash = (message: string) => { setSnapshotNotice(message); setTimeout(() => setSnapshotNotice(''), 3000); };
  /** W / D: enter the full-bleed comparison canvas in `mode`, or leave it when that mode is already showing. */
  const toggleCanvasMode = (mode: typeof compMode) => {
    if (playing) { flash('Pause playback to use the comparison canvas.'); return; }
    if (viewMode === 'compare' && compMode === mode) { setViewMode('side'); return; }
    if (mode !== 'wipe' && !(liveUrl || outputPreview.source)) { flash('No swapped frame yet: the diff map compares original against swapped.'); return; }
    setCompMode(mode); setViewMode('compare'); setSolo(null); setIsMaskEditing(false);
  };
  /** E: extract the selected (else left-most) face of the paused frame as the active source, without opening the menu. */
  const quickExtractFace = () => {
    if (!target) return;
    if (playing) { flash('Pause on a frame to extract its face.'); return; }
    const store = useFaceSelectionStore.getState(), faces = store.getFacesAtTimestamp(time, fps), face = faces.find(item => item.id === store.activeSelection);
    void extraction.run('source_gallery', { faceRank: face ? leftToRightRank(faces, face) : sourceFaceNumber - 1 });
  };
  const bufferShortcuts = Object.fromEntries(PIPELINE_NODES.map((node, index) => [`buffer${index + 1}`, () => selectSolo(node.id)]));
  useShortcutActions({
    play: target ? () => void toggle() : undefined,
    previous: () => seek(now() - frameStep), next: () => seek(now() + frameStep),
    markIn: () => { const at = now(); onTrimChange?.(at, trimEnd !== null && trimEnd <= at ? null : trimEnd); },
    markOut: () => { const at = now(); if (at > trimStart) onTrimChange?.(trimStart, at); },
    jumpIn: () => seek(trimStart), jumpOut: () => seek(trimEnd ?? duration), resetRange: () => onTrimChange?.(0, null),
    toggleWipe: () => toggleCanvasMode('wipe'), toggleDiff: () => toggleCanvasMode('diff'),
    ...bufferShortcuts,
    solo: () => selectSolo(selectedNode), bypass: () => bypassNode(selectedNode), sweep: () => { if (previewRequest) setSweepPanel('geometry'); },
    extractFace: quickExtractFace,
    snapshotA: () => void captureSlot('A'), snapshotB: () => void captureSlot('B'), snapshotC: () => void captureSlot('C'), snapshotD: () => void captureSlot('D'),
  }, active && !sweepPanel);
  // The preview window toggles work from every screen, so they live outside the faceswap-only map above.
  useShortcutActions({ floating: floating.toggle, pip: () => void floating.openPip() }, !sweepPanel);

  const mediaDims = probe ? { width: probe.width, height: probe.height } : { width: 1920, height: 1080 };
  const handleCompareContextMenu = useCallback((clientX: number, clientY: number) => {
    const rect = viewportRef.current?.getBoundingClientRect();
    if (!rect) return;
    const letterbox = calculateLetterbox(rect.width, rect.height, mediaDims.width, mediaDims.height);
    const store = useFaceSelectionStore.getState(), faces = store.getFacesAtTimestamp(time, fps);
    const x = clientX - rect.left, y = clientY - rect.top;
    const hit = faces.find(face => { const box = mapBoxToPixels(face.bbox, letterbox); return x >= box.x - 4 && x <= box.x + box.w + 4 && y >= box.y - 4 && y <= box.y + box.h + 4; });
    if (!hit) return;
    store.setActiveSelection(hit.id);
    setFaceMenu({ x: clientX, y: clientY, faceRank: leftToRightRank(faces, hit), pose: facePose(hit) });
  }, [time, fps, mediaDims.width, mediaDims.height]);

  // Preview Box zoom, pan, grid, peek, and display state
  const [previewZoom, setPreviewZoom] = preference('zoom');
  const [previewPan, setPreviewPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panOrigin, setPanOrigin] = useState({ x: 0, y: 0 });
  const [gridMode, setGridMode] = preference('grid_mode');
  const [showSourcePeek, setShowSourcePeek] = useState(false);
  const [showFaceOverlays, setShowFaceOverlays] = preference('show_overlays');
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
    if (!liveBlob) {
      setSnapshotNotice('No swapped preview is available to download yet.');
      setTimeout(() => setSnapshotNotice(''), 3000);
      return;
    }
    downloadBlob(liveBlob, previewFilename());
    setSnapshotNotice('Swapped preview downloaded');
    setTimeout(() => setSnapshotNotice(''), 3000);
  }, [liveBlob]);

  // Frame snapshot download
  const handleDownloadSnapshot = useCallback(() => {
    try {
      if (!showSourcePeek && liveBlob && !live.stale) { handleDownloadLive(); return; }
      let dataUrl = '';
      if (showSourcePeek && sourcePreview.source) {
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
  }, [showSourcePeek, sourcePreview.source, outputPreview.source, output, time, liveBlob, live.stale, handleDownloadLive]);

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
    const showLive = isOutput && !isShowingSourcePeek && !playing && !!liveUrl;

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
              onTimeUpdate={!isOutput ? event => { if (playing) setPlayhead(event.currentTarget.currentTime); } : undefined}
            />
            {!playing && !showLive && activeResult.source && (
              <img
                className="frame-preview"
                src={activeResult.source}
                alt={`${isOutput ? (isShowingSourcePeek ? 'Source peek' : 'Processed') : 'Source'} frame at ${time.toFixed(2)} seconds`}
              />
            )}
            {!isOutput && !isMaskEditing && showFaceOverlays && (
              <LiveTime fallback={time} live={playing}>{liveTime => <FaceOverlayCanvas
                videoRef={original}
                mediaDimensions={mediaDims}
                currentTime={liveTime}
                sources={sources}
                onAssignSource={onAssignSource}
                showOverlays={showFaceOverlays}
                showLandmarks={showLandmarks}
                showPose={showPose}
                onFaceContextMenu={openFaceMenu}
              />}</LiveTime>
            )}
            {isShowingSourcePeek && (
              <div className="source-peek-badge">
                <span>👁</span> SOURCE PEEK (A/B)
              </div>
            )}
          </>
        ) : !showLive && (
          <div className="viewport-empty">
            <span className="film-icon">{isOutput ? (live.status === 'loading' ? '⏳' : live.status === 'error' ? '⚠' : '◇') : '⊞'}</span>
            <h2>{isOutput ? (live.status === 'loading' ? 'Rendering live swap…' : live.status === 'error' ? 'Swap preview failed' : 'Your output starts here') : 'Bring a scene to life'}</h2>
            <p>{isOutput ? (live.status === 'error' ? (live.error || 'Check server logs for preview details.') : live.status === 'loading' ? 'Real-time frame preview is rendering…' : 'Completed renders appear alongside your source.') : 'Import a local video, inspect its faces, and assign an identity.'}</p>
            {isOutput && live.status === 'error' && (
              <button type="button" className="button small primary-soft" onClick={live.retry}>
                Retry preview
              </button>
            )}
          </div>
        )}
        {showLive && (
          <img
            className={`frame-preview live-swap-preview ${live.stale ? 'stale' : ''}`}
            src={liveUrl!}
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
      {!studio && <LayoutToolbar onFloat={floating.openWindow}/>}
      <details className="preview-tools" data-testid="preview-tools">
      <summary><span>Pipeline &amp; snapshots</span><small>Solo or bypass a stage · compare A/B/C/D snapshots</small></summary>
      <SnapshotSlots onCapture={(slot: string) => void captureSlot(slot)} onRestore={restoreWorkspace} busy={captureBusy} captureAvailable={!!previewRequest}/>
      {captureError && <p role="alert" className="inline-warning">{captureError}</p>}
      <PipelineNodeFlowBar selected={selectedNode} solo={solo} bypass={parameters.bypass} onSelect={setSelectedNode} onSolo={selectSolo} onBypass={bypassNode} busy={false} available={!!previewRequest}/>
      </details>
      <div className="viewport-toolbar">
        <span className="eyebrow">WORKSTATION MEDIA INSPECTOR</span>
        <button className="button small" disabled={!history.past.length} onClick={history.undo}>Undo</button><button className="button small" disabled={!history.future.length} onClick={history.redo}>Redo</button><button className="button small" onClick={() => setHistoryOpen(true)}>Edit history ({history.past.length})</button>
        <button className="button small" disabled={!previewRequest} onClick={() => { setPlaying(false); original.current?.pause(); processed.current?.pause(); setSweepPanel('geometry'); }}>3×3 Sweep</button>
        <button
          type="button"
          className={`button small use-face-frame-btn ${detectingFaces ? 'busy' : ''}`}
          disabled={!target || detectingFaces}
          onClick={handleUseFaceFromFrame}
          title="Detect and select target faces directly from the currently displayed preview frame (like Roop Ultimate)"
        >
          <span>🎯</span> {detectingFaces ? 'Detecting Faces…' : 'Use Face from this Frame'}
        </button>
        <span className="use-as-source-group" role="group" aria-label="Use a face from this frame as the source">
          <input type="number" min={1} max={16} value={sourceFaceNumber} aria-label="Face number, left to right" title="Which face in this frame (1 = leftmost)"
            onChange={event => setSourceFaceNumber(Math.max(1, Math.min(16, Math.round(Number(event.target.value)) || 1)))}/>
          <button type="button" className={`button small use-as-source-btn ${extraction.busy ? 'busy' : ''}`} disabled={!target || extraction.busy}
            onClick={() => void extraction.run('source_gallery', { faceRank: sourceFaceNumber - 1 })} title="Extract this face (aligned crop, ArcFace embedding, 3D pose) and make it the active source face">
            <span>🧑</span> {extraction.busy ? 'Extracting…' : 'Use as source'}
          </button>
        </span>
        {detectionNotice && <span className="detection-inline-notice">{detectionNotice}</span>}
        {extraction.toast && <span className={`face-toast ${extraction.toast.kind}`} role="status">{extraction.toast.text}</span>}
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

          {/* 3D pose & landmark HUD toggles, and the full-bleed comparison canvas */}
          <button type="button" className={`hud-btn ${showLandmarks ? 'active' : ''}`} aria-pressed={showLandmarks} onClick={() => setShowLandmarks(value => !value)}
            title="5-point ArcFace landmarks on every detected face"><span>⁙</span> Landmarks</button>
          <button type="button" className={`hud-btn ${showPose ? 'active' : ''}`} aria-pressed={showPose} onClick={() => setShowPose(value => !value)}
            title="Live yaw / pitch / roll in degrees (pitch + = looking up) on every detected face"><span>↻</span> Pose</button>
          <button type="button" className={`hud-btn compare-toggle ${viewMode === 'compare' ? 'active' : ''}`} aria-pressed={viewMode === 'compare'} disabled={playing}
            onClick={() => setViewMode(mode => (mode === 'compare' ? 'side' : 'compare'))}
            title="Full-bleed comparison canvas: wipe, crossfade, alpha blend, diff map and 3.5x loupe"><span>◫</span> Compare</button>

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
            disabled={!liveBlob}
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

          {/* Detachable always-visible preview: in-app floating window, or the browser's Picture-in-Picture window */}
          <button type="button" className={`hud-btn float-preview-btn ${floating.mode === 'window' ? 'active' : ''}`} aria-pressed={floating.mode === 'window'} onClick={floating.toggle}
            title="Floating preview: stays visible on every screen (Shift+F, remappable)"><span>◳</span> Float</button>
          {floating.supportsPip && <button type="button" className={`hud-btn pip-preview-btn ${floating.mode === 'pip' ? 'active' : ''}`} aria-pressed={floating.mode === 'pip'} onClick={() => void floating.openPip()}
            title="Always-on-top Picture-in-Picture preview window (Shift+G, remappable)"><span>⧉</span> PiP</button>}

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
          {liveUrl ? <img src={liveUrl} className={live.stale ? 'stale' : ''} alt={`${solo} stage output at frame ${live.meta?.frame_index ?? currentFrame}`}/> : <p role="status">{live.error || (live.status === 'busy' ? `GPU busy (${live.stage}); retrying` : live.status === 'loading' ? 'Rendering this stage…' : 'No buffer is available at this frame')}</p>}
          <span className="pane-label">SOLO · {PIPELINE_NODES.find(node => node.id === solo)?.label ?? solo} · frame {live.meta?.frame_index ?? currentFrame}</span>
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
          {viewMode === 'compare' && !playing && !isMaskEditing && !solo ? (
            <div className="compare-host">
              <WorkstationCanvas fullBleed output={output} originalUrl={sourcePreview.source} swappedUrl={liveUrl || outputPreview.source}
                mode={compMode} onModeChange={(next: string) => setCompMode(next as typeof compMode)} containerWidth={viewportSize.width} containerHeight={viewportSize.height}
                mediaDimensions={mediaDims} onContextMenuPoint={handleCompareContextMenu} currentTime={time} fps={fps}/>
              {showFaceOverlays && <FaceOverlayCanvas videoRef={original} mediaDimensions={mediaDims} currentTime={time} sources={sources} fps={fps}
                showOverlays showLandmarks={showLandmarks} showPose={showPose} interactive={false}/>}
            </div>
          ) : (
            <>
              {pane(target, sourcePreview, false)}
              {!isMaskEditing && pane(output, outputPreview, true)}
            </>
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
      {solo && live.status === 'error' && <p className="inline-warning" role="alert">{live.error}<button className="button small" onClick={live.retry}>Retry preview</button></p>}
      {solo && live.meta?.notes.map(note => <p className="muted buffer-note" key={note}>{note}</p>)}
      {(() => { const panels = <div className="parameter-panels">
        {(Object.keys(PANELS) as PanelId[]).map(panel => (
          <fieldset key={panel} className={`parameter-panel panel-${panel}`}>
            <legend>{PANELS[panel].title}</legend>
            <p className="muted panel-hint">{PANELS[panel].hint}</p>
            {panel === 'geometry' && <label className="numeric-control">Pose correction method<select aria-label="Pose correction method" value={parameters.pose_mode ?? 'retarget'} onChange={event => setParameters({ ...parameters, pose_mode: event.target.value as 'retarget' | 'mesh_blend' })}><option value="mesh_blend">2.5D / 3D mesh blend (canonical proxy)</option><option value="retarget">Reference LivePortrait retarget</option></select></label>}
            {panel === 'masking' && <label className="numeric-control">Occlusion handling<select aria-label="Occlusion handling" value={parameters.occlusion_mode ?? 'standard'} onChange={event => setParameters({ ...parameters, occlusion_mode: event.target.value as 'standard' | 'robust' })}><option value="standard">Standard (parser + landmark hull)</option><option value="robust">Robust (hands, hair, glasses, props, face depth order)</option></select></label>}
            {settingsOf(panel).map(spec => (
              <SettingControl key={spec.id} spec={spec} label={spec.id === 'pose_strength' && parameters.pose_mode !== 'mesh_blend' ? 'Reference retarget strength' : undefined}
                value={settingValue(spec, effectiveParameters)} onChange={value => editParameter(spec.id, value)}
                disabled={spec.id === 'restoration_blend' && !restorerAvailable} disabledReason="GPEN weights are unavailable, so the restorer cannot run."/>
            ))}
            <button type="button" className="button small sweep-btn" disabled={!previewRequest} aria-label={`3×3 Sweep for ${PANELS[panel].title}`}
              onClick={() => { setPlaying(false); original.current?.pause(); processed.current?.pause(); setSweepPanel(panel); }}>3×3 Sweep</button>
          </fieldset>
        ))}
      </div>; return studio ? (studio.panels ? createPortal(panels, studio.panels) : null) : panels; })()}
      {floating.portal}
      {historyOpen && <HistoryDrawer onClose={() => setHistoryOpen(false)}/>}
      {faceMenu && <FaceContextMenu target={faceMenu} activeSourceName={activeSourceName} busy={extraction.busy} onClose={() => setFaceMenu(null)}
        onChoose={(destination, menu) => { setFaceMenu(null); void extraction.run(destination, menu); }}/>}
      {sweepPanel && <ContactSheetModal panel={sweepPanel} parameters={parameters} frameLabel={`Frame ${currentFrame} · ${time.toFixed(3)} s`}
        payloadFor={(patch: Patch) => {
          const { fidelity_weight: _inactive, ...applied } = patch;
          const sweeping = 'restoration_blend' in applied;
          return buildLivePayload({ timestampSelection: true, request: previewRequest, time, fps, restorerAvailable, maskBase64, solo: 'final',
            parameters: { ...parameters, ...applied, bypass: sweeping ? parameters.bypass.filter(item => item !== 'restorer') : parameters.bypass } });
        }}
        onApply={(patch: Patch) => { const { fidelity_weight: _inactive, ...applied } = patch; setParameters({ ...parameters, ...applied }); }} onClose={() => setSweepPanel(null)}/>}
      {(() => { const timeline = <TimelineWorkstation
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
      />;
      const tray = <LiveTime fallback={time} live={playing}>{liveTime => <FaceSelectionTray
        currentTime={liveTime}
        sources={sources}
        onSeekTime={setTime}
        onAssignSource={onAssignSource}
        onUseFaceFromFrame={openMenuFromBar}
        canUseFaceFromFrame={!!target}
      />}</LiveTime>;
      if (studio) return <>{studio.timeline && createPortal(timeline, studio.timeline)}{studio.tray && createPortal(tray, studio.tray)}</>;
      return <DockArea>
      {/* Video Editor-Grade Workstation Timeline */}
      <DockPanel id="timeline" title="Timeline">{timeline}</DockPanel>
      {/* Target Face Selection Tray */}
      <DockPanel id="tray" title="Face Tray">{tray}</DockPanel>
      <DockPanel id="inspector" title="Inspector">{inspector ?? <p className="muted">Processing inspector</p>}</DockPanel>
      <DockPanel id="pose" title="3D Pose & Geometry">{posePanel ?? <p className="muted">Select a source identity to inspect geometry.</p>}</DockPanel>
      </DockArea>; })()}
    </section>
  );
}

export function formatTime(value: number) {
  return `${Math.floor(value / 60).toString().padStart(2, '0')}:${(value % 60).toFixed(2).padStart(5, '0')}`;
}
