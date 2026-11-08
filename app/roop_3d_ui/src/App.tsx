import FacesetBuilder from './components/facemgr/FacesetBuilder';
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type ComponentType } from 'react';
import { api, post, upload, type ApiError } from './api';
import { renderProbe } from './utils/renderProbe';
import type { Capabilities, Config, Coverage, FaceSet, IdentityLock, Inspection, Job, JobRequest, Probe, Project, Reconstruction, Track } from './contracts';
import { IdentityLockBar, IdentityTrackStats, lockProblem, useContactSheet } from './components/facemgr/IdentityLockPanel';
import { EventConnection } from './events';
import { publishHardware } from './stores/telemetryStore';
import { setPlayhead } from './stores/playheadStore';
import { selectEngine, useSelectedEngine } from './stores/engineSelection';
import { setRack, useRack } from './stores/rackSelection';
import { featureReason, jobActions, mappings, optionReason } from './state';
import { SettingsInspector } from './SettingsInspector';
import { VideoViewport, formatTime } from './VideoViewport';
import { JOB_START_EVENT } from './livePreview';
import { useFaceSelectionStore, faceEditState, restoreFaceEdit, buildFaceSegments, hydrateFaceSegments } from './useFaceSelectionStore';
import useHistoryStore, { executeCommand, replaceCurves } from './stores/useHistoryStore';
import FaceManager from './components/facemgr/FaceManager';
import AppRouter, { SCREENS, useHashRoute } from './router/AppRouter';
import Rail from './shell/Rail';
import TopBar from './shell/TopBar';
import CommandPalette, { type Command } from './shell/CommandPalette';
import { useShortcutActions } from './components/controls/Hotkeys';
import NumericInput from './components/controls/NumericInput';
import { DEFAULT_PARAMETERS, restorerWeights, type RestorerModel, type WorkstationParameters } from './components/workstation/pipeline';
import RenderFooter from './components/studio/RenderFooter';
import FaceMapping from './components/studio/FaceMapping';
import WorkstationScreens from './components/workstation/WorkstationScreens';
import PipelinePill from './components/hud/PipelinePill';
import RunPill from './components/faceswap/PipelinePill';
import QueueDrawer from './components/queue/QueueDrawer';
import PreFlightLinterModalComponent from './components/diagnostics/PreFlightLinterModal';
import { DEFAULT_EXPORT } from './components/export/VfxExportPanel';
import type { ExportOptions } from './contracts';
import FacesetLibrary from './components/facemgr/FacesetLibrary';
import QuickTour from './components/tour/QuickTour';
import { tourSeen } from './components/tour/tourSteps';
import ThemeGallery from './components/ThemeGallery';
import GuideDialog, { type GuideTab } from './components/guide/GuideDialog';
import WelcomeDialog from './components/guide/WelcomeDialog';
import NextStepBar from './components/guide/NextStepBar';
import RecoveryNotice from './components/guide/RecoveryNotice';
import { isOfflineError } from './components/guide/errorGuide';
import DialogFrame from './components/guide/DialogFrame';
import ExportPresetBar from './components/export/ExportPresetBar';
import { normalizeExport } from './components/export/exportOptions';
import { markOnboardingSeen, onboardingSeen, workflowProgress } from './components/guide/workflowState';
import type { WorkflowStepId } from './components/guide/helpContent';
import { applyMotionPref, watchSystemMotion } from './utils/motionPrefs';
import { useDialogA11y } from './components/controls/useDialogA11y';
import { THEMES, themeByName, applyThemeToDom } from './themes';
import TargetPanel, { trackLabel } from './components/studio/TargetPanel';
import FaceSetDetail from './components/studio/FaceSetDetail';
import StudioInspector from './components/studio/StudioInspector';
import TrackLanes from './components/studio/TrackLanes';
import PanelBoundary from './components/studio/PanelBoundary';
import { DEFAULT_VIEWPORT, type ViewportPreferences } from './previewPreferences';
import { parseFps } from './components/timeline/TimelineWorkstation';
const PreFlightLinterModal = PreFlightLinterModalComponent as unknown as ComponentType<any>;
const PoseCoverageInspector = lazy(() => import('./ThreeViewport').then(module => ({ default: module.PoseCoverageInspector })));

export default function App() {
  const selectedEngine = useSelectedEngine();
  // The Engine & Model Rack's swapper choice. null = follow the saved global setting.
  const rackSwapper = useRack(state => state.swapper);
  renderProbe('App');
  const [caps, setCaps] = useState<Capabilities | null>(null), [config, setConfig] = useState<Config | null>(null);
  const swapperModelId = rackSwapper ?? (typeof config?.swapper_model_id === 'string' ? config.swapper_model_id : undefined);
  const [sources, setSources] = useState<FaceSet[]>([]), [jobs, setJobs] = useState<Job[]>([]);
  // `time` is the *committed* playhead (seek / step / pause). Per-tick playback time lives in `playheadStore`, and
  // hardware telemetry in `telemetryStore`: neither is React state here, so neither re-renders the workstation.
  const [target, setTarget] = useState(''), [probe, setProbe] = useState<Probe | null>(null), [time, setCommittedTime] = useState(0), [keys, setKeys] = useState<number[]>([]);
  const setTime = useCallback((value: number) => { setCommittedTime(value); setPlayhead(value); }, []);
  const [inspection, setInspection] = useState<Inspection | null>(null), [selectedTrack, setSelectedTrack] = useState<string>('');
  const [identityLock, setIdentityLock] = useState<IdentityLock | null>(null);
  const [sourceId, setSourceId] = useState(''), [coverage, setCoverage] = useState<Coverage | null>(null), [mesh, setMesh] = useState<Reconstruction | null>(null);
  const [identitySourceB, setIdentitySourceB] = useState('');
  const [reconstructions, setReconstructions] = useState<{ reconstruction_id: string }[]>([]);
  const [selectedJob, setSelectedJob] = useState(''), [queueOpen, setQueueOpen] = useState(true), [connection, setConnection] = useState('connecting');
  const [project, setProject] = useState<Project | null>(null), [projectName, setProjectName] = useState('Untitled project'), [projects, setProjects] = useState<Pick<Project, 'project_id' | 'name' | 'saved_at' | 'last_completed_frame'>[]>([]), [projectsOpen, setProjectsOpen] = useState(false);
  const [previewView, setPreviewView] = useState<ViewportPreferences>(DEFAULT_VIEWPORT);
  const [trimStart, setTrimStart] = useState(0), [trimEnd, setTrimEnd] = useState<number | null>(null), [includeCheckpoint, setIncludeCheckpoint] = useState(false);
  const [maskBase64, setMaskRaw] = useState<string | null>(null);
  const [linterOpen, setLinterOpen] = useState(false);
  const [tourOpen, setTourOpen] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false), [guideTab, setGuideTab] = useState<GuideTab>('workflow'), [welcomeOpen, setWelcomeOpen] = useState(() => !onboardingSeen());
  const [errorInfo, setErrorInfo] = useState<{ message: string; code: string | null; requestId: string | null } | null>(null), [projectsLoading, setProjectsLoading] = useState(true);
  const projectDialog = useRef<HTMLElement>(null), workspaceRef = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState(''), [error, setError] = useState(''), [notice, setNotice] = useState(''), [uploadProgress, setUploadProgress] = useState(0);
  const [themeName, setThemeName] = useState(() => localStorage.getItem('roop3d-theme-name') ?? 'Default');
  const [themeGalleryOpen, setThemeGalleryOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if ((event.ctrlKey || event.metaKey) && !event.shiftKey && event.key.toLowerCase() === 'k') { event.preventDefault(); setPaletteOpen(open => !open); } };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  // Stage 7: 'studio' = four-panel workstation (target / identity / preview / inspector + bottom timeline); 'classic'
  // keeps the previous stacked workstation intact. Stored per browser only (a convenience, never project state).
  const [layoutMode, setLayoutMode] = useState<'studio' | 'classic'>(() => { try { return localStorage.getItem('roop3d-workstation-layout') === 'classic' ? 'classic' : 'studio'; } catch { return 'studio'; } });
  useEffect(() => { try { localStorage.setItem('roop3d-workstation-layout', layoutMode); } catch { /* per-session only */ } }, [layoutMode]);
  const [panelsSlot, setPanelsSlot] = useState<HTMLDivElement | null>(null), [timelineSlot, setTimelineSlot] = useState<HTMLDivElement | null>(null), [traySlot, setTraySlot] = useState<HTMLDivElement | null>(null);
  const [identityMesh, setIdentityMesh] = useState<Reconstruction | null>(null);
  const [reconstructionState, setReconstruction] = useState<{ state: 'idle' | 'loading' | 'ready' | 'error'; message?: string }>({ state: 'idle' });
  const [reconstructAttempt, setReconstructAttempt] = useState(0);
  const reconstruction = useMemo(() => {
    const again = () => setReconstructAttempt(value => value + 1);
    return { ...reconstructionState, retry: again, refresh: again, sourceId };
  }, [reconstructionState, sourceId]);
  const [zoom, setZoom] = useState(() => {
    const v = parseFloat(localStorage.getItem('roop3d_zoom') || '1');
    return Number.isFinite(v) && v >= 0.5 && v <= 1.6 ? v : 1;
  });
  const bumpZoom = useCallback((d: number) => setZoom(z => Math.min(1.6, Math.max(0.5, Math.round((z + d) * 20) / 20))), []);
  const resetZoom = useCallback(() => setZoom(1), []);
  useEffect(() => {
    (document.documentElement.style as any).zoom = String(zoom);
    localStorage.setItem('roop3d_zoom', String(zoom));
  }, [zoom]);
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        if (e.key === '=' || e.key === '+') { e.preventDefault(); bumpZoom(0.05); }
        else if (e.key === '-' || e.key === '_') { e.preventDefault(); bumpZoom(-0.05); }
        else if (e.key === '0') { e.preventDefault(); resetZoom(); }
      }
    };
    const handleWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        bumpZoom(e.deltaY < 0 ? 0.05 : -0.05);
      }
    };
    window.addEventListener('keydown', handleKey);
    window.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      window.removeEventListener('keydown', handleKey);
      window.removeEventListener('wheel', handleWheel);
    };
  }, [bumpZoom, resetZoom]);
  const route = useHashRoute();
  useEffect(() => { applyMotionPref(); return watchSystemMotion(); }, []);
  useEffect(() => { document.title = `${SCREENS.find(([id]) => `#/${id}` === route)?.[1] ?? 'Workstation'} · Roop 3D`; }, [route]);
  const [exportOptions, setExportOptions] = useState<ExportOptions>(DEFAULT_EXPORT);
  const [parameters, setParametersRaw] = useState<WorkstationParameters>({ ...DEFAULT_PARAMETERS });
  const assigned = useFaceSelectionStore(state => state.assignments);
  const clusters = useFaceSelectionStore(state => state.targetClusters);
  const curves = useHistoryStore(state => state.curves);
  const setAssigned = (value: Record<string, string> | ((previous: Record<string, string>) => Record<string, string>)) => useFaceSelectionStore.getState().replaceAssignments(typeof value === 'function' ? value(useFaceSelectionStore.getState().assignments) : value);
  const setParameters = (value: WorkstationParameters) => { if (JSON.stringify(value) === JSON.stringify(parameters)) return; const before = parameters; const changed = Object.keys(value).filter(key => JSON.stringify(value[key as keyof WorkstationParameters]) !== JSON.stringify(before[key as keyof WorkstationParameters])); executeCommand({ label: `Change ${changed.join(', ')}`, mergeKey: changed.length === 1 && changed[0] !== 'bypass' ? `parameter-${changed[0]}` : undefined, undo: () => setParametersRaw(before), redo: () => setParametersRaw(value) }); };
  const setMaskBase64 = (value: string | null) => { if (value === maskBase64) return; const before = maskBase64; executeCommand({ label: value ? 'Mask stroke' : 'Clear manual mask', bytes: (before?.length ?? 0) + (value?.length ?? 0), undo: () => setMaskRaw(before), redo: () => setMaskRaw(value) }); };
  const renderMappings = () => {
    const values: Record<string, string> = {};
    for (const [key, source] of Object.entries(assigned)) {
      if (source) values[key] = source;
    }
    for (const cluster of clusters) {
      const trackId = cluster.originTrackId ?? cluster.id;
      if (cluster.mode === 'swap') {
        const source = assigned[cluster.id] ?? assigned[trackId];
        if (source) values[trackId] = source;
      } else {
        delete values[trackId];
      }
    }
    return mappings(values, inspection?.tracks ?? [], sources);
  };
  const captureWorkspace = () => ({ parameters, exportOptions, maskBase64, curves, faces: faceEditState(), time, sourceId, identitySourceB, selectedTrack, trimStart, trimEnd });
  const restoreWorkspace = (value: ReturnType<typeof captureWorkspace>) => { const before = captureWorkspace(); const apply = (snapshot: typeof before) => { setParametersRaw(snapshot.parameters); setExportOptions(snapshot.exportOptions ?? DEFAULT_EXPORT); setMaskRaw(snapshot.maskBase64); replaceCurves(snapshot.curves); restoreFaceEdit(snapshot.faces); setTime(snapshot.time); setSourceId(snapshot.sourceId); setIdentitySourceB(snapshot.identitySourceB ?? ''); setSelectedTrack(snapshot.selectedTrack); setTrimStart(snapshot.trimStart); setTrimEnd(snapshot.trimEnd); useHistoryStore.getState().live(); }; executeCommand({ label: 'Restore snapshot settings', undo: () => apply(before), redo: () => apply(value) }); };
  const [previewLatency, setPreviewLatency] = useState<Record<string, number> | null>(null);
  const fileInput = useRef<HTMLInputElement>(null), sourceInput = useRef<HTMLInputElement>(null), projectInput = useRef<HTMLInputElement>(null);
  const uploadAbort = useRef<AbortController | null>(null), queueKey = useRef({ fingerprint: '', key: '' });
  const currentJob = jobs.find(job => job.job_id === selectedJob) ?? null;
  const output = currentJob?.state === 'succeeded' && currentJob.request.target_path === target ? currentJob.output?.output_path ?? '' : '';
  const { sheet } = useContactSheet(inspection, target);
  const currentTrack: Track | null = inspection?.tracks.find(track => track.track_id === selectedTrack) ?? null;
  const projectDirty = !project || project.name !== projectName || project.request.target_path !== target
    || (project.request.engine_id ?? 'reference_onnx') !== selectedEngine
    || (project.request.swapper_model_id ?? undefined) !== swapperModelId
    || JSON.stringify(project.preview ?? { timestamp: 0, selected_track: '', source_faceset_id: '', viewport: DEFAULT_VIEWPORT }) !== JSON.stringify({ timestamp: time, selected_track: selectedTrack, source_faceset_id: sourceId, viewport: previewView })
    || project.request.inspection_id !== inspection?.inspection_id || (project.request.trim_start_seconds ?? 0) !== trimStart
    || project.request.trim_end_seconds !== trimEnd
    || (project.request.identity_blend?.source_face_ids[1] ?? '') !== identitySourceB
    || JSON.stringify(project.request.face_mappings) !== JSON.stringify(renderMappings())
    || JSON.stringify(project.effective_settings) !== JSON.stringify(config)
    || JSON.stringify({ ...DEFAULT_PARAMETERS, ...project.request.workstation_parameters, pose_mode: project.request.workstation_parameters?.pose_mode ?? 'retarget' }) !== JSON.stringify(parameters)
    || JSON.stringify(project.request.parameter_curves?.tracks ?? {}) !== JSON.stringify(curves)
    || JSON.stringify(normalizeExport(project.request.export ?? DEFAULT_EXPORT)) !== JSON.stringify(normalizeExport(exportOptions))
    || (project.request.mask_base64 ?? null) !== maskBase64
    || !!project.request.face_segments && JSON.stringify(project.request.face_segments) !== JSON.stringify(buildFaceSegments(clusters, assigned));
  const reason = (name: string) => featureReason(caps, name);
  /** One undoable assignment through the face-selection store (the same path the tray and video overlay use). */
  const assignTrack = (trackId: string, facesetId: string | null) => {
    const store = useFaceSelectionStore.getState(), cluster = store.targetClusters.find(item => item.id === trackId);
    if (facesetId && cluster && cluster.mode !== 'swap') { setNotice(`${trackLabel(trackId)} is ${cluster.mode === 'ignore' ? 'a reference guide' : 'excluded'}; set its role to Swap to assign a source.`); return; }
    if (cluster) store.assignSource(trackId, facesetId);
    else setAssigned(previous => { const next = { ...previous }; if (facesetId) next[trackId] = facesetId; else delete next[trackId]; return next; });
    if (facesetId) { setSourceId(facesetId); setSelectedTrack(trackId); setNotice(`${sources.find(item => item.faceset_id === facesetId)?.name ?? 'FaceSet'} assigned to ${trackLabel(trackId)}. The live preview updates without a render.`); }
  };
  /** Select a target track everywhere: inspector, overlays (store selection) and, optionally, the playhead. */
  const selectTrack = (trackId: string, seek = true) => {
    setSelectedTrack(trackId);
    const track = inspection?.tracks.find(item => item.track_id === trackId);
    if (seek && track) setTime(track.first_timestamp_seconds);
    if (assigned[trackId]) setSourceId(assigned[trackId]);
    const store = useFaceSelectionStore.getState(), face = store.detectedFaces.find(item => item.clusterId === trackId);
    if (face && store.activeSelection !== face.id) store.setActiveSelection(face.id);
  };
  // Clicking a face in the video selects it in the store; follow it here so the inspector shows that track.
  const storeCluster = useFaceSelectionStore(state => (state as { selectedClusterId?: string | null }).selectedClusterId ?? null);
  useEffect(() => {
    if (!storeCluster) return;
    const cluster = useFaceSelectionStore.getState().targetClusters.find(item => item.id === storeCluster);
    const trackId = cluster?.originTrackId ?? storeCluster;
    if (inspection?.tracks.some(item => item.track_id === trackId)) setSelectedTrack(trackId);
  }, [storeCluster, inspection]);
  const refreshJobs = useCallback(async () => setJobs(await api<Job[]>('/jobs')), []);
  const refreshSources = useCallback(async () => setSources(await api<FaceSet[]>('/facesets')), []);
  const refreshProjects = useCallback(async () => { setProjectsLoading(true); try { setProjects(await api<typeof projects>('/projects')); } finally { setProjectsLoading(false); } }, []);
  const fail = (error: unknown) => { const message = (error as Error).message; setError(message); setErrorInfo({ message, code: (error as ApiError).code ?? null, requestId: (error as ApiError).requestId ?? null }); };
  const run = async (label: string, work: () => Promise<void>) => { setBusy(label); setError(''); setNotice(''); try { await work(); } catch (error) { if ((error as Error).name !== 'AbortError') fail(error); } finally { setBusy(''); } };
  const connect = useCallback(async () => {
    try {
      // The project list re-verifies every saved project's input hashes on the server, which can take many seconds with
      // large videos. It must never hold the rest of the workstation (capabilities, FaceSets, queue) hostage.
      setProjectsLoading(true);
      void api<typeof projects>('/projects').then(setProjects).catch(fail).finally(() => setProjectsLoading(false));
      const [capabilities, settings, sourceList, queue] = await Promise.all([api<Capabilities>('/system/capabilities'), api<Config>('/config'), api<FaceSet[]>('/facesets'), api<Job[]>('/jobs')]);
      setCaps(capabilities); setConfig(settings); setSources(sourceList); setJobs(queue); setError('');
    } catch (error) { fail(error); }
  }, []);
  useEffect(() => { void connect(); }, [connect]);
  useEffect(() => {
    const t = themeByName(themeName);
    applyThemeToDom(t);
    localStorage.setItem('roop3d-theme-name', themeName);
    localStorage.setItem('roop3d-theme', t.mode);
  }, [themeName]);
  useEffect(() => {
    const isHighPerf = Boolean(busy?.toLowerCase().includes('export') || busy?.toLowerCase().includes('render') || (config as any)?.performance_mode === 'high');
    if (isHighPerf) {
      document.documentElement.setAttribute('data-perf-mode', 'high');
    } else {
      document.documentElement.removeAttribute('data-perf-mode');
    }
  }, [busy, config]);
  useEffect(() => {
    let refresh: ReturnType<typeof setTimeout> | undefined;
    const events = new EventConnection(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/v1/ws/events`, event => {
      if (event.type === 'system.telemetry' && event.hardware) publishHardware(event.hardware);
      if (event.type.startsWith('job.') || event.type === 'stream.ready' || event.type === 'stream.gap') {
        if (!refresh) refresh = setTimeout(() => { refresh = undefined; void refreshJobs().catch(error => setError(error.message)); }, 120);
      }
    }, setConnection);
    events.start();
    return () => { events.stop(); if (refresh) clearTimeout(refresh); };
  }, [refreshJobs]);
  useEffect(() => () => uploadAbort.current?.abort(), []);
  useDialogA11y(projectDialog, () => setProjectsOpen(false), projectsOpen);
  useEffect(() => { setIdentityMesh(null); }, [sourceId, reconstructAttempt]);
  // One reconstruction per source FaceSet: the payload (both meshes, the UV texture, the observed pose points and the
  // coverage telemetry) goes straight to the 3D viewer. The target's pose is read by the viewer itself, so selecting a
  // different face or track never re-requests or re-parses it. The server caches by archive hash.
  useEffect(() => {
    if (!sourceId) { setCoverage(null); setMesh(null); setReconstruction({ state: 'idle' }); return; }
    const controller = new AbortController(); setCoverage(null); setMesh(null); setReconstruction({ state: 'loading' });
    // Always on: this runs for every render mode (including Standard identity swap). POST builds (or hits the server's
    // archive-hash cache), then GET 3d-state reads back the cached reconstruction the viewer binds to; if the read-back
    // fails for any reason other than an abort, the POST payload (identical content) is used.
    void post<Coverage>('/faceset/reconstruct-3d', { faceset_id: sourceId }, { signal: controller.signal })
      .then(async posted => {
        const cached = await api<Coverage>(`/faceset/3d-state?${new URLSearchParams({ faceset_id: sourceId })}`, { signal: controller.signal })
          .catch(error => { if (error.name === 'AbortError') throw error; return posted; });
        return cached?.faceset_id === sourceId ? cached : posted;
      })
      .then(value => { setCoverage(value); setReconstruction({ state: 'ready' }); })
      .catch(error => { if (error.name !== 'AbortError') setReconstruction({ state: 'error', message: error.message }); });
    return () => controller.abort();
  }, [sourceId, reconstructAttempt]);
  useEffect(() => { const refresh = () => setReconstructAttempt(v => v + 1); window.addEventListener("faceset-builder-saved", refresh); return () => window.removeEventListener("faceset-builder-saved", refresh); }, []);
  const loadTarget = async (path: string) => {
    const [info, frames, reconstructionList] = await Promise.all([post<Probe>('/media/probe', { path }), api<{ timestamps: number[] }>(`/media/keyframes?${new URLSearchParams({ path })}`), api<typeof reconstructions>('/reconstructions')]);
    setTarget(path); setProbe(info); setTime(0); setKeys(frames.timestamps); setInspection(null); setIdentityLock(null); setAssigned({}); setSelectedTrack(''); setMesh(null); setTrimStart(0); setTrimEnd(null); setReconstructions(reconstructionList); setSelectedJob(''); setProject(null); setIdentityMesh(null);
    useFaceSelectionStore.getState().reset(); useHistoryStore.getState().reset(); setMaskRaw(null);
    return info;
  };
  const importAssets = (files: File[]) => {
    if (!files.length) return;
    if (files.length > 32) { setError('Import at most 32 images or FaceSets at a time.'); return; }
    if (!caps) { setNotice('Still connecting to the local service. Try again in a moment.'); return; }
    void run('Importing asset', async () => {
      const controller = new AbortController(); uploadAbort.current = controller; setUploadProgress(0);
      const imagePaths: string[] = [];
      let facesetId = '';
      for (const [index, file] of files.entries()) {
        const result = await upload(file, caps, progress => setUploadProgress((index + progress) / files.length), controller.signal);
        if (result.kind === 'video' && result.path) await loadTarget(result.path);
        else if (result.kind === 'image' && result.path) imagePaths.push(result.path);
        else if (result.kind === 'faceset') facesetId = result.faceset_id ?? '';
      }
      if (imagePaths.length) {
        const result = await post<{ faceset_id: string }>('/faceset/reconstruct-3d', { source_image_paths: imagePaths }, { signal: controller.signal });
        facesetId = result.faceset_id;
      }
      if (facesetId) { await refreshSources(); setSourceId(facesetId); }
      setNotice('Local asset imported and validated.');
    });
  };
  const importAsset = (file: File | undefined) => { if (file) importAssets([file]); };
  const inspect = () => void run('Inspecting target faces', async () => {
    const result = await post<Inspection>('/media/inspect-faces', { path: target, identity_lock: true,
      full_video: true, time_budget_seconds: 900 });
    setInspection(result); setIdentityLock(null); setAssigned({}); setSelectedTrack(result.tracks[0]?.track_id ?? '');
    useFaceSelectionStore.getState().populateFromInspection(result, probe, {});
    if (!result.tracks.length) setNotice('No faces were detected. Try a clearer target frame or another clip.');
  });
  const request = (isLivePreview = false): JobRequest => {
    const selected = renderMappings();
    if(identitySourceB && identitySourceB === sourceId) throw new Error("Choose distinct Source A and Source B identities.");
    if (!target || !inspection || !selected.length) throw new Error('Inspect a target and map at least one face track before continuing.');
    const segments = buildFaceSegments(clusters, assigned);
    if (!isLivePreview) {
      const lockIssue = lockProblem(identityLock, Object.fromEntries(selected.map(item => [item.track_id, item.faceset_id])));
      if (lockIssue) throw new Error(`${lockIssue} Re-lock or release the identity lock before rendering.`);
    }
    return { engine_id: selectedEngine, swapper_model_id: swapperModelId, identity_lock_id: identityLock?.lock_id, identity_blend: identitySourceB && sourceId ? {method: 'slerp',source_face_ids: [sourceId, identitySourceB],weights:[1,0]} : undefined, target_path: target, inspection_id: inspection.inspection_id, face_mappings: selected, processing: 'local', remote_consent: false, trim_start_seconds: trimStart || null, trim_end_seconds: trimEnd, mask_base64: maskBase64 || undefined, workstation_parameters: parameters, parameter_curves: { tracks: curves }, face_segments: segments, export: exportOptions };
  };
  const executeQueue = () => void run('Queueing local render', async () => {
    const saved = await persistProject(), fingerprint = JSON.stringify({ project: saved.project_id, request: saved.request });
    if (queueKey.current.fingerprint !== fingerprint) queueKey.current = { fingerprint, key: crypto.randomUUID() };
    window.dispatchEvent(new Event(JOB_START_EVENT)); // drop in-flight live previews so the render can take the GPU promptly
    const job = await post<Job>(`/projects/${saved.project_id}/jobs`, {}, { headers: { 'Idempotency-Key': queueKey.current.key } });
    queueKey.current = { fingerprint: '', key: '' }; setSelectedJob(job.job_id); setQueueOpen(true); await refreshJobs(); setNotice('Render added to the local queue.');
  });
  const queue = () => {
    try {
      request();
    } catch (e: any) {
      setNotice(e.message);
      return;
    }
    const skip = sessionStorage.getItem('roop3d_skip_linter') === 'true';
    if (skip) {
      executeQueue();
    } else {
      setLinterOpen(true);
    }
  };
  const saveSettings = (value: Config) => void run('Applying settings', async () => { const saved = await api<Config>('/config', { method: 'PUT', body: JSON.stringify(value) }); setConfig(saved); await connect(); setNotice('Effective settings saved. Existing paused jobs retain their original settings identity.'); });
  const persistProject = async (copy = false) => {
    const checkpoint = includeCheckpoint && currentJob && jobActions(caps, currentJob.state).includes('resume') ? currentJob.job_id : null;
    const draft = { name: projectName, request: request(), checkpoint_job_id: checkpoint, revision: !copy ? project?.revision ?? 0 : undefined,
      preview: { timestamp: time, selected_track: selectedTrack, source_faceset_id: sourceId, viewport: previewView } };
    const saved = project && !copy
      ? await api<Project>(`/projects/${project.project_id}`, { method: 'PUT', body: JSON.stringify(draft) })
      : await post<Project>('/projects', draft);
    setProject(saved); await refreshProjects(); setNotice('Project saved with input and model hashes.');
    return saved;
  };
  const saveProject = () => void run('Saving project', async () => { await persistProject();
  });
  const openProject = (id: string) => void run('Verifying project hashes', async () => {
    const saved = await api<Project>(`/projects/${id}`);
    for (const [field, value] of Object.entries(saved.effective_settings)) {
      if (config?.[field] !== value && !caps?.workstation.read_only_parameters[field]) {
        const blocked = optionReason(caps, field, String(value)); if (blocked) throw new Error(`Saved ${field} is unavailable: ${blocked}`);
      }
    }
    const settings = await api<Config>('/config', { method: 'PUT', body: JSON.stringify(saved.effective_settings) });
    const savedProbe = await loadTarget(saved.request.target_path);
    const loaded = await api<Inspection>(`/media/inspections/${saved.request.inspection_id}?${new URLSearchParams({ path: saved.request.target_path })}`);
    const restoredLock = saved.request.identity_lock_id ? await api<IdentityLock>(`/identity/locks/${saved.request.identity_lock_id}?verify=true`) : null;
    const restoredMappings = Object.fromEntries(saved.request.face_mappings.map(item => [item.track_id, item.faceset_id]));
    selectEngine(saved.request.engine_id ?? 'reference_onnx');
    setRack({ swapper: saved.request.swapper_model_id ?? null });
    setConfig(settings); setInspection(loaded); setAssigned(restoredMappings); setSelectedTrack(loaded.tracks[0]?.track_id ?? '');
    setIdentityLock(restoredLock);
    setSelectedTrack(saved.preview?.selected_track || loaded.tracks[0]?.track_id || '');
    setSourceId(saved.preview?.source_faceset_id || saved.request.face_mappings[0]?.faceset_id || '');
    setTime(Math.min(saved.preview?.timestamp ?? 0, savedProbe.duration_seconds ?? 86400));
    setPreviewView({ ...DEFAULT_VIEWPORT, ...saved.preview?.viewport });
    // The probe gives the overlay real pixel → frame mapping; without it every face box falls back to one default rectangle.
    useFaceSelectionStore.getState().populateFromInspection(loaded, savedProbe, restoredMappings);
    if (saved.request.face_segments) hydrateFaceSegments(saved.request.face_segments);
    replaceCurves(saved.request.parameter_curves?.tracks ?? {}); setMaskRaw(saved.request.mask_base64 ?? null);
    setExportOptions(saved.request.export ?? DEFAULT_EXPORT);
    setIdentitySourceB(saved.request.identity_blend?.source_face_ids[1] ?? '');
    setParametersRaw({ ...DEFAULT_PARAMETERS, ...saved.request.workstation_parameters, pose_mode: saved.request.workstation_parameters?.pose_mode ?? 'retarget' });
    location.hash = '#/faceswap';
    setTrimStart(saved.request.trim_start_seconds ?? 0); setTrimEnd(saved.request.trim_end_seconds); setProject(saved); setProjectName(saved.name); setSelectedJob(saved.checkpoint_job_id ?? saved.output_history?.at(-1)?.job_id ?? ''); setIncludeCheckpoint(!!saved.checkpoint_job_id); setProjectsOpen(false); setNotice('Project input and model hashes verified. Saved settings restored.');
  });
  const importProject = (file: File | undefined) => { if (!file) return; void run('Importing project manifest', async () => { if (file.size > 512 * 1024) throw new Error('Project manifest exceeds 512 KiB.'); const saved = await post<Project>('/projects/import', JSON.parse(await file.text())); await refreshProjects(); setNotice(`Imported ${saved.name}. Open it from the project library.`); }); };
  const controlJob = (job: Job, action: string) => void run(`${action} job`, async () => {
    const path = action === 'resume' && project?.checkpoint_job_id === job.job_id ? `/projects/${project.project_id}/resume` : `/jobs/${job.job_id}/${action}`;
    await post(path, {}); await refreshJobs();
  });
  const showJob = (job: Job) => {
    selectEngine(job.request.engine_id ?? 'reference_onnx');
    setRack({ swapper: job.request.swapper_model_id ?? null });
    void run('Opening job target', async () => {
      const jobProbe = job.request.target_path === target && probe ? probe : await loadTarget(job.request.target_path);
      const loaded = await api<Inspection>(`/media/inspections/${job.request.inspection_id}?${new URLSearchParams({ path: job.request.target_path })}`);
      const linkedProject = job.project_id ? await api<Project>(`/projects/${job.project_id}`) : null;
      setProject(linkedProject); setProjectName(linkedProject?.name ?? 'Untitled project');
      if (job.effective_settings) {
        const settings = await api<Config>('/config', { method: 'PUT', body: JSON.stringify(job.effective_settings) });
        setConfig(settings);
      }
      const restoredMappings = Object.fromEntries(job.request.face_mappings.map(item => [item.track_id, item.faceset_id]));
      setInspection(loaded); setAssigned(restoredMappings);
      useFaceSelectionStore.getState().populateFromInspection(loaded, jobProbe, restoredMappings);
      setSelectedTrack(loaded.tracks[0]?.track_id ?? ''); setSelectedJob(job.job_id);
      setSourceId(job.request.face_mappings[0]?.faceset_id ?? '');
      setIdentityLock(job.request.identity_lock_id ? await api<IdentityLock>(`/identity/locks/${job.request.identity_lock_id}?verify=true`) : null);
      if (job.request.face_segments) hydrateFaceSegments(job.request.face_segments);
      replaceCurves(job.request.parameter_curves?.tracks ?? {}); setMaskRaw(job.request.mask_base64 ?? null);
      setExportOptions(job.request.export ?? DEFAULT_EXPORT);
      setIdentitySourceB(job.request.identity_blend?.source_face_ids[1] ?? '');
      setParametersRaw({ ...DEFAULT_PARAMETERS, ...job.request.workstation_parameters, pose_mode: job.request.workstation_parameters?.pose_mode ?? 'retarget' });
      setTrimStart(job.request.trim_start_seconds ?? 0); setTrimEnd(job.request.trim_end_seconds);
    });
  };
  useShortcutActions({ save: saveProject, undo: () => useHistoryStore.getState().undo(), redo: () => useHistoryStore.getState().redo() }, !projectsOpen);
  const previewRequest = (() => { try { return request(true); } catch { return null; } })();
  const checkpointAllowed = !!currentJob && jobActions(caps, currentJob.state).includes('resume');
  const selectedFaceset = sources.find(item => item.faceset_id === sourceId) ?? null;
  const saveTrackFace = async (name: string) => {
    const track = currentTrack ?? inspection?.tracks[0];
    if (!track?.thumbnail_base64) throw new Error('Inspect target faces first to extract an identity.');
    await post('/facesets/build', { name, views: [{ image_base64: track.thumbnail_base64, bbox: [0, 0, 100, 100] }] });
    await refreshSources();
  };
  const reconstructionPicker = <><FacesetBuilder sourceId={sourceId} onSaved={async id => { await refreshSources(); setSourceId(id); setReconstructAttempt(v => v + 1); }} /><label className="reconstruction-choice">Reconstructed artifact<select value="" disabled={!target || !reconstructions.length || !!reason('reconstruction_viewer')} onChange={event => { const id = event.target.value; if (!id) { setMesh(null); return; } void run('Verifying reconstruction', async () => { setIdentityMesh(null); setMesh(await api<Reconstruction>(`/reconstructions/${id}?${new URLSearchParams({ target_path: target })}`)); }); }}><option value="">{mesh ? 'Choose another artifact' : reconstructions.length ? 'No reconstruction loaded' : 'No reconstruction artifacts'}</option>{reconstructions.map(item => <option key={item.reconstruction_id} value={item.reconstruction_id}>{item.reconstruction_id.slice(0, 12)}</option>)}</select></label>{(mesh || identityMesh) && <button className="button small" onClick={() => { setMesh(null); setIdentityMesh(null); }}>Show pose coverage</button>}</>;
  const viewport = <VideoViewport previewView={previewView} onPreviewViewChange={setPreviewView} studio={layoutMode === 'studio' ? { panels: panelsSlot, timeline: timelineSlot, tray: traySlot } : null} activeSourceId={sourceId} onSourceExtracted={faceset => { setSources(previous => previous.some(item => item.faceset_id === faceset.faceset_id) ? previous : [...previous, faceset]); setSourceId(faceset.faceset_id); void refreshSources(); }} maskBase64={maskBase64} captureWorkspace={captureWorkspace} restoreWorkspace={restoreWorkspace} active={route === "#/faceswap"} parameters={parameters} setParameters={setParameters} previewRequest={previewRequest} restorerAvailable={!!caps?.models.some(model => model.name === restorerWeights(parameters.restorer_model) && model.state === "VERIFIED_LOCAL")} onPreviewLatency={setPreviewLatency} inspector={<SettingsInspector namespace="workstation-setting" caps={caps} config={config} save={saveSettings} busy={!!busy}/>} posePanel={<><FacesetBuilder sourceId={sourceId} onSaved={async id => { await refreshSources(); setSourceId(id); setReconstructAttempt(v => v + 1); }} /><Suspense fallback={<p className="muted empty-note">Loading pose inspection…</p>}><PoseCoverageInspector coverage={coverage} track={currentTrack} mesh={mesh} reconstruction={reconstruction} parameters={parameters} setParameters={setParameters}/></Suspense><label className="reconstruction-choice">Reconstructed artifact<select value="" disabled={!target || !reconstructions.length || !!reason('reconstruction_viewer')} onChange={event => { const id = event.target.value; if (!id) { setMesh(null); return; } void run('Verifying reconstruction', async () => setMesh(await api<Reconstruction>(`/reconstructions/${id}?${new URLSearchParams({ target_path: target })}`))); }}><option value="">{mesh ? 'Choose another artifact' : 'No reconstruction loaded'}</option>{reconstructions.map(item => <option key={item.reconstruction_id} value={item.reconstruction_id}>{item.reconstruction_id.slice(0, 12)}</option>)}</select></label>{mesh && <button className="button small" onClick={() => setMesh(null)}>Show pose coverage</button>}</>} target={target} output={output} time={time} setTime={setTime} duration={probe?.duration_seconds ?? 0} enabled={!reason('frame_preview')} keys={keys} probe={probe} sources={sources} onInspectionUpdate={setInspection} onMaskChange={setMaskBase64} trimStart={trimStart} trimEnd={trimEnd} onTrimChange={(start, end) => { setTrimStart(start); setTrimEnd(end); }} onAssignSource={(clusterId, facesetId) => assignTrack(clusterId, facesetId)}/>;
  const trimBar = <><div className="trim-bar">
        <span className="eyebrow">EXPORT RANGE</span>
        <button
          type="button"
          className="trim-btn jump-in"
          disabled={!target}
          onClick={() => setTime(trimStart)}
          title="Jump playhead to In Point [Shift+I]"
        >
          ⏮ In
        </button>
        <button
          type="button"
          className="trim-btn mark-in"
          disabled={!target}
          onClick={() => setTrimStart(Math.max(0, Math.min(time, (trimEnd ?? probe?.duration_seconds ?? 86400) - 0.01)))}
          title="Set In Point to current playhead [I]"
        >
          [ In
        </button>
        <NumericInput label="Trim start seconds" value={trimStart} min={0} max={Math.max(0, (trimEnd ?? probe?.duration_seconds ?? 86400) - .01)} step={.01} disabled={!!reason('trim') || !target} onChange={value => setTrimStart(value ?? 0)} />
        <NumericInput label="Trim end seconds" value={trimEnd} nullable placeholder="Full clip" min={trimStart + .01} max={probe?.duration_seconds ?? 86400} step={.01} disabled={!!reason('trim') || !target} onChange={setTrimEnd} />
        <button
          type="button"
          className="trim-btn mark-out"
          disabled={!target}
          onClick={() => setTrimEnd(Math.max(trimStart + 0.01, Math.min(time, probe?.duration_seconds ?? 86400)))}
          title="Set Out Point to current playhead [O]"
        >
          Out ]
        </button>
        <button
          type="button"
          className="trim-btn jump-out"
          disabled={!target}
          onClick={() => setTime(trimEnd ?? probe?.duration_seconds ?? 0)}
          title="Jump playhead to Out Point [Shift+O]"
        >
          Out ⏭
        </button>
        {(trimStart > 0 || trimEnd != null) && (
          <button
            type="button"
            className="trim-btn reset"
            onClick={() => { setTrimStart(0); setTrimEnd(null); }}
            title="Reset range to full clip [Alt+R]"
          >
            Reset Range
          </button>
        )}
        <span className="range-summary-pill" title="Active export range duration">
          {probe?.duration_seconds ? (
            `${Math.max(0, ((trimEnd ?? probe.duration_seconds) - trimStart)).toFixed(2)}s · ${Math.round(Math.max(0, ((trimEnd ?? probe.duration_seconds) - trimStart)) * (probe.fps_rational ? (parseInt(probe.fps_rational.split('/')[0], 10) / (parseInt(probe.fps_rational.split('/')[1], 10) || 1)) || 30 : 30))} frames`
          ) : 'Full clip'}
        </span>
      </div>
      <ExportPresetBar value={exportOptions} onChange={setExportOptions} disabled={!!busy} /></>;
  const queuePanel = <><QueueDrawer
        jobs={jobs}
        selectedJobId={selectedJob}
        onSelectJob={showJob}
        onRefreshJobs={refreshJobs}
        onControlJob={controlJob}
        isOpen={queueOpen}
        onToggleOpen={() => setQueueOpen(!queueOpen)}
      />
      {queueOpen && (
        <div className="checkpoint-container">
          <label className="checkpoint-option">
            <input
              type="checkbox"
              checked={includeCheckpoint && checkpointAllowed}
              disabled={!checkpointAllowed}
              onChange={event => setIncludeCheckpoint(event.target.checked)}
            />
            Include selected paused/interrupted checkpoint when saving a project
          </label>
          <small className="muted">Keeps the finished frames with the project, so you can close Roop 3D and resume this render later from Projects.</small>
        </div>
      )}</>;
  const workflow = workflowProgress({ projectSaved: !!project, target, faceSetCount: sources.length, inspected: !!inspection, mappingCount: renderMappings().length, previewSeen: !!previewLatency, jobs });
  const openGuide = (tab: GuideTab = 'workflow') => { setGuideTab(tab); setGuideOpen(true); };
  const goto = (hash: string) => { if (window.location.hash !== hash) window.location.hash = hash; };
  const guideAction = (id: WorkflowStepId) => {
    if (id === 'source') { goto('#/facemgr'); return; }
    goto('#/faceswap');
    if (id === 'project') window.setTimeout(focusProjectName, 80);
    else if (id === 'target') { const blocked = reason('uploads'); if (blocked) setNotice(blocked); else if (!busy) window.setTimeout(() => fileInput.current?.click(), 120); }
    else if (id === 'inspect') { const blocked = reason('face_inspection'); if (!target) setNotice('Import a target video first, then inspect its faces.'); else if (blocked) setNotice(blocked); else if (!busy) inspect(); }
    else if (id === 'assign') setNotice(inspection ? 'Drag a FaceSet card onto a face track, or choose one in the track’s “Source identity” menu.' : 'Inspect the target faces first; then you can assign a FaceSet to each track.');
    else if (id === 'preview') setNotice('Move the playhead to a hard moment (a turn, a hand over the face) and use Compare to check the swap.');
    else if (id === 'render') queue();
  };
  const focusProjectName = () => { const input = document.querySelector<HTMLInputElement>('input[aria-label="Project name"]'); input?.focus(); input?.select(); };
  const dismissWelcome = () => { markOnboardingSeen(); setWelcomeOpen(false); };
  const commands: Command[] = [
    ...SCREENS.map(([id, label]: string[]): Command => ({ id: `go-${id}`, label: `Go to ${label}`, group: 'Navigate', keywords: id, run: () => goto(`#/${id}`) })),
    { id: 'save', label: 'Save project', group: 'Project', hint: 'Ctrl S', disabled: !!reason('projects') || !inspection || !!busy, run: saveProject },
    { id: 'projects', label: 'Open project library', group: 'Project', disabled: !!reason('projects'), run: () => { setProjectsOpen(true); void refreshProjects(); } },
    { id: 'undo', label: 'Undo', group: 'Edit', hint: 'Ctrl Z', run: () => useHistoryStore.getState().undo() },
    { id: 'redo', label: 'Redo', group: 'Edit', hint: 'Ctrl Y', run: () => useHistoryStore.getState().redo() },
    { id: 'theme-mode', label: `Switch to ${themeByName(themeName).mode === 'dark' ? 'light' : 'dark'} theme`, group: 'Appearance', keywords: 'mode dark light', run: () => setThemeName(themeByName(themeName).mode === 'dark' ? 'Glass Light' : 'Default') },
    { id: 'themes', label: 'Browse themes', group: 'Appearance', keywords: 'swatches palette colour color', run: () => setThemeGalleryOpen(true) },
    { id: 'layout', label: `Use ${layoutMode === 'studio' ? 'classic' : 'studio'} layout`, group: 'Appearance', run: () => setLayoutMode(mode => (mode === 'studio' ? 'classic' : 'studio')) },
    { id: 'zoom-reset', label: 'Reset interface zoom', group: 'Appearance', hint: 'Ctrl 0', run: resetZoom },
    { id: 'guide', label: 'Open the guide', group: 'Help', run: () => openGuide() },
    { id: 'tour', label: 'Take the quick tour', group: 'Help', keywords: 'how to use', run: () => setTourOpen(true) },
  ];
  const isHighPerf = Boolean(busy?.toLowerCase().includes('export') || busy?.toLowerCase().includes('render') || (config as any)?.performance_mode === 'high');
  return <div className={`workstation ${isHighPerf ? 'perf-high' : ''}`} data-perf-mode={isHighPerf ? 'high' : 'normal'}>
    <button type="button" className="skip-link" onClick={() => workspaceRef.current?.focus()}>Skip to main content</button>
    <Rail route={route} />
    <div className="app-main">
    <TopBar title={SCREENS.find(([id]: string[]) => `#/${id}` === route)?.[1] ?? 'Workstation'} projectName={projectName} onProjectName={setProjectName} dirty={projectDirty} connection={connection}
      pill={<PipelinePill activeJob={currentJob} />} zoom={zoom} onZoom={bumpZoom} onZoomReset={resetZoom}
      themeMode={themeByName(themeName).mode} onToggleMode={() => setThemeName(themeByName(themeName).mode === 'dark' ? 'Glass Light' : 'Default')} onThemes={() => setThemeGalleryOpen(true)}
      tourFresh={!tourSeen()} onTour={() => setTourOpen(true)} onGuide={() => openGuide()}
      layoutMode={layoutMode} onLayout={() => setLayoutMode(mode => (mode === 'studio' ? 'classic' : 'studio'))}
      projectsDisabled={!!reason('projects')} projectsReason={reason('projects')} busy={!!busy} canSave={!!inspection}
      onPalette={() => setPaletteOpen(true)} onProjects={() => { setProjectsOpen(true); void refreshProjects(); }} onSave={saveProject} onSaveCopy={() => void run('Saving project copy', async () => { await persistProject(true); })} />
    <RunPill />
    {route === '#/faceswap' && <NextStepBar progress={workflow} onAction={guideAction} onGuide={() => openGuide()} blockedReason={workflow.next?.id === 'target' ? reason('uploads') : workflow.next?.id === 'inspect' && target ? reason('face_inspection') : null} busy={!!busy} />}
    {(error || notice || busy) && <div className={`notification ${error ? 'error' : ''}`} role={error ? 'alert' : 'status'}>{error ? <RecoveryNotice message={error} code={errorInfo?.message === error ? errorInfo.code : null} requestId={errorInfo?.message === error ? errorInfo.requestId : null} onGuide={() => openGuide('queue')} /> : <span>{busy || notice}</span>}{busy === 'Importing asset' && <><progress value={uploadProgress} max={1}/><button onClick={() => uploadAbort.current?.abort()}>Cancel import</button></>}{error && isOfflineError({ code: errorInfo?.message === error ? errorInfo.code : null, message: error }) && <button onClick={() => void connect()}>Reconnect</button>}{!busy && <button aria-label="Dismiss message" onClick={() => { setError(''); setNotice(''); }}>×</button>}</div>}
    <div id="workspace" ref={workspaceRef} tabIndex={-1} className="workspace-root">
    <AppRouter route={route} screen="facemgr">
      <FaceManager
        sources={sources}
        target={target}
        inspection={inspection}
        onRefreshSources={refreshSources}
        onAssignSource={(clusterId, facesetId) => {
          setAssigned(previous => {
            const next = { ...previous };
            if (facesetId) next[clusterId] = facesetId;
            else delete next[clusterId];
            return next;
          });
          if (facesetId) setSourceId(facesetId);
        }}
        onBackToWorkstation={() => { window.location.hash = '#/faceswap'; }}
      />
    </AppRouter>
    <AppRouter route={route} screen="faceswap">
      <input ref={fileInput} hidden type="file" accept=".mp4,.mov,.mkv,.avi,.webm,.m4v,.ts,.fsz,.png,.jpg,.jpeg,.webp,.bmp" onChange={event => { importAsset(event.target.files?.[0]); event.target.value = ''; }} />
      {layoutMode === 'studio' ? <div className="studio" data-testid="studio-layout">
        <div className="studio-grid">
          <aside className="studio-panel studio-identity" aria-label="Source identities" data-tour="source">
            <div className="studio-panel-head"><span className="eyebrow">01 · SOURCE FACE / FACESET</span><span className="muted">{sources.length.toString().padStart(2, '0')}</span></div>
            <PanelBoundary name="Identity panel"><FacesetLibrary variant="list" importDisabled={!!reason('uploads') || !!busy} sources={sources} selectedSourceId={sourceId} onSelectSource={id => setSourceId(id)} onRefreshSources={refreshSources} onImportFaceset={files => importAssets(files)}
              canSave={!!currentTrack?.thumbnail_base64} onSaveFace={saveTrackFace} notify={(message, type) => { if (type === 'error') setError(message); else setNotice(message); }} />
            {selectedFaceset ? <FaceSetDetail faceset={selectedFaceset} sources={sources} selectedTrackLabel={selectedTrack ? trackLabel(selectedTrack) : null}
              assignedHere={!!selectedTrack && assigned[selectedTrack] === selectedFaceset.faceset_id} onAssign={() => selectedTrack && assignTrack(selectedTrack, selectedFaceset.faceset_id)}
              identitySourceB={identitySourceB} onIdentitySourceB={setIdentitySourceB} onShowMesh={setIdentityMesh} meshShown={!!identityMesh} />
              : sources.length > 0 && <p className="muted studio-hint">Select a FaceSet to see its views, pose coverage and 3D identity asset.</p>}</PanelBoundary>
          </aside>
          <aside className="studio-panel studio-left" aria-label="Target media" data-tour="tracks">
            <div className="studio-panel-head"><span className="eyebrow">02 · TARGET MEDIA</span></div>
            <PanelBoundary name="Target panel"><TargetPanel target={target} probe={probe} busy={!!busy} uploadReason={reason('uploads')} onImport={() => fileInput.current?.click()}
              time={time} setTime={setTime} fps={parseFps(probe?.fps_rational, 30)} duration={probe?.duration_seconds ?? 0}
              inspection={inspection} inspectReason={reason('face_inspection')} onInspect={inspect} sheet={sheet}
              selectedTrack={selectedTrack} onSelectTrack={trackId => selectTrack(trackId)} sources={sources} assigned={assigned} onAssign={assignTrack}
              lockBar={inspection && <IdentityLockBar target={target} inspection={inspection} mappings={Object.fromEntries(renderMappings().map(item => [item.track_id, item.faceset_id]))} sources={sources} lock={identityLock} onLock={setIdentityLock} busy={!!busy} onError={setError} />} /></PanelBoundary>
          </aside>
          <aside className="studio-panel studio-mapping" aria-label="Target face mapping">
            <div className="studio-panel-head"><span className="eyebrow">03 · TARGET FACE MAPPING</span></div>
            <PanelBoundary name="Face mapping"><FaceMapping tracks={inspection?.tracks ?? []} sources={sources} assigned={assigned} selectedTrack={selectedTrack} onSelectTrack={trackId => selectTrack(trackId)} onAssign={assignTrack} /></PanelBoundary>
          </aside>
          <main className="studio-center" aria-label="Preview">
            <div className="studio-center-head"><div><span className="eyebrow">03 · PREVIEW</span><strong>{target ? target.split('/').at(-1) : 'No target loaded'}</strong></div>
              <button className="button accent" disabled={!!reason('queue') || !inspection || !renderMappings().length || !!busy} title={reason('queue') ?? (renderMappings().length ? 'Render mapped face tracks locally' : 'Assign a FaceSet to at least one track first')} onClick={queue}>Queue render <span>↗</span></button></div>
            {viewport}
          </main>
          <aside className="studio-panel studio-right" aria-label="Inspector">
            <div className="studio-panel-head"><span className="eyebrow">04 · INSPECTOR</span></div>
            <PanelBoundary name="Inspector"><StudioInspector caps={caps} config={config} onSaveConfig={saveSettings} track={currentTrack} sheet={sheet} sources={sources} assignedId={currentTrack ? assigned[currentTrack.track_id] ?? '' : ''} onAssign={assignTrack}
              parameters={parameters} setParameters={setParameters} coverage={coverage} reconstruction={reconstruction} mesh={identityMesh ?? mesh} panelsRef={setPanelsSlot}
              reconstructionPicker={reconstructionPicker}
              facesetCreator={<FacesetBuilder sourceId={sourceId} label="+ Create .fsz from Video/Photos" className="viewer-action-btn"
                onSaved={async id => { await refreshSources(); setSourceId(id); window.dispatchEvent(new Event('faceset-builder-saved')); }} />}
              processing={<SettingsInspector namespace="workstation-setting" caps={caps} config={config} save={saveSettings} busy={!!busy}/>}
              footer={<RenderFooter job={currentJob && currentJob.request.target_path === target ? currentJob : null} onStart={queue}
                startBlockedReason={reason('queue') ?? (!target ? 'Load a target video first.' : !inspection ? 'Inspect the target faces first.' : !renderMappings().length ? 'Assign a source FaceSet to at least one target face first.' : busy ? 'Busy' : null)}
                frameBlockedReason={reason('frame_preview') ?? (!target ? 'Load a target video first.' : !previewRequest ? 'Assign a source FaceSet to a target face first: the preview needs a face mapping.' : null)} />} /></PanelBoundary>
          </aside>
        </div>
        <section className="studio-bottom" aria-label="Timeline">
          <PanelBoundary name="Track lanes"><TrackLanes tracks={inspection?.tracks ?? []} sheet={sheet} sceneCuts={sheet?.scene_cut_frames ?? inspection?.scene_cut_frames ?? []} fps={parseFps(probe?.fps_rational, 30)} duration={probe?.duration_seconds ?? 0}
            trimStart={trimStart} trimEnd={trimEnd} job={currentJob && currentJob.request.target_path === target ? currentJob : null} selectedTrack={selectedTrack} assigned={assigned} sources={sources}
            onSelectTrack={trackId => selectTrack(trackId, false)} onSeek={value => setTime(Math.max(0, Math.min(probe?.duration_seconds ?? 0, value)))} onAssign={assignTrack} /></PanelBoundary>
          <div className="studio-timeline-slot" ref={setTimelineSlot} />
          {trimBar}
          <details className="studio-tray"><summary>Face tray <span className="muted">· faces in the current frame</span></summary><div ref={setTraySlot} /></details>
          {queuePanel}
        </section>
      </div> : <>      <div className="workstation-body"><aside className="asset-sidebar" data-tour="source"><div className="section-title"><span>ASSET LIBRARY</span><span className="muted">01</span></div><div className="asset-actions"><button className="button full primary-soft" disabled={!!reason('uploads') || !!busy} title={reason('uploads') ?? ''} onClick={() => fileInput.current?.click()}>＋ Import target video</button></div>
      <div className={`target-asset ${target ? 'loaded' : ''}`}><span className="asset-symbol">▤</span><div><strong>{target ? target.split('/').at(-1) : 'No target loaded'}</strong><small>{probe ? `${probe.width} × ${probe.height} · ${probe.fps_rational ?? '?'} fps` : 'Your files stay on this machine'}</small></div></div>
      <div className="section-title spaced"><span>SOURCE IDENTITIES</span><span>{sources.length.toString().padStart(2, '0')}</span></div>
      <FacesetLibrary
        sources={sources}
        selectedSourceId={sourceId}
        onSelectSource={id => setSourceId(id)}
        onRefreshSources={refreshSources}
        onImportFaceset={files => importAssets(files)}
        canSave={Boolean(inspection?.tracks?.length)}
        onSaveFace={async (name) => {
          const track = inspection?.tracks.find(t => t.track_id === selectedTrack) || inspection?.tracks[0];
          if (!track?.thumbnail_base64) {
            throw new Error('Inspect target faces first to extract an identity.');
          }
          await post('/facesets/build', {
            name,
            views: [{
              image_base64: track.thumbnail_base64,
              bbox: [0, 0, 100, 100],
            }],
          });
          await refreshSources();
        }}
        notify={(msg, type) => {
          if (type === 'error') setError(msg);
          else setNotice(msg);
        }}
      />
      <div className="library-footer"><span className="tiny-badge">LOCAL BY DEFAULT</span><p>Media, FaceSets, and identity embeddings stay in your local workspace.</p></div>
    </aside>
    <main className="main-workspace"><div className="workspace-heading"><div><span className="eyebrow">EDIT / INSPECT / RENDER</span><h1>Your scene. A new identity.</h1></div><button className="button accent" disabled={!!reason('queue') || !inspection || !renderMappings().length || !!busy} title={reason('queue') ?? 'Render mapped face tracks locally'} onClick={queue}>Queue render <span>↗</span></button></div>
      <label>Identity B for animated Slerp<select aria-label="Identity B for animated Slerp" value={identitySourceB} onChange={event=>setIdentitySourceB(event.target.value)}><option value="">No identity morph</option>{sources.filter(source=>source.faceset_id!==sourceId).map(source=><option key={source.faceset_id} value={source.faceset_id}>{source.name}</option>)}</select></label>
      <label>Colour transfer<select aria-label="Colour transfer" value={parameters.color_method ?? 'lab'} onChange={event=>setParameters({...parameters,color_method:event.target.value as 'lab' | 'histogram'})}><option value="lab">LAB statistics</option><option value="histogram">Stabilized histograms</option></select></label>
      <label>Restoration model<select aria-label="Restoration model" value={parameters.restorer_model ?? 'gpen'} onChange={event=>setParameters({...parameters,restorer_model:event.target.value as RestorerModel})}><option value="gpen">GPEN</option><option value="ultramax">UltraMax</option><option value="codeformer">CodeFormer</option></select></label>
      {viewport}
      {trimBar}
      <section className="face-inspector" data-tour="tracks"><div className="panel-heading"><div><span className="eyebrow">TARGET FACE INSPECTOR</span><h2>Assign an identity</h2></div><button className="button" disabled={!target || !!reason('face_inspection') || !!busy} title={reason('face_inspection') ?? ''} onClick={inspect}>{inspection ? 'Reinspect faces' : 'Inspect target faces'}</button></div>
        {!inspection ? <div className="mapping-empty"><span>◎</span><p>Inspect the target to find persistent face tracks.<small>Then map each track to an observed source FaceSet.</small></p></div> : <><div className="inspection-meta"><span>{inspection.tracks.length} detected tracks</span><span>Identity continuity is estimated</span>{inspection.truncated && <strong>Bounded sample — truncated</strong>}</div>{inspection.warnings.map(warning => <p key={warning} className="inline-warning">{warning}</p>)}
          <div className="track-list">{inspection.tracks.map(track => <div className={`track-card ${selectedTrack === track.track_id ? 'active' : ''}`} key={track.track_id}>
            <button className="track-select" aria-label={`Inspect ${track.track_id}`} onClick={() => { setSelectedTrack(track.track_id); setTime(track.first_timestamp_seconds); if (assigned[track.track_id]) setSourceId(assigned[track.track_id]); }}>{track.thumbnail_base64 ? <img src={`data:${track.thumbnail_mime};base64,${track.thumbnail_base64}`} alt={`${track.track_id} detected face`}/> : <span className="track-fallback">◎</span>}<span><strong>{track.track_id.replace('track_', 'Face ')}</strong><small>Frames {track.first_frame}–{track.last_frame}</small><small>{formatTime(track.first_timestamp_seconds)} → {formatTime(track.last_timestamp_seconds)}</small><IdentityTrackStats track={sheet?.tracks.find(item => item.track_id === track.track_id)} /></span></button><span className="mapping-arrow">→</span><label className="mapping-choice"><span>Source identity</span><select aria-label={`Identity for ${track.track_id}`} value={assigned[track.track_id] ?? ''} disabled={!sources.length} onChange={event => assignTrack(track.track_id, event.target.value || null)}><option value="">Keep original</option>{sources.map(source => <option key={source.faceset_id} value={source.faceset_id}>{source.name}</option>)}</select></label></div>)}</div><IdentityLockBar target={target} inspection={inspection} mappings={Object.fromEntries(renderMappings().map(item => [item.track_id, item.faceset_id]))} sources={sources} lock={identityLock} onLock={setIdentityLock} busy={!!busy} onError={setError} /></>}
      </section>
      {queuePanel}
    </main></div></>}</AppRouter>
    <WorkstationScreens route={route} caps={caps} config={config} sources={sources} jobs={jobs} projects={projects} busy={!!busy} connection={connection} openProject={openProject} showJob={showJob} controlJob={controlJob} saveSettings={saveSettings} refreshJobs={refreshJobs} openLibrary={() => { setProjectsOpen(true); void refreshProjects(); }} latency={previewLatency} target={target} probe={probe} inspection={inspection} refreshProjects={refreshProjects} exportOptions={exportOptions} onExportChange={setExportOptions} meshBlend={parameters.pose_mode === 'mesh_blend'} onQueue={queue} canQueue={!reason('queue') && !!inspection && !!renderMappings().length && !busy} previewRequest={inspection && renderMappings().length ? previewRequest : null} timestamp={time} themeName={themeName} onThemeChange={setThemeName} onStartTour={() => setTourOpen(true)} workflow={workflow} onWorkflowAction={guideAction} onOpenGuide={() => openGuide()} projectsLoading={projectsLoading}/>
    </div>
    <footer className="app-footer"><span>ROOP 3D / LOCAL WORKSTATION</span><span>Observed identities. Explicit processing. Synthetic media.</span><span>API :{config?.port ?? '—'}</span></footer>
    </div>
    {projectsOpen && <div className="modal-backdrop" onClick={() => setProjectsOpen(false)}><section ref={projectDialog} role="dialog" aria-modal="true" aria-labelledby="projects-title" className="project-modal" onClick={event => event.stopPropagation()}><div className="panel-heading"><div><span className="eyebrow">VERSIONED .R3DP MANIFESTS</span><h2 id="projects-title">Project library</h2></div><button aria-label="Close projects" className="icon-button" onClick={() => setProjectsOpen(false)}>×</button></div><p className="muted">Reopening verifies target, inspection, FaceSet, and model hashes. Project files contain references and settings, never raw media.</p><button className="button" disabled={!!busy} onClick={() => projectInput.current?.click()}>Import .r3dp</button><input ref={projectInput} hidden type="file" accept=".r3dp,application/json" onChange={event => { importProject(event.target.files?.[0]); event.target.value = ''; }}/>{!projects.length && <p className="empty-note muted" role="status">{projectsLoading ? 'Checking saved projects… each one is verified against its video and models, which can take a few seconds.' : 'No saved projects yet.'}</p>}{projects.map(item => <div key={item.project_id} className="saved-project"><div><strong>{item.name}</strong><small>{new Date(item.saved_at).toLocaleString()} · checkpoint {item.last_completed_frame >= 0 ? item.last_completed_frame : 'none'}</small></div><button className="button small" disabled={!!busy} onClick={() => openProject(item.project_id)}>Verify & open</button><a className="button small" href={`/api/v1/projects/${item.project_id}/download`} download>Export</a></div>)}{project?.checkpoint_job_id && <button className="button accent" disabled={!!busy || !!reason('queue')} onClick={() => void run('Verifying checkpoint continuation', async () => { const resumed = await post<Job>(`/projects/${project.project_id}/resume`, {}); setSelectedJob(resumed.job_id); await refreshJobs(); setProjectsOpen(false); })}>Verify & resume {project.name}</button>}</section></div>}
    {welcomeOpen && route === '#/home' && <WelcomeDialog caps={caps} onClose={dismissWelcome}
      onStart={() => { dismissWelcome(); goto('#/faceswap'); window.setTimeout(focusProjectName, 150); }}
      onTour={() => { dismissWelcome(); setTourOpen(true); }} onGuide={() => { dismissWelcome(); openGuide(); }} />}
    {guideOpen && <GuideDialog onClose={() => setGuideOpen(false)} progress={workflow} onAction={guideAction} onTour={() => { setGuideOpen(false); setTourOpen(true); }} caps={caps} initialTab={guideTab} />}
    <QuickTour open={tourOpen} onClose={() => setTourOpen(false)} onNavigate={route => { if (window.location.hash !== route) window.location.hash = route; }} />
    <PreFlightLinterModal
      isOpen={linterOpen}
      onClose={() => setLinterOpen(false)}
      onConfirmQueue={executeQueue}
      targetPath={target}
      inspectionId={inspection?.inspection_id || ''}
      faceMappings={renderMappings()}
      workstationParameters={parameters}
      maskBase64={maskBase64}
      hardware={caps}
      onApplySettings={(values: Config) => { if (config) saveSettings({ ...config, ...values }); }}
      onApplyParameter={(name: string, value: number) => setParameters({ ...parameters, [name]: value })}
    />
    <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} commands={commands} />
    {themeGalleryOpen && (
      <DialogFrame title={`Theme Gallery (${THEMES.length} Themes)`} eyebrow="WORKSTATION PALETTE & STYLES" onClose={() => setThemeGalleryOpen(false)} className="theme-gallery-modal" testId="theme-gallery">
        <p className="muted">Select from {THEMES.length} studio-calibrated dark, AMOLED pure-black, and high-contrast light themes.</p>
        <ThemeGallery value={themeName} onChange={name => setThemeName(name)} />
      </DialogFrame>
    )}
  </div>;
}
