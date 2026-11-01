import { lazy, Suspense, useCallback, useEffect, useRef, useState, type ComponentType } from 'react';
import { api, post, upload } from './api';
import type { Capabilities, Config, Coverage, FaceSet, Inspection, Job, JobRequest, Probe, Project, Reconstruction, Telemetry, Track } from './contracts';
import { EventConnection } from './events';
import { featureReason, jobActions, mappings, optionReason } from './state';
import { SettingsInspector } from './SettingsInspector';
import { VideoViewport, formatTime } from './VideoViewport';
import { JOB_START_EVENT } from './livePreview';
import { useFaceSelectionStore, faceEditState, restoreFaceEdit, buildFaceSegments, hydrateFaceSegments } from './useFaceSelectionStore';
import useHistoryStore, { executeCommand, replaceCurves } from './stores/useHistoryStore';
import FaceManager from './components/facemgr/FaceManager';
import AppRouter, { Navigation, useHashRoute } from './router/AppRouter';
import { useShortcutActions } from './components/controls/Hotkeys';
import NumericInput from './components/controls/NumericInput';
import { DEFAULT_PARAMETERS, type WorkstationParameters } from './components/workstation/pipeline';
import WorkstationScreens from './components/workstation/WorkstationScreens';
import PipelinePill from './components/hud/PipelinePill';
import QueueDrawer from './components/queue/QueueDrawer';
import PreFlightLinterModalComponent from './components/diagnostics/PreFlightLinterModal';
import { DEFAULT_EXPORT } from './components/export/VfxExportPanel';
import type { ExportOptions } from './contracts';
import FacesetLibrary from './components/facemgr/FacesetLibrary';
import ThemeGallery from './components/ThemeGallery';
import { THEMES, themeByName, applyThemeToDom } from './themes';
const PreFlightLinterModal = PreFlightLinterModalComponent as unknown as ComponentType<any>;
const PoseCoverageInspector = lazy(() => import('./ThreeViewport').then(module => ({ default: module.PoseCoverageInspector })));

export default function App() {
  const [caps, setCaps] = useState<Capabilities | null>(null), [config, setConfig] = useState<Config | null>(null);
  const [sources, setSources] = useState<FaceSet[]>([]), [jobs, setJobs] = useState<Job[]>([]), [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [target, setTarget] = useState(''), [probe, setProbe] = useState<Probe | null>(null), [time, setTime] = useState(0), [keys, setKeys] = useState<number[]>([]);
  const [inspection, setInspection] = useState<Inspection | null>(null), [selectedTrack, setSelectedTrack] = useState<string>('');
  const [sourceId, setSourceId] = useState(''), [coverage, setCoverage] = useState<Coverage | null>(null), [mesh, setMesh] = useState<Reconstruction | null>(null);
  const [reconstructions, setReconstructions] = useState<{ reconstruction_id: string }[]>([]);
  const [selectedJob, setSelectedJob] = useState(''), [queueOpen, setQueueOpen] = useState(true), [connection, setConnection] = useState('connecting');
  const [project, setProject] = useState<Project | null>(null), [projectName, setProjectName] = useState('Untitled project'), [projects, setProjects] = useState<Pick<Project, 'project_id' | 'name' | 'saved_at' | 'last_completed_frame'>[]>([]), [projectsOpen, setProjectsOpen] = useState(false);
  const [trimStart, setTrimStart] = useState(0), [trimEnd, setTrimEnd] = useState<number | null>(null), [includeCheckpoint, setIncludeCheckpoint] = useState(false);
  const [maskBase64, setMaskRaw] = useState<string | null>(null);
  const [linterOpen, setLinterOpen] = useState(false);
  const [busy, setBusy] = useState(''), [error, setError] = useState(''), [notice, setNotice] = useState(''), [uploadProgress, setUploadProgress] = useState(0);
  const [themeName, setThemeName] = useState(() => localStorage.getItem('roop3d-theme-name') ?? 'Default');
  const [themeGalleryOpen, setThemeGalleryOpen] = useState(false);
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
  const [exportOptions, setExportOptions] = useState<ExportOptions>(DEFAULT_EXPORT);
  const [parameters, setParametersRaw] = useState<WorkstationParameters>({ ...DEFAULT_PARAMETERS });
  const assigned = useFaceSelectionStore(state => state.assignments);
  const clusters = useFaceSelectionStore(state => state.targetClusters);
  const curves = useHistoryStore(state => state.curves);
  const setAssigned = (value: Record<string, string> | ((previous: Record<string, string>) => Record<string, string>)) => useFaceSelectionStore.getState().replaceAssignments(typeof value === 'function' ? value(useFaceSelectionStore.getState().assignments) : value);
  const setParameters = (value: WorkstationParameters) => { if (JSON.stringify(value) === JSON.stringify(parameters)) return; const before = parameters; const changed = Object.keys(value).filter(key => JSON.stringify(value[key as keyof WorkstationParameters]) !== JSON.stringify(before[key as keyof WorkstationParameters])); executeCommand({ label: `Change ${changed.join(', ')}`, mergeKey: changed.length === 1 && changed[0] !== 'bypass' ? `parameter-${changed[0]}` : undefined, undo: () => setParametersRaw(before), redo: () => setParametersRaw(value) }); };
  const setMaskBase64 = (value: string | null) => { if (value === maskBase64) return; const before = maskBase64; executeCommand({ label: value ? 'Mask stroke' : 'Clear manual mask', bytes: (before?.length ?? 0) + (value?.length ?? 0), undo: () => setMaskRaw(before), redo: () => setMaskRaw(value) }); };
  const renderMappings = () => { const values: Record<string, string> = {}; for (const cluster of clusters) if (cluster.mode === 'swap' && assigned[cluster.id]) values[cluster.originTrackId ?? cluster.id] = assigned[cluster.id]; return mappings(values, inspection?.tracks ?? [], sources); };
  const captureWorkspace = () => ({ parameters, exportOptions, maskBase64, curves, faces: faceEditState(), time, sourceId, selectedTrack, trimStart, trimEnd });
  const restoreWorkspace = (value: ReturnType<typeof captureWorkspace>) => { const before = captureWorkspace(); const apply = (snapshot: typeof before) => { setParametersRaw(snapshot.parameters); setExportOptions(snapshot.exportOptions ?? DEFAULT_EXPORT); setMaskRaw(snapshot.maskBase64); replaceCurves(snapshot.curves); restoreFaceEdit(snapshot.faces); setTime(snapshot.time); setSourceId(snapshot.sourceId); setSelectedTrack(snapshot.selectedTrack); setTrimStart(snapshot.trimStart); setTrimEnd(snapshot.trimEnd); useHistoryStore.getState().live(); }; executeCommand({ label: 'Restore snapshot settings', undo: () => apply(before), redo: () => apply(value) }); };
  const [previewLatency, setPreviewLatency] = useState<Record<string, number> | null>(null);
  const fileInput = useRef<HTMLInputElement>(null), sourceInput = useRef<HTMLInputElement>(null), projectInput = useRef<HTMLInputElement>(null);
  const uploadAbort = useRef<AbortController | null>(null), queueKey = useRef({ fingerprint: '', key: '' });
  const currentJob = jobs.find(job => job.job_id === selectedJob) ?? null;
  const output = currentJob?.state === 'succeeded' && currentJob.request.target_path === target ? currentJob.output?.output_path ?? '' : '';
  const currentTrack: Track | null = inspection?.tracks.find(track => track.track_id === selectedTrack) ?? null;
  const projectDirty = !project || project.name !== projectName || project.request.target_path !== target
    || project.request.inspection_id !== inspection?.inspection_id || (project.request.trim_start_seconds ?? 0) !== trimStart
    || project.request.trim_end_seconds !== trimEnd
    || JSON.stringify(project.request.face_mappings) !== JSON.stringify(renderMappings())
    || JSON.stringify(project.effective_settings) !== JSON.stringify(config)
    || JSON.stringify({ ...DEFAULT_PARAMETERS, ...project.request.workstation_parameters, pose_mode: project.request.workstation_parameters?.pose_mode ?? 'retarget' }) !== JSON.stringify(parameters)
    || JSON.stringify(project.request.parameter_curves?.tracks ?? {}) !== JSON.stringify(curves)
    || JSON.stringify(project.request.export ?? DEFAULT_EXPORT) !== JSON.stringify(exportOptions)
    || (project.request.mask_base64 ?? null) !== maskBase64
    || !!project.request.face_segments && JSON.stringify(project.request.face_segments) !== JSON.stringify(buildFaceSegments(clusters, assigned));
  const reason = (name: string) => featureReason(caps, name);
  const refreshJobs = useCallback(async () => setJobs(await api<Job[]>('/jobs')), []);
  const refreshSources = useCallback(async () => setSources(await api<FaceSet[]>('/facesets')), []);
  const refreshProjects = useCallback(async () => setProjects(await api<typeof projects>('/projects')), []);
  const run = async (label: string, work: () => Promise<void>) => { setBusy(label); setError(''); setNotice(''); try { await work(); } catch (error) { if ((error as Error).name !== 'AbortError') setError((error as Error).message); } finally { setBusy(''); } };
  const connect = useCallback(async () => {
    try {
      const [capabilities, settings, sourceList, queue, saved] = await Promise.all([api<Capabilities>('/system/capabilities'), api<Config>('/config'), api<FaceSet[]>('/facesets'), api<Job[]>('/jobs'), api<typeof projects>('/projects')]);
      setCaps(capabilities); setConfig(settings); setSources(sourceList); setJobs(queue); setProjects(saved); setError('');
    } catch (error) { setError((error as Error).message); }
  }, []);
  useEffect(() => { void connect(); }, [connect]);
  useEffect(() => {
    const t = themeByName(themeName);
    applyThemeToDom(t);
    localStorage.setItem('roop3d-theme-name', themeName);
    localStorage.setItem('roop3d-theme', t.mode);
  }, [themeName]);
  useEffect(() => {
    let refresh: ReturnType<typeof setTimeout> | undefined;
    const events = new EventConnection(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/v1/ws/events`, event => {
      if (event.type === 'system.telemetry' && event.hardware) setTelemetry(event.hardware);
      if (event.type.startsWith('job.') || event.type === 'stream.ready' || event.type === 'stream.gap') {
        if (!refresh) refresh = setTimeout(() => { refresh = undefined; void refreshJobs().catch(error => setError(error.message)); }, 120);
      }
    }, setConnection);
    events.start();
    return () => { events.stop(); if (refresh) clearTimeout(refresh); };
  }, [refreshJobs]);
  useEffect(() => () => uploadAbort.current?.abort(), []);
  useEffect(() => {
    if (!projectsOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    const dialog = document.querySelector<HTMLElement>('[role="dialog"]');
    const focusables = () => [...(dialog?.querySelectorAll<HTMLElement>('button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled)') ?? [])];
    focusables()[0]?.focus();
    const handle = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setProjectsOpen(false);
      if (event.key === 'Tab') { const items = focusables(), first = items[0], last = items.at(-1); if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); } else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); } }
    };
    document.addEventListener('keydown', handle);
    return () => { document.removeEventListener('keydown', handle); previous?.focus(); };
  }, [projectsOpen]);
  useEffect(() => {
    if (!sourceId) { setCoverage(null); return; }
    const controller = new AbortController(); setCoverage(null);
    void api<Coverage>(`/facesets/${sourceId}/coverage`, { signal: controller.signal }).then(setCoverage).catch(error => { if (error.name !== 'AbortError') setError(error.message); });
    return () => controller.abort();
  }, [sourceId]);
  const loadTarget = async (path: string) => {
    const [info, frames, reconstructionList] = await Promise.all([post<Probe>('/media/probe', { path }), api<{ timestamps: number[] }>(`/media/keyframes?${new URLSearchParams({ path })}`), api<typeof reconstructions>('/reconstructions')]);
    setTarget(path); setProbe(info); setTime(0); setKeys(frames.timestamps); setInspection(null); setAssigned({}); setSelectedTrack(''); setMesh(null); setTrimStart(0); setTrimEnd(null); setReconstructions(reconstructionList); setSelectedJob(''); setProject(null);
    useFaceSelectionStore.getState().reset(); useHistoryStore.getState().reset(); setMaskRaw(null);
  };
  const importAsset = (file: File | undefined) => {
    if (!file || !caps) return;
    void run('Importing asset', async () => {
      const controller = new AbortController(); uploadAbort.current = controller; setUploadProgress(0);
      const result = await upload(file, caps, setUploadProgress, controller.signal);
      if (result.kind === 'video' && result.path) await loadTarget(result.path);
      else { await refreshSources(); setSourceId(result.faceset_id ?? ''); }
      setNotice('Local asset imported and validated.');
    });
  };
  const inspect = () => void run('Inspecting target faces', async () => {
    const result = await post<Inspection>('/media/inspect-faces', { path: target });
    setInspection(result); setAssigned({}); setSelectedTrack(result.tracks[0]?.track_id ?? '');
    useFaceSelectionStore.getState().populateFromInspection(result, probe, {});
    if (!result.tracks.length) setNotice('No faces were detected. Try a clearer target frame or another clip.');
  });
  const request = (): JobRequest => {
    const selected = renderMappings();
    if (!target || !inspection || !selected.length) throw new Error('Inspect a target and map at least one face track before continuing.');
    return { target_path: target, inspection_id: inspection.inspection_id, face_mappings: selected, processing: 'local', remote_consent: false, trim_start_seconds: trimStart || null, trim_end_seconds: trimEnd, mask_base64: maskBase64 || undefined, workstation_parameters: parameters, parameter_curves: { tracks: curves }, face_segments: buildFaceSegments(clusters, assigned), export: exportOptions };
  };
  const executeQueue = () => void run('Queueing local render', async () => {
    const value = request(), fingerprint = JSON.stringify(value);
    if (queueKey.current.fingerprint !== fingerprint) queueKey.current = { fingerprint, key: crypto.randomUUID() };
    window.dispatchEvent(new Event(JOB_START_EVENT)); // drop in-flight live previews so the render can take the GPU promptly
    const job = await post<Job>('/jobs', value, { headers: { 'Idempotency-Key': queueKey.current.key } });
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
  const saveProject = () => void run('Saving project', async () => {
    const checkpoint = includeCheckpoint && currentJob && jobActions(caps, currentJob.state).includes('resume') ? currentJob.job_id : null;
    const saved = await post<Project>('/projects', { name: projectName, request: request(), checkpoint_job_id: checkpoint });
    setProject(saved); await refreshProjects(); setNotice('Project saved with input and model hashes.');
  });
  const openProject = (id: string) => void run('Verifying project hashes', async () => {
    const saved = await api<Project>(`/projects/${id}`);
    for (const [field, value] of Object.entries(saved.effective_settings)) {
      if (config?.[field] !== value && !caps?.workstation.read_only_parameters[field]) {
        const blocked = optionReason(caps, field, String(value)); if (blocked) throw new Error(`Saved ${field} is unavailable: ${blocked}`);
      }
    }
    const settings = await api<Config>('/config', { method: 'PUT', body: JSON.stringify(saved.effective_settings) });
    await loadTarget(saved.request.target_path);
    const loaded = await api<Inspection>(`/media/inspections/${saved.request.inspection_id}?${new URLSearchParams({ path: saved.request.target_path })}`);
    const restoredMappings = Object.fromEntries(saved.request.face_mappings.map(item => [item.track_id, item.faceset_id]));
    setConfig(settings); setInspection(loaded); setAssigned(restoredMappings); setSelectedTrack(loaded.tracks[0]?.track_id ?? '');
    useFaceSelectionStore.getState().populateFromInspection(loaded, null, restoredMappings);
    if (saved.request.face_segments) hydrateFaceSegments(saved.request.face_segments);
    replaceCurves(saved.request.parameter_curves?.tracks ?? {}); setMaskRaw(saved.request.mask_base64 ?? null);
    setExportOptions(saved.request.export ?? DEFAULT_EXPORT);
    setParametersRaw({ ...DEFAULT_PARAMETERS, ...saved.request.workstation_parameters, pose_mode: saved.request.workstation_parameters?.pose_mode ?? 'retarget' });
    location.hash = '#/faceswap';
    setTrimStart(saved.request.trim_start_seconds ?? 0); setTrimEnd(saved.request.trim_end_seconds); setProject(saved); setProjectName(saved.name); setSelectedJob(saved.checkpoint_job_id ?? ''); setIncludeCheckpoint(!!saved.checkpoint_job_id); setProjectsOpen(false); setNotice('Project input and model hashes verified. Saved settings restored.');
  });
  const importProject = (file: File | undefined) => { if (!file) return; void run('Importing project manifest', async () => { if (file.size > 512 * 1024) throw new Error('Project manifest exceeds 512 KiB.'); const saved = await post<Project>('/projects/import', JSON.parse(await file.text())); await refreshProjects(); setNotice(`Imported ${saved.name}. Open it from the project library.`); }); };
  const controlJob = (job: Job, action: string) => void run(`${action} job`, async () => {
    const path = action === 'resume' && project?.checkpoint_job_id === job.job_id ? `/projects/${project.project_id}/resume` : `/jobs/${job.job_id}/${action}`;
    await post(path, {}); await refreshJobs();
  });
  const showJob = (job: Job) => {
    if (job.request.target_path === target) { setSelectedJob(job.job_id); return; }
    void run('Opening job target', async () => {
      await loadTarget(job.request.target_path);
      const loaded = await api<Inspection>(`/media/inspections/${job.request.inspection_id}?${new URLSearchParams({ path: job.request.target_path })}`);
      const restoredMappings = Object.fromEntries(job.request.face_mappings.map(item => [item.track_id, item.faceset_id]));
      setInspection(loaded); setAssigned(restoredMappings);
      useFaceSelectionStore.getState().populateFromInspection(loaded, null, restoredMappings);
      setSelectedTrack(loaded.tracks[0]?.track_id ?? ''); setSelectedJob(job.job_id);
      if (job.request.face_segments) hydrateFaceSegments(job.request.face_segments);
      replaceCurves(job.request.parameter_curves?.tracks ?? {}); setMaskRaw(job.request.mask_base64 ?? null);
      setExportOptions(job.request.export ?? DEFAULT_EXPORT);
      setParametersRaw({ ...DEFAULT_PARAMETERS, ...job.request.workstation_parameters, pose_mode: job.request.workstation_parameters?.pose_mode ?? 'retarget' });
      setTrimStart(job.request.trim_start_seconds ?? 0); setTrimEnd(job.request.trim_end_seconds);
    });
  };
  useShortcutActions({ save: saveProject, undo: () => useHistoryStore.getState().undo(), redo: () => useHistoryStore.getState().redo() }, !projectsOpen);
  const previewRequest = (() => { try { return request(); } catch { return null; } })();
  const checkpointAllowed = !!currentJob && jobActions(caps, currentJob.state).includes('resume');
  return <div className="workstation">
    <header className="app-header"><a className="brand" href="#/home" aria-label="Roop 3D home"><span className="brand-mark">R<span>3</span></span><strong>roop<span>3D</span></strong></a><span className="header-divider"/><span className="workspace-label">VIDEO WORKSTATION</span>
      <div className="project-name"><input aria-label="Project name" maxLength={128} value={projectName} onChange={event => setProjectName(event.target.value)} /><span>{projectDirty ? 'UNSAVED' : 'SAVED'}</span></div>
      <span className="local-status"><i className={`dot ${connection === 'online' ? 'mint' : 'amber'}`}/>{connection === 'online' ? 'Local service' : connection}</span>
      <PipelinePill telemetry={telemetry} activeJob={currentJob} />
      <div role="group" aria-label="UI zoom" title="UI Zoom (Ctrl + / − / 0)" className="ui-zoom-group">
        <button type="button" onClick={() => bumpZoom(-0.05)} title="Zoom out (Ctrl −)" aria-label="Zoom out" className="zoom-btn">−</button>
        <button type="button" onClick={resetZoom} title="Reset zoom (Ctrl 0)" aria-label={`Reset zoom, ${Math.round(zoom * 100)}%`} className="zoom-reset-btn">{Math.round(zoom * 100)}%</button>
        <button type="button" onClick={() => bumpZoom(0.05)} title="Zoom in (Ctrl +)" aria-label="Zoom in" className="zoom-btn">+</button>
      </div>
      <div className="header-theme-group" title="Active Interface Theme">
        <span className="theme-badge-label">THEME:</span>
        <select
          aria-label="Select Theme"
          value={themeName}
          onChange={e => setThemeName(e.target.value)}
          className="theme-dropdown-select"
        >
          <optgroup label="Dark Themes">
            {THEMES.filter(t => t.mode === 'dark').map(t => (
              <option key={t.name} value={t.name}>{t.name}</option>
            ))}
          </optgroup>
          <optgroup label="Light Themes">
            {THEMES.filter(t => t.mode === 'light').map(t => (
              <option key={t.name} value={t.name}>{t.name}</option>
            ))}
          </optgroup>
        </select>
        <button
          type="button"
          className="button small theme-gallery-btn"
          title="Open visual swatch cards for all 37 themes"
          onClick={() => setThemeGalleryOpen(true)}
        >
          🎨 Swatches
        </button>
        <button
          type="button"
          className="icon-button"
          aria-label={`Switch to ${themeByName(themeName).mode === 'dark' ? 'light' : 'dark'} theme`}
          title={`Quick toggle: ${themeByName(themeName).mode === 'dark' ? 'Light mode' : 'Dark mode'}`}
          onClick={() => setThemeName(themeByName(themeName).mode === 'dark' ? 'Glass Light' : 'Default')}
        >
          {themeByName(themeName).mode === 'dark' ? '☀' : '◐'}
        </button>
      </div>
      <button className="button" disabled={!!reason('projects') || !!busy} title={reason('projects') ?? ''} onClick={() => { setProjectsOpen(true); void refreshProjects(); }}>Projects</button><button className="button" disabled={!!reason('projects') || !inspection || !!busy} onClick={saveProject}>Save project</button>
    </header>
    <Navigation route={route}/>
    {(error || notice || busy) && <div className={`notification ${error ? 'error' : ''}`} role={error ? 'alert' : 'status'}><span>{error || busy || notice}</span>{busy === 'Importing asset' && <><progress value={uploadProgress} max={1}/><button onClick={() => uploadAbort.current?.abort()}>Cancel import</button></>}{error && <button onClick={() => void connect()}>Reconnect</button>}{!busy && <button aria-label="Dismiss message" onClick={() => { setError(''); setNotice(''); }}>×</button>}</div>}
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
      <div className="workstation-body"><aside className="asset-sidebar"><div className="section-title"><span>ASSET LIBRARY</span><span className="muted">01</span></div><div className="asset-actions"><button className="button full primary-soft" disabled={!!reason('uploads') || !!busy} title={reason('uploads') ?? ''} onClick={() => fileInput.current?.click()}>＋ Import target video</button><input ref={fileInput} hidden type="file" accept=".mp4,.mov,.mkv,.avi,.webm,.m4v,.ts" onChange={event => { importAsset(event.target.files?.[0]); event.target.value = ''; }} /></div>
      <div className={`target-asset ${target ? 'loaded' : ''}`}><span className="asset-symbol">▤</span><div><strong>{target ? target.split('/').at(-1) : 'No target loaded'}</strong><small>{probe ? `${probe.width} × ${probe.height} · ${probe.fps_rational ?? '?'} fps` : 'Your files stay on this machine'}</small></div></div>
      <div className="section-title spaced"><span>SOURCE IDENTITIES</span><span>{sources.length.toString().padStart(2, '0')}</span></div>
      <FacesetLibrary
        sources={sources}
        selectedSourceId={sourceId}
        onSelectSource={id => setSourceId(id)}
        onRefreshSources={refreshSources}
        onImportFaceset={file => importAsset(file)}
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
      <VideoViewport onSourceExtracted={faceset => { setSources(previous => previous.some(item => item.faceset_id === faceset.faceset_id) ? previous : [...previous, faceset]); setSourceId(faceset.faceset_id); void refreshSources(); }} maskBase64={maskBase64} captureWorkspace={captureWorkspace} restoreWorkspace={restoreWorkspace} active={route === "#/faceswap"} parameters={parameters} setParameters={setParameters} previewRequest={previewRequest} restorerAvailable={!!caps?.models.some(model => model.name === "gpen_bfr_512.onnx" && model.state === "VERIFIED_LOCAL")} onPreviewLatency={setPreviewLatency} inspector={<SettingsInspector namespace="workstation-setting" caps={caps} config={config} save={saveSettings} busy={!!busy}/>} posePanel={<><Suspense fallback={<p className="muted empty-note">Loading pose inspection…</p>}><PoseCoverageInspector coverage={coverage} track={currentTrack} mesh={mesh}/></Suspense><label className="reconstruction-choice">Reconstructed artifact<select value="" disabled={!target || !reconstructions.length || !!reason('reconstruction_viewer')} onChange={event => { const id = event.target.value; if (!id) { setMesh(null); return; } void run('Verifying reconstruction', async () => setMesh(await api<Reconstruction>(`/reconstructions/${id}?${new URLSearchParams({ target_path: target })}`))); }}><option value="">{mesh ? 'Choose another artifact' : 'No reconstruction loaded'}</option>{reconstructions.map(item => <option key={item.reconstruction_id} value={item.reconstruction_id}>{item.reconstruction_id.slice(0, 12)}</option>)}</select></label>{mesh && <button className="button small" onClick={() => setMesh(null)}>Show pose coverage</button>}</>} target={target} output={output} time={time} setTime={setTime} duration={probe?.duration_seconds ?? 0} enabled={!reason('frame_preview')} keys={keys} probe={probe} sources={sources} onInspectionUpdate={setInspection} onMaskChange={setMaskBase64} trimStart={trimStart} trimEnd={trimEnd} onTrimChange={(start, end) => { setTrimStart(start); setTrimEnd(end); }} onAssignSource={(clusterId, facesetId) => {
        setAssigned(previous => {
          const next = { ...previous };
          if (facesetId) next[clusterId] = facesetId;
          else delete next[clusterId];
          return next;
        });
        if (facesetId) setSourceId(facesetId);
      }}/>
      <div className="trim-bar">
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
      <section className="face-inspector"><div className="panel-heading"><div><span className="eyebrow">TARGET FACE INSPECTOR</span><h2>Assign an identity</h2></div><button className="button" disabled={!target || !!reason('face_inspection') || !!busy} title={reason('face_inspection') ?? ''} onClick={inspect}>{inspection ? 'Reinspect faces' : 'Inspect target faces'}</button></div>
        {!inspection ? <div className="mapping-empty"><span>◎</span><p>Inspect the target to find persistent face tracks.<small>Then map each track to an observed source FaceSet.</small></p></div> : <><div className="inspection-meta"><span>{inspection.tracks.length} detected tracks</span><span>Identity continuity is estimated</span>{inspection.truncated && <strong>Bounded sample — truncated</strong>}</div>{inspection.warnings.map(warning => <p key={warning} className="inline-warning">{warning}</p>)}
          <div className="track-list">{inspection.tracks.map(track => <div className={`track-card ${selectedTrack === track.track_id ? 'active' : ''}`} key={track.track_id}>
            <button className="track-select" aria-label={`Inspect ${track.track_id}`} onClick={() => { setSelectedTrack(track.track_id); setTime(track.first_timestamp_seconds); if (assigned[track.track_id]) setSourceId(assigned[track.track_id]); }}>{track.thumbnail_base64 ? <img src={`data:${track.thumbnail_mime};base64,${track.thumbnail_base64}`} alt={`${track.track_id} detected face`}/> : <span className="track-fallback">◎</span>}<span><strong>{track.track_id.replace('track_', 'Face ')}</strong><small>Frames {track.first_frame}–{track.last_frame}</small><small>{formatTime(track.first_timestamp_seconds)} → {formatTime(track.last_timestamp_seconds)}</small></span></button><span className="mapping-arrow">→</span><label className="mapping-choice"><span>Source identity</span><select aria-label={`Identity for ${track.track_id}`} value={assigned[track.track_id] ?? ''} disabled={!sources.length} onChange={event => { const nextVal = event.target.value; setAssigned(previous => ({ ...previous, [track.track_id]: nextVal })); useFaceSelectionStore.getState().assignSource(track.track_id, nextVal || null); setSourceId(nextVal); }}><option value="">Keep original</option>{sources.map(source => <option key={source.faceset_id} value={source.faceset_id}>{source.name}</option>)}</select></label></div>)}</div></>}
      </section>
      <QueueDrawer
        jobs={jobs}
        selectedJobId={selectedJob}
        onSelectJob={showJob}
        onRefreshJobs={refreshJobs}
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
        </div>
      )}
    </main></div></AppRouter>
    <WorkstationScreens route={route} caps={caps} config={config} sources={sources} jobs={jobs} projects={projects} busy={!!busy} telemetry={telemetry} connection={connection} openProject={openProject} showJob={showJob} controlJob={controlJob} saveSettings={saveSettings} refreshJobs={refreshJobs} openLibrary={() => { setProjectsOpen(true); void refreshProjects(); }} latency={previewLatency} target={target} probe={probe} inspection={inspection} refreshProjects={refreshProjects} exportOptions={exportOptions} onExportChange={setExportOptions} meshBlend={parameters.pose_mode === 'mesh_blend'} onQueue={queue} canQueue={!reason('queue') && !!inspection && !!renderMappings().length && !busy} previewRequest={inspection && renderMappings().length ? request() : null} timestamp={time} themeName={themeName} onThemeChange={setThemeName}/>
    <footer className="app-footer"><span>ROOP 3D / LOCAL WORKSTATION</span><span>Observed identities. Explicit processing. Synthetic media.</span><span>API :{config?.port ?? '—'}</span></footer>
    {projectsOpen && <div className="modal-backdrop" onClick={() => setProjectsOpen(false)}><section role="dialog" aria-modal="true" aria-labelledby="projects-title" className="project-modal" onClick={event => event.stopPropagation()}><div className="panel-heading"><div><span className="eyebrow">VERSIONED .R3DP MANIFESTS</span><h2 id="projects-title">Project library</h2></div><button aria-label="Close projects" className="icon-button" onClick={() => setProjectsOpen(false)}>×</button></div><p className="muted">Reopening verifies target, inspection, FaceSet, and model hashes. Project files contain references and settings, never raw media.</p><button className="button" disabled={!!busy} onClick={() => projectInput.current?.click()}>Import .r3dp</button><input ref={projectInput} hidden type="file" accept=".r3dp,application/json" onChange={event => { importProject(event.target.files?.[0]); event.target.value = ''; }}/>{!projects.length && <p className="empty-note muted">No saved projects yet.</p>}{projects.map(item => <div key={item.project_id} className="saved-project"><div><strong>{item.name}</strong><small>{new Date(item.saved_at).toLocaleString()} · checkpoint {item.last_completed_frame >= 0 ? item.last_completed_frame : 'none'}</small></div><button className="button small" disabled={!!busy} onClick={() => openProject(item.project_id)}>Verify & open</button><a className="button small" href={`/api/v1/projects/${item.project_id}/download`} download>Export</a></div>)}{project?.checkpoint_job_id && <button className="button accent" disabled={!!busy || !!reason('queue')} onClick={() => void run('Verifying checkpoint continuation', async () => { const resumed = await post<Job>(`/projects/${project.project_id}/resume`, {}); setSelectedJob(resumed.job_id); await refreshJobs(); setProjectsOpen(false); })}>Verify & resume {project.name}</button>}</section></div>}
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
    />
    {themeGalleryOpen && (
      <div className="modal-backdrop" onClick={() => setThemeGalleryOpen(false)}>
        <section
          role="dialog"
          aria-modal="true"
          aria-labelledby="theme-modal-title"
          className="project-modal theme-gallery-modal"
          onClick={event => event.stopPropagation()}
        >
          <div className="panel-heading">
            <div>
              <span className="eyebrow">WORKSTATION PALETTE & STYLES</span>
              <h2 id="theme-modal-title">Theme Gallery ({THEMES.length} Themes)</h2>
            </div>
            <button aria-label="Close themes" className="icon-button" onClick={() => setThemeGalleryOpen(false)}>×</button>
          </div>
          <p className="muted">Select from 38 studio-calibrated dark, AMOLED pure-black, and high-contrast light themes.</p>
          <ThemeGallery
            value={themeName}
            onChange={(name) => {
              setThemeName(name);
            }}
          />
        </section>
      </div>
    )}
  </div>;
}
