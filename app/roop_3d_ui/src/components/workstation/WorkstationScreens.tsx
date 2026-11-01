import { useEffect, useState } from 'react';
import AppRouter from '../../router/AppRouter';
import { api, mediaURL, post } from '../../api';
import type { Capabilities, Config, FaceSet, Inspection, Job, Probe, Project } from '../../contracts';
import { useTelemetry } from '../../stores/telemetryStore';
import VirtualList from '../common/VirtualList';
import { featureReason, jobActions } from '../../state';
import { SettingsInspector } from '../../SettingsInspector';
import { HotkeyEditor } from '../controls/Hotkeys';
import BatchMatrixScreen from '../../screens/BatchMatrixScreen';
import HistoryScreen from '../../screens/HistoryScreen';
import BottleneckFlamegraphComponent from '../diagnostics/BottleneckFlamegraph';
import SettingsScreenComponent from '../../screens/SettingsScreen';
import ExtrasScreenComponent from '../../screens/ExtrasScreen';
import type { ExportOptions, JobRequest } from '../../contracts';
const ExtrasScreen = ExtrasScreenComponent as unknown as React.ComponentType<any>;

const BatchMatrix = BatchMatrixScreen as unknown as React.ComponentType<any>;
const History = HistoryScreen as unknown as React.ComponentType<any>;
const BottleneckFlamegraph = BottleneckFlamegraphComponent as unknown as React.ComponentType<any>;
const SettingsScreen = SettingsScreenComponent as unknown as React.ComponentType<any>;

/** Leaf subscriber: the only part of the Processing screen that re-renders when a hardware tick arrives. */
function GpuReadout() {
  const gpus = useTelemetry(state => state.hardware?.gpus);
  return <>{gpus?.map(gpu => (
    <span key={gpu.name}>
      {gpu.name} · {(gpu.free_vram_mib / 1024).toFixed(1)} GB free
    </span>
  ))}</>;
}

type ProjectSummary =Pick<Project, 'project_id' | 'name' | 'saved_at' | 'last_completed_frame'>;
type Props = {
  exportOptions?: ExportOptions;
  onExportChange?: (value: ExportOptions) => void;
  meshBlend?: boolean;
  onQueue?: () => void;
  canQueue?: boolean;
  previewRequest?: JobRequest | null;
  timestamp?: number;
  route: string;
  caps: Capabilities | null;
  config: Config | null;
  sources: FaceSet[];
  jobs: Job[];
  projects: ProjectSummary[];
  busy: boolean;
  connection: string;
  openProject: (id: string) => void;
  showJob: (job: Job) => void;
  controlJob: (job: Job, action: string) => void;
  saveSettings: (value: Config) => void;
  refreshJobs: () => Promise<void>;
  openLibrary: () => void;
  latency: Record<string, number> | null;
  target?: string;
  probe?: Probe | null;
  inspection?: Inspection | null;
  refreshProjects?: () => Promise<void>;
  themeName?: string;
  onThemeChange?: (name: string) => void;
  onStartTour?: () => void;
};

export default function WorkstationScreens(props: Props) {
  const {
    route,
    caps,
    config,
    sources,
    jobs,
    projects,
    busy,
    connection,
    openProject,
    showJob,
    controlJob,
    saveSettings,
    refreshJobs,
    openLibrary,
    latency,
    target = '',
    probe = null,
    inspection = null,
    refreshProjects = async () => {},
  } = props;

  const [health, setHealth] = useState<unknown>(null);
  const [healthError, setHealthError] = useState('');
  const [selectedOutput, setSelectedOutput] = useState('');
  const [probePath, setProbePath] = useState('');
  const [probeResult, setProbeResult] = useState<unknown>(null);
  const [probeError, setProbeError] = useState('');

  useEffect(() => {
    if (route !== '#/settings') return;
    const controller = new AbortController();
    void api('/health', { signal: controller.signal })
      .then(setHealth)
      .catch(e => {
        if (e.name !== 'AbortError') setHealthError(e.message);
      });
    return () => controller.abort();
  }, [route]);

  const renderJobs = [...jobs].reverse();
  const completed = renderJobs.filter(job => job.state === 'succeeded' && job.output);
  const output = completed.find(job => job.job_id === selectedOutput);
  const projectList = (items: ProjectSummary[]) =>
    items.length ? (
      items.map(item => (
        <div className="saved-project" key={item.project_id}>
          <div>
            <strong>{item.name}</strong>
            <small>
              {new Date(item.saved_at).toLocaleString()} · checkpoint{' '}
              {item.last_completed_frame >= 0 ? item.last_completed_frame : 'none'}
            </small>
          </div>
          <button className="button" disabled={busy} onClick={() => openProject(item.project_id)}>
            Verify & open
          </button>
          <a className="button" href={`/api/v1/projects/${item.project_id}/download`} download>
            Export
          </a>
        </div>
      ))
    ) : (
      <p className="muted empty-note">
        No saved projects yet. Inspect and map a target in the workstation, then save a project.
      </p>
    );

  return (
    <>
      <AppRouter route={route} screen="home">
        <main className="screen-page">
          <span className="eyebrow">LOCAL CREATIVE WORKSPACE</span>
          <h1>Dashboard</h1>
          <p className="muted">New here? Take the four-step tour: source face, target tracking, live preview, then the render queue.</p>
          <button className="button accent tour-start" data-testid="home-tour-button" onClick={() => props.onStartTour?.()}>How to Use · Quick Tour</button>
          <div className="dashboard-cards">
            <a href="#/faceswap" className="screen-card">
              <h2>Start a scene →</h2>
              <p>Import a target, inspect tracks, and assign a FaceSet.</p>
            </a>
            <a href="#/facemgr" className="screen-card">
              <h2>{sources.length} source identities</h2>
              <p>Build and inspect observed FaceSets.</p>
            </a>
            <a href="#/processing" className="screen-card">
              <h2>{jobs.filter(job => ['queued', 'preparing', 'running'].includes(job.state)).length} active jobs</h2>
              <p>Live local queue and execution progress.</p>
            </a>
          </div>
          <h2>Recent projects</h2>
          {projectList([...projects].sort((a, b) => b.saved_at.localeCompare(a.saved_at)).slice(0, 5))}
        </main>
      </AppRouter>

      <AppRouter route={route} screen="batch">
        <BatchMatrix
          sources={sources}
          projects={projects}
          jobs={jobs}
          caps={caps}
          config={config}
          target={target}
          probe={probe}
          inspection={inspection}
          refreshJobs={refreshJobs}
          busy={busy}
        />
      </AppRouter>

      <AppRouter route={route} screen="processing">
        <main className="screen-page">
          <h1>Live processing</h1>
          <div className="telemetry">
            <span>{connection} · local worker {caps?.rendering_worker_available ? 'available' : 'unavailable'}</span>
            <GpuReadout />

          </div>
          {!jobs.length && <p className="empty-note muted">Queue is empty.</p>}
          <VirtualList items={renderJobs} itemKey={job => job.job_id} estimateHeight={88} gap={0} aria-label="Render jobs"
            renderItem={job => (
            <div className="job">
              <button
                className="job-select"
                onClick={() => {
                  showJob(job);
                  location.hash = '#/faceswap';
                }}
              >
                <strong>{job.request.target_path.split(/[\\/]/).at(-1)}</strong>
              </button>
              <div>
                <span>
                  {job.state} · {job.progress.stage} · {job.progress.frame} / {job.progress.total_frames ?? '?'} frames
                </span>
                <progress
                  value={job.progress.frame}
                  max={job.progress.total_frames ?? Math.max(1, job.progress.frame)}
                />
                {job.failure && <p className="inline-warning">{job.failure.message}</p>}
              </div>
              {jobActions(caps, job.state).map(action => (
                <button className="button" key={action} disabled={busy} onClick={() => controlJob(job, action)}>
                  {action}
                </button>
              ))}
            </div>
          )} />
          <BottleneckFlamegraph latency={latency} activeJob={renderJobs.find(job => job.state === 'running') || renderJobs[0] || null} />
          <h2>Measured preview stage timings</h2>
          {latency ? (
            Object.entries(latency).map(([stage, ms]) => (
              <div className="latency-row" key={stage}>
                <span>{stage.replaceAll('_', ' ')}</span>
                <meter min={0} max={Math.max(...Object.values(latency), 1)} value={ms} />
                <span>{ms.toFixed(1)} ms</span>
              </div>
            ))
          ) : (
            <p className="muted">
              Capture a pipeline preview to inspect measured stage timings. Nested worker flamegraph data is unavailable.
            </p>
          )}
          <h2>Distributed worker monitor</h2>
          <p className="muted">Local worker only. No distributed worker telemetry is configured.</p>
        </main>
      </AppRouter>

      <AppRouter route={route} screen="extras">
        <ExtrasScreen active={route === '#/extras'} jobs={jobs} target={target} exportOptions={props.exportOptions} onExportChange={props.onExportChange} meshBlend={props.meshBlend} onQueue={props.onQueue} canQueue={props.canQueue} previewRequest={props.previewRequest} timestamp={props.timestamp}/>
      </AppRouter>

      <AppRouter route={route} screen="gallery">
        <main className="screen-page">
          <h1>Rendered outputs</h1>
          {!completed.length && <p className="muted empty-note">Completed renders appear here.</p>}
          <div className="gallery-grid-host">
            <VirtualList items={completed} itemKey={job => job.job_id} estimateHeight={96} minColumnWidth={230} gap={16} aria-label="Rendered outputs"
              renderItem={job => (
                <button
                  className={`screen-card ${selectedOutput === job.job_id ? 'selected' : ''}`}
                  onClick={() => setSelectedOutput(job.job_id)}
                >
                  <strong>{job.output!.output_path.split(/[\\/]/).at(-1)}</strong>
                  <small>
                    {job.output!.frames} frames · {job.job_id.slice(0, 8)}
                  </small>
                </button>
              )} />
          </div>
          {output?.output && (
            <section className="output-inspector">
              <video controls preload="metadata" src={mediaURL(output.output.output_path)} />
              <h2>Metadata inspector</h2>
              <pre>{JSON.stringify({ output: output.output, request: output.request, job_id: output.job_id }, null, 2)}</pre>
              <a className="button" href={mediaURL(output.output.output_path)} download>
                Download output
              </a>
            </section>
          )}
        </main>
      </AppRouter>

      <AppRouter route={route} screen="history">
        <History
          projects={projects}
          jobs={jobs}
          sources={sources}
          caps={caps}
          openProject={openProject}
          refreshProjects={refreshProjects}
          refreshJobs={refreshJobs}
          busy={busy}
        />
      </AppRouter>

      <AppRouter route={route} screen="settings">
        <SettingsScreen
          caps={caps}
          config={config}
          saveSettings={saveSettings}
          busy={busy}
          connection={connection}
          themeName={props.themeName}
          onThemeChange={props.onThemeChange}
        />
      </AppRouter>
    </>
  );
}
