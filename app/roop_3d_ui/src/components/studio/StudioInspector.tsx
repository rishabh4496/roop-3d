import { lazy, Suspense, useEffect, useState, type ReactNode, type Ref } from 'react';
import { api } from '../../api';
import type { Capabilities, Coverage, FaceSet, IdentitySheet, Reconstruction, Track } from '../../contracts';
import { useFaceSelectionStore } from '../../useFaceSelectionStore';
import { selectEngine, useSelectedEngine } from '../../stores/engineSelection';
import type { WorkstationParameters } from '../workstation/pipeline';
import { ErrorRetry } from './FaceSetDetail';
import { formatClock, trackLabel } from './TargetPanel';
import EngineExplainer from '../guide/EngineExplainer';
import HardwareExplainer from '../guide/HardwareExplainer';

const PoseGeometryViewer = lazy(() => import('../../ThreeViewport').then(module => ({ default: module.PoseGeometryViewer })));
const LightProbeSphere = lazy(() => import('../../ThreeViewport').then(module => ({ default: module.LightProbeSphere })));

export const RENDER_MODES = [
  { id: 'standard', label: 'Standard identity swap', hint: 'The proven 2D identity swap path.' },
  { id: 'pose_3d', label: '3D pose (source geometry)', hint: "Projects the swap through the source identity's own 3D asset. Falls back to standard per frame when geometry is unreliable." },
  { id: 'reenactment_3d', label: '3D reenactment', hint: 'Source identity geometry driven by the target expression. Falls back to standard per frame when geometry is unreliable.' },
] as const;

type Engine = { engine_id: string; name: string; status: string; selectable: boolean; actual_inference_available: boolean; evidence: string; purpose?: string };

/** Engines from the Stage 6 model hub; selection is the same store the Model Hub screen writes. */
function useEngines() {
  const [state, setState] = useState<{ engines: Engine[] | null; error: string; attempt: number }>({ engines: null, error: '', attempt: 0 });
  useEffect(() => {
    const controller = new AbortController();
    api<{ engines: Engine[] }>('/models/hub', { signal: controller.signal })
      .then(hub => setState(value => (Array.isArray(hub?.engines) ? { ...value, engines: hub.engines, error: '' } : { ...value, engines: null, error: 'unexpected response from the model hub' })))
      .catch((error: Error) => { if (error.name !== 'AbortError') setState(value => ({ ...value, engines: null, error: error.message })); });
    return () => controller.abort();
  }, [state.attempt]);
  return { ...state, retry: () => setState(value => ({ ...value, error: '', attempt: value.attempt + 1 })) };
}

function Section({ title, badge, open = true, children }: { title: string; badge?: ReactNode; open?: boolean; children: ReactNode }) {
  return <details className="inspector-section" open={open}><summary><span>{title}</span>{badge}</summary><div className="inspector-body">{children}</div></details>;
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return <div className="fact"><dt>{label}</dt><dd>{children}</dd></div>;
}
const deg = (value: [number, number] | null | undefined) => (value ? `${value[0].toFixed(0)}° … ${value[1].toFixed(0)}°` : 'not estimated');

export interface StudioInspectorProps {
  caps: Capabilities | null;
  track: Track | null; sheet: IdentitySheet | null;
  sources: FaceSet[]; assignedId: string; onAssign: (trackId: string, facesetId: string | null) => void;
  parameters: WorkstationParameters; setParameters: (value: WorkstationParameters) => void;
  coverage: Coverage | null; mesh: Reconstruction | null;
  reconstructionPicker: ReactNode;
  /** VideoViewport portals its Geometry / Masking / Restorer parameter panels into this element. */
  panelsRef: Ref<HTMLDivElement>;
  processing: ReactNode;
}

export default function StudioInspector({ caps, track, sheet, sources, assignedId, onAssign, parameters, setParameters, coverage, mesh, reconstructionPicker, panelsRef, processing }: StudioInspectorProps) {
  const engine = useSelectedEngine();
  const engines = useEngines();
  const cluster = useFaceSelectionStore(state => (track ? state.targetClusters.find(item => item.id === track.track_id) ?? null : null));
  const sheetTrack = track ? sheet?.tracks.find(item => item.track_id === track.track_id) : undefined;
  const restorerName = parameters.restorer_model === 'codeformer' ? 'CodeFormerv0.1.onnx' : 'gpen_bfr_512.onnx';
  const restorerReady = !!caps?.models.some(model => model.name === restorerName && model.state === 'VERIFIED_LOCAL');
  const renderMode = parameters.render_mode ?? 'standard';
  const set = (patch: Partial<WorkstationParameters>) => setParameters({ ...parameters, ...patch });

  return (
    <div className="studio-inspector">
      <Section title="Target identity" badge={track ? <span className="status-chip">{trackLabel(track.track_id)}</span> : undefined}>
        {!track ? <p className="muted">Click a face in the video, a lane in the timeline, or a card in the Target panel.</p> : <>
          <div className="inspector-target">
            {track.thumbnail_base64 && <img src={`data:${track.thumbnail_mime};base64,${track.thumbnail_base64}`} alt={`${track.track_id} detected face`} />}
            <dl className="facts">
              <Fact label="Frames">{track.first_frame}–{track.last_frame}</Fact>
              <Fact label="Time">{formatClock(track.first_timestamp_seconds)} → {formatClock(track.last_timestamp_seconds)}</Fact>
              <Fact label="Landmarks">{track.landmark_count}-point</Fact>
            </dl>
          </div>
          <dl className="facts two">
            <Fact label="Tracking confidence">{sheetTrack ? `${sheetTrack.confidence_mean.toFixed(2)} mean · ${sheetTrack.confidence_min.toFixed(2)} min` : 'identity sheet unavailable'}</Fact>
            <Fact label="Visible / swappable">{sheetTrack ? `${sheetTrack.visible_frames} / ${sheetTrack.swappable_frames} frames` : track.sampled_detections != null ? `${track.sampled_detections} sampled detections` : '—'}</Fact>
            <Fact label="Yaw range">{deg(track.pose_range.yaw_degrees)}</Fact>
            <Fact label="Pitch range">{deg(track.pose_range.pitch_degrees)}</Fact>
            <Fact label="Roll range">{deg(track.pose_range.roll_degrees)}</Fact>
            <Fact label="Pose method">{track.pose_range.method.replaceAll('_', ' ')}</Fact>
          </dl>
          <label className="inspector-field">Source FaceSet
            <select aria-label="Inspector source FaceSet" value={assignedId} onChange={event => onAssign(track.track_id, event.target.value || null)} disabled={!sources.length}>
              <option value="">Keep original</option>{sources.map(source => <option key={source.faceset_id} value={source.faceset_id}>{source.name}</option>)}
            </select>
          </label>
          {cluster && <label className="inspector-field">Role
            <select aria-label="Target face role" value={cluster.mode} onChange={event => useFaceSelectionStore.getState().setClusterMode(cluster.id, event.target.value as 'swap' | 'ignore' | 'exclude')}>
              <option value="swap">Swap</option><option value="ignore">Reference guide (keep original)</option><option value="exclude">Exclude</option>
            </select></label>}
        </>}
      </Section>

      <Section title="Models & engine">
        {engines.error ? <ErrorRetry message={`Model hub unavailable: ${engines.error}`} onRetry={engines.retry} /> : !engines.engines ? <div className="skeleton-line" aria-label="Loading engines" /> : (
          <label className="inspector-field">Swap engine
            <select aria-label="Swap engine" value={engine} onChange={event => selectEngine(event.target.value)}>
              {engines.engines.some(item => item.engine_id === engine) ? null : <option value={engine}>{engine}</option>}
              {engines.engines.map(item => <option key={item.engine_id} value={item.engine_id} disabled={!item.selectable}>{item.name}{item.selectable ? '' : ` (${item.status.toLowerCase().replaceAll('_', ' ')})`}</option>)}
            </select>
            <small className="muted">{engines.engines.find(item => item.engine_id === engine)?.evidence ?? 'Selection is saved with each job.'}</small>
          </label>)}
        {engines.engines && <EngineExplainer engines={engines.engines} selectedId={engine} onSelect={selectEngine} />}
        <HardwareExplainer caps={caps} />
        <label className="inspector-field">Restoration model
          <select aria-label="Restoration model" value={parameters.restorer_model ?? 'gpen'} onChange={event => set({ restorer_model: event.target.value as 'gpen' | 'codeformer' })}>
            <option value="gpen">GPEN</option><option value="codeformer">CodeFormer</option>
          </select>
          <small className={restorerReady ? 'muted' : 'identity-warn'}>{restorerReady ? 'Weights verified locally.' : 'Weights not verified locally: restoration stays off.'}</small>
        </label>
      </Section>

      <Section title="3D settings" badge={renderMode !== 'standard' ? <span className="status-chip ok">3D</span> : undefined}>
        <label className="inspector-field">Render mode
          <select aria-label="3D render mode" value={renderMode} onChange={event => { const value = event.target.value as NonNullable<WorkstationParameters['render_mode']>; const { render_mode: _old, ...rest } = parameters; setParameters(value === 'standard' ? rest : { ...rest, render_mode: value }); }}>
            {RENDER_MODES.map(mode => <option key={mode.id} value={mode.id}>{mode.label}</option>)}
          </select>
          <small className="muted">{RENDER_MODES.find(mode => mode.id === renderMode)?.hint}</small>
        </label>
        <div className="inspector-viewer">
          <Suspense fallback={<div className="skeleton-tile wide" aria-label="Loading 3D viewer" />}><PoseGeometryViewer coverage={coverage} track={track} mesh={mesh} /></Suspense>
        </div>
        {reconstructionPicker}
      </Section>

      <div className="inspector-panels" ref={panelsRef} aria-label="Pose, expression, masking and restoration parameters" />

      <Section title="Colour & lighting">
        <label className="inspector-field">Colour transfer
          <select aria-label="Colour transfer" value={parameters.color_method ?? 'lab'} onChange={event => set({ color_method: event.target.value as 'lab' | 'histogram' })}>
            <option value="lab">LAB statistics</option><option value="histogram">Stabilized histograms</option>
          </select>
        </label>
        <label className="inspector-toggle"><input type="checkbox" checked={!!parameters.relight} onChange={event => set({ relight: event.target.checked })} />3D relighting (SH scene lighting)</label>
        {parameters.relight
          ? <Suspense fallback={<div className="skeleton-tile wide" aria-label="Loading light probe" />}><LightProbeSphere parameters={parameters} setParameters={setParameters} /></Suspense>
          : <p className="muted">Relighting is off: the swapped face keeps colour-matched lighting only. Enable it to edit the light probe.</p>}
      </Section>

      <Section title="Processing" open={false}>{processing}</Section>
    </div>
  );
}
