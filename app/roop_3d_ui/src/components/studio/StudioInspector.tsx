import { lazy, Suspense, useEffect, useMemo, useState, type ReactNode, type Ref } from 'react';
import { api } from '../../api';
import type { Capabilities, Config, Coverage, FaceSet, IdentitySheet, Reconstruction, ReconstructionStatus, Track } from '../../contracts';
import EngineModelRack from '../modelhub/EngineModelRack';
import { PREVIEW_TAB_LABEL, RENDER_MODES, renderModeInfo } from '../workstation/renderModes';
import { useFaceSelectionStore } from '../../useFaceSelectionStore';
import { useModelCatalog } from '../../stores/modelCatalog';
import { selectEngine, useSelectedEngine } from '../../stores/engineSelection';
import { restorerAvailable, type RestorerModel, type WorkstationParameters } from '../workstation/pipeline';
import { ErrorRetry } from './FaceSetDetail';
import { formatClock, trackLabel } from './TargetPanel';
import EngineExplainer from '../guide/EngineExplainer';
import HardwareExplainer from '../guide/HardwareExplainer';
import CoverageRadarPanel from '../three/CoverageRadarPanel';
import IdentityMorphPanel from '../three/IdentityMorphPanel';
import { viewerLighting, viewerMorph } from '../three/viewerRigs';
import FaceExpressionEditor from './FaceExpressionEditor';
import HairNeckProtection from './HairNeckProtection';
import HeadModelBackbone from './HeadModelBackbone';
import SwapScopeSelector from './SwapScopeSelector';
import WorkstationControls3D from './WorkstationControls3D';
import { faceEditOf, jawDegToMouthOpen, mouthOpenToJawDeg, shLightOf, withFaceEdit, withManualLight, type ViewportTools } from '../three/viewportTools';

const PoseGeometryViewer = lazy(() => import('../../ThreeViewport').then(module => ({ default: module.PoseGeometryViewer })));
const LightProbeSphere = lazy(() => import('../../ThreeViewport').then(module => ({ default: module.LightProbeSphere })));

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
  /** Global engine settings (execution provider / precision) edited from the Engine & Model Rack. */
  config?: Config | null; onSaveConfig?: (value: Config) => void;
  track: Track | null; sheet: IdentitySheet | null;
  sources: FaceSet[]; assignedId: string; onAssign: (trackId: string, facesetId: string | null) => void;
  parameters: WorkstationParameters; setParameters: (value: WorkstationParameters) => void;
  coverage: Coverage | null; mesh: Reconstruction | null;
  /** Replaces the mesh shown in the 3D viewport (the topology / semantic-shape preview); null returns to the FaceSet's own. */
  onIdentityMesh?: (mesh: Reconstruction | null) => void;
  /** State of the FaceSet 3D reconstruction request (idle until a source is chosen). */
  reconstruction?: ReconstructionStatus | null;
  reconstructionPicker: ReactNode;
  /** "+ Create .fsz from Video/Photos" control rendered inside the 3D viewer panel. */
  facesetCreator?: ReactNode;
  /** VideoViewport portals its Geometry / Masking / Restorer parameter panels into this element (Advanced Diagnostics). */
  panelsRef: Ref<HTMLDivElement>;
  processing: ReactNode;
  /** Primary action footer (Render Current Frame / Start Full Render / progress / telemetry). */
  footer?: ReactNode;
}

export const ENHANCERS: readonly { id: 'none' | RestorerModel; label: string; hint: string }[] = [
  { id: 'none', label: 'None', hint: 'No face restoration: the raw swap is composited.' },
  { id: 'gpen', label: 'GPEN Realistic 512', hint: 'Natural skin and eyes; the fast default restorer.' },
  { id: 'ultramax', label: 'UltraMax High-Detail', hint: 'CodeFormer detail fused into the swap through a Laplacian pyramid, so colour and low frequencies stay yours.' },
  { id: 'codeformer', label: 'CodeFormer', hint: 'Strongest repair of blurry or damaged faces; uses the fidelity setting under Advanced.' },
];

/** Active enhancer: restoration only runs while its blend is above zero. */
export const activeEnhancer = (parameters: WorkstationParameters): 'none' | RestorerModel =>
  parameters.restoration_blend > 0 ? (parameters.restorer_model ?? 'gpen') : 'none';

export default function StudioInspector({ caps, config, onSaveConfig, track, sheet, sources, assignedId, onAssign, parameters, setParameters, coverage, mesh, onIdentityMesh, reconstruction, reconstructionPicker, facesetCreator, panelsRef, processing, footer }: StudioInspectorProps) {
  const engine = useSelectedEngine();
  const engines = useEngines();
  const cluster = useFaceSelectionStore(state => (track ? state.targetClusters.find(item => item.id === track.track_id) ?? null : null));
  const sheetTrack = track ? sheet?.tracks.find(item => item.track_id === track.track_id) : undefined;
  const enhancer = activeEnhancer(parameters);
  const catalog = useModelCatalog();
  const restorerReady = restorerAvailable(catalog.loaded ? catalog.models : caps?.models, parameters.restorer_model);
  const renderMode = parameters.render_mode ?? 'standard';
  const robust = parameters.occlusion_mode === 'robust';
  const set = (patch: Partial<WorkstationParameters>) => setParameters({ ...parameters, ...patch });
  const pickMode = (value: NonNullable<WorkstationParameters['render_mode']>) => {
    const { render_mode: _old, ...rest } = parameters;
    setParameters(value === 'standard' ? rest : { ...rest, render_mode: value });
  };
  const pickEnhancer = (value: 'none' | RestorerModel) => {
    if (value === 'none') { set({ restoration_blend: 0 }); return; }
    set({ restorer_model: value, restoration_blend: parameters.restoration_blend > 0 ? parameters.restoration_blend : 0.7 });
  };
  const info = renderModeInfo(renderMode);
  const [morphPreview, setMorphPreview] = useState(false);
  const lighting = useMemo(() => viewerLighting(parameters), [parameters]);
  const morph = useMemo(() => viewerMorph(parameters, morphPreview), [parameters, morphPreview]);
  // Gizmo bindings: jaw opening and gaze edit the selected face's real expression overrides, the light handle the manual key light.
  const trackId = track?.track_id ?? null;
  const faceEdit = faceEditOf(parameters, trackId);
  const keyLight = shLightOf(parameters);
  const viewportTools = useMemo<ViewportTools>(() => ({
    trackId,
    jawDeg: mouthOpenToJawDeg(faceEdit.mouth_open ?? 0),
    onJaw: deg => { if (trackId) setParameters(withFaceEdit(parameters, trackId, { mouth_open: Number(jawDegToMouthOpen(deg).toFixed(2)) })); },
    gaze: { x: faceEdit.gaze_x ?? 0, y: faceEdit.gaze_y ?? 0 },
    onGaze: (x, y) => { if (trackId) setParameters(withFaceEdit(parameters, trackId, { gaze_x: Number(x.toFixed(2)), gaze_y: Number(y.toFixed(2)) })); },
    light: { yaw: keyLight.yaw, pitch: keyLight.pitch, strength: keyLight.strength, ambient: keyLight.ambient, manual: parameters.lighting_mode === 'manual' },
    onLight: (yaw, pitch) => setParameters(withManualLight(parameters, yaw, pitch)),
    relightOn: !!parameters.relight,
    onRelight: on => setParameters({ ...parameters, relight: on }),
  }), [trackId, faceEdit.mouth_open, faceEdit.gaze_x, faceEdit.gaze_y, keyLight.yaw, keyLight.pitch, keyLight.strength, keyLight.ambient, parameters, setParameters]);

  return (
    <div className="studio-inspector workflow-inspector">
      <div className="workflow-scroll">
        <section className="workflow-step" aria-labelledby="step-swap" data-testid="step-swap">
          <h3 id="step-swap"><span className="step-no">1</span>Swap &amp; 3D mode{renderMode !== 'standard' && <span className="status-chip ok">3D</span>}</h3>
          <div className="mode-cards" role="radiogroup" aria-label="3D render mode">
            {RENDER_MODES.map(mode => (
              <button key={mode.id} type="button" role="radio" aria-checked={renderMode === mode.id} className={`mode-card ${renderMode === mode.id ? 'active' : ''}`}
                onClick={() => pickMode(mode.id)}><b>{mode.label}</b></button>
            ))}
          </div>
          <small className="muted" data-testid="render-mode-hint">{info.hint}</small>
          <small className="render-mode-preview">Viewer tab for this mode: <b>{PREVIEW_TAB_LABEL[info.preview]}</b></small>
          <SwapScopeSelector parameters={parameters} setParameters={setParameters} />
          <div className="inspector-viewer">
            <Suspense fallback={<div className="skeleton-tile wide" aria-label="Loading 3D viewer" />}><PoseGeometryViewer coverage={coverage} track={track} mesh={mesh} renderMode={renderMode} reconstruction={reconstruction} creator={facesetCreator} lighting={lighting} morph={morph} tools={viewportTools} /></Suspense>
          </div>
          {reconstructionPicker}
          <FaceExpressionEditor parameters={parameters} setParameters={setParameters} trackId={track?.track_id ?? null} />
          <details className="inspector-subsection" data-testid="coverage-section" open>
            <summary>Angular coverage<small>{coverage ? 'which head angles this FaceSet has' : 'load a FaceSet'}</small></summary>
            <CoverageRadarPanel coverage={coverage} track={track} compact />
          </details>
          <details className="inspector-subsection" data-testid="morph-section">
            <summary>3D head model &amp; influence<small>backbone · influence mode · jaw reshape</small></summary>
            <HeadModelBackbone />
            <IdentityMorphPanel parameters={parameters} setParameters={setParameters} preview={morphPreview} onPreview={setMorphPreview} />
          </details>
          <WorkstationControls3D parameters={parameters} setParameters={setParameters} trackId={trackId} facesetId={assignedId || coverage?.faceset_id || ''}
            onMesh={value => onIdentityMesh?.(value)} config={config} onSaveConfig={onSaveConfig} />
        </section>

        <section className="workflow-step" aria-labelledby="step-mask" data-testid="step-mask">
          <h3 id="step-mask"><span className="step-no">2</span>Masking &amp; occlusion</h3>
          <label className="inspector-toggle"><input type="checkbox" checked disabled readOnly />Face parser (BiSeNet) <small className="muted">always on: the base face matte</small></label>
          <label className="inspector-toggle"><input type="checkbox" aria-label="XSeg mask and smart occluder" checked={robust} onChange={event => set({ occlusion_mode: event.target.checked ? 'robust' : 'standard' })} />
            XSeg mask + Smart Occluder <small className="muted">keeps hands, glasses, hair and microphones in front of the face</small></label>
          <label className="inspector-field">Mask feather / blur <output>{(parameters.feather_px ?? 0).toFixed(0)} px</output>
            <input type="range" aria-label="Mask feather" min={0} max={32} step={1} value={parameters.feather_px ?? 0} onChange={event => set({ feather_px: Number(event.target.value) })} /></label>
          <HairNeckProtection parameters={parameters} setParameters={setParameters} enhancer={enhancer} />
        </section>

        <section className="workflow-step" aria-labelledby="step-enhance" data-testid="step-enhance">
          <h3 id="step-enhance"><span className="step-no">3</span>Face restoration</h3>
          <label className="inspector-field">Enhancer
            <select aria-label="Enhancer" value={enhancer} onChange={event => pickEnhancer(event.target.value as 'none' | RestorerModel)}>
              {ENHANCERS.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
            </select>
            <small className={enhancer === 'none' || restorerReady ? 'muted' : 'identity-warn'}>
              {enhancer !== 'none' && !restorerReady ? 'No usable weight file in app/models: restoration stays off.' : ENHANCERS.find(item => item.id === enhancer)?.hint}
            </small>
          </label>
          <label className="inspector-field">Blend amount <output>{Math.round(parameters.restoration_blend * 100)}%</output>
            <input type="range" aria-label="Enhancer blend amount" min={0} max={100} step={1} value={Math.round(parameters.restoration_blend * 100)}
              onChange={event => set({ restoration_blend: Number(event.target.value) / 100 })} /></label>
        </section>

        <section className="workflow-step" aria-labelledby="step-rack" data-testid="step-rack">
          <h3 id="step-rack"><span className="step-no">4</span>Engine &amp; Model Rack</h3>
          <EngineModelRack params={parameters} onParams={set} config={config ?? null} caps={caps} onSaveConfig={onSaveConfig}
            people={(sheet?.tracks ?? []).map(item => ({ key: item.track_id, label: trackLabel(item.track_id) }))} />
        </section>
      </div>

      <details className="advanced-drawer" data-testid="advanced-diagnostics">
        <summary><span>Advanced Diagnostics</span><small className="muted">engine, colour, lighting, tracks, experimental</small></summary>
        <div className="advanced-body">
          <Section title="Target identity" open={false} badge={track ? <span className="status-chip">{trackLabel(track.track_id)}</span> : undefined}>
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
              <select aria-label="Restoration model" value={parameters.restorer_model ?? 'gpen'} onChange={event => set({ restorer_model: event.target.value as RestorerModel })}>
                <option value="gpen">GPEN</option><option value="ultramax">UltraMax</option><option value="codeformer">CodeFormer</option>
              </select>
              <small className={restorerReady ? 'muted' : 'identity-warn'}>{restorerReady ? 'Weights verified locally.' : 'No usable weight file in app/models: restoration stays off.'}</small>
            </label>
          </Section>

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

          <Section title="Target sensor finishing">
            <p className="muted">Match shutter blur, lens softness and sensor grain around an enhanced face.</p>
            <label className="inspector-field">Camera shutter angle <output>{parameters.sensor_shutter_angle ?? 180}°</output>
              <input type="range" aria-label="Camera shutter angle" min={0} max={360} step={1}
                value={parameters.sensor_shutter_angle ?? 180}
                onChange={event => set({ sensor_shutter_angle: Number(event.target.value) })} /></label>
            <label className="inspector-field">Grain match strength <output>{parameters.grain_match_strength ?? 100}%</output>
              <input type="range" aria-label="Grain match strength" min={0} max={100} step={1}
                value={parameters.grain_match_strength ?? 100}
                onChange={event => set({ grain_match_strength: Number(event.target.value) })} /></label>
          </Section>

          <div className="inspector-panels" ref={panelsRef} aria-label="Pose, expression, masking and restoration parameters" />

          <Section title="Processing" open={false}>{processing}</Section>
        </div>
      </details>
      {footer}
    </div>
  );
}
