import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../../api';
import type { Config, Reconstruction } from '../../contracts';
import type { WorkstationParameters } from '../workstation/pipeline';
import { loadIdentityMesh, type MeshSource } from '../../utils/glbMesh';

/**
 * 3D topology preview and track stabilization controls. The preview is an inspection asset;
 * the render service remains unavailable until its full-head renderer is wired.
 */

// ---------------------------------------------------------------------------------------------------- smoothing <-> One-Euro cutoff
/** Light smoothing = a high cutoff, heavy = a low one; log scale because the filter responds to the cutoff's ratio. */
export const SMOOTH_CUTOFF_LIGHT = 1.5;
export const SMOOTH_CUTOFF_HEAVY = 0.03;
export const DEFAULT_MIN_CUTOFF = 0.15;
export const smoothingToCutoff = (smoothing: number): number => {
  const t = Math.min(1, Math.max(0, smoothing));
  return Math.exp(Math.log(SMOOTH_CUTOFF_LIGHT) + (Math.log(SMOOTH_CUTOFF_HEAVY) - Math.log(SMOOTH_CUTOFF_LIGHT)) * t);
};
export const cutoffToSmoothing = (cutoff: number): number => {
  const value = Number.isFinite(cutoff) && cutoff > 0 ? cutoff : DEFAULT_MIN_CUTOFF;
  const t = (Math.log(value) - Math.log(SMOOTH_CUTOFF_LIGHT)) / (Math.log(SMOOTH_CUTOFF_HEAVY) - Math.log(SMOOTH_CUTOFF_LIGHT));
  return Math.min(1, Math.max(0, t));
};
/** Pause after the last slider movement before the setting is saved (saving invalidates the preview caches, so it must not run per tick). */
export const SMOOTHING_COMMIT_MS = 500;

// ---------------------------------------------------------------------------------------------------- topology + semantic sliders
export interface TopologyRow { id: string; title: string; license_id: string; commercial_use: boolean; attribution?: string; default: boolean; available: boolean; state: string; detail: string | null }
export interface SemanticSliderDef { key: string; label: string; description: string; unit_mm: number; minimum: number; maximum: number; default: number }
export interface TopologyReport { default: string; topologies: TopologyRow[]; semantic_sliders: SemanticSliderDef[] }
export const isTopologyReport = (value: unknown): value is TopologyReport => {
  const item = value as TopologyReport | null;
  return !!item && typeof item === 'object' && Array.isArray(item.topologies) && Array.isArray(item.semantic_sliders) && item.topologies.every(row => typeof row?.id === 'string');
};
/**
 * The six craniofacial sliders the engine defines (`reconstruction/semantic_shape.py`), in display order, with the label the panel shows.
 * Forehead height and temple width are NOT among them: the 68-point landmark layout has nothing above the brows or at the temples,
 * and head width is deliberately held fixed (it is one of the nuisance measures the other sliders must not move).
 */
export const SEMANTIC_PANEL: readonly { key: string; label: string }[] = [
  { key: 'cheekbone_prominence', label: 'Cheekbone Arch' }, { key: 'jaw_width', label: 'Jawline Width' },
  { key: 'chin_projection', label: 'Chin Projection' }, { key: 'nose_bridge', label: 'Nose Bridge' },
  { key: 'brow_arch', label: 'Brow Arch' }, { key: 'face_fullness', label: 'Face Fullness' },
];
const TOPOLOGY_LABEL: Record<string, string> = { makehuman_cc0: 'MakeHuman CC0', flame2023_open: 'FLAME 2023 Open' };

export function meshQuery(topology: string, values: Record<string, number>): string {
  const params = new URLSearchParams({ template: 'makehuman', topology });
  for (const { key } of SEMANTIC_PANEL) if (values[key]) params.set(key, String(values[key]));
  return params.toString();
}

function TopologySemantic({ facesetId, onMesh }: {
  facesetId: string; onMesh: (mesh: Reconstruction | null) => void;
}) {
  const [report, setReport] = useState<{ value: TopologyReport | null; error: string; attempt: number }>({ value: null, error: '', attempt: 0 });
  const [topology, setTopology] = useState('');
  const [values, setValues] = useState<Record<string, number>>({});
  const [touched, setTouched] = useState(false);
  const [mesh, setMesh] = useState<{ state: 'idle' | 'loading' | 'ready' | 'error'; message: string; provenance: string; source?: MeshSource; fallbackReason?: string | null; bytes?: number | null }>({ state: 'idle', message: '', provenance: '' });
  useEffect(() => {
    const controller = new AbortController();
    api<unknown>('/morphable-topologies', { signal: controller.signal })
      .then(value => setReport(previous => (isTopologyReport(value) ? { ...previous, value, error: '' } : { ...previous, value: null, error: 'The local service returned an unexpected response.' })))
      .catch((error: Error) => { if (error.name !== 'AbortError') setReport(previous => ({ ...previous, value: null, error: error.message })); });
    return () => controller.abort();
  }, [report.attempt]);
  const active = topology || report.value?.default || 'makehuman_cc0';
  const defs = useMemo(() => new Map((report.value?.semantic_sliders ?? []).map(item => [item.key, item])), [report.value]);

  useEffect(() => { setTouched(false); setValues({}); setTopology(''); setMesh({ state: 'idle', message: '', provenance: '' }); }, [facesetId]);
  useEffect(() => {
    if (!touched || !facesetId) return undefined;
    const controller = new AbortController();
    let live = true;
    const timer = setTimeout(() => {
      setMesh(previous => ({ ...previous, state: 'loading', message: '' }));
      // Binary glTF first (smaller, one request, normal map included); the JSON route is the fallback.
      loadIdentityMesh(facesetId, meshQuery(active, values), { signal: controller.signal })
        .then(loaded => { if (live) { onMesh(loaded.mesh); setMesh({ state: 'ready', message: '', provenance: loaded.mesh.provenance ?? '', source: loaded.source, fallbackReason: loaded.fallbackReason, bytes: loaded.bytes }); } })
        .catch((error: Error) => { if (live && error.name !== 'AbortError') setMesh({ state: 'error', message: error.message, provenance: '' }); });
    }, 300);
    return () => { live = false; clearTimeout(timer); controller.abort(); };
  }, [touched, facesetId, active, values, onMesh]);

  const reset = () => { setValues({}); setTopology(''); setTouched(false); setMesh({ state: 'idle', message: '', provenance: '' }); onMesh(null); };

  return (
    <div className="topology-semantic" data-testid="topology-semantic">
      {report.error && <div className="error-retry" role="status"><span>{report.error}</span><button type="button" className="button small" onClick={() => setReport(previous => ({ ...previous, error: '', attempt: previous.attempt + 1 }))}>Retry</button></div>}
      <div className="influence-mode">
        <span id="topology-engine-label" className="influence-mode-title">Topology engine</span>
        <div className="mode-cards" role="radiogroup" aria-labelledby="topology-engine-label">
          {(report.value?.topologies ?? []).map(row => {
            const usable = row.available && row.commercial_use;
            return (
              <button key={row.id} type="button" role="radio" aria-checked={active === row.id} disabled={!usable} className={`mode-card ${active === row.id ? 'active' : ''}`}
                title={usable ? `${row.title} · ${row.license_id}` : row.detail ?? (row.commercial_use ? 'Not available' : 'Licence does not allow commercial use')}
                onClick={() => { setTopology(row.id); setTouched(true); }}>
                <b>{TOPOLOGY_LABEL[row.id] ?? row.title}</b><small className="muted"> {row.license_id}{usable ? '' : ` · ${row.detail ?? 'unavailable'}`}</small>
              </button>);
          })}
        </div>
      </div>
      {!facesetId && <small className="muted" role="status">Choose a source FaceSet to preview topology and semantic shape changes.</small>}
      {SEMANTIC_PANEL.map(({ key, label }) => {
        const def = defs.get(key);
        const value = values[key] ?? 0;
        return (
          <label key={key} className="inspector-field">{label}
            <output aria-live="off">{def ? `${value >= 0 ? '+' : ''}${(value * def.unit_mm).toFixed(1)} mm` : value.toFixed(2)}</output>
            <input type="range" aria-label={`Semantic ${label}`} min={def?.minimum ?? -3} max={def?.maximum ?? 3} step={0.1} value={value} disabled={!facesetId || !def}
              title={def?.description} onChange={event => { setValues(previous => ({ ...previous, [key]: Number(event.target.value) })); setTouched(true); }} />
          </label>);
      })}
      <div className="morph-actions">
        <button type="button" className="button small" disabled={!touched && !Object.keys(values).length} onClick={reset}>Reset shape</button>
        {mesh.state === 'loading' && <span className="build-progress" role="status">Rebuilding the mesh…</span>}
      </div>
      {mesh.state === 'error' && <div className="error-retry" role="alert"><span>{mesh.message}</span></div>}
      {mesh.state === 'ready' && mesh.provenance && <small className="muted" data-testid="topology-provenance">{mesh.provenance}</small>}
      {mesh.state === 'ready' && mesh.source && (
        <small className="muted" data-testid="mesh-source" title={mesh.fallbackReason ?? undefined}>
          {mesh.source === 'glb' ? `Loaded as binary glTF${mesh.bytes ? ` (${(mesh.bytes / 1024).toFixed(0)} KB)` : ''}` : `Loaded as JSON${mesh.fallbackReason ? ` (GLB unavailable: ${mesh.fallbackReason})` : ''}`}
        </small>)}
      <small className="muted" data-testid="semantic-limits">Forehead height and temple width are not offered: the engine has no landmarks above the brows or at the temples, and head width is held fixed.</small>
      <small className="muted">Previews the MakeHuman full-head asset for inspection. Rendering remains unavailable until the 3D renderer is wired.</small>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------- the panel set

export interface WorkstationControls3DProps {
  parameters: WorkstationParameters; setParameters: (value: WorkstationParameters) => void;
  facesetId: string;
  onMesh: (mesh: Reconstruction | null) => void;
  config?: Config | null; onSaveConfig?: (value: Config) => void;
}

export default function WorkstationControls3D({ parameters, setParameters, facesetId, onMesh, config, onSaveConfig }: WorkstationControls3DProps) {
  const [topologyOpen, setTopologyOpen] = useState(false);
  const colour = parameters.color_strength;
  const colourBypassed = parameters.bypass.includes('color');

  const cutoff = typeof config?.temporal_min_cutoff === 'number' ? config.temporal_min_cutoff : DEFAULT_MIN_CUTOFF;
  const [smoothing, setSmoothing] = useState(() => cutoffToSmoothing(cutoff));
  const lastSaved = useRef(cutoff);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => { lastSaved.current = cutoff; setSmoothing(cutoffToSmoothing(cutoff)); }, [cutoff]);
  useEffect(() => () => clearTimeout(timer.current), []);
  const changeSmoothing = (value: number) => {
    setSmoothing(value);
    clearTimeout(timer.current);
    if (!config || !onSaveConfig) return;
    timer.current = setTimeout(() => {
      const next = Number(smoothingToCutoff(value).toPrecision(3));
      if (Math.abs(next - lastSaved.current) / lastSaved.current < 0.01) return;
      lastSaved.current = next;
      onSaveConfig({ ...config, temporal_min_cutoff: next });
    }, SMOOTHING_COMMIT_MS);
  };

  return (
    <div className="workstation-controls-3d" data-testid="workstation-controls-3d">
      <details className="inspector-subsection" data-testid="topology-section" onToggle={event => setTopologyOpen((event.currentTarget as HTMLDetailsElement).open)}>
        <summary>Topology &amp; semantic shape<small>MakeHuman CC0 · FLAME 2023 Open · bone sliders</small></summary>
        {topologyOpen && <TopologySemantic facesetId={facesetId} onMesh={onMesh} />}
      </details>

      <details className="inspector-subsection" data-testid="tone-section">
        <summary>Skin tone vs. ambient light<small>colour-transfer weight</small></summary>
        <label className="inspector-field">Skin-tone preservation ↔ ambient adaptation <output aria-live="off">{Math.round(colour * 100)}% ambient</output>
          <input type="range" aria-label="Skin tone preservation versus ambient adaptation" min={0} max={100} step={1} value={Math.round(colour * 100)}
            aria-valuetext={`${Math.round(colour * 100)} percent adapted to the scene, ${100 - Math.round(colour * 100)} percent source skin tone kept`}
            onChange={event => setParameters({ ...parameters, color_strength: Number(event.target.value) / 100 })} />
          <span className="protection-scale" aria-hidden="true"><small>0 keep source tone</small><small>100 match scene</small></span>
        </label>
        {colourBypassed && <small className="identity-warn">Colour matching is bypassed in the render settings, so this slider has no effect until it is switched on again.</small>}
        <small className="muted">This is the swap's Lab colour-transfer weight, an original calibration, not the TRUST network (not installed).</small>
      </details>


      <details className="inspector-subsection" data-testid="adaptation-section">
        <summary>Scene adaptation &amp; smoothing<small>3 s adaptation · temporal filter</small></summary>
        <label className="inspector-toggle unsupported-control" data-testid="scene-adapt-unsupported" title="The tracker has no adaptation pass">
          <input type="checkbox" checked={false} disabled readOnly />3-second scene adaptation <span className="status-chip">not supported</span></label>
        <small className="muted">The tracker has no adaptation pass (SPARK is not installed). The temporal smoothing below is the real control.</small>
        <label className="inspector-field">Temporal smoothing <output aria-live="off">{smoothing < 0.34 ? 'light' : smoothing < 0.67 ? 'medium' : 'heavy'} · cutoff {smoothingToCutoff(smoothing).toFixed(2)}</output>
          <input type="range" aria-label="Temporal Kalman smoothing" min={0} max={100} step={1} value={Math.round(smoothing * 100)} disabled={!config || !onSaveConfig}
            onChange={event => changeSmoothing(Number(event.target.value) / 100)} />
          <span className="protection-scale" aria-hidden="true"><small>0 responsive</small><small>100 smooth</small></span>
        </label>
        <small className="muted">Sets the global track-stabiliser cutoff (<code>temporal_min_cutoff</code>, saved {SMOOTHING_COMMIT_MS} ms after you stop). Lower cutoff = smoother, more lag; the Kalman noise terms stay at the engine defaults.</small>
      </details>
    </div>
  );
}
