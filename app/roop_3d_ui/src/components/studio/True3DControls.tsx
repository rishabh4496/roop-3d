import type { WorkstationParameters } from '../workstation/pipeline';
import {
  AUTO_PRIOR_FOV_DEG, DEFAULT_CAMERA, DEFAULT_HEAD_TOPOLOGY, DEFAULT_SHADER_MODE, FOCAL_RANGE_PX, FOV_RANGE_DEG, HEAD_TOPOLOGY_ENGINES, NOT_WIRED_MESSAGE,
  SHADER_LABELS, SHADER_MODES, TOPOLOGY_LABELS, resolveIntrinsics, useTrue3DOptions, type CameraIntrinsics, type HeadTopologyEngine, type ShaderMode,
} from '../../true3d';
import { ErrorRetry } from './FaceSetDetail';

type Params = Partial<WorkstationParameters>;
export interface True3DControlsProps { params: Params; onParams: (patch: Params) => void; compact?: boolean }

const REFERENCE_FRAME = { width: 1920, height: 1080 } as const;
const number = (value: string): number | null => { const parsed = Number(value); return value.trim() !== '' && Number.isFinite(parsed) ? parsed : null; };

/** The camera a parameter set carries (auto when none is saved). */
export const cameraOf = (params: Params): CameraIntrinsics => params.camera ?? DEFAULT_CAMERA;

/**
 * Renderer readiness, as the server reports it. While the true-3D renderer is not connected this says so plainly
 * ("3D renderer not wired") and nothing downstream is presented as ready.
 */
export function RendererStatusNote({ wired, message, detail }: { wired: boolean; message?: string; detail?: string }) {
  if (wired) return <small className="muted" data-testid="renderer-status" role="status">3D renderer connected.</small>;
  return (
    <div className="identity-warn" data-testid="renderer-status" role="status">
      <b>{message ?? NOT_WIRED_MESSAGE}</b>
      <small>{detail ?? 'Previews, live studio and renders stay unavailable until the true-3D head renderer is connected.'}</small>
    </div>
  );
}

/**
 * 3D Head Topology Engine, 3D Photorealistic Shader Mode and perspective Camera Intrinsics. Every option's availability comes from
 * `GET /true3d/options`; an unavailable option is listed, disabled, with the server's reason. There is no orthographic or
 * weak-perspective camera: the projection is always a perspective pin-hole.
 */
export default function True3DControls({ params, onParams, compact = false }: True3DControlsProps) {
  const { options, error, retry } = useTrue3DOptions();
  const topology: HeadTopologyEngine = params.head_topology_engine ?? DEFAULT_HEAD_TOPOLOGY;
  const shader: ShaderMode = params.shader_mode ?? DEFAULT_SHADER_MODE;
  const camera = cameraOf(params);
  const setCamera = (patch: Partial<CameraIntrinsics>) => onParams({ camera: { ...camera, ...patch, projection: 'perspective' } });
  if (error) return <ErrorRetry message={`3D options unavailable: ${error}`} onRetry={retry} />;
  if (!options) return <div className="skeleton-line" aria-label="Loading 3D options" />;

  const topologyRow = (id: HeadTopologyEngine) => options.head_topology_engines.find(item => item.id === id);
  const shaderRow = (id: ShaderMode) => options.shader_modes.find(item => item.id === id);
  const picked = topologyRow(topology);
  const pickedShader = shaderRow(shader);
  const [fovLow, fovHigh] = options.camera?.fov_range_deg ?? FOV_RANGE_DEG;
  const [focalLow, focalHigh] = options.camera?.focal_range_px ?? FOCAL_RANGE_PX;
  const manual = camera.mode === 'manual';
  const resolved = resolveIntrinsics(camera, REFERENCE_FRAME.width, REFERENCE_FRAME.height);
  const point = camera.principal_point ?? [0.5, 0.5];

  return (
    <div className="true3d-controls" data-testid="true3d-controls">
      <RendererStatusNote wired={options.renderer.wired} message={options.renderer.message} detail={options.renderer.detail} />

      <label className="inspector-field">3D Head Topology Engine
        <select aria-label="3D head topology engine" data-testid="true3d-topology" value={topology} onChange={event => onParams({ head_topology_engine: event.target.value as HeadTopologyEngine })}>
          {HEAD_TOPOLOGY_ENGINES.map(id => {
            const row = topologyRow(id);
            return <option key={id} value={id} disabled={!row?.available}>{TOPOLOGY_LABELS[id]}{row?.available ? '' : ' — unavailable'}</option>;
          })}
        </select>
        <small className={picked?.available ? 'muted' : 'identity-warn'} data-testid="true3d-topology-reason">
          {picked?.available
            ? `${TOPOLOGY_LABELS[topology]} is installed${picked.license_id ? ` (${picked.license_id})` : ''}.`
            : picked?.reason ?? 'Availability is unknown.'}
        </small>
        {!compact && topologyRowsUnavailable(options.head_topology_engines).length > 0 && (
          <details className="backbone-unavailable"><summary>Why are some engines unavailable?</summary>
            <ul>{topologyRowsUnavailable(options.head_topology_engines).map(row => <li key={row.id}><b>{TOPOLOGY_LABELS[row.id]}</b>: {row.reason}</li>)}</ul>
          </details>)}
      </label>

      <label className="inspector-field">3D Photorealistic Shader Mode
        <select aria-label="3D photorealistic shader mode" data-testid="true3d-shader" value={shader} onChange={event => onParams({ shader_mode: event.target.value as ShaderMode })}>
          {SHADER_MODES.map(id => {
            const row = shaderRow(id);
            return <option key={id} value={id} disabled={!row?.available}>{SHADER_LABELS[id]}{row?.available ? '' : ' — unavailable'}</option>;
          })}
        </select>
        <small className={pickedShader?.available ? 'muted' : 'identity-warn'} data-testid="true3d-shader-reason">
          {pickedShader?.available ? `${SHADER_LABELS[shader]} is ready.` : pickedShader?.reason ?? NOT_WIRED_MESSAGE}
        </small>
      </label>

      <fieldset className="inspector-field true3d-camera" data-testid="true3d-camera">
        <legend>Camera intrinsics <small className="muted">perspective only</small></legend>
        <div className="mode-cards" role="radiogroup" aria-label="Camera focal length mode">
          {(['auto', 'manual'] as const).map(mode => (
            <button key={mode} type="button" role="radio" aria-checked={camera.mode === mode} className={`mode-card ${camera.mode === mode ? 'active' : ''}`}
              onClick={() => setCamera(mode === 'auto' ? { mode } : { mode, focal_unit: camera.focal_unit, fov_deg: camera.fov_deg ?? AUTO_PRIOR_FOV_DEG })}>
              <b>{mode === 'auto' ? 'Auto (dynamic solver)' : 'Manual'}</b>
            </button>
          ))}
        </div>
        {!manual && <small className="muted">The renderer solves the focal length per shot from the head's perspective foreshortening, starting from a {AUTO_PRIOR_FOV_DEG}° field of view.</small>}
        {manual && <>
          <label className="inspector-field">Focal length unit
            <select aria-label="Focal length unit" value={camera.focal_unit} onChange={event => {
              const unit = event.target.value === 'px' ? 'px' : 'fov_deg';
              setCamera(unit === 'px' ? { focal_unit: unit, focal_length_px: camera.focal_length_px ?? Math.round(resolved.fx) } : { focal_unit: unit, fov_deg: camera.fov_deg ?? Number(resolved.horizontalFovDeg.toFixed(1)) });
            }}>
              <option value="fov_deg">Horizontal field of view (degrees)</option><option value="px">Focal length (pixels)</option>
            </select></label>
          {camera.focal_unit === 'fov_deg'
            ? <label className="inspector-field">Field of view <output>{(camera.fov_deg ?? AUTO_PRIOR_FOV_DEG).toFixed(1)}°</output>
              <input type="range" aria-label="Field of view" min={fovLow} max={fovHigh} step={0.5} value={camera.fov_deg ?? AUTO_PRIOR_FOV_DEG}
                onChange={event => setCamera({ fov_deg: Number(event.target.value) })} /></label>
            : <label className="inspector-field">Focal length <output>{Math.round(camera.focal_length_px ?? resolved.fx)} px</output>
              <input type="number" aria-label="Focal length in pixels" min={focalLow} max={focalHigh} step={1} value={camera.focal_length_px ?? Math.round(resolved.fx)}
                onChange={event => { const value = number(event.target.value); if (value !== null) setCamera({ focal_length_px: Math.min(focalHigh, Math.max(focalLow, value)) }); }} /></label>}
          <div className="inspector-field" role="group" aria-label="Principal point">Principal point <small className="muted">fraction of the frame (0.5 / 0.5 = centre)</small>
            <span className="true3d-pp">
              <input type="number" aria-label="Principal point x" min={0} max={1} step={0.01} value={point[0]} onChange={event => { const value = number(event.target.value); if (value !== null && value >= 0 && value <= 1) setCamera({ principal_point: [value, point[1]] }); }} />
              <input type="number" aria-label="Principal point y" min={0} max={1} step={0.01} value={point[1]} onChange={event => { const value = number(event.target.value); if (value !== null && value >= 0 && value <= 1) setCamera({ principal_point: [point[0], value] }); }} />
            </span></div>
        </>}
        <small className="muted" data-testid="true3d-camera-k">
          At {REFERENCE_FRAME.width}×{REFERENCE_FRAME.height}: fx = fy = {resolved.fx.toFixed(0)} px, c = ({resolved.cx.toFixed(0)}, {resolved.cy.toFixed(0)}), {resolved.horizontalFovDeg.toFixed(1)}° horizontal{resolved.estimated ? ' (starting prior)' : ''}.
        </small>
      </fieldset>
    </div>
  );
}

const topologyRowsUnavailable = (rows: readonly { id: HeadTopologyEngine; available: boolean; reason: string | null }[]) => rows.filter(row => !row.available);
