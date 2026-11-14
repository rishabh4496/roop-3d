import { useState } from 'react';
import { post } from '../../api';
import type { Capabilities, Config } from '../../contracts';
import type { WorkstationParameters } from '../workstation/pipeline';
import True3DControls from '../studio/True3DControls';
import { modelStatus, useModelCatalog, type ModelStatus } from '../../stores/modelCatalog';
import { setRack, useRack } from '../../stores/rackSelection';
import ModelSelect, { StatusPill, type RackOption } from './ModelSelect';
import ActiveModelBadge from './ActiveModelBadge';
import CompositingControls from '../studio/CompositingControls';
import { occlusionModePatch } from '../studio/compositing';
import ComponentDiagnostics from './ComponentDiagnostics';

type Params = Partial<WorkstationParameters>;
export type RackProps = {
  /** Current values of the parameters the rack edits (the workspace parameters, or the Batch defaults). */
  params: Params; onParams: (patch: Params) => void;
  config: Config | null; caps: Capabilities | null;
  /** Persists global engine settings (execution provider / precision). Absent = those two controls are read-only. */
  onSaveConfig?: (value: Config) => void;
  variant?: 'workspace' | 'batch';
};

/** `live` = the value reaches the render engine. Anything else is shown but cannot be changed: it has no engine stage or setting to drive. */
const Wiring = ({ live }: { live: boolean }) => (
  <span className={`rack-wiring ${live ? 'live' : 'unsupported'}`} title={live ? 'Changes what the render engine runs.' : 'The render engine has no stage or setting for this, so it is listed but cannot be changed.'}>
    {live ? 'LIVE' : 'NOT SUPPORTED'}
  </span>
);

const DETECTORS: RackOption[] = [{ id: 'scrfd' }, { id: 'retinaface' }, { id: 'yoloface' }];

const RUNTIMES = [
  { id: 'auto', label: 'Auto (CUDA → DirectML → CPU)', provider: 'auto', precision: 'auto' },
  { id: 'trt_fp16', label: 'TensorRT (FP16 mixed precision)', provider: 'TensorrtExecutionProvider', precision: 'fp16' },
  { id: 'trt_fp32', label: 'TensorRT (FP32)', provider: 'TensorrtExecutionProvider', precision: 'fp32' },
  { id: 'cuda_fp16', label: 'CUDA (FP16)', provider: 'CUDAExecutionProvider', precision: 'fp16' },
  { id: 'cuda_fp32', label: 'CUDA (FP32)', provider: 'CUDAExecutionProvider', precision: 'fp32' },
  { id: 'dml', label: 'DirectML (FP32)', provider: 'DmlExecutionProvider', precision: 'fp32' },
  { id: 'cpu', label: 'CPU', provider: 'CPUExecutionProvider', precision: 'auto' },
] as const;

function CatalogPill({ modelId }: { modelId: string }) {
  const { models } = useModelCatalog();
  const model = models.find(item => item.id === modelId);
  return model ? <StatusPill status={modelStatus(model)} /> : null;
}

/** Best status among the registered artifacts with this file name, most usable first. */
const STATUS_RANK: readonly ModelStatus[] = ['ready', 'trt', 'disk', 'download', 'locate', 'blocked'];

/**
 * Status of a registry artifact by file name. The robust occluder loads `xseg_3.onnx` and `face_occluder.onnx`, which are not catalog ids
 * (the `xseg` row is the older `xseg.onnx`), so the file name is what says whether the networks it will run are there.
 */
function FilePill({ file }: { file: string }) {
  const { models, loaded } = useModelCatalog();
  if (!loaded) return null;
  const statuses = models.filter(item => item.name.toLowerCase() === file.toLowerCase()).map(modelStatus);
  const best = STATUS_RANK.find(status => statuses.includes(status));
  return <span className="rack-file"><code>{file}</code>{best ? <StatusPill status={best} /> : <small className="muted">not registered</small>}</span>;
}

export default function EngineModelRack({ params, onParams, config, caps, onSaveConfig, variant = 'workspace' }: RackProps) {
  const rack = useRack(state => state);
  const [trtMessage, setTrtMessage] = useState('');
  const [trtBusy, setTrtBusy] = useState(false);

  const detectorValue = typeof config?.detector_model_id === 'string' ? config.detector_model_id : rack.detector;
  /** Every model dropdown also persists to the backend config at once, so previews, renders and the read-back follow it. */
  const persist = (patch: Config) => {
    if (!config || !onSaveConfig) return;
    if (Object.entries(patch).every(([key, value]) => config[key] === value)) return;
    onSaveConfig({ ...config, ...patch });
  };
  const robust = params.occlusion_mode === 'robust';

  const provider = typeof config?.preferred_provider === 'string' ? config.preferred_provider : 'auto';
  const configuredPrecision = typeof config?.precision_policy === 'string' ? config.precision_policy : 'auto';
  const precision = configuredPrecision === 'auto' && provider === 'TensorrtExecutionProvider' ? 'fp16'
    : configuredPrecision === 'auto' && (provider === 'CUDAExecutionProvider' || provider === 'DmlExecutionProvider') ? 'fp32' : configuredPrecision;
  const runtimeValue = (RUNTIMES.find(item => item.provider === provider && item.precision === precision) ?? RUNTIMES.find(item => item.provider === provider && item.precision === 'auto') ?? RUNTIMES[0]).id;
  const mixedPrecisionStatus = caps?.workstation.tensorrt_mixed_precision;
  const runtimeReason = (item: (typeof RUNTIMES)[number]): string | null => {
    const controls = caps?.workstation.controls;
    if (!controls) return null;
    const providers = controls.preferred_provider;
    if (item.id === 'dml' && !Object.prototype.hasOwnProperty.call(providers ?? {}, item.provider)) return 'DirectML is not enabled by this service configuration.';
    const status = mixedPrecisionStatus;
    if (item.id === 'trt_fp16' && status && !status.tensorrt_available) {
      return providers?.[item.provider] ?? 'TensorRT Execution Provider is unavailable.';
    }
    if (item.id === 'cuda_fp16' && status && !status.cuda_fp16_available) {
      return providers?.[item.provider] ?? 'CUDA FP16 requires CUDA Execution Provider and torch.cuda.is_available().';
    }
    if (item.precision === 'fp16' && status && !status.available) {
      return controls.precision_policy?.fp16 ?? 'FP16 mixed precision is unavailable.';
    }
    if (item.precision === 'fp16' && !status) return controls.precision_policy?.fp16 ?? null;
    return providers?.[item.provider] ?? null;
  };
  const pickRuntime = (id: string) => {
    const item = RUNTIMES.find(entry => entry.id === id);
    if (item && config && onSaveConfig) onSaveConfig({ ...config, preferred_provider: item.provider, precision_policy: item.precision });
  };
  const rebuildTrt = async () => {
    setTrtBusy(true); setTrtMessage('');
    try {
      const result = await post<{ status?: string; results?: { fp16_enabled: boolean; canary_verified: boolean }[] }>('/models/compile_trt', { model_id: detectorValue, precision: precision === 'fp32' ? 'fp32' : 'fp16' });
      const built = result.results?.[0];
      setTrtMessage(`TensorRT engine for ${detectorValue}: ${result.status ?? 'accepted'}.${built ? ` ${built.fp16_enabled ? 'FP16 mixed precision' : 'FP32 (model safety policy)'}.${built.canary_verified ? ' Canary passed.' : ''}` : ''}`);
    } catch (reason) { setTrtMessage(reason instanceof Error ? reason.message : 'The TensorRT rebuild could not start.'); } finally { setTrtBusy(false); }
  };

  return (
    <div className="rack" data-testid="engine-model-rack" data-variant={variant}>
      <ActiveModelBadge selected={{ detector: detectorValue, head: params.head_topology_engine, shader: params.shader_mode }} />
      <ComponentDiagnostics />
      <section className="rack-section" aria-label="3D head reconstruction" data-testid="rack-true3d">
        <h4>3D head reconstruction <Wiring live={false} /></h4>
        <small className="muted" data-testid="render-mode-hint">The true-3D full-head renderer is not wired; render jobs fail closed.</small>
        <True3DControls params={params} onParams={onParams} compact={variant === 'batch'} />
      </section>

      <section className="rack-section" aria-label="Face detector">
        <h4>Face detector <Wiring live /></h4>
        <ModelSelect label="Face detector" category="detector" value={detectorValue} onChange={id => { setRack({ detector: id }); persist({ detector_model_id: id }); }} options={DETECTORS.map(option => !onSaveConfig ? { ...option, disabledReason: 'Change the detector from the Faceswap workspace or Settings.' } : option)} testId="rack-detector" />
        <small className="muted" data-testid="landmarker-note">3D landmarks are measured by the engine (5-point detector landmarks, 68-point and 106-point models) and are not selectable.</small>
      </section>

      <section className="rack-section" aria-label="Mask and occlusion engines">
        <h4>Mask &amp; occlusion engines</h4>
        <div className="rack-checks">
          <label className="inspector-toggle"><input type="checkbox" aria-label="XSeg occluder" checked={robust} onChange={event => onParams(occlusionModePatch(params, event.target.checked ? 'robust' : 'standard'))} /> XSeg occluder <Wiring live /></label>
          <div className="rack-engine-files" data-testid="robust-networks"><small className="muted">Robust occlusion runs</small> <FilePill file="xseg_3.onnx" /> <FilePill file="face_occluder.onnx" /></div>
          <label className="inspector-toggle"><input type="checkbox" aria-label="BiSeNet semantic parser" checked disabled readOnly /> BiSeNet semantic parser (skin, brows, eyes, mouth, hair) <CatalogPill modelId="bisenet" /> <small className="muted">always on: the base face matte</small></label>
          <label className="inspector-toggle"><input type="checkbox" aria-label="Face box feather" checked={(params.feather_px ?? 5) > 0} onChange={event => onParams({ feather_px: event.target.checked ? 5 : 0 })} /> Face box feather <Wiring live /></label>
        </div>
        {variant === 'batch'
          ? <CompositingControls params={params} onParams={onParams} />
          : <small className="muted" data-testid="fusion-pointer">XSeg occlusion strength and seam blending are set in step 2 · Masking &amp; occlusion.</small>}
      </section>

      <section className="rack-section" aria-label="Execution provider and precision">
        <h4>Execution provider &amp; precision <Wiring live /></h4>
        <label className="inspector-field">Provider / precision
          <select aria-label="Execution provider and precision" value={runtimeValue} disabled={!onSaveConfig || !config} onChange={event => pickRuntime(event.target.value)}>
            {RUNTIMES.map(item => { const reason = runtimeReason(item); return <option key={item.id} value={item.id} disabled={!!reason}>{item.label}{reason ? ' (unavailable)' : ''}</option>; })}
          </select>
          {RUNTIMES.filter(item => runtimeReason(item)).map(item => <small key={item.id}>{item.label}: {runtimeReason(item)}</small>)}
          {!onSaveConfig && <small>Global engine setting: change it from the Faceswap workspace or Settings.</small>}
        </label>
        {mixedPrecisionStatus && <small data-testid="mixed-precision-status">
          {mixedPrecisionStatus.active
            ? `FP16 active via ${mixedPrecisionStatus.active_provider ?? 'GPU provider'}.`
            : mixedPrecisionStatus.available
              ? `FP16 available via ${mixedPrecisionStatus.tensorrt_available ? 'TensorRT' : 'CUDA'}.`
              : mixedPrecisionStatus.diagnostic ?? 'FP16 mixed precision is unavailable.'}
        </small>}
        <div className="model-manager-row">
          <button type="button" className="button small" disabled={trtBusy} onClick={() => void rebuildTrt()}>{trtBusy ? 'Requesting…' : 'Rebuild TRT engine'}</button>
        </div>
        {trtMessage && <small role="status">{trtMessage}</small>}
      </section>
    </div>
  );
}
