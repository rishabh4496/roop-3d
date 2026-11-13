import { useMemo, useState } from 'react';
import { post } from '../../api';
import type { Capabilities, Config } from '../../contracts';
import { fidelityRestorer, type RestorerModel, type WorkstationParameters } from '../workstation/pipeline';
import { RENDER_MODES } from '../workstation/renderModes';
import { modelStatus, useModelCatalog, type ModelStatus } from '../../stores/modelCatalog';
import { setRack, useRack } from '../../stores/rackSelection';
import ModelSelect, { StatusPill, type RackOption } from './ModelSelect';
import ActiveModelBadge from './ActiveModelBadge';
import CompositingControls from '../studio/CompositingControls';
import { occlusionModePatch } from '../studio/compositing';

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

const SWAPPERS: RackOption[] = [
  { id: 'hyperswap_1a_256' }, { id: 'hyperswap_1b_256' }, { id: 'hyperswap_1c_256' },
  { id: 'realswap_512' }, { id: 'realswap_256' }, { id: 'simswap_512_unofficial' }, { id: 'simswap_256' },
  { id: 'inswapper_128' }, { id: 'inswapper_128_fp16' },
  { id: 'ghost_256_unet_1' }, { id: 'ghost_256_unet_2' }, { id: 'ghost_256_unet_3' },
  { id: 'blendswap_256' }, { id: 'uniface_256' },
];
/** Rack ids of the four restorers the engine has adapters for, and the `restorer_model` each one sends. */
const ENHANCER_ENGINE: Record<string, RestorerModel> = { ultramax: 'ultramax', gpen_bfr_512: 'gpen', codeformer: 'codeformer', gfpgan_v14: 'gfpgan' };
const ENGINE_ENHANCER: Record<RestorerModel, string> = { ultramax: 'ultramax', gpen: 'gpen_bfr_512', codeformer: 'codeformer', gfpgan: 'gfpgan_v14' };
const RESTORER_ADAPTERS = 'GPEN 512, CodeFormer, GFPGAN v1.4 and UltraMax';
const NO_ADAPTER = `No render adapter: the engine runs ${RESTORER_ADAPTERS} only.`;
const ENHANCERS: RackOption[] = [
  { id: 'none', modelId: null, label: 'None', note: 'No face restoration: the raw swap is composited.' },
  { id: 'ultramax', label: 'UltraMax', badge: 'CodeFormer fp16',
    note: 'The registered codeformer.fp16.onnx weights, then Laplacian detail fusion into the swap (not a GPEN / Real-ESRGAN cascade).' },
  { id: 'gpen_bfr_512', label: 'GPEN Realistic', badge: '512px' },
  { id: 'gpen_bfr_1024', label: 'GPEN Realistic', badge: '1024px', disabledReason: 'No render adapter: the GPEN restorer loads the 512 px model only.' },
  { id: 'codeformer' }, { id: 'gfpgan_v14' },
  { id: 'restoreformer_pp', disabledReason: NO_ADAPTER },
];
const ENHANCER_ARTIFACT = `Registered artifact without a render adapter: the engine runs ${RESTORER_ADAPTERS} by their pinned file names only.`;
const NO_UPSCALER = 'The render engine has no frame-upscaler stage: these weights are registry artifacts only.';
const UPSCALERS: RackOption[] = [{ id: 'none', modelId: null, label: 'None' }, ...['realesrgan_x2plus', 'realesrgan_x4plus', '4x_ultrasharp'].map(id => ({ id, disabledReason: NO_UPSCALER }))];
const DETECTORS: RackOption[] = [{ id: 'scrfd' }, { id: 'retinaface' }, { id: 'yoloface' }];

const RUNTIMES = [
  { id: 'auto', label: 'Auto (engine decides)', provider: 'auto', precision: 'auto' },
  { id: 'trt_fp16', label: 'TensorRT (FP16 mixed precision)', provider: 'TensorrtExecutionProvider', precision: 'fp16' },
  { id: 'trt_fp32', label: 'TensorRT (FP32)', provider: 'TensorrtExecutionProvider', precision: 'fp32' },
  { id: 'cuda_fp16', label: 'CUDA (FP16)', provider: 'CUDAExecutionProvider', precision: 'fp16' },
  { id: 'cuda_fp32', label: 'CUDA (FP32)', provider: 'CUDAExecutionProvider', precision: 'fp32' },
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
  const { models } = useModelCatalog();
  const [trtMessage, setTrtMessage] = useState('');
  const [trtBusy, setTrtBusy] = useState(false);

  const swapperValue = rack.swapper ?? (typeof config?.swapper_model_id === 'string' ? config.swapper_model_id : 'inswapper_128');
  const swapperOptions = useMemo(() => [...SWAPPERS, ...models.filter(model => model.category === "swapper" && model.local_format === "dfm").map(model => ({ id: model.id, label: `${model.name} (DFM)` }))].map(option => {
    const reason = caps?.workstation.controls.swapper_model_id?.[option.id];
    return reason ? { ...option, disabledReason: reason } : option;
  }), [caps, models]);

  const blend = params.restoration_blend ?? 0;
  const engine = params.restorer_model ?? 'gpen';
  const enhancerValue = blend > 0 ? ENGINE_ENHANCER[engine] : 'none';
  /** Every model dropdown also persists to the backend config at once, so previews, renders and the read-back follow it. */
  const persist = (patch: Config) => {
    if (!config || !onSaveConfig) return;
    if (Object.entries(patch).every(([key, value]) => config[key] === value)) return;
    onSaveConfig({ ...config, ...patch });
  };
  const pickEnhancer = (id: string) => {
    if (id === 'none') { onParams({ restoration_blend: 0 }); persist({ face_restoration_policy: 'off' }); return; }
    const model = ENHANCER_ENGINE[id];
    if (model) { onParams({ restorer_model: model, restoration_blend: blend > 0 ? blend : 0.7 }); persist({ face_restoration_policy: model }); }
  };
  const restorerRuns = blend > 0 && !(params.bypass ?? []).includes('restorer');
  const fidelityActive = fidelityRestorer(engine) && restorerRuns;
  const mode = params.render_mode ?? 'standard';
  const robust = params.occlusion_mode === 'robust';

  const provider = typeof config?.preferred_provider === 'string' ? config.preferred_provider : 'auto';
  const configuredPrecision = typeof config?.precision_policy === 'string' ? config.precision_policy : 'auto';
  const precision = configuredPrecision === 'auto' && provider === 'TensorrtExecutionProvider' ? 'fp16'
    : configuredPrecision === 'auto' && provider === 'CUDAExecutionProvider' ? 'fp32' : configuredPrecision;
  const runtimeValue = (RUNTIMES.find(item => item.provider === provider && item.precision === precision) ?? RUNTIMES.find(item => item.provider === provider && item.precision === 'auto') ?? RUNTIMES[0]).id;
  const mixedPrecisionStatus = caps?.workstation.tensorrt_mixed_precision;
  const runtimeReason = (item: (typeof RUNTIMES)[number]): string | null => {
    const controls = caps?.workstation.controls;
    if (!controls) return null;
    const providers = controls.preferred_provider;
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
      const result = await post<{ status?: string; results?: { fp16_enabled: boolean; canary_verified: boolean }[] }>('/models/compile_trt', { model_id: swapperValue, precision: precision === 'fp32' ? 'fp32' : 'fp16' });
      const built = result.results?.[0];
      setTrtMessage(`TensorRT engine for ${swapperValue}: ${result.status ?? 'accepted'}.${built ? ` ${built.fp16_enabled ? 'FP16 mixed precision' : 'FP32 (model safety policy)'}.${built.canary_verified ? ' Canary passed.' : ''}` : ''}`);
    } catch (reason) { setTrtMessage(reason instanceof Error ? reason.message : 'The TensorRT rebuild could not start.'); } finally { setTrtBusy(false); }
  };

  return (
    <div className="rack" data-testid="engine-model-rack" data-variant={variant}>
      <ActiveModelBadge selected={{ swapper: swapperValue, detector: typeof config?.detector_model_id === 'string' ? config.detector_model_id : rack.detector,
        enhancer: blend > 0 ? engine : 'off' }} />
      <section className="rack-section" aria-label="Face swapper model">
        <h4>Face swapper <Wiring live /></h4>
        <ModelSelect label="Swapper model" category="swapper" value={swapperValue} onChange={id => { setRack({ swapper: id }); persist({ swapper_model_id: id }); }} options={swapperOptions}
          registeredDisabledReason="Registered artifact without a render adapter: the engine cannot run it yet." testId="rack-swapper"
          hint="Sent with every preview and render job. Choosing a model that is not installed offers to download or locate it first." />
      </section>

      <section className="rack-section" aria-label="3D pipeline mode">
        <h4>3D pipeline mode <Wiring live /></h4>
        <div className="mode-cards" role="radiogroup" aria-label="3D pipeline mode">
          {RENDER_MODES.map(item => (
            <button key={item.id} type="button" role="radio" aria-checked={mode === item.id} className={`mode-card ${mode === item.id ? 'active' : ''}`} onClick={() => onParams({ render_mode: item.id })} title={item.hint}>
              <b>{item.label}</b>
            </button>
          ))}
        </div>
        <small className="muted">{RENDER_MODES.find(item => item.id === mode)?.hint}</small>
      </section>

      <section className="rack-section" aria-label="Face enhancer">
        <h4>Face enhancer / restorer <Wiring live /></h4>
        <ModelSelect label="Enhancer" category="enhancer" value={enhancerValue} onChange={pickEnhancer} options={ENHANCERS} registeredDisabledReason={ENHANCER_ARTIFACT} testId="rack-enhancer" />
        <label className="inspector-field">Enhancer blend <output>{Math.round(blend * 100)}%</output>
          <input type="range" aria-label="Enhancer blend" min={0} max={100} step={1} value={Math.round(blend * 100)} onChange={event => onParams({ restoration_blend: Number(event.target.value) / 100 })} /></label>
        <label className="inspector-field">Fidelity <output>{Math.round((params.fidelity_weight ?? 0.5) * 100)}%</output>
          <input type="range" aria-label="Enhancer fidelity" min={0} max={100} step={1} disabled={!fidelityActive} value={Math.round((params.fidelity_weight ?? 0.5) * 100)} onChange={event => onParams({ fidelity_weight: Number(event.target.value) / 100 })} />
          {!fidelityActive && <small>{fidelityRestorer(engine) ? 'Raise the enhancer blend above 0 to use fidelity.' : 'Fidelity applies to CodeFormer and UltraMax: GPEN and GFPGAN have no fidelity input.'}</small>}</label>
      </section>

      <section className="rack-section" aria-label="Frame upscaler">
        <h4>Frame upscaler <Wiring live={false} /></h4>
        <ModelSelect label="Upscaler" category="upscaler" value="none" onChange={() => undefined} options={UPSCALERS} registeredDisabledReason={NO_UPSCALER} testId="rack-upscaler" />
      </section>

      <section className="rack-section" aria-label="Face detector">
        <h4>Face detector <Wiring live /></h4>
        <ModelSelect label="Face detector" category="detector" value={typeof config?.detector_model_id === 'string' ? config.detector_model_id : rack.detector} onChange={id => { setRack({ detector: id }); persist({ detector_model_id: id }); }} options={DETECTORS.map(option => !onSaveConfig ? { ...option, disabledReason: 'Change the detector from the Faceswap workspace or Settings.' } : option)} testId="rack-detector" />
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

export { SWAPPERS, ENHANCERS };
