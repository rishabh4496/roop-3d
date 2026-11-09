import { useMemo, useState } from 'react';
import { post } from '../../api';
import type { Capabilities, Config } from '../../contracts';
import type { RestorerModel, WorkstationParameters } from '../workstation/pipeline';
import { modelStatus, useModelCatalog } from '../../stores/modelCatalog';
import { setRack, useRack } from '../../stores/rackSelection';
import ModelSelect, { StatusPill, type RackOption } from './ModelSelect';
import PersonModelPicker from './PersonModelPicker';
import ActiveModelBadge from './ActiveModelBadge';

type Params = Partial<WorkstationParameters>;
export type RackProps = {
  /** Current values of the parameters the rack edits (the workspace parameters, or the Batch defaults). */
  params: Params; onParams: (patch: Params) => void;
  config: Config | null; caps: Capabilities | null;
  /** Persists global engine settings (execution provider / precision). Absent = those two controls are read-only. */
  onSaveConfig?: (value: Config) => void;
  /** Detected people in the current target, for per-person overrides (workspace only). */
  people?: { key: string; label: string }[];
  variant?: 'workspace' | 'batch';
};

const Wiring = ({ live }: { live: boolean }) => (
  <span className={`rack-wiring ${live ? 'live' : ''}`} title={live ? 'Changes what the render engine runs.' : 'Remembered by the workstation. The render engine has no setting for this yet, so it does not change the output.'}>
    {live ? 'LIVE' : 'SAVED PREFERENCE'}
  </span>
);

const SWAPPERS: RackOption[] = [
  { id: 'hyperswap_1a_256' }, { id: 'hyperswap_1b_256' }, { id: 'hyperswap_1c_256' },
  { id: 'realswap_512' }, { id: 'realswap_256' }, { id: 'simswap_512_unofficial' }, { id: 'simswap_256' },
  { id: 'inswapper_128' }, { id: 'inswapper_128_fp16' },
  { id: 'ghost_256_unet_1' }, { id: 'ghost_256_unet_2' }, { id: 'ghost_256_unet_3' },
  { id: 'blendswap_256' }, { id: 'uniface_256' },
];
const ENHANCER_ENGINE: Record<string, RestorerModel> = { ultramax: 'ultramax', gpen_bfr_512: 'gpen', codeformer: 'codeformer', gfpgan_v14: 'gfpgan' };
const ENGINE_ENHANCER: Record<string, string> = { ultramax: 'ultramax', gpen: 'gpen_bfr_512', auto: 'gpen_bfr_512', codeformer: 'codeformer', gfpgan: 'gfpgan_v14' };
const NO_ADAPTER = 'The render engine has no adapter for this restorer yet (it runs GPEN at 512).';
const ENHANCERS: RackOption[] = [
  { id: 'none', modelId: null, label: 'None', note: 'No face restoration: the raw swap is composited.' },
  { id: 'ultramax' }, { id: 'gpen_bfr_512', label: 'GPEN Realistic', badge: '512px' },
  { id: 'gpen_bfr_1024', label: 'GPEN Realistic', badge: '1024px', disabledReason: NO_ADAPTER },
  { id: 'codeformer' }, { id: 'gfpgan_v14' },
  { id: 'restoreformer_pp', disabledReason: NO_ADAPTER },
];
const UPSCALERS: RackOption[] = [{ id: 'none', modelId: null, label: 'None' }, { id: 'realesrgan_x2plus' }, { id: 'realesrgan_x4plus' }, { id: '4x_ultrasharp' }];
const DETECTORS: RackOption[] = [{ id: 'scrfd' }, { id: 'retinaface' }, { id: 'yoloface' }];
const LANDMARKERS: RackOption[] = [{ id: 'dense_3d', modelId: null, label: 'Dense 3D 468/106 Landmarker', note: 'Built into the 3D stack; no separate weights.' }];

const MODES: { id: NonNullable<WorkstationParameters['render_mode']>; label: string; hint: string; relight?: boolean }[] = [
  { id: 'reenactment_3d', label: '3D Full-Head Hybrid', hint: 'Source head geometry carries the target expression, then the swapper runs on top.' },
  { id: 'pose_3d', label: '3D Pose-Guided Relit Swap', hint: 'The swap is projected on the source identity’s own geometry and relit to the scene.', relight: true },
  { id: 'standard', label: '2D Direct Swap', hint: 'The proven identity swap with no 3D projection.' },
];

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

export default function EngineModelRack({ params, onParams, config, caps, onSaveConfig, people = [], variant = 'workspace' }: RackProps) {
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
  const enhancerValue = blend > 0 ? ENGINE_ENHANCER[engine] ?? 'gpen_bfr_512' : 'none';
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
  const usesFidelity = engine === 'codeformer' || engine === 'ultramax';
  const mode = params.render_mode ?? 'standard';
  const pickMode = (id: (typeof MODES)[number]['id']) => {
    const next: Params = { render_mode: id };
    if (MODES.find(item => item.id === id)?.relight) next.relight = true;
    onParams(next);
  };

  const provider = typeof config?.preferred_provider === 'string' ? config.preferred_provider : 'auto';
  const configuredPrecision = typeof config?.precision_policy === 'string' ? config.precision_policy : 'auto';
  const precision = configuredPrecision === 'auto' && provider === 'TensorrtExecutionProvider' ? 'fp16'
    : configuredPrecision === 'auto' && provider === 'CUDAExecutionProvider' ? 'fp32' : configuredPrecision;
  const runtimeValue = (RUNTIMES.find(item => item.provider === provider && item.precision === precision) ?? RUNTIMES.find(item => item.provider === provider && item.precision === 'auto') ?? RUNTIMES[0]).id;
  const runtimeReason = (item: (typeof RUNTIMES)[number]): string | null => {
    const controls = caps?.workstation.controls;
    if (!controls) return null;
    return controls.preferred_provider?.[item.provider] ?? (item.precision === 'fp16' && item.provider !== 'TensorrtExecutionProvider' ? 'CUDA uses the selected model’s graph precision; use TensorRT for mixed precision.' : null);
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
          {MODES.map(item => (
            <button key={item.id} type="button" role="radio" aria-checked={mode === item.id} className={`mode-card ${mode === item.id ? 'active' : ''}`} onClick={() => pickMode(item.id)} title={item.hint}>
              <b>{item.label}</b>
            </button>
          ))}
        </div>
        <small className="muted">{MODES.find(item => item.id === mode)?.hint}</small>
      </section>

      <section className="rack-section" aria-label="Face enhancer">
        <h4>Face enhancer / restorer <Wiring live /></h4>
        <ModelSelect label="Enhancer" category="enhancer" value={enhancerValue} onChange={pickEnhancer} options={ENHANCERS} includeRegistered={false} testId="rack-enhancer" />
        <label className="inspector-field">Enhancer blend <output>{Math.round(blend * 100)}%</output>
          <input type="range" aria-label="Enhancer blend" min={0} max={100} step={1} value={Math.round(blend * 100)} onChange={event => onParams({ restoration_blend: Number(event.target.value) / 100 })} /></label>
        <label className="inspector-field">Fidelity <output>{Math.round((params.fidelity_weight ?? 0.5) * 100)}%</output>
          <input type="range" aria-label="Enhancer fidelity" min={0} max={100} step={1} disabled={!usesFidelity} value={Math.round((params.fidelity_weight ?? 0.5) * 100)} onChange={event => onParams({ fidelity_weight: Number(event.target.value) / 100 })} />
          {!usesFidelity && <small>Fidelity applies to CodeFormer and UltraMax.</small>}</label>
      </section>

      <section className="rack-section" aria-label="Frame upscaler">
        <h4>Frame upscaler <Wiring live={false} /></h4>
        <ModelSelect label="Upscaler" category="upscaler" value={rack.upscaler} onChange={id => setRack({ upscaler: id })} options={UPSCALERS} testId="rack-upscaler" />
      </section>

      <section className="rack-section" aria-label="Face detector and landmarker">
        <h4>Detector &amp; 3D landmarker <Wiring live /></h4>
        <ModelSelect label="Face detector" category="detector" value={typeof config?.detector_model_id === 'string' ? config.detector_model_id : rack.detector} onChange={id => { setRack({ detector: id }); persist({ detector_model_id: id }); }} options={DETECTORS.map(option => !onSaveConfig ? { ...option, disabledReason: 'Change the detector from the Faceswap workspace or Settings.' } : option)} testId="rack-detector" />
        <ModelSelect label="3D landmarker" category="landmarker" value={rack.landmarker} onChange={id => setRack({ landmarker: id })} options={LANDMARKERS} includeRegistered={false} testId="rack-landmarker" />
      </section>

      <section className="rack-section" aria-label="Mask and occlusion engines">
        <h4>Mask &amp; occlusion engines</h4>
        <div className="rack-checks">
          <label className="inspector-toggle"><input type="checkbox" aria-label="3D cranial silhouette" checked={rack.cranial} onChange={event => setRack({ cranial: event.target.checked })} /> 3D cranial silhouette <Wiring live={false} /></label>
          <label className="inspector-toggle"><input type="checkbox" aria-label="XSeg occluder" checked={params.occlusion_mode === 'robust'} onChange={event => onParams({ occlusion_mode: event.target.checked ? 'robust' : 'standard' })} /> XSeg occluder <CatalogPill modelId="xseg" /> <Wiring live /></label>
          <label className="inspector-toggle"><input type="checkbox" aria-label="BiSeNet semantic parser" checked disabled readOnly /> BiSeNet semantic parser (skin, brows, eyes, mouth, hair) <CatalogPill modelId="bisenet" /> <small className="muted">always on: the base face matte</small></label>
          <label className="inspector-toggle"><input type="checkbox" aria-label="Face box feather" checked={(params.feather_px ?? 5) > 0} onChange={event => onParams({ feather_px: event.target.checked ? 5 : 0 })} /> Face box feather <Wiring live /></label>
        </div>
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
        <div className="model-manager-row">
          <button type="button" className="button small" disabled={trtBusy} onClick={() => void rebuildTrt()}>{trtBusy ? 'Requesting…' : 'Rebuild TRT engine'}</button>
        </div>
        {trtMessage && <small role="status">{trtMessage}</small>}
      </section>

      {variant === 'workspace' && (
        <section className="rack-section" aria-label="Per-person models">
          <h4>Per-person models <Wiring live={false} /></h4>
          {people.length === 0 ? <small className="muted">Inspect a target to pin different models to each person. You can also right-click a face on the canvas.</small>
            : people.map(person => <PersonModelPicker key={person.key} personKey={person.key} label={person.label} swapperOptions={swapperOptions} enhancerOptions={ENHANCERS} />)}
          <small className="muted">The render engine applies one swapper and one enhancer per job; per-person choices are kept with this workspace for when it supports them.</small>
        </section>
      )}
    </div>
  );
}

export { SWAPPERS, ENHANCERS };
