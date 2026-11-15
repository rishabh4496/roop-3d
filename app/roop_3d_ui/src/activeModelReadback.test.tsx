// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import EngineModelRack from './components/modelhub/EngineModelRack';
import { buildLivePayload } from './livePreview';
import { modelCatalog, type CatalogModel } from './stores/modelCatalog';
import { activeModelStore, type ActiveModels } from './stores/activeModels';
import { rackStore } from './stores/rackSelection';
import { DEFAULT_PARAMETERS, type WorkstationParameters } from './components/workstation/pipeline';
import type { Capabilities, Config, JobRequest } from './contracts';

const api = vi.fn(), post = vi.fn();
vi.mock('./api', () => ({ api: (...args: unknown[]) => api(...args), post: (...args: unknown[]) => post(...args) }));
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const model = (id: string, category: string, patch: Partial<CatalogModel> = {}): CatalogModel => ({
  id, model_id: id, name: id, category, native_resolution: null, installed: true, selectable: true, vram_loaded: false, trt_cached: false,
  download_url: null, state: 'VERIFIED_LOCAL', sha256: null, size_bytes: null, ...patch,
});
const catalog = [model('scrfd', 'detector'), model('retinaface', 'detector'), model('yoloface', 'detector')];
const loaded = (stage: 'detector' | 'head', model_id: string, filename: string) => ({
  stage, model_id, model_path: `/app/models/${filename}`, filename, provider: 'CUDAExecutionProvider', precision: 'fp16', sha256: 'a'.repeat(64), loaded_at: 1, switches: 0,
});
let active: ActiveModels;

let host: HTMLDivElement, root: Root;
beforeEach(() => {
  active = {
    selected: { detector_model: 'scrfd', head_topology_engine: 'MakeHuman-CC0-Craniofacial-3D' },
    active: { detector: loaded('detector', 'scrfd', 'scrfd_10g_bnkps.onnx'), head: loaded('head', 'makehuman_cc0', 'makehuman_head_v1.npz') },
  };
  api.mockReset(); post.mockReset(); modelCatalog.reset(); rackStore.reset(); activeModelStore.reset(); window.localStorage.clear();
  api.mockImplementation(async (path: string) => {
    if (path === '/models') return catalog;
    if (path === '/models/active') return active;
    if (path === '/true3d/options') return { renderer: { wired: false, message: '3D renderer not wired' }, head_topology_engines: [], shader_modes: [] };
    throw new Error(`unexpected GET ${path}`);
  });
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); });
const flush = () => act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); });
const click = (element: Element | null | undefined) => act(async () => { (element as HTMLElement).click(); });
const option = (text: string) => [...host.querySelectorAll<HTMLElement>('[role="option"]')].find(item => item.textContent!.includes(text))!;

const config: Config = { preferred_provider: 'auto', precision_policy: 'auto', detector_model_id: 'scrfd' };
const caps = { workstation: { controls: { preferred_provider: { auto: null }, precision_policy: { auto: null } } } } as unknown as Capabilities;
const mount = async (params: Partial<WorkstationParameters> = {}, useConfig: Config = config) => {
  const onParams = vi.fn(), onSaveConfig = vi.fn();
  await act(async () => root.render(<EngineModelRack params={params} onParams={onParams} config={useConfig} caps={caps} onSaveConfig={onSaveConfig} />));
  await flush();
  return { onParams, onSaveConfig };
};

describe('active model read-back', () => {
  it('shows the exact weight file each stage has in VRAM', async () => {
    await mount();
    const text = (stage: string) => host.querySelector(`[data-testid="active-model-${stage}"]`)!.textContent!;
    expect(text('detector')).toContain('scrfd_10g_bnkps.onnx');
    expect(text('head')).toContain('makehuman_head_v1.npz');
    expect(text('head')).toContain('CUDA · fp16');
  });

  it('says a newly selected model has not loaded yet, and never claims it did', async () => {
    active.selected.detector_model = 'retinaface';
    await mount({}, { ...config, detector_model_id: 'retinaface' });
    const row = host.querySelector('[data-testid="active-model-detector"]')!;
    expect(row.textContent).toContain('scrfd_10g_bnkps.onnx');
    expect(row.textContent).toContain('Selected retinaface: loads on the next preview or render');
  });

  it('reports a stage with no session instead of inventing a file', async () => {
    active.active.head = null;
    await mount();
    expect(host.querySelector('[data-testid="active-model-head"]')!.textContent).toContain('Nothing loaded');
  });

  it('has no swapper or enhancer row: the only stages are the detector and what the 3D renderer reports', async () => {
    await mount();
    const stages = [...host.querySelectorAll('[data-testid^="active-model-"]')].map(row => row.getAttribute('data-stage'));
    expect(stages).toEqual(['detector', 'head']);
    expect(host.querySelector('[data-testid="rack-swapper"], [data-testid="rack-enhancer"]')).toBeNull();
  });
});

describe('every model dropdown persists to the backend config', () => {
  it('the detector dropdown saves its own config field', async () => {
    const { onSaveConfig } = await mount();
    await click(host.querySelector('[data-testid="rack-detector"] [role="combobox"]'));
    await click(option('RetinaFace'));
    expect(onSaveConfig).toHaveBeenLastCalledWith({ ...config, detector_model_id: 'retinaface' });
  });

  it('does not re-save a choice that is already the saved one', async () => {
    const { onSaveConfig } = await mount();
    await click(host.querySelector('[data-testid="rack-detector"] [role="combobox"]'));
    await click(option('SCRFD'));
    expect(onSaveConfig).not.toHaveBeenCalled();
  });
});

describe('model ids travel with every preview payload', () => {
  const request = { engine_id: 'reference_onnx', detector_model_id: 'yoloface', target_path: 'clip.mp4',
    inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }] } as unknown as JobRequest;
  it('carries the detector and the true-3D choices, and no swapper or restorer', () => {
    const payload = buildLivePayload({ request, time: 1, fps: 25,
      parameters: { ...DEFAULT_PARAMETERS, head_topology_engine: 'FLAME-FullHead-3D', shader_mode: '3D-Gaussian-Head-Avatar' } })!;
    expect(payload).toMatchObject({ detector_model_id: 'yoloface', head_topology_engine: 'FLAME-FullHead-3D', shader_mode: '3D-Gaussian-Head-Avatar' });
    expect(Object.keys(payload).join(' ')).not.toMatch(/swapper|restorer|restoration(?!_weight)|fidelity|ultramax/i);
  });
});
