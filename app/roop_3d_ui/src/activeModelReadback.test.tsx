// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import EngineModelRack from './components/modelhub/EngineModelRack';
import { buildLivePayload } from './livePreview';
import { modelCatalog, type CatalogModel } from './stores/modelCatalog';
import { activeModelStore, type ActiveModels } from './stores/activeModels';
import { rackStore } from './stores/rackSelection';
import { DEFAULT_PARAMETERS } from './components/workstation/pipeline';
import type { Capabilities, Config, JobRequest } from './contracts';

const api = vi.fn(), post = vi.fn();
vi.mock('./api', () => ({ api: (...args: unknown[]) => api(...args), post: (...args: unknown[]) => post(...args) }));
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const model = (id: string, category: string, patch: Partial<CatalogModel> = {}): CatalogModel => ({
  id, model_id: id, name: id, category, native_resolution: null, installed: true, selectable: true, vram_loaded: false, trt_cached: false,
  download_url: null, state: 'VERIFIED_LOCAL', sha256: null, size_bytes: null, ...patch,
});
const catalog = [model('inswapper_128', 'swapper'), model('hyperswap_1c_256', 'swapper'), model('gpen_bfr_512', 'enhancer'),
  model('codeformer', 'enhancer'), model('scrfd', 'detector'), model('retinaface', 'detector')];
const loaded = (stage: 'detector' | 'swapper' | 'enhancer', model_id: string, filename: string) => ({
  stage, model_id, model_path: `/app/models/${filename}`, filename, provider: 'CUDAExecutionProvider', precision: 'fp16', sha256: 'a'.repeat(64), loaded_at: 1, switches: 0,
});
let active: ActiveModels;

let host: HTMLDivElement, root: Root;
beforeEach(() => {
  active = {
    selected: { swapper_model: 'inswapper_128', detector_model: 'scrfd', enhancer_model: 'gpen' },
    active: { detector: loaded('detector', 'scrfd', 'scrfd_10g_bnkps.onnx'), swapper: loaded('swapper', 'inswapper_128', 'inswapper_128.onnx'),
      enhancer: loaded('enhancer', 'gpen_bfr_512', 'GPEN-BFR-512-fp16.onnx') },
  };
  api.mockReset(); post.mockReset(); modelCatalog.reset(); rackStore.reset(); activeModelStore.reset(); window.localStorage.clear();
  api.mockImplementation(async (path: string) => {
    if (path === '/models') return catalog;
    if (path === '/models/active') return active;
    throw new Error(`unexpected GET ${path}`);
  });
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); });
const flush = () => act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); });
const click = (element: Element | null | undefined) => act(async () => { (element as HTMLElement).click(); });
const option = (text: string) => [...host.querySelectorAll<HTMLElement>('[role="option"]')].find(item => item.textContent!.includes(text))!;

const config: Config = { preferred_provider: 'auto', precision_policy: 'auto', swapper_model_id: 'inswapper_128', detector_model_id: 'scrfd', face_restoration_policy: 'gpen' };
const caps = { workstation: { controls: { swapper_model_id: {}, preferred_provider: { auto: null }, precision_policy: { auto: null } } } } as unknown as Capabilities;
const mount = async (params = { restoration_blend: 0.7, restorer_model: 'gpen' as const }) => {
  const onParams = vi.fn(), onSaveConfig = vi.fn();
  await act(async () => root.render(<EngineModelRack params={params} onParams={onParams} config={config} caps={caps} onSaveConfig={onSaveConfig} />));
  await flush();
  return { onParams, onSaveConfig };
};

describe('active model read-back', () => {
  it('shows the exact weight file each stage has in VRAM', async () => {
    await mount();
    const text = (stage: string) => host.querySelector(`[data-testid="active-model-${stage}"]`)!.textContent!;
    expect(text('detector')).toContain('scrfd_10g_bnkps.onnx');
    expect(text('swapper')).toContain('inswapper_128.onnx');
    expect(text('enhancer')).toContain('GPEN-BFR-512-fp16.onnx');
    expect(text('swapper')).toContain('CUDA · fp16');
  });

  it('says a newly selected model has not loaded yet, and never claims it did', async () => {
    active.selected.swapper_model = 'hyperswap_1c_256';
    rackStore.set({ swapper: 'hyperswap_1c_256' });
    await mount();
    const row = host.querySelector('[data-testid="active-model-swapper"]')!;
    expect(row.textContent).toContain('inswapper_128.onnx');
    expect(row.textContent).toContain('Selected hyperswap_1c_256: loads on the next preview or render');
  });

  it('reports a stage with no session instead of inventing a file', async () => {
    active.active.enhancer = null;
    await mount({ restoration_blend: 0, restorer_model: 'gpen' });
    expect(host.querySelector('[data-testid="active-model-enhancer"]')!.textContent).toContain('Nothing loaded');
  });
});

describe('every model dropdown persists to the backend config', () => {
  it('swapper, detector and enhancer each save their own config field', async () => {
    const { onSaveConfig, onParams } = await mount();
    await click(host.querySelector('[data-testid="rack-swapper"] [role="combobox"]'));
    await click(option('HyperSwap 1C'));
    expect(onSaveConfig).toHaveBeenLastCalledWith({ ...config, swapper_model_id: 'hyperswap_1c_256' });

    await click(host.querySelector('[data-testid="rack-detector"] [role="combobox"]'));
    await click(option('RetinaFace'));
    expect(onSaveConfig).toHaveBeenLastCalledWith({ ...config, detector_model_id: 'retinaface' });

    await click(host.querySelector('[data-testid="rack-enhancer"] [role="combobox"]'));
    await click(option('CodeFormer'));
    expect(onParams).toHaveBeenLastCalledWith({ restorer_model: 'codeformer', restoration_blend: 0.7 });
    expect(onSaveConfig).toHaveBeenLastCalledWith({ ...config, face_restoration_policy: 'codeformer' });
  });

  it('does not re-save a choice that is already the saved one', async () => {
    const { onSaveConfig } = await mount();
    await click(host.querySelector('[data-testid="rack-detector"] [role="combobox"]'));
    await click(option('SCRFD'));
    expect(onSaveConfig).not.toHaveBeenCalled();
  });
});

describe('model ids travel with every preview payload', () => {
  const request = { engine_id: 'reference_onnx', swapper_model_id: 'hyperswap_1c_256', detector_model_id: 'yoloface', target_path: 'clip.mp4',
    inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }] } as unknown as JobRequest;
  it('carries swapper, detector and the enhancer architecture', () => {
    const payload = buildLivePayload({ request, time: 1, fps: 25, restorerAvailable: true,
      parameters: { ...DEFAULT_PARAMETERS, restorer_model: 'codeformer', restoration_blend: 0.6 } })!;
    expect(payload).toMatchObject({ swapper_model_id: 'hyperswap_1c_256', detector_model_id: 'yoloface', restorer: 'codeformer' });
  });
});
