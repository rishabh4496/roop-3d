// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ModelSelect from './components/modelhub/ModelSelect';
import EngineModelRack from './components/modelhub/EngineModelRack';
import type { RackOption } from './components/modelhub/ModelSelect';
import ModelManager from './components/modelhub/ModelManager';
import { modelCatalog, modelStatus, type CatalogModel } from './stores/modelCatalog';
import { rackStore } from './stores/rackSelection';
import type { Capabilities, Config } from './contracts';

const api = vi.fn(), post = vi.fn();
vi.mock('./api', () => ({ api: (...args: unknown[]) => api(...args), post: (...args: unknown[]) => post(...args) }));
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const model = (id: string, patch: Partial<CatalogModel> = {}): CatalogModel => ({
  id, model_id: id, name: id, category: 'detector', native_resolution: 256, installed: false, selectable: true, vram_loaded: false, trt_cached: false,
  download_url: null, state: 'CANDIDATE', sha256: null, size_bytes: null, ...patch,
});
let fixture: CatalogModel[];
const baseFixture = (): CatalogModel[] => [
  model('det_trt', { installed: true, state: 'VERIFIED_LOCAL', trt_cached: true }),
  model('det_ready', { installed: true, state: 'VERIFIED_LOCAL', native_resolution: 128 }),
  model('det_disk', { installed: true, state: 'CANDIDATE' }),
  model('det_download', { download_url: 'https://example.invalid/detector', native_resolution: 512 }),
  model('det_local'),
  model('det_extra', { native_resolution: null }),
  model('xseg', { category: 'occluder', installed: true, state: 'VERIFIED_LOCAL' }),
  model('bisenet', { category: 'occluder' }),
  model('scrfd', { category: 'detector', native_resolution: 640, installed: true, state: 'VERIFIED_LOCAL' }),
  model('retinaface', { category: 'detector', native_resolution: 640, installed: true, state: 'VERIFIED_LOCAL' }),
];
/** Generic selector options: the ModelSelect behaviour does not depend on which category it is used for. */
const OPTIONS: RackOption[] = [
  { id: 'det_trt', label: 'Detector A' }, { id: 'det_ready', label: 'Detector B' }, { id: 'det_disk', label: 'Detector C' },
  { id: 'det_download', label: 'Detector D' }, { id: 'det_local', label: 'Detector E' },
];

let host: HTMLDivElement, root: Root;
beforeEach(() => {
  fixture = baseFixture();
  api.mockReset(); post.mockReset(); modelCatalog.reset(); rackStore.reset(); window.localStorage.clear();
  api.mockImplementation(async (path: string) => { if (path === '/models') return fixture; throw new Error(`unexpected GET ${path}`); });
  post.mockImplementation(async (path: string) => { if (path === '/models/rescan') return { models: fixture.length, installed: fixture.filter(item => item.installed).length }; throw new Error(`unexpected POST ${path}`); });
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.useRealTimers(); });
const flush = () => act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); });
const render = async (node: React.ReactNode) => { await act(async () => root.render(node)); await flush(); };
const click = (element: Element | null | undefined) => act(async () => { (element as HTMLElement).click(); });
const typeInto = (input: HTMLInputElement, value: string) => act(async () => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
});
const options = () => [...host.querySelectorAll<HTMLElement>('[role="option"]')];
const option = (text: string) => options().find(item => item.textContent!.includes(text))!;
const trigger = (label?: string) => host.querySelector<HTMLButtonElement>(label ? `[aria-label="${label}"] [role="combobox"], [data-testid="${label}"] [role="combobox"]` : '[role="combobox"]')!;
const byText = (text: string) => [...host.querySelectorAll('button')].find(item => item.textContent?.includes(text)) as HTMLButtonElement | undefined;

describe('model status', () => {
  it('maps what is on disk to the five pills the workstation shows', () => {
    expect(modelStatus(model('a', { installed: true, state: 'VERIFIED_LOCAL' }))).toBe('ready');
    expect(modelStatus(model('a', { installed: true, state: 'VERIFIED_LOCAL', trt_cached: true }))).toBe('trt');
    expect(modelStatus(model('a', { installed: true, state: 'CANDIDATE' }))).toBe('disk');
    expect(modelStatus(model('a', { download_url: 'https://x' }))).toBe('download');
    expect(modelStatus(model('a'))).toBe('locate');
    expect(modelStatus(model('a', { state: 'BLOCKED' }))).toBe('blocked');
  });
});

describe('ModelSelect', () => {
  it('lists every registered model, installed or not, with a resolution badge and a status pill', async () => {
    await render(<ModelSelect label="Detector model" category="detector" value="det_ready" onChange={vi.fn()} options={OPTIONS} registeredDisabledReason="no adapter" />);
    await click(trigger());
    expect(option('Detector A • 256px').textContent).toContain('TRT FP16 cached');
    expect(option('Detector B • 128px').textContent).toContain('Ready');
    expect(option('Detector C • 256px').textContent).toContain('On disk');
    expect(option('Detector D • 512px').textContent).toContain('Click to download');
    expect(option('Detector E • 256px').textContent).toContain('Locate local file');
    // An artifact outside the render engine's list is still shown, but cannot be chosen.
    const extra = option('det_extra');
    expect(extra.getAttribute('aria-disabled')).toBe('true');
    expect(extra.textContent).toContain('no adapter');
    expect(host.textContent).toContain('Other registered detector artifacts');
  });

  it('activates an installed model immediately', async () => {
    const onChange = vi.fn();
    await render(<ModelSelect label="Detector model" category="detector" value="det_ready" onChange={onChange} options={OPTIONS} />);
    await click(trigger());
    await click(option('Detector A'));
    expect(onChange).toHaveBeenCalledWith('det_trt');
    expect(host.querySelector('[data-testid="rack-acquire"]')).toBeNull();
  });

  it('downloads a missing model inline with progress, then activates it by itself', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: false });
    const onChange = vi.fn();
    const job = (percent: number, state: string) => ({ download_id: 'dl1', model_id: 'det_download', state, bytes_downloaded: percent, total_bytes: 100, percent, error_code: null });
    post.mockImplementation(async (path: string, body: unknown) => {
      if (path === '/models/det_download/downloads') { expect(body).toEqual({ accepted_license: false }); return job(0, 'queued'); }
      throw new Error(`unexpected POST ${path}`);
    });
    const polls = [job(40, 'downloading'), job(100, 'completed')];
    api.mockImplementation(async (path: string) => {
      if (path === '/models') return fixture;
      if (path === '/models/downloads/dl1') { const next = polls.shift() ?? job(100, 'completed'); if (next.state === 'completed') fixture = fixture.map(item => (item.id === 'det_download' ? { ...item, installed: true, state: 'VERIFIED_LOCAL' } : item)); return next; }
      throw new Error(`unexpected GET ${path}`);
    });
    await act(async () => root.render(<ModelSelect label="Detector model" category="detector" value="det_ready" onChange={onChange} options={OPTIONS} />));
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    await click(trigger());
    await click(option('Detector D • 512px'));
    expect(onChange).not.toHaveBeenCalled();                        // not activated before the weights exist
    expect(host.querySelector('[data-testid="rack-acquire"]')!.textContent).toContain('is not in app/models yet');
    await click(byText('Download & activate'));
    expect(post).toHaveBeenCalledWith('/models/det_download/downloads', { accepted_license: false });
    await act(async () => { await vi.advanceTimersByTimeAsync(800); });
    const bar = host.querySelector('progress')!;
    expect(bar.value).toBe(40);
    await act(async () => { await vi.advanceTimersByTimeAsync(800); });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(onChange).toHaveBeenCalledWith('det_download');
    expect(host.querySelector('[data-testid="rack-acquire"]')).toBeNull();
  });

  it('offers a local import when no public source exists, rejects a non-matching file, and activates after a verified import', async () => {
    const onChange = vi.fn();
    post.mockImplementation(async (path: string, body: any) => {
      if (path === '/installer/models/import') {
        if (body.bundle_path === 'D:/wrong') return { status: 'EMPTY', imported: [], skipped: [], corrupted: [{ name: 'Detector E', file: 'det_local.onnx', error: 'SHA-256 mismatch' }] };
        fixture = fixture.map(item => (item.id === 'det_local' ? { ...item, installed: true, state: 'VERIFIED_LOCAL' } : item));
        return { status: 'PASS', imported: [{ name: 'Detector E', model_id: 'det_local' }], skipped: [], corrupted: [] };
      }
      if (path === '/models/rescan') return { models: 1, installed: 1 };
      throw new Error(`unexpected POST ${path}`);
    });
    await render(<ModelSelect label="Detector model" category="detector" value="det_ready" onChange={onChange} options={OPTIONS} />);
    await click(trigger());
    await click(option('Detector E • 256px'));
    const panel = host.querySelector('[data-testid="rack-acquire"]')!;
    expect(panel.textContent).toContain('No public source');
    expect(byText('Download & activate')).toBeUndefined();
    const input = panel.querySelector('input')! as HTMLInputElement;
    await typeInto(input, 'D:/wrong');
    await click(byText('Locate local .onnx'));
    await flush();
    expect(host.querySelector('[data-testid="rack-acquire"]')!.textContent).toContain('does not match the registered SHA-256');
    expect(onChange).not.toHaveBeenCalled();
    await typeInto(host.querySelector('[data-testid="rack-acquire"] input') as HTMLInputElement, 'D:/weights');
    await click(byText('Locate local .onnx'));
    await flush(); await flush();
    expect(post).toHaveBeenCalledWith('/installer/models/import', { bundle_path: 'D:/weights' });
    expect(onChange).toHaveBeenCalledWith('det_local');
  });

  it('supports the keyboard: arrows move, Enter chooses, Escape closes', async () => {
    const onChange = vi.fn();
    await render(<ModelSelect label="Detector model" category="detector" value="det_trt" onChange={onChange} options={OPTIONS} />);
    const combo = trigger();
    await act(async () => { combo.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true })); });
    expect(combo.getAttribute('aria-expanded')).toBe('true');
    const listIndex = (id: string) => OPTIONS.findIndex(item => item.id === id);
    const target = listIndex('det_ready');
    for (let step = listIndex('det_trt'); step < target; step += 1) await act(async () => { combo.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true })); });
    await act(async () => { combo.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })); });
    expect(onChange).toHaveBeenCalledWith('det_ready');
    await act(async () => { combo.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true })); });
    await act(async () => { combo.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); });
    expect(combo.getAttribute('aria-expanded')).toBe('false');
  });
});

describe('Engine & Model Rack', () => {
  const config: Config = { preferred_provider: 'auto', precision_policy: 'auto', detector_model_id: 'scrfd' };
  const caps = { workstation: { controls: {
    preferred_provider: { auto: null, CPUExecutionProvider: null, CUDAExecutionProvider: null, TensorrtExecutionProvider: 'TensorRT requires the stream backend' },
    precision_policy: { auto: null, fp32: null, fp16: 'The reference session uses observed FP32 graph contracts' },
  } } } as unknown as Capabilities;
  const mountRack = async (params = {}) => {
    const onParams = vi.fn(), onSaveConfig = vi.fn();
    await render(<EngineModelRack params={params} onParams={onParams} config={config} caps={caps} onSaveConfig={onSaveConfig} />);
    return { onParams, onSaveConfig };
  };

  it('shows only the groups the true-3D engine consumes: head, detector, occlusion and runtime', async () => {
    await mountRack();
    for (const name of ['3D head reconstruction', 'Face detector', 'Mask and occlusion engines', 'Execution provider and precision']) {
      expect(host.querySelector(`section[aria-label="${name}"]`), name).toBeTruthy();
    }
    for (const gone of ['Face swapper model', '3D pipeline mode', 'Face enhancer', 'Frame upscaler', 'Per-person models']) {
      expect(host.querySelector(`section[aria-label="${gone}"]`), gone).toBeNull();
    }
    expect(host.querySelector('[data-testid="landmarker-note"]')!.textContent).toContain('not selectable');
    expect(host.querySelector('[data-testid="fusion-pointer"]')!.textContent).toContain('step 2');   // workspace: strength and blending live in the Masking step
  });

  it('sends the chosen detector with jobs', async () => {
    await mountRack();
    await click(host.querySelector('[data-testid="rack-detector"] [role="combobox"]'));
    await click(option('RetinaFace'));
    expect(rackStore.get().detector).toBe('retinaface');
  });


  it('saves provider / precision globally and disables combinations the engine reports unavailable', async () => {
    const { onSaveConfig } = await mountRack();
    const select = host.querySelector('select[aria-label="Execution provider and precision"]') as HTMLSelectElement;
    const byLabel = (text: string) => [...select.options].find(item => item.textContent!.includes(text))!;
    expect(byLabel('TensorRT (FP16').disabled).toBe(true);
    expect(host.textContent).toContain('TensorRT requires the stream backend');
    expect(byLabel('CUDA (FP16)').disabled).toBe(true);
    await act(async () => { select.value = 'cpu'; select.dispatchEvent(new Event('change', { bubbles: true })); });
    expect(onSaveConfig).toHaveBeenCalledWith({ ...config, preferred_provider: 'CPUExecutionProvider', precision_policy: 'auto' });
  });

  it('reports the real answer when a TensorRT rebuild is refused', async () => {
    await mountRack();
    post.mockImplementation(async (path: string) => { if (path === '/models/compile_trt') throw new Error('No verified TensorRT build adapter is configured for these models'); throw new Error(path); });
    await click(byText('Rebuild TRT engine'));
    await flush();
    expect(host.textContent).toContain('No verified TensorRT build adapter');
  });

});

describe('Models & Engines Manager', () => {
  it('lists every registered model, rescans, and states that SHA-256 verification cannot be skipped', async () => {
    await render(<ModelManager />);
    await act(async () => { const select = host.querySelector('select[aria-label="Model category"]') as HTMLSelectElement; select.value = 'all'; select.dispatchEvent(new Event('change', { bubbles: true })); });
    expect(host.querySelectorAll('[data-testid^="manager-row-"]').length).toBe(fixture.length);   // installed or not, nothing is filtered out
    const sha = host.querySelector('input[aria-label="SHA-256 verification"]') as HTMLInputElement;
    expect(sha.checked).toBe(true); expect(sha.disabled).toBe(true);
    expect(host.querySelector('[data-testid="sha-policy"]')!.textContent).toContain('no “force unblock” switch');
    expect(host.querySelector('[data-testid="manager-row-det_download"]')!.textContent).toContain('Download');
    expect(host.querySelector('[data-testid="manager-row-det_local"]')!.textContent).toContain('import a local copy');
    const installedRow = host.querySelector('[data-testid="manager-row-det_ready"]')!;
    expect([...installedRow.querySelectorAll('button')].filter(item => item.disabled).map(item => item.textContent)).toEqual(['Preload to VRAM', 'Unload']);
    await click(byText('Rescan app/models folder'));
    await flush();
    expect(post).toHaveBeenCalledWith('/models/rescan', {});
    expect(host.textContent).toContain('Rescanned app/models: 6 of 10 registered models are installed.');
  });

  it('imports a local bundle and says so when nothing matches the registry', async () => {
    post.mockImplementation(async (path: string) => { if (path === '/installer/models/import') return { status: 'EMPTY', imported: [], skipped: [], corrupted: [] }; if (path === '/models/rescan') return { models: 1, installed: 1 }; throw new Error(path); });
    await render(<ModelManager />);
    await typeInto(host.querySelector('input[aria-label="Folder or .zip with model weights"]') as HTMLInputElement, 'D:/stuff');
    await click(byText('Import local .onnx model'));
    await flush();
    expect(host.querySelector('[role="alert"]')!.textContent).toContain('No registered weights found');
  });
});
