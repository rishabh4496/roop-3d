// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import EngineStatusBar from './components/hud/EngineStatusBar';
import ToastHost from './components/common/ToastHost';
import { forgetRaisedFailures, pipelineStatusStore, refreshPipelineStatus, type PipelineStatus } from './stores/pipelineStatus';
import { clearToasts, toastStore } from './stores/toasts';

const api = vi.fn();
vi.mock('./api', () => ({ api: (...args: unknown[]) => api(...args) }));
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const loaded = (stage: 'detector' | 'head' | 'shader', model_id: string, filename: string, sha256: string | null = 'a'.repeat(64)) => ({
  stage, model_id, model_path: `/app/models/${filename}`, filename, provider: 'TensorrtExecutionProvider', precision: 'fp16', sha256, loaded_at: 1, switches: 0,
});
const status = (patch: Partial<PipelineStatus> = {}): PipelineStatus => ({
  pipeline: { engine_state: 'IDLE', engine_label: null, engine_model: null, engine_precision: null },
  selected: { detector_model: 'scrfd', head_topology_engine: 'MakeHuman-CC0-Craniofacial-3D' },
  active: { detector: loaded('detector', 'scrfd', 'scrfd_2.5g_kps.onnx'), head: loaded('head', 'makehuman_cc0', 'makehuman_head_v1.npz'), shader: loaded('shader', 'pbr_sss', 'pbr_sss.onnx') },
  precision: { label: 'TensorRT FP16 Mixed', stage: 'head', stages: { head: 'TensorRT FP16 Mixed', detector: 'TensorRT FP16 Mixed' }, policy: 'fp16' },
  failures: [], ...patch,
});

let host: HTMLDivElement, root: Root, current: PipelineStatus;
beforeEach(() => {
  current = status(); api.mockReset(); pipelineStatusStore.reset(); forgetRaisedFailures(); clearToasts();
  api.mockImplementation(async (path: string) => { if (path === '/pipeline/status') return current; throw new Error(`unexpected GET ${path}`); });
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); });
const flush = () => act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); });
const text = (id: string) => host.querySelector(`[data-testid="${id}"]`)!.textContent!;

describe('EngineStatusBar', () => {
  it('shows the live detector, head model, shader (verified) and precision read back from /pipeline/status', async () => {
    await act(async () => root.render(<EngineStatusBar />));
    await flush();
    expect(api).toHaveBeenCalledWith('/pipeline/status');
    expect(text('engine-detector')).toBe('Detector:scrfd');
    expect(text('engine-head')).toBe('Head model:makehuman_cc0• Verified');
    expect(text('engine-shader')).toBe('Shader:pbr_sss• Verified');
    expect(text('engine-precision')).toBe('Precision:TensorRT FP16 Mixed');
    expect(host.querySelector('[data-testid=engine-head]')!.getAttribute('title')).toContain('makehuman_head_v1.npz');
    expect(host.querySelector('[data-testid=engine-swapper], [data-testid=engine-enhancer]')).toBeNull();
  });

  it('says so instead of inventing a model when nothing is loaded, and never labels an unhashed shader Verified', async () => {
    current = status({ active: { detector: null, head: null, shader: loaded('shader', 'gaussian_head', 'gaussian_head.onnx', null) }, precision: { label: null, stage: null, stages: {}, policy: 'auto' } });
    await act(async () => root.render(<EngineStatusBar />));
    await flush();
    expect(text('engine-head')).toBe('Head model:—');
    expect(text('engine-precision')).toBe('Precision:—');
    expect(text('engine-shader')).toContain('• Unverified');
    expect(text('engine-shader')).not.toContain('• Verified');
  });

  it('turns a failed model switch into an error toast with the exact backend reason, once, and flags the stage', async () => {
    current = status({ failures: [{ id: 1, stage: 'head', model_id: 'flame2023_open', code: 'MODEL_NOT_INSTALLED', reason: 'Selected head model has no installed local weights', at: 100 }] });
    await act(async () => root.render(<><EngineStatusBar /><ToastHost /></>));
    await flush();
    const alert = host.querySelector('[role=alert]')!;
    expect(alert.textContent).toContain('Head model did not switch to flame2023_open');
    expect(alert.textContent).toContain('Selected head model has no installed local weights [MODEL_NOT_INSTALLED]');
    expect(alert.textContent).toContain('Still using makehuman_cc0 (makehuman_head_v1.npz).');        // the old model is named, not silently kept
    expect(host.querySelector('[data-testid=engine-head]')!.className).toContain('failed');
    expect(text('engine-head')).toContain('flame2023_open failed');

    await act(async () => { await refreshPipelineStatus(); });                                       // the same failure on the next poll
    expect(toastStore.get().items).toHaveLength(1);
    current = status({ failures: [{ id: 2, stage: 'shader', model_id: 'gaussian_head', code: 'MODEL_SESSION_FAILED', reason: 'Verified model could not initialize on the requested provider', at: 200 }] });
    await act(async () => { await refreshPipelineStatus(); });
    expect(toastStore.get().items.map(item => item.title)).toEqual(['Head model did not switch to flame2023_open', 'Shader did not switch to gaussian_head']);
  });

  it('keeps working (offline note) when the status endpoint is unreachable', async () => {
    api.mockImplementation(async () => { throw new Error('Local service returned 503'); });
    await act(async () => root.render(<EngineStatusBar />));
    await flush();
    expect(host.querySelector('[data-testid=engine-status-bar]')!.getAttribute('data-state')).toBe('offline');
    expect(host.textContent).toContain('Engine status unavailable: Local service returned 503');
  });
});
