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

const loaded = (stage: 'detector' | 'swapper' | 'enhancer', model_id: string, filename: string, sha256: string | null = 'a'.repeat(64)) => ({
  stage, model_id, model_path: `/app/models/${filename}`, filename, provider: 'TensorrtExecutionProvider', precision: 'fp16', sha256, loaded_at: 1, switches: 0,
});
const status = (patch: Partial<PipelineStatus> = {}): PipelineStatus => ({
  pipeline: { engine_state: 'IDLE', engine_label: null, engine_model: null, engine_precision: null },
  selected: { swapper_model: 'inswapper_128', detector_model: 'scrfd', enhancer_model: 'gpen' },
  active: { detector: loaded('detector', 'scrfd', 'scrfd_2.5g_kps.onnx'), swapper: loaded('swapper', 'inswapper_128', 'inswapper_128.onnx'), enhancer: loaded('enhancer', 'gpen_bfr_512', 'GPEN-BFR-512.onnx') },
  precision: { label: 'TensorRT FP16 Mixed', stage: 'swapper', stages: { swapper: 'TensorRT FP16 Mixed', detector: 'TensorRT FP16 Mixed' }, policy: 'fp16' },
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
  it('shows the live detector, swapper, enhancer (verified) and precision read back from /pipeline/status', async () => {
    await act(async () => root.render(<EngineStatusBar />));
    await flush();
    expect(api).toHaveBeenCalledWith('/pipeline/status');
    expect(text('engine-detector')).toBe('Detector:scrfd');
    expect(text('engine-swapper')).toBe('Swapper:inswapper_128');
    expect(text('engine-enhancer')).toBe('Enhancer:gpen_bfr_512• Verified');
    expect(text('engine-precision')).toBe('Precision:TensorRT FP16 Mixed');
    expect(host.querySelector('[data-testid=engine-swapper]')!.getAttribute('title')).toContain('inswapper_128.onnx');
  });

  it('says so instead of inventing a model when nothing is loaded, and never labels an unhashed enhancer Verified', async () => {
    current = status({ active: { detector: null, swapper: null, enhancer: loaded('enhancer', 'codeformer', 'codeformer.onnx', null) }, precision: { label: null, stage: null, stages: {}, policy: 'auto' } });
    await act(async () => root.render(<EngineStatusBar />));
    await flush();
    expect(text('engine-swapper')).toBe('Swapper:—');
    expect(text('engine-precision')).toBe('Precision:—');
    expect(text('engine-enhancer')).toContain('• Unverified');
    expect(text('engine-enhancer')).not.toContain('• Verified');
  });

  it('turns a failed model switch into an error toast with the exact backend reason, once, and flags the stage', async () => {
    current = status({ failures: [{ id: 1, stage: 'swapper', model_id: 'simswap_256', code: 'MODEL_NOT_INSTALLED', reason: 'Selected swapper has no installed local ONNX weights', at: 100 }] });
    await act(async () => root.render(<><EngineStatusBar /><ToastHost /></>));
    await flush();
    const alert = host.querySelector('[role=alert]')!;
    expect(alert.textContent).toContain('Swapper did not switch to simswap_256');
    expect(alert.textContent).toContain('Selected swapper has no installed local ONNX weights [MODEL_NOT_INSTALLED]');
    expect(alert.textContent).toContain('Still using inswapper_128 (inswapper_128.onnx).');          // the old model is named, not silently kept
    expect(host.querySelector('[data-testid=engine-swapper]')!.className).toContain('failed');
    expect(text('engine-swapper')).toContain('simswap_256 failed');

    await act(async () => { await refreshPipelineStatus(); });                                       // the same failure on the next poll
    expect(toastStore.get().items).toHaveLength(1);
    current = status({ failures: [{ id: 2, stage: 'enhancer', model_id: 'codeformer', code: 'MODEL_SESSION_FAILED', reason: 'Verified model could not initialize on the requested provider', at: 200 }] });
    await act(async () => { await refreshPipelineStatus(); });
    expect(toastStore.get().items.map(item => item.title)).toEqual(['Swapper did not switch to simswap_256', 'Enhancer did not switch to codeformer']);
  });

  it('keeps working (offline note) when the status endpoint is unreachable', async () => {
    api.mockImplementation(async () => { throw new Error('Local service returned 503'); });
    await act(async () => root.render(<EngineStatusBar />));
    await flush();
    expect(host.querySelector('[data-testid=engine-status-bar]')!.getAttribute('data-state')).toBe('offline');
    expect(host.textContent).toContain('Engine status unavailable: Local service returned 503');
  });
});
