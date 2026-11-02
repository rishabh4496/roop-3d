// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import SettingsScreen from './screens/SettingsScreen';
import { api, post } from './api';
import { engineSelection, selectEngine } from './stores/engineSelection';

const Settings = SettingsScreen as React.ComponentType<any>;
const mockHub = vi.hoisted(() => ({
  failHub: false,
  activeDownloads: [] as { download_id: string; model_id: string; state: string; bytes_downloaded: number; total_bytes: number; percent: number; error_code: string | null }[],
  execution_policy: 'offline_by_default',
  download_policy: 'explicit_user_action_hash_verified_cancellable',
  model_states: ['AVAILABLE', 'NOT_INSTALLED', 'LICENSE_RESTRICTED', 'INCOMPATIBLE', 'UNVERIFIED', 'BROKEN', 'READY_FOR_INFERENCE'],
  engines: [
    { engine_id: 'reference_onnx', name: 'Roop 3D Reference ONNX', family: 'InsightFace-compatible', engine_type: 'ONNX Runtime',
      purpose: 'Local still-image and video face swap.', model_ids: ['swap'], minimum_vram_gib: null, recommended_vram_gib: null,
      dependencies: [], supported_precision: ['FP32'], supported_execution_providers: ['CPUExecutionProvider'], status: 'READY_FOR_INFERENCE',
      benchmark_status: 'NOT_RUN', actual_inference_available: true, selectable: true, selected: true, evidence: 'Hash checked.' },
    { engine_id: 'minimax_h3_ref2va_experimental', name: 'MiniMax H3 Ref2VA', family: 'ComfyUI LoRA', engine_type: 'ComfyUI',
      purpose: 'Experimental generative video.', model_ids: ['h3'], minimum_vram_gib: null, recommended_vram_gib: null,
      dependencies: ['MiniMax H3 base'], supported_precision: [], supported_execution_providers: [], status: 'UNVERIFIED',
      benchmark_status: 'NOT_RUN', actual_inference_available: false, selectable: false, selected: false, evidence: 'Video bridge unavailable.' },
  ],
  models: [
    { model_id: 'swap', name: 'inswapper_128.onnx', version: 'SHA-identified', family: 'InsightFace-compatible', engine_type: 'ONNX Runtime',
      purpose: 'Face identity transfer', source_repository: null,
      weight_files: [{ filename: 'inswapper_128.onnx', sha256: 'a'.repeat(64), size_bytes: 128, source_url: null }],
      license: { id: 'UNKNOWN', commercial_use: 'unknown', url: null, notice: 'Exact weight rights unknown.', acceptance_required: false, related_urls: [] },
      minimum_vram_gib: null, recommended_vram_gib: null, dependencies: [], supported_precision: ['FP32'],
      supported_execution_providers: ['CPUExecutionProvider'], provider_evidence: 'Test evidence', status: 'READY_FOR_INFERENCE',
      status_evidence: 'Hash checked.', benchmark_status: 'NOT_RUN', actual_inference_available: true,
      inference_evidence: 'test fixture', download_allowed: false, role: 'identity_swapper' },
    { model_id: 'h3', name: 'H3 face-swap LoRA', version: 'b2a5823', family: 'MiniMax H3 Ref2VA LoRA', engine_type: 'ComfyUI',
      purpose: 'Experimental generative video identity transfer', source_repository: 'https://huggingface.co/UntMods/FaceSwap_MiniMaxH3_REF2VA',
      weight_files: [{ filename: 'h3.safetensors', sha256: 'b'.repeat(64), size_bytes: 1024, source_url: 'https://huggingface.co/model' }],
      license: { id: 'Apache-2.0', commercial_use: 'allow', url: 'https://huggingface.co/UntMods/FaceSwap_MiniMaxH3_REF2VA', notice: 'Review base-model terms separately.', acceptance_required: true, related_urls: ['https://huggingface.co/MiniMaxAI/MiniMax-H3'] },
      minimum_vram_gib: null, recommended_vram_gib: null, dependencies: ['MiniMax H3 base'], supported_precision: [],
      supported_execution_providers: [], provider_evidence: 'Not tested', status: 'INCOMPATIBLE', status_evidence: 'No video adapter.',
      benchmark_status: 'NOT_RUN', actual_inference_available: false, inference_evidence: 'No adapter', download_allowed: true, role: 'diffusion_adapter' },
  ],
}));

vi.mock('./api', () => ({
  api: vi.fn((url: string, init?: RequestInit) => {
    if (url === '/models/hub') return mockHub.failHub ? Promise.reject(new Error('local service unavailable')) : Promise.resolve(mockHub);
    if (url === '/models/downloads') return Promise.resolve(mockHub.activeDownloads);
    if (url === '/models/downloads/download-1' && init?.method === 'DELETE') {
      return Promise.resolve({ download_id: 'download-1', model_id: 'h3', state: 'cancelled', bytes_downloaded: 0, total_bytes: 1024, percent: 0, error_code: null });
    }
    return Promise.resolve({});
  }),
  post: vi.fn(() => Promise.resolve({ download_id: 'download-1', model_id: 'h3', state: 'queued', bytes_downloaded: 0, total_bytes: 1024, percent: 0, error_code: null })),
  mediaURL: vi.fn((path: string) => path),
}));

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let host: HTMLDivElement, root: Root;

beforeEach(() => {
  engineSelection.reset();
  vi.clearAllMocks();
  mockHub.failHub = false;
  mockHub.activeDownloads = [];
  localStorage.clear();
  sessionStorage.clear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.restoreAllMocks();
});

const mount = async (element: React.ReactNode) => {
  await act(async () => root.render(element));
};

const click = async (element: Element | null) => {
  expect(element).toBeTruthy();
  await act(async () => element!.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true })));
};

describe('Stage 6 local model hub', () => {
  it('connects engine selection to the shared job preference and blocks unavailable engines', async () => {
    selectEngine('unavailable_saved_engine');
    await mount(<Settings activeTab="models" setActiveTab={() => {}} settings={{}} updateSetting={() => {}} onResetSettings={() => {}} />);
    const experimental = host.querySelector('[data-testid="engine-card-minimax_h3_ref2va_experimental"] button') as HTMLButtonElement;
    expect(experimental.disabled).toBe(true);
    expect(engineSelection.get().engineId).toBe('unavailable_saved_engine');
    await click(host.querySelector('[data-testid="engine-card-reference_onnx"] button'));
    expect(engineSelection.get().engineId).toBe('reference_onnx');
    expect(host.querySelector('[data-testid="engine-card-reference_onnx"] button')?.getAttribute('aria-pressed')).toBe('true');
  });
  it('shows the active production engine and reports the unverified H3 workflow as unavailable', async () => {
    await mount(<Settings activeTab="models" setActiveTab={() => {}} settings={{}} updateSetting={() => {}} onResetSettings={() => {}} />);
    expect(host.querySelector('[data-testid="model-hub-panel"]')?.textContent).toContain('Models and engines');
    expect(host.querySelector('[data-testid="engine-card-reference_onnx"]')?.textContent).toContain('Current production engine');
    expect(host.querySelector('[data-testid="engine-card-minimax_h3_ref2va_experimental"]')?.textContent).toContain('UNVERIFIED');
    expect(host.querySelector('[data-testid="engine-card-minimax_h3_ref2va_experimental"]')?.textContent).toContain('Unavailable for production jobs');
    expect(host.querySelector('[data-testid="model-card-swap"]')?.textContent).toContain('SHA-256:');
    expect(api).toHaveBeenCalledWith('/models/hub');
  });

  it('requires review before starting an explicit download and exposes cancellation', async () => {
    await mount(<Settings activeTab="models" setActiveTab={() => {}} settings={{}} updateSetting={() => {}} onResetSettings={() => {}} />);
    const card = host.querySelector('[data-testid="model-card-h3"]')!;
    const downloadButton = card.querySelector('button')!;
    expect((downloadButton as HTMLButtonElement).disabled).toBe(true);
    const checkbox = card.querySelector('input[type="checkbox"]')!;
    await click(checkbox);
    const enabledButton = card.querySelector('button')!;
    expect((enabledButton as HTMLButtonElement).disabled).toBe(false);
    await click(enabledButton);
    expect(post).toHaveBeenCalledWith('/models/h3/downloads', { accepted_license: true });
    expect(host.textContent).toContain('Download queued');
    expect(host.textContent).toContain('Cancel download');
    await click(host.querySelector('[data-testid="model-card-h3"] button:last-child'));
    expect(api).toHaveBeenCalledWith('/models/downloads/download-1', { method: 'DELETE' });
  });

  it('keeps an initial load error visible without retrying in a loop', async () => {
    mockHub.failHub = true;
    await mount(<Settings activeTab="models" setActiveTab={() => {}} settings={{}} updateSetting={() => {}} onResetSettings={() => {}} />);
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 20)); });
    expect(host.textContent).toContain('local service unavailable');
    expect(vi.mocked(api).mock.calls.filter(([url]) => url === '/models/hub')).toHaveLength(1);
    expect(host.querySelector('[data-testid="refresh-models-btn"]')).toBeTruthy();
  });

  it('reconnects to server-side active downloads after leaving and reopening Models', async () => {
    function NavigableSettings() {
      const [tab, setTab] = React.useState('models');
      return <>
        <button onClick={() => setTab(tab === 'models' ? 'preferences' : 'models')}>Toggle models tab</button>
        <Settings activeTab={tab} setActiveTab={setTab} settings={{}} updateSetting={() => {}} onResetSettings={() => {}} />
      </>;
    }
    await mount(<NavigableSettings />);
    await click(host.querySelector('[data-testid="model-card-h3"] input[type="checkbox"]'));
    await click(host.querySelector('[data-testid="model-card-h3"] button'));
    expect(host.textContent).toContain('Download queued');
    await click(host.querySelector('button'));
    mockHub.activeDownloads = [{ download_id: 'download-1', model_id: 'h3', state: 'downloading',
      bytes_downloaded: 256, total_bytes: 1024, percent: 25, error_code: null }];
    await click(host.querySelector('button'));
    expect(host.textContent).toContain('Download downloading: 25.0%');
    expect(host.textContent).toContain('Cancel download');
  });
});
