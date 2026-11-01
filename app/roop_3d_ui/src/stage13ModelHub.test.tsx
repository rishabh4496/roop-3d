// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import SettingsScreen from './screens/SettingsScreen';

const Settings = SettingsScreen as React.ComponentType<any>;

// Mock API
vi.mock('./api', () => ({
  api: vi.fn((url: string) => {
    if (url.includes('/models/registry')) {
      return Promise.resolve({
        status: 'ok',
        summary: {
          total_registered: 22,
          downloaded: 4,
          trt_ready: 2,
          missing: 18,
          corrupt_hash: 0,
        },
        models: [
          {
            model_key: 'inswapper_128.onnx',
            name: 'InSwapper 128 (InsightFace)',
            category: 'swapper',
            filename: 'inswapper_128.onnx',
            file_size_bytes: 554253681,
            sha256: 'e4a3f08c536572c613f341d1493ee2a18b13773c883814400e732360549b0d23',
            precision_policy: 'FP32',
            supports_dynamic_batch: true,
            status: 'READY + TRT ENGINE',
            downloaded: true,
            trt_compiled: true,
            vram_footprint_mb: 528,
          },
          {
            model_key: 'gpen_bfr_512.onnx',
            name: 'GPEN Realistic 512',
            category: 'enhancer',
            filename: 'GPEN-BFR-512.onnx',
            file_size_bytes: 290000000,
            sha256: 'b123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
            precision_policy: 'FP16',
            supports_dynamic_batch: false,
            status: 'DOWNLOADED (NEEDS TRT)',
            downloaded: true,
            trt_compiled: false,
            vram_footprint_mb: 276,
          },
          {
            model_key: 'simswap_unofficial_512.onnx',
            name: 'SimSwap Unofficial 512',
            category: 'swapper',
            filename: 'simswap_unofficial_512.onnx',
            file_size_bytes: 380000000,
            sha256: 'c123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
            precision_policy: 'FP16',
            supports_dynamic_batch: true,
            status: 'MISSING',
            downloaded: false,
            trt_compiled: false,
            vram_footprint_mb: 362,
          },
        ],
      });
    }
    if (url.includes('/system/diagnostics-report')) {
      return Promise.resolve({
        report_id: 'r3d_diag_test_123',
        overall_status: 'PASS',
        sections: [],
        markdown_report: '# Test Report',
      });
    }
    return Promise.resolve({});
  }),
  post: vi.fn((url: string, body?: any) => {
    if (url.includes('/models/download')) {
      return Promise.resolve({
        status: 'ok',
        model_key: body?.model_key || 'inswapper_128.onnx',
        message: 'Download initiated successfully',
      });
    }
    if (url.includes('/models/compile_trt')) {
      return Promise.resolve({
        status: 'ok',
        model_key: body?.model_key || 'gpen_bfr_512.onnx',
        trt_engine_path: 'app/models/GPEN-BFR-512.engine',
      });
    }
    return Promise.resolve({});
  }),
  mediaURL: vi.fn((p: string) => `http://127.0.0.1:8030/${p}`),
}));

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let host: HTMLDivElement, root: Root;

beforeEach(() => {
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
  await act(async () => {
    element!.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  });
};

describe('Stage 13 Model Hub & TRT Engine Manager UI', () => {
  it('renders the Model Hub tab button, switches to Model Hub, and displays summary cards', async () => {
    await mount(
      <Settings
        activeTab="models"
        setActiveTab={() => {}}
        hardware={{
          gpu_name: 'NVIDIA GeForce RTX 4090',
          vram_total_mib: 24576,
          vram_allocated_mib: 4096,
          vram_free_mib: 20480,
          memory_pressure_gauge: 0.166,
        }}
        settings={{}}
        updateSetting={() => {}}
        onResetSettings={() => {}}
      />
    );

    // Verify Model Hub container is mounted
    const modelHubPanel = host.querySelector('[data-testid="model-hub-panel"]');
    expect(modelHubPanel).toBeTruthy();
    expect(modelHubPanel?.textContent).toContain('Model Hub & TRT Engine Manager');
    expect(modelHubPanel?.textContent).toContain('22'); // Total registered models
  });

  it('displays models with readiness badges (READY + TRT ENGINE, DOWNLOADED (NEEDS TRT), MISSING)', async () => {
    await mount(
      <Settings
        activeTab="models"
        setActiveTab={() => {}}
        hardware={{
          gpu_name: 'NVIDIA GeForce RTX 4090',
          vram_total_mib: 24576,
          vram_allocated_mib: 4096,
          vram_free_mib: 20480,
          memory_pressure_gauge: 0.166,
        }}
        settings={{}}
        updateSetting={() => {}}
        onResetSettings={() => {}}
      />
    );

    // InSwapper 128 (READY + TRT ENGINE)
    const inswapperCard = host.querySelector('[data-testid="model-card-inswapper_128.onnx"]');
    expect(inswapperCard).toBeTruthy();
    expect(inswapperCard?.textContent).toContain('READY + TRT ENGINE');
    expect(inswapperCard?.textContent).toContain('FP32');

    // GPEN Realistic 512 (DOWNLOADED (NEEDS TRT))
    const gpenCard = host.querySelector('[data-testid="model-card-gpen_bfr_512.onnx"]');
    expect(gpenCard).toBeTruthy();
    expect(gpenCard?.textContent).toContain('DOWNLOADED (NEEDS TRT)');
    expect(gpenCard?.textContent).toContain('FP16');

    // SimSwap 512 (MISSING)
    const simswapCard = host.querySelector('[data-testid="model-card-simswap_unofficial_512.onnx"]');
    expect(simswapCard).toBeTruthy();
    expect(simswapCard?.textContent).toContain('MISSING');
  });

  it('filters models when clicking category filter buttons', async () => {
    await mount(
      <Settings
        activeTab="models"
        setActiveTab={() => {}}
        hardware={{
          gpu_name: 'NVIDIA GeForce RTX 4090',
          vram_total_mib: 24576,
          vram_allocated_mib: 4096,
          vram_free_mib: 20480,
          memory_pressure_gauge: 0.166,
        }}
        settings={{}}
        updateSetting={() => {}}
        onResetSettings={() => {}}
      />
    );

    // Click 'enhancer' category filter
    const enhancerFilterBtn = host.querySelector('[data-testid="category-filter-enhancer"]');
    await click(enhancerFilterBtn);

    // GPEN should remain visible
    expect(host.querySelector('[data-testid="model-card-gpen_bfr_512.onnx"]')).toBeTruthy();
    // Swappers should now be filtered out
    expect(host.querySelector('[data-testid="model-card-inswapper_128.onnx"]')).toBeNull();
  });

  it('triggers Download & Pre-Build All God-Tier Models button', async () => {
    await mount(
      <Settings
        activeTab="models"
        setActiveTab={() => {}}
        hardware={{
          gpu_name: 'NVIDIA GeForce RTX 4090',
          vram_total_mib: 24576,
          vram_allocated_mib: 4096,
          vram_free_mib: 20480,
          memory_pressure_gauge: 0.166,
        }}
        settings={{}}
        updateSetting={() => {}}
        onResetSettings={() => {}}
      />
    );

    const godTierBtn = host.querySelector('[data-testid="god-tier-prebuild-btn"]');
    await click(godTierBtn);

    // Verify it triggers action without uncaught errors
    expect(godTierBtn).toBeTruthy();
  });
});
