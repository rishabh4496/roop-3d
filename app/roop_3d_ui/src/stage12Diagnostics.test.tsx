// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import SettingsScreen from './screens/SettingsScreen';
import VfxExportPanel, { DEFAULT_EXPORT } from './components/export/VfxExportPanel';

const Settings = SettingsScreen as React.ComponentType<any>;

// Mock API
vi.mock('./api', () => ({
  api: vi.fn((url: string) => {
    if (url.includes('/system/diagnostics-report')) {
      return Promise.resolve({
        report_id: 'r3d_diag_test_123',
        report_timestamp: '2026-11-02T00:00:00Z',
        version: '0.1.0',
        overall_status: 'PASS',
        sections: [
          { section: 1, title: 'Python & Environment Integrity', status: 'PASS', python_version: '3.10.20', app_env_status: 'VERIFIED_DIR' },
          { section: 2, title: 'Host Hardware Architecture', status: 'PASS', primary_gpu: 'NVIDIA RTX 4090', memory_pressure_gauge: 0.45 },
          { section: 3, title: 'ONNX Runtime Providers & Acceleration', status: 'PASS', ort_version: '1.20.1', dll_alignment: 'ALIGNED' },
          { section: 4, title: 'TensorRT Engine Cache & Quantization State', status: 'VERIFIED_LOCAL', cached_engines: 3, engine_cache_validity: 'VALID' },
          { section: 5, title: 'FFmpeg Media & Hardware Codec Matrix', status: 'PASS', nvenc_encoders: ['h264_nvenc'] },
          { section: 6, title: 'Model Registry & SHA-256 Checksums', status: 'PASS', registered_models_count: 8, models_dir_status: 'VERIFIED_DIR' },
          { section: 7, title: 'Face Detection & ArcFace Alignment Pipeline', status: 'PASS', prepass_cache_hits: 12 },
          { section: 8, title: '3D Volumetric Mesh & LivePortrait Retargeting', status: 'PASS', stage0_matrix_status: 'POPULATED' },
          { section: 9, title: 'Identity Transfer & Inswapper 128 Graph', status: 'PASS' },
          { section: 10, title: 'Compositor, XSeg Occlusion & Alpha Seam Blending', status: 'PASS', watchdog_seam_logs: [] },
          { section: 11, title: 'Face Restoration Adapters (GPEN-512 / CodeFormer)', status: 'PASS' },
          { section: 12, title: 'VRAM Governor, Reservation & Profile Admission', status: 'PASS', memory_pressure_gauge: 0.45 },
          { section: 13, title: 'Temporal Landmark Smoothing & Flicker Control', status: 'PASS', per_node_latency_breakdown: { decode_ms: 2.5, detection_ms: 6.5, inswapper_ms: 9.8 } },
          { section: 14, title: 'Strict Offline Verification & Boundary Isolation (Zero CDN Calls)', status: 'PASS', app_facesets_status: 'VERIFIED_DIR' },
        ],
        markdown_report: '# Test Report',
      });
    }
    if (url.includes('/exports/job_1/timeline?format=xml')) {
      return Promise.resolve('<?xml version="1.0" encoding="UTF-8"?><xmeml version="5"></xmeml>');
    }
    if (url.includes('/exports/job_1/timeline')) {
      return Promise.resolve({
        name: 'Test Plan',
        frameRate: '25/1',
        frameCount: 100,
        sourceUrl: 'file:///source.mov',
        swappedUrl: 'file:///swapped.mov',
        ranges: [[0, 50]],
      });
    }
    return Promise.resolve({});
  }),
  post: vi.fn((url: string) => {
    if (url.includes('/system/purge-vram')) {
      return Promise.resolve({
        status: 'ok',
        freed_vram_mib: 512,
        available_vram_mib: 18432,
        total_vram_mib: 24576,
        gpu_name: 'NVIDIA RTX 4090',
        message: 'Idle VRAM cache purged and inactive sessions successfully unloaded.',
      });
    }
    if (url.includes('/system/check-compatibility')) {
      return Promise.resolve({
        status: 'UNVERIFIED',
        mode: 'offline',
        message: 'Offline-safe operation active. Remote compatibility is UNVERIFIED.',
        offline_compliant: true,
        zero_external_calls: true,
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
  await act(async () => (element as HTMLElement).click());
};

describe('Stage 12 Diagnostics & Environment-Health Suite UI', () => {
  const caps = {
    rendering_worker_available: true,
    legal_parameter_bounds: { properties: {} },
    workstation: { features: {}, controls: {}, read_only_parameters: {}, disabled_parameters: {}, job_states: [] },
    models: [],
  };
  const config = { execution_mode: 'swap', precision_policy: 'fp32', port: 8030, remote_processing_enabled: false };

  it('renders Live GPU/VRAM HUD with memory pressure gauge and purge button', async () => {
    await mount(
      <Settings
        caps={caps as any}
        config={config as any}
        saveSettings={vi.fn()}
      />
    );

    // Switch to diagnostics tab
    const diagTab = host.querySelector<HTMLElement>('[data-testid="tab-diagnostics"]');
    await click(diagTab);

    await act(async () => {
      await new Promise(r => setTimeout(r, 60));
    });

    // Check Live GPU / VRAM HUD card exists
    const hudCard = host.querySelector('[data-testid="gpu-vram-hud-card"]');
    expect(hudCard).toBeTruthy();
    expect(hudCard!.textContent).toContain('Live GPU / VRAM / CPU Telemetry HUD');
    expect(hudCard!.textContent).toContain('45% Utilized');
    expect(hudCard!.textContent).toContain('NOMINAL');

    // Click Purge Idle VRAM button
    const purgeBtn = host.querySelector<HTMLElement>('[data-testid="purge-vram-btn"]');
    expect(purgeBtn).toBeTruthy();
    await click(purgeBtn);

    await act(async () => {
      await new Promise(r => setTimeout(r, 50));
    });

    expect(hudCard!.textContent).toContain('Freed 512 MiB VRAM');
  });

  it('renders Standing Environment-Health Card with directory validations', async () => {
    await mount(
      <Settings
        caps={caps as any}
        config={config as any}
        saveSettings={vi.fn()}
      />
    );

    const diagTab = host.querySelector<HTMLElement>('[data-testid="tab-diagnostics"]');
    await click(diagTab);

    await act(async () => {
      await new Promise(r => setTimeout(r, 60));
    });

    const envCard = host.querySelector('[data-testid="environment-health-card"]');
    expect(envCard).toBeTruthy();
    expect(envCard!.textContent).toContain('Standing Environment-Health Card');
    expect(envCard!.textContent).toContain('app/env Virtualenv Directory');
    expect(envCard!.textContent).toContain('app/models Registry Directory');
    expect(envCard!.textContent).toContain('app/facesets Library Directory');
    expect(envCard!.textContent).toContain('CUDA / cuDNN / TensorRT DLL Alignment');
    expect(envCard!.textContent).toContain('TensorRT .engine Cache Validity');
  });

  it('performs Read-Only Offline-Safe Compatibility Check reporting UNVERIFIED cleanly', async () => {
    await mount(
      <Settings
        caps={caps as any}
        config={config as any}
        saveSettings={vi.fn()}
      />
    );

    const diagTab = host.querySelector<HTMLElement>('[data-testid="tab-diagnostics"]');
    await click(diagTab);

    await act(async () => {
      await new Promise(r => setTimeout(r, 60));
    });

    const compatBadge = host.querySelector('[data-testid="compatibility-status-badge"]');
    expect(compatBadge).toBeTruthy();
    expect(compatBadge!.textContent).toContain('UNVERIFIED');

    const checkBtn = host.querySelector<HTMLElement>('[data-testid="check-compatibility-btn"]');
    expect(checkBtn).toBeTruthy();
    await click(checkBtn);

    await act(async () => {
      await new Promise(r => setTimeout(r, 50));
    });

    expect(host.querySelector('[data-testid="compatibility-status-badge"]')!.textContent).toContain('UNVERIFIED');
  });

  it('renders VFX Export Panel with NLE interchange buttons including XML', async () => {
    const jobs = [
      {
        job_id: 'job_1',
        state: 'succeeded',
        request: { target_path: 'video.mp4' },
        output: { output_path: 'output.mp4', baked_video: true, passes: { alpha: 'alpha.mkv' } },
      },
    ];

    await mount(
      <VfxExportPanel
        value={DEFAULT_EXPORT as any}
        onChange={vi.fn()}
        onQueue={vi.fn()}
        jobs={jobs as any}
      />
    );

    expect(host.textContent).toContain('NLE timeline interchange');
    expect(host.textContent).toContain('Final Cut Pro XML (.fcpxml)');
    expect(host.textContent).toContain('Premiere / FCP 7 XML (.xml)');
    expect(host.textContent).toContain('CMX 3600 EDL (.edl)');
  });
});
