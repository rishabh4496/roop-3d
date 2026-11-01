// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import PreFlightLinterModal from './components/diagnostics/PreFlightLinterModal';
import BottleneckFlamegraph, { PIPELINE_STAGES } from './components/diagnostics/BottleneckFlamegraph';
import SettingsScreen from './screens/SettingsScreen';

const PreFlightLinter = PreFlightLinterModal as React.ComponentType<any>;
const Flamegraph = BottleneckFlamegraph as React.ComponentType<any>;
const Settings = SettingsScreen as React.ComponentType<any>;

// Mock API
vi.mock('./api', () => ({
  api: vi.fn((url: string) => {
    if (url.includes('/system/diagnostics-report')) {
      return Promise.resolve({
        report_timestamp: '2026-11-01T00:00:00Z',
        app_version: '0.1.0',
        sections: [
          { section: 1, title: 'Python & Environment Integrity', status: 'PASS', python_version: '3.10.12' },
          { section: 2, title: 'Host Hardware Architecture', status: 'PASS', primary_gpu: 'NVIDIA RTX 4090' },
          { section: 3, title: 'ONNX Runtime Providers & Acceleration', status: 'PASS', ort_version: '1.20.1' },
          { section: 4, title: 'TensorRT Engine Cache & Quantization State', status: 'VERIFIED_LOCAL', cached_engines: 2 },
          { section: 5, title: 'FFmpeg Media & Hardware Codec Matrix', status: 'PASS', nvenc_encoders: ['h264_nvenc'] },
          { section: 6, title: 'Model Registry & SHA-256 Checksums', status: 'PASS', registered_models_count: 8 },
          { section: 7, title: 'Face Detection & ArcFace Alignment Pipeline', status: 'PASS' },
          { section: 8, title: '3D Volumetric Mesh & LivePortrait Retargeting', status: 'PASS' },
          { section: 9, title: 'Identity Transfer & Inswapper 128 Graph', status: 'PASS' },
          { section: 10, title: 'Compositor, XSeg Occlusion & Alpha Seam Blending', status: 'PASS' },
          { section: 11, title: 'Face Restoration Adapters (GPEN-512 / CodeFormer)', status: 'PASS' },
          { section: 12, title: 'VRAM Governor, Reservation & Profile Admission', status: 'PASS' },
          { section: 13, title: 'Temporal Landmark Smoothing & Flicker Control', status: 'PASS' },
          { section: 14, title: 'Strict Offline Verification & Boundary Isolation (Zero CDN Calls)', status: 'PASS' },
        ],
        markdown_report: '# Test Report',
      });
    }
    return Promise.resolve({});
  }),
  post: vi.fn((url: string, body: any) => {
    if (url.includes('/system/preflight-lint')) {
      return Promise.resolve({
        status: 'PASS',
        ready_to_queue: true,
        checks: [
          {
            id: '3d_pose_coverage',
            name: '3D Pose Coverage & Orientation Sphere',
            status: 'PASS',
            message: '3D pose orientation coverage validated.',
            details: [],
          },
          {
            id: 'vram_headroom',
            name: 'VRAM Headroom & Allocation Budget',
            status: 'PASS',
            message: 'Peak VRAM headroom nominal.',
            estimated_vram_mib: 3072,
            available_vram_mib: 24576,
            details: [],
          },
          {
            id: 'out_of_frame_seams',
            name: 'Out-of-Frame & Seam Boundary Baseline',
            status: 'PASS',
            message: 'Bounding boxes within frame.',
            details: [],
          },
          {
            id: 'output_encoder_storage',
            name: 'Output Encoder & Storage Permissions',
            status: 'PASS',
            message: 'NVENC available and directory writable.',
            details: [],
          },
        ],
        timestamp: '2026-11-01T00:00:00Z',
      });
    }
    if (url.includes('/system/benchmark')) {
      return Promise.resolve({
        tested_frames: body.frames || 100,
        results: [
          { provider: 'CPUExecutionProvider', threads: 1, latency_ms: 45.0, fps: 22.2, peak_vram_mib: 0, status: 'VERIFIED_LOCAL' },
          { provider: 'CPUExecutionProvider', threads: 4, latency_ms: 22.5, fps: 44.4, peak_vram_mib: 0, status: 'VERIFIED_LOCAL' },
          { provider: 'CUDAExecutionProvider', threads: 4, latency_ms: 7.42, fps: 134.8, peak_vram_mib: 2480, status: 'VERIFIED_LOCAL', hardware: 'NVIDIA RTX 4090' },
          { provider: 'TensorrtExecutionProvider', threads: 4, latency_ms: 4.88, fps: 204.9, peak_vram_mib: 2150, status: 'VERIFIED_LOCAL', hardware: 'NVIDIA RTX 4090' },
        ],
        recommended_provider: 'TensorrtExecutionProvider',
        recommended_threads: 4,
        recommended_fps: 204.9,
        recommendation_reason: 'Optimal throughput achieved using TensorrtExecutionProvider with 4 threads (204.9 FPS).',
        timestamp: '2026-11-01T00:00:00Z',
      });
    }
    return Promise.resolve({});
  }),
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

describe('Pre-Flight Render Linter Modal', () => {
  it('renders pre-flight checks and dispatches queue confirmation', async () => {
    const onConfirmQueue = vi.fn();
    const onClose = vi.fn();

    await mount(
      <PreFlightLinter
        isOpen={true}
        onClose={onClose}
        onConfirmQueue={onConfirmQueue}
        targetPath="video.mp4"
        inspectionId="insp_123"
        faceMappings={[{ track_id: 'track_0', faceset_id: 'alice' }]}
      />
    );

    // Initial render has modal
    expect(host.querySelector('.preflight-linter-modal')).toBeTruthy();
    expect(host.textContent).toContain('Pre-Flight Render Linter');

    // Wait for async checks to render
    await act(async () => {
      await new Promise(r => setTimeout(r, 50));
    });

    expect(host.textContent).toContain('3D Pose Coverage & Orientation Sphere');
    expect(host.textContent).toContain('VRAM Headroom & Allocation Budget');
    expect(host.textContent).toContain('Out-of-Frame & Seam Boundary Baseline');
    expect(host.textContent).toContain('Output Encoder & Storage Permissions');
    expect(host.textContent).toContain('Estimated Peak VRAM: 3072 MiB');

    // Toggle skip linter checkbox
    const skipCheckbox = host.querySelector<HTMLInputElement>('.skip-linter-label input');
    expect(skipCheckbox).toBeTruthy();
    await click(skipCheckbox);

    // Click Proceed & Queue Render
    const proceedBtn = Array.from(host.querySelectorAll('button')).find(b => b.textContent?.includes('Proceed & Queue Render'));
    expect(proceedBtn).toBeTruthy();
    await click(proceedBtn!);

    expect(onConfirmQueue).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(sessionStorage.getItem('roop3d_skip_linter')).toBe('true');
  });
});

describe('Per-Frame Bottleneck Flamegraph & Temporal Controls', () => {
  it('shows the seven measured stages from live timings and flags the slowest as the bottleneck', async () => {
    const timings = { decode: 2.5, detection: 6.5, pose_tps: 5.0, inswapper: 9.8, masking: 4.0, restorer: 15.2, encode: 3.5 };

    await mount(<Flamegraph timings={timings} />);

    expect(host.querySelector('.bottleneck-flamegraph-container')).toBeTruthy();
    expect(host.textContent).toContain('Pipeline Bottleneck Flamegraph');
    expect(PIPELINE_STAGES.map(stage => stage.label)).toEqual(['Decode', 'Detection', '3D Pose / TPS', 'Inswapper', 'Masking', 'Restorer', 'Encode']);
    for (const stage of PIPELINE_STAGES) {
      expect(host.querySelector(`[data-testid="flamegraph-stage-${stage.id}"]`)).toBeTruthy();
    }
    expect(host.querySelector('[data-testid="flamegraph-stage-restorer"]')!.textContent).toContain('15.2');
    expect(host.querySelector('[data-testid="flamegraph-stage-restorer"]')!.className).toContain('bottleneck');
    expect(host.textContent).toContain('46.5 ms'); // measured total = sum of the seven stages
    expect(host.textContent).toContain('BOTTLENECK');
    expect(host.textContent).not.toContain('not measured');
  });

  it('never invents numbers: with no measurements every stage reads "not measured"', async () => {
    await mount(<Flamegraph />);
    expect(host.querySelector('[data-testid="flamegraph-source"]')!.textContent).toContain('No measured timings yet');
    expect(host.querySelectorAll('.stage-ms.muted')).toHaveLength(7);
    expect(host.textContent).not.toContain('BOTTLENECK');
    expect(host.textContent).not.toMatch(/\d+(\.\d+)? ms \(/);
  });

  it('prefers the running job’s per-frame timings and leaves unmeasured stages blank', async () => {
    const job = { job_id: 'abcdef1234567890', state: 'running', progress: { frame: 12, stage_timings_ms: { decode: 1.2, inswapper: 8, encode: 2 } } };
    await mount(<Flamegraph activeJob={job} latency={{ detection_ms: 99 }} />);
    expect(host.querySelector('[data-testid="flamegraph-source"]')!.textContent).toContain('job abcdef12');
    expect(host.querySelector('[data-testid="flamegraph-stage-inswapper"]')!.textContent).toContain('8');
    expect(host.querySelector('[data-testid="flamegraph-stage-detection"]')!.textContent).toContain('not measured'); // preview value ignored
    expect(host.querySelector('[data-testid="flamegraph-stage-restorer"]')!.textContent).toContain('not measured');
  });

  it('maps preview-capture latency onto detection, pose/TPS and swap only', async () => {
    await mount(<Flamegraph latency={{ detection_ms: 6, swap_ms: 9, retarget_ms: 3, projection_ms: 1.5, total_ms: 40 }} />);
    expect(host.querySelector('[data-testid="flamegraph-source"]')!.textContent).toContain('Preview capture');
    expect(host.querySelector('[data-testid="flamegraph-stage-pose_tps"]')!.textContent).toContain('4.5');
    expect(host.querySelector('[data-testid="flamegraph-stage-masking"]')!.textContent).toContain('not measured');
  });

  it('switches to Temporal Tracking & Flicker Control tab and adjusts parameters', async () => {
    const onTemporalChange = vi.fn();
    await mount(<Flamegraph onTemporalChange={onTemporalChange} />);

    // Click temporal tab
    const tabs = host.querySelectorAll('.tab-btn');
    const temporalTab = Array.from(tabs).find(t => t.textContent?.includes('Temporal Tracking'));
    expect(temporalTab).toBeTruthy();
    await click(temporalTab!);

    expect(host.querySelector('[data-testid="temporal-controls-panel"]')).toBeTruthy();
    expect(host.textContent).toContain('Temporal Face Tracking & Anti-Flicker Architecture');

    // Change filter select to kalman
    const select = host.querySelector<HTMLSelectElement>('#filter-select');
    expect(select).toBeTruthy();
    await act(async () => {
      select!.value = 'kalman';
      select!.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(onTemporalChange).toHaveBeenCalledWith(expect.objectContaining({ filter: 'kalman' }));

    // Toggle ROI reuse
    const checkboxes = host.querySelectorAll<HTMLInputElement>('.checkbox-label input');
    expect(checkboxes.length).toBeGreaterThan(0);
    await click(checkboxes[0]);
    expect(onTemporalChange).toHaveBeenCalledWith(expect.objectContaining({ roiReuse: false }));
  });
});

describe('Settings & Diagnostics Suite Screen', () => {
  const caps = {
    rendering_worker_available: true,
    legal_parameter_bounds: { properties: {} },
    workstation: {
      features: {},
      controls: {},
      read_only_parameters: {},
      disabled_parameters: {},
      job_states: [],
    },
    models: [],
  };
  const config = {
    execution_mode: 'swap',
    precision_policy: 'fp32',
    port: 8030,
    remote_processing_enabled: false,
  };

  it('renders preferences and switches to diagnostics suite', async () => {
    await mount(
      <Settings
        caps={caps as any}
        config={config as any}
        saveSettings={vi.fn()}
      />
    );

    expect(host.querySelector('[data-testid="settings-screen"]')).toBeTruthy();
    expect(host.querySelector('[data-testid="tab-preferences"]')).toBeTruthy();
    const diagTab = host.querySelector<HTMLElement>('[data-testid="tab-diagnostics"]');
    expect(diagTab).toBeTruthy();

    // Switch to diagnostics
    await click(diagTab);

    // Wait for report data to load
    await act(async () => {
      await new Promise(r => setTimeout(r, 50));
    });

    expect(host.querySelector('[data-testid="diagnostics-suite"]')).toBeTruthy();
    expect(host.querySelector('[data-testid="environment-health-card"]')).toBeTruthy();
    expect(host.querySelector('[data-testid="report-sections-card"]')).toBeTruthy();
    expect(host.querySelector('[data-testid="offline-verification-card"]')).toBeTruthy();

    // Check offline compliance features
    expect(host.textContent).toContain('Zero External CDN Calls');
    expect(host.textContent).toContain('Localhost Loopback Network Guard');

    // Check all 14 sections are rendered
    for (let s = 1; s <= 14; s++) {
      expect(host.querySelector(`[data-testid="report-section-${s}"]`)).toBeTruthy();
    }

    // Run benchmark runner
    const runBenchBtn = host.querySelector<HTMLElement>('[data-testid="run-benchmark-btn"]');
    expect(runBenchBtn).toBeTruthy();
    await click(runBenchBtn);

    await act(async () => {
      await new Promise(r => setTimeout(r, 50));
    });

    expect(host.textContent).toContain('Auto-Recommended Optimal Configuration');
    expect(host.textContent).toContain('TensorrtExecutionProvider');
    expect(host.textContent).toContain('204.9 FPS');
  });
});
