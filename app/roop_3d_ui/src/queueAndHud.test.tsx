import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import PipelinePill, {
  resolveProvider,
  computeBottleneck,
  PIPELINE_STAGES,
} from './components/hud/PipelinePill';
import QueueDrawer, {
  normalizeJobState,
  QUEUE_STATES,
} from './components/queue/QueueDrawer';

describe('Hardware Diagnostics & HUD Logic (PipelinePill)', () => {
  it('correctly resolves hardware execution provider from telemetry', () => {
    // Null telemetry -> CPU Reference
    expect(resolveProvider(null)).toEqual({
      name: 'CPU Reference',
      type: 'cpu',
      color: '#8d9c9f',
      badge: 'CPU',
    });

    // TensorRT
    expect(
      resolveProvider({
        advertised_providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider'],
        gpus: [{ name: 'NVIDIA RTX 4090' }],
      })
    ).toEqual({
      name: 'NVIDIA TensorRT',
      type: 'tensorrt',
      color: '#10b981',
      badge: 'TRT',
    });

    // CUDA
    expect(
      resolveProvider({
        advertised_providers: ['CUDAExecutionProvider'],
        gpus: [{ name: 'NVIDIA RTX 3080' }],
      })
    ).toEqual({
      name: 'NVIDIA CUDA',
      type: 'cuda',
      color: '#00f5d4',
      badge: 'CUDA',
    });

    // DirectML
    expect(
      resolveProvider({
        advertised_providers: ['DmlExecutionProvider', 'CPUExecutionProvider'],
        gpus: [],
      })
    ).toEqual({
      name: 'DirectML (Windows DX12)',
      type: 'directml',
      color: '#38bdf8',
      badge: 'DML',
    });

    // ROCm
    expect(
      resolveProvider({
        advertised_providers: ['ROCMExecutionProvider'],
        gpus: [],
      })
    ).toEqual({
      name: 'AMD ROCm (HIP)',
      type: 'rocm',
      color: '#f43f5e',
      badge: 'ROCM',
    });

    // Apple Silicon / CoreML
    expect(
      resolveProvider({
        advertised_providers: ['CoreMLExecutionProvider'],
        gpus: [],
      })
    ).toEqual({
      name: 'Apple Silicon (Metal/ANE)',
      type: 'coreml',
      color: '#a855f7',
      badge: 'APPLE',
    });
  });

  it('identifies real-time pipeline bottlenecks based on VRAM and FPS', () => {
    // 1. Healthy / Optimal
    const healthyTelemetry = {
      gpus: [{ name: 'RTX 4090', free_vram_mib: 16000, total_vram_mib: 24576 }],
    };
    expect(computeBottleneck(healthyTelemetry, 30.5, 'blend')).toEqual({
      label: 'Optimal Pipeline Throughput',
      severity: 'optimal',
    });

    // 2. VRAM Bound (>90% used)
    const criticalVramTelemetry = {
      gpus: [{ name: 'RTX 3060', free_vram_mib: 1000, total_vram_mib: 12288 }],
    };
    expect(computeBottleneck(criticalVramTelemetry, 20.0, 'warp')).toEqual({
      label: 'VRAM Bound (>90% memory allocation)',
      severity: 'critical',
    });

    // 3. VRAM Pressure (>80% used)
    const warningVramTelemetry = {
      gpus: [{ name: 'RTX 3060', free_vram_mib: 2000, total_vram_mib: 12288 }],
    };
    expect(computeBottleneck(warningVramTelemetry, 22.0, 'detection')).toEqual({
      label: 'VRAM Pressure (>80% allocation)',
      severity: 'warning',
    });

    // 4. Low FPS Bottlenecks
    expect(computeBottleneck(healthyTelemetry, 4.2, 'restoration')).toEqual({
      label: 'Face Restoration Model Bound',
      severity: 'warning',
    });
    expect(computeBottleneck(healthyTelemetry, 5.0, 'warp')).toEqual({
      label: 'Affine Warp / Tensor Flow Bound',
      severity: 'warning',
    });
    expect(computeBottleneck(healthyTelemetry, 3.1, 'detection')).toEqual({
      label: 'Compute Bound (Low FPS)',
      severity: 'warning',
    });
  });

  it('renders PipelinePill collapsed and expanded without crashing', () => {
    const mockTelemetry = {
      advertised_providers: ['CUDAExecutionProvider'],
      gpus: [{ name: 'NVIDIA RTX 4090', free_vram_mib: 14336, total_vram_mib: 24576, driver_version: '550.54' }],
      logical_cpu_count: 16,
    };
    const mockActiveJob = {
      job_id: 'job_test_12345678',
      state: 'running',
      progress: {
        stage: 'warp',
        frame: 120,
        total_frames: 400,
        fps: 28.4,
      },
    };

    // Collapsed rendering
    const collapsedHtml = renderToStaticMarkup(
      <PipelinePill telemetry={mockTelemetry} activeJob={mockActiveJob} defaultExpanded={false} />
    );
    expect(collapsedHtml).toContain('pipeline-hud-container');
    expect(collapsedHtml).toContain('CUDA');
    expect(collapsedHtml).toContain('28.4 FPS');
    expect(collapsedHtml).toContain('WARP');

    // Expanded rendering
    const expandedHtml = renderToStaticMarkup(
      <PipelinePill telemetry={mockTelemetry} activeJob={mockActiveJob} defaultExpanded={true} />
    );
    expect(expandedHtml).toContain('pipeline-hud-panel');
    expect(expandedHtml).toContain('LIVE TELEMETRY HUD');
    expect(expandedHtml).toContain('NVIDIA CUDA');
    expect(expandedHtml).toContain('10.0 GB / 24.0 GB');
    expect(expandedHtml).toContain('PIPELINE THROUGHPUT');
    // Ensure all 5 stages exist in the expanded breadcrumb
    for (const stage of PIPELINE_STAGES) {
      expect(expandedHtml).toContain(stage.label);
    }
  });
});

describe('Persistent Queue Manager (QueueDrawer)', () => {
  it('normalizes internal job states into user-facing queue states', () => {
    expect(normalizeJobState('queued')).toBe(QUEUE_STATES.QUEUED);
    expect(normalizeJobState('preparing')).toBe(QUEUE_STATES.ANALYZING);
    expect(normalizeJobState('running')).toBe(QUEUE_STATES.PROCESSING);
    expect(normalizeJobState('succeeded')).toBe(QUEUE_STATES.COMPLETED);
    expect(normalizeJobState('failed')).toBe(QUEUE_STATES.FAILED);
    expect(normalizeJobState('cancelled')).toBe(QUEUE_STATES.CANCELLED);
    expect(normalizeJobState('cancelling')).toBe(QUEUE_STATES.CANCELLED);
    expect(normalizeJobState('paused')).toBe(QUEUE_STATES.PAUSED);
    expect(normalizeJobState('pausing')).toBe(QUEUE_STATES.PAUSED);
    expect(normalizeJobState('interrupted')).toBe(QUEUE_STATES.PAUSED);
  });

  it('renders QueueDrawer with job cards, states, and reordering controls', () => {
    const mockJobs = [
      {
        job_id: 'job_alpha_11111111',
        state: 'running',
        request: { target_path: 'inputs/clip_a.mp4' },
        progress: { stage: 'restoration', frame: 50, total_frames: 100, fps: 18.5 },
      },
      {
        job_id: 'job_beta_22222222',
        state: 'succeeded',
        request: { target_path: 'inputs/clip_b.mp4' },
        progress: { stage: 'completed', frame: 200, total_frames: 200, fps: 24.0 },
        output: { output_path: 'outputs/clip_b_swap.mp4' },
      },
      {
        job_id: 'job_gamma_33333333',
        state: 'failed',
        request: { target_path: 'inputs/clip_c.mp4' },
        progress: { stage: 'failed', frame: 12, total_frames: 500, fps: 0 },
        failure: { message: 'Out of memory during affine alignment' },
      },
    ];

    const html = renderToStaticMarkup(
      <QueueDrawer
        jobs={mockJobs}
        selectedJobId="job_alpha_11111111"
        isOpen={true}
      />
    );

    expect(html).toContain('queue-drawer-panel');
    expect(html).toContain('PERSISTENT QUEUE');
    expect(html).toContain('3 Local Jobs');
    expect(html).toContain('1 Processing');
    expect(html).toContain('1 Failed');

    // Job cards
    expect(html).toContain('clip_a.mp4');
    expect(html).toContain('PROCESSING');
    expect(html).toContain('clip_b.mp4');
    expect(html).toContain('COMPLETED');
    expect(html).toContain('clip_c.mp4');
    expect(html).toContain('FAILED');
    expect(html).toContain('Out of memory during affine alignment');

    // Controls
    expect(html).toContain('Duplicate');
    expect(html).toContain('Retry');
    expect(html).toContain('Cancel');
    expect(html).toContain('Output ↗');
  });

  it('renders an empty state note when no jobs are present', () => {
    const html = renderToStaticMarkup(
      <QueueDrawer jobs={[]} isOpen={true} />
    );
    expect(html).toContain('Render queue is empty');
  });
});
