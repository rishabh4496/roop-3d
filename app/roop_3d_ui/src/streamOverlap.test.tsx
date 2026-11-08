// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import StreamOverlapPanel from './components/diagnostics/StreamOverlapPanel';
import { buildLaneModel, summarizePools, summarizeTransfers } from './components/diagnostics/lanesModel';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

// Field names mirror AsyncPipeline.snapshot() / GET /api/v1/system/pipeline exactly.
const pipeline = (extra: Record<string, unknown> = {}) => ({
  id: 'p1', name: 'render', running: true, frames_out: 12, fps: 31.5, bottleneck: 'swap',
  lanes: {
    window_ms: 100,
    stages: {
      decode: [[10, 0, 10], [11, 10, 20]], track: [[10, 10, 30], [11, 30, 50]], swap: [[10, 30, 70], [11, 70, 100]],
      composite: [[10, 70, 90]], write: [[10, 90, 95]],
    },
  },
  overlap_wall: { ge1: 1, ge2: 0.62, ge3: 0.18, mean_active: 1.7 },
  overlap_gpu_streams: null,
  gpu_stream_note: 'No CUDA stream windows were recorded: torch.cuda streams need PyTorch, which this environment does not use, so only host-thread overlap is measured.',
  pools: [{ backend: 'cuda_pinned', count: 10, in_flight: 4, bytes: 10 * 1024 * 1024, waits: 3, blocked_s: 0.42 }],
  transfers: {
    measured: true, scope: 'onnxruntime-boundary', installed_on: '1.23.2', gpu_runs: 36,
    total: { h2d_count: 36, h2d_bytes: 36 * 1048576, d2h_count: 36, d2h_bytes: 36 * 1048576, intermediate_copies: 72, boundary_copies: 0 },
    per_frame: { h2d_tensors: 3, d2h_tensors: 3, h2d_bytes: 3145728, d2h_bytes: 3145728, intermediate_copies: 6 },
    by_stage: { swap: { h2d_count: 24, d2h_count: 24, h2d_bytes: 24 * 1048576, d2h_bytes: 24 * 1048576 }, composite: { h2d_count: 12, d2h_count: 12, h2d_bytes: 12 * 1048576, d2h_bytes: 12 * 1048576 } },
    not_counted: ['torch / CV-CUDA / OpenCV-CUDA copies', 'NVENC / NVDEC internal transfers'], zero_intermediate_copies: false,
  },
  ...extra,
});
const payload = (p: unknown = pipeline()) => ({
  pipelines: [p], hardware: { backend: 'nvml', latest: { pcie_tx_mb_s: 812, pcie_rx_mb_s: 640, vram_used_mib: 5120, vram_total_mib: 12282 } },
  pcie_note: 'NVML reports PCIe throughput over a ~20 ms window; values are instantaneous, not averaged.',
});

describe('lane model', () => {
  it('lays every stage on one shared time axis in execution order with the real stage names', () => {
    const model = buildLaneModel(pipeline())!;
    expect(model.rows.map(row => row.name)).toEqual(['decode', 'track', 'swap', 'composite', 'write']);
    expect(model.rows.find(row => row.name === 'composite')!.label).toContain('UltraMax');
    expect(model.rows.find(row => row.name === 'write')!.label).toContain('PNG checkpoint');
    const swap = model.rows.find(row => row.name === 'swap')!;
    expect(swap.segments[0]).toMatchObject({ frame: 10, left: 30, width: 40 });
    expect(swap.busyMs).toBe(70);
    expect(model.frames).toBe(2);
    expect(model.overlap).toEqual({ two: 62, three: 18, mean: 1.7 });
  });

  it('draws nothing until a frame has been timed', () => {
    expect(buildLaneModel(null)).toBeNull();
    expect(buildLaneModel(pipeline({ lanes: { window_ms: 0, stages: {} } }))).toBeNull();
  });

  it('keeps an unknown stage and says CUDA stream windows were not measured instead of inventing them', () => {
    const model = buildLaneModel(pipeline({ lanes: { window_ms: 10, stages: { decode: [[0, 0, 5]], custom: [[0, 5, 9]] } } }))!;
    expect(model.rows.map(row => row.name)).toEqual(['decode', 'custom']);
    expect(model.gpuStreams).toBeNull();
    expect(model.gpuNote).toContain('PyTorch');
  });

  it('reports CUDA stream overlap only when the server measured it', () => {
    const model = buildLaneModel(pipeline({ overlap_gpu_streams: { ge1: 1, ge2: 0.4, ge3: 0.1, mean_active: 1.2 } }))!;
    expect(model.gpuStreams).toEqual({ two: 40, three: 10 });
    expect(model.gpuNote).toBeNull();
  });
});

describe('transfer summary', () => {
  it('reports copies between models as copies, never as zero-copy', () => {
    const summary = summarizeTransfers(pipeline().transfers)!;
    expect(summary.level).toBe('warn');
    expect(summary.verdict).toContain('6 tensor copies per frame');
    expect(summary.verdict).toContain('not zero-copy');
    expect(summary.perFrame).toEqual({ h2d: 3, d2h: 3, h2dMiB: 3, d2hMiB: 3 });
    expect(summary.stages.find(stage => stage.name === 'swap')).toMatchObject({ h2d: 24, d2h: 24, h2dMiB: 24 });
  });

  it('only says no intermediate copies when GPU runs were observed and none crossed PCIe', () => {
    const clean = { measured: true, scope: 'onnxruntime-boundary', gpu_runs: 50, total: { h2d_count: 1, h2d_bytes: 1, d2h_count: 1, d2h_bytes: 1, intermediate_copies: 0, boundary_copies: 2 }, by_stage: {}, zero_intermediate_copies: true };
    const summary = summarizeTransfers(clean)!;
    expect(summary.level).toBe('ok');
    expect(summary.verdict).toContain('No intermediate host↔device copies');
    expect(summary.verdict).toContain('50 GPU run(s)');
  });

  it('treats "nothing observed" as unknown, and an inactive ledger as inactive', () => {
    const unknown = summarizeTransfers({ measured: true, scope: 's', gpu_runs: 0, total: {}, by_stage: {}, zero_intermediate_copies: null })!;
    expect(unknown.level).toBe('idle');
    expect(unknown.verdict).toContain('unknown (not zero)');
    expect(summarizeTransfers({ measured: false, scope: 's', total: {}, by_stage: {} })!.verdict).toContain('not active');
    expect(summarizeTransfers(null)).toBeNull();
  });

  it('describes the ring buffer as page-locked host memory with its real occupancy', () => {
    const [pool] = summarizePools(pipeline().pools as never);
    expect(pool).toMatchObject({ backend: 'cuda_pinned', inFlight: 4, count: 10, fill: .4, mib: 10, waits: 3, blockedS: .42, pinned: true });
    expect(summarizePools(undefined)).toEqual([]);
  });
});

describe('StreamOverlapPanel', () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); });
  const text = (id: string) => host.querySelector(`[data-testid="${id}"]`)?.textContent ?? '';

  it('renders measured lanes, overlap, the PCIe link, copy verdict and ring buffer from one payload', () => {
    act(() => root.render(<StreamOverlapPanel metrics={payload()} />));
    expect(host.querySelectorAll('[data-testid^="lane-"]:not([data-testid="lanes-empty"])')).toHaveLength(5);
    expect(text('lane-swap')).toContain('Stage 2 · 3D swap');
    expect(text('overlap-line')).toContain('62%');
    expect(text('gpu-stream-note')).toContain('not measured');
    expect(host.textContent).toContain('812');                                 // PCIe TX MB/s from NVML
    expect(text('transfer-verdict')).toContain('6 tensor copies per frame');
    expect(text('transfer-totals')).toContain('onnxruntime-boundary');
    expect(text('not-counted')).toContain('NVENC / NVDEC internal transfers');
    expect(text('pool-row')).toContain('4/10 slots in flight');
    expect(host.textContent).toContain('page-locked host memory (not VRAM)');
  });

  it('states plainly when nothing has run: no lanes drawn, no pool, copies unknown', () => {
    const idle = pipeline({ running: false, frames_out: 0, lanes: { window_ms: 0, stages: {} }, pools: [], overlap_wall: null,
      transfers: { measured: true, scope: 'onnxruntime-boundary', gpu_runs: 0, total: {}, by_stage: {}, zero_intermediate_copies: null } });
    act(() => root.render(<StreamOverlapPanel metrics={payload(idle)} />));
    expect(text('lanes-empty')).toContain('Nothing is drawn');
    expect(text('pools-empty')).toContain('No frame pool');
    expect(text('transfer-verdict')).toContain('unknown (not zero)');
  });
});
