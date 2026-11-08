// Pure shaping of the /api/v1/system/pipeline payload into the stream-overlap view.
// Every figure is a server measurement. Where the server could not measure something (CUDA stream windows without PyTorch,
// copies before a GPU run, PCIe counters without NVML) the model carries an explicit `null` plus the reason, never a default.

const finite = value => (typeof value === 'number' && Number.isFinite(value) ? value : null);
const round = (value, digits = 1) => (finite(value) === null ? null : Number(value.toFixed(digits)));

/** The reference render graph's stages (roop_3d/async_pipeline/reference.py), in execution order. */
export const LANE_ORDER = ['decode', 'track', 'swap', 'composite', 'write'];
export const LANES = {
  decode: { label: 'Decode', hw: 'NVDEC / software decoder', color: '#38bdf8' },
  track: { label: 'Stage 1 · 3D pose & tracking', hw: 'Detector, landmarks, 6DoF pose', color: '#a78bfa' },
  swap: { label: 'Stage 2 · 3D swap', hw: 'Swapper model, mesh reenactment', color: '#f43f5e' },
  composite: { label: 'Stage 3 · Composite + restorer (UltraMax)', hw: 'Mask, relight, restorer, blend', color: '#f59e0b' },
  write: { label: 'Frame write (PNG checkpoint)', hw: 'Per-frame checkpoint; the final NVENC encode runs after the last frame, outside this graph', color: '#06b6d4' },
};

/** Lanes for the most recent frames: one row per stage on a shared time axis. null when no frame has been timed. */
export function buildLaneModel(pipeline) {
  const lanes = pipeline?.lanes;
  const total = finite(lanes?.window_ms);
  if (!lanes?.stages || total === null || total <= 0) return null;
  const names = [...LANE_ORDER.filter(name => name in lanes.stages), ...Object.keys(lanes.stages).filter(name => !LANE_ORDER.includes(name))];
  const rows = names.map(name => {
    const meta = LANES[name] ?? { label: name, hw: '', color: '#94a3b8' };
    const segments = (lanes.stages[name] ?? []).map(([frame, start, end]) => ({
      frame, start, end, left: (start / total) * 100, width: Math.max(((end - start) / total) * 100, 0.2),
    }));
    return { name, ...meta, segments, busyMs: round(segments.reduce((sum, item) => sum + (item.end - item.start), 0), 1) };
  });
  const frames = [...new Set(rows.flatMap(row => row.segments.map(segment => segment.frame)))];
  return {
    windowMs: round(total, 1), rows, frames: frames.length,
    overlap: pipeline.overlap_wall ? {
      two: Math.round(pipeline.overlap_wall.ge2 * 100), three: Math.round(pipeline.overlap_wall.ge3 * 100), mean: round(pipeline.overlap_wall.mean_active, 2),
    } : null,
    // CUDA stream windows exist only when torch.cuda streams ran; otherwise the server says why they are absent.
    gpuStreams: pipeline.overlap_gpu_streams ? {
      two: Math.round(pipeline.overlap_gpu_streams.ge2 * 100), three: Math.round(pipeline.overlap_gpu_streams.ge3 * 100),
    } : null,
    gpuNote: pipeline.overlap_gpu_streams ? null : (pipeline.gpu_stream_note ?? 'CUDA stream windows were not recorded.'),
  };
}

const mib = bytes => (finite(bytes) === null ? null : round(bytes / (1024 * 1024), 2));

/** Host<->device copies counted at the ONNX Runtime boundary, with an honest verdict. */
export function summarizeTransfers(transfers) {
  if (!transfers) return null;
  const totals = transfers.total ?? {};
  const perFrame = transfers.per_frame ?? null;
  let verdict, level;
  if (!transfers.measured) { level = 'idle'; verdict = 'The transfer ledger is not active (ONNX Runtime has not been loaded or it is disabled).'; }
  else if (transfers.zero_intermediate_copies === null || transfers.zero_intermediate_copies === undefined) {
    level = 'idle'; verdict = 'No GPU inference has been observed yet, so copies are unknown (not zero).';
  } else if (transfers.zero_intermediate_copies) {
    level = 'ok'; verdict = `No intermediate host↔device copies at the ONNX Runtime boundary (${transfers.gpu_runs} GPU run(s) observed).`;
  } else {
    level = 'warn';
    const frames = perFrame ? `${perFrame.intermediate_copies} tensor copies per frame` : `${totals.intermediate_copies} tensor copies`;
    verdict = `${frames} cross PCIe between models at the ONNX Runtime boundary: the stages hand frames over through host memory, so the chain is not zero-copy.`;
  }
  const stages = Object.entries(transfers.by_stage ?? {}).map(([name, value]) => ({
    name, h2d: value.h2d_count ?? 0, d2h: value.d2h_count ?? 0, h2dMiB: mib(value.h2d_bytes), d2hMiB: mib(value.d2h_bytes),
  }));
  return {
    level, verdict, scope: transfers.scope, gpuRuns: transfers.gpu_runs ?? 0,
    h2d: totals.h2d_count ?? 0, d2h: totals.d2h_count ?? 0, h2dMiB: mib(totals.h2d_bytes), d2hMiB: mib(totals.d2h_bytes),
    intermediate: totals.intermediate_copies ?? 0, boundary: totals.boundary_copies ?? 0,
    perFrame: perFrame ? { h2d: perFrame.h2d_tensors, d2h: perFrame.d2h_tensors, h2dMiB: mib(perFrame.h2d_bytes), d2hMiB: mib(perFrame.d2h_bytes) } : null,
    stages, notCounted: transfers.not_counted ?? [],
  };
}

/** Occupancy of the frame ring buffers (page-locked HOST memory, not VRAM). */
export function summarizePools(pools = []) {
  return pools.map((pool, index) => ({
    id: index, backend: pool.backend, count: pool.count ?? 0, inFlight: pool.in_flight ?? 0,
    fill: pool.count ? Math.min(1, (pool.in_flight ?? 0) / pool.count) : 0,
    mib: mib(pool.bytes), waits: pool.waits ?? 0, blockedS: round(pool.blocked_s, 2),
    pinned: pool.backend === 'cuda_pinned',
  }));
}
