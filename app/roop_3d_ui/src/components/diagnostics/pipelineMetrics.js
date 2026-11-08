// Pure helpers for the saturation panel: shape the /api/v1/system/pipeline payload into display rows.
// Every number is a measurement from the server; a value the driver or pipeline did not report stays null (shown as "—").

const number = value => (typeof value === 'number' && Number.isFinite(value) ? value : null);
const round = (value, digits = 1) => (number(value) === null ? null : Number(value.toFixed(digits)));

/** The pipeline that is running, else the most recently finished one. */
export function pickPipeline(pipelines = []) {
  return pipelines.find(item => item.running) ?? pipelines[pipelines.length - 1] ?? null;
}

export function summarizePipeline(metrics) {
  if (!metrics) return null;
  const latest = metrics.hardware?.latest ?? {};
  const window = metrics.hardware?.window ?? {};
  const pipeline = pickPipeline(metrics.pipelines);
  const stages = (pipeline?.stages ?? []).map(stage => ({
    name: stage.name,
    depth: number(stage.queue_out?.depth),
    capacity: number(stage.queue_out?.capacity),
    highWater: number(stage.queue_out?.high_water),
    fill: number(stage.queue_out?.depth) !== null && stage.queue_out.capacity ? stage.queue_out.depth / stage.queue_out.capacity : null,
    busyPct: round(stage.busy_pct),
    meanMs: round(stage.mean_ms, 2),
    cudaMs: round(stage.cuda_mean_ms, 2),   // CUDA-event stream latency; null for stages with no CUDA stream of their own
    starvedPct: round(stage.starved_pct),   // waiting for upstream
    blockedPct: round(stage.blocked_pct),   // waiting for downstream (back-pressure)
    stream: !!stage.cuda_stream,
    bottleneck: pipeline?.bottleneck === stage.name,
  }));
  return {
    gpu: round(latest.gpu_util_pct, 0), gpuMean: round(window.gpu_util_pct?.mean),
    cpu: round(latest.cpu_util_pct, 0), cpuMean: round(window.cpu_util_pct?.mean),
    encoder: round(latest.encoder_util_pct, 0), decoder: round(latest.decoder_util_pct, 0),
    pcieTx: round(latest.pcie_tx_mb_s), pcieRx: round(latest.pcie_rx_mb_s),
    vramUsed: number(latest.vram_used_mib), vramTotal: number(latest.vram_total_mib),
    source: metrics.hardware?.backend ?? null,
    fps: round(pipeline?.fps), running: !!pipeline?.running, frames: pipeline?.frames_out ?? 0,
    overlap2: pipeline?.overlap_wall ? Math.round(pipeline.overlap_wall.ge2 * 100) : null,
    overlap3: pipeline?.overlap_wall ? Math.round(pipeline.overlap_wall.ge3 * 100) : null,
    pools: pipeline?.pools ?? [],
    stages,
    live: metrics.live_stream ?? null,
  };
}

/** Plain-language verdict on whether the pipeline is keeping the GPU fed. */
export function saturationVerdict(summary) {
  if (!summary || !summary.running) return { level: 'idle', text: 'No render running' };
  if (summary.gpu === null) return { level: 'unknown', text: 'GPU utilisation unavailable' };
  if (summary.gpu >= 80) return { level: 'ok', text: `GPU saturated (${summary.gpu}%)` };
  const stalled = summary.stages.find(stage => stage.bottleneck);
  return { level: 'warn', text: `GPU at ${summary.gpu}%${stalled ? ` · limited by ${stalled.name}` : ''}` };
}
