import React, { useMemo } from 'react';
import { usePipelineMetrics } from './PipelineSaturationPanel';
import { pickPipeline } from './pipelineMetrics';
import { buildLaneModel, summarizePools, summarizeTransfers } from './lanesModel';

const dash = value => (value === null || value === undefined ? '—' : value);

/**
 * Per-frame stage lanes on one time axis (so overlap is visible), the PCIe / copy picture and the frame ring buffers.
 * Pass `metrics` to render a payload you already hold; otherwise it polls /api/v1/system/pipeline while mounted.
 */
/** @param {{ metrics?: any, enabled?: boolean }} props */
export default function StreamOverlapPanel({ metrics: given = null, enabled = true }) {
  const polled = usePipelineMetrics(1000, enabled && !given);
  const metrics = given ?? polled.metrics, error = given ? null : polled.error;
  const pipeline = useMemo(() => pickPipeline(metrics?.pipelines ?? []), [metrics]);
  const lanes = useMemo(() => buildLaneModel(pipeline), [pipeline]);
  const transfers = useMemo(() => summarizeTransfers(pipeline?.transfers ?? metrics?.transfers), [pipeline, metrics]);
  const pools = useMemo(() => summarizePools(pipeline?.pools), [pipeline]);
  const latest = metrics?.hardware?.latest ?? {};

  if (!metrics) return <div className="stream-panel" data-testid="stream-panel"><p className="muted">{error ? `Telemetry unavailable (${error})` : 'Reading pipeline telemetry…'}</p></div>;

  return (
    <div className="stream-panel" data-testid="stream-panel">
      <section aria-label="Stage lanes">
        <div className="hud-card-row"><span className="hud-label">PER-FRAME STAGE LANES</span>
          <span className="muted">{lanes ? `last ${lanes.frames} frames · ${lanes.windowMs} ms window` : 'no frame timed yet'}</span></div>
        {!lanes && <p className="muted" data-testid="lanes-empty">Start a render to see its stages overlap. Nothing is drawn until frames have been timed.</p>}
        {lanes && (
          <div className="lane-chart" role="img" aria-label={`Stage lanes: ${lanes.rows.map(row => row.label).join(', ')}`}>
            {lanes.rows.map(row => (
              <div className="lane-row" key={row.name} data-testid={`lane-${row.name}`}>
                <span className="lane-label" title={row.hw}>{row.label}</span>
                <div className="lane-track">
                  {row.segments.map((segment, index) => (
                    <i key={`${segment.frame}-${index}`} className="lane-seg" title={`frame ${segment.frame}: ${(segment.end - segment.start).toFixed(1)} ms`}
                      style={{ left: `${segment.left}%`, width: `${segment.width}%`, background: row.color }} />
                  ))}
                </div>
                <span className="lane-busy">{dash(row.busyMs)} ms</span>
              </div>
            ))}
          </div>
        )}
        {lanes?.overlap && (
          <p className="sat-line" data-testid="overlap-line">Host stages overlapping: ≥2 <b>{lanes.overlap.two}%</b> · ≥3 <b>{lanes.overlap.three}%</b> of the time · mean <b>{lanes.overlap.mean}</b> active</p>
        )}
        {lanes?.gpuStreams && <p className="sat-line">CUDA streams overlapping: ≥2 <b>{lanes.gpuStreams.two}%</b> · ≥3 <b>{lanes.gpuStreams.three}%</b></p>}
        {lanes && !lanes.gpuStreams && <p className="muted" data-testid="gpu-stream-note">CUDA stream overlap: not measured. {lanes.gpuNote}</p>}
      </section>

      <section aria-label="PCIe transfers">
        <div className="hud-card-row"><span className="hud-label">PCIe &amp; HOST↔DEVICE COPIES</span><span className="muted">NVML {metrics.hardware?.backend ?? 'unavailable'}</span></div>
        <p className="sat-line">Link now: ↑ <b>{dash(latest.pcie_tx_mb_s)}</b> MB/s · ↓ <b>{dash(latest.pcie_rx_mb_s)}</b> MB/s · VRAM <b>{dash(latest.vram_used_mib)}</b>/{dash(latest.vram_total_mib)} MiB
          <span className="muted"> · {metrics.pcie_note ?? 'instantaneous NVML sample'}</span></p>
        {transfers && (
          <>
            <p className={`sat-verdict sat-${transfers.level}`} role="status" data-testid="transfer-verdict">{transfers.verdict}</p>
            <div className="sat-line" data-testid="transfer-totals">
              Counted at the {transfers.scope}: H→D <b>{transfers.h2d}</b> ({dash(transfers.h2dMiB)} MiB) · D→H <b>{transfers.d2h}</b> ({dash(transfers.d2hMiB)} MiB)
              {transfers.perFrame && <> · per frame H→D <b>{transfers.perFrame.h2d}</b>, D→H <b>{transfers.perFrame.d2h}</b></>}
            </div>
            {transfers.stages.length > 0 && (
              <table className="sat-table" aria-label="Copies by stage">
                <thead><tr><th>Stage</th><th>H→D</th><th>D→H</th><th>H→D MiB</th><th>D→H MiB</th></tr></thead>
                <tbody>{transfers.stages.map(stage => <tr key={stage.name}><td>{stage.name}</td><td>{stage.h2d}</td><td>{stage.d2h}</td><td>{dash(stage.h2dMiB)}</td><td>{dash(stage.d2hMiB)}</td></tr>)}</tbody>
              </table>
            )}
            {transfers.notCounted.length > 0 && <p className="muted" data-testid="not-counted">Not counted: {transfers.notCounted.join('; ')}.</p>}
          </>
        )}
      </section>

      <section aria-label="Frame ring buffers">
        <div className="hud-card-row"><span className="hud-label">FRAME RING BUFFER</span><span className="muted">page-locked host memory (not VRAM)</span></div>
        {pools.length === 0 && <p className="muted" data-testid="pools-empty">No frame pool is allocated (no render is running, or pinned frames are disabled).</p>}
        {pools.map(pool => (
          <div className="pool-row" key={pool.id} data-testid="pool-row">
            <span>{pool.pinned ? 'Pinned pool' : 'Pageable pool'} ({pool.backend})</span>
            <div className="sat-bar sat-idle" role="meter" aria-label="Ring buffer occupancy" aria-valuemin={0} aria-valuemax={pool.count} aria-valuenow={pool.inFlight}><span style={{ width: `${pool.fill * 100}%` }} /></div>
            <b>{pool.inFlight}/{pool.count}</b> slots in flight · {dash(pool.mib)} MiB · waited {pool.waits}× ({dash(pool.blockedS)} s)
          </div>
        ))}
      </section>
    </div>
  );
}
