import React, { useEffect, useMemo, useState } from 'react';
import { summarizePipeline, saturationVerdict } from './pipelineMetrics';

const dash = value => (value === null || value === undefined ? '—' : value);

/** Polls the server while mounted (the HUD popover is open); nothing is requested when it is closed. */
export function usePipelineMetrics(intervalMs = 1000, enabled = true) {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    if (!enabled) return undefined;
    let alive = true, timer;
    const poll = async () => {
      try {
        const response = await fetch('/api/v1/system/pipeline', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const body = await response.json();
        if (alive) { setMetrics(body); setError(null); }
      } catch (failure) {
        if (alive) setError(failure.message);
      }
      if (alive) timer = setTimeout(poll, intervalMs);
    };
    poll();
    return () => { alive = false; clearTimeout(timer); };
  }, [intervalMs, enabled]);
  return { metrics, error };
}

function Bar({ value, max = 100, label, tone = 'ok' }) {
  const pct = value === null || value === undefined ? 0 : Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className={`sat-bar sat-${tone}`} role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={max} aria-valuenow={value ?? 0}>
      <span style={{ width: `${pct}%` }}/>
    </div>
  );
}

export default function PipelineSaturationPanel({ enabled = true }) {
  const { metrics, error } = usePipelineMetrics(1000, enabled);
  const summary = useMemo(() => summarizePipeline(metrics), [metrics]);
  const verdict = saturationVerdict(summary);
  if (!summary) return <div className="sat-panel" data-testid="saturation-panel"><span className="hud-label">PIPELINE SATURATION</span><p className="muted">{error ? `Telemetry unavailable (${error})` : 'Reading hardware counters…'}</p></div>;
  const tone = verdict.level === 'ok' ? 'ok' : verdict.level === 'warn' ? 'warn' : 'idle';
  return (
    <div className="sat-panel" data-testid="saturation-panel">
      <div className="hud-card-row"><span className="hud-label">PIPELINE SATURATION</span><span className={`sat-verdict sat-${tone}`}>{verdict.text}</span></div>
      <div className="sat-grid">
        <label>GPU <b>{dash(summary.gpu)}%</b><Bar value={summary.gpu} label="GPU utilisation" tone={tone}/></label>
        <label>CPU <b>{dash(summary.cpu)}%</b><Bar value={summary.cpu} label="CPU utilisation" tone="idle"/></label>
        <label>NVENC <b>{dash(summary.encoder)}%</b><Bar value={summary.encoder} label="Encoder engine" tone="idle"/></label>
        <label>NVDEC <b>{dash(summary.decoder)}%</b><Bar value={summary.decoder} label="Decoder engine" tone="idle"/></label>
      </div>
      <div className="sat-line">
        PCIe ↑ <b>{dash(summary.pcieTx)}</b> MB/s · ↓ <b>{dash(summary.pcieRx)}</b> MB/s · VRAM <b>{dash(summary.vramUsed)}</b>/{dash(summary.vramTotal)} MiB
        <span className="muted"> · {summary.source ?? 'no counter source'}</span>
      </div>
      {summary.stages.length > 0 && (
        <table className="sat-table" aria-label="Pipeline stages">
          <thead><tr><th>Stage</th><th>Queue</th><th>Wall ms</th><th>CUDA ms</th><th>Busy</th><th>Starved</th><th>Blocked</th></tr></thead>
          <tbody>
            {summary.stages.map(stage => (
              <tr key={stage.name} className={stage.bottleneck ? 'bottleneck' : ''}>
                <td>{stage.name}{stage.stream ? ' ⟿' : ''}</td>
                <td title={`high-water ${dash(stage.highWater)}`}>{dash(stage.depth)}/{dash(stage.capacity)}<Bar value={stage.depth} max={stage.capacity || 1} label={`${stage.name} queue depth`} tone="idle"/></td>
                <td>{dash(stage.meanMs)}</td><td>{dash(stage.cudaMs)}</td>
                <td>{dash(stage.busyPct)}%</td><td>{dash(stage.starvedPct)}%</td><td>{dash(stage.blockedPct)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="sat-line">
        {summary.running ? `${dash(summary.fps)} fps · ${summary.frames} frames` : 'idle'}
        {summary.overlap3 !== null && <> · stages overlapping: ≥2 <b>{summary.overlap2}%</b> · ≥3 <b>{summary.overlap3}%</b> of the time</>}
        {summary.live && <span className="muted"> · live JPEG {dash(summary.live.backend)} {dash(summary.live.mean_encode_ms)} ms · {summary.live.subscribers} viewer(s) · {summary.live.dropped} dropped</span>}
      </div>
    </div>
  );
}
