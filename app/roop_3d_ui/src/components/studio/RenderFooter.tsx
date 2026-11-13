import type { Job } from '../../contracts';
import { RENDER_FRAME_EVENT } from '../../livePreview';
import { useTelemetry } from '../../stores/telemetryStore';

const ACTIVE = new Set(['queued', 'preparing', 'running', 'pausing']);
const mib = (value: number) => `${(value / 1024).toFixed(1)} GB`;

export interface RenderFooterProps {
  /** The job for the loaded target, if any (drives the progress bar). */
  job: Job | null;
  /** Why "Start Full Render" is unavailable (shown as the button title); null = ready. */
  startBlockedReason: string | null;
  /** Why "Render Current Frame" is unavailable; null = ready. */
  frameBlockedReason: string | null;
  onStart: () => void;
  onCancel?: () => void;
}

/** GPU name + used / total VRAM from the telemetry stream; "GPU telemetry offline" until the first tick. */
function GpuPill() {
  const gpu = useTelemetry(state => state.hardware?.gpus?.[0] ?? null, (a, b) => a?.name === b?.name && a?.free_vram_mib === b?.free_vram_mib);
  if (!gpu) return <span className="gpu-pill offline" data-testid="gpu-pill" title="No telemetry yet">GPU telemetry offline</span>;
  return <span className="gpu-pill" data-testid="gpu-pill" title={`${gpu.name}: ${mib(gpu.total_vram_mib - gpu.free_vram_mib)} used of ${mib(gpu.total_vram_mib)}`}>
    <i aria-hidden /> {gpu.name.replace(/^NVIDIA GeForce /, '')} · {mib(gpu.total_vram_mib - gpu.free_vram_mib)} / {mib(gpu.total_vram_mib)}
  </span>;
}

export default function RenderFooter({ job, startBlockedReason, frameBlockedReason, onStart, onCancel }: RenderFooterProps) {
  const live = useTelemetry(state => state.progress, (a, b) => a === b);
  const active = !!job && ACTIVE.has(job.state);
  const total = job?.progress.total_frames ?? null;
  const frame = job?.progress.frame ?? 0;
  const percent = job ? (job.state === 'succeeded' ? 100 : total ? Math.min(100, Math.round((frame / total) * 100)) : 0) : 0;
  const fps = (active && typeof live?.fps === 'number' ? live.fps : job?.progress.fps) ?? 0;
  const status = !job ? 'No render yet' : job.state === 'succeeded' ? 'Render complete'
    : job.state === 'failed' ? `Render failed${job.failure?.message ? `: ${job.failure.message}` : ''}`
    : active ? `${job.progress.stage || job.state} · frame ${frame}${total ? ` / ${total}` : ''}` : job.state;
  return (
    <footer className="render-footer sticky-footer" aria-label="Render actions" data-testid="render-footer">
      <div className="render-actions">
        {active && onCancel ? (
          <button
            type="button"
            className="button danger render-cancel-btn"
            onClick={onCancel}
            title="Cancel and abort the active render immediately"
            data-testid="render-cancel-btn"
          >
            ✕ Cancel Render
          </button>
        ) : null}
        <button type="button" className="button" disabled={!!frameBlockedReason || active} title={frameBlockedReason ?? 'Swap the frame under the playhead and compare it with the original'}
          onClick={() => window.dispatchEvent(new Event(RENDER_FRAME_EVENT))}>⚡ Render current frame</button>
        <button type="button" className="button accent" disabled={!!startBlockedReason || active} title={startBlockedReason ?? 'Queue a full local render of the mapped faces'} onClick={onStart}>
          {active ? 'Rendering in progress…' : 'Start full render ↗'}
        </button>
      </div>
      <div className="render-progress" aria-hidden="true" data-testid="render-progress" data-percent={percent}>
        <div style={{ width: `${percent}%` }} />
      </div>
      <div className="render-status">
        <span data-testid="render-status">{status}{percent > 0 ? ` · ${percent}%` : ''}</span>
        <span data-testid="render-fps">{active || fps > 0 ? `${fps.toFixed(1)} fps` : '— fps'}</span>
      </div>
      <GpuPill />
    </footer>
  );
}
