import type { LivePreview } from '../../livePreview';
import type { RestorationReport } from '../../api';

export function describeStage(stage: string | null): string {
  if (!stage) return 'engine';
  const numbered = /^STAGE_(\d+)_(.+)$/.exec(stage);
  const words = (value: string) => value.toLowerCase().replace(/_/g, ' ');
  return numbered ? `stage ${numbered[1]} · ${words(numbered[2])}` : words(stage);
}

const BYPASS_REASONS: Record<string, string> = { small_face_bypass: 'small face' };
const ms = (value: number) => `${Math.round(value)} ms`;

/** One line of what the restorer really did for this frame: completed inferences, user-selected skips and failures, with measured times. */
export function describeRestoration(report: RestorationReport): string {
  const parts = [`${report.applied} applied`];
  if (report.mean_latency_ms !== null) parts[0] += ` (${ms(report.mean_latency_ms)}${report.mean_inference_ms !== null ? `, model ${ms(report.mean_inference_ms)}` : ''})`;
  if (report.bypassed) parts.push(`${report.bypassed} bypassed (${Object.keys(report.bypass_reasons).map(reason => BYPASS_REASONS[reason] ?? reason).join(', ') || 'selected'})`);
  if (report.failed) parts.push(`${report.failed} failed`);
  return `Restorer ${report.model}: ${parts.join(' · ')}`;
}

/** Small, non-modal status pill for the live swap preview (never covers the image centre). */
export default function LiveSwapStatus({ live }: { live: LivePreview }) {
  if (live.status === 'idle') return null;
  const { status, meta } = live;
  const total = meta?.latency?.total_ms;
  return (
    <div className={`live-swap-status ${status}`} role="status" aria-live="polite">
      {status === 'loading' && <span>Rendering swap…</span>}
      {status === 'busy' && <span title="The GPU is running a render job or compiling an engine; this preview retries automatically.">GPU busy · {describeStage(live.stage)} · retrying</span>}
      {status === 'error' && <><span>{live.error}</span><button type="button" className="live-swap-retry" onClick={live.retry}>Retry</button></>}
      {status === 'ready' && <>
        <span>{live.cached ? 'Cached' : 'Live'}{typeof total === 'number' && !live.cached ? ` · ${Math.round(total)} ms` : ''}</span>
        {!!meta?.unsupported.length && <span className="live-swap-unsupported" title={`Accepted but not applied by this engine: ${meta.unsupported.join(', ')}`}>{meta.unsupported.length} control{meta.unsupported.length > 1 ? 's' : ''} inactive</span>}
        {meta?.restoration && <span className="live-swap-restoration" data-testid="live-restoration" title="Observed by the restorer that rendered this frame; faces never offered to it (blend 0) are not counted.">{describeRestoration(meta.restoration)}</span>}
      </>}
    </div>
  );
}
