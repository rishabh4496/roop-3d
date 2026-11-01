import type { LivePreview } from '../../livePreview';

export function describeStage(stage: string | null): string {
  if (!stage) return 'engine';
  const numbered = /^STAGE_(\d+)_(.+)$/.exec(stage);
  const words = (value: string) => value.toLowerCase().replace(/_/g, ' ');
  return numbered ? `stage ${numbered[1]} · ${words(numbered[2])}` : words(stage);
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
      </>}
    </div>
  );
}
