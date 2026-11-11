import { useEffect, useMemo, useState } from 'react';
import { acquirePipelineStream, useStreamResyncs, useTransfers } from '../../stores/pipelineStream';
import { batchStore, formatBytes, planRequired, refreshManifest, startDownload, useBatch, useManifest } from '../../stores/modelManifest';
import './modelHub.css';

export const BANNER_DISMISS_KEY = 'roop3d-required-models-dismissed';
const readDismissed = (): string => { try { return localStorage.getItem(BANNER_DISMISS_KEY) ?? ''; } catch { return ''; } };
const writeDismissed = (value: string): void => { try { localStorage.setItem(BANNER_DISMISS_KEY, value); } catch { /* not remembered */ } };

/**
 * Non-intrusive strip under the top bar while required models are missing. One button downloads and verifies all of them.
 * Models under a non-open licence are only fetched after the acknowledgement checkbox is ticked (the service refuses them
 * otherwise). Dismissing hides the strip until the set of missing models changes.
 */
export default function RequiredModelsBanner({ onOpenHub }: { onOpenHub: () => void }) {
  const { manifest } = useManifest();
  const batch = useBatch();
  const transfers = useTransfers();
  const resyncs = useStreamResyncs();
  const [accepted, setAccepted] = useState(false);
  const [dismissed, setDismissed] = useState(readDismissed);

  useEffect(() => acquirePipelineStream(), []);
  useEffect(() => { void refreshManifest(); }, [resyncs]);

  const plan = useMemo(() => planRequired(manifest), [manifest]);
  const signature = manifest?.requiredMissing.join(',') ?? '';
  const busy = batch.phase === 'starting' || batch.phase === 'running' || batch.phase === 'verifying';

  // Overall progress of the models this run is fetching.
  const overall = useMemo(() => {
    let done = 0, total = 0;
    for (const id of batch.ids) { const item = transfers[id]; if (item) { done += item.downloadedBytes; total += item.totalBytes; } }
    return total > 0 ? Math.min(100, (done / total) * 100) : null;
  }, [batch.ids, transfers]);

  if (!manifest || !signature || (dismissed === signature && !busy)) return null;
  if (!busy && batch.phase === 'done' && plan.open.length + plan.needsAcceptance.length + plan.unavailable.length === 0) return null;

  const downloadable = plan.open.length + plan.needsAcceptance.length;
  const download = () => {
    const ids = [...plan.open, ...(accepted ? plan.needsAcceptance : [])].map(model => model.id);
    if (!ids.length) { batchStore.set({ phase: 'done', message: `${plan.needsAcceptance.length} required model${plan.needsAcceptance.length === 1 ? ' needs' : 's need'} your licence acknowledgement first.` }); return; }
    void startDownload(ids, { acceptedLicense: accepted });
  };

  return (
    <section className="required-banner" role="region" aria-label="Required models" data-testid="required-models-banner">
      <div className="required-banner-text">
        <strong>{manifest.requiredMissing.length} required model{manifest.requiredMissing.length === 1 ? ' is' : 's are'} not installed.</strong>
        <span className="muted" role="status">
          {busy ? batch.message || 'Working…'
            : batch.message ? batch.message
            : downloadable ? `${formatBytes(plan.bytesOpen + (accepted ? plan.bytesRestricted : 0))} to download${plan.unavailable.length ? `; ${plan.unavailable.length} need a manual copy` : ''}. Each file is checked against its SHA-256.`
            : `${plan.unavailable.length} have no public download. Open the Model Hub for details.`}
        </span>
      </div>
      <div className="required-banner-actions">
        {busy && overall != null && <progress max={100} value={overall} aria-label="Overall download progress" />}
        {plan.needsAcceptance.length > 0 && !busy && (
          <label><input type="checkbox" checked={accepted} onChange={event => setAccepted(event.target.checked)} data-testid="banner-accept" />
            I accept the licence terms of {plan.needsAcceptance.length} restricted model{plan.needsAcceptance.length === 1 ? '' : 's'}</label>)}
        <button type="button" className="button small accent" disabled={busy || downloadable === 0} onClick={download} data-testid="banner-download-all">Download All Required Models</button>
        <button type="button" className="button small" onClick={onOpenHub}>Model Hub</button>
        <button type="button" className="banner-dismiss" aria-label="Hide this message" disabled={busy} onClick={() => { writeDismissed(signature); setDismissed(signature); }}>×</button>
      </div>
    </section>
  );
}
