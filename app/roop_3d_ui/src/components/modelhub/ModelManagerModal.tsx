import { useEffect, useMemo, useState } from 'react';
import DialogFrame from '../guide/DialogFrame';
import { ACTIVE_TRANSFER, acquirePipelineStream, usePipelineConnection, useStreamResyncs, useTransfers, type ModelProgress } from '../../stores/pipelineStream';
import {
  cancelBatch, formatBytes, formatEta, formatSpeed, licenseText, planRequired, pruneFinishedTransfers, refreshManifest, runVerifyAndRepair, startDownload,
  useBatch, useManifest, type ManifestModel,
} from '../../stores/modelManifest';
import './modelHub.css';

type Filter = 'all' | 'required' | 'optional' | 'missing' | 'problems';
const FILTERS: readonly { id: Filter; label: string }[] = [
  { id: 'all', label: 'All models' }, { id: 'required', label: 'Required' }, { id: 'optional', label: 'Optional' },
  { id: 'missing', label: 'Not installed' }, { id: 'problems', label: 'Needs attention' },
];
const HASH_TEXT = { verified: '✓ SHA-256 verified', mismatch: '✗ SHA-256 mismatch', not_installed: '— not installed' } as const;

/** What the progress cell says for one row: live transfer first, then a partial download left on disk, then nothing. */
export function progressView(model: ManifestModel, transfer: ModelProgress | undefined): { percent: number; label: string; tone: 'busy' | 'verifying' | 'failed' | 'idle' } | null {
  if (transfer && ACTIVE_TRANSFER.includes(transfer.status)) {
    const percent = transfer.percent;
    if (transfer.status === 'queued') return { percent, label: 'Queued', tone: 'busy' };
    if (transfer.status === 'verifying') return { percent: 100, label: 'Verifying SHA-256…', tone: 'verifying' };
    if (transfer.status === 'compiling') return { percent, label: 'Building…', tone: 'busy' };
    return { percent, label: `${percent.toFixed(0)}% · ${formatSpeed(transfer.speedMbps)} · ${formatEta(transfer.etaSeconds)}`, tone: 'busy' };
  }
  if (transfer?.status === 'failed') return { percent: transfer.percent, label: transfer.detail ? `Failed: ${transfer.detail}` : 'Failed', tone: 'failed' };
  if (!model.installed && model.partialBytes > 0 && model.sizeBytes > 0) {
    const percent = Math.min(100, (model.partialBytes / model.sizeBytes) * 100);
    return { percent, label: `${percent.toFixed(0)}% on disk, resumes`, tone: 'idle' };
  }
  return null;
}

/**
 * Built-in Model Hub: every model the pipeline can use, its licence, size, SHA-256 state and live transfer progress, with
 * one-click download of all required models and a verify-and-repair pass. Live byte counts come from `/ws/v1/pipeline/status`
 * (which reconnects by itself); the manifest is re-read after every reconnect so a backend restart never leaves stale rows.
 */
export default function ModelManagerModal({ onClose }: { onClose: () => void }) {
  const { manifest, loading, error } = useManifest();
  const batch = useBatch();
  const transfers = useTransfers();
  const connection = usePipelineConnection();
  const resyncs = useStreamResyncs();
  const [filter, setFilter] = useState<Filter>('all');
  const [search, setSearch] = useState('');
  const [accepted, setAccepted] = useState(false);

  useEffect(() => acquirePipelineStream(), []);
  useEffect(() => { void refreshManifest({ force: true }); }, [resyncs]);
  useEffect(() => { if (manifest) pruneFinishedTransfers(manifest); }, [manifest]);

  const plan = useMemo(() => planRequired(manifest), [manifest]);
  const busy = batch.phase === 'starting' || batch.phase === 'running' || batch.phase === 'verifying';
  const rows = useMemo(() => (manifest?.models ?? []).filter(model => {
    const text = `${model.name} ${model.id} ${model.category} ${licenseText(model)}`.toLowerCase();
    if (search.trim() && !text.includes(search.trim().toLowerCase())) return false;
    const missingRequired = manifest?.requiredMissing.includes(model.id);
    switch (filter) {
      case 'required': return !model.optional;
      case 'optional': return model.optional;
      case 'missing': return !model.installed;
      case 'problems': return model.hashStatus === 'mismatch' || !!missingRequired || transfers[model.id]?.status === 'failed';
      default: return true;
    }
  }), [manifest, filter, search, transfers]);

  const required = (manifest?.models ?? []).filter(model => !model.optional);
  const installed = (manifest?.models ?? []).filter(model => model.installed).length;
  const downloadRequired = () => {
    const ids = [...plan.open, ...(accepted ? plan.needsAcceptance : [])].map(model => model.id);
    void startDownload(ids, { acceptedLicense: accepted });
  };
  const acceptanceNeeded = (model: ManifestModel) => model.requiresAcceptance && !accepted;
  const reachable = (model: ManifestModel) => model.mirrorCount > 0 || model.kind === 'compiled' || model.kind === 'bundled';

  return (
    <DialogFrame title="Model Hub" eyebrow="ONE-CLICK MODEL DOWNLOADER" onClose={onClose} className="model-hub-modal" testId="model-hub-modal">
      <div className="model-hub-summary" data-testid="model-hub-summary">
        <span><b>{installed}</b> of <b>{manifest?.models.length ?? 0}</b> installed</span>
        <span><b>{required.filter(model => model.installed).length}</b> of <b>{required.length}</b> required ready</span>
        {manifest && <span>Disk: <b>{formatBytes(manifest.disk.installedBytes)}</b> used{manifest.disk.freeBytes != null && <>, <b>{formatBytes(manifest.disk.freeBytes)}</b> free</>}</span>}
        {manifest && manifest.disk.downloadNeededBytes > 0 && <span>{formatBytes(manifest.disk.downloadNeededBytes)} to download{manifest.disk.enoughSpace === false && <b style={{ color: 'var(--danger)' }}> · not enough free space</b>}</span>}
        <span className="status-chip" data-testid="hub-connection" title="Live progress comes from /ws/v1/pipeline/status; it reconnects automatically">
          Live updates: {connection === 'online' ? 'connected' : connection === 'offline' ? 'off' : 'reconnecting…'}
        </span>
      </div>

      <div className="model-hub-actions">
        <button type="button" className="button accent" disabled={busy || (plan.open.length === 0 && !(accepted && plan.needsAcceptance.length > 0))}
          title={plan.open.length + plan.needsAcceptance.length === 0 ? 'Every required model that can be downloaded is installed.' : undefined} onClick={downloadRequired} data-testid="download-all-required">
          {batch.phase === 'starting' ? 'Starting…' : 'Download All Required Models'}
        </button>
        <button type="button" className="button" disabled={busy} onClick={() => void runVerifyAndRepair([], { acceptedLicense: accepted })} data-testid="verify-and-repair">Verify &amp; repair</button>
        <button type="button" className="button" disabled={loading || busy} onClick={() => void refreshManifest({ force: true })}>{loading ? 'Reading…' : 'Re-check files'}</button>
        {busy && batch.batchId && <button type="button" className="button" onClick={() => void cancelBatch()}>Cancel (keeps partial files)</button>}
        {plan.needsAcceptance.length > 0 && (
          <label title={plan.needsAcceptance.map(model => `${model.name}: ${model.license}`).join('\n')}>
            <input type="checkbox" checked={accepted} onChange={event => setAccepted(event.target.checked)} data-testid="accept-licenses" />
            I accept the licence terms of the {plan.needsAcceptance.length} restricted model{plan.needsAcceptance.length === 1 ? '' : 's'} ({formatBytes(plan.bytesRestricted)})
          </label>)}
      </div>

      {error && <div className="model-hub-note error" role="alert">Model manifest unavailable: {error} <button type="button" className="button small" onClick={() => void refreshManifest({ force: true })}>Retry</button></div>}
      {batch.message && <div className={`model-hub-note ${batch.phase === 'error' ? 'error' : ''}`} role={batch.phase === 'error' ? 'alert' : 'status'} data-testid="hub-message">{batch.message}</div>}
      {batch.skipped.length > 0 && (
        <details className="model-hub-note" data-testid="hub-skipped"><summary>{batch.skipped.length} model{batch.skipped.length === 1 ? ' was' : 's were'} not started</summary>
          <ul>{batch.skipped.map(item => <li key={item.id}><b>{item.id}</b>: {item.reason.replaceAll('_', ' ').toLowerCase()}. {item.detail}</li>)}</ul></details>)}
      {plan.unavailable.length > 0 && (
        <details className="model-hub-note"><summary>{plan.unavailable.length} required model{plan.unavailable.length === 1 ? ' has' : 's have'} no public download</summary>
          <ul>{plan.unavailable.map(model => <li key={model.id}><b>{model.name}</b>: {model.note || model.acquisition.replaceAll('_', ' ')}. Place the file in app/models, then use “Re-check files”.</li>)}</ul></details>)}

      <div className="model-hub-filters">
        <select aria-label="Filter models" value={filter} onChange={event => setFilter(event.target.value as Filter)}>{FILTERS.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select>
        <input aria-label="Search models" placeholder="Search by name, id, category or licence" value={search} onChange={event => setSearch(event.target.value)} />
      </div>

      <ul className="model-hub-list" data-testid="model-hub-list">
        {rows.map(model => {
          const transfer = transfers[model.id];
          const view = progressView(model, transfer);
          const active = !!transfer && ACTIVE_TRANSFER.includes(transfer.status);
          const missing = !model.installed && model.state !== 'NOT_APPLICABLE';
          return (
            <li key={model.id} className="model-hub-row" data-testid={`hub-row-${model.id}`}>
              <div className="model-name">
                <strong title={model.name}>{model.name}{model.optional ? '' : ' · required'}</strong>
                <small className="muted" title={model.detail || model.note}>{model.id} · {model.category}{model.detail ? ` · ${model.detail}` : ''}</small>
              </div>
              <a href={model.licenseUrl || undefined} target="_blank" rel="noreferrer" onClick={event => { if (!model.licenseUrl) event.preventDefault(); }}
                className={`license-badge ${model.licenseBadge}`} title={model.licenseNote || model.license} data-testid={`license-${model.id}`}>{licenseText(model)}</a>
              <span className="model-size" title={model.sha256 ? `SHA-256 ${model.sha256}` : undefined}>{formatBytes(model.sizeBytes)}</span>
              <div className="model-progress">
                <span className={`hash-check ${model.hashStatus}`} title={model.sha256 ? `SHA-256 ${model.sha256}` : 'No digest recorded'} data-testid={`hash-${model.id}`}>{HASH_TEXT[model.hashStatus]}</span>
                {view && (
                  <>
                    <div className={`bar ${view.tone}`} role="progressbar" aria-label={`${model.name} download`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(view.percent)}><i style={{ width: `${view.percent}%` }} /></div>
                    <small className="muted" data-testid={`progress-${model.id}`}>{view.label}</small>
                  </>)}
              </div>
              <div className="row-action">
                {missing && !active && (reachable(model)
                  ? <button type="button" className="button small" disabled={busy || acceptanceNeeded(model)} title={acceptanceNeeded(model) ? 'Tick the licence acknowledgement above first.' : undefined}
                      onClick={() => void startDownload([model.id], { acceptedLicense: accepted })}>Download</button>
                  : <small className="muted">No public source</small>)}
                {model.hashStatus === 'mismatch' && !active && <button type="button" className="button small" disabled={busy} onClick={() => void runVerifyAndRepair([model.id], { acceptedLicense: accepted })}>Repair</button>}
              </div>
            </li>);
        })}
        {!rows.length && <li className="muted">{loading ? 'Reading the manifest…' : manifest ? 'No model matches this filter.' : 'No manifest loaded.'}</li>}
      </ul>
    </DialogFrame>
  );
}
