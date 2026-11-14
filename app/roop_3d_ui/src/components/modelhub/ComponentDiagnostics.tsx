import type { ComponentDiagnostic } from '../../contracts';
import { refreshComponentStatus, useComponentStatus, useComponentStatusPolling } from '../../stores/componentStatus';
import './modelRack.css';

const LOAD_LABEL = {
  not_loaded: 'Not loaded', loaded: 'Session loaded', failed: 'Load failed', released: 'Session released',
  stale: 'Session record stale', not_applicable: 'No ONNX session (asset / math)',
};
const FORWARD_LABEL = { not_run: 'Not run', verified: 'Verified for this file', failed: 'Failed', stale: 'Stale — file changed' };

function Row({ row }: { row: ComponentDiagnostic }) {
  const runtime = row.runtime;
  return <li className={`component-diagnostic component-${row.state.toLowerCase()}`} data-testid={`component-diagnostic-${row.component}`}>
    <div className="component-diagnostic-heading"><strong>{row.label}{!row.required ? ' (optional)' : ''}</strong><span className="rack-pill">{row.state === 'READY' ? 'Available — not render readiness' : row.state}</span></div>
    <dl className="component-diagnostic-fields">
      <dt>Resolved file</dt><dd><code>{row.path ?? 'Not resolved'}</code></dd>
      <dt>Active provider</dt><dd>{runtime.active_provider ?? 'None — no active session'}</dd>
      <dt>Load status</dt><dd>{LOAD_LABEL[runtime.load_status]}</dd>
      <dt>Last standalone forward</dt><dd>{FORWARD_LABEL[row.last_forward.state]}{row.last_forward.provider ? ` · ${row.last_forward.provider}` : ''}</dd>
    </dl>
    {row.error && <pre className="component-diagnostic-error">{row.error.code}: {row.error.message}</pre>}
    {runtime.error && <pre className="component-diagnostic-error">{runtime.error}</pre>}
    {row.last_forward.error && <pre className="component-diagnostic-error">Forward: {row.last_forward.error}</pre>}
    {runtime.fallback_errors.length > 0 && <details><summary>Provider fallback diagnostics</summary><pre>{runtime.fallback_errors.join('\n')}</pre></details>}
    {runtime.warnings.length > 0 && <details><summary>Non-fatal provider warnings</summary><pre>{runtime.warnings.join('\n')}</pre></details>}
    <details><summary>Artifact and execution evidence</summary>
      <dl className="component-diagnostic-fields">
        <dt>Model ID</dt><dd><code>{row.model_id ?? 'No model weights'}</code></dd>
        <dt>Integrity</dt><dd>{row.integrity_verified ? 'Verified bytes' : 'Not verified'} · {row.trust_basis ?? 'No trust evidence'}</dd>
        <dt>SHA-256</dt><dd><code>{row.sha256 ?? 'Not recorded'}</code></dd>
        <dt>Live forward</dt><dd>{runtime.execution_verified ? 'Executed in this server' : 'Not recorded in this server'}</dd>
        <dt>Providers tried</dt><dd>{runtime.requested_providers.join(' → ') || 'No session attempt recorded'}</dd>
        {runtime.session_path && <><dt>Session graph</dt><dd><code>{runtime.session_path}</code></dd></>}
        {row.license && <><dt>License</dt><dd>{row.license.license_id} · {row.license.reviewed ? 'reviewed terms' : 'unreviewed terms'} · commercial use: {row.license.commercial_use}<p>{row.license.notice}</p></dd></>}
      </dl>
      {row.acquisition && <p>{row.acquisition}</p>}
    </details>
    {row.assets.length > 0 && <details open><summary>Resolved head assets</summary><ul className="component-diagnostic-list">{row.assets.map(asset => <Row key={asset.component} row={asset} />)}</ul></details>}
  </li>;
}

/** A resolved/installed file or a standalone forward never overrides the production renderer guard. */
export default function ComponentDiagnostics() {
  useComponentStatusPolling();
  const { value, error, loading } = useComponentStatus(state => state);
  return <section className="rack-section component-diagnostics" aria-label="Pipeline component diagnostics" data-testid="component-diagnostics">
    <h4>Pipeline component diagnostics</h4>
    <div className="model-manager-row"><button type="button" className="button small" disabled={loading} onClick={() => void refreshComponentStatus()}>{loading ? 'Reading diagnostics…' : 'Refresh diagnostics'}</button></div>
    {error && <p role="alert">{value ? 'Read-back is stale. ' : ''}{error}</p>}
    {!value && !error && <small role="status">Reading resolved files and recorded sessions…</small>}
    {value && <>
      <p className="component-renderer-state" role="status">{value.renderer.wired ? 'Renderer wired; component availability is not end-to-end render proof.' : `${value.renderer.code ?? 'RENDERER_NOT_WIRED'}: ${value.renderer.message ?? '3D renderer not wired'}. Render requests remain blocked.`}</p>
      <small className="muted">Installed / available ≠ loaded ≠ forward verified. Forward receipts refer to a separate standalone process, not models resident here. No inference sessions are created by this read-back.</small>
      <ul className="component-diagnostic-list">{value.components.map(row => <Row key={row.component} row={row} />)}</ul>
      <small className="muted">Advertised providers (not load proof): {value.providers.advertised.join(', ') || 'None'}</small>
      {value.providers.error && <pre className="component-diagnostic-error">{value.providers.error.code}: {value.providers.error.message}</pre>}
      <small className="muted">Standalone report: {value.verification_report.state}{value.verification_report.generated_at ? ` · ${value.verification_report.generated_at}` : ''}</small>
      {value.verification_report.error && <pre className="component-diagnostic-error">{value.verification_report.error.code}: {value.verification_report.error.message}</pre>}
      <small className="muted">File integrity / local placement never grants model license rights.</small>
    </>}
  </section>;
}
