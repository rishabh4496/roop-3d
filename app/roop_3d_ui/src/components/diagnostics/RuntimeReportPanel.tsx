import { useCallback, useEffect, useRef, useState } from 'react';
import { api, post } from '../../api';
import { pushToast } from '../../stores/toasts';
import { copyText, needsAttention, parseBenchmark, parseRuntimeReport, type BenchmarkStatus, type ReportSection, type RuntimeReport } from '../../utils/runtimeReport';
import './runtimeReport.css';

const BADGE: Record<ReportSection['status'], string> = { OK: 'ok', INFO: 'info', WARN: 'warn', UNAVAILABLE: 'warn', UNKNOWN: 'bad' };

function Section({ section }: { section: ReportSection }) {
  return (
    <details className={`rr-section rr-${BADGE[section.status]}`} open={needsAttention(section)} data-testid={`rr-section-${section.number}`}>
      <summary>
        <span className="rr-num">{section.number}</span>
        <span className="rr-title">{section.title}</span>
        <span className={`rr-badge rr-badge-${BADGE[section.status]}`}>{section.status}</span>
      </summary>
      <dl className="rr-rows">
        {section.rows.map((row, index) => (
          <div className="rr-row" key={`${row.label}-${index}`}><dt>{row.label.trim()}</dt><dd>{row.value}</dd></div>
        ))}
      </dl>
      {section.notes.map(note => <p className="rr-note" key={note}>{note}</p>)}
    </details>
  );
}

/**
 * The thread/pool benchmark that already exists on the server (`POST /settings/benchmark_threads`, `GET /settings/benchmark_status`).
 * It times a small synthetic affine + host-transfer workload for several thread and pool sizes, so its figures are samples per second
 * of that workload, not model or video FPS; the server's own note says so and is shown with the result. It never runs on its own.
 */
function BenchmarkRunner() {
  const [status, setStatus] = useState<BenchmarkStatus>({ state: 'IDLE', results: [], recommendation: null });
  const [error, setError] = useState('');
  const mounted = useRef(true);
  useEffect(() => () => { mounted.current = false; }, []);

  const refresh = useCallback(async () => {
    try {
      const next = parseBenchmark(await api<unknown>('/settings/benchmark_status'));
      if (mounted.current) { setStatus(next); setError(''); }
    } catch (reason) { if (mounted.current) setError(reason instanceof Error ? reason.message : 'Could not read the benchmark status.'); }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    if (status.state !== 'RUNNING') return undefined;
    const timer = window.setInterval(() => void refresh(), 1000);
    return () => window.clearInterval(timer);
  }, [status.state, refresh]);

  const start = async (profile: 'quick' | 'full') => {
    setError('');
    try {
      setStatus(parseBenchmark(await post<unknown>('/settings/benchmark_threads', { profile })));
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'The benchmark could not start.'); }
  };
  const running = status.state === 'RUNNING';
  const best = status.recommendation;
  return (
    <section className="rr-bench" aria-label="Thread and pool benchmark" data-testid="rr-bench">
      <h4>Thread / pool benchmark</h4>
      <p className="muted">Measures a small synthetic GPU/host-transfer workload for several thread and pool sizes. Run it while no job is rendering.</p>
      <div className="rr-actions">
        <button type="button" className="button small" disabled={running} onClick={() => void start('quick')} data-testid="rr-bench-quick">Quick run</button>
        <button type="button" className="button small" disabled={running} onClick={() => void start('full')} data-testid="rr-bench-full">Full run</button>
        <span className="rr-state" role="status" data-testid="rr-bench-state">
          {running ? `Running ${status.profile ?? ''} benchmark… ${status.results.length} configuration(s) measured` : status.state === 'DONE' ? 'Done' : status.state === 'FAILED' ? `Failed (${status.error ?? 'unknown'})` : status.state === 'CANCELLED' ? 'Cancelled' : 'Not run yet'}
        </span>
      </div>
      {error && <p className="rr-error" role="alert">{error}</p>}
      {status.results.length > 0 && (
        <table className="rr-table" data-testid="rr-bench-table">
          <thead><tr><th scope="col">Threads</th><th scope="col">Pool</th><th scope="col">Samples/s</th></tr></thead>
          <tbody>
            {status.results.map(row => {
              const isBest = !!best && best.execution_threads === row.execution_threads && best.pool_size === row.pool_size;
              return <tr key={`${row.execution_threads}-${row.pool_size}`} className={isBest ? 'rr-best' : undefined}>
                <td>{row.execution_threads}</td><td>{row.pool_size}</td><td>{row.samples_per_second.toFixed(1)}{isBest ? ' ★' : ''}</td></tr>;
            })}
          </tbody>
        </table>
      )}
      {best && status.state === 'DONE' && (
        <p className="rr-reco" data-testid="rr-bench-reco">
          Fastest measured: <b>{best.execution_threads}</b> execution thread(s) × pool <b>{best.pool_size}</b> ({best.samples_per_second.toFixed(1)} samples/s{status.provider ? `, ${status.provider.replace('ExecutionProvider', '')}` : ''}).
          {status.note && <small> {status.note}</small>}
        </p>
      )}
    </section>
  );
}

/**
 * Mirror of the Pinokio terminal's 14-section runtime report, read from the same collectors (`GET /system/runtime-report`), with a
 * one-click copy of the plain-text form and the benchmark runner. Loaded when opened, not on page load: building it probes hardware.
 */
export default function RuntimeReportPanel() {
  const [report, setReport] = useState<RuntimeReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState('');

  const load = useCallback(async () => {
    setLoading(true); setError('');
    try { setReport(parseRuntimeReport(await api<unknown>('/system/runtime-report'))); }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not read the runtime report.'); }
    finally { setLoading(false); }
  }, []);

  const copy = async () => {
    if (!report?.text) return;
    const ok = await copyText(report.text);
    setCopied(ok ? `Copied the ${report.sections.length}-section report (${report.text.length.toLocaleString()} characters).` : 'Copy was blocked by the browser. Select the report text and copy it manually.');
    pushToast(ok ? { kind: 'ok', text: 'Structured runtime report copied to the clipboard.' } : { kind: 'error', title: 'Could not copy', text: 'The browser blocked clipboard access.' });
  };

  const counts = report?.counts ?? {};
  return (
    <details className="rr-panel" data-testid="runtime-report-panel" onToggle={event => { if ((event.currentTarget as HTMLDetailsElement).open && !report && !loading) void load(); }}>
      <summary><span>Structured runtime report</span><small className="muted">14 sections · same as the Pinokio terminal</small></summary>
      <div className="rr-actions">
        <button type="button" className="button small" onClick={() => void copy()} disabled={!report?.text} data-testid="rr-copy">Copy Structured Runtime Report</button>
        <button type="button" className="button small" onClick={() => void load()} disabled={loading} data-testid="rr-refresh">{loading ? 'Reading…' : 'Refresh'}</button>
      </div>
      <p className="rr-copied" role="status" data-testid="rr-copied">{copied}</p>
      {error && <p className="rr-error" role="alert">{error}</p>}
      {report && (
        <>
          <p className="rr-summary" data-testid="rr-summary">
            <b>{report.overall}</b> · v{report.version} · {new Date(report.generated_at).toLocaleTimeString()} ·{' '}
            {counts.OK ?? 0} ok, {counts.INFO ?? 0} info, {counts.WARN ?? 0} warn, {counts.UNAVAILABLE ?? 0} unavailable
          </p>
          {report.sections.map(section => <Section key={section.key} section={section} />)}
        </>
      )}
      {!report && !error && loading && <p className="muted">Reading hardware and caches…</p>}
      <BenchmarkRunner />
    </details>
  );
}
