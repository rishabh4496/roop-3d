import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { api, post } from '../../api';
import { selectEngine, useSelectedEngine } from '../../stores/engineSelection';

type ModelState = 'AVAILABLE' | 'NOT_INSTALLED' | 'LICENSE_RESTRICTED' | 'INCOMPATIBLE' | 'UNVERIFIED' | 'BROKEN' | 'READY_FOR_INFERENCE';
type Model = {
  model_id: string; name: string; version: string; family: string; engine_type: string; purpose: string;
  source_repository: string | null; weight_files: { filename: string; sha256: string | null; size_bytes: number | null; source_url: string | null }[];
  license: { id: string; commercial_use: string; url: string | null; notice: string; acceptance_required: boolean; related_urls: string[] };
  minimum_vram_gib: number | null; recommended_vram_gib: number | null; dependencies: string[];
  supported_precision: string[]; supported_execution_providers: string[]; provider_evidence: string;
  status: ModelState; status_evidence: string; benchmark_status: string; actual_inference_available: boolean;
  inference_evidence: string; engine_compatibility: string; download_allowed: boolean; download_blocker?: string | null; role: string | null;
};
type Engine = {
  engine_id: string; name: string; family: string; engine_type: string; purpose: string; model_ids: string[];
  minimum_vram_gib: number | null; recommended_vram_gib: number | null; dependencies: string[];
  supported_precision: string[]; supported_execution_providers: string[]; status: ModelState;
  benchmark_status: string; actual_inference_available: boolean; selectable: boolean; selected: boolean; evidence: string;
};
type Download = { download_id: string; model_id: string; state: string; bytes_downloaded: number; total_bytes: number; percent: number; error_code: string | null };
type HubResponse = { models: Model[]; engines: Engine[]; model_states: ModelState[]; execution_policy: string; download_policy: string };

const sizeLabel = (bytes: number | null) => bytes == null ? 'Unknown' : `${(bytes / (1024 ** 2)).toFixed(1)} MiB`;
const vramLabel = (value: number | null) => value == null ? 'Not verified' : `${value} GiB`;

export default function ModelHubPanel({ active }: { active: boolean }) {
  const selectedEngine = useSelectedEngine();
  const [hub, setHub] = useState<HubResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [attempted, setAttempted] = useState(false);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [stateFilter, setStateFilter] = useState<ModelState | 'ALL'>('ALL');
  const [downloads, setDownloads] = useState<Record<string, Download>>({});
  const [licenseAccepted, setLicenseAccepted] = useState<Record<string, boolean>>({});

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [nextHub, activeDownloads] = await Promise.all([
        api<HubResponse>('/models/hub'),
        api<Download[]>('/models/downloads'),
      ]);
      setHub(nextHub);
      setDownloads(current => ({
        ...current,
        ...Object.fromEntries(activeDownloads.map(item => [item.model_id, item])),
      }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not load the local model catalog.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (active && !attempted && !loading) {
      setAttempted(true);
      void load();
    }
  }, [active, attempted, loading, load]);

  useEffect(() => {
    const pending = Object.values(downloads).filter(item => ['queued', 'downloading', 'cancelling'].includes(item.state));
    if (pending.length === 0) return undefined;
    const timer = window.setInterval(() => {
      pending.forEach(item => {
        void api<Download>(`/models/downloads/${item.download_id}`).then(next => {
          setDownloads(current => ({ ...current, [next.model_id]: next }));
          if (next.state === 'completed') void load();
        }).catch(() => undefined);
      });
    }, 600);
    return () => window.clearInterval(timer);
  }, [downloads, load]);

  const filteredModels = useMemo(() => (hub?.models ?? []).filter(model => {
    const matchesState = stateFilter === 'ALL' || model.status === stateFilter;
    const text = `${model.name} ${model.family} ${model.purpose} ${model.model_id}`.toLowerCase();
    return matchesState && text.includes(search.trim().toLowerCase());
  }), [hub, search, stateFilter]);

  const startDownload = async (model: Model) => {
    setError('');
    try {
      const job = await post<Download>(`/models/${encodeURIComponent(model.model_id)}/downloads`, {
        accepted_license: model.license.acceptance_required ? Boolean(licenseAccepted[model.model_id]) : false,
      });
      setDownloads(current => ({ ...current, [model.model_id]: job }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not start the model download.');
    }
  };

  const cancelDownload = async (modelId: string) => {
    const item = downloads[modelId];
    if (!item) return;
    try {
      const result = await api<Download>(`/models/downloads/${item.download_id}`, { method: 'DELETE' });
      setDownloads(current => ({ ...current, [modelId]: result }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not cancel the model download.');
    }
  };

  return (
    <div className="model-hub-container" data-testid="model-hub-panel" style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <section style={{ background: 'var(--surface)', border: '1px solid var(--line)', borderRadius: 8, padding: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <div>
            <span className="eyebrow">LOCAL MODEL CATALOG</span>
            <h2 style={{ margin: '4px 0' }}>Models and engines</h2>
            <p className="muted" style={{ margin: 0 }}>Local files are hash checked when this hub opens. Downloads start only after you choose a model.</p>
          </div>
          <button className="button small" onClick={() => void load()} disabled={loading} data-testid="refresh-models-btn">
            {loading ? 'Checking local files…' : 'Refresh model states'}
          </button>
        </div>
        <p className="muted" style={{ fontSize: 12, marginBottom: 0 }}>Execution: offline by default. Registry presence alone does not mean a weight is installed or an inference path is available.</p>
        {error && <div role="alert" className="notice-box" style={{ marginTop: 12 }}>{error}</div>}
      </section>

      <section aria-labelledby="engine-catalog-title">
        <h3 id="engine-catalog-title">Local engine for new jobs</h3>
        <p className="muted">Selection is saved with each job. Experimental engines remain unavailable until a compatible local adapter is verified.</p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 12 }}>
          {(hub?.engines ?? []).map(engine => (
            <article key={engine.engine_id} data-testid={`engine-card-${engine.engine_id}`} style={{ background: 'var(--surface)', border: '1px solid var(--line)', borderRadius: 8, padding: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                <strong>{engine.name}</strong>
                <span className="status-badge">{engine.engine_id === selectedEngine ? 'CURRENT' : engine.status}</span>
              </div>
              <p className="muted">{engine.purpose}</p>
              <p><strong>State:</strong> {engine.status} · <strong>Inference path:</strong> {engine.actual_inference_available ? 'available' : 'not verified'}</p>
              <p><strong>VRAM:</strong> minimum {vramLabel(engine.minimum_vram_gib)}, recommended {vramLabel(engine.recommended_vram_gib)}</p>
              <p><strong>Precision:</strong> {engine.supported_precision.join(', ') || 'Not verified'}</p>
              <p><strong>Providers:</strong> {engine.supported_execution_providers.join(', ') || 'Not verified'}</p>
              <p><strong>Benchmark:</strong> {engine.benchmark_status}</p>
              <p className="muted">{engine.evidence}</p>
              <button className="button small" disabled={!engine.selectable}
                aria-pressed={engine.engine_id === selectedEngine} onClick={() => selectEngine(engine.engine_id)}>
                {engine.engine_id === selectedEngine ? 'Current production engine' : engine.selectable ? 'Select for new jobs' : 'Unavailable for production jobs'}
              </button>
            </article>
          ))}
        </div>
      </section>

      <section aria-labelledby="model-list-title">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <h3 id="model-list-title">Registered artifacts ({filteredModels.length})</h3>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <input aria-label="Search models" value={search} onChange={event => setSearch(event.target.value)} placeholder="Search models" />
            <select aria-label="Filter by model status" value={stateFilter} onChange={event => setStateFilter(event.target.value as ModelState | 'ALL')}>
              <option value="ALL">All states</option>
              {(hub?.model_states ?? []).map(state => <option key={state} value={state}>{state}</option>)}
            </select>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 12 }}>
          {filteredModels.map(model => {
            const job = downloads[model.model_id];
            const isRunning = Boolean(job && ['queued', 'downloading', 'cancelling'].includes(job.state));
            return (
              <article key={model.model_id} data-testid={`model-card-${model.model_id}`} style={{ background: 'var(--surface)', border: '1px solid var(--line)', borderRadius: 8, padding: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                  <strong>{model.name}</strong><span className="status-badge">{model.status}</span>
                </div>
                <div className="muted">{model.family} · {model.engine_type} · version {model.version}</div>
                <p style={{ margin: 0 }}>{model.purpose}</p>
                <div><strong>License:</strong> {model.license.id} · commercial use: {model.license.commercial_use}</div>
                <p className="muted" style={{ margin: 0 }}>{model.license.notice}</p>
                {model.license.url && <a href={model.license.url} target="_blank" rel="noreferrer">Review model card and license</a>}
                {model.license.related_urls.map(url => <a key={url} href={url} target="_blank" rel="noreferrer">Review separate base-model license</a>)}
                <div><strong>VRAM:</strong> minimum {vramLabel(model.minimum_vram_gib)} · recommended {vramLabel(model.recommended_vram_gib)}</div>
                <div><strong>Precision:</strong> {model.supported_precision.join(', ') || 'Not verified'}</div>
                <div><strong>Execution providers:</strong> {model.supported_execution_providers.join(', ') || 'Not verified'}</div>
                <div><strong>Engine compatibility:</strong> {model.engine_compatibility}</div>
                <div><strong>Benchmark:</strong> {model.benchmark_status} · <strong>Actual inference:</strong> {model.actual_inference_available ? 'available' : 'not verified'}</div>
                <div><strong>Source:</strong> {model.source_repository ? <a href={model.source_repository} target="_blank" rel="noreferrer">{model.source_repository}</a> : 'Not verified'}</div>
                <div><strong>Dependencies:</strong> {model.dependencies.length ? model.dependencies.join('; ') : 'Not recorded'}</div>
                {model.weight_files.map(weight => (
                  <div key={weight.filename} style={{ overflowWrap: 'anywhere', fontSize: 12 }}>
                    <strong>Weight:</strong> {weight.filename} ({sizeLabel(weight.size_bytes)})<br />
                    <span className="muted">SHA-256: {weight.sha256 ?? 'Not recorded'}</span>
                  </div>
                ))}
                <p className="muted" style={{ margin: 0 }}>{model.status_evidence}</p>
                {model.license.acceptance_required && model.download_allowed && (
                  <label style={{ fontSize: 12 }}>
                    <input type="checkbox" checked={Boolean(licenseAccepted[model.model_id])}
                      onChange={event => setLicenseAccepted(current => ({ ...current, [model.model_id]: event.target.checked }))} />
                    {' '}{model.license.id === 'UNKNOWN'
                      ? 'The source publishes no license for these weights. I understand usage rights are unverified and that commercial use is not permitted.'
                      : 'I reviewed the adapter license and separate base-model terms.'}
                  </label>
                )}
                {job && <div role="status">Download {job.state}: {job.percent.toFixed(1)}%{job.error_code ? ` · ${job.error_code}` : ''}</div>}
                {isRunning ? (
                  <button className="button small" onClick={() => void cancelDownload(model.model_id)} disabled={job?.state === 'cancelling'}>Cancel download</button>
                ) : model.download_allowed ? (
                  <button className="button small primary" onClick={() => void startDownload(model)}
                    disabled={Boolean(model.license.acceptance_required && !licenseAccepted[model.model_id])}>
                    Download this model
                  </button>
                ) : <span className="muted">{model.download_blocker ?? 'Download unavailable until provenance and license are reviewed.'}</span>}
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}
