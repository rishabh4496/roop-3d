import { useMemo, useState } from 'react';
import { post } from '../../api';
import {
  ACTIVE_DOWNLOAD, importLocalWeights, isInstalled, modelStatus, modelTitle, rescanModels, startModelDownload, useModelCatalog, type CatalogModel,
} from '../../stores/modelCatalog';
import { StatusPill } from './ModelSelect';
import './modelRack.css';

const CATEGORIES = ['swapper', 'enhancer', 'detector', 'occluder', 'upscaler', 'other'] as const;
const NO_VRAM_API = 'The engine loads models on first use and releases them through the VRAM governor. There is no manual preload / unload API yet.';
const sizeLabel = (bytes: number | null) => (bytes == null ? '—' : bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GiB` : `${(bytes / 1024 ** 2).toFixed(1)} MiB`);

type Verify = { status?: string; verified?: number; failed?: number; [key: string]: unknown };

/** Settings → Models & Engines Manager: rescan, offline import, integrity check, and per-model actions. Every registered model is listed. */
export default function ModelManager() {
  const { models, loading, error, downloads } = useModelCatalog();
  const [category, setCategory] = useState<(typeof CATEGORIES)[number] | 'all'>('swapper');
  const [search, setSearch] = useState('');
  const [path, setPath] = useState('');
  const [busy, setBusy] = useState('');
  const [message, setMessage] = useState('');
  const [problem, setProblem] = useState('');
  const [rowMessage, setRowMessage] = useState<Record<string, string>>({});

  const rows = useMemo(() => models.filter(model => {
    const group = (CATEGORIES as readonly string[]).includes(model.category) ? model.category : 'other';
    return (category === 'all' || group === category) && `${model.name} ${model.id}`.toLowerCase().includes(search.trim().toLowerCase());
  }), [models, category, search]);
  const installed = models.filter(model => model.installed).length;

  const act = async (name: string, work: () => Promise<string>) => {
    setBusy(name); setMessage(''); setProblem('');
    try { setMessage(await work()); } catch (reason) { setProblem(reason instanceof Error ? reason.message : `${name} failed.`); } finally { setBusy(''); }
  };
  const rescan = () => act('rescan', async () => { const result = await rescanModels(); return `Rescanned app/models: ${result.installed} of ${result.models} registered models are installed.`; });
  const importPath = () => act('import', async () => {
    const report = await importLocalWeights(path);
    const parts = [`${report.imported?.length ?? 0} imported`, `${report.skipped?.length ?? 0} already installed`, `${report.corrupted?.length ?? 0} rejected (hash/size mismatch)`];
    if (!report.imported?.length && !report.skipped?.length) throw new Error(`No registered weights found there (${parts.join(', ')}). Only files whose SHA-256 and size match a registry record are installed.`);
    return `Import finished: ${parts.join(', ')}.`;
  });
  const verifyAll = () => act('verify', async () => {
    const result = await post<Verify>('/installer/models/verify', {});
    return `Integrity check: ${result.status ?? 'done'}${typeof result.verified === 'number' ? `, ${result.verified} verified` : ''}${typeof result.failed === 'number' ? `, ${result.failed} failed` : ''}.`;
  });
  const prebuildTrt = async (model: CatalogModel) => {
    setRowMessage(value => ({ ...value, [model.id]: 'Requesting…' }));
    try {
      const result = await post<{ status?: string }>('/models/compile_trt', { model_id: model.id, precision: 'fp16' });
      setRowMessage(value => ({ ...value, [model.id]: `TensorRT FP16 build: ${result.status ?? 'accepted'}.` }));
    } catch (reason) { setRowMessage(value => ({ ...value, [model.id]: reason instanceof Error ? reason.message : 'The TensorRT build could not start.' })); }
  };
  const download = async (model: CatalogModel) => {
    setRowMessage(value => ({ ...value, [model.id]: '' }));
    try { await startModelDownload(model, false); } catch (reason) { setRowMessage(value => ({ ...value, [model.id]: reason instanceof Error ? reason.message : 'The download could not start.' })); }
  };

  return (
    <section className="model-manager" data-testid="model-manager" aria-labelledby="model-manager-title">
      <div>
        <span className="eyebrow">MODELS &amp; ENGINES MANAGER</span>
        <h2 id="model-manager-title" style={{ margin: '4px 0' }}>Local weights</h2>
        <p className="muted" style={{ margin: 0 }}>{loading ? 'Reading the registry…' : `${models.length} registered models, ${installed} installed in app/models. Models that are not installed are listed too, never hidden.`}</p>
      </div>

      <div className="model-manager-row">
        <button type="button" className="button small" disabled={!!busy} onClick={() => void rescan()} data-testid="rescan-models-btn">{busy === 'rescan' ? 'Rescanning…' : 'Rescan app/models folder'}</button>
        <button type="button" className="button small" disabled={!!busy} onClick={() => void verifyAll()}>{busy === 'verify' ? 'Verifying…' : 'Verify SHA-256 of installed models'}</button>
      </div>
      <div className="model-manager-row">
        <input aria-label="Folder or .zip with model weights" placeholder="Folder or .zip containing .onnx weights" value={path} onChange={event => setPath(event.target.value)} />
        <button type="button" className="button small primary" disabled={!!busy || !path.trim()} onClick={() => void importPath()}>{busy === 'import' ? 'Importing…' : 'Import local .onnx model'}</button>
      </div>

      <div className="notice-box" data-testid="sha-policy">
        <label className="inspector-toggle"><input type="checkbox" checked disabled readOnly aria-label="SHA-256 verification" /> SHA-256 verification: always on</label>
        <small className="muted">The model provider opens only files whose SHA-256 and size match a registry record, so there is no “force unblock” switch for registry or downloaded models. The exception is a weight you place yourself in app/models: any file there over 1 MB is trusted by its own SHA-256 (logged as “Auto-trusted local weight file”), shown as verified and never blocked.</small>
      </div>

      {error && <div role="alert" className="notice-box">Model list unavailable: {error}</div>}
      {message && <div role="status" className="notice-box">{message}</div>}
      {problem && <div role="alert" className="notice-box">{problem}</div>}

      <div className="model-manager-row">
        <select aria-label="Model category" value={category} onChange={event => setCategory(event.target.value as typeof category)}>
          <option value="all">All categories ({models.length})</option>
          {CATEGORIES.map(item => <option key={item} value={item}>{item} ({models.filter(model => ((CATEGORIES as readonly string[]).includes(model.category) ? model.category : 'other') === item).length})</option>)}
        </select>
        <input aria-label="Search registered models" placeholder="Search models" value={search} onChange={event => setSearch(event.target.value)} />
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="model-manager-table">
          <thead><tr><th>Model</th><th>Status</th><th>Size</th><th>Actions</th></tr></thead>
          <tbody>
            {rows.map(model => {
              const status = modelStatus(model);
              const job = downloads[model.id];
              const running = !!job && ACTIVE_DOWNLOAD.includes(job.state);
              return (
                <tr key={model.id} data-testid={`manager-row-${model.id}`}>
                  <td>{modelTitle(model)}<br /><small className="muted">{model.id}</small></td>
                  <td><StatusPill status={status} />{model.vram_loaded && <small> · in VRAM</small>}</td>
                  <td>{sizeLabel(model.size_bytes)}</td>
                  <td>
                    <div className="model-manager-row">
                      {isInstalled(status) ? (<>
                        <button type="button" className="button small" disabled title={NO_VRAM_API}>Preload to VRAM</button>
                        <button type="button" className="button small" disabled title={NO_VRAM_API}>Unload</button>
                        <button type="button" className="button small" onClick={() => void prebuildTrt(model)}>Pre-build TRT FP16</button>
                      </>) : running ? <span role="status">Downloading {job?.percent.toFixed(0)}%</span>
                        : status === 'download' ? <button type="button" className="button small primary" onClick={() => void download(model)}>Download</button>
                          : status === 'blocked' ? <small className="muted">Blocked by the registry</small>
                            : <small className="muted">No public source: import a local copy above</small>}
                    </div>
                    {rowMessage[model.id] && <small role="status">{rowMessage[model.id]}</small>}
                  </td>
                </tr>
              );
            })}
            {rows.length === 0 && <tr><td colSpan={4} className="muted">{loading ? 'Loading…' : 'No registered model matches.'}</td></tr>}
          </tbody>
        </table>
      </div>
      <small className="muted">Preload / Unload are shown but disabled: {NO_VRAM_API}</small>
    </section>
  );
}
