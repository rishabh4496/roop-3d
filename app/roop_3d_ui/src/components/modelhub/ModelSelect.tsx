import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import {
  ACTIVE_DOWNLOAD, STATUS_META, cancelModelDownload, importLocalWeights, isInstalled, modelLabel, modelStatus, refreshCatalog, resolutionBadge, startModelDownload,
  useModelCatalog, type CatalogModel, type ModelStatus,
} from '../../stores/modelCatalog';
import './modelRack.css';

/**
 * One choice in a rack selector.
 * `modelId` is the catalog artifact it needs (defaults to `id`); `null` means it needs no weights ("None", engines built into the 3D stack).
 */
export type RackOption = { id: string; label?: string; modelId?: string | null; note?: string; disabledReason?: string; badge?: string };

type Props = {
  label: string; category: string; value: string | null; onChange: (id: string) => void;
  options: RackOption[];
  /** Also list every other registered artifact of this category (never hide a registered model). */
  includeRegistered?: boolean;
  /** When set, registered artifacts outside `options` are listed but cannot be chosen, with this reason. */
  registeredDisabledReason?: string;
  /** Short text under the selector (what the control changes). */
  hint?: string; testId?: string;
};

export function StatusPill({ status, compact = false }: { status: ModelStatus; compact?: boolean }) {
  const meta = STATUS_META[status];
  return <span className={`rack-pill rack-pill-${status}`} title={meta.hint}><span aria-hidden="true">{meta.glyph}</span>{compact ? <span className="sr-only">{meta.label}</span> : <span>{meta.label}</span>}</span>;
}

type Resolved = RackOption & { model: CatalogModel | null; status: ModelStatus; title: string };

export default function ModelSelect({ label, category, value, onChange, options, includeRegistered = true, registeredDisabledReason, hint, testId }: Props) {
  const { models, loading, error } = useModelCatalog();
  const uid = useId();
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState<string | null>(null);
  const [active, setActive] = useState(0);
  const listRef = useRef<HTMLUListElement>(null);

  const resolved = useMemo<Resolved[]>(() => {
    const byId = new Map(models.map(item => [item.id, item]));
    const make = (option: RackOption): Resolved => {
      const model = option.modelId === null ? null : byId.get(option.modelId ?? option.id) ?? null;
      const status: ModelStatus = option.modelId === null ? 'ready' : model ? modelStatus(model) : 'locate';
      const res = model ? resolutionBadge(model) : null;
      const base = option.label ?? (model ? modelLabel(model) : option.id);
      return { ...option, model, status, title: `${base}${option.badge ? ` • ${option.badge}` : res ? ` • ${res}` : ''}` };
    };
    const listed = options.map(make);
    if (!includeRegistered) return listed;
    const known = new Set(options.map(option => option.modelId ?? option.id));
    // The registry can hold several records for one filename (different byte builds, or the same build recorded twice). Show each distinct
    // build once, labelled with its SHA-256 so it is never mistaken for the build the render engine pins.
    const seen = new Set<string>();
    const rest = models.filter(model => {
      if (model.category !== category || known.has(model.id)) return false;
      const build = `${model.name}|${model.sha256 ?? model.id}`;
      if (seen.has(build)) return false;
      seen.add(build);
      return true;
    }).map(model => make({ id: model.id, label: `${modelLabel(model)}${model.sha256 ? ` · sha ${model.sha256.slice(0, 8)}` : ''}`, disabledReason: registeredDisabledReason }));
    return [...listed, ...rest];
  }, [models, options, category, includeRegistered, registeredDisabledReason]);

  const current = resolved.find(item => item.id === value) ?? null;
  const restStart = options.length;
  const target = pending ? resolved.find(item => item.id === pending) ?? null : null;

  // Auto-activate: once the pending model's weights show up (download finished or a local copy was imported), switch to it.
  useEffect(() => {
    if (target && isInstalled(target.status)) { onChange(target.id); setPending(null); }
  }, [target, onChange]);

  useEffect(() => { if (open) setActive(Math.max(0, resolved.findIndex(item => item.id === value))); }, [open]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (open) listRef.current?.querySelector<HTMLElement>(`[data-index="${active}"]`)?.scrollIntoView?.({ block: 'nearest' }); }, [active, open]);

  const choose = (item: Resolved) => {
    if (item.disabledReason) return;
    setOpen(false);
    if (item.status === 'blocked' || (!isInstalled(item.status) && item.modelId !== null)) { setPending(item.id); return; }
    setPending(null);
    onChange(item.id);
  };
  const onKey = (event: KeyboardEvent<HTMLElement>) => {
    if (!open) { if (event.key === 'ArrowDown' || event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setOpen(true); } return; }
    if (event.key === 'Escape') { event.preventDefault(); setOpen(false); }
    else if (event.key === 'ArrowDown') { event.preventDefault(); setActive(index => Math.min(resolved.length - 1, index + 1)); }
    else if (event.key === 'ArrowUp') { event.preventDefault(); setActive(index => Math.max(0, index - 1)); }
    else if (event.key === 'Home') { event.preventDefault(); setActive(0); }
    else if (event.key === 'End') { event.preventDefault(); setActive(resolved.length - 1); }
    else if (event.key === 'Enter') { event.preventDefault(); const item = resolved[active]; if (item) choose(item); }
  };

  return (
    <div className="rack-select" data-testid={testId}>
      <span className="rack-select-label" id={`${uid}-label`}>{label}</span>
      <button type="button" className="rack-select-trigger" role="combobox" aria-haspopup="listbox" aria-expanded={open} aria-controls={`${uid}-list`}
        aria-labelledby={`${uid}-label ${uid}-value`} onClick={() => setOpen(value => !value)} onKeyDown={onKey}>
        <span id={`${uid}-value`} className="rack-select-value">{current ? current.title : value ?? (loading ? 'Loading…' : 'Choose…')}</span>
        {current && <StatusPill status={current.status} />}
        <span aria-hidden="true" className="rack-caret">▾</span>
      </button>
      {open && (
        <ul id={`${uid}-list`} ref={listRef} className="rack-select-list" role="listbox" aria-labelledby={`${uid}-label`} tabIndex={-1} onKeyDown={onKey}>
          {resolved.map((item, index) => (
            <li key={item.id} role="option" data-index={index} aria-selected={item.id === value} aria-disabled={!!item.disabledReason}
              className={`rack-option ${index === active ? 'active' : ''} ${item.id === value ? 'selected' : ''} ${item.disabledReason ? 'disabled' : ''}`}
              onMouseEnter={() => setActive(index)} onClick={() => choose(item)} title={item.disabledReason ?? item.note}>
              {index === restStart && restStart > 0 && <span className="rack-group-title" role="presentation">Other registered {category} artifacts</span>}
              <span className="rack-option-main"><span className="rack-option-name">{item.title}</span>
                {(item.disabledReason ?? item.note) && <small>{item.disabledReason ?? item.note}</small>}</span>
              <StatusPill status={item.status} />
            </li>
          ))}
          {resolved.length === 0 && <li className="rack-option disabled" role="presentation">{error || (loading ? 'Loading model list…' : 'No models registered for this category.')}</li>}
        </ul>
      )}
      {error && !models.length && <small className="identity-warn" role="alert">Model list unavailable: {error} <button type="button" className="link-button" onClick={() => void refreshCatalog()}>Retry</button></small>}
      {hint && <small className="muted">{hint}</small>}
      {target && <AcquirePanel item={target} onDismiss={() => setPending(null)} />}
    </div>
  );
}

/** Inline, non-blocking: downloads (with progress and cancel) or imports a local copy, then the selector activates the model by itself. */
function AcquirePanel({ item, onDismiss }: { item: Resolved; onDismiss: () => void }) {
  const model = item.model;
  const { downloads } = useModelCatalog();
  const job = model ? downloads[model.id] : undefined;
  const running = !!job && ACTIVE_DOWNLOAD.includes(job.state);
  const [accepted, setAccepted] = useState(false);
  const [path, setPath] = useState('');
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [problem, setProblem] = useState('');
  const fail = (reason: unknown, fallback: string) => setProblem(reason instanceof Error ? reason.message : fallback);

  if (!model) {
    return <div className="rack-acquire" role="status"><b>{item.title}</b><small>This artifact is not in the model registry, so it cannot be installed from here.</small><button type="button" className="button small" onClick={onDismiss}>Dismiss</button></div>;
  }
  if (item.status === 'blocked') {
    return <div className="rack-acquire" role="alert"><b>{item.title} is blocked</b><small>{model.license_notice || 'The registry blocks this artifact until its licence and provenance are reviewed.'}</small><button type="button" className="button small" onClick={onDismiss}>Dismiss</button></div>;
  }
  const download = async () => {
    setProblem(''); setNotice('');
    try { await startModelDownload(model, accepted); } catch (reason) { fail(reason, 'The download could not start.'); }
  };
  const locate = async () => {
    if (!path.trim()) return;
    setBusy(true); setProblem(''); setNotice('');
    try {
      const report = await importLocalWeights(path);
      const matched = (report.imported?.length ?? 0) + (report.skipped?.length ?? 0);
      if (!matched) setProblem(report.corrupted?.length ? `Found ${report.corrupted[0].file} but it does not match the registered SHA-256 / size: ${report.corrupted[0].error}` : `No file in that location matches a registered weight for ${item.title}. The server only installs files whose SHA-256 and size match the registry.`);
      else setNotice(`Imported ${matched} verified file${matched === 1 ? '' : 's'}.`);
    } catch (reason) { fail(reason, 'The local import failed.'); } finally { setBusy(false); }
  };
  return (
    <div className="rack-acquire" role="group" aria-label={`Install ${item.title}`} data-testid="rack-acquire">
      <b>{item.title} is not in app/models yet</b>
      {running && job ? (
        <div className="rack-progress-row">
          <progress value={job.percent} max={100} aria-label={`Downloading ${item.title}`}>{job.percent.toFixed(0)}%</progress>
          <span>{job.state === 'cancelling' ? 'Cancelling…' : `${job.percent.toFixed(0)}%`}</span>
          <button type="button" className="button small" disabled={job.state === 'cancelling'} onClick={() => void cancelModelDownload(model.id).catch(reason => fail(reason, 'Could not cancel.'))}>Cancel</button>
        </div>
      ) : (
        <>
          {item.status === 'download' && (
            <div className="rack-acquire-row">
              {model.manual_acceptance_required && <label className="rack-license"><input type="checkbox" checked={accepted} onChange={event => setAccepted(event.target.checked)} /> I reviewed the licence ({model.license_id || 'unknown'}).</label>}
              <button type="button" className="button small primary" disabled={model.manual_acceptance_required && !accepted} onClick={() => void download()}>Download &amp; activate</button>
            </div>
          )}
          {item.status === 'locate' && <small className="muted">No public source with identical bytes is registered for this model, so it cannot be downloaded here.</small>}
          <div className="rack-acquire-row">
            <input aria-label={`Local folder or .zip containing ${item.title} weights`} placeholder="Folder or .zip with the .onnx file" value={path} onChange={event => setPath(event.target.value)} />
            <button type="button" className="button small" disabled={busy || !path.trim()} onClick={() => void locate()}>{busy ? 'Verifying…' : 'Locate local .onnx'}</button>
          </div>
        </>
      )}
      {job?.state === 'failed' && <small className="identity-warn" role="alert">Download failed{job.error_code ? ` (${job.error_code})` : ''}. You can locate a local copy instead.</small>}
      {problem && <small className="identity-warn" role="alert">{problem}</small>}
      {notice && <small role="status">{notice}</small>}
      <button type="button" className="link-button" onClick={onDismiss}>Keep current model</button>
    </div>
  );
}
