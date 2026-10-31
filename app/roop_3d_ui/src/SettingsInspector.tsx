import { useEffect, useState } from 'react';
import type { Capabilities, Config } from './contracts';
import { optionReason } from './state';
export function SettingsInspector({ caps, config, save, busy }: { caps: Capabilities | null; config: Config | null; save: (value: Config) => void; busy: boolean }) {
  const [draft, setDraft] = useState<Config>({});
  useEffect(() => { if (config) setDraft(config); }, [config]);
  const properties = caps?.legal_parameter_bounds.properties ?? {};
  const update = (name: string, value: string | number | boolean) => setDraft(previous => ({ ...previous, [name]: value }));
  return <section className="settings-panel"><div className="section-title"><span>PROCESSING</span><span className="tiny-badge">LOCAL</span></div>
    {!config && <p className="muted">Connecting to the capability catalog…</p>}
    <form onSubmit={event => { event.preventDefault(); save(draft); }}>
      {Object.entries(properties).map(([name, schema]) => {
        const reason = optionReason(caps, name);
        const value = draft[name];
        const label = name.replaceAll('_', ' ');
        const common = { id: `setting-${name}`, disabled: !!reason || busy, title: reason ?? schema.description };
        return <label className="setting" key={name} htmlFor={common.id}><span>{label}</span>
          {schema.enum ? <select {...common} value={String(value ?? '')} onChange={event => update(name, event.target.value)}>
            {schema.enum.map(option => { const unavailable = optionReason(caps, name, String(option)); return <option key={String(option)} value={String(option)} disabled={!!unavailable}>{String(option)}{unavailable ? ' · unavailable' : ''}</option>; })}
          </select> : schema.type === 'boolean' ? <input {...common} type="checkbox" checked={value === true} onChange={event => update(name, event.target.checked)} />
            : schema.type === 'integer' || schema.type === 'number' ? <input {...common} type="number" value={typeof value === 'number' ? value : ''} min={schema.minimum} max={schema.maximum} step={schema.type === 'integer' ? 1 : 'any'} onChange={event => update(name, event.target.value === '' ? '' : event.target.valueAsNumber)} required />
              : <input {...common} value={Array.isArray(value) ? value.join(', ') : String(value ?? '')} readOnly={Array.isArray(value)} onChange={event => update(name, event.target.value)} />}
          {reason && <small>{reason}</small>}
          {schema.enum && optionReason(caps, name, String(value)) && !reason && <small>{optionReason(caps, name, String(value))}</small>}
          {schema.enum && !reason && <details><summary>Availability</summary>{schema.enum.filter(option => optionReason(caps, name, String(option))).map(option => <small key={String(option)}>{String(option)} — {optionReason(caps, name, String(option))}</small>)}</details>}
        </label>;
      })}
      <button className="button full" disabled={!caps || !config || busy}>Apply settings</button>
    </form>
    <details className="model-list"><summary>Recorded model artifacts <span>{caps?.models.length ?? 0}</span></summary>{caps?.models.map(model => <div className="model" key={model.model_id}><strong>{model.name}</strong><small>{model.state} · license {model.license_state}</small><code>{model.sha256.slice(0, 16)}…</code></div>)}</details>
  </section>;
}
