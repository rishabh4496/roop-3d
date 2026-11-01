import { useEffect, useState } from 'react';
import { renderProbe } from './utils/renderProbe';
import type { Capabilities, Config } from './contracts';
import { optionReason } from './state';
import NumericInput from './components/controls/NumericInput';
export function SettingsInspector({ caps, config, save, busy, namespace = 'setting' }: { caps: Capabilities | null; config: Config | null; save: (value: Config) => void; busy: boolean; namespace?: string }) {
  const [draft, setDraft] = useState<Config>({});
  renderProbe('SettingsInspector');
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
        const common = { id: `${namespace}-${name}`, disabled: !!reason || busy, title: reason ?? schema.description };
        return <div className="setting" key={name}>{schema.type !== 'integer' && schema.type !== 'number' && <label htmlFor={common.id}>{label}</label>}
          {schema.enum ? <select {...common} value={String(value ?? '')} onChange={event => update(name, event.target.value)}>
            {schema.enum.map(option => { const unavailable = optionReason(caps, name, String(option)); return <option key={String(option)} value={String(option)} disabled={!!unavailable}>{String(option)}{unavailable ? ' · unavailable' : ''}</option>; })}
          </select> : schema.type === 'boolean' ? <input {...common} type="checkbox" checked={value === true} onChange={event => update(name, event.target.checked)} />
            : schema.type === 'integer' || schema.type === 'number' ? <NumericInput {...common} label={label} value={typeof value === 'number' ? value : null} min={schema.minimum} max={schema.maximum} step={schema.type === 'integer' ? 1 : .01} integer={schema.type === 'integer'} onChange={next => update(name, next ?? 0)} />
              : <input {...common} value={Array.isArray(value) ? value.join(', ') : String(value ?? '')} readOnly={Array.isArray(value)} onChange={event => update(name, event.target.value)} />}
          {reason && <small>{reason}</small>}
          {schema.enum && optionReason(caps, name, String(value)) && !reason && <small>{optionReason(caps, name, String(value))}</small>}
          {schema.enum && !reason && <details><summary>Availability</summary>{schema.enum.filter(option => optionReason(caps, name, String(option))).map(option => <small key={String(option)}>{String(option)} — {optionReason(caps, name, String(option))}</small>)}</details>}
        </div>;
      })}
      <button className="button full" disabled={!caps || !config || busy}>Apply settings</button>
    </form>
    <details className="model-list"><summary>Recorded model artifacts <span>{caps?.models.length ?? 0}</span></summary>{caps?.models.map(model => <div className="model" key={model.model_id}><strong>{model.name}</strong><small>{model.state} · license {model.license_state}</small><code>{model.sha256.slice(0, 16)}…</code></div>)}</details>
  </section>;
}
