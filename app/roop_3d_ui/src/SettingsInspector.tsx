import { useEffect, useState } from 'react';
import { renderProbe } from './utils/renderProbe';
import type { Capabilities, Config, Schema } from './contracts';
import { optionReason } from './state';
import NumericInput from './components/controls/NumericInput';
import HardwareExplainer from './components/guide/HardwareExplainer';

const DOMAIN_TABS = [
  { id: 'compute', label: 'Compute', description: 'Hardware acceleration, VRAM allocation, and thread budgets' },
  { id: 'spatial', label: '3D Spatial', description: '3D pose estimation, mesh projection, and temporal smoothing' },
  { id: 'pipeline', label: 'Pipeline', description: 'Restoration models, stream buffer ring, and worker scheduling' },
  { id: 'finishing', label: 'Finishing', description: 'Color transfer, timing policy, audio tracks, and export notices' },
  { id: 'ux', label: 'UX & Diagnostics', description: 'Diagnostics telemetry, concurrent job limits, and log levels' },
  { id: 'advanced', label: 'Advanced', description: 'Storage roots, network transport, and system infrastructure' },
] as const;

type DomainId = typeof DOMAIN_TABS[number]['id'];

const SETTING_DOMAINS: Record<string, DomainId> = {
  hardware_profile: 'compute',
  preferred_provider: 'compute',
  precision_policy: 'compute',
  vram_limit_mb: 'compute',
  memory_safety_reserve_mb: 'compute',
  execution_threads: 'compute',
  stream_batch_size: 'compute',
  stream_ring_slots: 'compute',
  render_backend: 'compute',

  execution_mode: 'spatial',
  pose_hysteresis_degrees: 'spatial',
  temporal_min_cutoff: 'spatial',
  temporal_beta: 'spatial',
  temporal_d_cutoff: 'spatial',

  face_restoration_policy: 'pipeline',
  swapper_model_id: 'pipeline',
  lip_sync_mode: 'pipeline',
  sparse_interval: 'pipeline',
  stream_segment_frames: 'pipeline',
  restorer_threads: 'pipeline',
  restorer_pool_size: 'pipeline',
  ultramax_native_bypass: 'pipeline',
  ultramax_min_pixels: 'pipeline',

  video_timing_policy: 'finishing',
  audio_policy: 'finishing',
  synthetic_media_notice: 'finishing',

  diagnostics_level: 'ux',
  max_concurrent_jobs: 'ux',
};

export function SettingsInspector({
  caps,
  config,
  save,
  busy,
  namespace = 'setting',
}: {
  caps: Capabilities | null;
  config: Config | null;
  save: (value: Config) => void;
  busy: boolean;
  namespace?: string;
}) {
  const [draft, setDraft] = useState<Config>({});
  const [activeTab, setActiveTab] = useState<DomainId>('compute');

  renderProbe('SettingsInspector');

  useEffect(() => {
    if (config) setDraft(config);
  }, [config]);

  const properties = caps?.legal_parameter_bounds.properties ?? {};
  const update = (name: string, value: string | number | boolean) =>
    setDraft(previous => ({ ...previous, [name]: value }));

  // Partition properties by domain
  const categorized: Record<DomainId, [string, Schema][]> = {
    compute: [],
    spatial: [],
    pipeline: [],
    finishing: [],
    ux: [],
    advanced: [],
  };

  for (const [name, schema] of Object.entries(properties)) {
    const domain = SETTING_DOMAINS[name] ?? 'advanced';
    categorized[domain].push([name, schema as Schema]);
  }

  const visibleTabs = DOMAIN_TABS.filter(tab => categorized[tab.id].length > 0);

  // Model lookup helper for pipeline models
  const modelState = (name: string) => {
    const found = caps?.models.find(m => m.name.toLowerCase().includes(name.toLowerCase()));
    return found ? found.state : null;
  };

  return (
    <section className="settings-panel">
      <div className="section-title">
        <span>PROCESSING</span>
        <span className="tiny-badge">LOCAL</span>
      </div>

      {!config && <p className="muted">Connecting to the capability catalog…</p>}

      <div className="settings-domain-tabs" role="tablist" aria-label="Settings categories">
        {visibleTabs.map(tab => {
          const isSelected = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              id={`tab-${tab.id}`}
              aria-selected={isSelected}
              aria-controls={`panel-${tab.id}`}
              className={`settings-domain-tab ${isSelected ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span>{tab.label}</span>
              <span className="tab-count">{categorized[tab.id].length}</span>
            </button>
          );
        })}
      </div>

      <form
        onSubmit={event => {
          event.preventDefault();
          save(draft);
        }}
      >
        {visibleTabs.map(tab => {
          const items = categorized[tab.id];
          const isSelected = activeTab === tab.id;

          return (
            <div
              key={tab.id}
              role="tabpanel"
              id={`panel-${tab.id}`}
              aria-labelledby={`tab-${tab.id}`}
              hidden={!isSelected}
              style={{ display: isSelected ? 'block' : 'none' }}
            >
              <div className="domain-description">{tab.description}</div>

              {tab.id === 'compute' && caps?.workstation && (
                <div className="profile-badge">
                  Active Profile:{' '}
                  <strong>{caps.workstation.active_hardware_profile ?? String(draft.hardware_profile ?? 'auto')}</strong>
                  {caps.workstation.resolved_hardware_profile && (
                    <span> · Resolved: {caps.workstation.resolved_hardware_profile}</span>
                  )}
                </div>
              )}
              {tab.id === 'compute' && <HardwareExplainer caps={caps} configured={String(draft.hardware_profile ?? 'auto')} />}

              {items.map(([name, schema]) => {
                const reason = optionReason(caps, name);
                const value = draft[name];
                const label = name === 'swapper_model_id' ? 'Swapper engine' : name.replaceAll('_', ' ');
                const common = {
                  id: `${namespace}-${name}`,
                  disabled: !!reason || busy,
                  title: reason ?? schema.description,
                };

                return (
                  <div className="setting" key={name}>
                    {schema.type !== 'integer' && schema.type !== 'number' && (
                      <label htmlFor={common.id}>
                        {label}
                        {name === 'face_restoration_policy' && (
                          <span style={{ marginLeft: 6, fontSize: 9, opacity: 0.8 }}>
                            {modelState('gpen') === 'VERIFIED_LOCAL' && modelState('codeformer') === 'VERIFIED_LOCAL'
                              ? '(GPEN & CodeFormer verified)'
                              : modelState('gpen') === 'VERIFIED_LOCAL'
                              ? '(GPEN verified)'
                              : ''}
                          </span>
                        )}
                      </label>
                    )}

                    {schema.enum ? (
                      <select
                        {...common}
                        value={String(value ?? '')}
                        onChange={event => update(name, event.target.value)}
                      >
                        {schema.enum.map((option: any) => {
                          const unavailable = optionReason(caps, name, String(option));
                          return (
                            <option
                              key={String(option)}
                              value={String(option)}
                              disabled={!!unavailable}
                            >
                              {String(option)}
                              {unavailable ? ' · unavailable' : ''}
                            </option>
                          );
                        })}
                      </select>
                    ) : schema.type === 'boolean' ? (
                      <input
                        {...common}
                        type="checkbox"
                        checked={value === true}
                        onChange={event => update(name, event.target.checked)}
                      />
                    ) : schema.type === 'integer' || schema.type === 'number' ? (
                      <NumericInput
                        {...common}
                        label={label}
                        value={typeof value === 'number' ? value : null}
                        min={schema.minimum}
                        max={schema.maximum}
                        step={schema.type === 'integer' ? 1 : 0.01}
                        integer={schema.type === 'integer'}
                        onChange={next => update(name, next ?? 0)}
                      />
                    ) : (
                      <input
                        {...common}
                        value={Array.isArray(value) ? value.join(', ') : String(value ?? '')}
                        readOnly={Array.isArray(value)}
                        onChange={event => update(name, event.target.value)}
                      />
                    )}

                    {reason && <small>{reason}</small>}
                    {schema.enum && optionReason(caps, name, String(value)) && !reason && (
                      <small>{optionReason(caps, name, String(value))}</small>
                    )}
                    {schema.enum && !reason && (
                      <details>
                        <summary>Availability</summary>
                        {schema.enum
                          .filter((option: any) => optionReason(caps, name, String(option)))
                          .map((option: any) => (
                            <small key={String(option)}>
                              {String(option)} — {optionReason(caps, name, String(option))}
                            </small>
                          ))}
                      </details>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}

        <button className="button full" disabled={!caps || !config || busy}>
          Apply settings
        </button>
      </form>

      <details className="model-list">
        <summary>
          Recorded model artifacts <span>{caps?.models.length ?? 0}</span>
        </summary>
        {caps?.models.map(model => (
          <div className="model" key={model.model_id}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <strong>{model.name}</strong>
              <span className={`model-badge ${model.state === 'VERIFIED_LOCAL' ? 'verified' : 'unverified'}`}>
                {model.state}
              </span>
            </div>
            <small>
              {model.license_state ? `license: ${model.license_state}` : ''}{' '}
              {model.inference_state ? `· inference: ${model.inference_state}` : ''}
            </small>
            <code>{model.sha256 ? `${model.sha256.slice(0, 16)}…` : 'Hash not recorded'}</code>
          </div>
        ))}
      </details>
    </section>
  );
}
