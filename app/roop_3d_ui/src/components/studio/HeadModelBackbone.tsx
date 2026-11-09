import { useEffect, useState } from 'react';
import { api } from '../../api';
import { ErrorRetry } from './FaceSetDetail';

export interface BackboneRow { id: string; label: string; available: boolean; active: boolean; reason: string }
export interface BackboneReport { default: string; active: string | null; backbones: BackboneRow[] }

export const BACKBONE_KEY = 'roop3d-head-backbone';
export const DEFAULT_BACKBONE = '3ddfa_v3_hrn';
export const readBackbone = (): string => { try { return localStorage.getItem(BACKBONE_KEY) || DEFAULT_BACKBONE; } catch { return DEFAULT_BACKBONE; } };
export const writeBackbone = (value: string) => { try { localStorage.setItem(BACKBONE_KEY, value); } catch { /* the choice just is not remembered */ } };

function useBackbones() {
  const [state, setState] = useState<{ report: BackboneReport | null; error: string; attempt: number }>({ report: null, error: '', attempt: 0 });
  useEffect(() => {
    const controller = new AbortController();
    api<BackboneReport>('/reconstruction/backbones', { signal: controller.signal })
      .then(report => setState(value => (Array.isArray(report?.backbones) ? { ...value, report, error: '' } : { ...value, report: null, error: 'unexpected response from the reconstruction service' })))
      .catch((error: Error) => { if (error.name !== 'AbortError') setState(value => ({ ...value, report: null, error: error.message })); });
    return () => controller.abort();
  }, [state.attempt]);
  return { ...state, retry: () => setState(value => ({ ...value, error: '', attempt: value.attempt + 1 })) };
}

/**
 * "3D Head Model Backbone" selector. Availability comes from the server (`GET /reconstruction/backbones`): a family is selectable only when
 * its bundle is really installed, and every unavailable one says why. The server loads exactly one dense-face bundle, so the choice is
 * a saved preference in this browser; it does not swap bundles.
 */
export default function HeadModelBackbone() {
  const { report, error, retry } = useBackbones();
  const [choice, setChoice] = useState(readBackbone);
  if (error) return <ErrorRetry message={`Backbone status unavailable: ${error}`} onRetry={retry} />;
  if (!report) return <div className="skeleton-line" aria-label="Loading 3D head model backbones" />;
  const rows = report.backbones;
  if (!rows.length) return <small className="identity-warn">The reconstruction service reported no backbones.</small>;
  const stored = rows.find(row => row.id === choice);
  // A saved choice that is not installed on this machine falls back to the installed backbone (or the default) instead of showing a lie.
  const selected = stored?.available ? stored : rows.find(row => row.active) ?? rows.find(row => row.id === report.default) ?? rows[0];
  const installed = rows.find(row => row.active);
  return (
    <div className="inspector-field head-backbone" data-testid="head-backbone">
      <label className="inspector-field">3D Head Model Backbone
        <select aria-label="3D head model backbone" title={selected.label} value={selected.id} onChange={event => { writeBackbone(event.target.value); setChoice(event.target.value); }}>
          {rows.map(row => <option key={row.id} value={row.id} disabled={!row.available}>{row.label}{row.available ? '' : ' — unavailable'}</option>)}
        </select>
      </label>
      <small className={selected.available ? 'muted' : 'identity-warn'} data-testid="backbone-reason">
        {selected.reason || (selected.active ? 'Installed: the 3D render modes use this bundle.' : '')}
      </small>
      <small className="muted">
        {installed ? `Installed backbone: ${installed.label}.` : 'No dense-face backbone is installed, so the 3D render modes are unavailable.'}{' '}
        The server loads only the installed bundle; this choice is saved in this browser.
      </small>
      {rows.filter(row => !row.available && row.id !== selected.id).length > 0 && (
        <details className="backbone-unavailable"><summary>Why are some backbones unavailable?</summary>
          <ul>{rows.filter(row => !row.available).map(row => <li key={row.id}><b>{row.label}</b>: {row.reason}</li>)}</ul>
        </details>)}
    </div>
  );
}
