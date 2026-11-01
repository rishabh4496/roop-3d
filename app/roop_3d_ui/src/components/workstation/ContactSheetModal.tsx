import { useEffect, useRef, useState } from 'react';
import Modal from '../controls/Modal';
import { PreviewBusyError, postLivePreview, type LivePreviewResult } from '../../api';
import { PreviewURL } from '../../state';
import { PANELS, SETTINGS, SETTING_BY_ID, settingValue, type PanelId, type SettingId } from '../controls/settingsRegistry';
import type { WorkstationParameters } from './pipeline';
import type { LivePayload } from '../../livePreview';

export const LEVELS = ['Low', 'Medium', 'High'] as const;
export type Patch = Partial<Record<SettingId, number>>;
type Tile = { row: number; column: number; x: number; y: number; url: string | null; state: 'pending' | 'loading' | 'ready' | 'error' | 'busy'; message?: string };

type Props = {
  /** Live-preview payload for the current frame with `patch` applied on top of the current settings, or null when no face is mapped. */
  payloadFor: (patch: Patch) => LivePayload | null;
  parameters: WorkstationParameters;
  frameLabel: string;
  onApply: (patch: Patch) => void;
  onClose: () => void;
  /** Panel the sweep was opened from: its first two settings are pre-selected. */
  panel?: PanelId;
  fidelity?: number;
  /** Test seam; defaults to POST /api/preview. */
  render?: (payload: LivePayload, signal: AbortSignal) => Promise<LivePreviewResult>;
  maxBusyRetries?: number;
};

const sweepable = SETTINGS;

export function defaultAxes(panel?: PanelId): [SettingId, SettingId] {
  const inPanel = sweepable.filter(spec => spec.panel === panel && !spec.inactive).map(spec => spec.id);
  const rest = sweepable.filter(spec => !spec.inactive && !inPanel.includes(spec.id)).map(spec => spec.id);
  const pool = [...inPanel, ...rest];
  return [pool[0] ?? 'tps_expression_weight', pool[1] ?? 'pose_strength'];
}

/**
 * 3×3 contact sheet: any two settings across Low / Medium / High, rendered as nine real `POST /api/preview`
 * frames of the current playhead (so each tile is cache-addressable and obeys the 503 busy contract). Clicking a
 * tile applies that tile's two values.
 */
export default function ContactSheetModal({ payloadFor, parameters, frameLabel, onApply, onClose, panel, fidelity, render = (payload, signal) => postLivePreview(payload, { signal }), maxBusyRetries = 20 }: Props) {
  const [initialX, initialY] = defaultAxes(panel);
  const [xId, setX] = useState<SettingId>(initialX), [yId, setY] = useState<SettingId>(initialY);
  const [tiles, setTiles] = useState<Tile[]>([]), [busy, setBusy] = useState(false), [error, setError] = useState('');
  const abort = useRef<AbortController | null>(null), leases = useRef<PreviewURL[]>([]), generation = useRef(0);
  const xSpec = SETTING_BY_ID[xId], ySpec = SETTING_BY_ID[yId];

  const release = () => { for (const lease of leases.current) lease.clear(); leases.current = []; };
  const cancel = () => { generation.current++; abort.current?.abort(); setBusy(false); };
  useEffect(() => () => { generation.current++; abort.current?.abort(); release(); }, []);
  // Changing an axis invalidates the sheet.
  useEffect(() => { cancel(); release(); setTiles([]); setError(''); }, [xId, yId]);

  const generate = async () => {
    if (xId === yId) { setError('Choose two different settings.'); return; }
    const token = ++generation.current, controller = new AbortController();
    abort.current = controller; release();
    setBusy(true); setError('');
    const grid: Tile[] = [];
    for (let row = 0; row < 3; row++) for (let column = 0; column < 3; column++) grid.push({ row, column, x: xSpec.sweep[column], y: ySpec.sweep[row], url: null, state: 'pending' });
    setTiles(grid);
    const update = (index: number, change: Partial<Tile>) => { if (token === generation.current) setTiles(previous => previous.map((tile, i) => (i === index ? { ...tile, ...change } : tile))); };
    for (let index = 0; index < grid.length && token === generation.current; index++) {
      const tile = grid[index];
      const payload = payloadFor({ [xId]: tile.x, [yId]: tile.y });
      if (!payload) { setError('Inspect and map a target face before generating a sweep.'); break; }
      update(index, { state: 'loading' });
      for (let attempt = 0; token === generation.current; attempt++) {
        try {
          const result = await render(payload, controller.signal);
          const lease = new PreviewURL(); leases.current.push(lease);
          update(index, { state: 'ready', url: lease.replace(result.blob) });
          break;
        } catch (e) {
          if (token !== generation.current || (e as Error).name === 'AbortError') break;
          if (e instanceof PreviewBusyError && attempt < maxBusyRetries) {
            update(index, { state: 'busy', message: `GPU busy (${e.stage}); retrying` });
            await new Promise(resolve => setTimeout(resolve, e.retryAfterMs));
            continue;
          }
          update(index, { state: 'error', message: (e as Error).message });
          break;
        }
      }
    }
    if (token === generation.current) setBusy(false);
  };

  const axis = (label: string, value: SettingId, change: (id: SettingId) => void, other: SettingId) => (
    <label className="sweep-axis">{label}
      <select aria-label={`${label} setting`} value={value} onChange={event => change(event.target.value as SettingId)}>
        {sweepable.map(spec => <option key={spec.id} value={spec.id} disabled={!!spec.inactive || spec.id === other} title={spec.inactive}>
          {PANELS[spec.panel].title} · {spec.label}{spec.inactive ? ' (inactive)' : ''}</option>)}
      </select>
      <small className="muted">{LEVELS.map((level, i) => `${level} ${(label === 'X axis' ? xSpec : ySpec).sweep[i]}`).join(' · ')}</small>
    </label>
  );

  return (
    <Modal title="3×3 Sweep" close={onClose}>
      <p className="muted">{frameLabel} · nine real previews of this frame · click a tile to apply both values. Current: {xSpec.label} {settingValue(xSpec, parameters, { fidelity_weight: fidelity })}, {ySpec.label} {settingValue(ySpec, parameters, { fidelity_weight: fidelity })}.</p>
      <div className="sweep-axes">{axis('X axis', xId, setX, yId)}{axis('Y axis', yId, setY, xId)}</div>
      <button className="button accent" disabled={busy} onClick={() => void generate()}>{busy ? 'Rendering nine previews…' : 'Generate contact sheet'}</button>
      {busy && <button className="button" onClick={cancel}>Cancel sweep</button>}
      {error && <p role="alert" className="inline-warning">{error}</p>}
      {tiles.length > 0 && (
        <div className="sweep-grid" role="grid" aria-label={`${xSpec.label} across, ${ySpec.label} down`}>
          {tiles.map(tile => (
            <button key={`${tile.row}-${tile.column}`} className={`sweep-cell ${tile.state}`} disabled={tile.state !== 'ready'}
              aria-label={`Apply ${xSpec.label} ${tile.x} and ${ySpec.label} ${tile.y}`}
              onClick={() => { onApply({ [xId]: tile.x, [yId]: tile.y }); onClose(); }}>
              {tile.url ? <img src={tile.url} alt={`${LEVELS[tile.column]} ${xSpec.label}, ${LEVELS[tile.row]} ${ySpec.label}`} /> : <span className="sweep-pending">{tile.message ?? (tile.state === 'loading' ? 'Rendering…' : 'Queued')}</span>}
              <span className="sweep-caption">{LEVELS[tile.column]} {xSpec.label} {tile.x} · {LEVELS[tile.row]} {ySpec.label} {tile.y}</span>
            </button>
          ))}
        </div>
      )}
    </Modal>
  );
}
