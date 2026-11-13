import { useEffect, useRef, useState } from 'react';
import Modal from '../controls/Modal';
import { PreviewBusyError, postLivePreview, type LivePreviewResult } from '../../api';
import { PreviewURL } from '../../state';
import { PANELS, SETTINGS, SETTING_BY_ID, inactiveReason, settingValue, type PanelId, type SettingId } from '../controls/settingsRegistry';
import { DEFAULT_PARAMETERS, type WorkstationParameters } from './pipeline';
import type { LivePayload } from '../../livePreview';
import { ENHANCER_CHOICES, RELIGHT_LEVELS, type EnhancerChoice, type Patch } from './sweepPatch';
import ZoomableImage from './ZoomableImage';

export type { Patch } from './sweepPatch';
export const LEVELS = ['Low', 'Medium', 'High'] as const;

type AxisId = SettingId | 'enhancer' | 'relight_strength';
type Value = number | EnhancerChoice;
type Tile = { row: number; column: number; x: Value; y: Value; url: string | null; state: 'pending' | 'loading' | 'ready' | 'error' | 'busy'; message?: string };

/** One sweepable dimension: a continuous setting (Low / Medium / High) or a categorical / stepped control. */
type Axis = {
  id: AxisId; label: string; group: string; values: [Value, Value, Value];
  /** Why the engine would ignore this axis for the current parameters (the selected restorer, occlusion mode, blend), or null when it acts. */
  inactive?: (parameters: WorkstationParameters) => string | null;
  /** Short text for the value, used in accessible names. */
  text: (value: Value, index: number) => string;
  /** What a tile shows under its image for this axis. */
  caption: (value: Value, index: number) => string;
  /** The alternative text fragment for this axis in a tile's image. */
  alt: (index: number) => string;
  current: (parameters: WorkstationParameters) => string;
  /** Hint under the selector listing the three values. */
  levels: () => string;
};

const continuous = (id: SettingId): Axis => {
  const spec = SETTING_BY_ID[id];
  return {
    id, label: spec.label, group: PANELS[spec.panel].title, inactive: parameters => inactiveReason(spec, parameters), values: spec.sweep,
    text: value => String(value), caption: (value, index) => `${LEVELS[index]} ${spec.label} ${value}`, alt: index => `${LEVELS[index]} ${spec.label}`,
    current: parameters => String(settingValue(spec, parameters)),
    levels: () => LEVELS.map((level, i) => `${level} ${spec.sweep[i]}`).join(' · '),
  };
};
const percent = (value: Value) => `${Math.round(Number(value) * 100)}%`;
const choiceLabel = (value: Value) => ENHANCER_CHOICES.find(choice => choice.id === value)?.label ?? String(value);

const ENHANCER_AXIS: Axis = {
  id: 'enhancer', label: 'Enhancer', group: 'Restorer', values: ENHANCER_CHOICES.map(choice => choice.id) as [EnhancerChoice, EnhancerChoice, EnhancerChoice],
  text: choiceLabel, caption: value => `Enhancer: ${choiceLabel(value)}`, alt: index => `${ENHANCER_CHOICES[index].label} enhancer`,
  current: parameters => (parameters.restoration_blend > 0 && !parameters.bypass.includes('restorer') ? choiceLabel(parameters.restorer_model === 'ultramax' ? 'ultramax' : 'gpen') : 'None'),
  levels: () => ENHANCER_CHOICES.map(choice => choice.label).join(' · '),
};
const RELIGHT_AXIS: Axis = {
  id: 'relight_strength', label: 'SH relighting strength', group: '3D Relighting', values: RELIGHT_LEVELS,
  text: percent, caption: value => `SH relighting ${percent(value)}`, alt: index => `${percent(RELIGHT_LEVELS[index])} SH relighting`,
  current: parameters => (parameters.relight ? percent(parameters.relight_strength ?? 0.8) : 'off'),
  levels: () => RELIGHT_LEVELS.map(percent).join(' · '),
};

/** Settings first (so panel defaults and ordering are unchanged), then the two stepped axes. */
export const AXES: Axis[] = [...SETTINGS.map(spec => continuous(spec.id)), ENHANCER_AXIS, RELIGHT_AXIS];
const AXIS_BY_ID = Object.fromEntries(AXES.map(axis => [axis.id, axis])) as Record<AxisId, Axis>;

type Props = {
  /** Live-preview payload for the current frame with `patch` applied on top of the current settings, or null when no face is mapped. */
  payloadFor: (patch: Patch) => LivePayload | null;
  parameters: WorkstationParameters;
  frameLabel: string;
  onApply: (patch: Patch) => void;
  onClose: () => void;
  /** Panel the sweep was opened from: its first two settings are pre-selected. */
  panel?: PanelId;
  /** Test seam; defaults to POST /api/preview. */
  render?: (payload: LivePayload, signal: AbortSignal) => Promise<LivePreviewResult>;
  maxBusyRetries?: number;
};

export function defaultAxes(panel?: PanelId, parameters: WorkstationParameters = DEFAULT_PARAMETERS): [SettingId, SettingId] {
  const active = SETTINGS.filter(spec => !inactiveReason(spec, parameters));
  const inPanel = active.filter(spec => spec.panel === panel).map(spec => spec.id);
  const rest = active.filter(spec => !inPanel.includes(spec.id)).map(spec => spec.id);
  const pool = [...inPanel, ...rest];
  return [pool[0] ?? 'tps_expression_weight', pool[1] ?? 'pose_strength'];
}

/**
 * 3×3 contact sheet: any two controls across three values each, rendered as nine real `POST /api/preview` frames of the
 * current playhead (so each tile is cache-addressable and obeys the 503 busy contract). Each tile can be inspected at native
 * pixel size; clicking a tile applies that tile's two values.
 */
export default function ContactSheetModal({ payloadFor, parameters, frameLabel, onApply, onClose, panel, render = (payload, signal) => postLivePreview(payload, { signal }), maxBusyRetries = 20 }: Props) {
  const [initialX, initialY] = defaultAxes(panel, parameters);
  const [xId, setX] = useState<AxisId>(initialX), [yId, setY] = useState<AxisId>(initialY);
  const [tiles, setTiles] = useState<Tile[]>([]), [busy, setBusy] = useState(false), [error, setError] = useState('');
  const [zoomed, setZoomed] = useState<number | null>(null);
  const abort = useRef<AbortController | null>(null), leases = useRef<PreviewURL[]>([]), generation = useRef(0);
  const xAxis = AXIS_BY_ID[xId], yAxis = AXIS_BY_ID[yId];

  const release = () => { for (const lease of leases.current) lease.clear(); leases.current = []; };
  const cancel = () => { generation.current++; abort.current?.abort(); setBusy(false); };
  useEffect(() => () => { generation.current++; abort.current?.abort(); release(); }, []);
  // Changing an axis invalidates the sheet.
  useEffect(() => { cancel(); release(); setTiles([]); setError(''); setZoomed(null); }, [xId, yId]);

  const patchOf = (tile: Tile): Patch => ({ [xId]: tile.x, [yId]: tile.y } as Patch);

  const generate = async () => {
    if (xId === yId) { setError('Choose two different settings.'); return; }
    const token = ++generation.current, controller = new AbortController();
    abort.current = controller; release(); setZoomed(null);
    setBusy(true); setError('');
    const grid: Tile[] = [];
    for (let row = 0; row < 3; row++) for (let column = 0; column < 3; column++) grid.push({ row, column, x: xAxis.values[column], y: yAxis.values[row], url: null, state: 'pending' });
    setTiles(grid);
    const update = (index: number, change: Partial<Tile>) => { if (token === generation.current) setTiles(previous => previous.map((tile, i) => (i === index ? { ...tile, ...change } : tile))); };
    for (let index = 0; index < grid.length && token === generation.current; index++) {
      const tile = grid[index];
      const payload = payloadFor(patchOf(tile));
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

  const choose = (tile: Tile) => { onApply(patchOf(tile)); onClose(); };
  const axis = (label: string, value: AxisId, change: (id: AxisId) => void, other: AxisId) => {
    const current = AXIS_BY_ID[value];
    return (
      <label className="sweep-axis">{label}
        <select aria-label={`${label} setting`} value={value} onChange={event => change(event.target.value as AxisId)}>
          {AXES.map(item => {
            const inert = item.inactive?.(parameters) ?? null;
            return <option key={item.id} value={item.id} disabled={!!inert || item.id === other} title={inert ?? undefined}>
              {item.group} · {item.label}{inert ? ' (inactive)' : ''}</option>;
          })}
        </select>
        <small className="muted">{current.levels()}</small>
      </label>
    );
  };
  const zoomTile = zoomed !== null ? tiles[zoomed] : null;
  const describe = (tile: Tile) => `${xAxis.alt(tile.column)}, ${yAxis.alt(tile.row)}`;

  return (
    <Modal title="3×3 Sweep" close={onClose}>
      <p className="muted">{frameLabel} · nine real previews of this frame · click a tile to apply both values. Current: {xAxis.label} {xAxis.current(parameters)}, {yAxis.label} {yAxis.current(parameters)}.</p>
      <div className="sweep-axes">{axis('X axis', xId, setX, yId)}{axis('Y axis', yId, setY, xId)}</div>
      <button className="button accent" disabled={busy} onClick={() => void generate()}>{busy ? 'Rendering nine previews…' : 'Generate contact sheet'}</button>
      {busy && <button className="button" onClick={cancel}>Cancel sweep</button>}
      {error && <p role="alert" className="inline-warning">{error}</p>}
      {zoomTile?.url && (
        <section className="sweep-zoom" aria-label="Zoomed tile" data-testid="sweep-zoom">
          <p className="muted">{xAxis.caption(zoomTile.x, zoomTile.column)} · {yAxis.caption(zoomTile.y, zoomTile.row)}</p>
          <ZoomableImage src={zoomTile.url} alt={describe(zoomTile)} />
          <div className="sweep-zoom-actions">
            <button type="button" className="button" onClick={() => setZoomed(null)}>Back to sheet</button>
            <button type="button" className="button accent" onClick={() => choose(zoomTile)}>Apply these settings</button>
          </div>
        </section>
      )}
      {tiles.length > 0 && (
        <div className="sweep-grid" role="grid" aria-label={`${xAxis.label} across, ${yAxis.label} down`} style={zoomTile?.url ? { display: 'none' } : undefined}>
          {tiles.map((tile, index) => (
            <div className="sweep-cell-wrap" key={`${tile.row}-${tile.column}`}>
              <button className={`sweep-cell ${tile.state}`} disabled={tile.state !== 'ready'}
                aria-label={`Apply ${xAxis.label} ${xAxis.text(tile.x, tile.column)} and ${yAxis.label} ${yAxis.text(tile.y, tile.row)}`}
                onClick={() => choose(tile)}>
                {tile.url ? <img src={tile.url} alt={describe(tile)} /> : <span className="sweep-pending">{tile.message ?? (tile.state === 'loading' ? 'Rendering…' : 'Queued')}</span>}
                <span className="sweep-caption">{xAxis.caption(tile.x, tile.column)} · {yAxis.caption(tile.y, tile.row)}</span>
              </button>
              {tile.state === 'ready' && <button type="button" className="sweep-zoom-button button small" aria-label={`Zoom tile ${index + 1}`} onClick={() => setZoomed(index)}>Zoom</button>}
            </div>
          ))}
        </div>
      )}
    </Modal>
  );
}
