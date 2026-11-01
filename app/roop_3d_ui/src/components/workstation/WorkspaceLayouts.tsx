import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
export const PANEL_IDS = ['timeline', 'tray', 'inspector', 'pose'] as const;
type PanelId = typeof PANEL_IDS[number];
type PanelState = { collapsed: boolean; width: number; height: number };
export type Layout = { order: PanelId[]; panels: Record<PanelId, PanelState>; floating: boolean };
function layout(order: PanelId[], widths: number[], collapsed: PanelId[] = [], floating = false): Layout {
  return { order, floating, panels: Object.fromEntries(order.map((id, i) => [id, { collapsed: collapsed.includes(id), width: widths[i], height: id === 'pose' ? 470 : 340 }])) as Layout['panels'] };
}
export const PRESETS: Record<string, Layout> = {
  'Standard': layout(['timeline', 'tray', 'inspector', 'pose'], [100, 100, 50, 50]),
  '3D Pose & Geometry Pro': layout(['pose', 'inspector', 'timeline', 'tray'], [70, 30, 100, 100], ['tray']),
  'Masking & VFX Compositor': layout(['inspector', 'timeline', 'tray', 'pose'], [100, 100, 100, 50], ['pose']),
  'Dual Monitor Floating Preview': layout(['timeline', 'inspector', 'pose', 'tray'], [100, 50, 50, 100], [], true),
};
export function validLayout(value: unknown): Layout {
  const fallback = PRESETS['Standard'];
  if (!value || typeof value !== 'object') return structuredClone(fallback);
  const v = value as Layout;
  if (!Array.isArray(v.order) || v.order.length !== 4 || new Set(v.order).size !== 4 || v.order.some(id => !PANEL_IDS.includes(id))) return structuredClone(fallback);
  if (!v.panels || PANEL_IDS.some(id => !v.panels[id] || typeof v.panels[id].collapsed !== 'boolean' || !Number.isFinite(v.panels[id].width) || !Number.isFinite(v.panels[id].height))) return structuredClone(fallback);
  return { order: v.order, floating: v.floating === true, panels: Object.fromEntries(PANEL_IDS.map(id => [id, { collapsed: v.panels[id].collapsed, width: Math.max(25, Math.min(100, v.panels[id].width)), height: Math.max(160, Math.min(1200, v.panels[id].height)) }])) as Layout['panels'] };
}
export function movePanel(order: PanelId[], from: PanelId, to: PanelId) {
  const next = order.filter(id => id !== from); next.splice(order.indexOf(to), 0, from); return next;
}
const LayoutContext = createContext<{ value: Layout; set: (v: Layout) => void }>({ value: PRESETS['Standard'], set: () => {} });
export function WorkspaceLayoutProvider({ children }: { children: ReactNode }) {
  const [value, set] = useState(() => { try { return validLayout(JSON.parse(localStorage.getItem('roop3d-layout-v1') ?? 'null')); } catch { return validLayout(null); } });
  useEffect(() => { try { localStorage.setItem('roop3d-layout-v1', JSON.stringify(value)); } catch { /* In-memory layout still works. */ } }, [value]);
  return <LayoutContext.Provider value={{ value, set }}>{children}</LayoutContext.Provider>;
}
export function LayoutToolbar({ onFloat }: { onFloat: () => void }) {
  const { value, set } = useContext(LayoutContext);
  const [custom, setCustom] = useState<Record<string, Layout>>(() => { try { const saved = JSON.parse(localStorage.getItem('roop3d-custom-layouts-v1') ?? '{}'); return Object.fromEntries(Object.entries(saved).slice(0, 30).map(([name, item]) => [name, validLayout(item)])); } catch { return {}; } });
  const [name, setName] = useState(''), [message, setMessage] = useState(''), [selected, setSelected] = useState('Standard');
  const store = (next: Record<string, Layout>) => { setCustom(next); try { localStorage.setItem('roop3d-custom-layouts-v1', JSON.stringify(next)); setMessage('Layout saved'); } catch { setMessage('Browser storage unavailable; layout saved for this session'); } };
  return <div className="layout-toolbar"><label>Workspace <select aria-label="Workspace layout" value={selected} onChange={e => { const key = e.target.value, next = validLayout(PRESETS[key] ?? custom[key]); set(next); setSelected(key); if (next.floating) onFloat(); }}>{Object.keys(PRESETS).map(key => <option key={key}>{key}</option>)}{Object.keys(custom).map(key => <option key={key}>{key}</option>)}</select></label>
    <input aria-label="Custom layout name" placeholder="Name this layout" maxLength={60} value={name} onChange={e => setName(e.target.value)} />
    <button className="button small" disabled={!name.trim() || !!PRESETS[name.trim()] || Object.keys(custom).length >= 30 && !custom[name.trim()]} onClick={() => { const key = name.trim(); store({ ...custom, [key]: structuredClone(value) }); setSelected(key); setName(''); }}>Save layout</button>
    <button className="button small" disabled={!custom[selected]} onClick={() => { const next = { ...custom }; delete next[selected]; store(next); setSelected('Standard'); set(validLayout(null)); }}>Delete saved</button>
    <button className="button small" onClick={onFloat}>Floating Preview ↗</button><small role="status">{message}</small>
  </div>;
}
export function DockArea({ children }: { children: ReactNode }) { return <div className="dock-area">{children}</div>; }
export function DockPanel({ id, title, children }: { id: PanelId; title: string; children: ReactNode }) {
  const { value, set } = useContext(LayoutContext), state = value.panels[id], index = value.order.indexOf(id);
  const ref = useRef<HTMLElement>(null), resize = useRef<{ x: number; y: number; width: number; height: number; parent: number } | null>(null);
  const patch = (next: Partial<PanelState>) => set({ ...value, panels: { ...value.panels, [id]: { ...state, ...next } } });
  const move = (from: PanelId, to: PanelId) => set({ ...value, order: movePanel(value.order, from, to) });
  return <section className={`dock-panel ${state.collapsed ? 'collapsed' : ''}`} ref={ref} style={{ order: index, flexBasis: `calc(${state.width}% - 8px)` }} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); const from = e.dataTransfer.getData('application/roop3d-panel') as PanelId; if (PANEL_IDS.includes(from)) move(from, id); }}>
    <div className="dock-heading" draggable onDragStart={e => e.dataTransfer.setData('application/roop3d-panel', id)}><strong>{title}</strong><span className="dock-actions"><button className="button small" disabled={index === 0} aria-label={`Move ${title} earlier`} onClick={() => move(id, value.order[index - 1])}>←</button><button className="button small" disabled={index === 3} aria-label={`Move ${title} later`} onClick={() => move(id, value.order[index + 1])}>→</button><button className="button small" aria-label={`${state.collapsed ? 'Expand' : 'Collapse'} ${title}`} aria-expanded={!state.collapsed} onClick={() => patch({ collapsed: !state.collapsed })}>{state.collapsed ? '+' : '−'}</button></span></div>
    <div className="dock-content" hidden={state.collapsed} style={{ height: state.height }}>{children}</div>
    {!state.collapsed && <button className="dock-resize" aria-label={`Resize ${title}`} title="Drag to resize; arrow keys adjust width and height"
      onPointerDown={e => { if (e.button !== 0 || !ref.current) return; e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId); resize.current = { x: e.clientX, y: e.clientY, width: state.width, height: state.height, parent: ref.current.parentElement!.clientWidth }; }}
      onPointerMove={e => { const start = resize.current; if (start) patch({ width: Math.max(25, Math.min(100, start.width + (e.clientX - start.x) / start.parent * 100)), height: Math.max(160, Math.min(1200, start.height + e.clientY - start.y)) }); }} onPointerUp={() => { resize.current = null; }} onPointerCancel={() => { resize.current = null; }} onLostPointerCapture={() => { resize.current = null; }}
      onKeyDown={e => { if (e.key.startsWith('Arrow')) { e.preventDefault(); patch({ width: Math.max(25, Math.min(100, state.width + (e.key === 'ArrowRight' ? 5 : e.key === 'ArrowLeft' ? -5 : 0))), height: Math.max(160, Math.min(1200, state.height + (e.key === 'ArrowDown' ? 20 : e.key === 'ArrowUp' ? -20 : 0))) }); } }}>◢</button>}
  </section>;
}
