import { createPortal } from 'react-dom';
import { useCallback, useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react';
import Modal from '../controls/Modal';
import { mediaURL } from '../../api';
import { createAtomicStore, shallowEqual, useStoreSelector } from '../../stores/atomicStore';
import { LiveTime } from './LiveTime';

export function FloatingPreviewMedia({ image, path, time, playing, label }: { image?: string; path: string; time: number; playing: boolean; label: string }) {
  const video = useRef<HTMLVideoElement>(null), [error, setError] = useState('');
  useEffect(() => {
    const element = video.current;
    if (!element) return;
    if (Number.isFinite(time) && Math.abs(element.currentTime - time) > .2) element.currentTime = time;
    if (playing) void element.play().catch(() => setError('Playback unavailable for this codec. Pause to inspect decoded frame previews.'));
    else element.pause();
  }, [path, time, playing, image]);
  useEffect(() => { setError(''); }, [path]);
  return <><div className="floating-caption">{label} · <LiveTime fallback={time} live={playing}>{live => live.toFixed(3)}</LiveTime> s</div>{image ? <img src={image} alt="Live workstation preview"/> : path ? <video ref={video} src={mediaURL(path)} muted playsInline preload="metadata" onLoadedMetadata={event => { event.currentTarget.currentTime = time; }}/>
    : <p className="muted">Capture a frame or solo a pipeline buffer to preview it here.</p>}{error && <p role="status" className="muted">{error}</p>}</>;
}

/* ───────────────────────── Stage 26: floating Picture-in-Picture preview ───────────────────────── */

export type PreviewMode = 'closed' | 'window' | 'pip' | 'popup';
export type Geometry = { x: number; y: number; width: number; height: number };
export const MIN_WINDOW = { width: 220, height: 150 };
export const DEFAULT_SIZE = { width: 380, height: 260 };
const STORAGE = 'roop3d-pip-v1';

/** Keep the window fully on screen and above its minimum size, whatever the viewport did since it was saved. */
export function clampGeometry(g: Geometry, viewportWidth: number, viewportHeight: number): Geometry {
  const width = Math.max(MIN_WINDOW.width, Math.min(g.width, viewportWidth)), height = Math.max(MIN_WINDOW.height, Math.min(g.height, viewportHeight));
  return { width, height, x: Math.max(0, Math.min(g.x, viewportWidth - width)), y: Math.max(0, Math.min(g.y, viewportHeight - height)) };
}
export const defaultGeometry = (viewportWidth: number, viewportHeight: number): Geometry => clampGeometry({ ...DEFAULT_SIZE, x: viewportWidth - DEFAULT_SIZE.width - 24, y: viewportHeight - DEFAULT_SIZE.height - 24 }, viewportWidth, viewportHeight);

type Stored = { mode: PreviewMode; geometry: Geometry | null };
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
export function validStored(value: unknown): Stored {
  const v = (value && typeof value === 'object' ? value : {}) as Record<string, any>, g = v.geometry;
  const geometry = g && [g.x, g.y, g.width, g.height].every(finite) ? { x: g.x, y: g.y, width: g.width, height: g.height } as Geometry : null;
  // Only the in-app window can be restored on load: a browser PiP / popup window needs a fresh user gesture.
  return { mode: v.mode === 'window' ? 'window' : 'closed', geometry };
}
const read = (): Stored => { try { return validStored(JSON.parse(localStorage.getItem(STORAGE) ?? 'null')); } catch { return validStored(null); } };
/** Shared across routes: the preview window belongs to the workstation shell, not to any one screen. */
export const pipStore = createAtomicStore<Stored>(read());
pipStore.subscribe(() => { try { const { mode, geometry } = pipStore.get(); localStorage.setItem(STORAGE, JSON.stringify({ mode: mode === 'window' ? 'window' : 'closed', geometry })); } catch { /* In-memory state still works. */ } });

type DocumentPip = { requestWindow: (options?: { width?: number; height?: number }) => Promise<Window>; window?: Window | null };
export const documentPip = (): DocumentPip | null => { const api = (window as unknown as { documentPictureInPicture?: DocumentPip }).documentPictureInPicture; return api && typeof api.requestWindow === 'function' ? api : null; };

/** Mirror keyboard events from a detached window to the workstation, so frame-step, W/D, 1–6 … work while it has focus. */
function forwardKeys(child: Window) {
  child.addEventListener('keydown', event => {
    if ((event.target as HTMLElement)?.closest?.('input,textarea,select,[contenteditable=true]')) return;
    const { key, code, ctrlKey, altKey, shiftKey, metaKey, repeat } = event;
    if (window.dispatchEvent(new KeyboardEvent('keydown', { key, code, ctrlKey, altKey, shiftKey, metaKey, repeat, bubbles: true, cancelable: true }))) return;
    event.preventDefault();
  });
}

function FloatingWindow({ children, onClose, onPopOut, onDetach }: { children: ReactNode; onClose: () => void; onPopOut?: () => void; onDetach?: () => void }) {
  const stored = useStoreSelector(pipStore, state => state.geometry, shallowEqual);
  const [live, setLive] = useState<Geometry | null>(null);
  const viewport = () => ({ width: window.innerWidth || 1280, height: window.innerHeight || 800 });
  const resolve = (): Geometry => { const { width, height } = viewport(); return clampGeometry(live ?? stored ?? defaultGeometry(width, height), width, height); };
  const geometry = resolve();
  const drag = useRef<{ kind: 'move' | 'resize'; x: number; y: number; origin: Geometry } | null>(null);
  const commit = useCallback((next: Geometry) => { const { width, height } = viewport(); pipStore.set({ geometry: clampGeometry(next, width, height) }); setLive(null); }, []);
  const begin = (kind: 'move' | 'resize') => (event: ReactPointerEvent) => {
    if (event.button !== 0 || (kind === 'move' && (event.target as HTMLElement).closest('button'))) return;
    event.preventDefault(); event.currentTarget.setPointerCapture?.(event.pointerId);
    drag.current = { kind, x: event.clientX, y: event.clientY, origin: geometry };
  };
  const move = (event: ReactPointerEvent) => {
    const start = drag.current; if (!start) return;
    const dx = event.clientX - start.x, dy = event.clientY - start.y, { width, height } = viewport();
    setLive(clampGeometry(start.kind === 'move' ? { ...start.origin, x: start.origin.x + dx, y: start.origin.y + dy } : { ...start.origin, width: start.origin.width + dx, height: start.origin.height + dy }, width, height));
  };
  const end = (event: ReactPointerEvent) => { if (drag.current && event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); if (drag.current) commit(resolve()); drag.current = null; };
  const keys = (event: ReactKeyboardEvent) => {
    const delta = ({ ArrowLeft: [-16, 0], ArrowRight: [16, 0], ArrowUp: [0, -16], ArrowDown: [0, 16] } as Record<string, [number, number]>)[event.key];
    if (event.key === 'Escape') { onClose(); return; }
    if (!delta) return; event.preventDefault(); event.stopPropagation();
    commit(event.shiftKey ? { ...geometry, width: geometry.width + delta[0], height: geometry.height + delta[1] } : { ...geometry, x: geometry.x + delta[0], y: geometry.y + delta[1] });
  };
  return createPortal(
    <aside className="floating-window" data-testid="floating-window" role="region" aria-label="Floating preview" style={{ left: geometry.x, top: geometry.y, width: geometry.width, height: geometry.height }}>
      <div className="floating-window-bar" tabIndex={0} aria-label="Floating preview title bar: drag to move, arrow keys move, Shift plus arrows resize" onPointerDown={begin('move')} onPointerMove={move} onPointerUp={end} onPointerCancel={end} onKeyDown={keys}>
        <strong>Live preview</strong>
        <span className="floating-window-actions">
          {onDetach && <button type="button" className="button small" onClick={onDetach} title="Detach into a separate browser window (for a second monitor)" aria-label="Detach to browser window">↗</button>}
          {onPopOut && <button type="button" className="button small" onClick={onPopOut} title="Pop out as an always-on-top browser Picture-in-Picture window" aria-label="Pop out to Picture-in-Picture">⇱</button>}
          <button type="button" className="button small" onClick={onClose} aria-label="Close floating preview">✕</button>
        </span>
      </div>
      <div className="floating-window-body floating-preview">{children}</div>
      <div className="floating-window-resize" role="presentation" onPointerDown={begin('resize')} onPointerMove={move} onPointerUp={end} onPointerCancel={end} />
    </aside>, document.body);
}

/**
 * Detachable live preview. Three presentations of the same React content, all fed by the same live state:
 *  - `window`: a draggable, resizable in-app window portalled to <body> (visible on every route, position persisted);
 *  - `pip`: the browser's Document Picture-in-Picture window (always on top, Chromium 116+), falling back to `window`;
 *  - `popup`: a classic secondary browser window (`open()`), falling back to a modal when popups are blocked.
 */
export function useFloatingPreview(content: ReactNode) {
  const popup = useRef<Window | null>(null), [host, setHost] = useState<HTMLElement | null>(null), [fallback, setFallback] = useState(false);
  const mode = useStoreSelector(pipStore, state => state.mode);
  const [windowName] = useState(() => `roop3d-preview-${typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : Math.random().toString(36).slice(2)}`);
  const closeExternal = () => { const child = popup.current; popup.current = null; setHost(null); try { child?.close(); } catch { /* already gone */ } };
  const close = () => { closeExternal(); setFallback(false); pipStore.set({ mode: 'closed' }); };
  useEffect(() => () => { const child = popup.current; popup.current = null; try { child?.close(); } catch { /* already gone */ } if (pipStore.get().mode === 'popup' || pipStore.get().mode === 'pip') pipStore.set({ mode: 'closed' }); }, []);

  const attach = (child: Window, title: string, next: PreviewMode) => {
    popup.current = child;
    child.document.title = title;
    child.document.body.replaceChildren();
    for (const node of document.querySelectorAll('link[rel=stylesheet],style')) child.document.head.append(node.cloneNode(true));
    child.document.documentElement.dataset.theme = document.documentElement.dataset.theme;
    const root = child.document.createElement('main'); root.className = 'floating-preview'; child.document.body.append(root); setHost(root);
    forwardKeys(child);
    child.addEventListener('pagehide', () => { if (popup.current === child) { popup.current = null; setHost(null); if (pipStore.get().mode === next) pipStore.set({ mode: 'closed' }); } }, { once: true });
    pipStore.set({ mode: next });
  };
  /** Legacy secondary window; a blocked popup degrades to a keyboard-dismissable modal. */
  const open = () => {
    if (popup.current && !popup.current.closed) { popup.current.focus(); return; }
    const child = window.open('', windowName, 'popup,width=1200,height=760');
    if (!child) { setFallback(true); return; }
    attach(child, 'Roop 3D — Floating Preview', 'popup');
  };
  const openWindow = () => { closeExternal(); setFallback(false); pipStore.set({ mode: 'window' }); };
  const openPip = async () => {
    const api = documentPip();
    if (!api) { openWindow(); return; }
    try {
      const geometry = pipStore.get().geometry;
      closeExternal();
      const child = await api.requestWindow({ width: Math.round(geometry?.width ?? DEFAULT_SIZE.width), height: Math.round(geometry?.height ?? DEFAULT_SIZE.height) });
      attach(child, 'Roop 3D — Live Preview', 'pip');
    } catch { openWindow(); }
  };
  const toggle = () => { if (pipStore.get().mode === 'closed') openWindow(); else close(); };

  const portal = host ? createPortal(content, host)
    : mode === 'window' ? <FloatingWindow onClose={close} onPopOut={documentPip() ? () => void openPip() : undefined} onDetach={open}>{content}</FloatingWindow>
    : fallback ? <Modal title="Floating preview" close={close}><div className="floating-preview-fallback"><p className="muted">Popup blocked. Allow popups for a secondary monitor window.</p>{content}</div></Modal> : null;
  return { open, openWindow, openPip, close, toggle, portal, mode, isOpen: mode !== 'closed' || fallback, supportsPip: documentPip() !== null };
}
