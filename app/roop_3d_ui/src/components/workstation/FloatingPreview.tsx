import { createPortal } from 'react-dom';
import { useEffect, useRef, useState, type ReactNode } from 'react';

// A same-origin React portal shares live state without transferring media to another service.
export function useFloatingPreview(content: ReactNode) {
  const popup = useRef<Window | null>(null), [host, setHost] = useState<HTMLElement | null>(null), [fallback, setFallback] = useState(false);
  const close = () => { popup.current?.close(); popup.current = null; setHost(null); setFallback(false); };
  useEffect(() => () => { popup.current?.close(); }, []);
  const open = () => {
    if (popup.current && !popup.current.closed) { popup.current.focus(); return; }
    const child = window.open('', 'roop3d-local-preview', 'popup,width=1200,height=760');
    if (!child) { setFallback(true); return; }
    popup.current = child;
    child.document.title = 'Roop 3D — Floating Preview';
    child.document.body.replaceChildren();
    for (const node of document.querySelectorAll('link[rel=stylesheet],style')) child.document.head.append(node.cloneNode(true));
    child.document.documentElement.dataset.theme = document.documentElement.dataset.theme;
    const root = child.document.createElement('main'); root.className = 'floating-preview'; child.document.body.append(root); setHost(root);
    child.addEventListener('pagehide', () => { popup.current = null; setHost(null); }, { once: true });
  };
  const portal = host ? createPortal(content, host) : fallback ? <div className="floating-fallback" role="dialog" aria-modal="true" aria-label="Floating preview"><button className="button" onClick={close}>Close floating preview</button><p className="muted">Popup blocked. Allow popups for a secondary monitor window.</p>{content}</div> : null;
  return { open, close, portal };
}
