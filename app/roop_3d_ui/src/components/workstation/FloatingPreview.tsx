import { createPortal } from 'react-dom';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import Modal from '../controls/Modal';
import { mediaURL } from '../../api';

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
  return <><div className="floating-caption">{label} · {time.toFixed(3)} s</div>{image ? <img src={image} alt="Live workstation preview"/> : path ? <video ref={video} src={mediaURL(path)} muted playsInline preload="metadata" onLoadedMetadata={event => { event.currentTarget.currentTime = time; }}/>
    : <p className="muted">Capture a frame or solo a pipeline buffer to preview it here.</p>}{error && <p role="status" className="muted">{error}</p>}</>;
}

// A same-origin React portal shares live state without transferring media to another service.
export function useFloatingPreview(content: ReactNode) {
  const popup = useRef<Window | null>(null), [host, setHost] = useState<HTMLElement | null>(null), [fallback, setFallback] = useState(false);
  const [windowName] = useState(() => `roop3d-preview-${typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : Math.random().toString(36).slice(2)}`);
  const close = () => { popup.current?.close(); popup.current = null; setHost(null); setFallback(false); };
  useEffect(() => () => { popup.current?.close(); }, []);
  const open = () => {
    if (popup.current && !popup.current.closed) { popup.current.focus(); return; }
    const child = window.open('', windowName, 'popup,width=1200,height=760');
    if (!child) { setFallback(true); return; }
    popup.current = child;
    child.document.title = 'Roop 3D — Floating Preview';
    child.document.body.replaceChildren();
    for (const node of document.querySelectorAll('link[rel=stylesheet],style')) child.document.head.append(node.cloneNode(true));
    child.document.documentElement.dataset.theme = document.documentElement.dataset.theme;
    const root = child.document.createElement('main'); root.className = 'floating-preview'; child.document.body.append(root); setHost(root);
    child.addEventListener('pagehide', () => { popup.current = null; setHost(null); }, { once: true });
  };
  const portal = host ? createPortal(content, host) : fallback ? <Modal title="Floating preview" close={close}><div className="floating-preview-fallback"><p className="muted">Popup blocked. Allow popups for a secondary monitor window.</p>{content}</div></Modal> : null;
  return { open, close, portal };
}
