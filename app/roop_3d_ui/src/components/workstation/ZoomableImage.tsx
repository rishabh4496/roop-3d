import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type WheelEvent as ReactWheelEvent } from 'react';

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 16;

/** Keep the zoomed image covering the viewport: pan is clamped so no empty margin appears. */
export function clampPan(pan: { x: number; y: number }, zoom: number, width: number, height: number) {
  const limitX = (width * (zoom - 1)) / 2, limitY = (height * (zoom - 1)) / 2;
  return { x: Math.max(-limitX, Math.min(limitX, pan.x)), y: Math.max(-limitY, Math.min(limitY, pan.y)) };
}

type Props = { src: string; alt: string; width?: number; height?: number };

/**
 * Native-pixel inspection of one preview frame: wheel / pinch-free buttons / keyboard zoom around the pointer, drag to pan,
 * double-click toggles fit and 4x. `data-zoom` exposes the current scale for tests and assistive tooling.
 */
export default function ZoomableImage({ src, alt, width = 720, height = 405 }: Props) {
  const [zoom, setZoom] = useState(1), [pan, setPan] = useState({ x: 0, y: 0 });
  const frame = useRef<HTMLDivElement | null>(null), drag = useRef<{ x: number; y: number; origin: { x: number; y: number } } | null>(null);
  useEffect(() => { setZoom(1); setPan({ x: 0, y: 0 }); }, [src]);

  const size = () => { const box = frame.current?.getBoundingClientRect(); return { width: box?.width || width, height: box?.height || height }; };
  const apply = (nextZoom: number, focus?: { x: number; y: number }) => {
    const target = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, nextZoom)), box = size();
    // Zoom about the focus point so the pixel under the pointer stays under the pointer.
    const ratio = target / zoom;
    const anchor = focus ?? { x: 0, y: 0 };
    setZoom(target);
    setPan(clampPan({ x: anchor.x - (anchor.x - pan.x) * ratio, y: anchor.y - (anchor.y - pan.y) * ratio }, target, box.width, box.height));
  };
  const focusOf = (clientX: number, clientY: number) => {
    const box = frame.current?.getBoundingClientRect();
    return box ? { x: clientX - box.left - box.width / 2, y: clientY - box.top - box.height / 2 } : undefined;
  };
  const onWheel = (event: ReactWheelEvent) => { event.preventDefault(); apply(zoom * (event.deltaY < 0 ? 1.25 : 0.8), focusOf(event.clientX, event.clientY)); };
  const onPointerDown = (event: ReactPointerEvent) => {
    if (zoom <= 1) return;
    drag.current = { x: event.clientX, y: event.clientY, origin: pan };
    (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
  };
  const onPointerMove = (event: ReactPointerEvent) => {
    if (!drag.current) return;
    const box = size();
    setPan(clampPan({ x: drag.current.origin.x + event.clientX - drag.current.x, y: drag.current.origin.y + event.clientY - drag.current.y }, zoom, box.width, box.height));
  };
  const end = () => { drag.current = null; };
  const reset = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  return (
    <div className="zoomable">
      <div ref={frame} className="zoomable-frame" data-testid="zoom-frame" data-zoom={zoom.toFixed(2)} tabIndex={0} role="group" aria-label={`${alt} (zoomable)`}
        style={{ cursor: zoom > 1 ? 'grab' : 'zoom-in', touchAction: 'none' }}
        onWheel={onWheel} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={end} onPointerCancel={end}
        onDoubleClick={event => (zoom > 1 ? reset() : apply(4, focusOf(event.clientX, event.clientY)))}
        onKeyDown={event => {
          if (event.key === '+' || event.key === '=') apply(zoom * 1.25);
          else if (event.key === '-') apply(zoom * 0.8);
          else if (event.key === '0') reset();
          else if (zoom > 1 && event.key.startsWith('Arrow')) {
            event.preventDefault();
            const box = size(), step = 40;
            setPan(clampPan({ x: pan.x + (event.key === 'ArrowLeft' ? step : event.key === 'ArrowRight' ? -step : 0), y: pan.y + (event.key === 'ArrowUp' ? step : event.key === 'ArrowDown' ? -step : 0) }, zoom, box.width, box.height));
          }
        }}>
        <img src={src} alt={alt} draggable={false} style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: 'center center' }} />
      </div>
      <div className="zoomable-bar">
        <button type="button" className="button small" aria-label="Zoom out" onClick={() => apply(zoom * 0.8)} disabled={zoom <= MIN_ZOOM}>−</button>
        <output aria-live="polite">{Math.round(zoom * 100)}%</output>
        <button type="button" className="button small" aria-label="Zoom in" onClick={() => apply(zoom * 1.25)} disabled={zoom >= MAX_ZOOM}>+</button>
        <button type="button" className="button small" onClick={reset} disabled={zoom === 1}>Fit</button>
        <button type="button" className="button small" onClick={() => apply(MAX_ZOOM / 4)}>400%</button>
        <small className="muted">Scroll to zoom · drag to pan · double-click toggles 400%</small>
      </div>
    </div>
  );
}
