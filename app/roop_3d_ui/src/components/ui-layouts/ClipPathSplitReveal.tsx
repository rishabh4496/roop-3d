import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react';
import { clamp, cn } from '../../lib/utils';

/**
 * Draggable before/after comparator. The "before" layer (original target frame) is clipped with `clip-path: polygon(...)`
 * over the "after" layer (3D photoreal swap); the divider is a real `role="slider"`, so it is keyboard and screen-reader operable.
 *
 *  - Drag anywhere on the frame, or use Left/Right (1%), Shift+Left/Right (10%), Home/End.
 *  - Hold Shift while hovering for a loupe (default 4x) that magnifies whichever side is under the cursor, at pixel level.
 *    The loupe is a scaled copy of the same layer (same `object-fit`), so it always matches what is on screen.
 * Both layers can be image URLs or arbitrary nodes (e.g. a canvas) via `before` / `after`.
 */
export interface ClipPathSplitRevealProps {
  beforeSrc?: string;
  afterSrc?: string;
  before?: ReactNode;
  after?: ReactNode;
  beforeLabel?: string;
  afterLabel?: string;
  beforeAlt?: string;
  afterAlt?: string;
  /** Initial divider position, 0..100 (percent from the left). */
  defaultPosition?: number;
  position?: number;
  onPositionChange?: (position: number) => void;
  /** Loupe magnification. 0 disables the loupe. */
  loupeZoom?: number;
  /** Loupe diameter in px. */
  loupeSize?: number;
  /** width / height of the frame. */
  aspectRatio?: number;
  className?: string;
  'aria-label'?: string;
}

/** `clip-path` that keeps only the part left of `position` percent. */
export const clipLeftOf = (position: number) => {
  const p = clamp(position, 0, 100);
  return `polygon(0% 0%, ${p}% 0%, ${p}% 100%, 0% 100%)`;
};

/** Where a magnified copy of a `size`-px frame must sit so that point (x, y) lands in the centre of a `loupe`-px lens. */
export function loupeOffset(x: number, y: number, zoom: number, loupe: number) {
  return { left: loupe / 2 - x * zoom, top: loupe / 2 - y * zoom };
}

interface LoupeState { x: number; y: number; w: number; h: number; side: 'before' | 'after' }

export function ClipPathSplitReveal({
  beforeSrc, afterSrc, before, after, beforeLabel = 'Original', afterLabel = '3D swap', beforeAlt = 'Original target frame', afterAlt = '3D photoreal swap',
  defaultPosition = 50, position, onPositionChange, loupeZoom = 4, loupeSize = 160, aspectRatio = 16 / 9, className, 'aria-label': ariaLabel = 'Before and after comparison',
}: ClipPathSplitRevealProps) {
  const frame = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const [inner, setInner] = useState(clamp(defaultPosition, 0, 100));
  const [loupe, setLoupe] = useState<LoupeState | null>(null);
  const pos = clamp(position ?? inner, 0, 100);

  const setPos = useCallback((next: number) => {
    const value = Math.round(clamp(next, 0, 100) * 10) / 10;
    if (position === undefined) setInner(value);
    onPositionChange?.(value);
  }, [position, onPositionChange]);

  const fromClientX = (clientX: number) => {
    const box = frame.current?.getBoundingClientRect();
    if (!box || box.width <= 0) return;
    setPos(((clientX - box.left) / box.width) * 100);
  };

  const updateLoupe = (event: PointerEvent<HTMLDivElement>) => {
    const box = frame.current?.getBoundingClientRect();
    if (!box || !loupeZoom || !event.shiftKey || dragging.current) { setLoupe(null); return; }
    const x = clamp(event.clientX - box.left, 0, box.width), y = clamp(event.clientY - box.top, 0, box.height);
    setLoupe({ x, y, w: box.width, h: box.height, side: (x / box.width) * 100 < pos ? 'before' : 'after' });
  };

  // Releasing Shift while the pointer rests still must close the loupe.
  useEffect(() => {
    if (!loupe) return;
    const onKeyUp = (event: globalThis.KeyboardEvent) => { if (event.key === 'Shift') setLoupe(null); };
    window.addEventListener('keyup', onKeyUp);
    return () => window.removeEventListener('keyup', onKeyUp);
  }, [loupe]);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 10 : 1;
    let next: number | undefined;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') next = pos - step;
    else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') next = pos + step;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = 100;
    if (next === undefined) return;
    event.preventDefault();
    setPos(next);
  };

  const layer = (src: string | undefined, node: ReactNode, alt: string) =>
    node ?? (src ? <img className="ul-reveal__img" src={src} alt={alt} draggable={false} /> : null);
  const beforeNode = layer(beforeSrc, before, beforeAlt), afterNode = layer(afterSrc, after, afterAlt);

  const lens = loupe && (() => {
    const offset = loupeOffset(loupe.x, loupe.y, loupeZoom, loupeSize);
    const src = loupe.side === 'before' ? beforeSrc : afterSrc;
    return (
      <div className="ul-reveal__loupe" aria-hidden="true" data-side={loupe.side} data-testid="reveal-loupe"
        style={{ width: loupeSize, height: loupeSize, left: loupe.x - loupeSize / 2, top: loupe.y - loupeSize / 2 }}>
        {src && <img src={src} alt="" draggable={false} style={{ position: 'absolute', width: loupe.w * loupeZoom, height: loupe.h * loupeZoom, left: offset.left, top: offset.top, maxWidth: 'none', objectFit: 'cover' }} />}
        <span className="ul-reveal__zoom ul-number">{loupeZoom}×</span>
      </div>
    );
  })();

  return (
    <div
      ref={frame} className={cn('ul-reveal', className)} style={{ aspectRatio }}
      onPointerDown={event => {
        if (event.button !== 0) return;
        dragging.current = true; setLoupe(null);
        event.currentTarget.setPointerCapture?.(event.pointerId);
        fromClientX(event.clientX);
      }}
      onPointerMove={event => { if (dragging.current) fromClientX(event.clientX); else updateLoupe(event); }}
      onPointerUp={event => { dragging.current = false; event.currentTarget.releasePointerCapture?.(event.pointerId); }}
      onPointerCancel={() => { dragging.current = false; }}
      onPointerLeave={() => { if (!dragging.current) setLoupe(null); }}
    >
      <div className="ul-reveal__layer ul-reveal__after">{afterNode}</div>
      <div className="ul-reveal__layer ul-reveal__before" style={{ clipPath: clipLeftOf(pos) }} data-testid="reveal-before">{beforeNode}</div>
      <span className="ul-reveal__tag ul-reveal__tag--before">{beforeLabel}</span>
      <span className="ul-reveal__tag ul-reveal__tag--after">{afterLabel}</span>
      <div
        className="ul-reveal__divider" style={{ left: `${pos}%` }}
        role="slider" tabIndex={0} aria-label={ariaLabel} aria-orientation="horizontal"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pos)} aria-valuetext={`${Math.round(pos)}% ${beforeLabel}, ${Math.round(100 - pos)}% ${afterLabel}`}
        onKeyDown={onKeyDown}
      >
        <span className="ul-reveal__grip" aria-hidden="true"><svg viewBox="0 0 24 24" width="16" height="16"><path d="M9 6l-6 6 6 6M15 6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg></span>
      </div>
      {lens}
      {loupeZoom > 0 && <span className="ul-reveal__hint">Hold Shift to magnify</span>}
    </div>
  );
}

export default ClipPathSplitReveal;
