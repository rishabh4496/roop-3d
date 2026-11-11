import { useEffect, useRef, type HTMLAttributes, type ReactNode } from 'react';
import { cn } from '../../lib/utils';

/**
 * Cursor-following radial spotlight. The glow lives in CSS custom properties written straight onto the element
 * (`--mouse-x`, `--mouse-y`, `--spot-opacity`), so tracking the pointer never re-renders React.
 *
 *  - Inside the card: `onMouseMove` tracks the cursor exactly.
 *  - Near the card: a passive window listener fades the glow in as the pointer approaches (within `proximity` px).
 *  - The 1.5px glowing border is the wrapper's padding, painted by the same gradient (`.ul-spotlight { padding: 1.5px }`).
 */
export interface SpotlightCardProps extends HTMLAttributes<HTMLDivElement> {
  children?: ReactNode;
  /** Distance in px from the card edge at which the glow starts to fade in. 0 disables proximity tracking. */
  proximity?: number;
  /** Class for the inner surface (the wrapper's `className` styles the glowing border shell). */
  bodyClassName?: string;
}

/** Opacity for a pointer at (x, y) relative to a rect: 1 inside, linear falloff to 0 at `proximity` px outside. */
export function proximityOpacity(x: number, y: number, rect: { left: number; top: number; right: number; bottom: number }, proximity: number): number {
  const dx = Math.max(rect.left - x, 0, x - rect.right);
  const dy = Math.max(rect.top - y, 0, y - rect.bottom);
  const distance = Math.hypot(dx, dy);
  if (distance === 0) return 1;
  if (proximity <= 0 || distance >= proximity) return 0;
  return 1 - distance / proximity;
}

export function SpotlightCard({ children, className, bodyClassName, proximity = 140, onMouseMove, onMouseLeave, ...rest }: SpotlightCardProps) {
  const ref = useRef<HTMLDivElement>(null);

  const place = (x: number, y: number, opacity: number) => {
    const node = ref.current;
    if (!node) return;
    const box = node.getBoundingClientRect();
    node.style.setProperty('--mouse-x', `${x - box.left}px`);
    node.style.setProperty('--mouse-y', `${y - box.top}px`);
    node.style.setProperty('--spot-opacity', opacity.toFixed(3));
  };

  useEffect(() => {
    if (proximity <= 0) return;
    let frame = 0;
    const onMove = (event: PointerEvent) => {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        const node = ref.current;
        if (!node) return;
        const opacity = proximityOpacity(event.clientX, event.clientY, node.getBoundingClientRect(), proximity);
        if (opacity > 0) place(event.clientX, event.clientY, opacity);
        else node.style.setProperty('--spot-opacity', '0');
      });
    };
    window.addEventListener('pointermove', onMove, { passive: true });
    return () => { window.removeEventListener('pointermove', onMove); if (frame) cancelAnimationFrame(frame); };
  }, [proximity]);

  return (
    <div
      {...rest}
      ref={ref}
      className={cn('ul-spotlight', className)}
      onMouseMove={event => { place(event.clientX, event.clientY, 1); onMouseMove?.(event); }}
      onMouseLeave={event => { ref.current?.style.setProperty('--spot-opacity', '0'); onMouseLeave?.(event); }}
    >
      <div className={cn('ul-spotlight__body', bodyClassName)}>
        <CursorFlowGradient />
        <div className="ul-spotlight__content">{children}</div>
      </div>
    </div>
  );
}

/** Soft hover-focus gradient that flows under the cursor inside a SpotlightCard (reads the card's --mouse-x/--mouse-y). */
export function CursorFlowGradient({ className }: { className?: string }) {
  return <span className={cn('ul-cursor-flow', className)} aria-hidden="true" />;
}

export default SpotlightCard;
