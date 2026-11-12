import { useRef, useState, type ComponentPropsWithoutRef, type KeyboardEvent, type MouseEvent, type ReactNode } from 'react';
import { motion, useDragControls, useReducedMotion } from 'framer-motion';
import { cn } from '../../lib/utils';

/**
 * Liquid glass card (ui-layouts "Liquid Glass", re-implemented on this app's CSS tokens: there is no Tailwind here).
 *
 *  - `blurIntensity`   backdrop blur: sm 8px · md 16px · lg 24px · xl 40px
 *  - `glowIntensity`   outer glow radius, none … 2xl
 *  - `shadowIntensity` inner shadow depth, none … 2xl
 *  - `draggable`       free drag with an elastic spring back to where it sat
 *  - `expandable`      click (or the toggle button, for keyboard and screen readers) to animate between the collapsed and expanded size
 *
 * Extension over the upstream API: `dragHandle` limits dragging to a grip, so a card full of inputs (the brush toolbar) stays usable.
 */
export type BlurIntensity = 'sm' | 'md' | 'lg' | 'xl';
export type EffectIntensity = 'none' | 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl';

export const BLUR_PX: Record<BlurIntensity, number> = { sm: 8, md: 16, lg: 24, xl: 40 };
export const GLOW_PX: Record<EffectIntensity, number> = { none: 0, xs: 6, sm: 12, md: 20, lg: 32, xl: 48, '2xl': 72 };
export const SHADOW_PX: Record<EffectIntensity, number> = { none: 0, xs: 3, sm: 6, md: 12, lg: 20, xl: 30, '2xl': 44 };

/** Unknown values fall back to the default instead of producing `NaNpx`. */
export function glassVars(blur: string, glow: string, shadow: string, radius: string): Record<string, string> {
  const b = BLUR_PX[blur as BlurIntensity] ?? BLUR_PX.sm;
  const g = GLOW_PX[glow as EffectIntensity] ?? GLOW_PX.sm;
  const s = SHADOW_PX[shadow as EffectIntensity] ?? SHADOW_PX.md;
  return { '--ul-glass-blur': `${b}px`, '--ul-glass-glow': `${g}px`, '--ul-glass-shadow': `${s}px`, '--ul-glass-radius': radius };
}

/** Elastic return: the card is released anywhere and springs back to its home position. */
export const ELASTIC = { dragElastic: 0.45, dragTransition: { bounceStiffness: 420, bounceDamping: 20 } } as const;

export interface LiquidGlassCardProps extends Omit<ComponentPropsWithoutRef<typeof motion.div>, 'children' | 'drag'> {
  children?: ReactNode;
  className?: string;
  draggable?: boolean;
  /** Only the grip starts a drag (extension). Implies `draggable`. */
  dragHandle?: boolean;
  expandable?: boolean;
  defaultExpanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
  width?: string;
  height?: string;
  expandedWidth?: string;
  expandedHeight?: string;
  blurIntensity?: BlurIntensity;
  borderRadius?: string;
  glowIntensity?: EffectIntensity;
  shadowIntensity?: EffectIntensity;
}

const INTERACTIVE = 'button, a, input, select, textarea, label, summary, [role="slider"], [data-no-expand]';

export function LiquidGlassCard({
  children, className, draggable = false, dragHandle = false, expandable = false, defaultExpanded = false, onExpandedChange,
  width, height, expandedWidth, expandedHeight, blurIntensity = 'sm', borderRadius = '32px', glowIntensity = 'sm', shadowIntensity = 'md',
  style, onClick, ...rest
}: LiquidGlassCardProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const controls = useDragControls();
  const reduce = useReducedMotion();
  const bodyId = useRef(`ul-glass-${Math.random().toString(36).slice(2, 9)}`).current;
  const canDrag = draggable || dragHandle;

  const toggle = (next = !expanded) => { setExpanded(next); onExpandedChange?.(next); };
  const size = expanded
    ? { width: expandedWidth ?? width ?? 'auto', height: expandedHeight ?? height ?? 'auto' }
    : { width: width ?? 'auto', height: height ?? 'auto' };

  const handleClick = (event: MouseEvent<HTMLDivElement>) => {
    onClick?.(event);
    if (!expandable || event.defaultPrevented) return;
    if ((event.target as HTMLElement).closest(INTERACTIVE)) return;   // a click on a control inside the card is that control's, not an expand
    toggle();
  };
  const handleKey = (event: KeyboardEvent<HTMLButtonElement>) => { if (event.key === 'Escape' && expanded) { event.stopPropagation(); toggle(false); } };

  return (
    <motion.div
      {...rest}
      className={cn('ul-glass', canDrag && 'ul-glass--draggable', expandable && 'ul-glass--expandable', expanded && 'is-expanded', className)}
      style={{ ...glassVars(blurIntensity, glowIntensity, shadowIntensity, borderRadius), ...style }}
      data-expanded={expandable ? expanded : undefined}
      animate={expandable ? size : undefined}
      initial={false}
      transition={reduce ? { duration: 0 } : { type: 'spring', stiffness: 260, damping: 28 }}
      {...(canDrag ? {
        drag: true as const,
        dragListener: !dragHandle,
        dragControls: controls,
        dragConstraints: { left: 0, right: 0, top: 0, bottom: 0 },
        dragElastic: reduce ? 0 : ELASTIC.dragElastic,
        dragTransition: ELASTIC.dragTransition,
        dragMomentum: false,
        whileDrag: reduce ? undefined : { scale: 1.02 },
      } : {})}
      onClick={handleClick}
    >
      <span className="ul-glass__sheen" aria-hidden />
      {dragHandle && <button type="button" className="ul-glass__grip" aria-label="Drag to move" title="Drag to move; it springs back when released"
        onPointerDown={event => controls.start(event)}><span aria-hidden>⠿</span></button>}
      {expandable && <button type="button" className="ul-glass__toggle" aria-expanded={expanded} aria-controls={bodyId} aria-label={expanded ? 'Collapse' : 'Expand'}
        onClick={event => { event.stopPropagation(); toggle(); }} onKeyDown={handleKey}><span aria-hidden>{expanded ? '–' : '+'}</span></button>}
      <div id={bodyId} className="ul-glass__body">{children}</div>
    </motion.div>
  );
}
