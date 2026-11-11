import { useEffect, useId, useState, type RefObject } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { cn } from '../../lib/utils';

/**
 * SVG beams that travel between DOM nodes, used to draw the live 3D swap execution graph (detect → track → mesh → relight → composite).
 * Geometry is measured from the nodes' bounding boxes relative to `containerRef`, and re-measured when the container or either node resizes.
 *
 *  - `direction="forward"`  : light travels from → to
 *  - `direction="reverse"`  : light travels to → from
 *  - `direction="both"`     : two out-of-phase pulses, for a bidirectional link (e.g. tracker ⇄ mesh feedback)
 *  - `ConvergingBeams`      : many sources into one target (multi-input convergence), fanned out so the curves don't overlap
 *
 * `active={false}` keeps the resting path but stops the pulse (idle stage). Reduced motion also freezes the pulse.
 */
export type BeamDirection = 'forward' | 'reverse' | 'both';

export interface AnimatedBeamProps {
  containerRef: RefObject<HTMLElement | null>;
  fromRef: RefObject<HTMLElement | null>;
  toRef: RefObject<HTMLElement | null>;
  direction?: BeamDirection;
  /** Vertical bow of the curve in px; negative bows the other way. 0 is a straight line. */
  curvature?: number;
  /** Seconds for one pulse to cross the path. */
  duration?: number;
  /** Seconds before the first pulse. */
  delay?: number;
  pathWidth?: number;
  pathOpacity?: number;
  /** Resting path colour. Defaults to the hairline token. */
  pathColor?: string;
  gradientStartColor?: string;
  gradientStopColor?: string;
  /** Pixel nudges for where the beam attaches to each node (from its centre). */
  /** Which edge of each node the beam attaches to: its centre (default), or its top/bottom edge inset by 2px (a lit rail along a row of cards). */
  anchor?: 'center' | 'top' | 'bottom';
  startXOffset?: number;
  startYOffset?: number;
  endXOffset?: number;
  endYOffset?: number;
  active?: boolean;
  className?: string;
}

interface Box { left: number; top: number; width: number; height: number }

export interface BeamGeometry { path: string; width: number; height: number; startX: number; endX: number }

/** Pure geometry: a quadratic curve between the two node centres (plus offsets), in container-local coordinates. */
export function computeBeamPath(container: Box, from: Box, to: Box, { curvature = 0, startXOffset = 0, startYOffset = 0, endXOffset = 0, endYOffset = 0, anchor = 'center' as 'center' | 'top' | 'bottom' } = {}): BeamGeometry {
  const edge = (box: Box) => (anchor === 'top' ? -box.height / 2 + 2 : anchor === 'bottom' ? box.height / 2 - 2 : 0);
  const startX = from.left - container.left + from.width / 2 + startXOffset;
  const startY = from.top - container.top + from.height / 2 + edge(from) + startYOffset;
  const endX = to.left - container.left + to.width / 2 + endXOffset;
  const endY = to.top - container.top + to.height / 2 + edge(to) + endYOffset;
  const controlY = startY - curvature;
  const r = (n: number) => Math.round(n * 100) / 100;
  return { path: `M ${r(startX)},${r(startY)} Q ${r((startX + endX) / 2)},${r(controlY)} ${r(endX)},${r(endY)}`, width: container.width, height: container.height, startX: r(startX), endX: r(endX) };
}

/** Gradient vector keyframes (user-space px) for a pulse whose head travels a→b and whose tail trails it by `tail` px. */
export function pulseKeyframes(a: number, b: number, tail: number) {
  const d = b >= a ? 1 : -1;
  return { x1: [a, b + d * tail], x2: [a - d * tail, b] };
}

/**
 * Bounding box in the element's own CSS pixels. Under CSS `zoom` (the workstation's interface-size control) getBoundingClientRect reports
 * zoomed pixels, so every box is divided by the zoom factor (rendered width / layout width of the container); without it the SVG came
 * out too large and the path drifted off its nodes. The factor is 1 wherever the browser already reports unzoomed values.
 */
const boxOf = (node: Element, scale = 1): Box => { const b = node.getBoundingClientRect(); return { left: b.left / scale, top: b.top / scale, width: b.width / scale, height: b.height / scale }; };
const zoomOf = (container: HTMLElement): number => {
  const factor = container.offsetWidth > 0 ? container.getBoundingClientRect().width / container.offsetWidth : 1;
  return Number.isFinite(factor) && factor > 0 ? factor : 1;
};

export function AnimatedBeam({
  containerRef, fromRef, toRef, direction = 'forward', curvature = 0, duration = 2.4, delay = 0,
  pathWidth = 2, pathOpacity = 0.35, pathColor = 'var(--hairline-strong)',
  gradientStartColor = 'var(--accent-indigo)', gradientStopColor = 'var(--accent-cyan)',
  anchor = 'center', startXOffset = 0, startYOffset = 0, endXOffset = 0, endYOffset = 0, active = true, className,
}: AnimatedBeamProps) {
  const id = useId().replace(/:/g, '');
  const reduce = useReducedMotion();
  const [geometry, setGeometry] = useState<BeamGeometry | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const measure = () => {
      const from = fromRef.current, to = toRef.current;
      if (!from || !to) return;
      const k = zoomOf(container);
      const next = computeBeamPath(boxOf(container, k), boxOf(from, k), boxOf(to, k), { curvature, anchor, startXOffset, startYOffset, endXOffset, endYOffset });
      setGeometry(prev => (prev && prev.path === next.path && prev.width === next.width && prev.height === next.height ? prev : next));
    };
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure);
    [container, fromRef.current, toRef.current].forEach(node => node && observer.observe(node));
    window.addEventListener('resize', measure);
    container.addEventListener('scroll', measure, true); // a scrolling row of nodes (capture: scroll does not bubble)
    return () => { observer.disconnect(); window.removeEventListener('resize', measure); container.removeEventListener('scroll', measure, true); };
  }, [containerRef, fromRef, toRef, curvature, anchor, startXOffset, startYOffset, endXOffset, endYOffset]);

  if (!geometry) return null;
  const animate = active && !reduce;
  // The pulse is a gradient window sliding along the path's own x-span (user-space px), so short hops animate as fully as long ones.
  const span = Math.abs(geometry.endX - geometry.startX), w = geometry.width;
  const tail = Math.max(span * 0.4, 24);
  const forward = pulseKeyframes(geometry.startX, geometry.endX, tail);
  const reverse = pulseKeyframes(geometry.endX, geometry.startX, tail);
  const lanes: { key: string; keyframes: { x1: number[]; x2: number[] }; offset: number }[] =
    direction === 'forward' ? [{ key: 'f', keyframes: forward, offset: 0 }]
    : direction === 'reverse' ? [{ key: 'r', keyframes: reverse, offset: 0 }]
    : [{ key: 'f', keyframes: forward, offset: 0 }, { key: 'r', keyframes: reverse, offset: duration / 2 }];

  return (
    <svg
      className={cn('ul-beam', className)}
      width={w} height={geometry.height}
      viewBox={`0 0 ${w} ${geometry.height}`}
      fill="none" aria-hidden="true" data-direction={direction} data-active={animate ? 'true' : 'false'}
    >
      <path d={geometry.path} stroke={pathColor} strokeWidth={pathWidth} strokeOpacity={pathOpacity} strokeLinecap="round" />
      {animate && lanes.map(lane => (
        <path key={lane.key} d={geometry.path} stroke={`url(#${id}-${lane.key})`} strokeWidth={pathWidth} strokeLinecap="round" className="ul-beam__pulse" />
      ))}
      <defs>
        {lanes.map(lane => (
          <motion.linearGradient
            key={lane.key} id={`${id}-${lane.key}`} gradientUnits="userSpaceOnUse"
            x1={lane.keyframes.x1[0]} x2={lane.keyframes.x2[0]} y1={0} y2={0}
            animate={animate ? { x1: lane.keyframes.x1, x2: lane.keyframes.x2 } : undefined}
            transition={{ duration, delay: delay + lane.offset, ease: [0.16, 1, 0.3, 1], repeat: Infinity, repeatDelay: 0.4 }}
          >
            <stop stopColor={gradientStartColor} stopOpacity="0" />
            <stop stopColor={gradientStartColor} />
            <stop offset="32.5%" stopColor={gradientStopColor} />
            <stop offset="100%" stopColor={gradientStopColor} stopOpacity="0" />
          </motion.linearGradient>
        ))}
      </defs>
    </svg>
  );
}

export interface ConvergingBeamsProps extends Omit<AnimatedBeamProps, 'fromRef' | 'toRef' | 'curvature' | 'delay'> {
  sources: RefObject<HTMLElement | null>[];
  target: RefObject<HTMLElement | null>;
  /** Largest bow in px; sources are spread symmetrically between -spread and +spread. */
  spread?: number;
  /** Seconds between consecutive sources' pulses. */
  stagger?: number;
}

/** Multi-input convergence: every source feeds one target. Curvature fans out so overlapping paths stay distinguishable. */
export function ConvergingBeams({ sources, target, spread = 60, stagger = 0.35, ...beam }: ConvergingBeamsProps) {
  const last = Math.max(sources.length - 1, 1);
  return (
    <>
      {sources.map((source, index) => (
        <AnimatedBeam
          key={index} {...beam} fromRef={source} toRef={target}
          curvature={sources.length === 1 ? 0 : -spread + (2 * spread * index) / last}
          delay={index * stagger}
        />
      ))}
    </>
  );
}

export default AnimatedBeam;
