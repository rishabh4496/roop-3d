import { createElement, type ComponentPropsWithoutRef, type ElementType, type ReactNode } from 'react';
import { motion, useReducedMotion, type Variants } from 'framer-motion';

/**
 * Timeline animation (ui-layouts "Timeline Animation"): children reveal one after another as they scroll into view.
 * `animationNum` is the position in the sequence; each step starts `STEP` seconds after the previous one.
 * Under reduced motion everything is simply shown.
 */
export interface ViewportOptions { amount?: number; margin?: string; once?: boolean }
/**
 * Deliberately not upstream's `{ amount: 0.3, margin: '0px 0px -120px 0px' }`: that margin shrinks the bottom of the viewport by 120px, so
 * anything docked within 120px of the window's bottom edge (the frame dock, the last queue card) would never count as "in view" and
 * would stay invisible. This app lays out panels inside a fixed shell, not a long page, so the reveal triggers on any visibility.
 */
export const DEFAULT_VIEWPORT: Required<ViewportOptions> = { amount: 0.05, margin: '0px', once: true };
export const STEP = 0.1;
export const SLOW_STEP = 0.25;

/** Seconds before step `index` begins. Negative or non-finite indices start immediately. */
export function revealDelay(index: number, slow = false): number {
  return Number.isFinite(index) && index > 0 ? index * (slow ? SLOW_STEP : STEP) : 0;
}

export const defaultVariants = (slow = false): Variants => ({
  hidden: { opacity: 0, y: 14, filter: 'blur(6px)' },
  visible: (index: number = 0) => ({ opacity: 1, y: 0, filter: 'blur(0px)', transition: { delay: revealDelay(index, slow), duration: 0.45, ease: [0.22, 1, 0.36, 1] } }),
});

type Tag = Extract<ElementType, keyof typeof motion>;
export type TimelineAnimationProps<T extends Tag = 'div'> = {
  as?: T;
  animationNum?: number;
  viewport?: ViewportOptions;
  customVariants?: Variants;
  /** Use the slower 0.25s step between items. */
  delay?: boolean;
  className?: string;
  children?: ReactNode;
} & Omit<ComponentPropsWithoutRef<typeof motion.div>, 'children' | 'variants' | 'custom' | 'viewport' | 'initial' | 'whileInView'>;

export function TimelineAnimation<T extends Tag = 'div'>({
  as, animationNum = 0, viewport, customVariants, delay = false, className, children, ...rest
}: TimelineAnimationProps<T>) {
  const reduce = useReducedMotion();
  // Without IntersectionObserver (old webviews, jsdom) a scroll reveal cannot fire and framer would throw: just show the content.
  const still = !!reduce || typeof IntersectionObserver === 'undefined';
  const Component = (motion as unknown as Record<string, ElementType>)[as ?? 'div'] ?? motion.div;
  return createElement(Component, {
    ...rest,
    className,
    custom: animationNum,
    variants: customVariants ?? defaultVariants(delay),
    initial: still ? 'visible' : 'hidden',
    ...(still ? {} : { whileInView: 'visible', viewport: { ...DEFAULT_VIEWPORT, ...viewport } }),
  }, children);
}
