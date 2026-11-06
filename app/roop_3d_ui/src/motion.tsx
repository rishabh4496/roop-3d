// Shared motion vocabulary (ported from Roop Ultimate's react-ui): one set of springs and entrance variants so
// every animated surface feels like the same instrument. Signature easing: cubic-bezier(0.16, 1, 0.3, 1).
// Tab-level transitions are pure CSS (see shell.css `view-in`) and compositor-only; this module is for cards.
import { useRef, type ReactNode, type MouseEvent } from 'react';
import { motion, useMotionValue, useReducedMotion, useSpring, useTransform, type Variants } from 'framer-motion';

export { motion, AnimatePresence, MotionConfig, LayoutGroup, useReducedMotion } from 'framer-motion';

export const spring = {
  snappy: { type: 'spring', stiffness: 520, damping: 34, mass: 0.7 },
  smooth: { type: 'spring', stiffness: 260, damping: 30 },
  bouncy: { type: 'spring', stiffness: 460, damping: 20, mass: 0.8 },
  soft: { type: 'spring', stiffness: 170, damping: 22 },
} as const;

export const EASE_SETTLE = [0.16, 1, 0.3, 1] as const;

/** Rise + settle entrance for cards and panels. No filter blur: panels already carry a backdrop blur. */
export const fadeUp: Variants = {
  hidden: { opacity: 0, y: 24, scale: 0.98 },
  show: { opacity: 1, y: 0, scale: 1, transition: { ...spring.smooth, opacity: { duration: 0.4 } } },
  exit: { opacity: 0, y: -12, scale: 0.985, transition: { duration: 0.2 } },
};

export const staggerParent: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06, delayChildren: 0.05 } },
  exit: { transition: { staggerChildren: 0.02, staggerDirection: -1 } },
};

export function Stagger({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <motion.div className={className} variants={staggerParent} initial="hidden" animate="show">{children}</motion.div>;
}

export function Reveal({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <motion.div className={className} variants={fadeUp}>{children}</motion.div>;
}

/**
 * Mouse-follow tilt with a cursor-tracking glare. Runs on motion values, not React state, so hovering never
 * re-renders. Honours prefers-reduced-motion.
 */
export function useTilt({ max = 7, perspective = 900 } = {}) {
  const ref = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();
  const px = useMotionValue(0.5), py = useMotionValue(0.5);
  const rx = useSpring(useTransform(py, [0, 1], [max, -max]), spring.soft);
  const ry = useSpring(useTransform(px, [0, 1], [-max, max]), spring.soft);
  const gx = useTransform(px, [0, 1], ['0%', '100%']);
  const gy = useTransform(py, [0, 1], ['0%', '100%']);
  const glare = useTransform([gx, gy], ([x, y]) => `radial-gradient(520px circle at ${x} ${y}, var(--accent-glow), transparent 42%)`);
  const onMouseMove = (event: MouseEvent) => {
    if (reduce || !ref.current) return;
    const box = ref.current.getBoundingClientRect();
    px.set((event.clientX - box.left) / box.width);
    py.set((event.clientY - box.top) / box.height);
  };
  const onMouseLeave = () => { px.set(0.5); py.set(0.5); };
  const style = reduce ? undefined : { rotateX: rx, rotateY: ry, transformPerspective: perspective };
  return { ref, reduce, onMouseMove, onMouseLeave, style, glare };
}

export function TiltCard({ children, className = '', max = 7, ...rest }: { children: ReactNode; className?: string; max?: number } & Record<string, unknown>) {
  const tilt = useTilt({ max });
  return (
    <motion.div ref={tilt.ref} onMouseMove={tilt.onMouseMove} onMouseLeave={tilt.onMouseLeave} style={tilt.style} className={`tilt-card ${className}`} {...rest}>
      {children}
      {!tilt.reduce && <motion.div aria-hidden className="tilt-glare" style={{ background: tilt.glare }} />}
    </motion.div>
  );
}
