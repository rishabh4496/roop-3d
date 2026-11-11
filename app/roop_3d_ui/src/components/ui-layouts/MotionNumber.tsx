import { useCallback, useEffect, useRef, useState } from 'react';
import { useReducedMotion, useSpring } from 'framer-motion';
import { cn } from '../../lib/utils';

/**
 * Spring-interpolated numeric ticker for live telemetry (FPS, VRAM GB, stage latency ms, 6DoF angles).
 *
 * The animated text is written directly to the DOM from the spring's change event, so a 60 Hz ticker costs no React renders.
 * Layout cannot jitter: digits are tabular (`.ul-number`), and the number sits in an inline-block sized to the widest value
 * it can show (`minChars`, measured in `ch`). Screen readers get the settled value from a visually hidden sibling, never the in-flight frames.
 */
export interface MotionNumberProps {
  value: number;
  /** Fraction digits. Default 0. */
  decimals?: number;
  prefix?: string;
  suffix?: string;
  /** Always show a leading + for positive values (angles, deltas). */
  signed?: boolean;
  /** Reserve this many characters of width so growing values never push neighbours. Default: the formatted length of `value`. */
  minChars?: number;
  /** Spring tuning. Defaults feel like a gauge needle: quick, no overshoot. */
  stiffness?: number;
  damping?: number;
  className?: string;
  /** Overrides the formatter entirely; receives the in-flight value. */
  format?: (value: number) => string;
}

export function formatNumber(value: number, { decimals = 0, signed = false, prefix = '', suffix = '' }: { decimals?: number; signed?: boolean; prefix?: string; suffix?: string } = {}): string {
  if (!Number.isFinite(value)) return `${prefix}—${suffix}`;
  // Avoid "-0" / "-0.0": a needle settling at zero from below must not flash a minus sign.
  const fixed = Number(value.toFixed(decimals));
  const text = Math.abs(fixed).toFixed(decimals);
  const sign = fixed < 0 ? '−' : signed && fixed > 0 ? '+' : '';
  return `${prefix}${sign}${text}${suffix}`;
}

export function MotionNumber({ value, decimals = 0, prefix = '', suffix = '', signed = false, minChars, stiffness = 220, damping = 30, className, format }: MotionNumberProps) {
  const reduce = useReducedMotion();
  const node = useRef<HTMLSpanElement>(null);
  const render = (v: number) => (format ? format(v) : formatNumber(v, { decimals, signed, prefix, suffix }));
  const renderRef = useRef(render);
  renderRef.current = render;

  const spring = useSpring(Number.isFinite(value) ? value : 0, { stiffness, damping, mass: 0.8, restDelta: 0.001 });

  const write = useCallback((v: number) => { if (node.current) node.current.textContent = renderRef.current(v); }, []);
  const valueRef = useRef(value);
  valueRef.current = value;

  useEffect(() => {
    const stopChange = spring.on('change', write);
    // Whatever the last spring frame was, the DOM must end on the exact settled value.
    const stopDone = spring.on('animationComplete', () => write(valueRef.current));
    return () => { stopChange(); stopDone(); };
  }, [spring, write]);

  useEffect(() => {
    if (!Number.isFinite(value)) { write(value); return; }
    if (reduce) { spring.jump(value); write(value); }
    else spring.set(value);
  }, [value, reduce, spring, write]);

  // The live node's React children are fixed at mount: React never touches its text again, so the spring owns it
  // and a new `value` can't flash its final text for a frame before the animation starts.
  const [initial] = useState(() => render(value));
  const settled = render(value);
  const width = minChars ?? settled.length;
  return (
    <span className={cn('ul-number ul-motion-number', className)} data-value={Number.isFinite(value) ? value : ''}>
      <span ref={node} className="ul-motion-number__live" aria-hidden="true" style={{ minWidth: `${width}ch` }}>{initial}</span>
      <span className="ul-sr-only">{settled}</span>
    </span>
  );
}

export default MotionNumber;
