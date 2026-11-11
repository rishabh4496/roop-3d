// Foundation utilities for the ui-layouts primitives (src/components/ui-layouts). Dependency-free on purpose: the workstation
// ships fully offline, so there is no clsx / tailwind-merge here. This app has no Tailwind either, so `cn` only joins class names.
import { useCallback, useSyncExternalStore } from 'react';

export type ClassValue = string | number | null | undefined | false | ClassValue[] | { [className: string]: unknown };

/** Joins class names, skipping falsy values. Accepts strings, arrays and `{ className: condition }` maps, like `clsx`. */
export function cn(...inputs: ClassValue[]): string {
  const out: string[] = [];
  const walk = (value: ClassValue) => {
    if (!value && value !== 0) return;
    if (typeof value === 'string') { if (value) out.push(value); }
    else if (typeof value === 'number') out.push(String(value));
    else if (Array.isArray(value)) value.forEach(walk);
    else if (typeof value === 'object') { for (const key of Object.keys(value)) if (value[key]) out.push(key); }
  };
  inputs.forEach(walk);
  return out.join(' ');
}

const matchMediaAvailable = () => typeof window !== 'undefined' && typeof window.matchMedia === 'function';

/**
 * Reactive `matchMedia`. Subscribes through `useSyncExternalStore`, so it is tear-free and correct on the first render
 * (no flash from a `false` default). Returns `false` where `matchMedia` does not exist (SSR, some test environments).
 */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback((notify: () => void) => {
    if (!matchMediaAvailable()) return () => {};
    const list = window.matchMedia(query);
    list.addEventListener('change', notify);
    return () => list.removeEventListener('change', notify);
  }, [query]);
  const getSnapshot = useCallback(() => (matchMediaAvailable() ? window.matchMedia(query).matches : false), [query]);
  return useSyncExternalStore(subscribe, getSnapshot, () => false);
}

export const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

/** Decimal places needed to display `step` exactly (0.25 → 2, 5 → 0). */
export function decimalsOf(step: number): number {
  if (!Number.isFinite(step) || step <= 0 || Number.isInteger(step)) return 0;
  const text = String(step);
  if (text.includes('e-')) return Number(text.split('e-')[1]);
  return text.split('.')[1]?.length ?? 0;
}
