import { useSyncExternalStore } from 'react';

/**
 * Motion preference: follow the operating system ("system"), always reduce ("reduce"), or always allow ("full").
 * One effective boolean drives both the CSS (`:root[data-motion="reduce"]`) and the JavaScript animation loops
 * (idle turntable, eased camera, smooth scrolling), so the setting really stops motion everywhere.
 */
export type MotionPref = 'system' | 'reduce' | 'full';
export const MOTION_KEY = 'roop3d-motion-v1';

const listeners = new Set<() => void>();
const query = () => (typeof window !== 'undefined' && typeof window.matchMedia === 'function' ? window.matchMedia('(prefers-reduced-motion: reduce)') : null);

export function getMotionPref(): MotionPref {
  try { const value = localStorage.getItem(MOTION_KEY); return value === 'reduce' || value === 'full' ? value : 'system'; } catch { return 'system'; }
}

/** True when animation should be suppressed right now. */
export function reducedMotion(pref: MotionPref = getMotionPref()): boolean {
  if (pref === 'reduce') return true;
  if (pref === 'full') return false;
  return !!query()?.matches;
}

export function applyMotionPref(pref: MotionPref = getMotionPref()): void {
  if (typeof document === 'undefined') return;
  document.documentElement.dataset.motion = reducedMotion(pref) ? 'reduce' : 'full';
}

export function setMotionPref(pref: MotionPref): void {
  try { if (pref === 'system') localStorage.removeItem(MOTION_KEY); else localStorage.setItem(MOTION_KEY, pref); } catch { /* applies for this session only */ }
  applyMotionPref(pref);
  listeners.forEach(listener => listener());
}

/** Keep `data-motion` in step with the OS setting while the preference is "system". Returns an unsubscribe. */
export function watchSystemMotion(): () => void {
  const media = query();
  if (!media) return () => undefined;
  const onChange = () => { applyMotionPref(); listeners.forEach(listener => listener()); };
  media.addEventListener?.('change', onChange);
  return () => media.removeEventListener?.('change', onChange);
}

const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; };
/** `[preference, effectiveReduced, setPreference]` for React components. */
export function useMotionPref(): [MotionPref, boolean, (pref: MotionPref) => void] {
  const pref = useSyncExternalStore(subscribe, getMotionPref, () => 'system' as MotionPref);
  const reduced = useSyncExternalStore(subscribe, () => reducedMotion(), () => false);
  return [pref, reduced, setMotionPref];
}
