import { createAtomicStore, useStoreSelector } from './atomicStore';
import { SPEEDS } from '../components/timeline/dockMath';

/**
 * One playback speed for the whole workstation (0.25x - 4x). The timeline's own speed buttons and the Stage 24 dock both
 * write it, and every `<video>` that follows it (the viewport's two panes, the finishing preview) reads it.
 */
const store = createAtomicStore<{ rate: number }>({ rate: 1 });

/** Snaps to the nearest supported speed; anything non-finite is ignored. */
export function setPlaybackRate(rate: number): number {
  if (!Number.isFinite(rate)) return store.get().rate;
  const snapped = SPEEDS.reduce((best, speed) => (Math.abs(speed - rate) < Math.abs(best - rate) ? speed : best), SPEEDS[0]);
  store.set({ rate: snapped });
  return snapped;
}
export const getPlaybackRate = () => store.get().rate;
export const usePlaybackRate = () => useStoreSelector(store, state => state.rate);
export const resetPlaybackRate = () => store.reset();
