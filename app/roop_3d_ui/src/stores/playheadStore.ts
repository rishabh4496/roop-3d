import { createAtomicStore, useStoreSelector } from './atomicStore';

/**
 * Stage 11: the live playhead timecode (seconds).
 *
 * `App` keeps a *committed* `time` in React state: it changes on seek, frame-step, pause and when a target loads, so
 * previews, live-swap requests and project dirtiness react to it. While a video is *playing* the browser fires
 * `timeupdate` many times a second; those ticks go only into this store, so only the playhead needle, the timecode
 * readout and the face overlay re-render. On pause the final value is committed back to `App`.
 */
export const playheadStore = createAtomicStore<{ time: number }>({ time: 0 });

export const setPlayhead = (time: number) => { if (Number.isFinite(time)) playheadStore.set({ time }); };
export const getPlayhead = () => playheadStore.get().time;

/** Re-render only when `selector(time)` changes, e.g. `usePlayhead(t => Math.floor(t * fps))` for frame-accurate UI. */
export function usePlayhead<S = number>(selector: (time: number) => S = time => time as unknown as S): S {
  return useStoreSelector(playheadStore, state => selector(state.time));
}
