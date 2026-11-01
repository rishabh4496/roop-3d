import type { ReactNode } from 'react';
import { usePlayhead } from '../../stores/playheadStore';

/**
 * Render-prop leaf that follows the live playhead only while `live` (playing). When paused it shows the committed
 * `fallback` time and ignores per-tick store writes, so a playback tick re-renders this subtree and nothing above it.
 */
export function LiveTime({ fallback, live, children }: { fallback: number; live: boolean; children: (time: number) => ReactNode }) {
  const time = usePlayhead(playhead => (live ? playhead : fallback));
  return <>{children(time)}</>;
}
