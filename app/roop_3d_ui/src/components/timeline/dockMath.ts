/** Pure maths for the Stage 24 timeline dock (Embla filmstrip + trim rail). No React, no DOM: everything here is unit-tested. */

export const SPEEDS = [0.25, 0.5, 1, 2, 4] as const;
export type Speed = (typeof SPEEDS)[number];

export interface DockFrame { index: number; timestamp: number; url: string; frameIndex: number | null }

const finite = (value: number, fallback = 0) => (Number.isFinite(value) ? value : fallback);
export const clamp = (value: number, low: number, high: number) => Math.min(Math.max(value, low), Math.max(low, high));

/** Where `time` sits on a track of `duration`, as a 0..100 percentage. An unknown duration parks at 0, never NaN%. */
export function percentOf(time: number, duration: number): number {
  return duration > 0 ? clamp((finite(time) / duration) * 100, 0, 100) : 0;
}

/** Compact ruler label: `m:ss.f` under an hour, `h:mm:ss` above; the fraction is dropped when `precise` is false. */
export function rulerLabel(seconds: number, precise = true): string {
  const total = Math.max(0, finite(seconds));
  const whole = Math.floor(total);
  const h = Math.floor(whole / 3600), m = Math.floor((whole % 3600) / 60), s = whole % 60;
  const tenth = Math.floor((total - whole) * 10);
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}${precise ? `.${tenth}` : ''}`;
}

/** `count` evenly spaced sample times across the clip (slide centres), used when the server filmstrip is unavailable. */
export function placeholderFrames(duration: number, count: number): DockFrame[] {
  const n = Math.max(1, Math.floor(count));
  const length = Math.max(0, finite(duration));
  return Array.from({ length: n }, (_, index) => ({ index, timestamp: ((index + 0.5) / n) * length, url: '', frameIndex: null }));
}

/** Index of the slide that owns `time`: the last slide whose timestamp is at or before it (0 before the first). */
export function activeSlide(frames: readonly Pick<DockFrame, 'timestamp'>[], time: number): number {
  let found = 0;
  for (let i = 0; i < frames.length; i += 1) { if (frames[i].timestamp <= time + 1e-6) found = i; else break; }
  return found;
}

/** Slide `index` covers [its timestamp, the next slide's timestamp); the first slide also owns the clip start. Used to dim slides outside the In/Out range. */
export function slideSpan(frames: readonly Pick<DockFrame, 'timestamp'>[], index: number, duration: number): [number, number] {
  const start = index === 0 ? 0 : frames[index]?.timestamp ?? 0;
  const end = index + 1 < frames.length ? frames[index + 1].timestamp : Math.max(start, duration);
  return [start, end];
}
export function inTrimRange(span: [number, number], trimStart: number, trimEnd: number | null): boolean {
  return span[1] > trimStart && (trimEnd === null || span[0] < trimEnd);
}

/** The slide a chapter marker belongs to, so each slide can show a flag. */
export function chapterSlides(chapters: readonly { id: string; time: number }[], frames: readonly Pick<DockFrame, 'timestamp'>[]): Map<number, string[]> {
  const map = new Map<number, string[]>();
  for (const chapter of chapters) {
    const index = activeSlide(frames, chapter.time);
    map.set(index, [...(map.get(index) ?? []), chapter.id]);
  }
  return map;
}

export interface Trim { start: number; end: number | null }
/**
 * Move one trim handle to `time`, keeping at least one frame between In and Out and never leaving the clip.
 * An Out at (or beyond) the end is stored as `null`, meaning "to the end of the clip", which is how jobs encode it.
 */
export function moveTrimHandle(which: 'in' | 'out', time: number, trim: Trim, duration: number, fps: number): Trim {
  const frame = 1 / Math.max(1, finite(fps, 30));
  const length = Math.max(0, finite(duration));
  const end = trim.end ?? length;
  if (which === 'in') return { start: clamp(finite(time), 0, Math.max(0, end - frame)), end: trim.end };
  const next = clamp(finite(time), Math.min(length, trim.start + frame), length);
  return { start: trim.start, end: next >= length - 1e-6 ? null : next };
}

/** Keyboard step for a handle: one frame, or one second with Shift (a tenth of the clip with Ctrl is deliberately not offered). */
export function stepSeconds(fps: number, shift: boolean): number {
  return shift ? 1 : 1 / Math.max(1, finite(fps, 30));
}

export function nextSpeed(current: number, direction: 1 | -1): Speed {
  const index = SPEEDS.findIndex(speed => speed === current);
  const from = index === -1 ? SPEEDS.indexOf(1) : index;
  return SPEEDS[clamp(from + direction, 0, SPEEDS.length - 1)];
}
