/** Chapter markers for the timeline: pure list operations plus tolerant (de)serialisation. */

export const MAX_CHAPTERS = 64;
const EPSILON = 1e-3;

/** @typedef {{ id: string, time: number, label: string }} Chapter */

/** @param {Chapter[]} chapters */
export const sortChapters = chapters => [...chapters].sort((a, b) => a.time - b.time);

/**
 * Add a chapter at `time` (clamped into the clip). A chapter already within one frame of `time` is returned instead
 * of adding a duplicate, so double-clicking "Add chapter" is harmless.
 * @param {Chapter[]} chapters
 * @param {number} time
 * @param {{ duration: number, fps?: number, label?: string }} options
 */
export function addChapter(chapters, time, { duration, fps = 30, label } = { duration: 0 }) {
  if (!Number.isFinite(time)) return chapters;
  const at = Math.max(0, Math.min(Math.max(duration, 0), time));
  if (chapters.some(chapter => Math.abs(chapter.time - at) < 1 / Math.max(1, fps) - EPSILON)) return chapters;
  if (chapters.length >= MAX_CHAPTERS) return chapters;
  const next = { id: `chapter-${Math.round(at * 1000)}-${chapters.length}`, time: Number(at.toFixed(3)), label: label?.trim() || `Chapter ${chapters.length + 1}` };
  return sortChapters([...chapters, next]);
}

/** @param {Chapter[]} chapters @param {string} id */
export const removeChapter = (chapters, id) => chapters.filter(chapter => chapter.id !== id);

/** @param {Chapter[]} chapters @param {string} id @param {string} label */
export const renameChapter = (chapters, id, label) =>
  chapters.map(chapter => (chapter.id === id ? { ...chapter, label: label.trim().slice(0, 48) || chapter.label } : chapter));

/** Chapter containing `time` (the last one at or before it), or null before the first chapter. */
export const chapterAt = (chapters, time) => [...sortChapters(chapters)].reverse().find(chapter => chapter.time <= time + EPSILON) ?? null;

/** Next / previous chapter relative to `time`, for keyboard navigation. */
export const adjacentChapter = (chapters, time, direction) => {
  const ordered = sortChapters(chapters);
  return direction > 0 ? ordered.find(chapter => chapter.time > time + EPSILON) ?? null : [...ordered].reverse().find(chapter => chapter.time < time - EPSILON) ?? null;
};

export const storageKey = target => `roop3d.chapters:${target}`;

/** Fired on `window` after chapters are saved, so other views of the same clip (the Stage 24 dock) refresh without a reload. */
export const CHAPTERS_CHANGED = 'roop3d:chapters';

/** Parse stored JSON defensively: anything malformed is dropped rather than trusted. */
export function parseChapters(text) {
  try {
    const value = JSON.parse(text ?? '[]');
    if (!Array.isArray(value)) return [];
    return sortChapters(value
      .filter(item => item && typeof item.id === 'string' && Number.isFinite(item.time) && item.time >= 0 && typeof item.label === 'string')
      .slice(0, MAX_CHAPTERS)
      .map(item => ({ id: item.id.slice(0, 64), time: item.time, label: item.label.slice(0, 48) })));
  } catch {
    return [];
  }
}

export function loadChapters(target) {
  if (!target) return [];
  try { return parseChapters(localStorage.getItem(storageKey(target))); } catch { return []; }
}

export function saveChapters(target, chapters) {
  if (!target) return;
  try { localStorage.setItem(storageKey(target), JSON.stringify(chapters)); } catch { /* storage unavailable: chapters stay in memory */ }
  try { window.dispatchEvent(new CustomEvent(CHAPTERS_CHANGED, { detail: { target } })); } catch { /* no window (SSR / tests) */ }
}
