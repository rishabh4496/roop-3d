/**
 * Stage 23: hand-painted keep/exclude occlusion masks that apply to a frame range instead of the whole clip.
 *
 * A range is inclusive on both ends; `end_frame: null` means "until the clip ends". The server rejects overlapping ranges
 * (`mask_keyframes` on a job), so every insert here keeps the list sorted and non-overlapping: a newer stroke wins the
 * frames it covers and trims or splits whatever it lands on.
 */
export interface MaskKeyframe { start_frame: number; end_frame: number | null; mask_base64: string }

const endOf = (item: MaskKeyframe) => (item.end_frame === null ? Number.POSITIVE_INFINITY : item.end_frame);
const sorted = (items: MaskKeyframe[]) => [...items].sort((a, b) => a.start_frame - b.start_frame);

/** The range mask covering `frame`, if any. */
export function keyframeAt(keyframes: readonly MaskKeyframe[] | null | undefined, frame: number): MaskKeyframe | null {
  if (!keyframes?.length || !Number.isFinite(frame)) return null;
  return keyframes.find(item => frame >= item.start_frame && frame <= endOf(item)) ?? null;
}

/** The mask the compositor uses on `frame`: a painted range wins over the whole-clip mask (mirrors `FrameEditPlan`). */
export function maskAtFrame(whole: string | null | undefined, keyframes: readonly MaskKeyframe[] | null | undefined, frame: number): string | null {
  return keyframeAt(keyframes, frame)?.mask_base64 ?? whole ?? null;
}

/** Add `next`, which owns the frames it covers: overlapped ranges are trimmed, split around it, or dropped when swallowed. */
export function insertMaskKeyframe(items: readonly MaskKeyframe[], next: MaskKeyframe): MaskKeyframe[] {
  const start = Math.max(0, Math.floor(next.start_frame));
  const end = next.end_frame === null ? null : Math.max(start, Math.floor(next.end_frame));
  const incoming: MaskKeyframe = { start_frame: start, end_frame: end, mask_base64: next.mask_base64 };
  const incomingEnd = endOf(incoming);
  const kept: MaskKeyframe[] = [];
  for (const item of items) {
    if (endOf(item) < start || item.start_frame > incomingEnd) { kept.push(item); continue; } // no overlap
    if (item.start_frame < start) kept.push({ ...item, end_frame: start - 1 });               // keep the part before
    if (endOf(item) > incomingEnd) kept.push({ ...item, start_frame: incomingEnd + 1 });       // keep the part after
  }
  return sorted([...kept, incoming]);
}

export function removeMaskKeyframe(items: readonly MaskKeyframe[], startFrame: number): MaskKeyframe[] {
  return items.filter(item => item.start_frame !== startFrame);
}

export function describeRange(item: Pick<MaskKeyframe, 'start_frame' | 'end_frame'>): string {
  if (item.end_frame === item.start_frame) return `frame ${item.start_frame}`;
  return `frames ${item.start_frame}–${item.end_frame ?? 'end'}`;
}

/** Tolerant reader for saved projects / jobs: drops anything malformed rather than failing the whole load. */
export function readMaskKeyframes(value: unknown): MaskKeyframe[] {
  if (!Array.isArray(value)) return [];
  const clean = value.flatMap((raw): MaskKeyframe[] => {
    const item = raw as Partial<MaskKeyframe> | null;
    if (!item || typeof item.mask_base64 !== 'string' || !Number.isInteger(item.start_frame) || (item.start_frame as number) < 0) return [];
    const end = item.end_frame;
    if (end !== null && end !== undefined && (!Number.isInteger(end) || end < (item.start_frame as number))) return [];
    return [{ start_frame: item.start_frame as number, end_frame: end ?? null, mask_base64: item.mask_base64 }];
  });
  return clean.reduce<MaskKeyframe[]>((acc, item) => insertMaskKeyframe(acc, item), []);
}
