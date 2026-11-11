import { describe, expect, it } from 'vitest';
import { describeRange, insertMaskKeyframe, keyframeAt, maskAtFrame, readMaskKeyframes, removeMaskKeyframe, type MaskKeyframe } from './utils/maskRanges';

const kf = (start: number, end: number | null, tag = `${start}-${end}`): MaskKeyframe => ({ start_frame: start, end_frame: end, mask_base64: tag });
const spans = (items: MaskKeyframe[]) => items.map(item => [item.start_frame, item.end_frame, item.mask_base64]);

describe('mask frame ranges', () => {
  it('lets a painted range win over the whole-clip mask only inside the range', () => {
    const items = [kf(10, 20, 'a')];
    expect(maskAtFrame('whole', items, 15)).toBe('a');
    expect(maskAtFrame('whole', items, 21)).toBe('whole');
    expect(maskAtFrame(null, items, 5)).toBeNull();
    expect(keyframeAt(items, 10)?.mask_base64).toBe('a');
    expect(keyframeAt(items, 20)?.mask_base64).toBe('a');
    expect(keyframeAt([kf(30, null, 'open')], 9999)?.mask_base64).toBe('open');
  });

  it('keeps ranges sorted and never overlapping: the newest stroke owns the frames it covers', () => {
    let items: MaskKeyframe[] = [];
    items = insertMaskKeyframe(items, kf(0, 99, 'base'));
    items = insertMaskKeyframe(items, kf(40, 59, 'mid'));
    expect(spans(items)).toEqual([[0, 39, 'base'], [40, 59, 'mid'], [60, 99, 'base']]); // split around the new range
    items = insertMaskKeyframe(items, kf(30, 70, 'wide'));
    expect(spans(items)).toEqual([[0, 29, 'base'], [30, 70, 'wide'], [71, 99, 'base']]); // swallowed + trimmed
    items = insertMaskKeyframe(items, kf(80, null, 'tail'));
    expect(spans(items)).toEqual([[0, 29, 'base'], [30, 70, 'wide'], [71, 79, 'base'], [80, null, 'tail']]);
    for (let i = 1; i < items.length; i += 1) expect(items[i].start_frame).toBeGreaterThan(items[i - 1].end_frame as number);
  });

  it('replaces a range painted again on the same frames and removes by start frame', () => {
    const items = insertMaskKeyframe([kf(5, 9, 'old')], kf(5, 9, 'new'));
    expect(spans(items)).toEqual([[5, 9, 'new']]);
    expect(removeMaskKeyframe(items, 5)).toEqual([]);
    expect(removeMaskKeyframe(items, 6)).toHaveLength(1);
  });

  it('normalises a reversed or fractional range and describes it', () => {
    expect(spans(insertMaskKeyframe([], kf(4.9, 2, 'x')))).toEqual([[4, 4, 'x']]);
    expect(describeRange(kf(7, 7))).toBe('frame 7');
    expect(describeRange(kf(7, 9))).toBe('frames 7–9');
    expect(describeRange(kf(7, null))).toBe('frames 7–end');
  });

  it('reads saved data tolerantly and repairs overlaps', () => {
    expect(readMaskKeyframes(undefined)).toEqual([]);
    const loaded = readMaskKeyframes([{ start_frame: 0, end_frame: 10, mask_base64: 'a' }, { start_frame: 5, end_frame: 20, mask_base64: 'b' },
      { start_frame: -1, end_frame: 3, mask_base64: 'bad' }, { start_frame: 3, end_frame: 1, mask_base64: 'bad' }, { start_frame: 1, mask_base64: 7 }, null]);
    expect(spans(loaded)).toEqual([[0, 4, 'a'], [5, 20, 'b']]);
  });
});
