import { describe, expect, it } from 'vitest';
import { PreviewFrameCache } from './previewFrameCache';

const frame = (size = 4) => ({ blob: new Blob([new Uint8Array(size)]), meta: null, cached: false });

describe('preview frame cache', () => {
  it('evicts least recently used frames by count and encoded byte budget', () => {
    const cache = new PreviewFrameCache(24, 2);
    cache.put('a', frame(), 0); cache.put('b', frame(), 0);
    expect(cache.get('a', 1)?.cached).toBe(true);
    cache.put('c', frame(), 1);
    expect(cache.get('b', 2)).toBeUndefined();
    cache.put('d', frame(16), 2);
    expect(cache.get('a', 3)).toBeUndefined();
    expect(cache.get('c', 3)).toBeDefined();
    expect(cache.get('d', 3)).toBeDefined();
  });

  it('expires without extending freshness on reads and releases everything on clear', () => {
    const cache = new PreviewFrameCache(100, 12, 30);
    cache.put('a', frame(), 0);
    expect(cache.get('a', 29)).toBeDefined();
    expect(cache.get('a', 30)).toBeUndefined();
    cache.put('b', frame(), 30); cache.clear();
    expect(cache.get('b', 31)).toBeUndefined();
  });

  it('does not cache cancelled, empty or oversized results', () => {
    const cache = new PreviewFrameCache(10);
    cache.put('a', { ...frame(), cancelled: true });
    cache.put('b', frame(0)); cache.put('c', frame(20));
    for (const key of ['a', 'b', 'c']) expect(cache.get(key)).toBeUndefined();
  });
});
