// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { renderToStaticMarkup } from 'react-dom/server';
import TimelineWorkstation from './components/timeline/TimelineWorkstation';
import { addChapter, adjacentChapter, chapterAt, loadChapters, parseChapters, removeChapter, renameChapter, saveChapters, storageKey, MAX_CHAPTERS } from './components/timeline/chapters';
import WorkstationCanvas, { DIFF_NOISE_FLOOR, LOUPE_ZOOM, MODES, clientToMedia, crossfadeValue, diffHeatmap } from './components/canvas/WorkstationCanvas';
import { calculateLetterbox } from './utils/letterbox';

describe('chapter list operations', () => {
  const at = (...times: number[]) => times.reduce((list, time) => addChapter(list, time, { duration: 60, fps: 25 }), [] as ReturnType<typeof parseChapters>);

  it('adds chapters in time order with default names, clamped into the clip', () => {
    const list = at(30, 5, 99);
    expect(list.map(chapter => [chapter.time, chapter.label])).toEqual([[5, 'Chapter 2'], [30, 'Chapter 1'], [60, 'Chapter 3']]);
    expect(addChapter([], -4, { duration: 60 })[0].time).toBe(0);
    expect(addChapter([], 10, { duration: 60, label: '  Intro  ' })[0].label).toBe('Intro');
    expect(addChapter([], NaN, { duration: 60 })).toEqual([]);
  });

  it('ignores a second marker within one frame and caps the list', () => {
    const list = at(10);
    expect(addChapter(list, 10.02, { duration: 60, fps: 25 })).toBe(list);              // < 1 frame (0.04 s) away
    expect(addChapter(list, 10.1, { duration: 60, fps: 25 })).toHaveLength(2);
    let full = [] as ReturnType<typeof parseChapters>;
    for (let i = 0; i < MAX_CHAPTERS + 5; i++) full = addChapter(full, i * 0.5, { duration: 600, fps: 25 });
    expect(full).toHaveLength(MAX_CHAPTERS);
  });

  it('removes, renames, and finds the current / adjacent chapter', () => {
    const list = at(10, 20, 40);
    expect(removeChapter(list, list[1].id).map(chapter => chapter.time)).toEqual([10, 40]);
    expect(renameChapter(list, list[0].id, '  Opening  ')[0].label).toBe('Opening');
    expect(renameChapter(list, list[0].id, '   ')[0].label).toBe(list[0].label);       // blank keeps the old name
    expect([chapterAt(list, 5), chapterAt(list, 10)?.time, chapterAt(list, 39.9)?.time, chapterAt(list, 100)?.time]).toEqual([null, 10, 20, 40]);
    expect([adjacentChapter(list, 15, 1)?.time, adjacentChapter(list, 15, -1)?.time, adjacentChapter(list, 40, 1), adjacentChapter(list, 10, -1)]).toEqual([20, 10, null, null]);
  });

  it('drops malformed stored data instead of trusting it', () => {
    expect(parseChapters('not json')).toEqual([]);
    expect(parseChapters('{"a":1}')).toEqual([]);
    expect(parseChapters(JSON.stringify([{ id: 'x', time: 3, label: 'ok' }, { id: 5, time: 1, label: 'bad id' }, { id: 'y', time: -1, label: 'neg' }, { id: 'z', time: 'a', label: 'nan' }, null])))
      .toEqual([{ id: 'x', time: 3, label: 'ok' }]);
  });
});

describe('Timeline chapters', () => {
  let host: HTMLDivElement, root: Root;
  const onSeek = vi.fn();
  const mount = (props: Record<string, unknown> = {}) => act(async () => {
    root.render(<TimelineWorkstation target="roop_3d/.runtime/media/a.mp4" output="" currentTime={12} duration={60} fps={25} playing={false} keys={[]} trimStart={0} trimEnd={null} onSeek={onSeek} {...props}/>);
  });
  const click = (selector: string) => act(async () => (host.querySelector(selector) as HTMLElement).click());

  beforeEach(() => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true; localStorage.clear(); onSeek.mockReset();
    vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

  it('adds a chapter at the playhead, seeks to it from its flag, renames and removes it', async () => {
    await mount();
    await click('.chapter-btn.add');
    expect(host.querySelectorAll('.chapter-marker')).toHaveLength(1);
    const marker = host.querySelector<HTMLElement>('.chapter-marker')!;
    expect(marker.style.left).toBe('20%');                                 // 12 s of a 60 s clip
    expect(host.querySelector('.chapter-flag')!.getAttribute('aria-label')).toContain('Chapter Chapter 1 at 00:00:12:00');
    await click('.chapter-btn.add');                                       // same frame again: no duplicate
    expect(host.querySelectorAll('.chapter-marker')).toHaveLength(1);
    await click('.chapter-flag');
    expect(onSeek).toHaveBeenLastCalledWith(12);

    await act(async () => host.querySelector('.chapter-flag')!.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));
    const input = host.querySelector<HTMLInputElement>('.chapter-rename')!;
    await act(async () => { input.value = 'Scene A'; Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, 'Scene A'); input.dispatchEvent(new Event('input', { bubbles: true })); input.focus(); input.blur(); });
    expect(host.querySelector('.chapter-flag')!.textContent).toContain('Scene A');
    expect(host.querySelector('.chapter-current')!.textContent).toBe('Scene A');

    await click('.chapter-remove');
    expect(host.querySelectorAll('.chapter-marker')).toHaveLength(0);
  });

  it('navigates between chapters with previous / next', async () => {
    saveChapters('roop_3d/.runtime/media/a.mp4', [{ id: 'a', time: 5, label: 'A' }, { id: 'b', time: 30, label: 'B' }]);
    await mount({ currentTime: 12 });
    await click('.chapter-btn.next');
    expect(onSeek).toHaveBeenLastCalledWith(30);
    await click('.chapter-btn.prev');
    expect(onSeek).toHaveBeenLastCalledWith(5);
    await mount({ currentTime: 40 });
    expect(host.querySelector<HTMLButtonElement>('.chapter-btn.next')!.disabled).toBe(true);
  });

  it('persists per target clip and reloads when the clip changes', async () => {
    await mount();
    await click('.chapter-btn.add');
    expect(JSON.parse(localStorage.getItem(storageKey('roop_3d/.runtime/media/a.mp4'))!)).toHaveLength(1);
    await mount({ target: 'roop_3d/.runtime/media/b.mp4' });
    expect(host.querySelectorAll('.chapter-marker')).toHaveLength(0);       // another clip has its own chapters
    await mount({ target: 'roop_3d/.runtime/media/a.mp4' });
    expect(host.querySelectorAll('.chapter-marker')).toHaveLength(1);
    expect(loadChapters('')).toEqual([]);
  });

  it('still offers 0.25x to 4x playback speed, the ruler, the filmstrip and in/out trim', async () => {
    await mount();
    expect([...host.querySelectorAll('.speed-btn')].map(button => button.textContent)).toEqual(['0.25x', '0.5x', '1x', '2x', '4x']);
    for (const selector of ['.timecode-ruler-track', '.filmstrip-track', '.trim-handle-marker.in', '.trim-handle-marker.out']) expect(host.querySelector(selector), selector).toBeTruthy();
  });
});

describe('comparison canvas maths', () => {
  it('maps pointer positions to media pixels with sub-pixel precision', () => {
    const stage = { left: 100, top: 50, width: 800, height: 450 };
    const media = { width: 1920, height: 1080 };
    expect(clientToMedia(stage, media, 500, 275)).toEqual({ x: 960, y: 540, u: 0.5, v: 0.5 });
    const p = clientToMedia(stage, media, 100.25, 50.1)!;
    expect(p.x).toBeCloseTo(0.6, 5);
    expect(p.y).toBeCloseTo(0.24, 5);
    expect(clientToMedia(stage, media, 99, 100)).toBeNull();               // outside the picture
    expect(clientToMedia({ left: 0, top: 0, width: 0, height: 0 }, media, 1, 1)).toBeNull();
  });

  it('crossfades smoothly from original to swapped and back, repeating every period', () => {
    const v = (ms: number) => crossfadeValue(ms, 2);
    expect([v(0), v(1000), v(2000), v(4000)].map(x => Number(x.toFixed(6)))).toEqual([0, 1, 0, 0]);
    expect(v(500)).toBeCloseTo(0.5, 6);
    expect(v(500)).toBeCloseTo(v(1500), 6);                                // symmetric
  });

  it('shows exactly the pixels that changed, above the JPEG noise floor, and scales intensity with the boost', () => {
    const a = new Uint8ClampedArray([100, 100, 100, 255, 100, 100, 100, 255, 100, 100, 100, 255]);
    const b = new Uint8ClampedArray([100, 100, 100, 255, 102, 102, 102, 255, 160, 100, 100, 255]);
    const out = diffHeatmap(a, b, 2, new Uint8ClampedArray(12));
    expect(DIFF_NOISE_FLOOR).toBe(3);
    expect([...out.slice(0, 4)]).toEqual([15, 23, 26, 210]);                // identical: dark slate
    expect([...out.slice(4, 8)]).toEqual([15, 23, 26, 210]);                // 2-level noise: still "unchanged"
    expect(out[8]).toBe(Math.round((20 * 2 / 128) * 244));                  // real change (mean 20 levels, boost 2): red rises with intensity
    expect(out[11]).toBe(Math.round(180 + (20 * 2 / 128) * 75));             // and the cell becomes more opaque than an unchanged one
    const low = diffHeatmap(a, b, 1, new Uint8ClampedArray(12)), high = diffHeatmap(a, b, 8, new Uint8ClampedArray(12));
    expect(high[8]).toBeGreaterThan(low[8]);                                // boost makes the same change hotter
  });
});

describe('comparison canvas', () => {
  const props = { output: 'live', originalUrl: 'blob:original', swappedUrl: 'blob:swapped', containerWidth: 1000, containerHeight: 500, mediaDimensions: { width: 1920, height: 1080 } };
  const html = (extra: Record<string, unknown>) => renderToStaticMarkup(<WorkstationCanvas {...props} {...extra}/>);

  it('offers wipe, crossfade, alpha blend, diff map and a 3.5x loupe', () => {
    expect(MODES.map(mode => mode.label)).toEqual(['Wipe (A/B)', 'Crossfade', 'Alpha Blend', 'Diff Map', 'Loupe (3.5x)']);
    expect(LOUPE_ZOOM).toBe(3.5);
    for (const mode of MODES) expect(html({ mode: mode.id })).toContain(mode.label);
  });

  it('composites the real frames: original underneath, swapped on top, clipped by the wipe', () => {
    const wipe = html({ mode: 'wipe' });
    expect(wipe).toContain('class="compare-a" src="blob:original"');
    expect(wipe).toContain('class="compare-b" src="blob:swapped"');
    expect(wipe).toContain('clip-path:inset(0 0 0 50.00%)');                // swapped visible right of the divider
    expect(html({ mode: 'blend' })).toContain('opacity:0.5');
    expect(html({ mode: 'diff' })).not.toContain('class="compare-b"');      // the heat-map replaces the swapped layer
  });

  it('aligns the picture to the letterboxed media rectangle when full-bleed', () => {
    const box = calculateLetterbox(1000, 500, 1920, 1080);
    expect(box.displayW).toBeCloseTo(888.888, 2);
    const out = html({ mode: 'wipe', fullBleed: true });
    expect(out).toContain('full-bleed');
    expect(out).toContain(`left:${box.offsetX}px`);
    expect(out).toContain(`width:${box.displayW}px`);
  });

  describe('interaction', () => {
    let host: HTMLDivElement, root: Root;
    beforeEach(() => { (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true; host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
    afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

    it('keeps crossfading until paused, and shows the live blend', async () => {
      const callbacks: FrameRequestCallback[] = [];
      vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => { callbacks.push(cb); return callbacks.length; });
      vi.stubGlobal('cancelAnimationFrame', vi.fn());
      let clock = 0;
      await act(async () => root.render(<WorkstationCanvas {...props} mode="crossfade" now={() => clock}/>));
      const step = async (ms: number) => { clock = ms; await act(async () => { callbacks.splice(0).forEach(cb => cb(ms)); }); };
      await step(0);
      expect(host.querySelector<HTMLElement>('.compare-b')!.style.opacity).toBe('0');
      await step(1000);                                                      // half of a 2 s period: fully swapped
      expect(host.querySelector<HTMLElement>('.compare-b')!.style.opacity).toBe('1');
      expect(host.querySelector('.crossfade-controls .control-val')!.textContent).toBe('100% swapped');
      await act(async () => (host.querySelector('.crossfade-toggle') as HTMLButtonElement).click());
      expect(host.querySelector('.crossfade-toggle')!.getAttribute('aria-pressed')).toBe('false');
      const pending = callbacks.length;
      await step(1500);
      expect(callbacks.length).toBe(pending);                                // paused: no further animation frames requested
    });

    it('shows the media coordinate under the pointer to two decimals', async () => {
      vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({ left: 0, top: 0, right: 1000, bottom: 500, width: 1000, height: 500, x: 0, y: 0, toJSON() {} } as DOMRect);
      await act(async () => root.render(<WorkstationCanvas {...props} mode="wipe" fullBleed/>));
      const box = calculateLetterbox(1000, 500, 1920, 1080);
      const centerX = box.offsetX + box.displayW / 2, centerY = box.offsetY + box.displayH / 2;
      await act(async () => { host.firstElementChild!.dispatchEvent(new MouseEvent('pointermove', { bubbles: true, clientX: centerX, clientY: centerY })); });
      expect(host.querySelector('.coord-readout')!.textContent).toBe('x 960.00 · y 540.00 px');
      await act(async () => { host.firstElementChild!.dispatchEvent(new MouseEvent('pointermove', { bubbles: true, clientX: 2, clientY: 2 })); });
      expect(host.querySelector('.coord-readout')).toBeNull();                // outside the picture (letterbox bar)
    });

    it('forwards right-clicks to the owner so a face under the pointer can open its menu', async () => {
      const onContextMenuPoint = vi.fn();
      await act(async () => root.render(<WorkstationCanvas {...props} mode="wipe" fullBleed onContextMenuPoint={onContextMenuPoint}/>));
      const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 300, clientY: 200 });
      await act(async () => { host.firstElementChild!.dispatchEvent(event); });
      expect(onContextMenuPoint).toHaveBeenCalledWith(300, 200);
      expect(event.defaultPrevented).toBe(true);
    });
  });
});
