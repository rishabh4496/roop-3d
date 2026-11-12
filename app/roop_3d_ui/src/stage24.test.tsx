// @vitest-environment jsdom
import { act, useState, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  BLUR_PX, DragItems, GLOW_PX, LiquidGlassCard, SHADOW_PX, TimelineAnimation, glassVars, moveItem, revealDelay, sameOrder,
} from './components/ui-layouts';
import TimelineDock from './components/timeline/TimelineDock';
import {
  SPEEDS, activeSlide, chapterSlides, inTrimRange, moveTrimHandle, nextSpeed, percentOf, placeholderFrames, rulerLabel, slideSpan, stepSeconds, type DockFrame,
} from './components/timeline/dockMath';
import { getPlaybackRate, resetPlaybackRate, setPlaybackRate } from './stores/playbackRateStore';
import BackendQueuePanel from './components/queue/BackendQueuePanel';
import { failedSlices, isReorderable, lifecycleSteps, progressPercent, queueActions, reorderPayload, type QueueJob } from './components/queue/backendQueue';
import { insideStage, loupeCrop } from './components/canvas/loupeCrop';
import { BeforeAfterLoupeHud, ViewportTelemetryPill } from './components/canvas/CanvasHuds';
import { CHAPTERS_CHANGED, saveChapters } from './components/timeline/chapters';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

beforeAll(() => {
  // Reduced motion on: springs jump and exits take 0s, so interaction tests need no timers.
  window.matchMedia = ((query: string) => ({
    matches: query.includes('prefers-reduced-motion'), media: query, onchange: null, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
  // Embla observes sizes; jsdom has no ResizeObserver / IntersectionObserver.
  class Observer { observe() {} unobserve() {} disconnect() {} takeRecords() { return []; } }
  Object.assign(globalThis, { ResizeObserver: Observer, IntersectionObserver: Observer });
});
beforeEach(() => resetPlaybackRate());

const hosts: { unmount: () => Promise<void> }[] = [];
const mount = async (node: ReactNode) => {
  const host = document.createElement('div'); document.body.append(host);
  const root = createRoot(host);
  await act(async () => root.render(node));
  const handle = { host, unmount: async () => { await act(async () => root.unmount()); host.remove(); } };
  hosts.push(handle);
  return handle;
};
afterEach(async () => { while (hosts.length) await hosts.pop()!.unmount(); document.body.innerHTML = ''; });
const click = (target: Element) => act(async () => { (target as HTMLElement).click(); });
const key = (target: Element, k: string, init: KeyboardEventInit = {}) => act(async () => { target.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...init })); });

describe('timeline dock maths', () => {
  it('places times on the track and never yields NaN%', () => {
    expect(percentOf(5, 20)).toBe(25);
    expect(percentOf(50, 20)).toBe(100);
    expect(percentOf(-3, 20)).toBe(0);
    expect(percentOf(5, 0)).toBe(0);
    expect(percentOf(Number.NaN, 20)).toBe(0);
  });

  it('measures the ruler in m:ss.f and h:mm:ss', () => {
    expect(rulerLabel(0)).toBe('0:00.0');
    expect(rulerLabel(65.46)).toBe('1:05.4');
    expect(rulerLabel(65.46, false)).toBe('1:05');
    expect(rulerLabel(3725)).toBe('1:02:05');
    expect(rulerLabel(-4)).toBe('0:00.0');
    expect(rulerLabel(Number.NaN)).toBe('0:00.0');
  });

  it('finds the slide under the playhead and each slide\'s span', () => {
    const frames = placeholderFrames(10, 5);
    expect(frames.map(frame => frame.timestamp)).toEqual([1, 3, 5, 7, 9]);
    expect(activeSlide(frames, 0)).toBe(0);
    expect(activeSlide(frames, 4.9)).toBe(1);
    expect(activeSlide(frames, 5)).toBe(2);
    expect(activeSlide(frames, 99)).toBe(4);
    expect(slideSpan(frames, 1, 10)).toEqual([3, 5]);
    expect(slideSpan(frames, 0, 10)).toEqual([0, 3]);
    expect(slideSpan(frames, 4, 10)).toEqual([9, 10]);
    expect(placeholderFrames(10, 0)).toHaveLength(1);
  });

  it('dims only the slides outside the In/Out range', () => {
    const frames = placeholderFrames(10, 5);
    const kept = frames.map((_, index) => inTrimRange(slideSpan(frames, index, 10), 2.5, 7.5));
    expect(kept).toEqual([true, true, true, true, false]);                  // slide 0 spans 0-3 s, so it overlaps a 2.5 s In point
    expect(frames.map((_, index) => inTrimRange(slideSpan(frames, index, 10), 3.5, 7.5))).toEqual([false, true, true, true, false]);
    expect(inTrimRange([0, 10], 0, null)).toBe(true);
  });

  it('assigns each chapter to the slide that contains it', () => {
    const frames = placeholderFrames(10, 5);
    const flags = chapterSlides([{ id: 'a', time: 0 }, { id: 'b', time: 4 }, { id: 'c', time: 4.5 }], frames);
    expect(flags.get(0)).toEqual(['a']);
    expect(flags.get(1)).toEqual(['b', 'c']);
  });

  it('moves a trim handle without crossing the other or leaving the clip, and stores a full-length Out as null', () => {
    const base = { start: 2, end: 8 as number | null };
    expect(moveTrimHandle('in', 5, base, 10, 25)).toEqual({ start: 5, end: 8 });
    expect(moveTrimHandle('in', 9, base, 10, 25)).toEqual({ start: 8 - 1 / 25, end: 8 });     // one frame always stays
    expect(moveTrimHandle('in', -4, base, 10, 25)).toEqual({ start: 0, end: 8 });
    expect(moveTrimHandle('out', 6, base, 10, 25)).toEqual({ start: 2, end: 6 });
    expect(moveTrimHandle('out', 1, base, 10, 25)).toEqual({ start: 2, end: 2 + 1 / 25 });
    expect(moveTrimHandle('out', 10, base, 10, 25)).toEqual({ start: 2, end: null });
    expect(moveTrimHandle('out', 99, base, 10, 25)).toEqual({ start: 2, end: null });
    expect(moveTrimHandle('in', 9.99, { start: 0, end: null }, 10, 25).start).toBeCloseTo(10 - 1 / 25, 6);
    expect(moveTrimHandle('in', Number.NaN, base, 10, 25).start).toBe(0);
  });

  it('steps a frame, or a second with Shift, and walks the 0.25x - 4x speeds', () => {
    expect(stepSeconds(25, false)).toBe(1 / 25);
    expect(stepSeconds(25, true)).toBe(1);
    expect(SPEEDS).toEqual([0.25, 0.5, 1, 2, 4]);
    expect(nextSpeed(1, 1)).toBe(2);
    expect(nextSpeed(4, 1)).toBe(4);
    expect(nextSpeed(0.25, -1)).toBe(0.25);
    expect(nextSpeed(3, 1)).toBe(2);   // an unsupported rate counts as 1x
  });

  it('shares one snapped playback speed across the app', () => {
    expect(getPlaybackRate()).toBe(1);
    expect(setPlaybackRate(3.4)).toBe(4);
    expect(setPlaybackRate(0.3)).toBe(0.25);
    expect(setPlaybackRate(Number.NaN)).toBe(0.25);
    expect(getPlaybackRate()).toBe(0.25);
  });
});

describe('TimelineDock', () => {
  const frames: DockFrame[] = Array.from({ length: 6 }, (_, index) => ({ index, timestamp: index * 2 + 1, url: `data:image/png;base64,AA${index}`, frameIndex: index * 50 }));
  const props = { target: 'clip.mp4', duration: 12, fps: 25, currentTime: 5, onSeek: () => {}, frames };

  it('shows a ruler label and a thumbnail for every frame, the active one marked', () => {
    const html = renderToStaticMarkup(<TimelineDock {...props} />);
    expect(html).toContain('aria-roledescription="carousel"');
    expect((html.match(/aria-roledescription="slide"/g) ?? []).length).toBe(6);
    expect((html.match(/<img /g) ?? []).length).toBe(6);
    expect(html).toContain('data-active-slide="2"');
    expect(html).toContain('aria-current="true"');
    for (const label of ['0:01', '0:03', '0:05', '0:07', '0:09', '0:11']) expect(html).toContain(label);
    expect(html).toContain('00:00:05:00');                                    // SMPTE playhead readout
  });

  it('offers In and Out sliders with real values, and a 0.25x - 4x speed group', () => {
    const html = renderToStaticMarkup(<TimelineDock {...props} trimStart={2} trimEnd={9} onTrimChange={() => {}} />);
    expect(html).toMatch(/role="slider"[^>]*aria-label="In point"[^>]*aria-valuenow="2"/);
    expect(html).toMatch(/role="slider"[^>]*aria-label="Out point"[^>]*aria-valuenow="9"/);
    expect(html).toContain('7:00'.replace('7:00', '0:09.0'));                 // Out label
    expect(html).toContain('role="radiogroup" aria-label="Playback speed"');
    expect((html.match(/role="radio"/g) ?? []).length).toBe(5);
    expect(html).toMatch(/role="radio" aria-checked="true"[^>]*>1x</);
    for (const speed of ['0.25x', '0.5x', '2x', '4x']) expect(html).toContain(`>${speed}<`);
    expect(renderToStaticMarkup(<TimelineDock {...props} showSpeed={false} />)).not.toContain('Playback speed');
  });

  it('is inert before a video is loaded and read-only without a trim callback', () => {
    const empty = renderToStaticMarkup(<TimelineDock {...props} target="" duration={0} frames={[]} />);
    expect(empty).toContain('Load a target video to see its frames.');
    expect(empty).toMatch(/aria-label="In point"[^>]*disabled/);
    const readOnly = renderToStaticMarkup(<TimelineDock {...props} />);
    expect(readOnly).toContain('aria-readonly="true"');
  });

  it('seeks from a thumbnail, sets In/Out at the playhead, and drives the shared speed', async () => {
    const onSeek = vi.fn(), onTrim = vi.fn();
    const view = await mount(<TimelineDock {...props} onSeek={onSeek} onTrimChange={onTrim} />);
    await click(view.host.querySelectorAll('.dock-thumb')[3]);
    expect(onSeek).toHaveBeenCalledWith(7);
    await click([...view.host.querySelectorAll('button')].find(b => b.textContent === 'Set In')!);
    expect(onTrim).toHaveBeenLastCalledWith(5, null);
    await click([...view.host.querySelectorAll('button')].find(b => b.textContent === 'Set Out')!);
    expect(onTrim).toHaveBeenLastCalledWith(0, 5);
    await click([...view.host.querySelectorAll('[role="radio"]')].find(b => b.textContent === '0.5x')!);
    expect(getPlaybackRate()).toBe(0.5);
    expect(view.host.querySelector('[role="radio"][aria-checked="true"]')!.textContent).toBe('0.5x');
  });

  it('moves a handle with the keyboard: a frame per arrow, a second with Shift, Home and End jump to the ends', async () => {
    const onTrim = vi.fn();
    const view = await mount(<TimelineDock {...props} trimStart={2} trimEnd={9} onTrimChange={onTrim} />);
    const handleIn = view.host.querySelector('[aria-label="In point"]')!, handleOut = view.host.querySelector('[aria-label="Out point"]')!;
    await key(handleIn, 'ArrowRight');
    expect(onTrim).toHaveBeenLastCalledWith(2 + 1 / 25, 9);
    await key(handleIn, 'ArrowRight', { shiftKey: true });
    expect(onTrim).toHaveBeenLastCalledWith(3, 9);
    await key(handleOut, 'ArrowLeft', { shiftKey: true });
    expect(onTrim).toHaveBeenLastCalledWith(2, 8);
    await key(handleOut, 'End');
    expect(onTrim).toHaveBeenLastCalledWith(2, null);
    await key(handleIn, 'Home');
    expect(onTrim).toHaveBeenLastCalledWith(0, 9);
  });

  it('shows chapter markers and refreshes when they change', async () => {
    saveChapters('clip.mp4', []);
    const view = await mount(<TimelineDock {...props} />);
    expect(view.host.querySelectorAll('.dock-chapter')).toHaveLength(0);
    await act(async () => { saveChapters('clip.mp4', [{ id: 'c1', time: 4, label: 'Intro' }]); });
    expect(CHAPTERS_CHANGED).toBe('roop3d:chapters');
    expect(view.host.querySelectorAll('.dock-chapter')).toHaveLength(1);
    expect(view.host.querySelector('.dock-chapter')!.getAttribute('aria-label')).toBe('Chapter Intro at 0:04.0');
    expect(view.host.querySelector('.dock-flag')).not.toBeNull();
    saveChapters('clip.mp4', []);
  });
});

describe('LiquidGlassCard', () => {
  it('maps blur, glow and shadow intensities to CSS variables and falls back on unknown values', () => {
    expect(BLUR_PX).toEqual({ sm: 8, md: 16, lg: 24, xl: 40 });
    expect(Object.keys(GLOW_PX)).toEqual(['none', 'xs', 'sm', 'md', 'lg', 'xl', '2xl']);
    expect(Object.keys(SHADOW_PX)).toEqual(['none', 'xs', 'sm', 'md', 'lg', 'xl', '2xl']);
    expect(glassVars('xl', '2xl', 'none', '12px')).toEqual({ '--ul-glass-blur': '40px', '--ul-glass-glow': '72px', '--ul-glass-shadow': '0px', '--ul-glass-radius': '12px' });
    expect(glassVars('nope', 'nope', 'nope', '32px')['--ul-glass-blur']).toBe('8px');
  });

  it('renders the content on a glass surface and adds a grip or toggle only when asked', () => {
    const plain = renderToStaticMarkup(<LiquidGlassCard blurIntensity="lg" glowIntensity="md">Hello</LiquidGlassCard>);
    expect(plain).toContain('class="ul-glass');
    expect(plain).toContain('--ul-glass-blur:24px');
    expect(plain).toContain('Hello');
    expect(plain).not.toContain('ul-glass__grip');
    expect(plain).not.toContain('ul-glass__toggle');
    const rich = renderToStaticMarkup(<LiquidGlassCard dragHandle expandable width="200px" expandedWidth="400px">Body</LiquidGlassCard>);
    expect(rich).toContain('aria-label="Drag to move"');
    expect(rich).toMatch(/ul-glass__toggle[^>]*aria-expanded="false"|aria-expanded="false"[^>]*ul-glass__toggle/);
    expect(rich).toContain('ul-glass--draggable');
  });

  it('expands from the toggle button and from a click on the card, but not from a click on a control inside it', async () => {
    const changes: boolean[] = [];
    const view = await mount(<LiquidGlassCard expandable width="100px" expandedWidth="300px" onExpandedChange={value => changes.push(value)}><button type="button" id="inside">Inside</button><p id="text">Text</p></LiquidGlassCard>);
    const card = view.host.querySelector('.ul-glass')!;
    expect(card.getAttribute('data-expanded')).toBe('false');
    await click(view.host.querySelector('.ul-glass__toggle')!);
    expect(card.getAttribute('data-expanded')).toBe('true');
    expect(view.host.querySelector('.ul-glass__toggle')!.getAttribute('aria-expanded')).toBe('true');
    await click(view.host.querySelector('#inside')!);
    expect(card.getAttribute('data-expanded')).toBe('true');          // a control inside is its own click
    await click(view.host.querySelector('#text')!);
    expect(card.getAttribute('data-expanded')).toBe('false');
    expect(changes).toEqual([true, false]);
  });
});

describe('TimelineAnimation', () => {
  it('staggers by position and shows everything at once without motion', () => {
    expect(revealDelay(0)).toBe(0);
    expect(revealDelay(3)).toBeCloseTo(0.3, 6);
    expect(revealDelay(3, true)).toBeCloseTo(0.75, 6);
    expect(revealDelay(-2)).toBe(0);
    expect(revealDelay(Number.NaN)).toBe(0);
    const html = renderToStaticMarkup(<TimelineAnimation as="li" className="x" animationNum={2}>Item</TimelineAnimation>);
    expect(html).toMatch(/^<li class="x"/);
    expect(html).toContain('Item');
  });
});

describe('DragItems', () => {
  it('moves an item, clamps at the ends and is a no-op in place', () => {
    expect(moveItem(['a', 'b', 'c'], 0, 2)).toEqual(['b', 'c', 'a']);
    expect(moveItem(['a', 'b', 'c'], 2, 0)).toEqual(['c', 'a', 'b']);
    expect(moveItem(['a', 'b', 'c'], 1, 99)).toEqual(['a', 'c', 'b']);
    expect(moveItem(['a', 'b', 'c'], 1, -5)).toEqual(['b', 'a', 'c']);
    expect(moveItem(['a', 'b', 'c'], 1, 1)).toEqual(['a', 'b', 'c']);
    expect(moveItem(['a', 'b'], 7, 0)).toEqual(['a', 'b']);
    expect(sameOrder(['a', 'b'], ['a', 'b'], x => x)).toBe(true);
    expect(sameOrder(['a', 'b'], ['b', 'a'], x => x)).toBe(false);
  });

  function Harness({ onCommit }: { onCommit: (items: string[]) => void }) {
    const [items, setItems] = useState(['alpha', 'beta', 'gamma']);
    return <DragItems items={items} getKey={item => item} label="Letters" onReorder={setItems} onCommit={onCommit}
      renderItem={(item, { controls, index }) => <div className="row">{controls}<span className="name">{index + 1}. {item}</span></div>} />;
  }
  const names = (host: HTMLElement) => [...host.querySelectorAll('.name')].map(node => node.textContent);

  it('reorders with the move buttons, commits once per move and announces the result', async () => {
    const onCommit = vi.fn();
    const view = await mount(<Harness onCommit={onCommit} />);
    expect(view.host.querySelector('ol')!.getAttribute('aria-label')).toBe('Letters');
    expect(names(view.host)).toEqual(['1. alpha', '2. beta', '3. gamma']);
    expect(view.host.querySelector('[aria-label="Move alpha up"]')!.hasAttribute('disabled')).toBe(true);
    expect(view.host.querySelector('[aria-label="Move gamma down"]')!.hasAttribute('disabled')).toBe(true);
    await click(view.host.querySelector('[aria-label="Move alpha down"]')!);
    expect(names(view.host)).toEqual(['1. beta', '2. alpha', '3. gamma']);
    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(onCommit).toHaveBeenLastCalledWith(['beta', 'alpha', 'gamma']);
    expect(view.host.querySelector('[role="status"]')!.textContent).toBe('alpha moved to position 2 of 3.');
  });

  it('reorders from the keyboard with Alt+arrows on the grip and ignores plain arrows', async () => {
    const onCommit = vi.fn();
    const view = await mount(<Harness onCommit={onCommit} />);
    const grip = (label: string) => view.host.querySelector(`.ul-drag__grip[aria-label^="Reorder ${label}"]`)!;
    await key(grip('gamma'), 'ArrowUp');
    expect(onCommit).not.toHaveBeenCalled();
    await key(grip('gamma'), 'ArrowUp', { altKey: true });
    expect(names(view.host)).toEqual(['1. alpha', '2. gamma', '3. beta']);
    await key(grip('gamma'), 'ArrowUp', { altKey: true });
    await key(grip('gamma'), 'ArrowUp', { altKey: true });     // already first: no further commit
    expect(names(view.host)).toEqual(['1. gamma', '2. alpha', '3. beta']);
    expect(onCommit).toHaveBeenCalledTimes(2);
  });

  it('follows a changed list from the parent and can be disabled', async () => {
    const view = await mount(<DragItems items={['x', 'y']} getKey={item => item} label="L" disabled renderItem={(item, { controls }) => <div className="row">{controls}<span className="name">{item}</span></div>} />);
    expect(view.host.querySelectorAll('button:not([disabled])')).toHaveLength(0);
    expect(names(view.host)).toEqual(['x', 'y']);
  });
});

describe('persistent queue panel', () => {
  const job = (overrides: Partial<QueueJob> & { job_id: string }): QueueJob => ({
    name: overrides.job_id, state: 'QUEUED', lifecycle_state: 'queued', queue_order: 0, slices: [], revision: 1, output: null, error_message: null,
    progress: { frame: 0, total_frames: 100, fps: 0, stage: 'QUEUED' }, ...overrides,
  });
  const jobs = [
    job({ job_id: 'run', name: 'Running clip', lifecycle_state: 'rendering', progress: { frame: 40, total_frames: 100, fps: 20, stage: 'RENDERING' } }),
    job({ job_id: 'q1', name: 'Queued one' }),
    job({ job_id: 'trt', name: 'Engine build', lifecycle_state: 'compiling_trt' }),
    job({ job_id: 'q2', name: 'Queued two', lifecycle_state: 'paused' }),
    job({ job_id: 'bad', name: 'Broken', lifecycle_state: 'failed', error_message: 'ffmpeg exited 1', slices: [{ slice_id: 's1', slice_index: 1, status: 'COMPLETED' }, { slice_id: 's2', slice_index: 2, status: 'FAILED', error: 'oom' }] }),
  ];

  it('offers only the controls that make sense for a state', () => {
    const actions = (value: QueueJob) => queueActions(value);
    expect(actions(jobs[0])).toEqual(['pause', 'cancel', 'duplicate']);
    expect(actions(jobs[3])).toEqual(['resume', 'cancel', 'duplicate']);
    expect(actions(jobs[4])).toEqual(['duplicate']);
    const joinable = job({ job_id: 'j', lifecycle_state: 'muxing_audio', slices: [{ slice_id: 'a', slice_index: 1, status: 'COMPLETED' }, { slice_id: 'b', slice_index: 2, status: 'COMPLETED' }] });
    expect(actions(joinable)).toContain('join');
    expect(actions({ ...joinable, output: { path: 'x' } })).not.toContain('join');
    expect(failedSlices(jobs[4]).map(slice => slice.slice_id)).toEqual(['s2']);
  });

  it('lets only queued and paused jobs move, and a drop never relocates a running one', () => {
    expect(jobs.map(isReorderable)).toEqual([false, true, false, true, false]);
    // Dragging "Queued two" above "Queued one": the movable jobs swap slots, everything else keeps its place.
    const after = [jobs[0], jobs[3], jobs[2], jobs[1], jobs[4]];
    expect(reorderPayload(jobs, after)).toEqual(['run', 'q2', 'trt', 'q1', 'bad']);
    expect(reorderPayload(jobs, jobs)).toEqual(jobs.map(item => item.job_id));
  });

  it('reports progress and the lifecycle strip position', () => {
    expect(progressPercent(jobs[0])).toBe(40);
    expect(progressPercent(job({ job_id: 'c', lifecycle_state: 'completed' }))).toBe(100);
    expect(progressPercent(job({ job_id: 'n', progress: { frame: 5, total_frames: null, fps: 0, stage: 'x' } }))).toBe(0);
    const steps = lifecycleSteps('tracking_3d');
    expect(steps.map(step => step.status)).toEqual(['done', 'done', 'done', 'current', 'upcoming', 'upcoming', 'upcoming']);
    expect(lifecycleSteps('failed').every(step => step.status === 'upcoming')).toBe(true);
  });

  it('renders every job with its lifecycle chip, locks running jobs and offers segment retry', () => {
    const html = renderToStaticMarkup(<BackendQueuePanel jobs={jobs} />);
    expect(html).toContain('aria-label="Persistent job queue"');
    for (const label of ['Rendering', 'Compiling TensorRT', 'Queued', 'Paused', 'Failed']) expect(html).toContain(`>${label}<`);
    expect(html).toContain('Retry segment 2');
    expect((html.match(/ul-drag__controls--locked/g) ?? []).length).toBe(3);   // rendering, compiling, failed
    expect(html).toContain('ffmpeg exited 1');
    expect(html).toContain('1/2 segments');
    expect(renderToStaticMarkup(<BackendQueuePanel jobs={[]} />)).toContain('No batch or queued jobs yet');
  });
});

describe('canvas HUDs', () => {
  it('crops the lens inside the image and ignores a stage with no area', () => {
    const stage = { left: 10, top: 20, width: 400, height: 200 };
    const natural = { width: 1600, height: 800 };
    const mid = loupeCrop({ x: 210, y: 120 }, stage, natural, 100, 4)!;
    expect(mid.sw).toBeCloseTo(100, 6);          // (100 / 4) * 4 px of source per screen px
    expect(mid.sx + mid.sw / 2).toBeCloseTo(800, 6);
    expect(mid.sy + mid.sh / 2).toBeCloseTo(400, 6);
    const corner = loupeCrop({ x: 10, y: 20 }, stage, natural, 100, 4)!;
    expect([corner.sx, corner.sy]).toEqual([0, 0]);
    const far = loupeCrop({ x: 9999, y: 9999 }, stage, natural, 100, 4)!;
    expect(far.sx + far.sw).toBeCloseTo(1600, 6);
    expect(far.sy + far.sh).toBeCloseTo(800, 6);
    expect(loupeCrop({ x: 0, y: 0 }, { ...stage, width: 0 }, natural, 100, 4)).toBeNull();
    expect(loupeCrop({ x: 0, y: 0 }, stage, natural, 100, 0)).toBeNull();
    expect(insideStage({ x: 50, y: 50 }, stage)).toBe(true);
    expect(insideStage({ x: 5, y: 50 }, stage)).toBe(false);
  });

  it('shows yaw / pitch / roll of the selected face, or dashes when there is none', () => {
    const face = { id: 'f1', pose3d: { yaw: 12.34, pitch: -4.06, roll: 0.5 } };
    const html = renderToStaticMarkup(<ViewportTelemetryPill faces={[face]} activeFaceId="f1" mode="loupe" frame={42} zoom={4} />);
    expect(html).toContain('aria-label="3D viewport telemetry"');
    expect(html).toContain('+12.3°');
    expect(html).toContain('−4.1°');
    expect(html).toContain('1 FACE');
    expect(html).toContain('F42');
    expect(html).toContain('loupe 4x');
    const none = renderToStaticMarkup(<ViewportTelemetryPill faces={[]} mode="wipe" frame={0} />);
    expect(none).toContain('—');
    expect(none).toContain('0 FACES');
  });

  it('renders a before and an after lens', () => {
    const html = renderToStaticMarkup(<BeforeAfterLoupeHud imageA={null} imageB={null} stage={{ left: 0, top: 0, width: 100, height: 100 }} pointer={{ x: 5, y: 5, active: true }} zoom={4} />);
    expect(html).toContain('aria-label="Before and after loupe at 4x"');
    expect(html).toContain('BEFORE');
    expect(html).toContain('AFTER · 4x');
  });
});
