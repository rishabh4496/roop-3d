// @vitest-environment jsdom
/** Stage 11 / 26 units: atomic stores, the virtualized list, the timeline's playhead isolation, hotkey migration, the floating window and Document PiP. */
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createAtomicStore, shallowEqual, useStoreSelector } from './stores/atomicStore';
import { playheadStore, setPlayhead, usePlayhead } from './stores/playheadStore';
import { publishHardware, resetTelemetry, telemetryStore, useTelemetry } from './stores/telemetryStore';
import VirtualList, { columnsFor, windowRange } from './components/common/VirtualList';
import TimelineWorkstation from './components/timeline/TimelineWorkstation';
import { DEFAULT_KEYS, HotkeyEditor, HotkeyProvider, KEY_GROUPS, KEY_LABELS, keyConflict, validKeys } from './components/controls/Hotkeys';
import { DEFAULT_SIZE, clampGeometry, defaultGeometry, pipStore, useFloatingPreview, validStored } from './components/workstation/FloatingPreview';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let host: HTMLDivElement, root: Root;
const probe: Record<string, number> = {};
beforeEach(() => {
  localStorage.clear(); for (const key of Object.keys(probe)) delete probe[key];
  (globalThis as any).__R3D_RENDER_PROBE__ = probe;
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  resetTelemetry(); playheadStore.reset(); pipStore.set({ mode: 'closed', geometry: null });
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); delete (globalThis as any).__R3D_RENDER_PROBE__; vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('atomic stores', () => {
  it('notifies only on a real change and supports transient (render-free) selector subscriptions', () => {
    const store = createAtomicStore({ a: 1, b: { n: 1 } }), all = vi.fn(), onlyA = vi.fn();
    store.subscribe(all); store.subscribeSelector(state => state.a, onlyA);
    store.set({ a: 1 }); expect(all).not.toHaveBeenCalled();
    store.set({ b: { n: 2 } }); expect(all).toHaveBeenCalledTimes(1); expect(onlyA).not.toHaveBeenCalled();
    store.set(state => ({ a: state.a + 1 })); expect(onlyA).toHaveBeenCalledWith(2, 1);
    store.reset(); expect(store.get()).toEqual({ a: 1, b: { n: 1 } });
  });
  it('re-renders a subscriber only when its selected slice changes', async () => {
    const store = createAtomicStore({ fps: 0, vram: 0 }); let renders = 0;
    function Fps() { renders++; return <>{useStoreSelector(store, state => state.fps)}</>; }
    await act(async () => root.render(<Fps/>)); expect(renders).toBe(1);
    for (let i = 0; i < 90; i++) await act(async () => store.set({ vram: i + 1 }));          // 90 ticks of an unrelated field
    expect(renders).toBe(1);
    await act(async () => store.set({ fps: 30 })); expect(renders).toBe(2); expect(host.textContent).toBe('30');
  });
  it('shallowEqual compares object slices by value', () => { expect(shallowEqual({ x: 1, y: 2 }, { x: 1, y: 2 })).toBe(true); expect(shallowEqual({ x: 1 }, { x: 1, y: 2 })).toBe(false); expect(shallowEqual(null, null)).toBe(true); });
  it('telemetry store: a leaf selecting one GPU field ignores the other fields streaming past it', async () => {
    let renders = 0;
    function Name() { renders++; return <>{useTelemetry(state => state.hardware?.gpus[0]?.name ?? 'none')}</>; }
    await act(async () => root.render(<Name/>));
    for (let i = 0; i < 30; i++) await act(async () => publishHardware({ gpus: [{ name: 'RTX 4090', total_vram_mib: 24576, free_vram_mib: 20000 - i }], logical_cpu_count: 16 }));
    expect(host.textContent).toBe('RTX 4090'); expect(renders).toBe(2);                         // initial + the one name change
    expect(telemetryStore.get().hardware?.gpus[0].free_vram_mib).toBe(19971);                    // the store itself is always fresh
  });
  it('playhead hook: frame-quantised selectors re-render once per frame, not once per tick', async () => {
    let renders = 0;
    function Frame() { renders++; return <>{usePlayhead(time => Math.floor(time * 30))}</>; }
    await act(async () => root.render(<Frame/>));
    for (let i = 0; i < 100; i++) await act(async () => setPlayhead(i * 0.001));                // 100 sub-frame ticks inside the first 100 ms
    expect(host.textContent).toBe('2'); expect(renders).toBeLessThanOrEqual(4);
    setPlayhead(Number.NaN); expect(playheadStore.get().time).toBeCloseTo(0.099);               // non-finite times are ignored
  });
});

describe('VirtualList', () => {
  const rect = (top: number, height: number) => ({ top, bottom: top + height, left: 0, right: 800, width: 800, height, x: 0, y: top, toJSON() {} }) as DOMRect;
  it('windowRange mounts only the visible band plus overscan and reports the full extent', () => {
    const items = { rowCount: 10_000, heights: new Map<number, number>(), estimate: 50, overscan: 2 };
    expect(windowRange({ ...items, viewTop: 0, viewBottom: 500 })).toEqual({ first: 0, last: 12, offset: 0, total: 500_000 });
    const mid = windowRange({ ...items, viewTop: 250_000, viewBottom: 250_500 });
    expect(mid.first).toBe(4998); expect(mid.last).toBe(5012); expect(mid.offset).toBe(4998 * 50); expect(mid.total).toBe(500_000);
    expect(windowRange({ ...items, viewTop: 10_000_000, viewBottom: 10_000_500 }).last).toBe(9999);   // scrolled past the end clamps
    expect(windowRange({ ...items, rowCount: 0, viewTop: 0, viewBottom: 500 })).toEqual({ first: 0, last: -1, offset: 0, total: 0 });
  });
  it('measured row heights replace the estimate in offsets and total', () => {
    const heights = new Map([[0, 100], [1, 100]]);
    const result = windowRange({ rowCount: 4, heights, estimate: 50, viewTop: 150, viewBottom: 260, overscan: 0 });
    expect(result.first).toBe(1); expect(result.total).toBe(300); expect(result.offset).toBe(100);
  });
  it('derives responsive column counts', () => { expect(columnsFor(1000, 230, 16)).toBe(4); expect(columnsFor(100, 230, 16)).toBe(1); });

  it('keeps the DOM footprint constant for 20,000 face crops and follows the scroll position', async () => {
    const items = Array.from({ length: 20_000 }, (_, id) => ({ id }));
    let scrollTop = 0;
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      if (this.classList.contains('virtual-spacer')) return rect(-scrollTop, 20_000 * 60);
      if (this.classList.contains('virtual-list')) return rect(0, 300);
      return rect(0, 0);
    });
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { callback(0); return 1; });
    await act(async () => root.render(<VirtualList items={items} itemKey={item => item.id} estimateHeight={60} gap={0} maxHeight={300} aria-label="Faces" renderItem={item => <span className="crop">face {item.id}</span>}/>));
    const list = () => host.querySelector<HTMLElement>('.virtual-list')!;
    expect(list().dataset.virtualCount).toBe('20000');
    const mounted = () => host.querySelectorAll('.crop').length;
    expect(mounted()).toBeGreaterThan(4); expect(mounted()).toBeLessThan(20);
    expect(host.querySelector('.crop')!.textContent).toBe('face 0');
    scrollTop = 30_000;                                                                       // 500 rows down
    await act(async () => { list().dispatchEvent(new Event('scroll')); });
    expect(mounted()).toBeLessThan(20);
    const texts = [...host.querySelectorAll('.crop')].map(node => node.textContent);
    expect(texts).toContain('face 500'); expect(texts).not.toContain('face 0');
    const item = host.querySelector('[role="listitem"]')!; expect(item.getAttribute('aria-setsize')).toBe('20000'); expect(Number(item.getAttribute('aria-posinset'))).toBeGreaterThan(400);
  });
  it('lays a thumbnail grid out in columns and renders an empty slot for no items', async () => {
    await act(async () => root.render(<VirtualList items={Array.from({ length: 12 }, (_, id) => id)} itemKey={id => id} estimateHeight={80} columns={3} renderItem={id => <b>{id}</b>}/>));
    const rows = host.querySelectorAll('[data-row]'); expect(rows).toHaveLength(4); expect((rows[0] as HTMLElement).style.gridTemplateColumns).toContain('repeat(3');
    await act(async () => root.render(<VirtualList items={[]} itemKey={() => 0} estimateHeight={10} renderItem={() => null} empty={<p>Nothing yet</p>}/>));
    expect(host.textContent).toBe('Nothing yet');
  });
});

describe('timeline playhead isolation', () => {
  const timeline = (props: Record<string, unknown> = {}) => <TimelineWorkstation target="clip.mp4" duration={10} fps={30} currentTime={1} playing {...props}/>;
  it('a playback tick re-renders only the needle, timecode and frame counter, never the timeline body', async () => {
    await act(async () => root.render(timeline()));
    const body = probe.TimelineWorkstation;
    const needle = () => host.querySelector('.needle-head-tag')!.textContent, timecode = () => host.querySelector('.current-tc')!.textContent;
    for (let i = 0; i < 60; i++) await act(async () => setPlayhead(1 + i / 30));
    expect(probe.TimelineWorkstation).toBe(body);                                              // 60 ticks, 0 body renders
    expect(timecode()).toBe('00:00:02:29'); expect(needle()).toBe('00:02:29');
    expect(host.querySelector('.frame-counter')!.textContent).toBe('[F: 89 / 300]');
    expect((host.querySelector('.timeline-playhead-needle') as HTMLElement).style.left).toBe(`${(1 + 59 / 30) / 10 * 100}%`);
    expect(Number(host.querySelector<HTMLInputElement>('[aria-label="Timeline scrubber"]')!.value)).toBeCloseTo(1 + 59 / 30, 3);
  });
  it('paused, it shows the committed time and ignores stray store writes', async () => {
    await act(async () => root.render(timeline({ playing: false, currentTime: 2 })));
    await act(async () => setPlayhead(7));
    expect(host.querySelector('.current-tc')!.textContent).toBe('00:00:02:00');
  });
  it('wraps the range loop on a store tick without a React render', async () => {
    const onSeek = vi.fn();
    await act(async () => root.render(timeline({ onSeek, trimStart: 1, trimEnd: 3, currentTime: 1 })));
    await act(async () => (host.querySelector('.timeline-loop-btn') as HTMLElement).click());
    const body = probe.TimelineWorkstation; onSeek.mockClear();
    await act(async () => setPlayhead(2.5)); expect(onSeek).not.toHaveBeenCalled();
    await act(async () => setPlayhead(3)); expect(onSeek).toHaveBeenCalledWith(1);
    expect(probe.TimelineWorkstation).toBe(body);
  });
});

describe('hotkeys: new actions, migration and conflicts', () => {
  it('every action has a label, exactly one group, and a unique non-empty default chord', () => {
    const actions = Object.keys(DEFAULT_KEYS), grouped = KEY_GROUPS.flatMap(group => group.actions);
    expect([...grouped].sort()).toEqual([...actions].sort());
    expect(actions.every(action => KEY_LABELS[action])).toBe(true);
    const chords = Object.values(DEFAULT_KEYS); expect(new Set(chords).size).toBe(chords.length); expect(chords.every(Boolean)).toBe(true);
    for (const [action, chord] of Object.entries({ previous: 'ArrowLeft', next: 'ArrowRight', markIn: 'I', markOut: 'O', toggleWipe: 'W', toggleDiff: 'D', extractFace: 'E', buffer1: '1', buffer6: '6', snapshotA: 'Shift+A', snapshotD: 'Shift+D' })) expect(DEFAULT_KEYS[action]).toBe(chord);
  });
  it('keeps a stored v1 map’s remaps when new actions are added, and leaves a new action unassigned if its default is taken', () => {
    const { toggleWipe, toggleDiff, extractFace, buffer1, ...legacy } = DEFAULT_KEYS; void toggleWipe; void toggleDiff; void extractFace; void buffer1;
    const migrated = validKeys({ ...legacy, play: 'K', solo: 'W' });                           // the user already put Solo on W
    expect(migrated.play).toBe('K'); expect(migrated.solo).toBe('W');
    expect(migrated.toggleWipe).toBe('');                                                      // W is taken: unassigned, not stolen
    expect(migrated.toggleDiff).toBe('D'); expect(migrated.extractFace).toBe('E'); expect(migrated.buffer1).toBe('1');
    expect(validKeys({ ...legacy })).toEqual(DEFAULT_KEYS);
    expect(validKeys({ ...DEFAULT_KEYS, play: 'S' })).toEqual(DEFAULT_KEYS);                   // a present duplicate still resets everything
    expect(validKeys({ ...DEFAULT_KEYS, play: 'ctrl+!!' })).toEqual(DEFAULT_KEYS);
    expect(validKeys('garbage')).toEqual(DEFAULT_KEYS);
    expect(keyConflict({ a: 'X', b: '' }, 'b', '')).toBeUndefined(); expect(keyConflict({ a: 'X', b: 'Y' }, 'b', 'X')).toBe('a');
  });
  it('the editor lists every group, unassigns with Backspace, persists, and adopts remaps made in another tab', async () => {
    await act(async () => root.render(<HotkeyProvider><HotkeyEditor/></HotkeyProvider>));
    expect([...host.querySelectorAll('.hotkey-group legend')].map(node => node.textContent)).toEqual(KEY_GROUPS.map(group => group.title));
    const field = () => host.querySelector<HTMLInputElement>('[aria-label="Shortcut for Extract face from frame"]')!;
    expect(field().value).toBe('E');
    await act(async () => { field().dispatchEvent(new KeyboardEvent('keydown', { key: 'Backspace', bubbles: true, cancelable: true })); });
    expect(field().value).toBe(''); expect(JSON.parse(localStorage.getItem('roop3d-hotkeys-v1')!).extractFace).toBe('');
    const other = { ...DEFAULT_KEYS, extractFace: 'F8' };
    localStorage.setItem('roop3d-hotkeys-v1', JSON.stringify(other));
    await act(async () => { window.dispatchEvent(new StorageEvent('storage', { key: 'roop3d-hotkeys-v1' })); });
    expect(field().value).toBe('F8');
  });
});

describe('floating preview window', () => {
  it('clamps geometry on screen and validates stored state (only the in-app window is restorable)', () => {
    expect(clampGeometry({ x: -50, y: 9999, width: 5000, height: 10 }, 1280, 800)).toEqual({ x: 0, y: 650, width: 1280, height: 150 });
    expect(defaultGeometry(1280, 800)).toEqual({ ...DEFAULT_SIZE, x: 1280 - DEFAULT_SIZE.width - 24, y: 800 - DEFAULT_SIZE.height - 24 });
    expect(validStored({ mode: 'pip', geometry: { x: 1, y: 2, width: 300, height: 200 } })).toEqual({ mode: 'closed', geometry: { x: 1, y: 2, width: 300, height: 200 } });
    expect(validStored({ mode: 'window', geometry: { x: 'a' } })).toEqual({ mode: 'window', geometry: null });
    expect(validStored(null)).toEqual({ mode: 'closed', geometry: null });
  });

  function Host({ label = 'frame' }: { label?: string }) {
    const floating = useFloatingPreview(<span data-testid="content">{label}</span>);
    return <><button onClick={floating.toggle}>toggle</button><button onClick={() => void floating.openPip()}>pip</button><output data-mode>{floating.mode}</output>{floating.portal}</>;
  }
  it('toggles an in-app window portalled to <body>, moves by pointer drag, persists geometry, and clamps to the viewport', async () => {
    vi.stubGlobal('innerWidth', 1000); vi.stubGlobal('innerHeight', 700);
    await act(async () => root.render(<Host/>));
    await act(async () => (host.querySelectorAll('button')[0] as HTMLElement).click());
    const win = () => document.querySelector<HTMLElement>('[data-testid="floating-window"]')!;
    expect(win()).toBeTruthy(); expect(win().parentElement).toBe(document.body); expect(win().textContent).toContain('frame');
    const start = { left: parseFloat(win().style.left), top: parseFloat(win().style.top) };
    const bar = win().querySelector<HTMLElement>('.floating-window-bar')!;
    const pointer = (type: string, x: number, y: number) => act(async () => { bar.dispatchEvent(new MouseEvent(type, { bubbles: true, clientX: x, clientY: y, button: 0 })); });
    await pointer('pointerdown', 100, 100); await pointer('pointermove', 60, 70); await pointer('pointerup', 60, 70);
    expect(parseFloat(win().style.left)).toBe(start.left - 40); expect(parseFloat(win().style.top)).toBe(start.top - 30);
    expect(JSON.parse(localStorage.getItem('roop3d-pip-v1')!).geometry.x).toBe(start.left - 40);
    await pointer('pointerdown', 100, 100); await pointer('pointermove', -5000, -5000); await pointer('pointerup', -5000, -5000);
    expect([parseFloat(win().style.left), parseFloat(win().style.top)]).toEqual([0, 0]);      // cannot be dragged off screen
    await act(async () => bar.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', shiftKey: true, bubbles: true }))); // Shift+arrow resizes
    expect(parseFloat(win().style.width)).toBe(DEFAULT_SIZE.width + 16);
    await act(async () => (host.querySelector('[aria-label="Close floating preview"]') ?? document.querySelector('[aria-label="Close floating preview"]') as HTMLElement).dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(win()).toBeNull(); expect(JSON.parse(localStorage.getItem('roop3d-pip-v1')!).mode).toBe('closed');
  });
  it('restores an open window from storage (state shared by the whole shell)', async () => {
    pipStore.set({ mode: 'window', geometry: { x: 40, y: 50, width: 300, height: 200 } });
    await act(async () => root.render(<Host/>));
    const win = document.querySelector<HTMLElement>('[data-testid="floating-window"]')!;
    expect([win.style.left, win.style.top, win.style.width, win.style.height]).toEqual(['40px', '50px', '300px', '200px']);
  });

  it('opens the browser Document Picture-in-Picture window with live content, forwards its keys, and closes with the host', async () => {
    const childDocument = document.implementation.createHTMLDocument(''), listeners: Record<string, (event: any) => void> = {}, close = vi.fn();
    const child = { document: childDocument, addEventListener: (type: string, fn: (event: any) => void) => { listeners[type] = fn; }, close, closed: false, focus: vi.fn() };
    const requestWindow = vi.fn(async () => child as unknown as Window);
    vi.stubGlobal('documentPictureInPicture', { requestWindow });
    const seen: string[] = []; window.addEventListener('keydown', event => seen.push(event.key));
    const view = (label: string) => <Host label={label}/>;
    await act(async () => root.render(view('A')));
    await act(async () => (host.querySelectorAll('button')[1] as HTMLElement).click());
    expect(requestWindow).toHaveBeenCalledWith(expect.objectContaining({ width: DEFAULT_SIZE.width, height: DEFAULT_SIZE.height }));
    expect(childDocument.body.textContent).toContain('A'); expect(host.querySelector('[data-mode]')!.textContent).toBe('pip');
    expect(document.querySelector('[data-testid="floating-window"]')).toBeNull();
    await act(async () => root.render(view('B'))); expect(childDocument.body.textContent).toContain('B');   // same live React state
    listeners.keydown({ key: 'ArrowRight', code: 'ArrowRight', target: childDocument.body, preventDefault() {} });
    expect(seen).toContain('ArrowRight');                                                      // frame-step works while the PiP window has focus
    await act(async () => { listeners.pagehide({}); });
    expect(host.querySelector('[data-mode]')!.textContent).toBe('closed');
    await act(async () => root.unmount()); root = createRoot(host);
  });
  it('falls back to the in-app window when Document PiP is unsupported or rejected', async () => {
    await act(async () => root.render(<Host/>));
    await act(async () => (host.querySelectorAll('button')[1] as HTMLElement).click());        // jsdom: no documentPictureInPicture
    expect(document.querySelector('[data-testid="floating-window"]')).toBeTruthy();
    await act(async () => (host.querySelectorAll('button')[0] as HTMLElement).click());
    vi.stubGlobal('documentPictureInPicture', { requestWindow: async () => { throw new Error('needs a user gesture'); } });
    await act(async () => (host.querySelectorAll('button')[1] as HTMLElement).click());
    expect(document.querySelector('[data-testid="floating-window"]')).toBeTruthy();
  });
});
