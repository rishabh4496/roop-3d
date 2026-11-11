// @vitest-environment jsdom
import { act, useRef, useState, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { clamp, cn, decimalsOf, useMediaQuery } from './lib/utils';
import {
  AnimatedBeam, AnimatedTabs, pulseKeyframes, ClipPathSplitReveal, ConvergingBeams, MotionDrawer, MotionNumber, MultiAccordion, PrecisionRangeSlider, SpotlightCard,
  clipLeftOf, computeBeamPath, fillSegment, formatNumber, loupeOffset, proximityOpacity, snapToStep, type Sides,
} from './components/ui-layouts';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

// matchMedia stub: reduced motion is always on (springs jump, exits take 0s), other queries are driven by `matching`.
const matching = new Set<string>();
const listeners = new Map<string, Set<() => void>>();
beforeAll(() => {
  window.matchMedia = ((query: string) => ({
    get matches() { return query.includes('prefers-reduced-motion') || matching.has(query); },
    media: query, onchange: null,
    addEventListener: (_: string, fn: () => void) => { (listeners.get(query) ?? listeners.set(query, new Set()).get(query)!).add(fn); },
    removeEventListener: (_: string, fn: () => void) => { listeners.get(query)?.delete(fn); },
    addListener() {}, removeListener() {}, dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
});

const hosts: { unmount: () => Promise<void> }[] = [];
const mount = async (node: ReactNode) => {
  const host = document.createElement('div'); document.body.append(host);
  const root = createRoot(host);
  await act(async () => root.render(node));
  const handle = { host, rerender: (next: ReactNode) => act(async () => root.render(next)), unmount: async () => { await act(async () => root.unmount()); host.remove(); } };
  hosts.push(handle);
  return handle;
};
afterEach(async () => { while (hosts.length) await hosts.pop()!.unmount(); document.body.innerHTML = ''; matching.clear(); });

const flush = (ms = 30) => act(async () => { await new Promise(resolve => setTimeout(resolve, ms)); });
const key = (target: Element, k: string, init: KeyboardEventInit = {}) => act(async () => { target.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...init })); });
const pointer = (target: Element, type: string, init: MouseEventInit = {}) => act(async () => { target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, button: 0, ...init })); });
const click = (target: Element) => act(async () => { (target as HTMLElement).click(); });
const rectOf = (left: number, top: number, width: number, height: number) => ({ left, top, width, height, right: left + width, bottom: top + height, x: left, y: top, toJSON() {} }) as DOMRect;

describe('foundation utilities', () => {
  it('cn joins strings, arrays and condition maps and drops falsy values', () => {
    expect(cn('a', false, null, undefined, '', ['b', ['c']], { d: true, e: 0, f: 'yes' }, 0)).toBe('a b c d f 0');
    expect(cn()).toBe('');
  });
  it('clamp and decimalsOf', () => {
    expect(clamp(5, 0, 3)).toBe(3); expect(clamp(-1, 0, 3)).toBe(0);
    expect([decimalsOf(1), decimalsOf(0.25), decimalsOf(0.1), decimalsOf(1e-7), decimalsOf(0)]).toEqual([0, 2, 1, 7, 0]);
  });
  it('useMediaQuery is correct on first render and follows changes', async () => {
    matching.add('(min-width: 900px)');
    const seen: boolean[] = [];
    const Probe = () => { const v = useMediaQuery('(min-width: 900px)'); seen.push(v); return <span data-v={String(v)} />; };
    const view = await mount(<Probe />);
    expect(seen[0]).toBe(true); // no false→true flash
    matching.delete('(min-width: 900px)');
    await act(async () => { listeners.get('(min-width: 900px)')?.forEach(fn => fn()); });
    expect(view.host.querySelector('span')!.dataset.v).toBe('false');
    await view.unmount();
    expect(listeners.get('(min-width: 900px)')?.size).toBe(0); // unsubscribed
  });
  it('Obsidian tokens are defined and no stylesheet reaches off-device', async () => {
    const fs = await import('node:fs'), path = await import('node:path');
    const css = fs.readFileSync(path.resolve(__dirname, 'index.css'), 'utf8');
    for (const token of ['--bg-obsidian: #070709', '--bg-panel: #0d0e12', 'rgba(255, 255, 255, 0.08)', '--accent-indigo', '--accent-cyan', 'tabular-nums']) expect(css).toContain(token);
    for (const file of ['index.css', 'components/ui-layouts/ui-layouts.css']) {
      const text = fs.readFileSync(path.resolve(__dirname, file), 'utf8');
      expect(text).not.toMatch(/https?:\/\//); expect(text).not.toMatch(/@import|url\(/);
    }
    const dir = path.resolve(__dirname, 'components/ui-layouts');
    for (const file of fs.readdirSync(dir).filter(name => /\.(tsx?|css)$/.test(name))) expect(fs.readFileSync(path.join(dir, file), 'utf8'), file).not.toMatch(/https?:\/\//);
  });
});

describe('SpotlightCard', () => {
  it('proximityOpacity is 1 inside, fades with distance, 0 beyond', () => {
    const rect = { left: 100, top: 100, right: 200, bottom: 200 };
    expect(proximityOpacity(150, 150, rect, 100)).toBe(1);
    expect(proximityOpacity(250, 150, rect, 100)).toBeCloseTo(0.5);
    expect(proximityOpacity(400, 150, rect, 100)).toBe(0);
    expect(proximityOpacity(210, 150, rect, 0)).toBe(0);
  });
  it('writes --mouse-x/--mouse-y on mouse move without re-rendering, and hides the glow on leave', async () => {
    let renders = 0;
    const Card = () => { renders++; return <SpotlightCard data-testid="card">hello</SpotlightCard>; };
    await mount(<Card />);
    const card = document.querySelector<HTMLElement>('[data-testid=card]')!;
    card.getBoundingClientRect = () => rectOf(10, 20, 300, 200);
    const before = renders;
    await pointer(card, 'mousemove', { clientX: 110, clientY: 70 });
    expect(card.style.getPropertyValue('--mouse-x')).toBe('100px');
    expect(card.style.getPropertyValue('--mouse-y')).toBe('50px');
    expect(card.style.getPropertyValue('--spot-opacity')).toBe('1.000');
    expect(renders).toBe(before);
    await act(async () => { card.dispatchEvent(new MouseEvent('mouseout', { bubbles: true, relatedTarget: document.body })); }); // React derives mouseleave from mouseout
    expect(card.style.getPropertyValue('--spot-opacity')).toBe('0');
    expect(card.querySelector('.ul-cursor-flow')).not.toBeNull();
    expect(card.textContent).toBe('hello');
  });
});

describe('AnimatedBeam', () => {
  it('computes a container-local quadratic path between node centres', () => {
    const g = computeBeamPath({ left: 10, top: 10, width: 400, height: 200 }, { left: 20, top: 30, width: 20, height: 20 }, { left: 310, top: 130, width: 20, height: 20 }, { curvature: 40 });
    expect(g.path).toBe('M 20,30 Q 165,-10 310,130'); expect(g.width).toBe(400);
    expect(computeBeamPath({ left: 0, top: 0, width: 1, height: 1 }, { left: 0, top: 0, width: 10, height: 10 }, { left: 20, top: 0, width: 10, height: 10 }, { startXOffset: 5 }).path).toBe('M 10,5 Q 17.5,5 25,5');
  });

  it('anchors to a node edge and builds pixel-space pulse keyframes that run head-first in either direction', () => {
    const box = { left: 0, top: 0, width: 400, height: 100 };
    const a = { left: 10, top: 20, width: 100, height: 60 }, b = { left: 200, top: 20, width: 100, height: 60 };
    expect(computeBeamPath(box, a, b, { anchor: 'top' }).path).toBe('M 60,22 Q 155,22 250,22');
    expect(computeBeamPath(box, a, b, { anchor: 'bottom' }).path).toBe('M 60,78 Q 155,78 250,78');
    const g = computeBeamPath(box, a, b);
    expect([g.startX, g.endX]).toEqual([60, 250]);
    expect(pulseKeyframes(60, 250, 40)).toEqual({ x1: [60, 290], x2: [20, 250] });
    expect(pulseKeyframes(250, 60, 40)).toEqual({ x1: [250, 20], x2: [290, 60] });
  });

  const Graph = ({ direction, converge }: { direction?: 'forward' | 'reverse' | 'both'; converge?: boolean }) => {
    const container = useRef<HTMLDivElement>(null), a = useRef<HTMLDivElement>(null), b = useRef<HTMLDivElement>(null), c = useRef<HTMLDivElement>(null);
    return (
      <div ref={container} data-testid="graph">
        <div ref={a} data-testid="a" /><div ref={b} data-testid="b" /><div ref={c} data-testid="c" />
        {converge ? <ConvergingBeams containerRef={container} sources={[a, b]} target={c} /> : <AnimatedBeam containerRef={container} fromRef={a} toRef={c} direction={direction} curvature={30} />}
      </div>
    );
  };
  const place = () => {
    const boxes: Record<string, DOMRect> = { graph: rectOf(0, 0, 300, 100), a: rectOf(10, 40, 20, 20), b: rectOf(10, 70, 20, 20), c: rectOf(270, 40, 20, 20) };
    for (const [id, box] of Object.entries(boxes)) document.querySelector<HTMLElement>(`[data-testid=${id}]`)!.getBoundingClientRect = () => box;
  };

  it('stays absent until the nodes exist, then draws the resting path (static under reduced motion)', async () => {
    const view = await mount(<Graph direction="both" />);
    // jsdom rects are all zero, so a path is still produced; re-measure with real boxes via a fresh mount.
    expect(view.host.querySelectorAll('svg.ul-beam').length).toBe(1);
    const svg = view.host.querySelector('svg')!;
    expect(svg.getAttribute('aria-hidden')).toBe('true');
    expect(svg.dataset.direction).toBe('both');
    expect(svg.dataset.active).toBe('false'); // reduced motion → no pulse
    expect(svg.querySelectorAll('path').length).toBe(1);
  });
  it('re-measures when the container resizes', async () => {
    let notify: () => void = () => {};
    const OriginalRO = globalThis.ResizeObserver;
    globalThis.ResizeObserver = class { constructor(cb: () => void) { notify = cb; } observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver;
    try {
      const view = await mount(<Graph />);
      place(); await act(async () => notify());
      expect(view.host.querySelector('svg path')!.getAttribute('d')).toBe('M 20,50 Q 150,20 280,50');
    } finally { globalThis.ResizeObserver = OriginalRO; }
  });
  it('divides CSS zoom out of every measurement, so the path and the SVG stay in layout pixels', async () => {
    let notify: () => void = () => {};
    const OriginalRO = globalThis.ResizeObserver;
    globalThis.ResizeObserver = class { constructor(cb: () => void) { notify = cb; } observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver;
    try {
      const view = await mount(<Graph />);
      // The same layout as above rendered at zoom 1.1: the browser reports every rect 10% larger, offsetWidth stays the layout width.
      const k = 1.1, boxes: Record<string, DOMRect> = { graph: rectOf(0, 0, 300 * k, 100 * k), a: rectOf(10 * k, 40 * k, 20 * k, 20 * k), b: rectOf(10 * k, 70 * k, 20 * k, 20 * k), c: rectOf(270 * k, 40 * k, 20 * k, 20 * k) };
      for (const [id, box] of Object.entries(boxes)) {
        const node = document.querySelector<HTMLElement>(`[data-testid=${id}]`)!;
        node.getBoundingClientRect = () => box;
        if (id === 'graph') Object.defineProperty(node, 'offsetWidth', { configurable: true, value: 300 });
      }
      await act(async () => notify());
      const svg = view.host.querySelector('svg')!;
      expect(svg.querySelector('path')!.getAttribute('d')).toBe('M 20,50 Q 150,20 280,50');
      expect([svg.getAttribute('width'), svg.getAttribute('height')]).toEqual(['300', '100']);
    } finally { globalThis.ResizeObserver = OriginalRO; }
  });
  it('ConvergingBeams draws one beam per source into the same target', async () => {
    const view = await mount(<Graph converge />);
    expect(view.host.querySelectorAll('svg.ul-beam').length).toBe(2);
  });
});

describe('MotionNumber', () => {
  it('formats with fixed decimals, signs and never shows -0', () => {
    expect(formatNumber(59.94, { decimals: 1 })).toBe('59.9');
    expect(formatNumber(12.5, { decimals: 1, suffix: ' ms' })).toBe('12.5 ms');
    expect(formatNumber(-4.2, { decimals: 1, suffix: '°' })).toBe('−4.2°');
    expect(formatNumber(4.2, { decimals: 1, signed: true })).toBe('+4.2');
    expect(formatNumber(-0.04, { decimals: 1 })).toBe('0.0');
    expect(formatNumber(Number.NaN, { suffix: ' fps' })).toBe('— fps');
  });
  it('renders the value, announces only the settled value, and follows new values (reduced motion jumps)', async () => {
    const view = await mount(<MotionNumber value={24} decimals={1} suffix=" GB" minChars={8} />);
    const live = () => view.host.querySelector<HTMLElement>('.ul-motion-number__live')!;
    expect(live().textContent).toBe('24.0 GB');
    expect(live().getAttribute('aria-hidden')).toBe('true');
    expect(live().style.minWidth).toBe('8ch');
    expect(view.host.querySelector('.ul-sr-only')!.textContent).toBe('24.0 GB');
    await view.rerender(<MotionNumber value={31.5} decimals={1} suffix=" GB" minChars={8} />);
    await flush();
    expect(live().textContent).toBe('31.5 GB');
    expect(view.host.querySelector('.ul-sr-only')!.textContent).toBe('31.5 GB');
    expect(view.host.querySelector('.ul-motion-number')!.classList.contains('ul-number')).toBe(true);
  });
});

describe('MultiAccordion', () => {
  const items = [
    { id: 'a', title: 'Gaze', content: <p>gaze body</p>, badge: '3' },
    { id: 'b', title: 'Jaw', content: <p>jaw body</p> },
    { id: 'c', title: 'Light', content: <p>light body</p>, disabled: true },
  ];
  it('opens several sections at once, wires ARIA, and closes again', async () => {
    const onOpenChange = vi.fn();
    const view = await mount(<MultiAccordion items={items} defaultOpen={['a']} onOpenChange={onOpenChange} aria-label="Parameters" />);
    const [a, b, c] = [...view.host.querySelectorAll<HTMLButtonElement>('button.ul-accordion__trigger')];
    expect(a.getAttribute('aria-expanded')).toBe('true'); expect(b.getAttribute('aria-expanded')).toBe('false'); expect(c.disabled).toBe(true);
    expect(view.host.textContent).toContain('gaze body'); expect(view.host.textContent).not.toContain('jaw body');
    expect(view.host.querySelector('[role=region]')!.getAttribute('aria-labelledby')).toBe(a.id);
    expect(a.getAttribute('aria-controls')).toBe(view.host.querySelector('[role=region]')!.id);
    await click(b); await flush();
    expect(onOpenChange).toHaveBeenLastCalledWith(['a', 'b']);
    expect(view.host.querySelectorAll('[role=region]').length).toBe(2); // multi-expand
    await click(a); await flush(60);
    expect(view.host.textContent).not.toContain('gaze body'); expect(view.host.textContent).toContain('jaw body');
  });
  it('moves focus with arrow keys, skipping disabled headers, wrapping, Home/End', async () => {
    const view = await mount(<MultiAccordion items={items} />);
    const [a, b] = [...view.host.querySelectorAll<HTMLButtonElement>('button.ul-accordion__trigger')];
    a.focus(); await key(a, 'ArrowDown'); expect(document.activeElement).toBe(b);
    await key(b, 'ArrowDown'); expect(document.activeElement).toBe(a); // wrapped past the disabled one
    await key(a, 'End'); expect(document.activeElement).toBe(b);
    await key(b, 'Home'); expect(document.activeElement).toBe(a);
  });
  it('can be controlled', async () => {
    const view = await mount(<MultiAccordion items={items} open={['b']} />);
    await click(view.host.querySelectorAll('button')[0]); await flush();
    expect(view.host.textContent).toContain('jaw body'); expect(view.host.textContent).not.toContain('gaze body'); // parent did not accept the change
  });
});

describe('AnimatedTabs', () => {
  const tabs = [{ id: 'one', label: 'Mesh', content: <p>mesh panel</p> }, { id: 'two', label: 'Light', content: <p>light panel</p> }, { id: 'three', label: 'Off', content: <p>off</p>, disabled: true }];
  it('implements the tabs pattern with a roving tabindex and one shared indicator', async () => {
    const onValueChange = vi.fn();
    const view = await mount(<AnimatedTabs tabs={tabs} onValueChange={onValueChange} aria-label="Sections" />);
    const [one, two] = [...view.host.querySelectorAll<HTMLButtonElement>('[role=tab]')];
    expect(one.getAttribute('aria-selected')).toBe('true'); expect(one.tabIndex).toBe(0); expect(two.tabIndex).toBe(-1);
    expect(view.host.querySelector('[role=tabpanel]')!.getAttribute('aria-labelledby')).toBe(one.id);
    expect(view.host.querySelectorAll('.ul-tabs__pill').length).toBe(1);
    one.focus(); await key(one, 'ArrowRight');
    expect(onValueChange).toHaveBeenLastCalledWith('two'); expect(two.getAttribute('aria-selected')).toBe('true'); expect(document.activeElement).toBe(two);
    expect(view.host.textContent).toContain('light panel'); expect(view.host.querySelectorAll('.ul-tabs__pill').length).toBe(1);
    await key(two, 'ArrowRight'); expect(one.getAttribute('aria-selected')).toBe('true'); // wraps past disabled
    await key(one, 'End'); expect(two.getAttribute('aria-selected')).toBe('true');
  });
  it('works as a panel-less segmented control and when controlled', async () => {
    const Controlled = () => { const [v, setV] = useState('two'); return <AnimatedTabs tabs={tabs.map(t => ({ id: t.id, label: t.label }))} value={v} onValueChange={setV} />; };
    const view = await mount(<Controlled />);
    expect(view.host.querySelector('[role=tabpanel]')).toBeNull();
    const [one, two] = [...view.host.querySelectorAll<HTMLButtonElement>('[role=tab]')];
    expect(two.getAttribute('aria-selected')).toBe('true');
    await click(one); expect(one.getAttribute('aria-selected')).toBe('true');
  });
});

describe('ClipPathSplitReveal', () => {
  it('builds the polygon wipe and loupe offsets', () => {
    expect(clipLeftOf(30)).toBe('polygon(0% 0%, 30% 0%, 30% 100%, 0% 100%)');
    expect(clipLeftOf(140)).toBe('polygon(0% 0%, 100% 0%, 100% 100%, 0% 100%)');
    expect(loupeOffset(100, 50, 4, 160)).toEqual({ left: -320, top: -120 });
  });
  const frame = () => document.querySelector<HTMLElement>('.ul-reveal')!;
  const slider = () => document.querySelector<HTMLElement>('[role=slider]')!;
  const before = () => document.querySelector<HTMLElement>('[data-testid=reveal-before]')!;

  it('is a keyboard-operable slider that moves the clip-path', async () => {
    const onPositionChange = vi.fn();
    await mount(<ClipPathSplitReveal beforeSrc="a.png" afterSrc="b.png" defaultPosition={40} onPositionChange={onPositionChange} />);
    expect(slider().getAttribute('aria-valuenow')).toBe('40'); expect(before().style.clipPath).toBe(clipLeftOf(40));
    await key(slider(), 'ArrowRight'); expect(slider().getAttribute('aria-valuenow')).toBe('41');
    await key(slider(), 'ArrowLeft', { shiftKey: true }); expect(slider().getAttribute('aria-valuenow')).toBe('31');
    await key(slider(), 'End'); expect(before().style.clipPath).toBe(clipLeftOf(100)); expect(onPositionChange).toHaveBeenLastCalledWith(100);
    await key(slider(), 'Home'); expect(slider().getAttribute('aria-valuenow')).toBe('0');
    expect(document.querySelectorAll('img').length).toBe(2);
  });
  it('drags the divider with the pointer', async () => {
    await mount(<ClipPathSplitReveal beforeSrc="a.png" afterSrc="b.png" />);
    frame().getBoundingClientRect = () => rectOf(0, 0, 400, 225);
    await pointer(frame(), 'pointerdown', { clientX: 100, clientY: 50 });
    expect(slider().getAttribute('aria-valuenow')).toBe('25');
    await pointer(frame(), 'pointermove', { clientX: 300, clientY: 50 });
    expect(slider().getAttribute('aria-valuenow')).toBe('75');
    await pointer(frame(), 'pointerup', {});
    await pointer(frame(), 'pointermove', { clientX: 20, clientY: 50 });
    expect(slider().getAttribute('aria-valuenow')).toBe('75'); // no drag in progress
  });
  it('shows a 4x loupe on Shift+hover over the correct side, and closes it without Shift', async () => {
    await mount(<ClipPathSplitReveal beforeSrc="a.png" afterSrc="b.png" defaultPosition={50} />);
    frame().getBoundingClientRect = () => rectOf(0, 0, 400, 225);
    await pointer(frame(), 'pointermove', { clientX: 100, clientY: 100 });
    expect(document.querySelector('[data-testid=reveal-loupe]')).toBeNull();
    await pointer(frame(), 'pointermove', { clientX: 100, clientY: 100, shiftKey: true });
    const loupe = document.querySelector<HTMLElement>('[data-testid=reveal-loupe]')!;
    expect(loupe.dataset.side).toBe('before');
    const img = loupe.querySelector('img')!;
    expect(img.getAttribute('src')).toBe('a.png'); expect(img.style.width).toBe('1600px'); expect(img.style.height).toBe('900px');
    expect([img.style.left, img.style.top]).toEqual(['-320px', '-320px']);
    await pointer(frame(), 'pointermove', { clientX: 300, clientY: 100, shiftKey: true });
    expect(document.querySelector<HTMLElement>('[data-testid=reveal-loupe]')!.dataset.side).toBe('after');
    expect(document.querySelector('[data-testid=reveal-loupe] img')!.getAttribute('src')).toBe('b.png');
    await act(async () => { window.dispatchEvent(new KeyboardEvent('keyup', { key: 'Shift' })); });
    expect(document.querySelector('[data-testid=reveal-loupe]')).toBeNull();
  });
  it('disables the loupe with loupeZoom={0}', async () => {
    await mount(<ClipPathSplitReveal beforeSrc="a.png" afterSrc="b.png" loupeZoom={0} />);
    frame().getBoundingClientRect = () => rectOf(0, 0, 400, 225);
    await pointer(frame(), 'pointermove', { clientX: 100, clientY: 100, shiftKey: true });
    expect(document.querySelector('[data-testid=reveal-loupe]')).toBeNull();
  });
});

describe('MotionDrawer', () => {
  it('is a labelled modal dialog: portals to body, closes on Esc and backdrop, restores focus', async () => {
    const onClose = vi.fn();
    const opener = document.createElement('button'); document.body.append(opener); opener.focus();
    const view = await mount(<MotionDrawer open onClose={onClose} title="Advanced" description="Rarely used knobs" footer={<span>foot</span>}><button>Inner</button></MotionDrawer>);
    const dialog = document.querySelector<HTMLElement>('[role=dialog]')!;
    expect(dialog.parentElement!.parentElement).toBe(document.body); // portalled
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    expect(document.getElementById(dialog.getAttribute('aria-labelledby')!)!.textContent).toBe('Advanced');
    expect(document.getElementById(dialog.getAttribute('aria-describedby')!)!.textContent).toBe('Rarely used knobs');
    expect(dialog.classList.contains('ul-drawer--right')).toBe(true);
    expect(dialog.contains(document.activeElement)).toBe(true);
    await key(document.body, 'Escape'); expect(onClose).toHaveBeenCalledTimes(1);
    await click(document.querySelector('.ul-drawer__close')!); expect(onClose).toHaveBeenCalledTimes(2);
    await pointer(document.querySelector('[data-testid=drawer-backdrop]')!, 'mousedown'); expect(onClose).toHaveBeenCalledTimes(3);
    await view.rerender(<MotionDrawer open={false} onClose={onClose} title="Advanced" />); await flush(60);
    expect(document.querySelector('[role=dialog]')).toBeNull();
    expect(document.activeElement).toBe(opener);
  });
  it('supports other sides and sizes', async () => {
    await mount(<MotionDrawer open onClose={() => {}} title="Bottom" side="bottom" size="40vh" />);
    const dialog = document.querySelector<HTMLElement>('[role=dialog]')!;
    expect(dialog.classList.contains('ul-drawer--bottom')).toBe(true); expect(dialog.style.height).toBe('40vh');
  });
});

describe('PrecisionRangeSlider', () => {
  it('snaps to the step grid without float noise and finds the fill segment', () => {
    expect(snapToStep(0.30000000000000004, 0, 1, 0.1)).toBe(0.3);
    expect(snapToStep(7.4, 0, 10, 5)).toBe(5); expect(snapToStep(99, 0, 10, 1)).toBe(10); expect(snapToStep(-3, -2, 2, 0.5)).toBe(-2);
    expect(fillSegment(25, 0, 100)).toEqual({ start: 0, size: 25 });
    expect(fillSegment(-45, -90, 90)).toEqual({ start: 25, size: 25 }); // outward from centre
    expect(fillSegment(45, -90, 90)).toEqual({ start: 50, size: 25 });
  });

  const Single = ({ onChange, def = 20 }: { onChange?: (v: number) => void; def?: number }) => {
    const [v, setV] = useState(60);
    return <PrecisionRangeSlider label="Yaw" min={0} max={100} step={5} unit="°" value={v} defaultValue={def} onChange={n => { setV(n); onChange?.(n); }} />;
  };
  const thumb = () => document.querySelector<HTMLElement>('[role=slider]')!;

  it('exposes slider semantics and a live fill', async () => {
    await mount(<Single />);
    expect(thumb().getAttribute('aria-label')).toBe('Yaw');
    expect([thumb().getAttribute('aria-valuemin'), thumb().getAttribute('aria-valuemax'), thumb().getAttribute('aria-valuenow'), thumb().getAttribute('aria-valuetext')]).toEqual(['0', '100', '60', '60°']);
    expect(document.querySelector<HTMLElement>('[data-testid=slider-fill]')!.style.width).toBe('60%');
    expect(document.querySelector('.ul-slider__readout')!.textContent).toBe('60°');
  });
  it('responds to the keyboard and resets on double-click or Escape', async () => {
    const onChange = vi.fn();
    await mount(<Single onChange={onChange} />);
    await key(thumb(), 'ArrowRight'); expect(thumb().getAttribute('aria-valuenow')).toBe('65');
    await key(thumb(), 'ArrowLeft', { shiftKey: true }); expect(thumb().getAttribute('aria-valuenow')).toBe('15');
    await key(thumb(), 'End'); expect(thumb().getAttribute('aria-valuenow')).toBe('100');
    await key(thumb(), 'Home'); expect(thumb().getAttribute('aria-valuenow')).toBe('0');
    await key(thumb(), 'PageUp'); expect(thumb().getAttribute('aria-valuenow')).toBe('50');
    await act(async () => { document.querySelector('[data-testid=slider-track]')!.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })); });
    expect(thumb().getAttribute('aria-valuenow')).toBe('20');
    await key(thumb(), 'End'); await key(thumb(), 'Escape'); expect(thumb().getAttribute('aria-valuenow')).toBe('20');
    await key(thumb(), 'End'); await act(async () => { document.querySelector('.ul-slider__readout')!.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })); });
    expect(thumb().getAttribute('aria-valuenow')).toBe('20');
  });
  it('scrubs from the track and from the readout (Shift = fine)', async () => {
    await mount(<Single />);
    const track = document.querySelector<HTMLElement>('[data-testid=slider-track]')!;
    track.getBoundingClientRect = () => rectOf(0, 0, 200, 22);
    await pointer(track, 'pointerdown', { clientX: 100 }); expect(thumb().getAttribute('aria-valuenow')).toBe('50');
    await pointer(track, 'pointermove', { clientX: 150 }); expect(thumb().getAttribute('aria-valuenow')).toBe('75');
    await pointer(track, 'pointerup'); await pointer(track, 'pointermove', { clientX: 10 }); expect(thumb().getAttribute('aria-valuenow')).toBe('75');
    const readout = document.querySelector<HTMLElement>('.ul-slider__readout')!;
    await pointer(readout, 'pointerdown', { clientX: 0 });
    await pointer(readout, 'pointermove', { clientX: 30 }); expect(thumb().getAttribute('aria-valuenow')).toBe('85'); // 30px * (100/300) = +10
    await pointer(readout, 'pointermove', { clientX: 30, shiftKey: true }); expect(thumb().getAttribute('aria-valuenow')).toBe('75'); // +1 → snaps back to 75
    await pointer(readout, 'pointerup');
  });
  it('fills outward from zero for signed ranges', async () => {
    await mount(<PrecisionRangeSlider label="Pitch" min={-90} max={90} value={-45} onChange={() => {}} />);
    const fill = document.querySelector<HTMLElement>('[data-testid=slider-fill]')!;
    expect([fill.style.left, fill.style.width]).toEqual(['25%', '25%']);
  });
  it('ignores input when disabled', async () => {
    const onChange = vi.fn();
    await mount(<PrecisionRangeSlider label="Yaw" min={0} max={10} value={5} disabled onChange={onChange} />);
    await key(thumb(), 'ArrowRight'); expect(onChange).not.toHaveBeenCalled();
    expect(thumb().tabIndex).toBe(-1);
  });

  const Split = ({ onSides }: { onSides?: (s: Sides) => void }) => {
    const [sides, setSides] = useState<Sides>({ left: 10, right: 30 });
    return <PrecisionRangeSlider label="Brow" min={0} max={100} step={5} sides={sides} defaultSides={{ left: 0, right: 0 }} defaultLinked={false} onSidesChange={s => { setSides(s); onSides?.(s); }} />;
  };
  it('splits into independent Left/Right tracks, and re-linking collapses them onto the mean', async () => {
    const onSides = vi.fn();
    await mount(<Split onSides={onSides} />);
    const sliders = () => [...document.querySelectorAll<HTMLElement>('[role=slider]')];
    expect(sliders().map(s => s.getAttribute('aria-label'))).toEqual(['Brow Left', 'Brow Right']);
    const toggle = document.querySelector<HTMLButtonElement>('.ul-slider__link')!;
    expect(toggle.getAttribute('aria-pressed')).toBe('false');
    await key(sliders()[0], 'ArrowRight'); expect(onSides).toHaveBeenLastCalledWith({ left: 15, right: 30 });
    await key(sliders()[1], 'ArrowLeft'); expect(onSides).toHaveBeenLastCalledWith({ left: 15, right: 25 });
    await act(async () => { sliders()[1].closest('.ul-slider__row')!.querySelector('[data-testid=slider-track]')!.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })); });
    expect(onSides).toHaveBeenLastCalledWith({ left: 15, right: 0 }); // per-side reset target
    await click(toggle);
    expect(toggle.getAttribute('aria-pressed')).toBe('true');
    expect(onSides).toHaveBeenLastCalledWith({ left: 10, right: 10 }); // mean(15, 0) = 7.5 → snaps to the 5-step grid
    expect(sliders().length).toBe(1);
    await key(sliders()[0], 'ArrowRight'); expect(onSides).toHaveBeenLastCalledWith({ left: 15, right: 15 }); // linked: one track drives both
  });
});
