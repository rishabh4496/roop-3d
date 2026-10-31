// @vitest-environment jsdom
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import AppRouter, { Navigation, SCREENS, normalizeRoute, useHashRoute } from './router/AppRouter';
import NumericInput, { evaluateExpression, scrubDelta } from './components/controls/NumericInput';
import { DEFAULT_KEYS, HotkeyEditor, HotkeyProvider, useShortcutActions, validKeys } from './components/controls/Hotkeys';
import { DockArea, DockPanel, LayoutToolbar, PRESETS, WorkspaceLayoutProvider, validLayout } from './components/workstation/WorkspaceLayouts';
import { FloatingPreviewMedia, useFloatingPreview } from './components/workstation/FloatingPreview';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let host: HTMLDivElement, root: Root;
beforeEach(() => { localStorage.clear(); history.replaceState(null, '', '/#/faceswap'); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.restoreAllMocks(); });
const mount = async (element: React.ReactNode) => { await act(async () => root.render(element)); };
const click = async (element: Element | null) => { expect(element).toBeTruthy(); await act(async () => (element as HTMLElement).click()); };
const input = async (element: HTMLInputElement, value: string) => { await act(async () => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(element, value); element.dispatchEvent(new Event('input', { bubbles: true })); }); };
const key = async (element: Element, value: string, modifiers = {}) => { await act(async () => element.dispatchEvent(new KeyboardEvent('keydown', { key: value, bubbles: true, cancelable: true, ...modifiers }))); };

it('keeps all nine deep-linked screens mounted and preserves local edits across back/forward', async () => {
  function Draft() { const [value, setValue] = useState(''); return <input aria-label="Persistent draft" value={value} onChange={e => setValue(e.target.value)}/>; }
  function Shell() { const route = useHashRoute(); return <><Navigation route={route}/>{SCREENS.map(([screen]) => <AppRouter route={route} screen={screen} key={screen}>{screen === 'faceswap' ? <Draft/> : screen}</AppRouter>)}</>; }
  await mount(<Shell/>);
  await input(host.querySelector('input')!, 'scene 42');
  for (const [screen] of SCREENS) {
    await act(async () => { location.hash = `#/${screen}`; window.dispatchEvent(new HashChangeEvent('hashchange')); });
    expect(host.querySelectorAll('[data-screen]')).toHaveLength(9);
    expect(host.querySelector('[data-screen]:not([hidden])')?.getAttribute('data-screen')).toBe(screen);
  }
  await act(async () => { location.hash = '#/faceswap'; window.dispatchEvent(new HashChangeEvent('hashchange')); });
  expect(host.querySelector<HTMLInputElement>('input')!.value).toBe('scene 42');
  expect(host.querySelector('[aria-current=page]')?.getAttribute('href')).toBe('#/faceswap');
  expect(normalizeRoute('#/garbage')).toBe('#/home');
  expect(normalizeRoute('#/')).toBe('#/faceswap');
});

it('evaluates arithmetic safely, supports relative increments and rejects executable text', () => {
  expect(evaluateExpression('512 * 1.5')).toBe(768);
  expect(evaluateExpression('100 / 3')).toBeCloseTo(33.33333);
  expect(evaluateExpression('+15', 25)).toBe(40);
  expect(evaluateExpression('(2 + 3) * -4')).toBe(-20);
  for (const value of ['1/0', 'window.alert(1)', '2**3', '1;2', '', '(1+2', 'NaN', '1'.repeat(129)]) expect(() => evaluateExpression(value)).toThrow();
  expect(scrubDelta(2, .5, true)).toBe(10);
  expect(scrubDelta(2, .5, false, true)).toBe(.1);
});

it('commits expressions once on Enter/blur, clamps scrubs, and isolates invalid drafts', async () => {
  function Field() { const [value, setValue] = useState<number | null>(100); return <NumericInput label="Resolution" value={value} min={0} max={200} onChange={setValue}/>; }
  await mount(<Field/>); const field = host.querySelector('input')!;
  await input(field, '+15'); await key(field, 'Enter'); expect(field.value).toBe('115');
  await act(async () => field.dispatchEvent(new FocusEvent('focusout', { bubbles: true }))); expect(field.value).toBe('115');
  await input(field, '100 / 0'); await key(field, 'Enter'); expect(field.getAttribute('aria-invalid')).toBe('true');
  await key(field, 'Escape'); expect(field.value).toBe('115');
  await key(host.querySelector('label')!, 'ArrowRight', { shiftKey: true }); expect(field.value).toBe('125');
  await input(field, '900'); await key(field, 'Enter'); expect(field.value).toBe('200');
});

it('detects shortcut conflicts, persists remaps, and invokes the remapped action only outside inputs', async () => {
  const play = vi.fn();
  function Actions() { useShortcutActions({ play }); return <HotkeyEditor/>; }
  await mount(<HotkeyProvider><Actions/></HotkeyProvider>);
  const field = host.querySelector<HTMLInputElement>('[aria-label="Shortcut for Play / pause"]')!;
  await key(field, 's'); expect(host.textContent).toContain('already assigned'); expect(field.value).toBe('Space');
  await key(field, 'k'); expect(field.value).toBe('K');
  await key(document.body, 'k'); expect(play).toHaveBeenCalledTimes(1);
  await key(field, 'k'); expect(play).toHaveBeenCalledTimes(1);
  expect(JSON.parse(localStorage.getItem('roop3d-hotkeys-v1')!).play).toBe('K');
  expect(validKeys({ ...DEFAULT_KEYS, play: 'S' })).toEqual(DEFAULT_KEYS);
});

it('saves named panel positions, sizes and collapsed states and restores them after remount', async () => {
  function Layout() { return <WorkspaceLayoutProvider><LayoutToolbar onFloat={() => {}}/><DockArea><DockPanel id="timeline" title="Timeline"><input defaultValue="keep this draft"/></DockPanel><DockPanel id="tray" title="Face Tray">faces</DockPanel></DockArea></WorkspaceLayoutProvider>; }
  await mount(<Layout/>);
  await click(host.querySelector('[aria-label="Collapse Timeline"]'));
  await click(host.querySelector('[aria-label="Move Timeline later"]'));
  await key(host.querySelector('[aria-label="Resize Face Tray"]')!, 'ArrowUp');
  await input(host.querySelector('[aria-label="Custom layout name"]')!, 'My edit');
  await click([...host.querySelectorAll('button')].find(button => button.textContent === 'Save layout')!);
  const saved = JSON.parse(localStorage.getItem('roop3d-custom-layouts-v1')!)['My edit'];
  expect(saved.order[0]).toBe('tray'); expect(saved.panels.timeline.collapsed).toBe(true); expect(saved.panels.tray.height).toBe(320);
  await act(async () => root.unmount()); root = createRoot(host); await mount(<Layout/>);
  expect(host.querySelector('[aria-label="Expand Timeline"]')).toBeTruthy();
  expect(host.querySelector<HTMLInputElement>('.dock-content input')!.value).toBe('keep this draft');
  expect(validLayout({ order: ['pose'] })).toEqual(PRESETS['Standard Workstation']);
});

it('shares live preview content with a secondary window and closes it on teardown', async () => {
  const childDocument = document.implementation.createHTMLDocument(''), close = vi.fn();
  vi.spyOn(window, 'open').mockReturnValue({ document: childDocument, addEventListener: vi.fn(), close, closed: false, focus: vi.fn() } as unknown as Window);
  function Floating() { const [frame, setFrame] = useState(1); const floating = useFloatingPreview(<span>Frame {frame}</span>); return <><button onClick={floating.open}>Detach</button><button onClick={() => setFrame(2)}>Seek</button>{floating.portal}</>; }
  await mount(<Floating/>); await click(host.querySelector('button')); expect(childDocument.body.textContent).toContain('Frame 1');
  await click(host.querySelectorAll('button')[1]); expect(childDocument.body.textContent).toContain('Frame 2');
  await act(async () => root.unmount()); expect(close).toHaveBeenCalledOnce(); root = createRoot(host);
});

it('synchronizes playback and seeks in the floating video, and shows paused buffer images', async () => {
  const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  await mount(<FloatingPreviewMedia path="local.mp4" time={2} playing label="OUTPUT"/>);
  expect(play).toHaveBeenCalledOnce(); expect(host.querySelector('video')!.currentTime).toBe(2);
  await mount(<FloatingPreviewMedia path="local.mp4" time={3} playing={false} label="OUTPUT"/>);
  expect(pause).toHaveBeenCalledOnce(); expect(host.querySelector('video')!.currentTime).toBe(3);
  await mount(<FloatingPreviewMedia image="data:image/jpeg;base64,AQ==" path="" time={3} playing={false} label="SOLO"/>);
  expect(host.querySelector('video')).toBeNull(); expect(host.querySelector('img')).toBeTruthy();
});

it('provides a keyboard-dismissable modal when popups are blocked', async () => {
  vi.spyOn(window, 'open').mockReturnValue(null);
  function Fallback() { const floating = useFloatingPreview(<span>Preview frame</span>); return <><button onClick={floating.open}>Detach</button>{floating.portal}</>; }
  await mount(<Fallback/>); await click(host.querySelector('button'));
  expect(host.querySelector('[role=dialog]')).toBeTruthy();
  await key(document.activeElement!, 'Escape'); expect(host.querySelector('[role=dialog]')).toBeNull();
});
