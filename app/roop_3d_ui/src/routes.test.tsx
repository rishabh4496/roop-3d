// @vitest-environment jsdom
import { act, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import { HotkeyProvider } from './components/controls/Hotkeys';
import { WorkspaceLayoutProvider } from './components/workstation/WorkspaceLayouts';
import App from './App';
import useHistoryStore from './stores/useHistoryStore';

vi.mock('./events', () => ({ EventConnection: class { start() {} stop() {} } }));
vi.mock('./ThreeViewport', () => ({ PoseCoverageInspector: () => <div>Pose inspector</div> }));
vi.mock('./components/hud/PipelinePill', () => ({ default: () => null }));
vi.mock('./components/queue/QueueDrawer', () => ({ default: () => null }));
vi.mock('./components/facemgr/FaceManager', () => ({ default: function Manager() { const [draft, setDraft] = useState(''); return <input aria-label="Identity draft" value={draft} onChange={e => setDraft(e.target.value)}/>; } }));
vi.mock('./VideoViewport', () => ({ formatTime: (value: number) => String(value), VideoViewport: function Canvas(props: any) { const [count, setCount] = useState(0); return <><button onClick={() => setCount(count + 1)}>Canvas draft {count}</button><button onClick={() => props.setParameters({ ...props.parameters, swap_weight: .2 })}>Change parameter</button><button onClick={() => props.onMaskChange('data:image/png;base64,AQ==')}>Mask stroke</button><output data-workspace>{JSON.stringify(props.captureWorkspace())}</output></>; } }));
vi.mock('./api', () => ({ post: vi.fn(), upload: vi.fn(), api: async (path: string) => path === '/system/capabilities' ? { models: [], job_parameter_bounds: {}, legal_parameter_bounds: {}, workstation: { features: {}, read_only_parameters: {}, disabled_parameters: {}, controls: {}, job_controls: {} } } : path === '/config' ? {} : [] }));

import { SCREENS, normalizeRoute } from './router/AppRouter';

const ROUTES = [
  ['#/home', 'Home'], ['#/faceswap', 'Face Swap'], ['#/batch', 'Batch Matrix'], ['#/processing', 'Processing'], ['#/facemgr', 'Face Manager'],
  ['#/extras', 'Editor'], ['#/gallery', 'Outputs'], ['#/history', 'History'], ['#/settings', 'Settings'],
] as const;

const mount = async (hash: string) => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); localStorage.clear(); location.hash = hash;
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  await act(async () => root.render(<HotkeyProvider><WorkspaceLayoutProvider><App/></WorkspaceLayoutProvider></HotkeyProvider>));
  return { host, unmount: async () => { await act(async () => root.unmount()); host.remove(); } };
};
const visible = (host: HTMLElement) => [...host.querySelectorAll('[data-screen]:not([hidden])')].map(node => node.getAttribute('data-screen'));

it('defines exactly the nine deep-linked routes with their names', () => {
  expect(SCREENS.map(([id, label]) => [`#/${id}`, label])).toEqual(ROUTES.map(route => [...route]));
  for (const [route] of ROUTES) expect(normalizeRoute(route)).toBe(route);
  expect(normalizeRoute('#/batch/')).toBe('#/batch');                    // trailing slash
  expect(normalizeRoute('#/gallery?clip=3')).toBe('#/gallery');          // query is not part of the route
  expect(normalizeRoute('#/workstation')).toBe('#/faceswap');            // legacy alias
  expect([normalizeRoute(''), normalizeRoute('#')]).toEqual(['#/faceswap', '#/faceswap']);
  expect(normalizeRoute('#/nope')).toBe('#/home');                       // unknown routes land on Home, never a blank screen
});

it('renders each of the nine routes: one visible screen, a current nav link, every other screen hidden and inert', async () => {
  const { host, unmount } = await mount('#/home');
  try {
    expect(host.querySelectorAll('[data-screen]')).toHaveLength(9);
    for (const [route, label] of ROUTES) {
      await act(async () => { location.hash = route; window.dispatchEvent(new HashChangeEvent('hashchange')); });
      const id = route.slice(2);
      expect(visible(host)).toEqual([id]);
      const current = host.querySelectorAll('.screen-nav a[aria-current="page"]');
      expect(current).toHaveLength(1);
      expect([current[0].getAttribute('href'), current[0].textContent]).toEqual([route, label]);
      for (const hidden of host.querySelectorAll('[data-screen][hidden]')) expect(hidden.hasAttribute('inert')).toBe(true);
      expect(host.querySelector(`[data-screen="${id}"]`)!.hasAttribute('inert')).toBe(false);
    }
  } finally { await unmount(); }
});

it('restores the exact screen after a refresh: a fresh mount on each hash opens that screen with no hashchange event', async () => {
  for (const [route] of ROUTES) {
    const { host, unmount } = await mount(route);
    try {
      expect(visible(host)).toEqual([route.slice(2)]);
      expect(location.hash).toBe(route);
    } finally { await unmount(); }
  }
});

it('cleans up an unknown or empty hash instead of leaving a blank page', async () => {
  const unknown = await mount('#/not-a-screen');
  try { expect(visible(unknown.host)).toEqual(['home']); expect(location.hash).toBe('#/home'); } finally { await unknown.unmount(); }
  const empty = await mount('');
  try { expect(visible(empty.host)).toEqual(['faceswap']); } finally { await empty.unmount(); }
});
