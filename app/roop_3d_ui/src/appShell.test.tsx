// @vitest-environment jsdom
import { act, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import { HotkeyProvider } from './components/controls/Hotkeys';
import { WorkspaceLayoutProvider } from './components/workstation/WorkspaceLayouts';
import App from './App';
import useHistoryStore from './stores/useHistoryStore';

vi.mock('./events', () => ({ EventConnection: class { start() {} stop() {} } }));
vi.mock('./ThreeViewport', () => ({ PoseCoverageInspector: () => <div>Pose inspector</div>, PoseGeometryViewer: () => <div>Pose viewer</div>, LightProbeSphere: () => <div>Light probe</div> }));
vi.mock('./components/hud/PipelinePill', () => ({ default: () => null }));
vi.mock('./components/queue/QueueDrawer', () => ({ default: () => null }));
vi.mock('./components/facemgr/FaceManager', () => ({ default: function Manager() { const [draft, setDraft] = useState(''); return <input aria-label="Identity draft" value={draft} onChange={e => setDraft(e.target.value)}/>; } }));
vi.mock('./VideoViewport', () => ({ formatTime: (value: number) => String(value), VideoViewport: function Canvas(props: any) { const [count, setCount] = useState(0); return <><button onClick={() => setCount(count + 1)}>Canvas draft {count}</button><button onClick={() => props.setParameters({ ...props.parameters, swap_weight: .2 })}>Change parameter</button><button onClick={() => props.onMaskChange('data:image/png;base64,AQ==')}>Mask stroke</button><output data-workspace>{JSON.stringify(props.captureWorkspace())}</output></>; } }));
vi.mock('./api', () => ({ post: vi.fn(), upload: vi.fn(), api: async (path: string) => path === '/system/capabilities' ? { models: [], job_parameter_bounds: {}, legal_parameter_bounds: {}, workstation: { features: {}, read_only_parameters: {}, disabled_parameters: {}, controls: {}, job_controls: {} } } : path === '/config' ? {} : [] }));

it('mounts the integrated application shell with all nine screens and keeps canvas drafts across navigation', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); localStorage.clear(); location.hash = '#/faceswap';
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  try {
    await act(async () => root.render(<HotkeyProvider><WorkspaceLayoutProvider><App/></WorkspaceLayoutProvider></HotkeyProvider>));
    expect(host.querySelectorAll('[data-screen]')).toHaveLength(9);
    expect(host.querySelectorAll('.screen-nav a')).toHaveLength(9);
    const draft = [...host.querySelectorAll('button')].find(button => button.textContent === 'Canvas draft 0')!;
    await act(async () => draft.click());
    for (const route of ['#/facemgr', '#/settings', '#/history', '#/gallery', '#/faceswap']) {
      await act(async () => { location.hash = route; window.dispatchEvent(new HashChangeEvent('hashchange')); });
      expect(host.querySelector('[data-screen]:not([hidden])')?.getAttribute('data-screen')).toBe(route.slice(2));
    }
    expect(host.querySelector('[data-screen="faceswap"]')?.textContent).toContain('Canvas draft 1');
  } finally { await act(async () => root.unmount()); host.remove(); }
});

it('undoes parameter changes and mask strokes through the integrated Ctrl+Z / Ctrl+Y shortcuts', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); localStorage.clear(); useHistoryStore.getState().reset();
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  try {
    await act(async () => root.render(<HotkeyProvider><WorkspaceLayoutProvider><App/></WorkspaceLayoutProvider></HotkeyProvider>));
    await act(async () => [...host.querySelectorAll('button')].find(button => button.textContent === 'Change parameter')!.click());
    await act(async () => [...host.querySelectorAll('button')].find(button => button.textContent === 'Mask stroke')!.click());
    expect(useHistoryStore.getState().past).toHaveLength(2);
    await act(async () => document.body.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'z', ctrlKey: true })));
    expect(JSON.parse(host.querySelector('[data-workspace]')!.textContent!).maskBase64).toBeNull();
    await act(async () => document.body.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'z', ctrlKey: true })));
    expect(JSON.parse(host.querySelector('[data-workspace]')!.textContent!).parameters.swap_weight).toBe(1);
    await act(async () => document.body.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'y', ctrlKey: true })));
    expect(JSON.parse(host.querySelector('[data-workspace]')!.textContent!).parameters.swap_weight).toBe(.2);
  } finally { await act(async () => root.unmount()); host.remove(); }
});

it('opens the Quick Tour from the header and from #/home, walks all four steps, and logs zero console errors', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); localStorage.clear(); location.hash = '#/home';
  const errors = vi.spyOn(console, 'error');
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  const named = (label: string) => [...host.querySelectorAll('button')].find(button => button.textContent?.includes(label))!;
  try {
    await act(async () => root.render(<HotkeyProvider><WorkspaceLayoutProvider><App/></WorkspaceLayoutProvider></HotkeyProvider>));
    expect(host.querySelector('[data-testid="header-tour-button"]')?.textContent).toBe('How to Use');
    expect(host.querySelector('[data-testid="home-tour-button"]')).toBeTruthy();
    await act(async () => (host.querySelector('[data-testid="home-tour-button"]') as HTMLElement).click());
    expect(host.querySelector('[role="dialog"][aria-labelledby="tour-title"]')).toBeTruthy();
    expect(location.hash).toBe('#/faceswap');                       // the tour moves to the workstation for step 1
    for (let step = 0; step < 3; step++) await act(async () => named('Next').click());
    expect(host.querySelector('.tour-card')?.getAttribute('data-step')).toBe('render');
    await act(async () => named('Done').click());
    expect(host.querySelector('[data-testid="quick-tour"]')).toBeNull();
    await act(async () => (host.querySelector('[data-testid="header-tour-button"]') as HTMLElement).click());
    expect(host.querySelector('.tour-card')?.getAttribute('data-step')).toBe('source'); // header button restarts the tour
    await act(async () => (host.querySelector('[aria-label="Close tour"]') as HTMLElement).click());
    expect(errors.mock.calls.map(call => String(call[0]))).toEqual([]);
  } finally { errors.mockRestore(); await act(async () => root.unmount()); host.remove(); }
});
