// @vitest-environment jsdom
import { act, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import { HotkeyProvider } from './components/controls/Hotkeys';
import { WorkspaceLayoutProvider } from './components/workstation/WorkspaceLayouts';
import App from './App';

vi.mock('./events', () => ({ EventConnection: class { start() {} stop() {} } }));
vi.mock('./ThreeViewport', () => ({ PoseCoverageInspector: () => <div>Pose inspector</div> }));
vi.mock('./components/hud/PipelinePill', () => ({ default: () => null }));
vi.mock('./components/queue/QueueDrawer', () => ({ default: () => null }));
vi.mock('./components/facemgr/FaceManager', () => ({ default: function Manager() { const [draft, setDraft] = useState(''); return <input aria-label="Identity draft" value={draft} onChange={e => setDraft(e.target.value)}/>; } }));
vi.mock('./VideoViewport', () => ({ formatTime: (value: number) => String(value), VideoViewport: function Canvas() { const [count, setCount] = useState(0); return <button onClick={() => setCount(count + 1)}>Canvas draft {count}</button>; } }));
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
