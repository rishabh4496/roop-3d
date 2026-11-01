// @vitest-environment jsdom
/**
 * Stage 11 verification gate (React Profiler + render probes against the REAL <App/>):
 *  1. simulated 30 Hz /ws/telemetry (+ /api/v1/ws/events + the HUD socket) bursts trigger 0 re-renders in the parameter
 *     sidebar (SettingControl panels, SettingsInspector), the timeline, the Face Manager, VideoViewport and App itself;
 *  3. the floating PiP window and the hotkey remapper persist across route changes.
 * (2, inline expression math, is covered in stage11Math.test.tsx.)
 */
import { Profiler, act, useState, type ComponentType } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HotkeyProvider } from './components/controls/Hotkeys';
import { WorkspaceLayoutProvider } from './components/workstation/WorkspaceLayouts';
import { pipStore } from './components/workstation/FloatingPreview';
import { resetTelemetry } from './stores/telemetryStore';
import { renderProbe } from './utils/renderProbe';
import * as events from './events';
import App from './App';

const profiled = vi.hoisted(() => [] as { id: string; phase: string }[]);
vi.mock('./events', () => {
  const links: any[] = [];
  class EventConnection { constructor(public url: string, public receive: (event: any) => void, public status: (state: string) => void) { links.push(this); } start() { this.status('online'); } stop() {} }
  return { EventConnection, __links: links };
});
vi.mock('./ThreeViewport', () => ({ PoseCoverageInspector: () => <div>Pose inspector</div> }));
vi.mock('./components/queue/QueueDrawer', () => ({ default: () => null }));
vi.mock('./components/canvas/FaceOverlayCanvas', () => ({ default: () => null }));
vi.mock('./components/canvas/MaskBrushOverlay', () => ({ default: () => null }));
vi.mock('./components/tray/FaceSelectionTray', () => ({ default: () => null }));
// The Face Manager's 3D sphere needs a real GPU; its render count (not its pixels) is what this gate measures.
vi.mock('./components/facemgr/FaceManager', () => ({ default: function FaceManager() { renderProbe('FaceManager'); const [draft, setDraft] = useState(''); return <input aria-label="Identity draft" value={draft} onChange={event => setDraft(event.target.value)}/>; } }));
// React Profiler around the real parameter controls: any commit that touches them is recorded.
vi.mock('./components/controls/SettingControl', async () => {
  const actual: any = await vi.importActual('./components/controls/SettingControl');
  return { default: (props: any) => <Profiler id="SettingControl" onRender={(id, phase) => profiled.push({ id, phase })}><actual.default {...props}/></Profiler> };
});
vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof import('./api')>()), post: vi.fn(), upload: vi.fn(),
  api: async (path: string) => path === '/system/capabilities' ? { models: [], job_parameter_bounds: {}, legal_parameter_bounds: {}, workstation: { features: {}, read_only_parameters: {}, disabled_parameters: {}, controls: {}, job_controls: {} } } : path === '/config' ? {} : [],
}));

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
const probe: Record<string, number> = {};
class FakeSocket {
  static instances: FakeSocket[] = [];
  onopen: (() => void) | null = null; onmessage: ((event: { data: string }) => void) | null = null; onclose: (() => void) | null = null; onerror: (() => void) | null = null;
  constructor(public url: string) { FakeSocket.instances.push(this); }
  close() {}
}
let host: HTMLDivElement, root: Root;
/** Wait until the app stops rendering on its own (capability fetch, config → draft effects), so a baseline is a true idle. */
const settle = async () => {
  for (let round = 0; round < 25; round++) {
    const before = JSON.stringify(probe);
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 15)); });
    if (JSON.stringify(probe) === before) return;
  }
  throw new Error('App never settled');
};
const mount = async () => {
  await act(async () => root.render(<HotkeyProvider><WorkspaceLayoutProvider><App/></WorkspaceLayoutProvider></HotkeyProvider>));
  await settle();
};
const go = async (route: string) => act(async () => { location.hash = route; window.dispatchEvent(new HashChangeEvent('hashchange')); });
const press = async (key: string, modifiers: KeyboardEventInit = {}, target: EventTarget = document.body) => act(async () => { target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...modifiers })); });
const snapshot = (): Record<string, number> => ({ ...probe, commits: profiled.length });

beforeEach(() => {
  localStorage.clear(); sessionStorage.clear(); sessionStorage.setItem('roop3d_skip_linter', 'true');
  for (const key of Object.keys(probe)) delete probe[key];
  profiled.length = 0; FakeSocket.instances = []; (events as any).__links.length = 0;
  (globalThis as any).__R3D_RENDER_PROBE__ = probe;
  vi.stubGlobal('WebSocket', FakeSocket);
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
  vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 503 })));
  resetTelemetry(); pipStore.set({ mode: 'closed', geometry: null });
  history.replaceState(null, '', '/#/faceswap');
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); delete (globalThis as any).__R3D_RENDER_PROBE__; vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('30 Hz telemetry isolation', () => {
  it('re-renders none of the parameter sidebar, timeline, Face Manager, VideoViewport or App while telemetry streams', async () => {
    await mount();
    const links = (events as any).__links as { url: string; receive: (event: any) => void }[];
    const eventsLink = links.find(link => link.url.endsWith('/api/v1/ws/events'))!, runLink = links.find(link => link.url.endsWith('/ws/telemetry'))!;
    const hud = FakeSocket.instances.find(socket => socket.url.endsWith('/ws/telemetry'))!;
    expect(eventsLink && runLink && hud).toBeTruthy();
    // The harness is live: panels did render once on mount.
    expect(probe.SettingControl).toBeGreaterThan(0); expect(probe.TimelineWorkstation).toBeGreaterThan(0); expect(probe.FaceManager).toBeGreaterThan(0);

    const before = snapshot();
    const stages = { STAGE_1_SPARSE_PRESCAN: 'DONE', STAGE_2_ACTIVE_PREPASS: 'DONE', STAGE_3_MAIN_SWAP: 'RUNNING' };
    for (let tick = 0; tick < 90; tick++) {                                                  // 3 seconds at 30 Hz
      await act(async () => {
        eventsLink.receive({ type: 'system.telemetry', hardware: { gpus: [{ name: 'RTX 4090', total_vram_mib: 24576, free_vram_mib: 20000 - tick * 40 }], logical_cpu_count: 16 } });
        runLink.receive({ type: 'pipeline.status', stage: 'STAGE_3_MAIN_SWAP', stages, stage_progress_pct: tick, fps: 24 + tick / 10, eta_seconds: 90 - tick, vram_available_gb: 12 - tick * 0.01, skipped_frames_pct: 1.5 });
        hud.onmessage?.({ data: JSON.stringify({ type: 'job.progress', stage: 'warp', fps: 20 + (tick % 5), stage_timings_ms: { decode: 3 } }) });
      });
    }
    const after = snapshot();
    for (const name of ['App', 'VideoViewport', 'TimelineWorkstation', 'SettingControl', 'SettingsInspector', 'FaceManager'] as const) {
      expect({ name, renders: (after[name] ?? 0) - (before[name] ?? 0) }).toEqual({ name, renders: 0 });
    }
    expect(after.commits - before.commits).toBe(0);                                          // React Profiler: no commit touched a SettingControl
    // ...and the data really flowed to the subscribers that are meant to re-render.
    expect(host.querySelector('[data-testid="fps"]')?.textContent).toBe('32.9');
    expect(host.querySelector('.pipeline-pill-btn')?.textContent).toContain('FPS');
  });

  it('control: an App state change DOES re-render the panels, proving the probes and Profiler see renders', async () => {
    await mount();
    const before = snapshot();
    await act(async () => (host.querySelector('[data-testid="header-tour-button"]') as HTMLElement).click());
    const after = snapshot();
    expect((after.App ?? 0) - (before.App ?? 0)).toBeGreaterThan(0);
    expect(after.commits - before.commits).toBeGreaterThan(0);
  });
});

describe('Floating PiP preview and hotkey remapper persist across route changes', () => {
  it('keeps the draggable floating preview visible and live on every screen', async () => {
    await mount();
    await press('F', { shiftKey: true });                                                   // default floating-preview chord
    const floating = () => document.querySelector<HTMLElement>('[data-testid="floating-window"]');
    expect(floating()).toBeTruthy(); expect(pipStore.get().mode).toBe('window');
    for (const route of ['#/facemgr', '#/batch', '#/settings', '#/history', '#/gallery', '#/processing', '#/faceswap']) {
      await go(route);
      expect(floating()).toBeTruthy();
      expect(floating()!.closest('[hidden],[inert]')).toBeNull();                          // not trapped inside a hidden screen
      expect(floating()!.parentElement).toBe(document.body);
    }
    const stored = JSON.parse(localStorage.getItem('roop3d-pip-v1')!);
    expect(stored.mode).toBe('window');
    // Dragging by the title bar moves it and persists the new position.
    const bar = floating()!.querySelector<HTMLElement>('.floating-window-bar')!, startLeft = parseFloat(floating()!.style.left);
    await act(async () => bar.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true })));
    expect(parseFloat(floating()!.style.left)).toBe(Math.max(0, startLeft - 16));
    expect(JSON.parse(localStorage.getItem('roop3d-pip-v1')!).geometry.x).toBe(Math.max(0, startLeft - 16));
    await press('F', { shiftKey: true });                                                   // toggles off again
    expect(floating()).toBeNull();
  });

  it('persists a remapped hotkey across routes, rejects conflicts, and fires the remapped action', async () => {
    await mount();
    await go('#/settings');
    const field = () => host.querySelector<HTMLInputElement>('[aria-label="Shortcut for Toggle split wipe"]')!;
    expect(field().value).toBe('W');
    await press('X', {}, field());
    expect(field().value).toBe('X');
    await press('D', {}, field());                                                          // D belongs to "Toggle diff map"
    expect(host.textContent).toContain('already assigned'); expect(field().value).toBe('X');
    for (const route of ['#/facemgr', '#/batch', '#/faceswap', '#/settings']) await go(route);
    expect(field().value).toBe('X');
    expect(JSON.parse(localStorage.getItem('roop3d-hotkeys-v1')!).toggleWipe).toBe('X');

    await go('#/faceswap');
    expect(host.querySelector('.compare-host')).toBeNull();
    await press('W');                                                                       // the old chord no longer does anything
    expect(host.querySelector('.compare-host')).toBeNull();
    await press('X');
    expect(host.querySelector('.compare-host')).toBeTruthy();                                // remapped chord opens the wipe canvas
    await press('X');
    expect(host.querySelector('.compare-host')).toBeNull();
  });
});

// `ComponentType` import keeps the vi.mock factory typings honest for the lazily imported App.
export type _Unused = ComponentType;
