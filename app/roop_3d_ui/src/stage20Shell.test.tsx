// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { drawerOf, isDeepLink, normalizeRoute, useHashDrawer, useHashRoute } from './router/AppRouter';
import MagneticDock, { DOCK_ITEMS, activeDockItem, magnetScale } from './shell/MagneticDock';
import ShellDrawers from './shell/ShellDrawers';
import TelemetryHeader, { GRAPH_NODES, nodeForStage, swapModeLabel } from './shell/TelemetryHeader';
import { DEFAULT_PARAMETERS } from './components/workstation/pipeline';
import { publishHardware, publishProgress, resetTelemetry } from './stores/telemetryStore';

vi.mock('./api', () => ({ api: vi.fn(() => Promise.reject(new Error('offline'))), post: vi.fn() }));
// The drawer bodies are exercised by their own tests; here only the shell around them matters.
vi.mock('./components/modelhub/ModelManager', () => ({ default: () => <p>model manager body</p> }));
vi.mock('./components/modelhub/ModelHubPanel', () => ({ default: () => <p>model hub body</p> }));
vi.mock('./components/diagnostics/RuntimeReportPanel', () => ({ default: () => <p>runtime report body</p> }));

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
const hosts: { unmount: () => Promise<void> }[] = [];
const mount = async (node: React.ReactNode) => {
  const host = document.createElement('div'); document.body.append(host);
  const root = createRoot(host);
  await act(async () => root.render(node));
  const handle = { host, unmount: async () => { await act(async () => root.unmount()); host.remove(); } };
  hosts.push(handle);
  return handle;
};
const flush = (ms = 40) => act(async () => { await new Promise(resolve => setTimeout(resolve, ms)); });
afterEach(async () => { while (hosts.length) await hosts.pop()!.unmount(); resetTelemetry(); location.hash = ''; document.body.innerHTML = ''; });

describe('deep-link routing', () => {
  it('maps the five Stage 20 links onto the existing screens and keeps every legacy route', () => {
    expect([normalizeRoute('#/studio'), normalizeRoute('#/identity-forge'), normalizeRoute('#/queue')]).toEqual(['#/faceswap', '#/facemgr', '#/processing']);
    expect([normalizeRoute('#/models'), normalizeRoute('#/diagnostics')]).toEqual(['#/faceswap', '#/faceswap']); // drawers sit over the Studio
    expect([drawerOf('#/models'), drawerOf('#/diagnostics?tab=2'), drawerOf('#/studio'), drawerOf('#/faceswap')]).toEqual(['models', 'diagnostics', null, null]);
    expect(normalizeRoute('#/studio/')).toBe('#/faceswap');
    for (const legacy of ['#/home', '#/faceswap', '#/batch', '#/processing', '#/facemgr', '#/extras', '#/gallery', '#/history', '#/settings']) expect(normalizeRoute(legacy)).toBe(legacy);
    expect(normalizeRoute('#/nope')).toBe('#/home');
    expect([isDeepLink('#/queue'), isDeepLink('#/models'), isDeepLink('#/faceswap'), isDeepLink('#/nope')]).toEqual([true, true, false, false]);
  });

  it('keeps the deep link in the address bar on load and on navigation, but still rewrites unknown hashes', async () => {
    const seen: { route: string; drawer: string | null }[] = [];
    const Probe = () => { const route = useHashRoute(); const drawer = useHashDrawer(); seen.push({ route, drawer }); return null; };
    location.hash = '#/identity-forge';
    await mount(<Probe />);
    expect(location.hash).toBe('#/identity-forge');                       // not rewritten to #/facemgr: the URL is shareable as typed
    expect(seen.at(-1)).toEqual({ route: '#/facemgr', drawer: null });
    await act(async () => { location.hash = '#/models'; window.dispatchEvent(new HashChangeEvent('hashchange')); });
    expect(location.hash).toBe('#/models');
    expect(seen.at(-1)).toEqual({ route: '#/faceswap', drawer: 'models' });
    await act(async () => { location.hash = '#/queue'; window.dispatchEvent(new HashChangeEvent('hashchange')); });
    expect(seen.at(-1)).toEqual({ route: '#/processing', drawer: null });
    await act(async () => { location.hash = '#/definitely-not-a-screen'; window.dispatchEvent(new HashChangeEvent('hashchange')); });
    expect(location.hash).toBe('#/home');
  });
});

describe('MagneticDock', () => {
  it('has one real link per deep link plus a palette button, and marks the current one', async () => {
    const onPalette = vi.fn();
    const { host } = await mount(<MagneticDock route="#/facemgr" drawer={null} onPalette={onPalette} />);
    const links = [...host.querySelectorAll<HTMLAnchorElement>('a.dock-item')];
    expect(links.map(link => [link.getAttribute('href'), link.getAttribute('aria-label')])).toEqual([
      ['#/studio', 'Studio'], ['#/identity-forge', 'Identity Forge'], ['#/queue', 'Batch Queue'], ['#/models', 'Model Hub'], ['#/diagnostics', 'Runtime Report'],
    ]);
    expect(links.filter(link => link.getAttribute('aria-current') === 'page').map(link => link.dataset.dock)).toEqual(['identity-forge']);
    expect(host.querySelector('nav')!.getAttribute('aria-label')).toBe('Quick dock');
    await act(async () => host.querySelector<HTMLButtonElement>('button.dock-palette')!.click());
    expect(onPalette).toHaveBeenCalledTimes(1);
  });

  it('lets an open drawer win over the screen behind it', () => {
    expect(activeDockItem('#/faceswap', null)).toBe('studio');
    expect(activeDockItem('#/faceswap', 'models')).toBe('models');
    expect(activeDockItem('#/processing', null)).toBe('queue');
    expect(activeDockItem('#/settings', null)).toBeNull();
    expect(DOCK_ITEMS.map(item => item.id)).toEqual(['studio', 'identity-forge', 'queue', 'models', 'diagnostics']);
  });

  it('magnifies by proximity: 1.55x under the pointer, back to 1x beyond 140px, symmetric', () => {
    expect(magnetScale(0)).toBeCloseTo(1.55);
    expect(magnetScale(140)).toBe(1);
    expect(magnetScale(500)).toBe(1);
    expect(magnetScale(-60)).toBeCloseTo(magnetScale(60));
    expect(magnetScale(30)).toBeGreaterThan(magnetScale(70));
  });
});

describe('ShellDrawers', () => {
  it('opens the Model Hub and the runtime report as labelled dialogs, and closing returns to the Studio', async () => {
    location.hash = '#/models';
    const view = await mount(<ShellDrawers drawer="models" />);
    await flush();
    const dialog = document.querySelector<HTMLElement>('[role=dialog]')!;
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    expect(document.getElementById(dialog.getAttribute('aria-labelledby')!)!.textContent).toBe('Model Hub');
    expect(dialog.textContent).toContain('model manager body'); expect(dialog.textContent).toContain('model hub body');
    await act(async () => document.querySelector<HTMLButtonElement>('.ul-drawer__close')!.click());
    expect(location.hash).toBe('#/studio');
    await view.unmount();

    location.hash = '#/diagnostics';
    await mount(<ShellDrawers drawer="diagnostics" />);
    await flush();
    const report = document.querySelector<HTMLElement>('[role=dialog]')!;
    expect(document.getElementById(report.getAttribute('aria-labelledby')!)!.textContent).toBe('Runtime report');
    expect(report.textContent).toContain('runtime report body');
  });
  it('renders nothing when no drawer is requested', async () => {
    await mount(<ShellDrawers drawer={null} />);
    expect(document.querySelector('[role=dialog]')).toBeNull();
  });
});

describe('TelemetryHeader', () => {
  it('maps engine stage names to graph nodes, and lights nothing for an unknown or idle stage', () => {
    expect(GRAPH_NODES.map(node => node.id)).toEqual(['identity', 'target', 'lbs', 'relight', 'output']);
    expect(['detection', 'landmark', 'warp', 'restoration', 'blend', 'encode'].map(nodeForStage)).toEqual(['target', 'target', 'lbs', 'relight', 'relight', 'output']);
    expect([nodeForStage('idle'), nodeForStage(''), nodeForStage(null), nodeForStage('something_new')]).toEqual([null, null, null, null]);
  });

  it('names the swap mode from the real swap scope', () => {
    expect(swapModeLabel(DEFAULT_PARAMETERS)).toBe('100% FULL CRANIOFACIAL LOCK');
    expect(swapModeLabel({ ...DEFAULT_PARAMETERS, swap_scope: 'adaptive_skull', skull_blend: 0.6 })).toBe('ADAPTIVE SKULL BLEND · 60% SOURCE');
    expect(swapModeLabel({ ...DEFAULT_PARAMETERS, swap_scope: 'inner_face' })).toBe('INNER FACE ONLY');
  });

  it('says n/a for everything the server has not reported, instead of inventing figures', async () => {
    const { host } = await mount(<TelemetryHeader parameters={DEFAULT_PARAMETERS} />);
    await flush();
    expect(host.querySelector('[data-testid=thm-gpu]')!.textContent).toContain('n/a');
    expect(host.querySelector('[data-testid=thm-trt]')!.textContent).toContain('n/a');
    expect(host.querySelector('[data-testid=thm-vram]')!.textContent).toContain('n/a');
    expect(host.querySelector('[data-testid=thm-fps] .ul-sr-only')!.textContent).toBe('—');
    expect(host.querySelector('[data-testid=thm-swap-mode]')!.textContent).toBe('100% FULL CRANIOFACIAL LOCK');
    expect(host.querySelector('[data-testid=thm-map]')!.getAttribute('aria-label')).toBe('Pipeline graph, idle');
    expect(host.querySelectorAll('.thm-node[data-active=true]')).toHaveLength(0);
  });

  it('shows the reported GPU, VRAM in use / total, live FPS and glows the node of the running stage', async () => {
    const { host } = await mount(<TelemetryHeader parameters={DEFAULT_PARAMETERS} />);
    await act(async () => {
      publishHardware({ gpus: [{ name: 'NVIDIA GeForce RTX 4090', total_vram_mib: 24576, free_vram_mib: 18432 }], logical_cpu_count: 16 });
      publishProgress({ stage: 'warp', fps: 23.46 });
    });
    await flush();
    expect(host.querySelector('[data-testid=thm-gpu]')!.textContent).toContain('NVIDIA GeForce RTX 4090');
    const meter = host.querySelector<HTMLElement>('[role=meter]')!;
    expect([meter.getAttribute('aria-valuenow'), meter.getAttribute('aria-valuemax')]).toEqual(['6', '24']);        // 6144 of 24576 MiB in use
    expect(meter.getAttribute('aria-valuetext')).toBe('6.0 of 24.0 GB in use');
    expect(host.querySelector<HTMLElement>('.thm-bar i')!.style.width).toBe('25%');
    expect(host.querySelector('[data-testid=thm-fps] .ul-sr-only')!.textContent).toBe('23.5');
    expect([...host.querySelectorAll('.thm-node[data-active=true]')].map(node => (node as HTMLElement).dataset.node)).toEqual(['lbs']);
    expect(host.querySelector('[data-testid=thm-map]')!.getAttribute('aria-label')).toBe('Pipeline graph, running: warp');
    await act(async () => publishProgress({ stage: 'restoration', fps: 20 }));
    expect([...host.querySelectorAll('.thm-node[data-active=true]')].map(node => (node as HTMLElement).dataset.node)).toEqual(['relight']);
  });

  it('every graph node is a link, so the graph is navigable by keyboard', async () => {
    const { host } = await mount(<TelemetryHeader parameters={DEFAULT_PARAMETERS} />);
    expect([...host.querySelectorAll<HTMLAnchorElement>('a.thm-node')].map(node => node.getAttribute('href'))).toEqual(['#/identity-forge', '#/studio', '#/studio', '#/studio', '#/queue']);
  });
});
