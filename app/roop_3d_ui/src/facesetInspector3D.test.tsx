// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import FacesetInspector3D from './components/facemgr/FacesetInspector3D';

// The R3F viewer needs WebGL: replace it with a probe that records the mesh it is handed.
vi.mock('./components/three/PoseGeometryViewer', () => ({
  default: (props: { coverage: { mesh_full_3d?: { reconstruction_level?: string } } | null }) => <div data-testid="head-viewer">{props.coverage?.mesh_full_3d?.reconstruction_level ?? 'no mesh'}</div>,
}));

const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const VIEWS = [{ index: 0, yaw: 0, pitch: 0, roll: 0 }, { index: 1, yaw: -45, pitch: 10, roll: 0 }, { index: 2, yaw: 44, pitch: -15, roll: 0 }];
const surface = () => ({ faceset_id: 'alice', views: [], bins: {},
  mesh_full_3d: { reconstruction_level: 'three_tier_facial_surface', vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], indices: [[0, 1, 2]], uvs: [], normals: [], texture_base64: '', limitations: [], backbone: { available: true } } });
const SOURCES = [{ faceset_id: 'alice', name: 'Alice' }, { faceset_id: 'bob', name: 'Bob' }];

let calls: { url: string; method: string }[] = [];
let built = false;
beforeEach(() => {
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
  calls = []; built = false; localStorage.clear();
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET'; calls.push({ url: String(url), method });
    const path = String(url).replace('/api/v1', '');
    if (/\/views$/.test(path)) return json(VIEWS);
    if (path === '/faceset/reconstruct-3d' && method === 'POST') { built = true; return json(surface()); }
    if (path.startsWith('/faceset/3d-state')) return built ? json(surface()) : json({ error: { message: 'No fitted facial surface for this FaceSet yet.' } }, 404);
    return new Response('', { status: 404 });
  }));
});
afterEach(() => { vi.unstubAllGlobals(); document.body.innerHTML = ''; });

const flush = () => act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); });

it('shows the radar for the stored views (pitch converted to the standard frame) and offers an explicit build, never building by itself', async () => {
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  try {
    await act(async () => root.render(<FacesetInspector3D sources={SOURCES} />)); await flush(); await flush();
    expect(host.querySelectorAll('.radar-cell.filled')).toHaveLength(3);
    // legacy pitch +10 is looking down: it must land in the standard pitch -10 row (0 .. -15 = row 3), not +10 (row 4).
    expect(host.querySelector('[data-cell="' + (3 * 12 + 3) + '"]')?.getAttribute('data-count')).toBe('1');          // yaw -45 starts column 3, standard pitch -10 is row 3
    expect(host.querySelector('[data-testid="head-viewer"]')).toBeNull();
    expect(host.textContent).toContain('No fitted facial surface for this FaceSet yet.');
    expect(calls.some(call => call.method === 'POST')).toBe(false);
    await act(async () => host.querySelector<HTMLButtonElement>('.fsz-inspector-placeholder .button')!.click()); await flush(); await flush(); await flush();
    expect(calls.filter(call => call.method === 'POST').map(call => call.url)).toEqual(['/api/v1/faceset/reconstruct-3d']);
    expect(host.querySelector('[data-testid="head-viewer"]')?.textContent).toBe('three_tier_facial_surface');
  } finally { await act(async () => root.unmount()); host.remove(); }
});

it('uses the facial reconstruction regardless of a saved legacy template and follows the FaceSet picker', async () => {
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  try {
    localStorage.setItem('roop3d-head-template', 'makehuman');
    await act(async () => root.render(<FacesetInspector3D sources={SOURCES} />)); await flush(); await flush();
    expect(calls.some(call => call.url === '/api/v1/faceset/3d-state?faceset_id=alice')).toBe(true);
    expect(calls.some(call => call.url.includes('identity-asset'))).toBe(false);
    expect(host.textContent).toContain('Anatomical facial surface');
    const select = host.querySelector<HTMLSelectElement>('[aria-label="FaceSet to inspect in 3D"]')!;
    await act(async () => { Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')!.set!.call(select, 'bob'); select.dispatchEvent(new Event('change', { bubbles: true })); }); await flush(); await flush();
    expect(calls.some(call => call.url === '/api/v1/facesets/bob/views')).toBe(true);
    expect(calls.some(call => call.url === '/api/v1/faceset/3d-state?faceset_id=bob')).toBe(true);
  } finally { await act(async () => root.unmount()); host.remove(); }
});

it('reports a failed read with Retry instead of an empty viewer, and an empty library plainly', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => json({ error: { message: 'FaceSet is not loaded' } }, 404)));
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  try {
    await act(async () => root.render(<FacesetInspector3D sources={SOURCES} />)); await flush(); await flush();
    expect(host.textContent).toContain('FaceSet is not loaded');
    expect([...host.querySelectorAll('button')].filter(button => button.textContent === 'Retry').length).toBeGreaterThan(0);
    await act(async () => root.render(<FacesetInspector3D sources={[]} />));
    expect(host.querySelector('[data-testid="fsz-inspector-empty"]')).toBeTruthy();
  } finally { await act(async () => root.unmount()); host.remove(); }
});
