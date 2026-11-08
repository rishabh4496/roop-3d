// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import FacesetInspector3D from './components/facemgr/FacesetInspector3D';

// The R3F viewer needs WebGL: replace it with a probe that records the mesh it is handed.
vi.mock('./components/three/PoseGeometryViewer', () => ({
  default: (props: { mesh: { provenance?: string } | null }) => <div data-testid="head-viewer">{props.mesh?.provenance ?? 'no mesh'}</div>,
}));

const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const VIEWS = [{ index: 0, yaw: 0, pitch: 0, roll: 0 }, { index: 1, yaw: -45, pitch: 10, roll: 0 }, { index: 2, yaw: 44, pitch: -15, roll: 0 }];
const status = (built: boolean, template: string) => ({ faceset_id: 'alice', name: 'Alice', state: built ? 'built' : 'not_built', template, build_key: 'k', note: '', flame: {},
  coverage: { readiness: 'partial_coverage', view_count: 3, usable_count: 3, yaw_range_deg: [-45, 44], yaw_span_deg: 89, pitch_span_deg: 25, yaw_coverage: {}, pitch_coverage: {}, missing: [], requests: [], duplicate_groups: [], capabilities: {} },
  asset: built ? { build_key: 'k', vertex_count: 4, face_count: 4 } : null });
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
    if (/identity-asset\/mesh/.test(path)) return json({ method: 'identity_asset', coordinate_system: 'opengl', vertices: [], faces: [], uvs: null, texture: null, provenance: `mesh for ${path}` });
    if (/identity-asset/.test(path)) { if (method === 'POST') built = true; return json(status(built, path.includes('makehuman') ? 'makehuman' : 'canonical'), method === 'POST' ? 201 : 200); }
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
    expect(host.textContent).toContain('No 3D canonical head for this FaceSet yet.');
    expect(calls.some(call => call.method === 'POST')).toBe(false);
    await act(async () => host.querySelector<HTMLButtonElement>('.fsz-inspector-placeholder .button')!.click()); await flush(); await flush(); await flush();
    expect(calls.filter(call => call.method === 'POST').map(call => call.url)).toEqual(['/api/v1/facesets/alice/identity-asset']);
    expect(host.querySelector('[data-testid="head-viewer"]')?.textContent).toBe('mesh for /facesets/alice/identity-asset/mesh');
  } finally { await act(async () => root.unmount()); host.remove(); }
});

it('asks for the MakeHuman template when that head model is chosen, remembers the choice, and follows the FaceSet picker', async () => {
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  try {
    await act(async () => root.render(<FacesetInspector3D sources={SOURCES} />)); await flush();
    await act(async () => [...host.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find(button => button.textContent === 'MakeHuman CC0 head')!.click()); await flush(); await flush();
    expect(calls.some(call => call.url === '/api/v1/facesets/alice/identity-asset?template=makehuman')).toBe(true);
    expect(localStorage.getItem('roop3d-head-template')).toBe('makehuman');
    expect(host.textContent).toContain('No 3D MakeHuman head for this FaceSet yet.');
    const select = host.querySelector<HTMLSelectElement>('[aria-label="FaceSet to inspect in 3D"]')!;
    await act(async () => { Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')!.set!.call(select, 'bob'); select.dispatchEvent(new Event('change', { bubbles: true })); }); await flush(); await flush();
    expect(calls.some(call => call.url === '/api/v1/facesets/bob/views')).toBe(true);
    expect(calls.some(call => call.url === '/api/v1/facesets/bob/identity-asset?template=makehuman')).toBe(true);
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
