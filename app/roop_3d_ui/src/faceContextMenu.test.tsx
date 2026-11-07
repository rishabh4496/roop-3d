// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { VideoViewport } from './VideoViewport';
import { DEFAULT_PARAMETERS } from './components/workstation/pipeline';
import type { JobRequest } from './contracts';
import FaceContextMenu from './components/canvas/FaceContextMenu';
import { useFaceExtraction } from './components/canvas/useFaceExtraction';

// The overlay and tray are replaced by probes that call the real callbacks VideoViewport hands them.
vi.mock('./components/canvas/FaceOverlayCanvas', () => ({
  default: (props: any) => <button data-testid="face" onContextMenu={event => {
    event.preventDefault();
    props.onFaceContextMenu?.({ face: { id: 'f' }, faceRank: 1, pose: { yaw: 28, pitch: -16, roll: 0 }, x: 200, y: 120 });
  }}>face</button>,
  calculateLetterbox: () => ({}), mapBoxToPixels: () => ({}), mapPointToPixels: () => ({}),
}));
vi.mock('./components/tray/FaceSelectionTray', () => ({
  default: (props: any) => <button data-testid="tray-action" disabled={props.canUseFaceFromFrame === false} onClick={() => props.onUseFaceFromFrame?.({ x: 40, y: 60 })}>Use Face From Frame</button>,
}));
vi.mock('./components/canvas/WorkstationCanvas', () => ({ default: () => null }));
vi.mock('./components/canvas/MaskBrushOverlay', () => ({ default: () => null }));
vi.mock('./components/timeline/TimelineWorkstation', () => ({ default: () => null, parseFps: () => 25 }));

const REQUEST: JobRequest = { target_path: 'roop_3d/.runtime/media/t.mp4', inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }],
  processing: 'local', remote_consent: false, trim_start_seconds: null, trim_end_seconds: null };
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const PNG = () => new Response(new Uint8Array([137, 80, 78, 71]), { status: 200, headers: { 'Content-Type': 'image/png', 'X-Preview-Meta': JSON.stringify({ frame_index: 5, solo_node: 'final', faces: [], applied: [], unsupported: [], notes: [], latency: {} }) } });

const face = (cell: string) => ({ face_id: 'face_1', frame_index: 5, face_index: 1, pose: { yaw: 28, pitch: -16, roll: 0 }, pose_cell: cell, embedding_dim: 512, thumbnail_base64: '' });
const REPLIES: Record<string, unknown> = {
  source_gallery: { destination: 'source_gallery', face: face('right_down'), active_source: { faceset_id: 'Frame_ab12', name: 'Frame 5 · face 1' } },
  active_faceset: { destination: 'active_faceset', face: face('right_down'), faceset: { faceset_id: 'alice', name: 'Alice', item_count: 4 }, inserted: { view_index: 3, pose_cell: 'right_down' } },
  target_cluster: { destination: 'target_cluster', face: face('right_down'), cluster: { rank: 1, created: true } },
};

describe('"Use Face From Frame" on-canvas context menu', () => {
  let host: HTMLDivElement, root: Root, fetchMock: ReturnType<typeof vi.fn>;
  const extractCalls = () => fetchMock.mock.calls.filter(([url]) => String(url).endsWith('/faces/extract_from_frame')).map(([, init]) => JSON.parse(init.body));
  const previewCalls = () => fetchMock.mock.calls.filter(([url]) => url === '/api/preview').length;
  const menuItems = () => [...host.ownerDocument.querySelectorAll<HTMLButtonElement>('.face-context-menu [role="menuitem"]')];
  const mount = async (props: Record<string, unknown> = {}) => {
    await act(async () => root.render(<VideoViewport target="roop_3d/.runtime/media/t.mp4" output="" time={0.2} setTime={() => {}} duration={4} enabled keys={[]}
      previewRequest={REQUEST} parameters={DEFAULT_PARAMETERS} restorerAvailable={false} {...props}/>));
    await act(async () => { await vi.advanceTimersByTimeAsync(250); });
  };
  const rightClickFace = () => act(async () => { host.querySelector('[data-testid="face"]')!.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true })); });
  const choose = (text: string) => act(async () => { menuItems().find(item => item.textContent!.includes(text))!.click(); await vi.advanceTimersByTimeAsync(0); });

  beforeEach(() => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
    vi.useFakeTimers();
    vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
    URL.createObjectURL = vi.fn(() => 'blob:menu'); URL.revokeObjectURL = vi.fn();
    fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      if (url === '/api/preview') return PNG();
      if (String(url).endsWith('/faces/extract_from_frame')) return json(REPLIES[JSON.parse(String(init.body)).destination], 201);
      return new Response('', { status: 404 });
    });
    vi.stubGlobal('fetch', fetchMock);
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); document.body.innerHTML = ''; vi.useRealTimers(); vi.unstubAllGlobals(); });

  it('right-clicking a face opens a menu with the three extraction actions and the four Stage 12 target actions, labelled with the face\'s pose cell', async () => {
    await mount({ activeSourceId: 'alice', sources: [{ faceset_id: 'alice', name: 'Alice', item_count: 3 }] });
    expect(menuItems()).toHaveLength(0);
    await rightClickFace();
    const items = menuItems();
    expect(host.ownerDocument.querySelector('.face-context-menu')!.getAttribute('role')).toBe('menu');
    expect(items.map(item => item.textContent!.replace(/\s+/g, ' ').replace(/^\S+ /, '').trim())).toEqual([
      'Use as Active Source Face',
      'Add to .fsz Faceset (Auto-Pose Cell [Yaw +28°, Pitch -16°] → right_down)',
      'Lock as Target Identity Cluster',
      'Lock as Primary Target',
      'Add This Angle to Target Reference BankLock a primary target first',
      'Bind Specific Source / .fsz to This Person',
      'Exclude / Ignore This Face in Current Segment',
    ]);
    expect(items.filter(item => item.disabled).map(item => item.textContent)).toEqual([expect.stringContaining('Add This Angle')]);   // needs a lock first
    expect(host.ownerDocument.querySelector('.face-menu-title')!.textContent).toContain('Face 2');
  });

  it('"Use as Active Source Face" dispatches the POST, updates the source gallery and refreshes the swapped preview', async () => {
    const onSourceExtracted = vi.fn();
    await mount({ onSourceExtracted });
    const previewsBefore = previewCalls();
    await rightClickFace();
    await choose('Use as Active Source Face');
    expect(extractCalls()).toEqual([{ target_media_path: 'roop_3d/.runtime/media/t.mp4', frame_index: 5, face_index: 1, destination: 'source_gallery' }]);
    expect(fetchMock.mock.calls.find(([url]) => String(url).endsWith('/faces/extract_from_frame'))![0]).toBe('/api/v1/faces/extract_from_frame');
    expect(onSourceExtracted).toHaveBeenCalledWith({ faceset_id: 'Frame_ab12', name: 'Frame 5 · face 1', item_count: 1 });
    expect(host.querySelector('.face-toast.ok')!.textContent).toBe('Active source face set: Frame 5 · face 1 (right / down)');
    expect(menuItems()).toHaveLength(0);                                   // the menu closes
    await act(async () => { await vi.advanceTimersByTimeAsync(250); });
    expect(previewCalls()).toBeGreaterThan(previewsBefore);                // the live swap re-renders with the new identity
    await act(async () => { await vi.advanceTimersByTimeAsync(4100); });
    expect(host.querySelector('.face-toast')).toBeNull();                  // the toast clears itself
  });

  it('"Add to .fsz Faceset" extends the active source at the auto-detected pose cell', async () => {
    const onSourceExtracted = vi.fn();
    await mount({ onSourceExtracted, activeSourceId: 'alice', sources: [{ faceset_id: 'alice', name: 'Alice', item_count: 3 }] });
    await rightClickFace();
    await choose('Add to .fsz Faceset');
    expect(extractCalls()).toEqual([{ target_media_path: 'roop_3d/.runtime/media/t.mp4', frame_index: 5, face_index: 1, destination: 'active_faceset', faceset_name: 'alice' }]);
    expect(onSourceExtracted).toHaveBeenCalledWith({ faceset_id: 'alice', name: 'Alice', item_count: 4 });
    expect(host.querySelector('.face-toast')!.textContent).toBe('Added to Alice in cell right / down (4 faces)');
  });

  it('"Lock as Target Identity Cluster" pins the face', async () => {
    await mount();
    await rightClickFace();
    await choose('Lock as Target Identity Cluster');
    expect(extractCalls()).toEqual([{ target_media_path: 'roop_3d/.runtime/media/t.mp4', frame_index: 5, face_index: 1, destination: 'target_cluster' }]);
    expect(host.querySelector('.face-toast')!.textContent).toBe('Locked as target identity #2');
  });

  it('disables "Add to .fsz Faceset" until a source FaceSet is active, and never posts for it', async () => {
    await mount();
    await rightClickFace();
    const add = menuItems().find(item => item.textContent!.includes('Add to .fsz Faceset'))!;
    expect(add.disabled).toBe(true);
    expect(add.textContent).toContain('Select a source FaceSet first');
    add.click();
    expect(extractCalls()).toEqual([]);
  });

  it('the Target Faces bar action opens the same menu', async () => {
    await mount({ activeSourceId: 'alice', sources: [{ faceset_id: 'alice', name: 'Alice', item_count: 3 }] });
    await act(async () => (host.querySelector('[data-testid="tray-action"]') as HTMLButtonElement).click());
    expect(menuItems()).toHaveLength(7);
    const style = (host.ownerDocument.querySelector('.face-context-menu') as HTMLElement).style;
    expect([style.left, style.top]).toEqual(['40px', '60px']);             // anchored at the button
  });

  it('reports a failed extraction as an error toast and keeps the workspace unchanged', async () => {
    fetchMock.mockImplementation(async (url: string) => String(url).endsWith('/faces/extract_from_frame')
      ? json({ error: { code: 'FACE_NOT_FOUND', message: 'Frame has 1 detected face(s); face_index 1 is out of range' } }, 422) : PNG());
    const onSourceExtracted = vi.fn();
    await mount({ onSourceExtracted });
    await rightClickFace();
    await choose('Use as Active Source Face');
    expect(onSourceExtracted).not.toHaveBeenCalled();
    expect(host.querySelector('.face-toast.error')!.textContent).toContain('face_index 1 is out of range');
  });

  it('closes on Escape and on an outside click without posting anything', async () => {
    await mount();
    await rightClickFace();
    await act(async () => { document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' })); });
    expect(menuItems()).toHaveLength(0);
    await rightClickFace();
    expect(menuItems()).toHaveLength(7);
    await act(async () => { document.body.dispatchEvent(new Event('pointerdown', { bubbles: true })); });
    expect(menuItems()).toHaveLength(0);
    expect(extractCalls()).toEqual([]);
  });
});

describe('FaceContextMenu keyboard and extraction hook', () => {
  it('moves focus with the arrow keys, skipping disabled items', async () => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
    const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
    try {
      await act(async () => root.render(<FaceContextMenu target={{ x: 10, y: 10, faceRank: 0, pose: { yaw: 0, pitch: 0, roll: 0 } }} activeSourceName={null} onChoose={() => {}} onClose={() => {}}/>));
      const items = [...document.querySelectorAll<HTMLButtonElement>('[role="menuitem"]')];
      expect(document.activeElement).toBe(items[0]);
      await act(async () => { document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown' })); });
      expect(document.activeElement).toBe(items[2]);                       // the disabled "Add to .fsz Faceset" is skipped
      await act(async () => { document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp' })); });
      expect(document.activeElement).toBe(items[0]);
    } finally { await act(async () => root.unmount()); host.remove(); }
  });

  it('is single-flight: a second request while one is running is ignored', async () => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
    let release: (value: any) => void = () => {};
    const extract = vi.fn(() => new Promise<any>(resolve => { release = resolve; }));
    let api: ReturnType<typeof useFaceExtraction> | null = null;
    function Probe() { api = useFaceExtraction({ target: 't.mp4', frameIndex: 1, activeSourceId: null, extract }); return null; }
    const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
    try {
      await act(async () => root.render(<Probe/>));
      let first!: Promise<unknown>;
      await act(async () => { first = api!.run('source_gallery', { faceRank: 0 }); });
      await act(async () => { await api!.run('source_gallery', { faceRank: 0 }); });
      expect(extract).toHaveBeenCalledTimes(1);
      await act(async () => { release({ destination: 'source_gallery', face: { pose_cell: 'center_level' }, active_source: { faceset_id: 'x', name: 'X' } }); await first; });
      expect(api!.busy).toBe(false);
    } finally { await act(async () => root.unmount()); host.remove(); }
  });
});
