// @vitest-environment jsdom
import { act, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import { VideoViewport } from './VideoViewport';
import { DEFAULT_PARAMETERS, PIPELINE_NODES } from './components/workstation/pipeline';
import type { JobRequest } from './contracts';
import useHistoryStore from './stores/useHistoryStore';

vi.mock('./components/canvas/FaceOverlayCanvas', () => ({ default: () => null }));
vi.mock('./components/canvas/WorkstationCanvas', () => ({ default: () => <div>Comparison canvas</div> }));
vi.mock('./components/canvas/MaskBrushOverlay', () => ({ default: () => null }));
vi.mock('./components/tray/FaceSelectionTray', () => ({ default: () => null }));
vi.mock('./components/timeline/TimelineWorkstation', () => ({ default: () => <div>Timeline</div>, parseFps: () => 25 }));

const REQUEST: JobRequest = { target_path: 'local.mp4', inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }], processing: 'local', remote_consent: false, trim_start_seconds: null, trim_end_seconds: null };
const png = () => new Uint8Array([137, 80, 78, 71]);
const previewReply = (frame: number, soloNode = 'final') => new Response(png(), { status: 200, headers: { 'Content-Type': 'image/png', 'X-Preview-Cache': 'MISS', 'X-Preview-Meta': JSON.stringify({ frame_index: frame, solo_node: soloNode, faces: [], applied: [], unsupported: [], notes: ['Matte uses the reference BiSeNet parser.'], latency: {} }) } });
const previewBodies = (fetch: ReturnType<typeof vi.fn>) => fetch.mock.calls.filter(([url]) => url === '/api/preview').map(([, init]) => JSON.parse(init.body));

it('solos a pipeline node through POST /api/preview solo_node, returns to the composite, and discards stale frame results', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
  let urls = 0; URL.createObjectURL = vi.fn(() => `blob:solo-${++urls}`); URL.revokeObjectURL = vi.fn();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  vi.useFakeTimers();
  const replies: Array<{ resolve: (value: Response) => void; signal: AbortSignal }> = [];
  const fetch = vi.fn((url: string, init: RequestInit) => url === '/api/preview'
    ? new Promise<Response>(resolve => replies.push({ resolve, signal: init.signal as AbortSignal }))
    : Promise.resolve(new Response('', { status: 404 })));
  vi.stubGlobal('fetch', fetch);
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  const props = { target: '', output: '', setTime: vi.fn(), duration: 3, enabled: true, keys: [], previewRequest: REQUEST, parameters: DEFAULT_PARAMETERS };
  try {
    await act(async () => root.render(<VideoViewport {...props} time={1}/>));
    await act(async () => { await vi.advanceTimersByTimeAsync(200); });
    expect(previewBodies(fetch)[0].solo_node).toBeUndefined();              // not soloed: the final composite
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Solo Occlusion & Seam Mask"]')!.click());
    await act(async () => { await vi.advanceTimersByTimeAsync(200); });
    expect(previewBodies(fetch).at(-1)).toMatchObject({ solo_node: 'matte', frame_index: 25 });
    await act(async () => { replies.at(-1)!.resolve(previewReply(25, 'matte')); await vi.advanceTimersByTimeAsync(0); });
    expect(host.querySelector('.solo-viewport img')?.getAttribute('src')).toMatch(/^blob:solo-/);
    expect(host.querySelector('.solo-viewport .pane-label')?.textContent).toContain('Occlusion & Seam Mask');
    expect(host.textContent).toContain('Matte uses the reference BiSeNet parser.');
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Solo Occlusion & Seam Mask"]')!.click());
    expect(host.querySelector('.solo-viewport')).toBeNull();
    await act(async () => root.render(<VideoViewport {...props} time={2}/>));
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Solo Raw Swap"]')!.click());
    await act(async () => { await vi.advanceTimersByTimeAsync(200); });
    const stale = replies.at(-1)!;
    expect(previewBodies(fetch).at(-1)).toMatchObject({ solo_node: 'swapper' });
    await act(async () => root.render(<VideoViewport {...props} time={3}/>));
    expect(stale.signal.aborted).toBe(true);
    await act(async () => { stale.resolve(previewReply(50, 'swapper')); await vi.advanceTimersByTimeAsync(0); });
    expect(host.querySelector('.solo-viewport img')).toBeNull();            // the aborted frame's late answer is never shown
  } finally { await act(async () => root.unmount()); host.remove(); vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); }
});

it('offers exactly the six pipeline stages and maps each to its server buffer', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  try {
    await act(async () => root.render(<VideoViewport target="" output="" time={0} setTime={() => {}} duration={0} enabled={false} keys={[]} previewRequest={REQUEST}/>));
    const names = [...host.querySelectorAll('.pipeline-node .node-select')].map(node => node.textContent?.replace(/^\d{2}/, ''));
    expect(names).toEqual(['Input', '3D Pose & TPS Grid', '3D Normal Map', '3D Relighting (SH)', 'Raw Swap', 'Occlusion & Seam Mask', 'Restorer', 'Final Output']);
    expect(PIPELINE_NODES.map(node => node.buffer)).toEqual(['detector', 'pose', 'normal_map', 'sh_lighting', 'swapper', 'matte', 'restorer', 'final']);
  } finally { await act(async () => root.unmount()); host.remove(); vi.unstubAllGlobals(); }
});

it('captures a rendered final composite and full settings into a snapshot slot', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); useHistoryStore.getState().reset();
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  const fetch = vi.fn(async () => previewReply(25));
  vi.stubGlobal('fetch', fetch);
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  const workspace = { parameters: DEFAULT_PARAMETERS, maskBase64: 'mask-state', curves: { swap_weight: [] }, faces: { assignments: {} } };
  try {
    await act(async () => root.render(<VideoViewport target="" output="" time={1} setTime={() => {}} duration={2} enabled={false} keys={[]} previewRequest={REQUEST} captureWorkspace={() => workspace}/>));
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Capture snapshot A"]')!.click());
    await vi.waitFor(() => expect(useHistoryStore.getState().slots.A).toBeTruthy());
    expect(previewBodies(fetch)).toHaveLength(1);
    expect(previewBodies(fetch)[0]).toMatchObject({ solo_node: 'final', frame_index: 25 });
    expect(useHistoryStore.getState().slots.A).toMatchObject({ frame: 25, workspace });
    expect(useHistoryStore.getState().slots.A!.image).toMatch(/^data:/);     // self-contained: survives the preview URL being revoked
    expect(host.querySelector('[aria-label="Snapshot A"]')).toBeTruthy();
  } finally { await act(async () => root.unmount()); host.remove(); vi.unstubAllGlobals(); vi.restoreAllMocks(); }
});

it('offers mesh blending and an explicitly separate reference retarget mode', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); useHistoryStore.getState().reset();
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  function Workstation() { const [parameters, setParameters] = useState(DEFAULT_PARAMETERS); return <VideoViewport target="" output="" time={0} setTime={() => {}} duration={0} enabled={false} keys={[]} parameters={parameters} setParameters={setParameters}/>; }
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  try {
    await act(async () => root.render(<Workstation/>));
    expect(host.querySelector('[aria-label="2.5D → 3D pose blend"]')).toBeTruthy();
    const select = host.querySelector<HTMLSelectElement>('[aria-label="Pose correction method"]')!;
    await act(async () => { Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')!.set!.call(select, 'retarget'); select.dispatchEvent(new Event('change', { bubbles: true })); });
    expect(host.querySelector('[aria-label="Reference retarget strength"]')).toBeTruthy();
  } finally { await act(async () => root.unmount()); host.remove(); vi.unstubAllGlobals(); vi.restoreAllMocks(); }
});

it('solos 3D Normal Map and 3D Relighting nodes through POST /api/preview', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  vi.useFakeTimers();
  const fetch = vi.fn(async () => previewReply(25, 'normal_map'));
  vi.stubGlobal('fetch', fetch);
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  try {
    await act(async () => root.render(<VideoViewport target="" output="" time={1} setTime={() => {}} duration={3} enabled={true} keys={[]} previewRequest={REQUEST} parameters={DEFAULT_PARAMETERS}/>));
    await act(async () => { await vi.advanceTimersByTimeAsync(200); });
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Solo 3D Normal Map"]')!.click());
    await act(async () => { await vi.advanceTimersByTimeAsync(200); });
    expect(previewBodies(fetch).at(-1)).toMatchObject({ solo_node: 'normal_map' });
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Solo 3D Relighting (SH)"]')!.click());
    await act(async () => { await vi.advanceTimersByTimeAsync(200); });
    expect(previewBodies(fetch).at(-1)).toMatchObject({ solo_node: 'sh_lighting' });
  } finally { await act(async () => root.unmount()); host.remove(); vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); }
});
