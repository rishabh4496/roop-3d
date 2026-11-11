// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { FaceSet, Job, Track } from './contracts';
import { buildLivePayload } from './livePreview';
import { DEFAULT_PARAMETERS, type WorkstationParameters } from './components/workstation/pipeline';
import { FACESET_MIME, carriesFaceset, droppedFaceset, facesetDropProps } from './utils/facesetDrag';
import FaceSetDetail, { coverageGrid, isAssetStatus, isViewList } from './components/studio/FaceSetDetail';
import TrackLanes, { renderFraction, trackSpans } from './components/studio/TrackLanes';
import { FrameNavigator, trackLabel } from './components/studio/TargetPanel';
import StudioInspector from './components/studio/StudioInspector';
import PanelBoundary from './components/studio/PanelBoundary';
import { engineSelection } from './stores/engineSelection';
import { setPlayhead } from './stores/playheadStore';

const api = vi.fn(), post = vi.fn();
vi.mock('./api', () => ({ api: (...args: unknown[]) => api(...args), post: (...args: unknown[]) => post(...args) }));
vi.mock('./ThreeViewport', () => ({ PoseGeometryViewer: ({ mesh }: { mesh: unknown }) => <div data-testid="viewer">{mesh ? 'mesh' : 'estimate'}</div>, LightProbeSphere: () => <div data-testid="light-probe" /> }));
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

let host: HTMLDivElement, root: Root;
beforeEach(() => { api.mockReset(); post.mockReset(); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.restoreAllMocks(); });
const flush = () => act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); });
const render = (node: React.ReactNode) => act(async () => root.render(node));
const byText = (text: string) => Array.from(host.querySelectorAll('button')).find(item => item.textContent?.includes(text)) as HTMLButtonElement | undefined;
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(r => { resolve = r; }); return { promise, resolve }; };

const transfer = (types: string[], data: Record<string, string> = {}) => ({ types, getData: (type: string) => data[type] ?? '', setData: vi.fn(), dropEffect: 'none', effectAllowed: 'all' });
const dragEvent = (types: string[], data: Record<string, string> = {}) => ({ dataTransfer: transfer(types, data), preventDefault: vi.fn(), stopPropagation: vi.fn() }) as any;

describe('FaceSet drag payload', () => {
  it('accepts only a well-formed FaceSet drag and ignores files or text', () => {
    expect(carriesFaceset(dragEvent([FACESET_MIME]))).toBe(true);
    expect(carriesFaceset(dragEvent(['Files']))).toBe(false);
    expect(droppedFaceset(dragEvent([FACESET_MIME], { [FACESET_MIME]: 'alice_01' }))).toBe('alice_01');
    expect(droppedFaceset(dragEvent([FACESET_MIME], { [FACESET_MIME]: '../etc/passwd' }))).toBeNull();
    expect(droppedFaceset(dragEvent(['text/plain'], { 'text/plain': 'alice' }))).toBeNull();
    const onDrop = vi.fn(), setOver = vi.fn(), props = facesetDropProps(onDrop, setOver);
    const over = dragEvent([FACESET_MIME]); props.onDragOver(over);
    expect(over.preventDefault).toHaveBeenCalled(); expect(setOver).toHaveBeenLastCalledWith(true);
    const fileOver = dragEvent(['Files']); props.onDragOver(fileOver);
    expect(fileOver.preventDefault).not.toHaveBeenCalled();
    props.onDrop(dragEvent([FACESET_MIME], { [FACESET_MIME]: 'bob' }));
    expect(onDrop).toHaveBeenCalledWith('bob'); expect(setOver).toHaveBeenLastCalledWith(false);
    props.onDrop(dragEvent(['Files']));
    expect(onDrop).toHaveBeenCalledTimes(1);
  });
});

describe('identity panel model', () => {
  it('bins stored poses into the yaw x pitch indicator in the standard pitch frame', () => {
    // stored pitch + = looking down; -15 stored = looking UP
    const { grid, covered, total } = coverageGrid([{ index: 0, yaw: 0, pitch: 0, roll: 0 }, { index: 1, yaw: 3, pitch: 2, roll: 0 }, { index: 2, yaw: -70, pitch: -15, roll: 0 }, { index: 3, yaw: 40, pitch: 20, roll: 0 }]);
    expect(total).toBe(15); expect(covered).toBe(3);
    expect(grid[1][2]).toBe(2);   // level / front
    expect(grid[0][0]).toBe(1);   // up / far left
    expect(grid[2][3]).toBe(1);   // down / right
  });
  it('rejects malformed responses instead of rendering them', () => {
    expect(isViewList([{ index: 0, yaw: 1, pitch: 2, roll: 0 }])).toBe(true);
    expect(isViewList({})).toBe(false);
    expect(isViewList([{ index: 'a' }])).toBe(false);
    expect(isAssetStatus([])).toBe(false);
    expect(isAssetStatus({ state: 'built', coverage: { readiness: 'x', requests: [] }, asset: null })).toBe(false);
    expect(isAssetStatus({ state: 'not_built', coverage: { readiness: 'front_only', requests: [] }, asset: null })).toBe(true);
  });
});

const STATUS = (built: boolean) => ({ faceset_id: 'alice', name: 'Alice', state: built ? 'built' : 'not_built', build_key: 'k', note: '', flame: {},
  coverage: { readiness: 'front_only', view_count: 1, usable_count: 1, yaw_range_deg: [0, 0], yaw_span_deg: 0, pitch_span_deg: 0, yaw_coverage: {}, pitch_coverage: {}, missing: [], requests: ['Add a left profile'], duplicate_groups: [], capabilities: {} },
  asset: built ? { build_key: 'k', created_at: '', algorithm_version: '', reconstruction_level: 'single_view', landmark_layout: '5pt', claims: {}, missing_information: [], limitations: [], statistics: {}, sources_used: 1, sources_total: 1, vertex_count: 10, face_count: 12, texture_size: [8, 8] } : null });
const SOURCES: FaceSet[] = [{ faceset_id: 'alice', name: 'Alice', item_count: 1 }, { faceset_id: 'bob', name: 'Bob', item_count: 2 }];
const detail = (faceset: FaceSet, extra: Record<string, unknown> = {}) => <FaceSetDetail faceset={faceset} sources={SOURCES} selectedTrackLabel="Face 0" assignedHere={false} onAssign={vi.fn()}
  identitySourceB="" onIdentitySourceB={vi.fn()} onShowMesh={vi.fn()} meshShown={false} {...extra} />;

describe('FaceSetDetail', () => {
  it('never shows a late answer for a FaceSet that is no longer selected', async () => {
    const slow = deferred<unknown>();
    api.mockImplementation((path: string) => path === '/facesets/alice/views' ? slow.promise
      : path.endsWith('/views') ? Promise.resolve([{ index: 0, yaw: 0, pitch: 0, roll: 0 }, { index: 1, yaw: 45, pitch: 0, roll: 0 }])
        : Promise.resolve(STATUS(false)));
    await render(detail(SOURCES[0]));
    await render(detail(SOURCES[1]));
    await flush();
    expect(host.querySelectorAll('.view-tile')).toHaveLength(2);
    slow.resolve([{ index: 0, yaw: 0, pitch: 0, roll: 0 }, { index: 1, yaw: 0, pitch: 0, roll: 0 }, { index: 2, yaw: 0, pitch: 0, roll: 0 }]);
    await flush();
    expect(host.querySelectorAll('.view-tile')).toHaveLength(2);
    expect(host.querySelector('[aria-label="Bob views"]')).toBeTruthy();
  });

  it('builds only on request, then offers the 3D view; failures show the reason and a retry', async () => {
    let built = false;
    api.mockImplementation((path: string) => path.endsWith('/views') ? Promise.resolve([]) : path.endsWith('/mesh') ? Promise.resolve({ method: 'identity_asset' }) : Promise.resolve(STATUS(built)));
    post.mockRejectedValueOnce(new Error('FaceSet contains no views to analyse'));
    const onShowMesh = vi.fn();
    await render(detail(SOURCES[0], { onShowMesh }));
    await flush();
    expect(post).not.toHaveBeenCalled();
    expect(host.textContent).toContain('Not built yet');
    expect(host.textContent).toContain('1 suggested additional angle');
    await act(async () => byText('Build 3D identity asset')!.click());
    await flush();
    expect(host.querySelector('.error-retry')?.textContent).toContain('FaceSet contains no views to analyse');
    post.mockImplementationOnce(async () => { built = true; return STATUS(true); });
    await act(async () => byText('Build 3D identity asset')!.click());
    await flush(); await flush();
    expect(post).toHaveBeenLastCalledWith('/facesets/alice/identity-asset', {});
    expect(host.querySelector('.asset-facts')?.textContent).toContain('single_view');
    await act(async () => byText('View 3D asset')!.click());
    await flush();
    expect(onShowMesh).toHaveBeenCalledWith({ method: 'identity_asset' });
  });

  it('bakes and links the four local PBR maps at the selected resolution', async () => {
    api.mockImplementation((path: string) => Promise.resolve(path.endsWith('/views') ? [] : STATUS(false)));
    post.mockResolvedValue({ faceset_id: 'alice', build_key: 'k', resolution: 2048, source_resolution: [512, 512], asset_reused: true,
      albedo_source: 'observed multi-view identity asset', trust_mask: 'PNG alpha', relief_available: true,
      maps: { basecolor: '/atlas/albedo.png', normal: '/atlas/normal.png', orm: '/atlas/orm.png', cavity: '/atlas/cavity.png' }, manifest: 'atlas_2k_manifest.json', claims: {} });
    await render(detail(SOURCES[0]));
    await flush();
    await act(async () => byText('Bake PBR atlas')!.click());
    await flush();
    expect(post).toHaveBeenCalledWith('/facesets/alice/bake-atlas', { resolution: 2048 });
    expect(host.querySelectorAll('.atlas-downloads a')).toHaveLength(4);
    expect(host.querySelector('.atlas-downloads')?.textContent).toContain('measured UV relief included');
  });

  it('assignment needs a selected target track', async () => {
    api.mockImplementation((path: string) => Promise.resolve(path.endsWith('/views') ? [] : STATUS(false)));
    await render(detail(SOURCES[0], { selectedTrackLabel: null }));
    expect(byText('Select a target track to assign')!.disabled).toBe(true);
    const onAssign = vi.fn();
    await render(detail(SOURCES[0], { onAssign }));
    await act(async () => byText('Assign to Face 0')!.click());
    expect(onAssign).toHaveBeenCalledTimes(1);
  });
});

const TRACK = (id: string, first: number, last: number): Track => ({ track_id: id, first_frame: first, last_frame: last, first_timestamp_seconds: first / 25, last_timestamp_seconds: last / 25, thumbnail_base64: '', thumbnail_mime: 'image/jpeg', landmark_count: 106,
  pose_range: { yaw_degrees: [-10, 12], pitch_degrees: [0, 3], roll_degrees: [-1, 1], method: 'five_point_geometric_estimate' } });
const job = (state: Job['state'], frame: number, total: number | null): Job => ({ job_id: 'j', state, request: { target_path: 't', inspection_id: 'i', face_mappings: [], processing: 'local', remote_consent: false, trim_start_seconds: 1, trim_end_seconds: 3 },
  progress: { frame, total_frames: total, fps: 0, stage: 'x' }, output: null, failure: null });

describe('TrackLanes', () => {
  it('derives spans and render progress from real data only', () => {
    expect(trackSpans(TRACK('track_0000', 5, 40))).toEqual([[5, 40]]);
    expect(trackSpans(TRACK('track_0000', 5, 40), { segments: [[5, 9, 0, 0], [20, 40, 0, 0]] } as any)).toEqual([[5, 9], [20, 40]]);
    expect(renderFraction(null)).toBeNull();
    expect(renderFraction(job('running', 25, 50))).toBe(0.5);
    expect(renderFraction(job('running', 25, null))).toBeNull();
    expect(renderFraction(job('succeeded', 0, null))).toBe(1);
  });

  it('seeks, selects, accepts dropped FaceSets and reports render progress', async () => {
    const onSeek = vi.fn(), onSelectTrack = vi.fn(), onAssign = vi.fn();
    await render(<TrackLanes tracks={[TRACK('track_0000', 0, 49), TRACK('track_0001', 50, 99)]} sheet={null} sceneCuts={[50]} fps={25} duration={4} trimStart={0} trimEnd={null}
      job={job('running', 10, 50)} selectedTrack="track_0001" assigned={{ track_0001: 'bob' }} sources={SOURCES} onSelectTrack={onSelectTrack} onSeek={onSeek} onAssign={onAssign} />);
    expect(host.querySelector('[data-lane-track="track_0001"]')!.className).toContain('selected');
    expect(host.querySelector('[data-lane-track="track_0001"] .lane-label')!.textContent).toContain('Bob');
    expect(host.querySelector('[role=progressbar]')!.getAttribute('aria-valuenow')).toBe('20');
    await act(async () => (host.querySelector('[aria-label="Scene cut at frame 50"]') as HTMLButtonElement).click());
    expect(onSeek).toHaveBeenLastCalledWith(2);
    const body = host.querySelector('[data-lane-track="track_0000"] .lane-body') as HTMLElement;
    body.getBoundingClientRect = () => ({ left: 100, width: 400, top: 0, height: 18, right: 500, bottom: 18, x: 100, y: 0, toJSON: () => ({}) });
    await act(async () => body.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: 200 })));
    expect(onSelectTrack).toHaveBeenLastCalledWith('track_0000');
    expect(onSeek).toHaveBeenLastCalledWith(1);
    const lane = host.querySelector('[data-lane-track="track_0000"]') as HTMLElement;
    const drop = new Event('drop', { bubbles: true, cancelable: true });
    Object.assign(drop, { dataTransfer: transfer([FACESET_MIME], { [FACESET_MIME]: 'alice' }) });
    await act(async () => lane.dispatchEvent(drop));
    expect(onAssign).toHaveBeenCalledWith('track_0000', 'alice');
  });

  it('renders nothing without a duration (no fake timeline)', async () => {
    await render(<TrackLanes tracks={[]} sheet={null} sceneCuts={[]} fps={25} duration={0} trimStart={0} trimEnd={null} job={null} selectedTrack="" assigned={{}} sources={[]} onSelectTrack={vi.fn()} onSeek={vi.fn()} onAssign={vi.fn()} />);
    expect(host.innerHTML).toBe('');
  });
});

describe('FrameNavigator', () => {
  it('steps exactly one frame and commits typed frame numbers', async () => {
    const setTime = vi.fn();
    setPlayhead(2);   // frame 50 at 25 fps
    await render(<FrameNavigator time={2} setTime={setTime} fps={25} duration={4} disabled={false} />);
    const input = host.querySelector('input[aria-label="Frame number"]') as HTMLInputElement;
    expect(input.value).toBe('50');
    await act(async () => (host.querySelector('[aria-label="Next frame"]') as HTMLButtonElement).click());
    expect(setTime).toHaveBeenLastCalledWith(51 / 25);
    await act(async () => (host.querySelector('[aria-label="Previous frame"]') as HTMLButtonElement).click());
    expect(setTime).toHaveBeenLastCalledWith(49 / 25);
    await act(async () => (host.querySelector('[aria-label="Last frame"]') as HTMLButtonElement).click());
    expect(setTime).toHaveBeenLastCalledWith(99 / 25);
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
    await act(async () => { setter.call(input, '10'); input.dispatchEvent(new Event('input', { bubbles: true })); });
    await act(async () => input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })));
    expect(setTime).toHaveBeenLastCalledWith(10 / 25);
    expect(trackLabel('track_0003')).toBe('Face 3');
  });
});

describe('StudioInspector', () => {
  const base = { caps: null, sheet: null, sources: SOURCES, assignedId: '', onAssign: vi.fn(), coverage: null, mesh: null, reconstructionPicker: null, panelsRef: () => {}, processing: null };
  it('3D render mode adds and removes its key; relighting gates the light probe; unselectable engines are disabled', async () => {
    api.mockResolvedValue({ engines: [{ engine_id: 'reference_onnx', name: 'Reference', status: 'READY_FOR_INFERENCE', selectable: true, actual_inference_available: true, evidence: 'ok' },
      { engine_id: 'x', name: 'X', status: 'NOT_INSTALLED', selectable: false, actual_inference_available: false, evidence: 'absent' }] });
    let parameters: WorkstationParameters = { ...DEFAULT_PARAMETERS };
    const setParameters = vi.fn((value: WorkstationParameters) => { parameters = value; });
    const mount = () => render(<StudioInspector {...base} track={TRACK('track_0000', 0, 9)} parameters={parameters} setParameters={setParameters} />);
    engineSelection.set({ engineId: 'reference_onnx' });
    await mount(); await flush();
    const engine = host.querySelector('select[aria-label="Swap engine"]') as HTMLSelectElement;
    expect((engine.querySelector('option[value="x"]') as HTMLOptionElement).disabled).toBe(true);
    const radio = (label: string) => Array.from(host.querySelectorAll('[role="radiogroup"][aria-label="3D render mode"] [role="radio"]')).find(item => item.textContent === label) as HTMLButtonElement;
    await act(async () => radio('3D Facial Warp').click());
    expect(parameters.render_mode).toBe('pose_3d');
    await mount();
    expect(radio('3D Facial Warp').getAttribute('aria-checked')).toBe('true');
    await act(async () => radio('Standard 2D Identity Swap').click());
    expect('render_mode' in parameters).toBe(false);
    expect(host.querySelector('[data-testid="light-probe"]')).toBeNull();
    const relight = Array.from(host.querySelectorAll('.inspector-toggle')).find(item => item.textContent?.includes('3D relighting'))!.querySelector('input') as HTMLInputElement;
    await act(async () => relight.click());
    expect(parameters.relight).toBe(true);
    await mount(); await flush();
    expect(host.querySelector('[data-testid="light-probe"]')).toBeTruthy();
    expect(host.textContent).toContain('-10° … 12°');
  });

  it('lays the pipeline out as 3 steps with the experimental controls inside a closed Advanced Diagnostics drawer', async () => {
    api.mockResolvedValue({ engines: [] });
    let parameters: WorkstationParameters = { ...DEFAULT_PARAMETERS };
    const setParameters = vi.fn((value: WorkstationParameters) => { parameters = value; });
    const mount = () => render(<StudioInspector {...base} track={null} parameters={parameters} setParameters={setParameters} footer={<div data-testid="footer-slot" />} />);
    await mount(); await flush();
    for (const id of ['step-swap', 'step-mask', 'step-enhance']) expect(host.querySelector('[data-testid="' + id + '"]')).toBeTruthy();
    const drawer = host.querySelector('[data-testid="advanced-diagnostics"]') as HTMLDetailsElement;
    expect(drawer.open).toBe(false);
    expect(drawer.querySelector('select[aria-label="Swap engine"]')).toBeTruthy();
    expect(host.querySelector('[data-testid="step-swap"] select[aria-label="Swap engine"]')).toBeNull();
    expect(host.querySelector('[data-testid="footer-slot"]')).toBeTruthy();
    // Step 2: XSeg + Smart Occluder is the compositor's robust mode; BiSeNet is the always-on base matte.
    const toggle = host.querySelector('input[aria-label="XSeg mask and smart occluder"]') as HTMLInputElement;
    await act(async () => toggle.click());
    expect(parameters.occlusion_mode).toBe('robust');
    // Step 3: choosing an enhancer turns restoration on at a sensible blend; None turns it off.
    await mount();
    const enhancer = host.querySelector('select[aria-label="Enhancer"]') as HTMLSelectElement;
    await act(async () => { enhancer.value = 'ultramax'; enhancer.dispatchEvent(new Event('change', { bubbles: true })); });
    expect(parameters.restorer_model).toBe('ultramax');
    expect(parameters.restoration_blend).toBeCloseTo(0.7);
    await mount();
    expect((host.querySelector('select[aria-label="Enhancer"]') as HTMLSelectElement).value).toBe('ultramax');
    const none = host.querySelector('select[aria-label="Enhancer"]') as HTMLSelectElement;
    await act(async () => { none.value = 'none'; none.dispatchEvent(new Event('change', { bubbles: true })); });
    expect(parameters.restoration_blend).toBe(0);
  });

  it('a failed model hub shows the reason and retries on request (no endless skeleton)', async () => {
    // The Engine & Model Rack also reads /models and /models/active; answer them by path so only the hub call fails once.
    let hubCalls = 0;
    api.mockImplementation(async (path: string) => { if (path === '/models') return []; if (path === '/models/active') return { selected: {}, active: {} }; if (path === '/reconstruction/backbones') return { default: '3ddfa_v3_hrn', active: null, backbones: [] }; hubCalls += 1; if (hubCalls === 1) throw new Error('offline'); return { engines: [] }; });
    await render(<StudioInspector {...base} track={null} parameters={{ ...DEFAULT_PARAMETERS }} setParameters={vi.fn()} />);
    await flush();
    expect(host.textContent).toContain('Model hub unavailable: offline');
    await act(async () => byText('Retry')!.click());
    await flush();
    expect(host.querySelector('select[aria-label="Swap engine"]')).toBeTruthy();
    expect(hubCalls).toBe(2);
  });
});

describe('live preview payload (preview without a full render)', () => {
  const request = { target_path: 't.mp4', inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }], processing: 'local' as const, remote_consent: false as const, trim_start_seconds: null, trim_end_seconds: null };
  it('omits Stage 7 fields at their defaults and carries them when set', () => {
    const plain = buildLivePayload({ request, time: 1, fps: 25, parameters: { ...DEFAULT_PARAMETERS }, restorerAvailable: false })!;
    expect(JSON.parse(JSON.stringify(plain))).not.toHaveProperty('render_mode');
    expect(JSON.parse(JSON.stringify(plain))).not.toHaveProperty('occlusion_mode');
    expect(JSON.parse(JSON.stringify(plain))).not.toHaveProperty('relight');
    const set = buildLivePayload({ request, time: 1, fps: 25, restorerAvailable: false, parameters: { ...DEFAULT_PARAMETERS, render_mode: 'reenactment_3d', occlusion_mode: 'robust', relight: true, lighting_mode: 'manual', manual_light_yaw: 10 } })!;
    expect(set).toMatchObject({ render_mode: 'reenactment_3d', occlusion_mode: 'robust', relight: true, lighting_mode: 'manual', manual_light_yaw: 10, relight_strength: 0.8 });
  });
});

describe('PanelBoundary', () => {
  it('keeps a panel failure local and remounts on Retry', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {}); vi.spyOn(console, 'warn').mockImplementation(() => {});
    let fail = true;
    const Flaky = () => { if (fail) throw new Error('bad data'); return <p>recovered</p>; };
    await render(<div><PanelBoundary name="Inspector"><Flaky /></PanelBoundary><p>preview still here</p></div>);
    expect(host.textContent).toContain('Inspector could not be displayed: bad data');
    expect(host.textContent).toContain('preview still here');
    fail = false;
    await act(async () => byText('Retry')!.click());
    expect(host.textContent).toContain('recovered');
  });
});
