// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeEach, expect, it, vi } from 'vitest';
import { HotkeyProvider } from './components/controls/Hotkeys';
import { WorkspaceLayoutProvider } from './components/workstation/WorkspaceLayouts';
import App from './App';
import useHistoryStore from './stores/useHistoryStore';
import { SESSION_KEY, activeJobToResume, clearSession, parseSession, readSession, writeSession } from './sessionSnapshot';

const calls: string[] = [];
let jobList: any[] = [];
let missingTarget = false;

vi.mock('./events', () => ({ EventConnection: class { start() {} stop() {} } }));
vi.mock('./ThreeViewport', () => ({ PoseCoverageInspector: () => null, PoseGeometryViewer: () => null, LightProbeSphere: () => null }));
vi.mock('./components/hud/PipelinePill', () => ({ default: () => null }));
vi.mock('./components/queue/QueueDrawer', () => ({ default: () => null }));
vi.mock('./components/facemgr/FaceManager', () => ({ default: () => null }));
vi.mock('./VideoViewport', () => ({ formatTime: (value: number) => String(value),
  VideoViewport: (props: any) => <><output data-target>{props.target}</output><output data-workspace>{JSON.stringify(props.captureWorkspace())}</output></> }));
vi.mock('./api', () => ({
  post: vi.fn(async (path: string, body: any) => {
    calls.push(`POST ${path}`);
    if (path === '/media/probe') { if (missingTarget) throw new Error('Local video file does not exist'); return { width: 1280, height: 694, duration_seconds: 300, fps_rational: '30/1', declared_frame_count: 9000, audio_tracks: [] }; }
    return {};
  }),
  upload: vi.fn(),
  api: vi.fn(async (path: string) => {
    calls.push(`GET ${path}`);
    if (path === '/system/capabilities') return { models: [], job_parameter_bounds: {}, legal_parameter_bounds: {}, workstation: { features: {}, read_only_parameters: {}, disabled_parameters: {}, controls: {}, job_controls: {} } };
    if (path === '/config') return {};
    if (path === '/jobs') return jobList;
    if (path.startsWith('/media/keyframes')) return { timestamps: [0, 10] };
    if (path.startsWith('/media/inspections/')) return { inspection_id: 'insp1', target_sha256: 'a'.repeat(64), tracks: [{ track_id: 'track_0004', first_frame: 90, last_frame: 2700, first_timestamp_seconds: 3, last_timestamp_seconds: 90, thumbnail_base64: '', thumbnail_mime: 'image/jpeg', landmark_count: 5, pose_range: { yaw_degrees: [-20, 20], pitch_degrees: [-5, 5], roll_degrees: [-3, 3], method: 'test' } }], warnings: [], truncated: false };
    return [];
  }),
}));

const TARGET = 'roop_3d/.runtime/media/803b3b70f4334d17a2835b843ff5dd99.mp4';
const snapshot = (over: Record<string, unknown> = {}) => ({
  target: TARGET, inspectionId: 'insp1', selectedJob: 'job46', projectId: '', projectName: 'Untitled project', identityLockId: '', engineId: 'reference_onnx',
  time: 42, sourceId: 'fs1', identitySourceB: '', selectedTrack: 'track_0004', trimStart: 5, trimEnd: 60, parameters: { swap_weight: 0.3 },
  exportOptions: null, curves: {}, assigned: { track_0004: 'fs1' }, faceSegments: [], ...over });

beforeEach(() => { calls.length = 0; jobList = []; missingTarget = false; localStorage.clear(); useHistoryStore.getState().reset(); location.hash = '#/faceswap'; });

async function mount() {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  await act(async () => root.render(<HotkeyProvider><WorkspaceLayoutProvider><App/></WorkspaceLayoutProvider></HotkeyProvider>));
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 30)); });
  return { host, unmount: async () => { await act(async () => root.unmount()); host.remove(); } };
}
const workspace = (host: HTMLElement) => JSON.parse(host.querySelector('[data-workspace]')!.textContent!);

it('rebuilds the workstation from the saved session after the page reloads (Pinokio Run -> Dev -> Run)', async () => {
  jobList = [{ job_id: 'job46', state: 'running', request: { target_path: TARGET, inspection_id: 'insp1', face_mappings: [] } }];
  writeSession(snapshot() as any);
  const { host, unmount } = await mount();
  try {
    expect(host.querySelector('[data-target]')?.textContent).toBe(TARGET);
    const restored = workspace(host);
    expect(restored.time).toBe(42); expect(restored.sourceId).toBe('fs1'); expect(restored.trimStart).toBe(5); expect(restored.trimEnd).toBe(60);
    expect(restored.selectedTrack).toBe('track_0004'); expect(restored.parameters.swap_weight).toBe(0.3);
    expect(calls).toContain('POST /media/probe'); expect(calls.some(call => call.startsWith('GET /media/inspections/insp1'))).toBe(true);
    expect(host.textContent).toContain('restored after the page reloaded');
  } finally { await unmount(); }
});

it('keeps saving the session as it changes, and writes it as the page goes away', async () => {
  writeSession(snapshot({ selectedJob: '' }) as any);
  const first = await mount();
  try {
    localStorage.removeItem(SESSION_KEY);                       // prove the page itself writes it back
    await act(async () => { window.dispatchEvent(new Event('pagehide')); });
    expect(readSession()?.target).toBe(TARGET);
    expect(readSession()?.time).toBe(42);
  } finally { await first.unmount(); }
  const second = await mount();                                 // "Run" again: same state, no new action needed
  try { expect(second.host.querySelector('[data-target]')?.textContent).toBe(TARGET); } finally { await second.unmount(); }
});

it('drops a session whose video is gone instead of showing a broken workstation', async () => {
  writeSession(snapshot() as any); missingTarget = true;
  const { host, unmount } = await mount();
  try {
    expect(host.querySelector('[data-target]')?.textContent).toBe('');
    expect(host.textContent).toContain('could not be restored');
    expect(readSession()).toBeNull();
  } finally { await unmount(); }
});

it('with nothing saved, brings back the render that is still running', async () => {
  jobList = [{ job_id: 'done1', state: 'succeeded', request: { target_path: 'x.mp4', inspection_id: 'i0', face_mappings: [] } },
    { job_id: 'job46', state: 'running', request: { target_path: TARGET, inspection_id: 'insp1', face_mappings: [{ track_id: 'track_0004', faceset_id: 'fs1' }] } }];
  const { host, unmount } = await mount();
  try { expect(host.querySelector('[data-target]')?.textContent).toBe(TARGET); } finally { await unmount(); }
});

it('does nothing on an empty backend with nothing saved', async () => {
  const { host, unmount } = await mount();
  try { expect(host.querySelector('[data-target]')?.textContent).toBe(''); expect(calls).not.toContain('POST /media/probe'); } finally { await unmount(); }
});

it('validates what it reads back from storage', () => {
  expect(parseSession(null)).toBeNull();
  expect(parseSession('not json')).toBeNull();
  expect(parseSession(JSON.stringify({ v: 99, target: 'a.mp4' }))).toBeNull();           // unknown version
  expect(parseSession(JSON.stringify({ v: 1, target: '' }))).toBeNull();                  // nothing to restore
  expect(parseSession('x'.repeat(300_000))).toBeNull();                                   // oversize
  const parsed = parseSession(JSON.stringify({ v: 1, target: 'a.mp4', time: -5, trimEnd: 'bad', assigned: { a: 'f', b: 7 }, parameters: [1], faceSegments: 'no' }))!;
  expect(parsed.time).toBe(0); expect(parsed.trimEnd).toBeNull(); expect(parsed.assigned).toEqual({ a: 'f' });
  expect(parsed.parameters).toEqual({}); expect(parsed.faceSegments).toEqual([]);
  expect(writeSession({ ...snapshot(), target: '' } as any)).toBe(false);                 // never stores an empty session
  const failing = { getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('quota'); }, removeItem: () => { throw new Error('blocked'); } };
  expect(readSession(failing)).toBeNull(); expect(writeSession(snapshot() as any, failing)).toBe(false); expect(() => clearSession(failing)).not.toThrow();
});

it('picks the running render first, then any other live one', () => {
  expect(activeJobToResume([])).toBeNull();
  expect(activeJobToResume([{ job_id: 'a', state: 'succeeded' }, { job_id: 'b', state: 'failed' }])).toBeNull();
  expect(activeJobToResume([{ job_id: 'a', state: 'queued' }, { job_id: 'b', state: 'running' }])?.job_id).toBe('b');
  expect(activeJobToResume([{ job_id: 'a', state: 'paused' }])?.job_id).toBe('a');
});
