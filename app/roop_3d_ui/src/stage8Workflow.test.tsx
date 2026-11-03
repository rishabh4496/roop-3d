// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { beforeEach, afterEach, expect, it, vi } from 'vitest';
import QueueDrawer from './components/queue/QueueDrawer';
import { buildLivePayload } from './livePreview';
import { DEFAULT_PARAMETERS } from './components/workstation/pipeline';
import type { JobRequest } from './contracts';

const post = vi.fn();
vi.mock('./api', () => ({ post: (...args: unknown[]) => post(...args) }));
vi.mock('./components/diagnostics/BottleneckFlamegraph', () => ({ default: () => null }));
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let host: HTMLDivElement, root: Root;
beforeEach(() => { post.mockResolvedValue({}); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(async () => { await act(async () => root.unmount()); host.remove(); localStorage.clear(); });
const job = (state: string) => ({ job_id: state, state, request: { target_path: 'local/target.mp4' }, progress: { frame: 2, total_frames: 10, fps: 1, stage: 'render' }, output: { output_path: 'owned/render.mp4', frames: 10, manifest_path: 'owned/render.json' } });
const render = async (jobs: any[]) => act(async () => root.render(<QueueDrawer jobs={jobs} isOpen />));
const button = (name: string) => [...host.querySelectorAll('button')].find(item => item.textContent === name)!;

it('uses the server queue even when stale completed jobs exist in localStorage', async () => {
  localStorage.setItem('roop3d_queue_cache', JSON.stringify([job('succeeded')]));
  await render([job('running')]);
  await render([]);
  expect(host.textContent).toContain('Render queue is empty');
  expect(host.querySelectorAll('.queue-job-card')).toHaveLength(0);
});

it('offers legal controls and exports only successful completed output', async () => {
  await render([job('running')]);
  expect(host.querySelectorAll('a')).toHaveLength(0);
  await act(async () => button('Pause').click());
  expect(post).toHaveBeenLastCalledWith('/jobs/running/pause', {});
  await render([job('paused')]);
  await act(async () => button('Resume').click());
  expect(post).toHaveBeenLastCalledWith('/jobs/paused/resume', {});
  await render([job('failed')]);
  expect(host.querySelectorAll('a')).toHaveLength(0);
  await render([job('succeeded')]);
  expect(host.querySelector('a[href$="/artifacts/manifest"]')).not.toBeNull();
  await act(async () => button('Inspect output').click());
  expect(host.querySelector('video')?.getAttribute('src')).toBe('/api/v1/jobs/succeeded/artifacts/video');
  expect(host.querySelector('[role="dialog"]')?.textContent).toContain('Synthetic media processed locally');
});

it('uses exact playhead timestamps for VFR preview and carries target lock and motion controls', () => {
  const request = { target_path: 'owned/video.mp4', inspection_id: 'a'.repeat(32), identity_lock_id: 'b'.repeat(32), face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }] } as JobRequest;
  const payload = buildLivePayload({ request, time: .12, fps: 29.97, timestampSelection: true,
    parameters: { ...DEFAULT_PARAMETERS, pose_strength: .7 }, restorerAvailable: false });
  expect(payload).toMatchObject({ timestamp: .12, identity_lock_id: request.identity_lock_id, pose_strength: .7 });
  expect(payload).not.toHaveProperty('frame_index');
});
