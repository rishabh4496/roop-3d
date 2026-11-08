// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';
import FaceExpressionEditor, { FACE_EDIT_CONTROLS } from './components/studio/FaceExpressionEditor';
import PhotoSynthesisAction from './components/facemgr/PhotoSynthesisAction';
import { DEFAULT_PARAMETERS } from './components/workstation/pipeline';
import { buildLivePayload } from './livePreview';
import { api, post } from './api';
import type { JobRequest } from './contracts';

vi.mock('./api', () => ({ api: vi.fn(), post: vi.fn(), postLivePreview: vi.fn(), PreviewBusyError: class extends Error {} }));
(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
afterEach(() => { vi.useRealTimers(); vi.clearAllMocks(); });

it('shows every requested 3D edit, scopes values and reset to the selected target face', async () => {
  const host = document.createElement('div'), root = createRoot(host), set = vi.fn();
  document.body.append(host);
  const parameters = { ...DEFAULT_PARAMETERS, expression_restore_factor: .7, face_expression_overrides: {
    track_0001: { head_yaw: 12, blink_left: 1 }, track_0002: { mouth_smile: .3 },
  } };
  await act(async () => root.render(<FaceExpressionEditor parameters={parameters} setParameters={set} trackId="track_0001"/>));
  expect(host.querySelectorAll('input[type=range]')).toHaveLength(FACE_EDIT_CONTROLS.length+1);
  expect(host.querySelector<HTMLInputElement>('[aria-label="Head Yaw Offset"]')?.value).toBe('12');
  expect(host.querySelector<HTMLInputElement>('[aria-label="Mouth Smile"]')?.value).toBe('0');
  await act(async () => host.querySelector<HTMLButtonElement>('button')!.click());
  expect(set).toHaveBeenCalledWith({ ...parameters, face_expression_overrides: { track_0002: { mouth_smile: .3 } } });
  await act(async () => root.render(<FaceExpressionEditor parameters={parameters} setParameters={set} trackId={null}/>));
  expect(host.querySelector('fieldset')?.disabled).toBe(true);
  await act(async () => root.unmount()); host.remove();
});

it('sends the same global restore factor and per-face edits to the debounced live endpoint', () => {
  const parameters = { ...DEFAULT_PARAMETERS, expression_restore_factor: .8, face_expression_overrides: { track_0002: { head_pitch: -4, gaze_x: .4 } } };
  const request = { target_path: 'media/target.mp4', inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0002', faceset_id: 'source' }] } as JobRequest;
  const payload = buildLivePayload({ request, parameters, time: 0, fps: 30, restorerAvailable: false });
  expect(payload?.expression_restore_factor).toBe(.8);
  expect(payload?.face_expression_overrides).toEqual(parameters.face_expression_overrides);
});

it('synthesizes imported photos with one click, polls progress and activates the result', async () => {
  vi.useFakeTimers();
  const host = document.createElement('div'), root = createRoot(host), saved = vi.fn();
  document.body.append(host);
  vi.mocked(post).mockResolvedValue({ job_id: 'b'.repeat(32), status: 'synthesizing', progress: 0, scanned: 0 });
  vi.mocked(api).mockResolvedValue({ job_id: 'b'.repeat(32), status: 'ready', progress: 1, scanned: 36, result: { faceset_id: 'generated' } });
  await act(async () => root.render(<PhotoSynthesisAction images={['data:image/png;base64,AA==']} name="Alice" onSaved={saved}/>));
  await act(async () => host.querySelector<HTMLButtonElement>('button')!.click());
  expect(post).toHaveBeenCalledWith('/faceset/synthesize-photos', { images: ['data:image/png;base64,AA=='], name: 'Alice' });
  expect(host.querySelector<HTMLButtonElement>('button')?.disabled).toBe(true);
  await act(async () => { await vi.advanceTimersByTimeAsync(750); });
  expect(saved).toHaveBeenCalledWith('generated');
  expect(host.textContent).toContain('saved and activated');
  await act(async () => root.unmount()); host.remove();
});
