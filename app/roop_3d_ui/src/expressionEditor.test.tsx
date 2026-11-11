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
import { isDisabled, setSlider, sliderValue, thumb } from './sliderTestKit';

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
  // restore factor + the 12 single controls + the blink pair, which is shown split here because its two channels differ (linked it would be one slider)
  expect(host.querySelectorAll('[role=slider]')).toHaveLength(FACE_EDIT_CONTROLS.length + 1);
  expect(sliderValue(thumb(host, 'Head Yaw Offset'))).toBe(12);
  expect(thumb(host, 'Head Yaw Offset').getAttribute('aria-valuetext')).toBe('12°');
  expect(sliderValue(thumb(host, 'Mouth Smile'))).toBe(0);
  // blink_left is 1 and blink_right unset: they differ, so the pair opens split rather than hiding the difference
  expect(sliderValue(thumb(host, 'Eye Blink Left'))).toBe(1); expect(sliderValue(thumb(host, 'Eye Blink Right'))).toBe(0);
  await act(async () => [...host.querySelectorAll('button')].find(item => item.textContent === 'Reset selected face')!.click());
  expect(set).toHaveBeenCalledWith({ ...parameters, face_expression_overrides: { track_0002: { mouth_smile: .3 } } });
  await act(async () => root.render(<FaceExpressionEditor parameters={parameters} setParameters={set} trackId={null}/>));
  expect(host.querySelector('fieldset')?.disabled).toBe(true);
  expect([...host.querySelectorAll<HTMLElement>('[role=slider]')].filter(item => item.getAttribute('aria-label') !== 'Expression restore factor').every(isDisabled)).toBe(true);
  await act(async () => root.unmount()); host.remove();
});

it('edits gaze and jaw per face, and the blink pair moves together until split', async () => {
  const host = document.createElement('div'), root = createRoot(host), set = vi.fn();
  document.body.append(host);
  const parameters = { ...DEFAULT_PARAMETERS, face_expression_overrides: {} };
  await act(async () => root.render(<FaceExpressionEditor parameters={parameters} setParameters={set} trackId="track_0001"/>));
  await setSlider(thumb(host, 'Eyeball Gaze X'), 0.4);
  expect(set).toHaveBeenLastCalledWith({ ...parameters, face_expression_overrides: { track_0001: { gaze_x: 0.4 } } });
  await setSlider(thumb(host, 'Jaw Width 3D Morph'), -0.25);
  expect(set).toHaveBeenLastCalledWith({ ...parameters, face_expression_overrides: { track_0001: { jaw_width: -0.25 } } });
  await setSlider(thumb(host, 'Head Pitch Offset'), 10);
  expect(set).toHaveBeenLastCalledWith({ ...parameters, face_expression_overrides: { track_0001: { head_pitch: 10 } } });
  // linked: a single "Eye Blink" slider drives both channels in one update
  await setSlider(thumb(host, 'Eye Blink'), 0.5);
  expect(set).toHaveBeenLastCalledWith({ ...parameters, face_expression_overrides: { track_0001: { blink_left: 0.5, blink_right: 0.5 } } });
  // back to zero clears both keys (sparse overrides)
  const linked = { ...parameters, face_expression_overrides: { track_0001: { blink_left: 0.5, blink_right: 0.5, gaze_y: 0.1 } } };
  await act(async () => root.render(<FaceExpressionEditor parameters={linked} setParameters={set} trackId="track_0001"/>));
  await setSlider(thumb(host, 'Eye Blink'), 0);
  expect(set).toHaveBeenLastCalledWith({ ...linked, face_expression_overrides: { track_0001: { gaze_y: 0.1 } } });
  // split: left and right are independent
  await act(async () => host.querySelector<HTMLButtonElement>('.ul-slider__link')!.click());
  await setSlider(thumb(host, 'Eye Blink Right'), 0.2);
  expect(set).toHaveBeenLastCalledWith({ ...linked, face_expression_overrides: { track_0001: { blink_left: 0.5, blink_right: 0.2, gaze_y: 0.1 } } });
  await act(async () => root.unmount()); host.remove();
});

it('sends the same global restore factor and per-face edits to the debounced live endpoint', () => {
  const parameters = { ...DEFAULT_PARAMETERS, expression_restore_factor: .8, face_expression_overrides: { track_0002: { head_pitch: -4, gaze_x: .4 } } };
  const request = { target_path: 'media/target.mp4', inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0002', faceset_id: 'source' }] } as JobRequest;
  const payload = buildLivePayload({ request, parameters, time: 0, fps: 30, restorerAvailable: false });
  expect(payload?.expression_restore_factor).toBe(.8);
  expect(payload?.face_expression_overrides).toEqual(parameters.face_expression_overrides);
});

it('exposes all 52 ARKit sliders and stores sparse values on the selected face', async () => {
  const host = document.createElement('div'), root = createRoot(host), set = vi.fn();
  document.body.append(host);
  const parameters = { ...DEFAULT_PARAMETERS, face_expression_overrides: {} };
  await act(async () => root.render(<FaceExpressionEditor parameters={parameters} setParameters={set} trackId="track_0001"/>));
  await act(async () => host.querySelector('.arkit-slider-panel summary')!.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  expect(host.querySelectorAll('.arkit-slider-list input[type=range]')).toHaveLength(52);
  const slider = host.querySelector<HTMLInputElement>('[aria-label="ARKit Jaw Open"]')!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(slider, '0.65');
    slider.dispatchEvent(new Event('input', { bubbles: true }));
  });
  expect(set).toHaveBeenLastCalledWith({ ...parameters, face_expression_overrides: {
    track_0001: { arkit_blendshapes: { jawOpen: 0.65 } },
  } });
  await act(async () => root.unmount()); host.remove();
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
