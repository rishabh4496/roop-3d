// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import ContactSheetModal, { AXES, type Patch } from './components/workstation/ContactSheetModal';
import ZoomableImage, { MAX_ZOOM, clampPan } from './components/workstation/ZoomableImage';
import { CAMERA_FOV_LEVELS, applySweepPatch } from './components/workstation/sweepPatch';
import { DEFAULT_PARAMETERS } from './components/workstation/pipeline';
import { buildLivePayload } from './livePreview';
import CurveEditorTrack from './components/timeline/CurveEditorTrack';
import useHistoryStore from './stores/useHistoryStore';
import { CURVE_HINTS, CURVE_PARAMETERS, evaluateParameters } from './stores/parameterCurves';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

describe('sweep patch semantics', () => {
  it('sweeping relight strength switches relighting on, since it does nothing otherwise', () => {
    expect(applySweepPatch(DEFAULT_PARAMETERS, { relight_strength: .25 })).toMatchObject({ relight: true, relight_strength: .25 });
  });

  it('a swept field of view is a manual perspective camera for that tile, never an orthographic one', () => {
    const next = applySweepPatch(DEFAULT_PARAMETERS, { camera_fov_deg: 85 });
    expect(next.camera).toMatchObject({ mode: 'manual', projection: 'perspective', focal_unit: 'fov_deg', fov_deg: 85 });
  });

  it('the tile preview and the applied parameters are the same object of settings', () => {
    const patch: Patch = { camera_fov_deg: 30, relight_strength: .6 };
    const request = { engine_id: 'reference_onnx', target_path: 't.mp4', inspection_id: 'i', face_mappings: [{ track_id: 't0', faceset_id: 'f' }] } as never;
    const parameters = applySweepPatch(DEFAULT_PARAMETERS, patch);
    const payload = buildLivePayload({ request, time: 1, fps: 25, parameters, solo: 'final' })!;
    expect(payload.camera).toMatchObject({ mode: 'manual', projection: 'perspective', fov_deg: 30 });
    expect(payload.relight).toBe(true);
    expect(payload.relight_strength).toBe(.6);
  });
});

describe('sweep axes cover the requested comparison', () => {
  it('offers a perspective camera field-of-view axis and a 25 / 60 / 100 % relighting axis, and no enhancer axis', () => {
    const camera = AXES.find(axis => axis.id === 'camera_fov_deg')!, relight = AXES.find(axis => axis.id === 'relight_strength')!;
    expect(AXES.find(axis => (axis.id as string) === 'enhancer')).toBeUndefined();
    expect(camera.values).toEqual(CAMERA_FOV_LEVELS);
    expect(camera.levels()).toBe('30° · 55° · 85°');
    expect(relight.values).toEqual([.25, .6, 1]);
    expect(relight.levels()).toBe('25% · 60% · 100%');
  });
});

describe('3×3 contact sheet with stepped axes and native-pixel zoom', () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { vi.useFakeTimers(); URL.createObjectURL = vi.fn(() => 'blob:tile'); URL.revokeObjectURL = vi.fn(); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); document.body.innerHTML = ''; vi.useRealTimers(); });
  const tile = () => ({ blob: new Blob([new Uint8Array([1])], { type: 'image/png' }), meta: null, cached: false });
  const setup = () => {
    const onApply = vi.fn(), onClose = vi.fn(), render = vi.fn(async () => tile());
    const payloadFor = vi.fn((patch: Patch) => ({ patch, frame_index: 7 }));
    act(() => { root.render(<ContactSheetModal panel="color" parameters={DEFAULT_PARAMETERS} frameLabel="Frame 7" payloadFor={payloadFor as never} onApply={onApply} onClose={onClose} render={render as never} />); });
    return { onApply, onClose, render, payloadFor };
  };
  const choose = (label: string, value: string) => act(async () => {
    const el = document.querySelector<HTMLSelectElement>(`select[aria-label="${label}"]`)!;
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')!.set!.call(el, value);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
  const generate = async () => act(async () => { [...document.querySelectorAll('button')].find(b => b.textContent === 'Generate contact sheet')!.click(); await vi.advanceTimersByTimeAsync(0); });

  it('renders the camera field of view (X) by relighting strength (Y) grid in row-major order and applies the clicked cell', async () => {
    const { onApply, onClose, payloadFor } = setup();
    await choose('X axis setting', 'camera_fov_deg');
    await choose('Y axis setting', 'relight_strength');
    await generate();
    expect(payloadFor.mock.calls.map(([patch]) => [patch.camera_fov_deg, patch.relight_strength])).toEqual([
      [30, .25], [55, .25], [85, .25], [30, .6], [55, .6], [85, .6], [30, 1], [55, 1], [85, 1]]);
    const cells = [...document.querySelectorAll<HTMLButtonElement>('.sweep-cell')];
    expect(cells).toHaveLength(9);
    expect(cells[8].textContent).toContain('Camera FOV 85°');
    expect(cells[8].textContent).toContain('SH relighting 100%');
    await act(async () => cells[4].click());
    expect(onApply).toHaveBeenCalledWith({ camera_fov_deg: 55, relight_strength: .6 });
    expect(onClose).toHaveBeenCalled();
  });

  it('zooms one tile to native pixels, can go back to the sheet, and applies from the zoomed view', async () => {
    const { onApply } = setup();
    await generate();
    const zoomButtons = [...document.querySelectorAll<HTMLButtonElement>('button[aria-label^="Zoom tile"]')];
    expect(zoomButtons).toHaveLength(9);
    await act(async () => zoomButtons[2].click());
    const frame = document.querySelector<HTMLElement>('[data-testid="zoom-frame"]')!;
    expect(frame.dataset.zoom).toBe('1.00');
    expect(document.querySelector<HTMLElement>('.sweep-grid')!.style.display).toBe('none');
    await act(async () => [...document.querySelectorAll('button')].find(b => b.textContent === '400%')!.click());
    expect(frame.dataset.zoom).toBe('4.00');
    await act(async () => [...document.querySelectorAll('button')].find(b => b.textContent === 'Back to sheet')!.click());
    expect(document.querySelector('[data-testid="sweep-zoom"]')).toBeNull();
    expect(document.querySelector<HTMLElement>('.sweep-grid')!.style.display).not.toBe('none');
    await act(async () => zoomButtons[2].click());
    await act(async () => [...document.querySelectorAll('button')].find(b => b.textContent === 'Apply these settings')!.click());
    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onApply.mock.calls[0][0]).toEqual({ color_strength: .9, swap_weight: .3 });   // row 0 (Low identity weight), column 2 (High colour)
  });
});

describe('ZoomableImage', () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); });
  const frame = () => document.querySelector<HTMLElement>('[data-testid="zoom-frame"]')!;
  const key = (value: string) => act(async () => { frame().dispatchEvent(new KeyboardEvent('keydown', { key: value, bubbles: true })); });

  it('clamps the pan so the zoomed image always covers its viewport', () => {
    expect(clampPan({ x: 999, y: -999 }, 2, 400, 200)).toEqual({ x: 200, y: -100 });
    expect(clampPan({ x: 50, y: 50 }, 1, 400, 200)).toEqual({ x: 0, y: 0 });
  });

  it('zooms from the keyboard within its bounds and resets with 0', async () => {
    act(() => root.render(<ZoomableImage src="blob:x" alt="tile" />));
    expect(frame().dataset.zoom).toBe('1.00');
    await key('+'); await key('+');
    expect(frame().dataset.zoom).toBe((1.25 * 1.25).toFixed(2));
    for (let i = 0; i < 40; i++) await key('+');
    expect(Number(frame().dataset.zoom)).toBe(MAX_ZOOM);
    await key('0');
    expect(frame().dataset.zoom).toBe('1.00');
    for (let i = 0; i < 10; i++) await key('-');
    expect(frame().dataset.zoom).toBe('1.00');
  });

  it('wheel zooms in and out, and a new image starts back at fit', async () => {
    act(() => root.render(<ZoomableImage src="blob:a" alt="tile" />));
    await act(async () => { frame().dispatchEvent(new WheelEvent('wheel', { deltaY: -100, bubbles: true, cancelable: true })); });
    expect(frame().dataset.zoom).toBe("1.25");
    await act(async () => { frame().dispatchEvent(new WheelEvent('wheel', { deltaY: 100, bubbles: true, cancelable: true })); });
    expect(frame().dataset.zoom).toBe('1.00');
    await key('+');
    act(() => root.render(<ZoomableImage src="blob:b" alt="tile" />));
    expect(frame().dataset.zoom).toBe('1.00');
  });
});

describe('parameter curves for the animatable workstation controls', () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { useHistoryStore.getState().reset(); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); });

  it('lists the animatable parameters, each mapped to a real engine field, and none from the removed 2D / 2.5D path', () => {
    const order = Object.keys(CURVE_PARAMETERS);
    expect(order.slice(0, 3)).toEqual(['relight_strength', 'mask_feather', 'occlusion_threshold']);
    expect(order.join(' ')).not.toMatch(/pose_correction|fidelity|tps|frontaliz|restor/);
    expect(CURVE_PARAMETERS.relight_strength.field).toBe('relight_strength');
    expect(CURVE_PARAMETERS.occlusion_threshold.field).toBe('occlusion_threshold');
    expect(CURVE_PARAMETERS.mask_feather.field).toBe('feather_px');
  });

  it('evaluates the new curves into the same field names the backend validates', () => {
    const keys = [{ frame: 0, value: 0, easing: 'linear' }, { frame: 100, value: 1, easing: 'linear' }];
    const result = evaluateParameters({}, { relight_strength: keys, occlusion_threshold: keys }, 25);
    expect(result).toMatchObject({ relight_strength: .25, occlusion_threshold: .25 });
  });

  it('says honestly that the occlusion control drives the BiSeNet parser, not a separate XSeg network', () => {
    expect(CURVE_HINTS.occlusion_threshold).toMatch(/BiSeNet/);
    expect(CURVE_HINTS.occlusion_threshold).toMatch(/no separate XSeg network/);
    expect(CURVE_PARAMETERS.occlusion_threshold.label).toMatch(/XSeg \/ BiSeNet/);
  });

  it('shows the hint for the selected curve and animates the relight curve', async () => {
    act(() => root.render(<CurveEditorTrack frame={10} totalFrames={100} parameters={{ relight_strength: .8 }} />));
    const select = host.querySelector<HTMLSelectElement>('[aria-label="Curve parameter"]')!;
    await act(async () => { Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')!.set!.call(select, 'relight_strength'); select.dispatchEvent(new Event('change', { bubbles: true })); });
    expect(host.querySelector('[data-testid="curve-hint"]')!.textContent).toContain('3D Relighting is switched on');
    await act(async () => [...host.querySelectorAll('button')].find(b => b.textContent === 'Add key at playhead')!.click());
    expect(useHistoryStore.getState().curves.relight_strength[0]).toMatchObject({ frame: 10, value: .8 });
    await act(async () => useHistoryStore.getState().undo());
    expect(useHistoryStore.getState().curves.relight_strength ?? []).toHaveLength(0);
  });
});
