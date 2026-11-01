// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import SettingControl from './components/controls/SettingControl';
import { PANELS, SETTINGS, SETTING_BY_ID, recommendedText, settingValue, settingsOf, type SettingId } from './components/controls/settingsRegistry';
import ContactSheetModal, { LEVELS, defaultAxes, type Patch } from './components/workstation/ContactSheetModal';
import { DEFAULT_PARAMETERS } from './components/workstation/pipeline';
import { PreviewBusyError } from './api';

describe('settings registry', () => {
  it('documents every setting with sane bounds, a recommended range inside them, and three distinct sweep values', () => {
    expect(SETTINGS.map(spec => spec.id).sort()).toEqual(['color_strength', 'erosion_px', 'feather_px', 'fidelity_weight', 'pose_frontalization', 'pose_strength',
      'restoration_blend', 'swap_weight', 'tps_expression_weight']);
    for (const spec of SETTINGS) {
      expect(spec.min, spec.id).toBeLessThan(spec.max);
      expect(spec.default, spec.id).toBeGreaterThanOrEqual(spec.min);
      expect(spec.default, spec.id).toBeLessThanOrEqual(spec.max);
      expect(spec.recommended[0], spec.id).toBeGreaterThanOrEqual(spec.min);
      expect(spec.recommended[1], spec.id).toBeLessThanOrEqual(spec.max);
      expect(spec.recommended[0], spec.id).toBeLessThan(spec.recommended[1]);
      expect(new Set(spec.sweep).size, spec.id).toBe(3);
      expect([...spec.sweep].sort((a, b) => a - b), spec.id).toEqual([...spec.sweep]);                       // Low < Medium < High
      expect(spec.sweep.every(value => value >= spec.min && value <= spec.max), spec.id).toBe(true);
      expect(spec.effect.length, spec.id).toBeGreaterThan(60);                                                // says what you will SEE
    }
  });

  it('agrees with the application\'s real defaults, so "reset" restores what a fresh session has', () => {
    for (const id of ['pose_strength', 'swap_weight', 'erosion_px', 'feather_px', 'color_strength', 'restoration_blend'] as const) {
      expect(SETTING_BY_ID[id].default, id).toBe((DEFAULT_PARAMETERS as any)[id]);
    }
    expect(settingValue(SETTING_BY_ID.tps_expression_weight, DEFAULT_PARAMETERS)).toBe(0);                  // omitted until touched
    expect(settingValue(SETTING_BY_ID.tps_expression_weight, { ...DEFAULT_PARAMETERS, tps_expression_weight: .4 })).toBe(.4);
  });

  it('groups settings into the 3D Geometry, Masking and Restorer panels', () => {
    expect(Object.values(PANELS).map(panel => panel.title)).toEqual(['3D Geometry', 'Masking', 'Restorer']);
    expect(settingsOf('geometry').map(spec => spec.id)).toEqual(['pose_strength', 'pose_frontalization', 'tps_expression_weight']);
    expect(settingsOf('masking').map(spec => spec.id)).toEqual(['swap_weight', 'erosion_px', 'feather_px']);
    expect(settingsOf('restorer').map(spec => spec.id)).toEqual(['color_strength', 'restoration_blend', 'fidelity_weight']);
    expect(recommendedText(SETTING_BY_ID.feather_px)).toBe('3–12 px');
    expect(recommendedText(SETTING_BY_ID.tps_expression_weight)).toBe('0.30–0.80');
  });

  it('is honest that CodeFormer fidelity currently has no effect', () => {
    expect(SETTING_BY_ID.fidelity_weight.inactive).toMatch(/not wired/i);
  });
});

describe('SettingControl', () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true; host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); });

  function Harness({ id, start, disabled = false, reason }: { id: SettingId; start: number; disabled?: boolean; reason?: string }) {
    const [value, setValue] = useState(start);
    return <><SettingControl spec={SETTING_BY_ID[id]} value={value} onChange={setValue} disabled={disabled} disabledReason={reason}/><output data-testid="value">{value}</output></>;
  }
  const value = () => Number(host.querySelector('[data-testid="value"]')!.textContent);
  const setInput = (el: HTMLInputElement, text: string) => act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(el, text);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });

  it('resets to the default with one click, and only offers reset when the value differs', async () => {
    await act(async () => root.render(<Harness id="feather_px" start={20}/>));
    const reset = host.querySelector<HTMLButtonElement>('[aria-label^="Reset Mask feather"]')!;
    expect(reset.getAttribute('aria-label')).toBe('Reset Mask feather to default (5)');
    expect(reset.disabled).toBe(false);
    await act(async () => reset.click());
    expect(value()).toBe(5);
    expect(reset.disabled).toBe(true);
  });

  it('keeps the slider and the numeric field in step, clamped to the registered range', async () => {
    await act(async () => root.render(<Harness id="swap_weight" start={1}/>));
    const slider = host.querySelector<HTMLInputElement>('input[type="range"]')!;
    expect([slider.min, slider.max, slider.step]).toEqual(['0', '1', '0.01']);
    await setInput(slider, '0.4');
    expect(value()).toBe(0.4);
    expect(host.querySelector<HTMLInputElement>('input[aria-label="Swap weight"]')!.value).toBe('0.4');
  });

  it('accepts inline expression math in the numeric field', async () => {
    await act(async () => root.render(<Harness id="tps_expression_weight" start={0}/>));
    const field = host.querySelector<HTMLInputElement>('input[aria-label="TPS expression weight"]')!;
    await setInput(field, '0.5 + 0.25');
    await act(async () => { field.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })); });
    expect(value()).toBe(0.75);
    await setInput(field, '9 * 9');
    await act(async () => { field.focus(); field.blur(); });
    expect(value()).toBe(1);                                               // clamped to the registered maximum
  });

  it('scrubs by dragging the label (Shift = 10×)', async () => {
    await act(async () => root.render(<Harness id="erosion_px" start={0}/>));
    const label = host.querySelector<HTMLElement>('.scrub-label')!;
    label.setPointerCapture = vi.fn(); label.releasePointerCapture = vi.fn(); label.hasPointerCapture = () => false;
    const fire = (type: string, init: Record<string, unknown>) => act(async () => { label.dispatchEvent(Object.assign(new MouseEvent(type, { bubbles: true, button: 0, ...init }), { pointerId: 1 })); });
    await fire('pointerdown', { clientX: 100 });
    await fire('pointermove', { clientX: 104 });
    expect(value()).toBe(4);
    await fire('pointermove', { clientX: 104, shiftKey: true });         // 4 px * step 1 * 10 from the drag origin
    expect(value()).toBe(32);                                              // clamped to +32
    await fire('pointerup', { clientX: 104 });
  });

  it('explains itself on hover/focus: the visible effect, the recommended range, the default and the bounds', async () => {
    await act(async () => root.render(<Harness id="tps_expression_weight" start={0.3}/>));
    const badge = host.querySelector('.help-badge')!;
    const tip = host.querySelector('[role="tooltip"]')!;
    expect(badge.getAttribute('aria-describedby')).toBe(tip.id);
    expect(tip.textContent).toContain('mouth, eyebrows and eyelids');
    expect(tip.textContent).toContain('face outline is pinned');
    expect(tip.textContent).toContain('Recommended: 0.30–0.80');
    expect(tip.textContent).toContain('Default: 0');
    expect(tip.textContent).toContain('Range: 0–1');
  });

  it('disables an inert control and says why, in the control and in its help', async () => {
    await act(async () => root.render(<Harness id="fidelity_weight" start={0.5}/>));
    expect(host.querySelector<HTMLInputElement>('input[aria-label="CodeFormer fidelity"]')!.disabled).toBe(true);
    expect(host.querySelector<HTMLInputElement>('input[type="range"]')!.disabled).toBe(true);
    expect(host.querySelector<HTMLButtonElement>('.reset-setting')!.disabled).toBe(true);
    expect(host.querySelector('.tip-warning')!.textContent).toMatch(/changes nothing/);
    await act(async () => root.render(<Harness id="restoration_blend" start={0.5} disabled reason="GPEN weights are unavailable"/>));
    expect(host.querySelector('.tip-warning')!.textContent).toBe('GPEN weights are unavailable');
  });
});

describe('3×3 contact sheet', () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true; vi.useFakeTimers(); URL.createObjectURL = vi.fn(() => 'blob:tile'); URL.revokeObjectURL = vi.fn(); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); document.body.innerHTML = ''; vi.useRealTimers(); });
  const tileResult = () => ({ blob: new Blob([new Uint8Array([1])], { type: 'image/png' }), meta: null, cached: false });
  const open = (props: Record<string, unknown> = {}) => {
    const onApply = vi.fn(), onClose = vi.fn(), render = vi.fn(async () => tileResult());
    const payloadFor = vi.fn((patch: Patch) => ({ patch, frame_index: 7 }));
    act(() => { root.render(<ContactSheetModal panel="geometry" parameters={DEFAULT_PARAMETERS} frameLabel="Frame 7 · 0.280 s" payloadFor={payloadFor as any}
      onApply={onApply} onClose={onClose} render={render as any} {...props}/>); });
    return { onApply, onClose, render, payloadFor };
  };
  const generate = async () => { await act(async () => { [...document.querySelectorAll('button')].find(b => b.textContent === 'Generate contact sheet')!.click(); await vi.advanceTimersByTimeAsync(0); }); };
  const select = (label: string, value: string) => act(async () => {
    const el = document.querySelector<HTMLSelectElement>(`select[aria-label="${label}"]`)!;
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')!.set!.call(el, value); el.dispatchEvent(new Event('change', { bubbles: true }));
  });

  it('starts from the panel that opened it', () => {
    expect(defaultAxes('geometry')).toEqual(['pose_strength', 'pose_frontalization']);
    expect(defaultAxes('masking')).toEqual(['swap_weight', 'erosion_px']);
    expect(defaultAxes('restorer')).toEqual(['color_strength', 'restoration_blend']);   // fidelity_weight is inactive, so never pre-selected
    expect(defaultAxes(undefined)).toHaveLength(2);
  });

  it('renders nine real previews across Low / Medium / High in row-major order', async () => {
    const { render, payloadFor } = open();
    await select('X axis setting', 'tps_expression_weight');
    await select('Y axis setting', 'pose_frontalization');
    await generate();
    expect(render).toHaveBeenCalledTimes(9);
    const combos = payloadFor.mock.calls.map(([patch]) => [patch.tps_expression_weight, patch.pose_frontalization]);
    expect(combos).toEqual([[0, 0], [.5, 0], [1, 0], [0, .5], [.5, .5], [1, .5], [0, 1], [.5, 1], [1, 1]]);
    const tiles = [...document.querySelectorAll<HTMLButtonElement>('.sweep-cell')];
    expect(tiles).toHaveLength(9);
    expect(tiles.every(tile => tile.querySelector('img') && !tile.disabled)).toBe(true);
    expect(tiles[5].querySelector('img')!.getAttribute('alt')).toBe('High TPS expression weight, Medium Pose frontalization');
    expect(tiles[5].textContent).toContain('High TPS expression weight 1 · Medium Pose frontalization 0.5');
    expect(LEVELS).toEqual(['Low', 'Medium', 'High']);
  });

  it('clicking a tile applies both of its values and closes', async () => {
    const { onApply, onClose } = open();
    await generate();                                                        // default axes: pose_strength (X) x pose_frontalization (Y)
    await act(async () => document.querySelectorAll<HTMLButtonElement>('.sweep-cell')[7].click());   // Medium X, High Y
    expect(onApply).toHaveBeenCalledWith({ pose_strength: .5, pose_frontalization: 1 });
    expect(onClose).toHaveBeenCalled();
  });

  it('cannot pick an inactive setting or the same setting twice', async () => {
    open();
    const options = [...document.querySelectorAll<HTMLOptionElement>('select[aria-label="X axis setting"] option')];
    expect(options.find(option => option.value === 'fidelity_weight')!.disabled).toBe(true);
    expect(options.find(option => option.value === 'fidelity_weight')!.textContent).toContain('(inactive)');
    expect(options.find(option => option.value === 'pose_frontalization')!.disabled).toBe(true);   // already the Y axis
  });

  it('retries a busy tile after retry_after_ms instead of failing the sheet', async () => {
    let busy = 2;
    const render = vi.fn(async () => { if (busy-- > 0) throw new PreviewBusyError('STAGE_1_SPARSE_PRESCAN', 1000); return tileResult(); });
    open({ render });
    await act(async () => { [...document.querySelectorAll('button')].find(b => b.textContent === 'Generate contact sheet')!.click(); await vi.advanceTimersByTimeAsync(0); });
    expect(document.querySelector('.sweep-cell.busy .sweep-pending')!.textContent).toContain('GPU busy (STAGE_1_SPARSE_PRESCAN)');
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(render).toHaveBeenCalledTimes(2);                               // not before the server's retry_after_ms
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(document.querySelectorAll('.sweep-cell.ready').length).toBeGreaterThanOrEqual(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(document.querySelectorAll('.sweep-cell.ready')).toHaveLength(9);
  });

  it('shows a per-tile error, and cancelling aborts the remaining tiles', async () => {
    const render = vi.fn(async (_payload: unknown, signal: AbortSignal) => {
      if (render.mock.calls.length === 2) throw new Error('No mapped face is visible at this playhead frame');
      if (render.mock.calls.length === 4) await new Promise((_, reject) => signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError'))));
      return tileResult();
    });
    open({ render });
    await generate();
    expect(document.querySelector('.sweep-cell.error .sweep-pending')!.textContent).toContain('No mapped face');
    expect(render).toHaveBeenCalledTimes(4);                               // stuck on tile 4
    await act(async () => { [...document.querySelectorAll('button')].find(b => b.textContent === 'Cancel sweep')!.click(); await vi.advanceTimersByTimeAsync(0); });
    expect(render).toHaveBeenCalledTimes(4);                               // nothing further is requested
    expect([...document.querySelectorAll('button')].some(b => b.textContent === 'Cancel sweep')).toBe(false);
  });

  it('asks for a mapped face when there is nothing to render', async () => {
    open({ payloadFor: () => null });
    await generate();
    expect(document.querySelector('[role="alert"]')!.textContent).toContain('Inspect and map a target face');
  });
});
