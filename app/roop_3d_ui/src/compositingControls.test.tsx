// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import CompositingControls from './components/studio/CompositingControls';
import { occlusionModePatch, pyramidInactiveReason, xsegInactiveReason } from './components/studio/compositing';
import { describeRestoration } from './components/workstation/LiveSwapStatus';
import { selectEngine } from './stores/engineSelection';
import type { RestorationReport } from './api';
import type { WorkstationParameters } from './components/workstation/pipeline';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

let host: HTMLDivElement, root: Root;
beforeEach(() => { host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(async () => { await act(async () => root.unmount()); host.remove(); selectEngine('reference_onnx'); });

const mount = async (params: Partial<WorkstationParameters>, onParams = vi.fn()) => {
  await act(async () => {
    root.render(<CompositingControls params={params} onParams={onParams} />);
  });
  return onParams;
};
const slider = (label: string) => host.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)!;
const set = (input: HTMLInputElement, value: string) => act(async () => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
});

describe('XSeg occlusion strength', () => {
  it('is disabled with the reason while the standard compositor runs (it loads no XSeg network)', async () => {
    await mount({});
    expect(slider('XSeg occlusion strength').disabled).toBe(true);
    expect(host.querySelector('[data-testid="xseg-strength-note"]')!.textContent).toContain('only run in Robust occlusion mode');
  });

  it('is live under Robust occlusion and sends the 0..1 value the engine bounds', async () => {
    const onParams = await mount({ occlusion_mode: 'robust' });
    expect(slider('XSeg occlusion strength').disabled).toBe(false);
    expect(slider('XSeg occlusion strength').value).toBe('100');
    await set(slider('XSeg occlusion strength'), '40');
    expect(onParams).toHaveBeenLastCalledWith({ xseg_occlusion_strength: 0.4 });
  });
});

describe('seam blending', () => {
  it('defaults to the four-band multi-band pyramid and sends the chosen fusion', async () => {
    const onParams = await mount({});
    const select = host.querySelector<HTMLSelectElement>('select[aria-label="Seam blending"]')!;
    expect(select.value).toBe('multi_band');
    expect(slider('Pyramid bands').value).toBe('4');
    expect([slider('Pyramid bands').min, slider('Pyramid bands').max]).toEqual(['4', '6']);
    await act(async () => { select.value = 'poisson'; select.dispatchEvent(new Event('change', { bubbles: true })); });
    expect(onParams).toHaveBeenLastCalledWith({ blend_mode: 'poisson' });
  });

  it('disables the band count where the fusion has no pyramid, and says why', async () => {
    await mount({ blend_mode: 'feather' });
    expect(slider('Pyramid bands').disabled).toBe(true);
    expect(host.querySelector('[data-testid="pyramid-note"]')!.textContent).toContain('no pyramid');
    expect(pyramidInactiveReason({ blend_mode: 'frequency_separated' })).toBeNull();
    expect(pyramidInactiveReason({})).toBeNull();
  });

  it('blocks Frequency-separated under Robust occlusion, and Robust returns it to the default fusion', async () => {
    await mount({ occlusion_mode: 'robust' });
    const frequency = [...host.querySelectorAll<HTMLOptionElement>('select[aria-label="Seam blending"] option')].find(item => item.value === 'frequency_separated')!;
    expect(frequency.disabled).toBe(true);
    expect(occlusionModePatch({ blend_mode: 'frequency_separated' }, 'robust')).toEqual({ occlusion_mode: 'robust', blend_mode: 'multi_band' });
    expect(occlusionModePatch({ blend_mode: 'poisson' }, 'robust')).toEqual({ occlusion_mode: 'robust' });
    expect(occlusionModePatch({ blend_mode: 'frequency_separated' }, 'standard')).toEqual({ occlusion_mode: 'standard' });
    expect(xsegInactiveReason({ occlusion_mode: 'robust' })).toBeNull();
  });

  it('says the Photoreal engine ignores the blend mode but follows the band count', async () => {
    await mount({});
    expect(host.querySelector('[data-testid="photoreal-blend-note"]')).toBeNull();
    selectEngine('photoreal_3d_ffmpeg');
    await mount({});
    expect(host.querySelector('[data-testid="photoreal-blend-note"]')!.textContent).toContain('ignores the seam blending mode');
  });
});

describe('restoration readout', () => {
  const report = (patch: Partial<RestorationReport> = {}): RestorationReport => ({
    model: 'codeformer.fp16.onnx', applied: 2, bypassed: 0, failed: 0, bypass_reasons: {}, mean_latency_ms: 143.2, mean_inference_ms: 98.4, last: null, ...patch,
  });
  it('reports observed outcomes only: applied with measured times, user-selected skips with the reason, failures', () => {
    expect(describeRestoration(report())).toBe('Restorer codeformer.fp16.onnx: 2 applied (143 ms, model 98 ms)');
    expect(describeRestoration(report({ applied: 0, mean_latency_ms: null, mean_inference_ms: null, bypassed: 1, bypass_reasons: { small_face_bypass: 1 } })))
      .toBe('Restorer codeformer.fp16.onnx: 0 applied · 1 bypassed (small face)');
    expect(describeRestoration(report({ failed: 1 }))).toContain('1 failed');
  });
});
