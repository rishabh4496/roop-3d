// @vitest-environment jsdom
import { act, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import SwapScopeSelector from './components/studio/SwapScopeSelector';
import {
  ADAPTIVE_START_BLEND, DEFAULT_SWAP_SCOPE, SWAP_SCOPES, skullBlendOf, swapScopeApplies, swapScopeOf, swapScopeTouched, withSkullBlend, withSwapScope,
} from './components/studio/swapScope';
import { DEFAULT_PARAMETERS, type WorkstationParameters } from './components/workstation/pipeline';
import { buildLivePayload } from './livePreview';
import type { JobRequest } from './contracts';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

describe('swap scope helpers keep untouched requests byte-identical', () => {
  it('lists the three scopes with the full craniofacial lock first and as the default', () => {
    expect(SWAP_SCOPES.map(item => item.id)).toEqual(['full_craniofacial', 'adaptive_skull', 'inner_face']);
    expect(SWAP_SCOPES[0].label).toBe('100% Full Craniofacial + Jaw + Forehead + Background Inpaint');
    expect(DEFAULT_SWAP_SCOPE).toBe('full_craniofacial');
    expect(swapScopeOf(DEFAULT_PARAMETERS)).toBe('full_craniofacial');
    expect(skullBlendOf(DEFAULT_PARAMETERS)).toBe(1);
  });

  it('picking the default removes both keys', () => {
    const adaptive = withSwapScope(DEFAULT_PARAMETERS, 'adaptive_skull');
    expect(adaptive).toMatchObject({ swap_scope: 'adaptive_skull', skull_blend: ADAPTIVE_START_BLEND });
    expect(withSwapScope(adaptive, 'full_craniofacial')).toEqual(DEFAULT_PARAMETERS);
    expect(swapScopeTouched(DEFAULT_PARAMETERS)).toBe(false);
    expect(swapScopeTouched(adaptive)).toBe(true);
  });

  it('skull_blend only travels with the adaptive scope', () => {
    const adaptive = withSkullBlend(withSwapScope(DEFAULT_PARAMETERS, 'adaptive_skull'), 0.3);
    expect(adaptive.skull_blend).toBe(0.3);
    expect(withSkullBlend(adaptive, 4).skull_blend).toBe(1);
    expect(withSkullBlend(adaptive, -1).skull_blend).toBe(0);
    const legacy = withSwapScope(adaptive, 'inner_face');
    expect(legacy.swap_scope).toBe('inner_face');
    expect('skull_blend' in legacy).toBe(false);
    expect(withSkullBlend(DEFAULT_PARAMETERS, 0.2)).toBe(DEFAULT_PARAMETERS);        // ignored outside the adaptive scope
    // Re-entering the adaptive scope keeps the user's earlier share rather than resetting it.
    expect(withSwapScope({ ...adaptive }, 'adaptive_skull').skull_blend).toBe(0.3);
  });

  it('only applies in a 3D render mode', () => {
    expect(swapScopeApplies(DEFAULT_PARAMETERS)).toBe(false);
    expect(swapScopeApplies({ ...DEFAULT_PARAMETERS, render_mode: 'pose_3d' })).toBe(true);
  });
});

describe('SwapScopeSelector', () => {
  const mounted: { unmount: () => Promise<void> }[] = [];
  afterEach(async () => { while (mounted.length) await mounted.pop()!.unmount(); });

  function Harness({ initial = DEFAULT_PARAMETERS, onChange }: { initial?: WorkstationParameters; onChange: (value: WorkstationParameters) => void }) {
    const [parameters, setParameters] = useState(initial);
    return <SwapScopeSelector parameters={parameters} setParameters={value => { onChange(value); setParameters(value); }} />;
  }
  async function mount(initial?: WorkstationParameters) {
    const seen: WorkstationParameters[] = [];
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => root.render(<Harness initial={initial} onChange={value => seen.push(value)} />));
    const unmount = async () => { await act(async () => root.unmount()); host.remove(); };
    mounted.push({ unmount });
    return { host, seen, unmount };
  }
  const radios = (host: HTMLElement) => [...host.querySelectorAll<HTMLElement>('[role="radio"]')];
  const change = (input: HTMLInputElement, value: string) => act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });

  it('shows three options with the full lock selected by default and no slider', async () => {
    const { host } = await mount();
    const options = radios(host);
    expect(options.map(item => item.querySelector('b')!.textContent)).toEqual(['100% Full Craniofacial + Jaw + Forehead + Background Inpaint default', 'Adaptive Skull Blend', 'Inner Face']);
    expect(options.map(item => item.getAttribute('aria-checked'))).toEqual(['true', 'false', 'false']);
    expect(host.querySelector('input[type="range"]')).toBeNull();
    expect(host.querySelector('[role="radiogroup"]')!.getAttribute('aria-labelledby')).toBe('swap-scope-label');
  });

  it('the adaptive option reveals the β_source / β_target slider and reports every change', async () => {
    const { host, seen } = await mount();
    await act(async () => radios(host)[1].click());
    expect(seen.at(-1)).toMatchObject({ swap_scope: 'adaptive_skull', skull_blend: 0.5 });
    const slider = host.querySelector<HTMLInputElement>('input[aria-label="Source skull share"]')!;
    expect(slider.value).toBe('50');
    expect(host.textContent).toContain('β source 50% · β target 50%');
    await change(slider, '80');
    expect(seen.at(-1)!.skull_blend).toBeCloseTo(0.8);
    expect(host.textContent).toContain('β source 80% · β target 20%');
    expect(slider.getAttribute('aria-valuetext')).toBe('80 percent source skull, 20 percent target skull');
  });

  it('legacy removes the slider; the default removes the keys again', async () => {
    const { host, seen } = await mount({ ...DEFAULT_PARAMETERS, swap_scope: 'adaptive_skull', skull_blend: 0.4 });
    expect(host.querySelector('input[type="range"]')).not.toBeNull();
    await act(async () => radios(host)[2].click());
    expect(seen.at(-1)).toMatchObject({ swap_scope: 'inner_face' });
    expect('skull_blend' in seen.at(-1)!).toBe(false);
    expect(host.querySelector('input[type="range"]')).toBeNull();
    await act(async () => radios(host)[0].click());
    expect(seen.at(-1)).toEqual(DEFAULT_PARAMETERS);
  });

  it('says plainly that the scope needs a 3D render mode, and stops saying it once one is on', async () => {
    const plain = await mount();
    expect(plain.host.querySelector('[data-testid="swap-scope-applies"]')!.textContent).toContain('3D render mode');
    await plain.unmount();
    const three = await mount({ ...DEFAULT_PARAMETERS, render_mode: 'pose_3d' });
    expect(three.host.querySelector('[data-testid="swap-scope-applies"]')).toBeNull();
  });
});

describe('live preview payload', () => {
  const request = { engine_id: 'reference_onnx', target_path: 'clip.mp4', inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }] } as unknown as JobRequest;
  const payload = (parameters: WorkstationParameters) => JSON.parse(JSON.stringify(buildLivePayload({ request, time: 1, fps: 25, restorerAvailable: true, parameters })!));

  it('leaves the scope out until it is changed, so earlier previews keep their cache keys', () => {
    const plain = payload(DEFAULT_PARAMETERS);
    expect('swap_scope' in plain).toBe(false);
    expect('skull_blend' in plain).toBe(false);
  });

  it('sends the scope under the engine names, and the blend only for the adaptive scope', () => {
    expect(payload(withSwapScope(DEFAULT_PARAMETERS, 'inner_face'))).toMatchObject({ swap_scope: 'inner_face' });
    expect('skull_blend' in payload(withSwapScope(DEFAULT_PARAMETERS, 'inner_face'))).toBe(false);
    expect(payload(withSkullBlend(withSwapScope(DEFAULT_PARAMETERS, 'adaptive_skull'), 0.25))).toMatchObject({ swap_scope: 'adaptive_skull', skull_blend: 0.25 });
  });
});
