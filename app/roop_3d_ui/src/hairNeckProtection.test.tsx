// @vitest-environment jsdom
import { act, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import HairNeckProtection from './components/studio/HairNeckProtection';
import HeadModelBackbone, { BACKBONE_KEY, type BackboneReport } from './components/studio/HeadModelBackbone';
import {
  PROTECTION_DEFAULTS, hairlineInset, matteProtectionApplies, protectionTouched, resetProtection, withHairlineInset, withProtection,
} from './components/studio/protection';
import { DEFAULT_PARAMETERS, type WorkstationParameters } from './components/workstation/pipeline';
import { buildLivePayload } from './livePreview';
import type { JobRequest } from './contracts';

const api = vi.fn();
vi.mock('./api', () => ({ api: (...args: unknown[]) => api(...args) }));

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
const change = (input: HTMLInputElement, value: string) => act(async () => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
});
const flush = () => act(async () => { await Promise.resolve(); await Promise.resolve(); });

describe('protection helpers keep untouched requests byte-identical', () => {
  it('removes a control that is set back to its engine default', () => {
    const touched = withProtection(DEFAULT_PARAMETERS, 'neck_protection', false);
    expect(touched.neck_protection).toBe(false);
    expect('neck_protection' in withProtection(touched, 'neck_protection', true)).toBe(false);
    expect('jaw_feather_px' in withProtection(DEFAULT_PARAMETERS, 'jaw_feather_px', PROTECTION_DEFAULTS.jaw_feather_px)).toBe(false);
    expect(withProtection(DEFAULT_PARAMETERS, 'skin_texture_retention', 0.6).skin_texture_retention).toBe(0.6);
  });

  it('the hairline inset is the engine offset with the sign flipped, and is omitted at zero', () => {
    expect(withHairlineInset(DEFAULT_PARAMETERS, 8).hairline_offset_px).toBe(-8);
    expect(withHairlineInset(DEFAULT_PARAMETERS, -5).hairline_offset_px).toBe(5);
    expect('hairline_offset_px' in withHairlineInset({ ...DEFAULT_PARAMETERS, hairline_offset_px: 4 }, 0)).toBe(false);
    expect(hairlineInset({ ...DEFAULT_PARAMETERS, hairline_offset_px: -8 })).toBe(8);
    expect(Object.is(hairlineInset(DEFAULT_PARAMETERS), 0)).toBe(true);
  });

  it('reset removes all five controls and the hairline offset, and only those', () => {
    let parameters: WorkstationParameters = { ...DEFAULT_PARAMETERS, feather_px: 9 };
    parameters = withHairlineInset(withProtection(withProtection(parameters, 'hair_matting', false), 'jaw_feather_px', 12), 3);
    expect(protectionTouched(parameters)).toBe(true);
    const reset = resetProtection(parameters);
    expect(reset).toEqual({ ...DEFAULT_PARAMETERS, feather_px: 9 });
    expect(protectionTouched(reset)).toBe(false);
  });

  it('the matte controls only apply to a 3D-projected matte', () => {
    expect(matteProtectionApplies(DEFAULT_PARAMETERS)).toBe(false);
    expect(matteProtectionApplies({ ...DEFAULT_PARAMETERS, render_mode: 'pose_3d' })).toBe(true);
    expect(matteProtectionApplies({ ...DEFAULT_PARAMETERS, occlusion_mode: 'robust' })).toBe(true);
  });
});

describe('HairNeckProtection', () => {
  function Harness({ initial = DEFAULT_PARAMETERS, enhancer = 'ultramax', onChange }: { initial?: WorkstationParameters; enhancer?: string; onChange: (value: WorkstationParameters) => void }) {
    const [parameters, setParameters] = useState(initial);
    return <HairNeckProtection parameters={parameters} enhancer={enhancer} setParameters={value => { onChange(value); setParameters(value); }} />;
  }
  async function mount(props: Parameters<typeof Harness>[0]) {
    const host = document.createElement('div'); document.body.append(host);
    const root = createRoot(host);
    await act(async () => root.render(<Harness {...props} />));
    return { host, unmount: async () => { await act(async () => root.unmount()); host.remove(); } };
  }
  const field = (host: HTMLElement, label: string) => host.querySelector<HTMLInputElement>(`[aria-label="${label}"]`)!;

  it('shows the engine defaults and sends nothing until a control is touched', async () => {
    const seen: WorkstationParameters[] = [];
    const { host, unmount } = await mount({ onChange: value => seen.push(value) });
    try {
      for (const label of ['Strict Neck & Throat Protection', 'Hair-Strand Guided Matting', 'Preserve Target Ears & Eyeglasses']) expect(field(host, label).checked).toBe(true);
      expect(field(host, 'Forehead hairline inset').value).toBe('0');
      expect(field(host, 'Jaw-to-neck feather falloff').value).toBe('6');
      expect(field(host, 'Skin micro-texture and pore retention').value).toBe('25');
      expect(seen).toEqual([]);
      expect(host.querySelector<HTMLButtonElement>('.protection-head button')!.disabled).toBe(true);
    } finally { await unmount(); }
  });

  it('writes each toggle and slider to its engine parameter and omits it again at the default', async () => {
    const seen: WorkstationParameters[] = [];
    const { host, unmount } = await mount({ onChange: value => seen.push(value) });
    try {
      await act(async () => field(host, 'Strict Neck & Throat Protection').click());
      expect(seen.at(-1)?.neck_protection).toBe(false);
      await act(async () => field(host, 'Strict Neck & Throat Protection').click());
      expect('neck_protection' in seen.at(-1)!).toBe(false);
      await act(async () => field(host, 'Hair-Strand Guided Matting').click());
      expect(seen.at(-1)?.hair_matting).toBe(false);
      await act(async () => field(host, 'Preserve Target Ears & Eyeglasses').click());
      expect(seen.at(-1)).toMatchObject({ hair_matting: false, preserve_ears_glasses: false });
      await change(field(host, 'Jaw-to-neck feather falloff'), '18');
      expect(seen.at(-1)?.jaw_feather_px).toBe(18);
      await change(field(host, 'Skin micro-texture and pore retention'), '60');
      expect(seen.at(-1)?.skin_texture_retention).toBeCloseTo(0.6);
      await change(field(host, 'Forehead hairline inset'), '14');
      expect(seen.at(-1)?.hairline_offset_px).toBe(-14);
      expect(host.querySelector('.inspector-field output')?.textContent).toBe('+14 px');
      await change(field(host, 'Forehead hairline inset'), '0');
      expect('hairline_offset_px' in seen.at(-1)!).toBe(false);
      await act(async () => host.querySelector<HTMLButtonElement>('.protection-head button')!.click());
      expect(seen.at(-1)).toEqual(DEFAULT_PARAMETERS);
    } finally { await unmount(); }
  });

  it('keeps the slider ranges the spec asks for', async () => {
    const { host, unmount } = await mount({ onChange: () => {} });
    try {
      const inset = field(host, 'Forehead hairline inset'), jaw = field(host, 'Jaw-to-neck feather falloff'), pores = field(host, 'Skin micro-texture and pore retention');
      expect([inset.min, inset.max]).toEqual(['-20', '20']);
      expect([jaw.min, jaw.max]).toEqual(['2', '24']);
      expect([pores.min, pores.max]).toEqual(['0', '100']);
    } finally { await unmount(); }
  });

  it('says why a control has no effect instead of pretending: pores need UltraMax, the jaw feather needs neck protection, the matte needs a 3D matte', async () => {
    const { host, unmount } = await mount({ enhancer: 'gpen', initial: withProtection(DEFAULT_PARAMETERS, 'neck_protection', false), onChange: () => {} });
    try {
      expect(field(host, 'Skin micro-texture and pore retention').disabled).toBe(true);
      expect(host.textContent).toContain('UltraMax enhancer only');
      expect(field(host, 'Jaw-to-neck feather falloff').disabled).toBe(true);
      expect(host.querySelector('[data-testid="protection-scope"]')?.textContent).toContain('3D render mode');
    } finally { await unmount(); }
    const robust = await mount({ initial: { ...DEFAULT_PARAMETERS, occlusion_mode: 'robust' }, onChange: () => {} });
    try {
      expect(robust.host.querySelector('[data-testid="protection-scope"]')).toBeNull();
      expect(field(robust.host, 'Skin micro-texture and pore retention').disabled).toBe(false);
    } finally { await robust.unmount(); }
  });
});

describe('HeadModelBackbone', () => {
  const REPORT: BackboneReport = { default: '3ddfa_v3_hrn', active: '3ddfa_v3_hrn', backbones: [
    { id: '3ddfa_v3_hrn', label: '3DDFA-V3 Dense 35k + HRN Detail (Photorealistic Default)', available: true, active: true, reason: '' },
    { id: 'flame_smirk', label: 'FLAME 2023 Open + SMIRK Micro-Expressions', available: false, active: false, reason: 'Needs its own bundle.' },
    { id: 'faceverse_v4', label: 'FaceVerse V4 (52 ARKit Blendshapes)', available: false, active: false, reason: 'Not supported: no FaceVerse topology.' },
    { id: 'makehuman_legacy', label: 'Legacy MakeHuman CC0 (Debug Only)', available: true, active: false, reason: 'Inspection and export only: the live swap never uses it.' },
  ] };
  beforeEach(() => { api.mockReset(); try { localStorage.clear(); } catch { /* jsdom without storage */ } });
  afterEach(() => { document.body.innerHTML = ''; });

  async function mount() {
    const host = document.createElement('div'); document.body.append(host);
    const root = createRoot(host);
    await act(async () => root.render(<HeadModelBackbone />));
    await flush();
    return { host, unmount: async () => { await act(async () => root.unmount()); host.remove(); } };
  }

  it('offers the four backbones, disables the ones that are not installed and explains why', async () => {
    api.mockResolvedValue(REPORT);
    const { host, unmount } = await mount();
    try {
      expect(api).toHaveBeenCalledWith('/reconstruction/backbones', expect.anything());
      const select = host.querySelector<HTMLSelectElement>('[aria-label="3D head model backbone"]')!;
      const options = [...select.options];
      expect(options.map(option => option.textContent)).toEqual([
        '3DDFA-V3 Dense 35k + HRN Detail (Photorealistic Default)', 'FLAME 2023 Open + SMIRK Micro-Expressions — unavailable',
        'FaceVerse V4 (52 ARKit Blendshapes) — unavailable', 'Legacy MakeHuman CC0 (Debug Only)']);
      expect(options.map(option => option.disabled)).toEqual([false, true, true, false]);
      expect(select.value).toBe('3ddfa_v3_hrn');
      expect(host.textContent).toContain('Installed backbone: 3DDFA-V3');
      expect(host.querySelector('.backbone-unavailable')?.textContent).toContain('Not supported: no FaceVerse topology.');
    } finally { await unmount(); }
  });

  it('remembers an available choice, and ignores a saved one that is not installed', async () => {
    api.mockResolvedValue(REPORT);
    const first = await mount();
    try {
      const select = first.host.querySelector<HTMLSelectElement>('select')!;
      await act(async () => { select.value = 'makehuman_legacy'; select.dispatchEvent(new Event('change', { bubbles: true })); });
      expect(localStorage.getItem(BACKBONE_KEY)).toBe('makehuman_legacy');
      expect(first.host.querySelector('[data-testid="backbone-reason"]')?.textContent).toContain('the live swap never uses it');   // the debug-only caveat stays visible
    } finally { await first.unmount(); }
    localStorage.setItem(BACKBONE_KEY, 'faceverse_v4');
    const second = await mount();
    try { expect(second.host.querySelector<HTMLSelectElement>('select')!.value).toBe('3ddfa_v3_hrn'); } finally { await second.unmount(); }
  });

  it('reports a missing bundle plainly when nothing is installed', async () => {
    api.mockResolvedValue({ default: '3ddfa_v3_hrn', active: null, backbones: REPORT.backbones.map(row => ({ ...row, available: false, active: false, reason: 'No dense-face bundle is installed.' })) });
    const { host, unmount } = await mount();
    try {
      expect(host.textContent).toContain('No dense-face backbone is installed');
      expect(host.querySelector('[data-testid="backbone-reason"]')?.textContent).toBe('No dense-face bundle is installed.');
    } finally { await unmount(); }
  });

  it('shows an error with a retry when the service is down', async () => {
    api.mockRejectedValueOnce(new Error('connection refused')).mockResolvedValueOnce(REPORT);
    const { host, unmount } = await mount();
    try {
      expect(host.textContent).toContain('Backbone status unavailable: connection refused');
      await act(async () => [...host.querySelectorAll('button')].find(button => /retry/i.test(button.textContent ?? ''))!.click());
      await flush();
      expect(host.querySelector('select')).not.toBeNull();
    } finally { await unmount(); }
  });
});

describe('live preview payload', () => {
  const request = { engine_id: 'reference_onnx', target_path: 'clip.mp4', inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }] } as unknown as JobRequest;
  const payload = (parameters: WorkstationParameters) => JSON.parse(JSON.stringify(buildLivePayload({ request, time: 1, fps: 25, restorerAvailable: true, parameters })!));

  it('leaves the new fields out until they are touched, so earlier previews keep their cache keys', () => {
    const plain = payload(DEFAULT_PARAMETERS);
    for (const key of ['neck_protection', 'hair_matting', 'preserve_ears_glasses', 'jaw_feather_px', 'skin_texture_retention', 'hairline_offset_px']) expect(key in plain).toBe(false);
  });

  it('sends exactly the touched controls under the engine names', () => {
    let parameters = withProtection(withProtection(DEFAULT_PARAMETERS, 'hair_matting', false), 'skin_texture_retention', 0.5);
    parameters = withHairlineInset(parameters, 7);
    const body = payload(parameters);
    expect(body).toMatchObject({ hair_matting: false, skin_texture_retention: 0.5, hairline_offset_px: -7 });
    expect('neck_protection' in body).toBe(false);
  });
});
