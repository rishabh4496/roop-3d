// @vitest-environment jsdom
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import WorkstationControls3D, { LIP_LOCK_KEY, SCENE_ADAPT_KEY, SMOOTHING_COMMIT_MS, type TopologyReport } from './components/studio/WorkstationControls3D';
import { DEFAULT_PARAMETERS, type WorkstationParameters } from './components/workstation/pipeline';
import { buildLivePayload } from './livePreview';
import type { Config, JobRequest, Reconstruction } from './contracts';

const api = vi.fn();
vi.mock('./api', () => ({ api: (...args: unknown[]) => api(...args) }));
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const report: TopologyReport = {
  default: 'makehuman_cc0',
  topologies: [
    { id: 'makehuman_cc0', title: 'MakeHuman', license_id: 'CC0-1.0', commercial_use: true, default: true, available: true, state: 'ready', detail: null },
    { id: 'flame2023_open', title: 'FLAME 2023 Open', license_id: 'CC-BY-4.0', commercial_use: true, default: false, available: false, state: 'asset_missing', detail: 'flame2023_Open.pkl was not found' },
  ],
  semantic_sliders: [
    { key: 'jaw_width', label: 'Jaw width', description: 'Bigonial width', unit_mm: 4, minimum: -3, maximum: 3, default: 0 },
    { key: 'cheekbone_prominence', label: 'Cheekbone prominence', description: '', unit_mm: 2, minimum: -3, maximum: 3, default: 0 },
    { key: 'chin_projection', label: 'Chin projection', description: '', unit_mm: 3, minimum: -3, maximum: 3, default: 0 },
    { key: 'nose_bridge', label: 'Nose bridge', description: '', unit_mm: 2, minimum: -3, maximum: 3, default: 0 },
  ],
};
const mesh: Reconstruction = { method: 'identity_asset', coordinate_system: 'mesh', vertices: [], faces: [], uvs: null, texture: null, provenance: 'makehuman · semantic sliders applied' };

const change = (input: HTMLInputElement, value: string) => act(async () => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
});
const flush = () => act(async () => { for (let i = 0; i < 4; i++) await Promise.resolve(); });
const slider = (host: HTMLElement, label: string) => host.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)!;

let host: HTMLDivElement, root: Root;
let latest: { parameters: WorkstationParameters };
const saveConfig = vi.fn();
const onMesh = vi.fn();

function Harness({ trackId = 'track_0001', facesetId = 'fs1', config = { temporal_min_cutoff: 0.15 } as Config | null }: { trackId?: string | null; facesetId?: string; config?: Config | null }) {
  const [parameters, setParameters] = useState<WorkstationParameters>({ ...DEFAULT_PARAMETERS });
  latest = { parameters };
  return <WorkstationControls3D parameters={parameters} setParameters={setParameters} trackId={trackId} facesetId={facesetId} onMesh={onMesh} config={config} onSaveConfig={saveConfig} />;
}
const mount = async (props: Parameters<typeof Harness>[0] = {}) => { await act(async () => { root.render(<Harness {...props} />); }); };
const open = async (testid: string) => act(async () => {
  const details = host.querySelector<HTMLDetailsElement>(`[data-testid="${testid}"]`)!;
  details.open = true; details.dispatchEvent(new Event('toggle'));
});

beforeEach(() => {
  api.mockReset(); saveConfig.mockReset(); onMesh.mockReset();
  api.mockImplementation(async (path: string) => (path.startsWith('/morphable-topologies') ? report : mesh));
  localStorage.clear();
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.useRealTimers(); });

describe('skin tone vs ambient adaptation', () => {
  it('drives only the colour-transfer weight', async () => {
    await mount();
    await change(slider(host, 'Skin tone preservation versus ambient adaptation'), '20');
    expect(latest.parameters.color_strength).toBeCloseTo(0.2);
    expect({ ...latest.parameters, color_strength: DEFAULT_PARAMETERS.color_strength }).toEqual(DEFAULT_PARAMETERS);
  });
  it('says so when colour matching is bypassed, and never claims to be the TRUST network', async () => {
    await mount();
    expect(host.textContent).toMatch(/not the TRUST network/);
    expect(host.textContent).not.toMatch(/bypassed/);
  });
});

describe('expression offsets', () => {
  it('edit the selected face through the real face_expression_overrides, and clear when back at zero', async () => {
    await mount();
    await change(slider(host, 'Expression offset smile'), '0.6');
    await change(slider(host, 'Expression offset mouth open'), '0.3');
    expect(latest.parameters.face_expression_overrides).toEqual({ track_0001: { mouth_smile: 0.6, mouth_open: 0.3 } });
    await act(async () => { [...host.querySelectorAll('button')].find(b => b.textContent === 'Reset offsets')!.click(); });
    expect(latest.parameters.face_expression_overrides).toEqual({});
  });
  it('are disabled until a target face is selected', async () => {
    await mount({ trackId: null });
    expect(slider(host, 'Expression offset smile').closest('fieldset')!.disabled).toBe(true);
    expect(host.textContent).toMatch(/select a target face/);
  });
});

describe('preference-only toggles (no engine consumer yet)', () => {
  it('are remembered locally, labelled as such, and never reach the preview request', async () => {
    await mount();
    const toggles = [...host.querySelectorAll<HTMLInputElement>('input[type=checkbox]')];
    const lip = toggles.find(t => t.closest('label')?.textContent?.includes('Speech-lip lock'))!;
    const adapt = toggles.find(t => t.closest('label')?.textContent?.includes('3-second scene adaptation'))!;
    await act(async () => { lip.click(); adapt.click(); });
    expect(localStorage.getItem(LIP_LOCK_KEY)).toBe('1');
    expect(localStorage.getItem(SCENE_ADAPT_KEY)).toBe('1');
    expect(host.textContent?.match(/saved preference only/g)).toHaveLength(2);
    expect(latest.parameters).toEqual(DEFAULT_PARAMETERS);
    const request = { engine_id: 'x', target_path: 't', inspection_id: 'i', face_mappings: [{ track_id: 'track_0001', faceset_id: 'fs1' }] } as unknown as JobRequest;
    const payload = buildLivePayload({ request, time: 0, fps: 30, parameters: latest.parameters, restorerAvailable: false })!;
    expect(JSON.stringify(payload)).not.toMatch(/lip|spectre|spark|adapt/i);
  });
  it('restores the saved state on the next visit', async () => {
    localStorage.setItem(LIP_LOCK_KEY, '1');
    await mount();
    const lip = [...host.querySelectorAll<HTMLInputElement>('input[type=checkbox]')].find(t => t.closest('label')?.textContent?.includes('Speech-lip lock'))!;
    expect(lip.checked).toBe(true);
  });
});

describe('temporal smoothing', () => {
  it('saves the global cutoff once, after the slider stops moving', async () => {
    vi.useFakeTimers();
    await mount();
    const input = slider(host, 'Temporal Kalman smoothing');
    await change(input, '80'); await change(input, '90');
    expect(saveConfig).not.toHaveBeenCalled();
    await act(async () => { vi.advanceTimersByTime(SMOOTHING_COMMIT_MS + 10); });
    expect(saveConfig).toHaveBeenCalledTimes(1);
    const saved = saveConfig.mock.calls[0][0] as Config;
    expect(saved.temporal_min_cutoff).toBeLessThan(0.15);   // heavier smoothing = lower cutoff
    expect(saved.temporal_min_cutoff).toBeGreaterThan(0.03);
  });
  it('does not save when the value lands where it already was, and is disabled without a config', async () => {
    vi.useFakeTimers();
    await mount();
    await change(slider(host, 'Temporal Kalman smoothing'), String(Math.round(0.5883 * 100)));
    await act(async () => { vi.advanceTimersByTime(SMOOTHING_COMMIT_MS + 10); });
    expect(saveConfig).not.toHaveBeenCalled();
    await mount({ config: null });
    expect(slider(host, 'Temporal Kalman smoothing').disabled).toBe(true);
  });
});

describe('topology engine and semantic shape', () => {
  it('fetches nothing until the section is opened', async () => {
    await mount();
    await flush();
    expect(api).not.toHaveBeenCalled();
    await open('topology-section');
    await flush();
    expect(api).toHaveBeenCalledWith('/morphable-topologies', expect.anything());
  });

  it('offers both topologies and disables the one whose weights are missing, with the reason', async () => {
    await mount(); await open('topology-section'); await flush();
    const radios = [...host.querySelectorAll<HTMLButtonElement>('[role=radio]')].filter(r => /MakeHuman CC0|FLAME 2023 Open/.test(r.textContent ?? ''));
    expect(radios.map(r => r.textContent)).toEqual([expect.stringContaining('MakeHuman CC0'), expect.stringContaining('FLAME 2023 Open')]);
    expect(radios[0].getAttribute('aria-checked')).toBe('true');
    expect(radios[1].disabled).toBe(true);
    expect(radios[1].textContent).toMatch(/was not found/);
  });

  it('rebuilds the mesh from the slider values and hands it to the viewport', async () => {
    vi.useFakeTimers();
    await mount(); await open('topology-section'); await flush();
    expect(onMesh).not.toHaveBeenCalled();
    await change(slider(host, 'Semantic Jawline Width'), '1.5');
    expect(host.textContent).toContain('+6.0 mm');
    await act(async () => { vi.advanceTimersByTime(350); });
    await flush();
    const call = api.mock.calls.find(([path]) => String(path).includes('/identity-asset/mesh'))!;
    expect(call[0]).toBe('/facesets/fs1/identity-asset/mesh?template=makehuman&topology=makehuman_cc0&jaw_width=1.5');
    expect(onMesh).toHaveBeenCalledWith(mesh);
    expect(host.querySelector('[data-testid=topology-provenance]')?.textContent).toMatch(/semantic sliders applied/);
  });

  it('shows the server error instead of failing silently', async () => {
    vi.useFakeTimers();
    api.mockImplementation(async (path: string) => { if (path.startsWith('/morphable-topologies')) return report; throw new Error('Build the 3D identity asset first'); });
    await mount(); await open('topology-section'); await flush();
    await change(slider(host, 'Semantic Chin Projection'), '-2');
    await act(async () => { vi.advanceTimersByTime(350); });
    await flush();
    expect(host.querySelector('[role=alert]')?.textContent).toContain('Build the 3D identity asset first');
    expect(onMesh).not.toHaveBeenCalled();
  });

  it('sliders are disabled until a FaceSet exists', async () => {
    await mount({ facesetId: '' }); await open('topology-section'); await flush();
    expect(slider(host, 'Semantic Jawline Width').disabled).toBe(true);
  });

  it('beta blend switches to the adaptive skull scope and sets the real skull_blend', async () => {
    await mount(); await open('topology-section'); await flush();
    const beta = slider(host, 'Source to target beta blend');
    expect(beta.value).toBe('100');
    await change(beta, '30');
    expect(latest.parameters).toMatchObject({ swap_scope: 'adaptive_skull', skull_blend: 0.3 });
    expect(host.textContent).toContain('β source 30% · β target 70%');
  });

  it('reset clears the preview mesh', async () => {
    vi.useFakeTimers();
    await mount(); await open('topology-section'); await flush();
    await change(slider(host, 'Semantic Nose Bridge'), '1');
    await act(async () => { vi.advanceTimersByTime(350); });
    await flush();
    onMesh.mockClear();
    await act(async () => { [...host.querySelectorAll('button')].find(b => b.textContent === 'Reset shape')!.click(); });
    expect(onMesh).toHaveBeenCalledWith(null);
    expect(slider(host, 'Semantic Nose Bridge').value).toBe('0');
  });
});
