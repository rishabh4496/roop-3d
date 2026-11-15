// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { FaceSet, Job, JobRequest, Track } from './contracts';
import { RENDER_FRAME_EVENT, buildLivePayload } from './livePreview';
import { DEFAULT_PARAMETERS, wireParameters } from './components/workstation/pipeline';
import FaceMapping from './components/studio/FaceMapping';
import RenderFooter from './components/studio/RenderFooter';
import { publishHardware, resetTelemetry } from './stores/telemetryStore';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let host: HTMLDivElement, root: Root;
beforeEach(() => { host = document.createElement('div'); document.body.append(host); root = createRoot(host); resetTelemetry(); });
afterEach(async () => { await act(async () => root.unmount()); host.remove(); });
const render = (node: React.ReactNode) => act(async () => root.render(node));

const job = (patch: Partial<Job> & { progress?: Partial<Job['progress']> }): Job => ({
  job_id: 'a'.repeat(32), state: 'running',
  request: { target_path: 't.mp4', inspection_id: 'b'.repeat(32), face_mappings: [], processing: 'local', remote_consent: false, trim_start_seconds: null, trim_end_seconds: null } as unknown as Job['request'],
  progress: { frame: 30, total_frames: 120, fps: 12.5, stage: 'rendering', ...patch.progress }, output: null, failure: null, ...patch, } as Job);

describe('RenderFooter', () => {
  it('shows live progress and fps, and a GPU pill once telemetry arrives', async () => {
    await render(<RenderFooter job={job({})} startBlockedReason={null} frameBlockedReason={null} onStart={vi.fn()} />);
    expect(host.querySelector('[data-testid="render-progress"]')?.getAttribute('data-percent')).toBe('25');
    expect(host.querySelector('[data-testid="render-fps"]')?.textContent).toBe('12.5 fps');
    expect(host.querySelector('[data-testid="render-status"]')?.textContent).toContain('frame 30 / 120');
    expect(host.querySelector('[data-testid="gpu-pill"]')?.textContent).toContain('telemetry offline');
    await act(async () => publishHardware({ gpus: [{ name: 'NVIDIA GeForce RTX 4070', total_vram_mib: 12288, free_vram_mib: 9216 }], logical_cpu_count: 16 }));
    expect(host.querySelector('[data-testid="gpu-pill"]')?.textContent).toContain('RTX 4070 · 3.0 GB / 12.0 GB');
  });

  it('start is gated by its reason, the frame button asks the viewport to render, and a finished job reads 100%', async () => {
    const onStart = vi.fn(), listener = vi.fn();
    window.addEventListener(RENDER_FRAME_EVENT, listener);
    await render(<RenderFooter job={null} startBlockedReason="Inspect the target faces first." frameBlockedReason={null} onStart={onStart} />);
    const start = Array.from(host.querySelectorAll('button')).find(item => item.textContent?.includes('Start full render')) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    expect(start.title).toBe('Inspect the target faces first.');
    await act(async () => (Array.from(host.querySelectorAll('button')).find(item => item.textContent?.includes('Render current frame')) as HTMLButtonElement).click());
    expect(listener).toHaveBeenCalledTimes(1);
    window.removeEventListener(RENDER_FRAME_EVENT, listener);
    await render(<RenderFooter job={job({ state: 'succeeded' })} startBlockedReason={null} frameBlockedReason={null} onStart={onStart} />);
    expect(host.querySelector('[data-testid="render-progress"]')?.getAttribute('data-percent')).toBe('100');
    await act(async () => (Array.from(host.querySelectorAll('button')).find(item => item.textContent?.includes('Start full render')) as HTMLButtonElement).click());
    expect(onStart).toHaveBeenCalledTimes(1);
  });
});

describe('FaceMapping', () => {
  const tracks = [{ track_id: 'track_0000', thumbnail_base64: 'AAAA', thumbnail_mime: 'image/png' }, { track_id: 'track_0001' }] as unknown as Track[];
  const sources: FaceSet[] = [{ faceset_id: 'alice', name: 'Alice', item_count: 7 }, { faceset_id: 'bob', name: 'Bob', item_count: 1 }];

  it('lists Source -> Target rows with the angle count and assigns through the row select', async () => {
    const onAssign = vi.fn();
    await render(<FaceMapping tracks={tracks} sources={sources} assigned={{ track_0000: 'alice' }} selectedTrack="track_0000" onSelectTrack={vi.fn()} onAssign={onAssign} />);
    expect(host.textContent).toContain('7 angles loaded');
    expect(host.querySelectorAll('.face-mapping-row').length).toBe(2);
    const select = host.querySelector('select[aria-label="Face 1 source"]') as HTMLSelectElement;
    await act(async () => { select.value = 'bob'; select.dispatchEvent(new Event('change', { bubbles: true })); });
    expect(onAssign).toHaveBeenCalledWith('track_0001', 'bob');
  });

  it('explains itself before any face is detected', async () => {
    await render(<FaceMapping tracks={[]} sources={sources} assigned={{}} selectedTrack={null} onSelectTrack={vi.fn()} onAssign={vi.fn()} />);
    expect(host.textContent).toContain('Inspect the target');
  });
});

describe('2D swapper / restorer options are not on the wire', () => {
  const request = { target_path: 't.mp4', inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }], processing: 'local' as const, remote_consent: false as const, trim_start_seconds: null, trim_end_seconds: null } as JobRequest;
  it('sends only the detector and true-3D choices; no swapper, restorer, fidelity or UltraMax key', () => {
    const payload = buildLivePayload({ request: { ...request, detector_model_id: 'scrfd' }, time: 1, fps: 25,
      parameters: { ...DEFAULT_PARAMETERS, head_topology_engine: 'FLAME-FullHead-3D' } })!;
    expect(payload).toMatchObject({ detector_model_id: 'scrfd', head_topology_engine: 'FLAME-FullHead-3D' });
    expect(Object.keys(payload).join(' ')).not.toMatch(/swapper|restorer|restoration(?!_weight)|fidelity|ultramax|frontaliz|tps_|pose_mode|render_mode/i);
  });
});

describe('controls the selection cannot act on stay off the wire', () => {
  const request = { target_path: 't.mp4', inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }], processing: 'local' as const, remote_consent: false as const, trim_start_seconds: null, trim_end_seconds: null } as JobRequest;
  it('drops the XSeg vote outside Robust occlusion and bands for Feather fusion', () => {
    const wire = wireParameters({ ...DEFAULT_PARAMETERS, xseg_occlusion_strength: .4, blend_mode: 'feather', pyramid_levels: 5 });
    expect(wire).not.toHaveProperty('xseg_occlusion_strength');
    expect(wire).not.toHaveProperty('pyramid_levels');
    expect(wire.blend_mode).toBe('feather');
    const kept = wireParameters({ ...DEFAULT_PARAMETERS, occlusion_mode: 'robust', xseg_occlusion_strength: .4, pyramid_levels: 5 });
    expect(kept).toMatchObject({ xseg_occlusion_strength: .4, pyramid_levels: 5 });
  });
  it('sends the fusion controls to the live preview only when they act and differ from the engine defaults', () => {
    const send = (parameters: Parameters<typeof buildLivePayload>[0]['parameters']) => buildLivePayload({ request, time: 1, fps: 25, parameters })!;
    const plain = send({ ...DEFAULT_PARAMETERS, xseg_occlusion_strength: .4, pyramid_levels: 5 });
    expect(plain).not.toHaveProperty('xseg_occlusion_strength', .4);       // standard occlusion runs no XSeg network
    expect(plain.xseg_occlusion_strength).toBeUndefined();
    const robust = send({ ...DEFAULT_PARAMETERS, occlusion_mode: 'robust', xseg_occlusion_strength: .4, blend_mode: 'poisson', pyramid_levels: 5 });
    expect(robust).toMatchObject({ occlusion_mode: 'robust', xseg_occlusion_strength: .4, blend_mode: 'poisson' });
    expect(robust.pyramid_levels).toBeUndefined();                         // Poisson fusion has no pyramid
    expect(send({ ...DEFAULT_PARAMETERS, occlusion_mode: 'robust', xseg_occlusion_strength: 1, blend_mode: 'multi_band', pyramid_levels: 4 })).not.toHaveProperty('blend_mode', 'multi_band');
  });
});
