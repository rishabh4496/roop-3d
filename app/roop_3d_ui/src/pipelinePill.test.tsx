// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import PipelinePill from './components/faceswap/PipelinePill';
import { STEPS, buildPillModel, elapsedSeconds, formatElapsed, isPipelineFrame } from './components/faceswap/pillModel';

const frame = (patch: Record<string, unknown> = {}) => ({
  type: 'pipeline.status', schema_version: 1, sequence: 1, job_id: 'job-1', stage: null,
  stages: { STAGE_1_SPARSE_PRESCAN: 'PENDING', STAGE_2_ACTIVE_PREPASS: 'PENDING', STAGE_3_MAIN_SWAP: 'PENDING' },
  stage_progress_pct: null, active_ranges_count: null, verified_target_frames: null, skipped_frames_pct: null, eta_seconds: null,
  fps: 0, engine_state: 'IDLE', engine_label: null, engine_model: null, engine_precision: null, engine_since: null, ...patch,
});
const running = (stage: string, done: string[], patch: Record<string, unknown> = {}) => frame({
  stage, stages: { STAGE_1_SPARSE_PRESCAN: 'PENDING', STAGE_2_ACTIVE_PREPASS: 'PENDING', STAGE_3_MAIN_SWAP: 'PENDING',
    ...Object.fromEntries(done.map(id => [id, 'DONE'])), [stage]: 'RUNNING' }, ...patch });

describe('pill model', () => {
  it('maps the three server stages onto Pre-Scan, Range Lock and Render & Mux', () => {
    expect(STEPS.map(step => [step.number, step.label, step.id])).toEqual([
      [1, 'Pre-Scan', 'STAGE_1_SPARSE_PRESCAN'], [2, 'Range Lock', 'STAGE_2_ACTIVE_PREPASS'], [3, 'Render & Mux', 'STAGE_3_MAIN_SWAP']]);
    const model = buildPillModel(running('STAGE_2_ACTIVE_PREPASS', ['STAGE_1_SPARSE_PRESCAN'], { stage_progress_pct: 41.5, skipped_frames_pct: 63.25, fps: 12.4 }), 0);
    expect(model.steps.map(step => step.status)).toEqual(['done', 'running', 'pending']);
    expect(model.steps[1]).toMatchObject({ active: true, progressPct: 41.5 });
    expect(model.steps[0].progressPct).toBeNull();                       // only the running step shows a progress bar
    expect(model).toMatchObject({ skippedFramesPct: 63.25, fps: 12.4, muxing: false, idle: false, banner: null });
  });

  it('treats a checkpoint-restored step as done, a failed step as failed, and unmeasured values as null (never 0)', () => {
    const model = buildPillModel(frame({ stages: { STAGE_1_SPARSE_PRESCAN: 'RESTORED', STAGE_2_ACTIVE_PREPASS: 'SKIPPED', STAGE_3_MAIN_SWAP: 'FAILED' } }), 0);
    expect(model.steps.map(step => step.status)).toEqual(['done', 'skipped', 'failed']);
    expect(model.skippedFramesPct).toBeNull();
    expect(model.etaSeconds).toBeNull();
    expect(buildPillModel(null, 0)).toMatchObject({ connected: false, idle: true, fps: 0, skippedFramesPct: null, banner: null });
  });

  it('reports Muxing only once every frame is rendered but the job has not finished', () => {
    const at = (pct: number | null) => buildPillModel(running('STAGE_3_MAIN_SWAP', ['STAGE_1_SPARSE_PRESCAN', 'STAGE_2_ACTIVE_PREPASS'], { stage_progress_pct: pct }), 0).muxing;
    expect([at(10), at(99.9), at(null)]).toEqual([false, false, false]);
    expect(at(100)).toBe(true);
    const finished = buildPillModel(frame({ stage: null, stages: { STAGE_1_SPARSE_PRESCAN: 'DONE', STAGE_2_ACTIVE_PREPASS: 'DONE', STAGE_3_MAIN_SWAP: 'DONE' }, stage_progress_pct: 100 }), 0);
    expect(finished.muxing).toBe(false);                                   // DONE is not "muxing"
  });

  it('builds the compilation banner with elapsed seconds computed from engine_since', () => {
    const since = 1_000_000;
    const compiling = buildPillModel(frame({ engine_state: 'COMPILING_TRT', engine_label: 'inswapper_128 fp16', engine_model: 'inswapper_128.onnx', engine_precision: 'fp16', engine_since: since }), (since + 75.9) * 1000);
    expect(compiling.banner).toMatchObject({ kind: 'compiling', title: 'Compiling TensorRT engine', label: 'inswapper_128 fp16', model: 'inswapper_128.onnx', precision: 'fp16', elapsedSeconds: 75 });
    expect(compiling.banner!.text).toBe('Compiling TensorRT engine · inswapper_128 fp16 · inswapper_128.onnx · fp16 · 1:15');
    expect(compiling.idle).toBe(false);
    const canary = buildPillModel(frame({ engine_state: 'RUNNING_CANARY', engine_since: since }), (since + 4) * 1000);
    expect(canary.banner).toMatchObject({ kind: 'canary', title: 'Running engine canary', elapsedSeconds: 4 });
    expect(elapsedSeconds(since, (since - 50) * 1000)).toBe(0);            // clock skew never yields negative elapsed
    expect(elapsedSeconds(null, 1000)).toBeNull();
    expect([formatElapsed(7), formatElapsed(60), formatElapsed(605), formatElapsed(null)]).toEqual(['7s', '1:00', '10:05', '']);
  });

  it('only recognises pipeline.status frames', () => {
    expect([isPipelineFrame(frame()), isPipelineFrame({ type: 'system.telemetry' }), isPipelineFrame(null), isPipelineFrame('x')]).toEqual([true, false, false, false]);
  });
});

class FakeSocket {
  static instances: FakeSocket[] = [];
  onopen: (() => void) | null = null; onmessage: ((event: { data: string }) => void) | null = null; onclose: (() => void) | null = null; onerror: (() => void) | null = null;
  closed = false;
  constructor(public url: string) { FakeSocket.instances.push(this); }
  close() { this.closed = true; }
  push(value: unknown) { this.onmessage?.({ data: typeof value === 'string' ? value : JSON.stringify(value) }); }
}

describe('PipelinePill component (/ws/telemetry)', () => {
  let host: HTMLDivElement, root: Root;
  const mount = (props: Record<string, unknown> = {}) => act(async () => {
    root.render(<PipelinePill makeSocket={(url: string) => new FakeSocket(url) as unknown as WebSocket} {...props}/>);
  });
  const socket = () => FakeSocket.instances.at(-1)!;
  const text = (selector: string) => host.querySelector(selector)?.textContent;

  beforeEach(() => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
    FakeSocket.instances = [];
    vi.useFakeTimers();
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.useRealTimers(); });

  it('subscribes to /ws/telemetry and closes the socket on unmount', async () => {
    await mount();
    expect(FakeSocket.instances).toHaveLength(1);
    expect(socket().url).toMatch(/^wss?:\/\/[^/]+\/ws\/telemetry$/);
    await act(async () => root.unmount());
    expect(socket().closed).toBe(true);
    root = createRoot(host);                                              // afterEach unmounts a fresh root safely
  });

  it('renders the three steps and updates live from telemetry frames', async () => {
    await mount();
    expect(host.querySelectorAll('.run-step')).toHaveLength(3);
    expect([...host.querySelectorAll('.step-label')].map(node => node.textContent)).toEqual(['Step 1 · Pre-Scan', 'Step 2 · Range Lock', 'Step 3 · Render & Mux']);
    expect(host.querySelector('.run-pill')!.getAttribute('data-state')).toBe('idle');
    expect(text('[data-testid="skipped-pct"]')).toBe('—');               // nothing measured yet: no fake zeros
    expect(text('[data-testid="fps"]')).toBe('—');

    await act(async () => socket().push(running('STAGE_1_SPARSE_PRESCAN', [], { stage_progress_pct: 40, skipped_frames_pct: 12.5 })));
    expect(host.querySelector('.run-step.running .step-label')!.textContent).toContain('Pre-Scan');
    expect(host.querySelector('.run-step.running')!.getAttribute('aria-current')).toBe('step');
    expect(text('.run-step.running .step-state')).toBe('40%');
    expect(text('[data-testid="skipped-pct"]')).toBe('12.5%');

    await act(async () => socket().push(running('STAGE_3_MAIN_SWAP', ['STAGE_1_SPARSE_PRESCAN', 'STAGE_2_ACTIVE_PREPASS'], { stage_progress_pct: 63, skipped_frames_pct: 70.04, fps: 23.46 })));
    expect([...host.querySelectorAll('.run-step')].map(step => step.getAttribute('data-status'))).toEqual(['done', 'done', 'running']);
    expect(text('[data-testid="skipped-pct"]')).toBe('70.0%');
    expect(text('[data-testid="fps"]')).toBe('23.5');
    expect(host.querySelector('[data-testid="muxing"]')).toBeNull();
    expect(host.querySelector('.run-pill')!.getAttribute('data-state')).toBe('running');

    await act(async () => socket().push(running('STAGE_3_MAIN_SWAP', ['STAGE_1_SPARSE_PRESCAN', 'STAGE_2_ACTIVE_PREPASS'], { stage_progress_pct: 100, fps: 30 })));
    expect(text('[data-testid="muxing"]')).toBe('Muxing');
    expect(text('.run-step.running .step-state')).toBe('Muxing…');
    expect(host.querySelector('.run-pill')!.getAttribute('data-state')).toBe('muxing');
  });

  it('shows a prominent compile banner whose elapsed time keeps ticking between frames', async () => {
    let wall = 2_000_000 * 1000;
    await mount({ now: () => wall });
    await act(async () => socket().push(frame({ engine_state: 'COMPILING_TRT', engine_label: 'inswapper_128 fp16', engine_model: 'inswapper_128.onnx', engine_precision: 'fp16', engine_since: 2_000_000 - 9 })));
    const banner = host.querySelector('.compile-banner')!;
    expect(banner.getAttribute('role')).toBe('alert');
    expect(banner.textContent).toContain('Compiling TensorRT engine');
    expect(banner.textContent).toContain('inswapper_128 fp16');
    expect(text('.engine-model')).toBe('inswapper_128.onnx');
    expect(text('.engine-precision')).toBe('fp16');
    expect(text('.engine-elapsed')).toBe('9s elapsed');
    wall += 3000;
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });  // no new frame: the pill's own clock advances
    expect(text('.engine-elapsed')).toBe('12s elapsed');
    expect(host.querySelector('.run-pill')!.getAttribute('data-state')).toBe('compiling');

    await act(async () => socket().push(frame({ engine_state: 'RUNNING_CANARY', engine_label: 'canary', engine_since: 2_000_003 })));
    expect(text('.compile-banner strong')).toBe('Running engine canary');
    expect(host.querySelector('.run-pill')!.getAttribute('data-state')).toBe('canary');

    await act(async () => socket().push(frame()));                         // engine finished
    expect(host.querySelector('.compile-banner')).toBeNull();
  });

  it('ignores other event types and malformed payloads without losing the last good frame', async () => {
    await mount();
    await act(async () => socket().push(running('STAGE_1_SPARSE_PRESCAN', [], { skipped_frames_pct: 5 })));
    await act(async () => { socket().push({ type: 'system.telemetry', hardware: {} }); socket().push('not json'); socket().push({ type: 'job.progress', frame: 1 }); });
    expect(text('[data-testid="skipped-pct"]')).toBe('5.0%');
    expect(host.querySelectorAll('.run-step.running')).toHaveLength(1);
  });

  it('reports the link state and reconnects after the socket drops', async () => {
    await mount();
    expect(host.querySelector('.run-link')!.getAttribute('aria-label')).toBe('Telemetry connecting');
    await act(async () => socket().onopen?.());
    expect(host.querySelector('.run-link')!.getAttribute('aria-label')).toBe('Telemetry online');
    await act(async () => socket().onclose?.());
    expect(host.querySelector('.run-link')!.getAttribute('aria-label')).toBe('Telemetry reconnecting');
    await act(async () => { await vi.advanceTimersByTimeAsync(600); });
    expect(FakeSocket.instances).toHaveLength(2);                          // exponential-backoff reconnect
  });
});
