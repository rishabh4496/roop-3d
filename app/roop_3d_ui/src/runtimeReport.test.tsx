// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import EngineStatusBar from './components/hud/EngineStatusBar';
import RuntimeReportPanel from './components/diagnostics/RuntimeReportPanel';
import { ingestFrame, onConnectionState, parseTrtFrame, pipelineStream, resetPipelineStream, TRT_BUILD_MAX_S, TRT_VRAM_FRESH_MS, trtElapsedSeconds } from './stores/pipelineStream';
import { pipelineStatusStore } from './stores/pipelineStatus';
import { clearToasts, toastStore } from './stores/toasts';
import { copyText, needsAttention, parseBenchmark, parseRuntimeReport } from './utils/runtimeReport';

const api = vi.fn();
const post = vi.fn();
vi.mock('./api', () => ({ api: (...args: unknown[]) => api(...args), post: (...args: unknown[]) => post(...args) }));
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const TITLES = ['Host / OS / Venv Isolation', 'CPU Cores & Worker Pool Allocation', 'GPU Architecture & Compute Capability', 'Execution Provider Chain & Mixed Precision',
  'TensorRT Engine Cache', 'Zero-Copy OrtValue Ring Buffer', 'VRAM Budget & OOM-Guard Thresholds', 'Model Governance & SHA-256 Verification',
  '3D Head Topology & Anatomical Sub-Mesh', 'Full-Craniofacial Swap Lock', 'Physical Camera & SH Relighting', 'GOP Video Slicer & Audio Remux',
  'Offline Isolation Audit', 'Loopback Server & Deep-Link URLs'];
const STATUS = ['OK', 'INFO', 'OK', 'OK', 'INFO', 'UNAVAILABLE', 'OK', 'WARN', 'OK', 'OK', 'INFO', 'OK', 'OK', 'OK'];
const reportBody = () => ({
  report_id: 'r3d-1', generated_at: '2026-11-11T10:00:00+00:00', version: '0.1.0', trigger: 'on demand', overall: 'ATTENTION',
  counts: { OK: 8, INFO: 3, WARN: 1, UNAVAILABLE: 1, UNKNOWN: 0 },
  sections: TITLES.map((title, index) => ({ number: index + 1, key: `s${index + 1}`, title, status: STATUS[index],
    rows: [{ label: 'Row A', value: `value ${index + 1}` }], notes: STATUS[index] === 'WARN' ? ['6 models have no recorded hash'] : [] })),
  text: '╔ ROOP 3D RUNTIME REPORT ╗\n' + TITLES.map((title, index) => `┌ ${index + 1} · ${title} ┐`).join('\n'),
});

let host: HTMLDivElement, root: Root;
beforeEach(() => {
  api.mockReset(); post.mockReset(); api.mockRejectedValue(new Error('offline')); resetPipelineStream(); pipelineStatusStore.reset(); clearToasts();
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.useRealTimers(); vi.restoreAllMocks(); });
const flush = () => act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); });
const byId = (id: string) => host.querySelector(`[data-testid="${id}"]`) as HTMLElement | null;
const open = async () => {
  const panel = byId('runtime-report-panel') as HTMLDetailsElement;
  await act(async () => { panel.open = true; panel.dispatchEvent(new Event('toggle')); });
  await flush();
};
const apiRoutes = (overrides: Record<string, unknown> = {}) => api.mockImplementation(async (path: string) => {
  if (path in overrides) { const value = overrides[path]; if (value instanceof Error) throw value; return value; }
  if (path === '/system/runtime-report') return reportBody();
  if (path === '/settings/benchmark_status') return { state: 'IDLE', results: [], recommendation: null };
  throw new Error(`unexpected GET ${path}`);
});

describe('trt_compilation_progress frames', () => {
  const frame = { type: 'trt_compilation_progress', status: 'compiling', model: 'inswapper_128.onnx', precision: 'FP16', precision_label: 'FP16+FP32 Mix',
    opt_shape: '4x3x512x512', elapsed_seconds: 14, vram_gb: 3.21, vram_scope: 'process' };

  it('parses a build frame, clears on finished, and ignores everything malformed', () => {
    const parsed = parseTrtFrame(frame, 5);
    expect(parsed).toMatchObject({ finished: false, build: { model: 'inswapper_128.onnx', optShape: '4x3x512x512', elapsedSeconds: 14, vramGb: 3.21, vramScope: 'process', precisionLabel: 'FP16+FP32 Mix', receivedAt: 5 } });
    expect(parseTrtFrame({ ...frame, status: 'finished' })).toEqual({ finished: true });
    for (const bad of [null, 'x', {}, { type: 'other' }, { ...frame, model: '' }, { ...frame, model: 3 }]) expect(parseTrtFrame(bad)).toBeNull();
    const odd = parseTrtFrame({ ...frame, elapsed_seconds: -4, vram_gb: -1, vram_scope: 'gpu', opt_shape: null, precision_label: null });
    expect(odd).toMatchObject({ build: { elapsedSeconds: 0, vramGb: null, vramScope: null, optShape: null, precisionLabel: null } });
  });

  it('shows a live strip while compiling, with only measured values, and removes it when the build finishes', async () => {
    await act(async () => root.render(<EngineStatusBar />));
    expect(byId('trt-build')).toBeNull();
    await act(async () => ingestFrame(frame));
    const strip = byId('trt-build')!;
    expect(strip.getAttribute('role')).toBe('status');
    expect(strip.textContent).toContain('Compiling TensorRT engine inswapper_128.onnx (FP16+FP32 Mix · opt=4x3x512x512)');
    expect(byId('trt-build-clock')!.textContent).toBe('14.0 s · VRAM 3.2 GB');
    expect(byId('trt-build-clock')!.getAttribute('aria-hidden')).toBe('true');   // the ticking clock is not announced every second
    expect(strip.textContent).not.toMatch(/%/);                                 // TensorRT reports no progress, so none is shown
    await act(async () => ingestFrame({ ...frame, elapsed_seconds: 15 }));
    expect(byId('trt-build-clock')!.textContent).toContain('15.0 s');
    await act(async () => ingestFrame({ ...frame, status: 'finished', reason: 'ended' }));
    expect(byId('trt-build')).toBeNull();
  });

  it('says n/a for what the build path does not know, and labels device-wide VRAM', async () => {
    await act(async () => root.render(<EngineStatusBar />));
    await act(async () => ingestFrame({ ...frame, opt_shape: null, vram_gb: null, vram_scope: null, precision: null, precision_label: null }));
    expect(byId('trt-build')!.textContent).toContain('precision n/a · opt=n/a');
    expect(byId('trt-build-clock')!.textContent).toContain('VRAM n/a');
    await act(async () => ingestFrame({ ...frame, vram_scope: 'device' }));
    expect(byId('trt-build-clock')!.textContent).toContain('3.2 GB (device)');
  });

  it('counts its own clock while the server is silent (ONNX Runtime holds the GIL during a build) and stops claiming VRAM it no longer has', async () => {
    vi.useFakeTimers();
    await act(async () => root.render(<EngineStatusBar />));
    // The start frame is the last thing the server can send before the build takes the GIL: elapsed 0, no VRAM yet.
    await act(async () => ingestFrame({ ...frame, elapsed_seconds: 0, vram_gb: 2.0 }, Date.now()));
    expect(byId('trt-build-clock')!.textContent).toBe('0.0 s · VRAM 2.0 GB');
    expect(byId('trt-build-note')!.textContent).toBe('First use only; this can take 15-60 s.');
    await act(async () => { vi.advanceTimersByTime(TRT_VRAM_FRESH_MS + 3500); });
    expect(byId('trt-build')).not.toBeNull();
    const clock = byId('trt-build-clock')!.textContent!;
    expect(parseFloat(clock)).toBeGreaterThanOrEqual(5);                       // it kept counting with no frames at all
    expect(clock).toContain('VRAM n/a');                                       // a six-second-old figure is not shown as current
    expect(byId('trt-build-note')!.textContent).toContain('live VRAM is in the terminal');
    await act(async () => ingestFrame({ ...frame, elapsed_seconds: 20, vram_gb: 3.4 }, Date.now()));   // frames resume if the process can run again
    expect(byId('trt-build-clock')!.textContent).toBe('20.0 s · VRAM 3.4 GB');
  });

  it('believes a build only while the socket is up, and never past the 15 minute ceiling', async () => {
    vi.useFakeTimers();
    await act(async () => root.render(<EngineStatusBar />));
    await act(async () => ingestFrame(frame, Date.now()));
    expect(byId('trt-build')).not.toBeNull();
    await act(async () => onConnectionState('online'));
    expect(byId('trt-build')).not.toBeNull();
    await act(async () => onConnectionState('reconnecting'));   // the server went away mid-build, and took the build with it
    expect(byId('trt-build')).toBeNull();
    await act(async () => ingestFrame(frame, Date.now()));
    await act(async () => { vi.advanceTimersByTime((TRT_BUILD_MAX_S + 5) * 1000); });
    expect(byId('trt-build')).toBeNull();                                         // a finish frame that never came cannot pin the strip up
  });

  it('extrapolates elapsed time from the last frame only forwards', () => {
    const build = parseTrtFrame(frame, 1000);
    if (!build || build.finished) throw new Error('expected a build');
    expect(trtElapsedSeconds(build.build, 1000)).toBe(14);
    expect(trtElapsedSeconds(build.build, 3500)).toBe(16.5);
    expect(trtElapsedSeconds(build.build, 500)).toBe(14);   // a clock that went backwards never makes the build shorter
  });

  it('does not disturb model-download frames sharing the stream', async () => {
    await act(async () => ingestFrame({ type: 'model_download_progress', model_id: 'm', status: 'downloading', total_bytes: 10, downloaded_bytes: 5 }));
    await act(async () => ingestFrame(frame));
    expect(Object.keys(pipelineStream.get().transfers)).toEqual(['m']);
    expect(pipelineStream.get().trtBuild?.model).toBe('inswapper_128.onnx');
  });
});

describe('report parsing', () => {
  it('accepts a full report and defaults every missing field instead of throwing at render time', () => {
    const report = parseRuntimeReport(reportBody());
    expect(report.sections).toHaveLength(14);
    const sparse = parseRuntimeReport({ sections: [{ title: 'T', status: 'nonsense', rows: [{ label: 3 }, null], notes: [4, 'ok'] }, 7] });
    expect(sparse.sections).toEqual([{ number: 1, key: 'section-1', title: 'T', status: 'UNKNOWN', rows: [{ label: '', value: '' }], notes: ['ok'] }]);
    expect(sparse.overall).toBe('OK');
  });

  it('rejects responses that are not a report', () => {
    for (const bad of [null, 'x', {}, { sections: 'no' }, { sections: [] }]) expect(() => parseRuntimeReport(bad)).toThrow();
  });

  it('marks WARN, UNAVAILABLE and UNKNOWN sections as needing a look', () => {
    const sections = parseRuntimeReport(reportBody()).sections;
    expect(sections.filter(needsAttention).map(s => s.number)).toEqual([6, 8]);
  });

  it('parses benchmark status defensively', () => {
    expect(parseBenchmark(null)).toEqual({ state: 'IDLE', results: [], recommendation: null, profile: undefined, provider: undefined, note: undefined, error: undefined });
    const row = { execution_threads: 2, pool_size: 1, samples: 12, seconds: 0.5, samples_per_second: 24, provider: 'CUDAExecutionProvider' };
    const parsed = parseBenchmark({ state: 'DONE', profile: 'quick', results: [row, { execution_threads: 'x' }, null], recommendation: row, note: 'n' });
    expect(parsed.results).toEqual([row]);
    expect(parsed.recommendation).toEqual(row);
    expect(parseBenchmark({ state: 'WAT' }).state).toBe('IDLE');
  });
});

describe('copyText', () => {
  it('uses the async clipboard when available', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal('navigator', { clipboard: { writeText } });
    expect(await copyText('abc')).toBe(true);
    expect(writeText).toHaveBeenCalledWith('abc');
    vi.unstubAllGlobals();
  });

  it('falls back to a selected textarea when the clipboard API is missing or refuses, and cleans up after itself', async () => {
    vi.stubGlobal('navigator', { clipboard: { writeText: vi.fn().mockRejectedValue(new Error('denied')) } });
    document.execCommand = vi.fn().mockReturnValue(true);
    expect(await copyText('abc')).toBe(true);
    expect(document.execCommand).toHaveBeenCalledWith('copy');
    expect(document.querySelector('textarea')).toBeNull();
    (document.execCommand as ReturnType<typeof vi.fn>).mockReturnValue(false);
    expect(await copyText('abc')).toBe(false);
    vi.unstubAllGlobals();
  });
});

describe('RuntimeReportPanel', () => {
  it('reads nothing until it is opened (building the report probes hardware)', async () => {
    apiRoutes();
    await act(async () => root.render(<RuntimeReportPanel />));
    await flush();
    expect(api.mock.calls.map(call => call[0])).not.toContain('/system/runtime-report');
    expect(byId('rr-copy')!.hasAttribute('disabled')).toBe(true);
    await open();
    expect(api).toHaveBeenCalledWith('/system/runtime-report');
  });

  it('mirrors all fourteen sections, opening the ones that need attention and summarising the counts', async () => {
    apiRoutes();
    await act(async () => root.render(<RuntimeReportPanel />));
    await open();
    const sections = [...host.querySelectorAll('[data-testid^="rr-section-"]')] as HTMLDetailsElement[];
    expect(sections).toHaveLength(14);
    expect(sections.map(s => s.querySelector('.rr-title')!.textContent)).toEqual(TITLES);
    expect(sections.filter(s => s.open).map(s => s.dataset.testid)).toEqual(['rr-section-6', 'rr-section-8']);
    expect(sections[7].textContent).toContain('6 models have no recorded hash');
    expect(byId('rr-summary')!.textContent).toContain('ATTENTION');
    expect(byId('rr-summary')!.textContent).toContain('8 ok, 3 info, 1 warn, 1 unavailable');
  });

  it('copies the plain-text report with one click and confirms it, in the page and as a toast', async () => {
    apiRoutes();
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal('navigator', { clipboard: { writeText } });
    await act(async () => root.render(<RuntimeReportPanel />));
    await open();
    expect(byId('rr-copy')!.textContent).toBe('Copy Structured Runtime Report');
    await act(async () => { byId('rr-copy')!.click(); });
    await flush();
    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText.mock.calls[0][0]).toBe(reportBody().text);
    expect(byId('rr-copied')!.textContent).toContain('Copied the 14-section report');
    expect(toastStore.get().items.at(-1)).toMatchObject({ kind: 'ok' });
    vi.unstubAllGlobals();
  });

  it('says so when the browser blocks the copy, instead of claiming success', async () => {
    apiRoutes();
    vi.stubGlobal('navigator', { clipboard: { writeText: vi.fn().mockRejectedValue(new Error('denied')) } });
    document.execCommand = vi.fn().mockReturnValue(false);
    await act(async () => root.render(<RuntimeReportPanel />));
    await open();
    await act(async () => { byId('rr-copy')!.click(); });
    await flush();
    expect(byId('rr-copied')!.textContent).toContain('Copy was blocked');
    expect(toastStore.get().items.at(-1)).toMatchObject({ kind: 'error' });
    vi.unstubAllGlobals();
  });

  it('shows the failure and keeps Copy disabled when the report cannot be read', async () => {
    apiRoutes({ '/system/runtime-report': new Error('Local service returned 500') });
    await act(async () => root.render(<RuntimeReportPanel />));
    await open();
    expect(host.querySelector('[role=alert]')!.textContent).toContain('Local service returned 500');
    expect(byId('rr-copy')!.hasAttribute('disabled')).toBe(true);
  });

  it('refreshes on demand', async () => {
    apiRoutes();
    await act(async () => root.render(<RuntimeReportPanel />));
    await open();
    await act(async () => { byId('rr-refresh')!.click(); });
    await flush();
    expect(api.mock.calls.filter(call => call[0] === '/system/runtime-report')).toHaveLength(2);
  });
});

describe('thread / pool benchmark runner', () => {
  const row = (threads: number, pool: number, rate: number) => ({ execution_threads: threads, pool_size: pool, samples: 12, seconds: 12 / rate, samples_per_second: rate, provider: 'CUDAExecutionProvider' });

  it('starts a quick run, polls while it runs, and marks the fastest configuration with the server\'s own caveat', async () => {
    vi.useFakeTimers();
    let polls = 0;
    let started = false;
    api.mockImplementation(async (path: string) => {
      if (path === '/settings/benchmark_status') {
        if (!started) return { state: 'IDLE', results: [], recommendation: null };
        polls += 1;
        return polls < 3 ? { state: 'RUNNING', profile: 'quick', results: [row(1, 1, 20)], recommendation: null }
          : { state: 'DONE', profile: 'quick', provider: 'CUDAExecutionProvider', results: [row(1, 1, 20), row(2, 1, 41.5), row(2, 2, 30)], recommendation: row(2, 1, 41.5),
            note: 'Recommendation is for the measured synthetic workload, not model or video-pipeline FPS.' };
      }
      return { state: 'IDLE', results: [], recommendation: null };
    });
    post.mockImplementation(async () => { started = true; return { state: 'RUNNING', profile: 'quick', results: [], recommendation: null }; });
    await act(async () => root.render(<RuntimeReportPanel />));
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    await act(async () => { byId('rr-bench-quick')!.click(); });
    expect(post).toHaveBeenCalledWith('/settings/benchmark_threads', { profile: 'quick' });
    expect(byId('rr-bench-quick')!.hasAttribute('disabled')).toBe(true);
    expect(byId('rr-bench-state')!.textContent).toContain('Running quick benchmark');
    await act(async () => { await vi.advanceTimersByTimeAsync(3200); });
    expect(byId('rr-bench-state')!.textContent).toBe('Done');
    expect(byId('rr-bench-quick')!.hasAttribute('disabled')).toBe(false);
    const rows = [...host.querySelectorAll('[data-testid=rr-bench-table] tbody tr')];
    expect(rows).toHaveLength(3);
    expect(rows.filter(r => r.classList.contains('rr-best')).map(r => [...r.children].map(cell => cell.textContent))).toEqual([['2', '1', '41.5 ★']]);
    expect(byId('rr-bench-reco')!.textContent).toContain('Fastest measured: 2 execution thread(s) × pool 1 (41.5 samples/s, CUDA)');
    expect(byId('rr-bench-reco')!.textContent).toContain('not model or video-pipeline FPS');
    const callsAfterDone = api.mock.calls.length;
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(api.mock.calls.length).toBe(callsAfterDone);   // polling stops once the run is done
  });

  it('shows the server\'s reason when a job is rendering, and never starts a second run', async () => {
    apiRoutes();
    post.mockRejectedValue(new Error('Wait for active jobs before benchmarking GPU concurrency'));
    await act(async () => root.render(<RuntimeReportPanel />));
    await flush();
    await act(async () => { byId('rr-bench-full')!.click(); });
    await flush();
    expect(post).toHaveBeenCalledWith('/settings/benchmark_threads', { profile: 'full' });
    expect(host.querySelector('.rr-bench [role=alert]')!.textContent).toContain('Wait for active jobs');
    expect(byId('rr-bench-state')!.textContent).toBe('Not run yet');
  });

  it('reports a failed run by its error type and offers no recommendation', async () => {
    apiRoutes({ '/settings/benchmark_status': { state: 'FAILED', error: 'RuntimeError', results: [], recommendation: null } });
    await act(async () => root.render(<RuntimeReportPanel />));
    await flush();
    expect(byId('rr-bench-state')!.textContent).toBe('Failed (RuntimeError)');
    expect(byId('rr-bench-reco')).toBeNull();
  });
});
