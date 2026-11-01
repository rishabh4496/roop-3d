// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import PreFlightLinterModal from './components/diagnostics/PreFlightLinterModal';
import PipelinePill from './components/hud/PipelinePill';
import QuickTour from './components/tour/QuickTour';
import { TOUR_STEPS, TOUR_STORAGE_KEY, tourSeen } from './components/tour/tourSteps';
import { PRESETS, validLayout } from './components/workstation/WorkspaceLayouts';

const post = vi.fn();
vi.mock('./api', () => ({ post: (...args: unknown[]) => post(...args), api: vi.fn(async () => ({})) }));

const Linter = PreFlightLinterModal as React.ComponentType<any>;
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

let host: HTMLDivElement, root: Root, consoleError: ReturnType<typeof vi.spyOn>;
beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  post.mockReset();
  consoleError = vi.spyOn(console, 'error');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  document.body.querySelectorAll('[data-tour]').forEach(node => node.remove());
  // Zero console errors is part of the Stage 7 gate.
  expect(consoleError.mock.calls.map((call: unknown[]) => String(call[0]))).toEqual([]);
  vi.restoreAllMocks();
});

const flush = () => act(async () => { await new Promise(resolve => setTimeout(resolve, 20)); });
const click = async (element: Element | null | undefined) => { expect(element).toBeTruthy(); await act(async () => (element as HTMLElement).click()); };
const button = (label: string) => Array.from(host.querySelectorAll('button')).find(b => b.textContent?.includes(label));

const CHECKS = (statuses: Partial<Record<string, any>> = {}) => ({
  status: 'WARN', ready_to_queue: true,
  checks: [
    { id: 'trt_dynamic_models', name: 'TensorRT Dynamic-Shape Models', status: 'WARN', message: 'TensorRT is selected but _dynamic.onnx models are missing.', details: ['inswapper_128.onnx -> inswapper_128_dynamic.onnx not found'],
      action: { id: 'optimize_models', label: 'Convert & Optimize Now', method: 'POST', endpoint: '/system/optimize-models', models: ['inswapper_128.onnx'] }, ...statuses.trt },
    { id: 'frontalization_vs_pose_matrix', name: 'Pose Frontalization vs 3x3 Pose Matrix', status: 'WARN', message: 'Over-rotation risk.', details: [],
      action: { id: 'set_parameter', label: 'Lower frontalization to 0.5', parameter: 'pose_frontalization', value: 0.5 } },
    { id: 'vram_threads', name: 'VRAM vs Execution Threads', status: 'WARN', message: 'Recommended: 2 thread(s).', details: [],
      action: { id: 'set_setting', label: 'Use 2 thread(s)', settings: { execution_threads: 2 } } },
  ],
});

const mountLinter = async (extra: Record<string, unknown> = {}) => {
  await act(async () => root.render(<Linter isOpen onClose={() => {}} onConfirmQueue={() => {}} targetPath="t.mp4" inspectionId="" faceMappings={[]} {...extra} />));
  await flush();
};

describe('Pre-flight linter one-click fixes', () => {
  it('runs “Convert & Optimize Now”, reports the result and re-lints', async () => {
    post.mockImplementation(async (url: string) => url === '/system/preflight-lint' ? CHECKS()
      : { converted: 1, results: [{ model: 'inswapper_128.onnx', state: 'CONVERTED', output: 'inswapper_128_dynamic.onnx' }], remaining: [] });
    await mountLinter();
    expect(host.textContent).toContain('TensorRT Dynamic-Shape Models');
    const lintCalls = () => post.mock.calls.filter(([url]) => url === '/system/preflight-lint').length;
    expect(lintCalls()).toBe(1);
    await click(host.querySelector('[data-testid="lint-action-trt_dynamic_models"]'));
    await flush();
    expect(post).toHaveBeenCalledWith('/system/optimize-models', { models: ['inswapper_128.onnx'] });
    expect(host.textContent).toContain('1 model(s) converted.');
    expect(lintCalls()).toBe(2); // the fix is verified by running the lint again
  });

  it('surfaces a conversion that did not complete as an error, not a success', async () => {
    post.mockImplementation(async (url: string) => url === '/system/preflight-lint' ? CHECKS()
      : { converted: 0, results: [{ model: 'inswapper_128.onnx', state: 'FAILED', reason: 'boom' }], remaining: [{ model: 'inswapper_128.onnx' }] });
    await mountLinter();
    await click(host.querySelector('[data-testid="lint-action-trt_dynamic_models"]'));
    await flush();
    const alert = host.querySelector('.check-action-row [role="alert"]');
    expect(alert?.textContent).toContain('inswapper_128.onnx: FAILED');
  });

  it('applies parameter and setting fixes through the host callbacks', async () => {
    post.mockResolvedValue(CHECKS());
    const onApplyParameter = vi.fn(), onApplySettings = vi.fn(async () => {});
    await mountLinter({ onApplyParameter, onApplySettings });
    await click(host.querySelector('[data-testid="lint-action-frontalization_vs_pose_matrix"]'));
    expect(onApplyParameter).toHaveBeenCalledWith('pose_frontalization', 0.5);
    await click(host.querySelector('[data-testid="lint-action-vram_threads"]'));
    await flush();
    expect(onApplySettings).toHaveBeenCalledWith({ execution_threads: 2 });
  });

  it('reports a fix that cannot be applied from this screen', async () => {
    post.mockResolvedValue(CHECKS());
    await mountLinter();
    await click(host.querySelector('[data-testid="lint-action-vram_threads"]'));
    expect(host.querySelector('.check-action-row [role="alert"]')?.textContent).toContain('not available');
  });
});

describe('Quick Tour', () => {
  const addTarget = (name: string) => { const node = document.createElement('div'); node.setAttribute('data-tour', name); document.body.append(node); node.getBoundingClientRect = () => ({ top: 10, left: 20, width: 200, height: 100, right: 220, bottom: 110, x: 20, y: 10, toJSON() {} }) as DOMRect; };

  it('walks the four steps in order, navigating each step’s screen and highlighting its target', async () => {
    ['source', 'tracks', 'pipeline-bar', 'run-pill'].forEach(addTarget);
    const onNavigate = vi.fn(), onClose = vi.fn();
    await act(async () => root.render(<QuickTour open onClose={onClose} onNavigate={onNavigate} />));
    expect(TOUR_STEPS.map(step => step.id)).toEqual(['source', 'target', 'preview', 'render']);
    const seen: string[] = [], text: string[] = [];
    for (let i = 0; i < 4; i++) {
      const card = host.querySelector('.tour-card')!;
      seen.push(card.getAttribute('data-step')!);
      text.push(card.textContent!);
      expect(card.textContent).toContain(`STEP ${i + 1} OF 4`);
      expect(host.querySelector('[data-testid="tour-spotlight"]')).toBeTruthy();
      if (i < 3) await click(button('Next'));
    }
    expect(seen).toEqual(['source', 'target', 'preview', 'render']);
    expect(text[0]).toContain('Use Face From Frame');
    expect(text[0]).toContain('.fsz');
    expect(text[1]).toContain('Yaw / Pitch / Roll');
    expect(text[2]).toContain('Split Comparison');
    expect(text[2]).toContain('3×3 Parameter Sweep');
    expect(text[3]).toContain('Pre-Scan, Range Lock, then Render & Mux');
    expect(text[3]).toContain('persistent backend queue');
    expect(onNavigate).toHaveBeenCalledWith('#/faceswap');
    expect(button('Done')).toBeTruthy();
    await click(button('Done'));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(tourSeen()).toBe(true);
    expect(localStorage.getItem(TOUR_STORAGE_KEY)).toBe('1');
  });

  it('supports keyboard navigation and Escape, and renders nothing while closed', async () => {
    const onClose = vi.fn();
    await act(async () => root.render(<QuickTour open={false} onClose={onClose} onNavigate={() => {}} />));
    expect(host.querySelector('[data-testid="quick-tour"]')).toBeNull();
    await act(async () => root.render(<QuickTour open onClose={onClose} onNavigate={() => {}} />));
    await act(async () => { window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' })); });
    expect(host.querySelector('.tour-card')!.getAttribute('data-step')).toBe('target');
    await act(async () => { window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft' })); });
    expect(host.querySelector('.tour-card')!.getAttribute('data-step')).toBe('source');
    await act(async () => { window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' })); });
    expect(onClose).toHaveBeenCalled();
  });

  it('still shows the step (centred, with a note) when its target is not on screen', async () => {
    await act(async () => root.render(<QuickTour open onClose={() => {}} onNavigate={() => {}} />));
    expect(host.querySelector('[data-testid="tour-spotlight"]')).toBeNull();
    expect(host.querySelector('.tour-note')).toBeTruthy();
    expect(host.querySelector('[role="dialog"]')!.getAttribute('aria-modal')).toBe('true');
  });

  it('survives blocked browser storage', async () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked'); });
    const onClose = vi.fn();
    await act(async () => root.render(<QuickTour open onClose={onClose} onNavigate={() => {}} />));
    await click(host.querySelector('[aria-label="Close tour"]'));
    expect(onClose).toHaveBeenCalled();
    expect(tourSeen()).toBe(false);
  });
});

describe('Workspace layouts and HUD flamegraph', () => {
  it('ships the three named workspaces and keeps layouts valid', () => {
    expect(Object.keys(PRESETS).slice(0, 3)).toEqual(['Standard', '3D Pose & Geometry Pro', 'Masking & VFX Compositor']);
    expect(PRESETS['3D Pose & Geometry Pro'].order[0]).toBe('pose');
    expect(PRESETS['Masking & VFX Compositor'].panels.pose.collapsed).toBe(true);
    for (const layout of Object.values(PRESETS)) expect(validLayout(layout)).toEqual(layout);
  });

  it('embeds the measured flamegraph in the expanded diagnostics HUD', () => {
    const html = renderToStaticMarkup(<PipelinePill telemetry={null} activeJob={{ job_id: 'abcdef12', state: 'running', progress: { frame: 3, fps: 20, stage: 'render', stage_timings_ms: { decode: 2, inswapper: 9 } } }} defaultExpanded />);
    expect(html).toContain('data-testid="hud-flamegraph"');
    expect(html).toContain('job abcdef12');
    expect(html).toContain('Inswapper');
  });
});
