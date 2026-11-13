// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import PreFlightLinterModal from './components/diagnostics/PreFlightLinterModal';
import VfxExportPanel, { DEFAULT_EXPORT } from './components/export/VfxExportPanel';
import { post } from './api';

const Linter = PreFlightLinterModal as React.ComponentType<any>;
const Panel = VfxExportPanel as React.ComponentType<any>;

vi.mock('./api', () => ({
  api: vi.fn(() => Promise.resolve({})),
  post: vi.fn(() => Promise.resolve({
    status: 'WARN', ready_to_queue: true,
    checks: [
      { id: 'fsz_extreme_yaw', name: '.fsz Extreme-Yaw Coverage', status: 'WARN', message: '1 extreme-yaw angle(s) the target reaches are missing', details: ["Track 't0' turns 62 deg left but FaceSet 'alice' it has no left-facing view; add a view near 62 deg left in the FaceSet Builder."] },
      { id: 'trt_engines_built', name: 'TensorRT Engines Built', status: 'WARN', message: 'No cached TensorRT engine for ultramax.', details: ['Engine cached: inswapper_128', 'Engine missing: ultramax'] },
      { id: 'ultramax_on_4k', name: 'UltraMax on 4K Targets', status: 'WARN', message: 'The target is 4K but UltraMax is off.', details: [], action: { id: 'set_parameter', label: 'Use UltraMax', parameter: 'restorer_model', value: 'ultramax' } },
      { id: 'storage_disk_space', name: 'Storage & Free Disk Space', status: 'FAIL', message: 'Not enough disk space: about 41.2 GB needed, 12.0 GB free on the output volume.', details: [] },
    ],
  })),
  mediaURL: (value: string) => value,
}));

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let host: HTMLDivElement, root: Root;
beforeEach(() => { host = document.createElement('div'); document.body.append(host); root = createRoot(host); vi.mocked(post).mockClear(); });
afterEach(async () => { await act(async () => root.unmount()); host.remove(); document.body.innerHTML = ''; });
const flush = async () => { await act(async () => { await Promise.resolve(); await Promise.resolve(); }); };

describe('pre-flight linter modal', () => {
  const mount = async (props: Record<string, unknown> = {}) => {
    act(() => root.render(<Linter isOpen targetPath="clip.mp4" inspectionId={'a'.repeat(32)} faceMappings={[]} workstationParameters={{ restoration_blend: 0 }}
      swapperModelId="hyperswap_1a_256" exportPasses={['alpha', 'relit']} {...props} />));
    await flush();
  };

  it('sends the swapper and the export passes so the TensorRT and disk estimates are about THIS render', async () => {
    await mount();
    const [url, body] = vi.mocked(post).mock.calls[0] as [string, Record<string, unknown>];
    expect(url).toBe('/system/preflight-lint');
    expect(body).toMatchObject({ swapper_model_id: 'hyperswap_1a_256', export_passes: ['alpha', 'relit'], target_path: 'clip.mp4' });
  });

  it('does not re-request the lint on every render when the optional context is omitted', async () => {
    await mount({ swapperModelId: undefined, exportPasses: undefined });
    await flush(); await flush();
    expect(vi.mocked(post)).toHaveBeenCalledTimes(1);
    expect((vi.mocked(post).mock.calls[0][1] as Record<string, unknown>).export_passes).toEqual([]);
  });

  it('shows the four readiness checks and blocks queueing on the disk failure', async () => {
    await mount();
    for (const name of ['.fsz Extreme-Yaw Coverage', 'TensorRT Engines Built', 'UltraMax on 4K Targets', 'Storage & Free Disk Space']) expect(host.textContent).toContain(name);
    expect(host.textContent).toContain('no left-facing view');
    expect(host.textContent).toContain('41.2 GB needed');
    const proceed = [...host.querySelectorAll('button')].find(button => /Blocked by Errors|Proceed/.test(button.textContent ?? ''))!;
    expect(proceed.textContent).toContain('Blocked by Errors');
    expect(proceed.disabled).toBe(true);
  });

  it('lists failures and warnings before passes so the scrolling list explains the overall status', async () => {
    vi.mocked(post).mockResolvedValueOnce({
      status: 'WARN', ready_to_queue: true,
      checks: [
        { id: 'a', name: 'Pass A', status: 'PASS', message: 'ok', details: [] },
        { id: 'b', name: 'Pass B', status: 'PASS', message: 'ok', details: [] },
        { id: 'c', name: 'Warn C', status: 'WARN', message: 'careful', details: [] },
        { id: 'd', name: 'Pass D', status: 'PASS', message: 'ok', details: [] },
        { id: 'e', name: 'Fail E', status: 'FAIL', message: 'broken', details: [] },
      ],
    });
    await mount();
    const order = [...host.querySelectorAll('.linter-check-card strong')].map(node => node.textContent);
    expect(order).toEqual(['Fail E', 'Warn C', 'Pass A', 'Pass B', 'Pass D']);
  });

  it('applies the one-click UltraMax fix through the parameter callback, then re-lints', async () => {
    const onApplyParameter = vi.fn();
    await mount({ onApplyParameter });
    await act(async () => host.querySelector<HTMLButtonElement>('[data-testid="lint-action-ultramax_on_4k"]')!.click());
    await flush();
    expect(onApplyParameter).toHaveBeenCalledWith('restorer_model', 'ultramax');
    expect(vi.mocked(post).mock.calls.length).toBeGreaterThanOrEqual(2);
  });
});

describe('clean relit face pass in the export panel', () => {
  const panel = (relight: boolean, passes: string[] = ['relit']) => {
    const onQueue = vi.fn();
    act(() => root.render(<Panel value={{ ...DEFAULT_EXPORT, passes }} onChange={() => {}} onQueue={onQueue} canQueue meshBlend relight={relight} jobs={[]} />));
    return { onQueue, queue: () => [...host.querySelectorAll('button')].find(b => b.textContent === 'Queue export render')! as HTMLButtonElement };
  };

  it('is offered next to the other passes', () => {
    panel(true);
    expect(host.textContent).toContain('Clean Relit Face Pass');
  });

  it('cannot be queued while relighting is off, and says why', () => {
    const { queue } = panel(false);
    expect(host.querySelector('[data-testid="relit-needs-relight"]')!.textContent).toContain('Turn on 3D Relighting');
    expect(queue().disabled).toBe(true);
  });

  it('can be queued once relighting is on', () => {
    const { queue } = panel(true);
    expect(host.querySelector('[data-testid="relit-needs-relight"]')).toBeNull();
    expect(queue().disabled).toBe(false);
  });

  it('does not nag about relighting when the pass is not selected', () => {
    const { queue } = panel(false, ['alpha']);
    expect(host.querySelector('[data-testid="relit-needs-relight"]')).toBeNull();
    expect(queue().disabled).toBe(false);
  });
});

describe('batch pre-flight', () => {
  const payload = (target: string, extra: Record<string, unknown> = {}) => ({ target_path: target, inspection_id: 'i', face_mappings: [{ track_id: 't0', faceset_id: 'alice', extra: 1 }], ...extra });

  it('lints each distinct target once, with the rack parameters, swapper and passes of the batch', async () => {
    const { lintBatch, lintRequestFor } = await import('./components/diagnostics/batchLint');
    const post = vi.fn(async (_url: string, _body: unknown) => ({ status: 'PASS', checks: [] }));
    const result = await lintBatch(post, [payload('a.mp4'), payload('a.mp4'), payload('b.mp4', { export: { passes: ['alpha'] } })], { rackParameters: { restorer_model: 'ultramax' }, swapperId: 'hyperswap_1a_256' });
    expect(post).toHaveBeenCalledTimes(2);
    expect(result.targets).toBe(2);
    const body = post.mock.calls[1][1] as Record<string, unknown>;
    expect(body).toMatchObject({ target_path: 'b.mp4', swapper_model_id: 'hyperswap_1a_256', export_passes: ['alpha'], workstation_parameters: { restorer_model: 'ultramax' } });
    expect(lintRequestFor(payload('a.mp4')).face_mappings).toEqual([{ track_id: 't0', faceset_id: 'alice' }]);   // only the string pairs the endpoint accepts
  });

  it('blocks on a FAIL, warns on a WARN, and never lets an unreachable linter stop the batch', async () => {
    const { lintBatch, describeLint } = await import('./components/diagnostics/batchLint');
    const post = vi.fn(async (_url: string, body: { target_path: string }) => {
      if (body.target_path === 'down.mp4') throw new Error('offline');
      return { checks: [{ id: 'storage_disk_space', name: 'Storage & Free Disk Space', status: body.target_path.endsWith('big.mp4') ? 'FAIL' : 'WARN', message: 'about 41 GB needed' }] };
    });
    const result = await lintBatch(post, [payload("C:\\clips\\big.mp4"), payload('ok.mp4'), payload('down.mp4')]);
    expect(result.failures.map(item => item.target)).toEqual(['big.mp4']);
    expect(result.warnings.map(item => item.target)).toEqual(['ok.mp4']);
    expect(result.unavailable).toBe(true);
    const text = describeLint(result);
    expect(text.block).toContain('Pre-flight blocked the batch');
    expect(text.block).toContain('about 41 GB needed');
    expect(text.note).toContain('1 pre-flight warning(s)');
    expect(text.note).toContain('unavailable');
    expect(describeLint({ failures: [], warnings: [], unavailable: false, targets: 1 })).toEqual({ block: '', note: '' });
  });
});
