// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import ParameterSweepModal from './components/workstation/ParameterSweepModal';
import { DEFAULT_PARAMETERS, sweepAxis } from './components/workstation/pipeline';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let host: HTMLDivElement, root: Root;
beforeEach(() => { host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.unstubAllGlobals(); });
const request = { target_path: 'local.mp4', inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }], processing: 'local', remote_consent: false };

it('dispatches one batch for the exact playhead and applies the selected coordinates', async () => {
  const apply = vi.fn(), close = vi.fn();
  const fetch = vi.fn(async (_url, options) => {
    const body = JSON.parse(options.body);
    const cells = body.y.values.flatMap((y: number, row: number) => body.x.values.map((x: number, column: number) => ({ row, column, coordinates: { [body.x.parameter]: x, [body.y.parameter]: y }, parameters: { ...body.parameters, [body.x.parameter]: x, [body.y.parameter]: y }, image: 'data:image/jpeg;base64,AQ==' })));
    return { ok: true, json: async () => ({ cells, timestamp: 1.25, frame_index: 31, notes: [] }) };
  });
  vi.stubGlobal('fetch', fetch);
  await act(async () => root.render(<ParameterSweepModal request={request} timestamp={1.25} parameters={DEFAULT_PARAMETERS} onApply={apply} onClose={close}/>));
  const generate = [...host.querySelectorAll('button')].find(button => button.textContent === 'Generate contact sheet')!;
  await act(async () => generate.click());
  expect(fetch).toHaveBeenCalledOnce(); expect(fetch.mock.calls[0][0]).toBe('/api/v1/workstation/preview');
  const payload = JSON.parse(fetch.mock.calls[0][1].body); expect(payload.timestamp).toBe(1.25); expect(payload.x.values).toEqual([.25, .5, .75]);
  const cells = host.querySelectorAll<HTMLButtonElement>('.sweep-cell'); expect(cells).toHaveLength(9);
  await act(async () => cells[7].click());
  expect(apply).toHaveBeenCalledWith({ ...DEFAULT_PARAMETERS, color_strength: .5, feather_px: 10 }); expect(close).toHaveBeenCalledOnce();
});

it('aborts and discards stale results when the playhead changes', async () => {
  let finish: (value: unknown) => void = () => {};
  const fetch = vi.fn((_url, _options) => new Promise(resolve => { finish = resolve; })); vi.stubGlobal('fetch', fetch);
  const props = { request, parameters: DEFAULT_PARAMETERS, onApply: vi.fn(), onClose: vi.fn() };
  await act(async () => root.render(<ParameterSweepModal {...props} timestamp={1}/>));
  await act(async () => [...host.querySelectorAll('button')].find(button => button.textContent === 'Generate contact sheet')!.click());
  const signal = (fetch.mock.calls[0][1] as RequestInit).signal!;
  await act(async () => root.render(<ParameterSweepModal {...props} timestamp={2}/>)); expect(signal.aborted).toBe(true);
  await act(async () => finish({ ok: true, json: async () => ({ cells: [{ image: 'stale' }], timestamp: 1, notes: [] }) }));
  expect(host.querySelectorAll('.sweep-cell')).toHaveLength(0);
});

it('rejects duplicate, blank, out-of-range and mismatched coordinates', () => {
  for (const values of ['0,0,1', '0,,1', '-1,0,1', '0,1', 'a,b,c']) expect(() => sweepAxis('color_strength', values)).toThrow();
  expect(sweepAxis('feather_px', '0, 5, 10').values).toEqual([0, 5, 10]);
});
