import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { SettingsInspector } from './SettingsInspector';
import type { Capabilities, FaceSet, JobState, Track } from './contracts';
import { EventConnection } from './events';
import { featureReason, jobActions, mappings, optionReason, PreviewURL } from './state';

const states: JobState[] = ['queued', 'preparing', 'running', 'pausing', 'paused', 'cancelling', 'cancelled', 'succeeded', 'failed', 'interrupted'];
const caps: Capabilities = { enabled_modes: [], rendering_worker_available: true, models: [], inspection_parameter_bounds: {}, job_parameter_bounds: {},
  legal_parameter_bounds: { properties: { execution_mode: { enum: ['swap', 'diffusion_vfx'] }, precision_policy: { enum: ['fp32', 'fp16'] }, remote_processing_enabled: { type: 'boolean' }, port: { type: 'integer' } } },
  workstation: { features: { queue: { enabled: true, reason: null }, face_inspection: { enabled: false, reason: 'Detector unavailable' } }, controls: { execution_mode: { swap: null, diffusion_vfx: 'No verified loader' }, precision_policy: { fp32: null, fp16: 'Graph requires FP32' } }, read_only_parameters: { port: 'Offline transport setting' }, disabled_parameters: { remote_processing_enabled: 'No remote adapter' }, job_states: states,
    job_controls: { pause: ['queued', 'preparing', 'running'], resume: ['paused', 'interrupted'], cancel: ['queued', 'preparing', 'running', 'pausing', 'paused', 'interrupted'] }, upload_chunk_bytes: 1048576, max_asset_bytes: 8589934592 } };

afterEach(() => { vi.useRealTimers(); });
describe('capability and config gates', () => {
  it('fails closed before capability load and for unknown features/options', () => {
    expect(featureReason(null, 'queue')).toBeTruthy();
    expect(featureReason(caps, 'unadvertised')).toBeTruthy();
    expect(featureReason(caps, 'queue')).toBeNull();
    expect(featureReason(caps, 'face_inspection')).toBe('Detector unavailable');
    expect(optionReason(caps, 'execution_mode', 'unknown')).toBeTruthy();
    expect(optionReason(caps, 'precision_policy', 'fp16')).toBe('Graph requires FP32');
  });
  it('renders schema controls with unavailable choices and reasons', () => {
    const html = renderToStaticMarkup(<SettingsInspector caps={caps} config={{ execution_mode: 'swap', precision_policy: 'fp32', port: 8030, remote_processing_enabled: false }} busy={false} save={() => { throw new Error('SSR must not mutate settings'); }}/>);
    expect(html).toContain('value="diffusion_vfx" disabled=""');
    expect(html).toContain('No verified loader');
    expect(html).toContain('Graph requires FP32');
    expect(html).toMatch(/id="setting-port"[^>]*disabled=""/);
    expect(html).toContain('No remote adapter');
  });
  it.each(states)('uses backend actions for job state %s', state => {
    const expected = Object.entries(caps.workstation.job_controls).filter(([, allowed]) => allowed.includes(state)).map(([action]) => action);
    expect(jobActions(caps, state)).toEqual(expected);
    if (['succeeded', 'failed', 'cancelled', 'cancelling'].includes(state)) expect(jobActions(caps, state)).toEqual([]);
  });
  it('does not invent job controls when capability contract is missing', () => expect(jobActions(null, 'running')).toEqual([]));
});

it('maps only known target tracks to loaded FaceSets and preserves keep-original', () => {
  const tracks = [{ track_id: 'track_0000' }, { track_id: 'track_0001' }] as Track[];
  const sources = [{ faceset_id: 'alice' }] as FaceSet[];
  expect(mappings({ track_0000: 'alice', track_0001: '', track_9999: 'alice' }, tracks, sources)).toEqual([{ track_id: 'track_0000', faceset_id: 'alice' }]);
  expect(mappings({ track_0000: 'alice' }, tracks, [])).toEqual([]);
});

it('revokes every preview URL during repeated scrubs and cleanup', () => {
  let serial = 0;
  const create = vi.fn(() => `blob:preview-${serial++}`), revoke = vi.fn();
  const resource = new PreviewURL(create, revoke);
  for (let index = 0; index < 100; index++) resource.replace(new Blob(['frame']));
  expect(revoke).toHaveBeenCalledTimes(99);
  resource.clear(); resource.clear();
  expect(revoke).toHaveBeenCalledTimes(100);
  expect(new Set(revoke.mock.calls.map(([url]) => url)).size).toBe(100);
});

describe('WebSocket lifecycle', () => {
  function fixture() {
    const sockets: WebSocket[] = [];
    const receive = vi.fn(), status = vi.fn();
    const connection = new EventConnection('ws://127.0.0.1:8030/api/v1/ws/events', receive, status, () => {
      const socket = { onopen: null, onmessage: null, onclose: null, onerror: null, close: vi.fn() } as unknown as WebSocket;
      sockets.push(socket); return socket;
    });
    return { connection, sockets, receive, status };
  }
  it('reconnects and resynchronizes without leaking after disposal', () => {
    vi.useFakeTimers();
    const { connection, sockets, receive, status } = fixture();
    connection.start(); const first = sockets[0];
    first.onopen?.call(first, new Event('open'));
    first.onmessage?.call(first, new MessageEvent('message', { data: JSON.stringify({ type: 'stream.ready', sequence: 1 }) }));
    expect(receive).toHaveBeenCalledWith({ type: 'stream.ready', sequence: 1 });
    first.onclose?.call(first, new Event('close') as CloseEvent);
    vi.advanceTimersByTime(499); expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(1); expect(sockets).toHaveLength(2);
    sockets[1].onmessage?.call(sockets[1], new MessageEvent('message', { data: JSON.stringify({ type: 'job.progress', sequence: 4 }) }));
    expect(receive).toHaveBeenCalledWith({ type: 'stream.gap' });
    connection.stop();
    sockets[1].onclose?.call(sockets[1], new Event('close') as CloseEvent);
    vi.advanceTimersByTime(30000); expect(sockets).toHaveLength(2);
    expect(status).toHaveBeenLastCalledWith('offline');
    expect(sockets[1].close).toHaveBeenCalledOnce();
  });
  it('bounds backoff and cancels failed connection retry timers', () => {
    vi.useFakeTimers();
    const create = vi.fn(() => { throw new Error('connection failed'); });
    const connection = new EventConnection('ws://127.0.0.1/ws', vi.fn(), vi.fn(), create);
    connection.start(); vi.advanceTimersByTime(31500);
    expect(create.mock.calls.length).toBeGreaterThan(4);
    expect(create.mock.calls.length).toBeLessThan(10);
    connection.stop(); const attempts = create.mock.calls.length;
    vi.advanceTimersByTime(60000); expect(create).toHaveBeenCalledTimes(attempts);
  });
  it('requests a refresh for malformed events', () => {
    const { connection, sockets, receive } = fixture(); connection.start();
    sockets[0].onmessage?.call(sockets[0], new MessageEvent('message', { data: '{bad' }));
    expect(receive).toHaveBeenCalledWith({ type: 'stream.gap' }); connection.stop();
  });
});
