// @vitest-environment node
import { afterAll, beforeAll, expect, test } from 'vitest';
import { WebSocket } from 'ws';
import { startMockServer } from './server.ts';

let mock: Awaited<ReturnType<typeof startMockServer>>;
beforeAll(async () => { mock = await startMockServer(0); });
afterAll(async () => { await mock.close(); });

test('/api/meta identifies the mock and every response is marked', async () => {
  for (const path of ['/api/meta', '/api/v1/meta']) {
    const response = await fetch(mock.url + path);
    expect(response.status).toBe(200);
    expect(response.headers.get('x-mock-server')).toBe('true');
    expect(await response.json()).toEqual({ mock: true });
  }
  const missing = await fetch(mock.url + '/api/no-such-route');
  expect(missing.status).toBe(404);
  expect(missing.headers.get('x-mock-server')).toBe('true');
});

test('hardware, telemetry and health are served under unversioned aliases', async () => {
  const hardware = await (await fetch(mock.url + '/api/system/hardware')).json();
  expect(hardware.mock).toBe(true);
  expect(hardware).toHaveProperty('vram_total_gb');
  const telemetry = await (await fetch(mock.url + '/api/system/telemetry')).json();
  expect(telemetry.pipeline.type).toBe('pipeline.status');
  expect((await fetch(mock.url + '/api/health')).status).toBe(200);
});

test('pipeline status frames stream over the websocket route', async () => {
  const socket = new WebSocket(mock.url.replace('http', 'ws') + '/ws/v1/pipeline/status');
  const messages: any[] = [];
  await new Promise<void>((resolve, reject) => {
    socket.on('message', data => { messages.push(JSON.parse(String(data))); if (messages.length === 2) resolve(); });
    socket.on('error', reject);
  });
  socket.close();
  expect(messages[0].type).toBe('stream.ready');
  expect(Object.keys(messages[1].stages)).toEqual(['STAGE_1_SPARSE_PRESCAN', 'STAGE_2_ACTIVE_PREPASS', 'STAGE_3_MAIN_SWAP']);
});
