/**
 * GPU-free Express backend for UI development (`npm run dev:mock`).
 *
 * Every response carries `X-Mock-Server: true`, and `GET /api/meta` answers `{"mock": true}`, so a
 * page can never mistake this for the real service. All figures below are fixed placeholders;
 * nothing here measures hardware or runs a model.
 */
import express from 'express';
import type { Server } from 'node:http';
import { WebSocketServer } from 'ws';

export const MOCK_HEADER = 'X-Mock-Server';

const STAGES = ['STAGE_1_SPARSE_PRESCAN', 'STAGE_2_ACTIVE_PREPASS', 'STAGE_3_MAIN_SWAP'] as const;

export const mockHardware = () => ({
  timestamp: new Date().toISOString(), mock: true,
  gpu_name: 'Mock GPU (no hardware)', compute_capability: '8.6', vram_total_gb: 12, vram_available_gb: 10,
  cuda_version: '12.4', tensorrt_version: null, onnxruntime_providers: ['CPUExecutionProvider'],
  tensor_core_precisions: ['fp16', 'int8', 'bf16', 'tf32'], tensor_core_state: 'INFERRED_FROM_COMPUTE_CAPABILITY',
  nvdec_available: false, nvenc_available: false, gpus: [], advertised_providers: ['CPUExecutionProvider'],
});

export const mockFrame = (tick: number) => {
  const running = tick % 20 < 10;
  return {
    type: 'pipeline.status', schema_version: 1, sequence: tick, timestamp: new Date().toISOString(), job_id: running ? 'mock-job' : null,
    stage: running ? STAGES[2] : null,
    stages: { [STAGES[0]]: 'SKIPPED', [STAGES[1]]: 'SKIPPED', [STAGES[2]]: running ? 'RUNNING' : 'PENDING' },
    stage_progress_pct: running ? (tick % 10) * 10 : null, active_ranges_count: null, verified_target_frames: null,
    skipped_frames_pct: null, eta_seconds: running ? 10 - (tick % 10) : null, fps: running ? 12 : 0,
    engine_state: 'IDLE', engine_label: null, engine_model: null, engine_precision: null, engine_since: null,
  };
};

export function createMockApp() {
  const app = express();
  app.disable('x-powered-by');
  app.use((_request, response, next) => {
    response.setHeader(MOCK_HEADER, 'true');
    next();
  });
  // Same `/api/<x>` -> `/api/v1/<x>` aliasing as the real service.
  app.use((request, _response, next) => {
    if (/^\/api\/(?!v1\/)/.test(request.url)) request.url = request.url.replace(/^\/api\//, '/api/v1/');
    next();
  });
  const api = express.Router();
  api.get('/meta', (_request, response) => { response.json({ mock: true }); });
  api.get('/health', (_request, response) => { response.json({ status: 'ok', application: 'roop_3d_mock', mock: true }); });
  api.get('/system/hardware', (_request, response) => { response.json(mockHardware()); });
  api.get('/system/telemetry', (_request, response) => { response.json({ ...mockHardware(), pipeline: mockFrame(0) }); });
  for (const empty of ['jobs', 'projects', 'facesets', 'models']) api.get(`/${empty}`, (_request, response) => { response.json([]); });
  api.use((_request, response) => { response.status(404).json({ error: { code: 'MOCK_NOT_IMPLEMENTED', message: 'This endpoint is not mocked' } }); });
  app.use('/api/v1', api);
  return app;
}

export function startMockServer(port = 8031, host = '127.0.0.1'): Promise<{ server: Server; url: string; close: () => Promise<void> }> {
  const app = createMockApp();
  return new Promise((resolve, reject) => {
    const server = app.listen(port, host, () => {
      const address = server.address();
      const actual = typeof address === 'object' && address ? address.port : port;
      const sockets = new WebSocketServer({ noServer: true });
      let tick = 0;
      const timer = setInterval(() => {
        tick += 1;
        for (const client of sockets.clients) client.send(JSON.stringify(mockFrame(tick)));
      }, 1000);
      server.on('upgrade', (request, socket, head) => {
        const path = (request.url ?? '').split('?')[0];
        if (path !== '/ws/telemetry' && path !== '/ws/v1/pipeline/status') { socket.destroy(); return; }
        sockets.handleUpgrade(request, socket, head, client => {
          client.send(JSON.stringify({ type: 'stream.ready', schema_version: 1, sequence: tick, action: 'refresh_jobs' }));
          client.send(JSON.stringify(mockFrame(tick)));
        });
      });
      resolve({
        server, url: `http://${host}:${actual}`,
        close: () => new Promise(done => { clearInterval(timer); for (const client of sockets.clients) client.terminate(); server.close(() => done()); }),
      });
    });
    server.on('error', reject);
  });
}
