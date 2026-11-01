/** `npm run dev:mock`: the mock backend plus Vite HMR on 5174, proxied to it. No GPU, models or Python needed. */
import { createServer } from 'vite';
import { startMockServer } from './server.ts';

const mock = await startMockServer(Number(process.env.ROOP3D_MOCK_PORT ?? 8031));
const vite = await createServer({
  server: {
    host: '127.0.0.1', port: 5174, strictPort: true,
    proxy: { '/api': { target: mock.url, changeOrigin: true }, '/ws': { target: mock.url, ws: true, changeOrigin: true } },
  },
});
await vite.listen();
vite.printUrls();
console.log(`[mock] backend ${mock.url} (X-Mock-Server: true)`);

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => { void Promise.all([vite.close(), mock.close()]).finally(() => process.exit(0)); });
}
