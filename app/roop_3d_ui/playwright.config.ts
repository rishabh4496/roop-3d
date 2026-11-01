import { defineConfig } from '@playwright/test';

/**
 * Real-browser gates (`npm run test:e2e`). They drive the system Chrome (no browser download) against the Vite dev
 * server: the WebGL2 shader specs need a real GL context, and the telemetry spec needs real React + real WebSockets
 * (mocked per-test with page.routeWebSocket, so no backend, GPU or model is required).
 */
export default defineConfig({
  testDir: 'e2e',
  testMatch: '**/*.e2e.ts',
  timeout: 90_000,
  workers: 1,
  fullyParallel: false,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:5174',
    channel: 'chrome',
    headless: true,
    // Headless Chrome falls back to SwiftShader when there is no usable GPU; allow it so WebGL2 is always available.
    launchOptions: { args: ['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--use-angle=swiftshader-webgl'] },
  },
  webServer: { command: 'npx vite --host 127.0.0.1 --port 5174 --strictPort', url: 'http://127.0.0.1:5174', reuseExistingServer: true, timeout: 90_000 },
});
