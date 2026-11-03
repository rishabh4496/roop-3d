import { defineConfig } from '@playwright/test';
import { localBrowserTemp } from './browserPaths';

localBrowserTemp();

// Requires a running production serve.py on loopback and manual_stage08_acceptance.py fixtures.
// This campaign uses real endpoints and model inference. It is intentionally separate from mocked browser tests.
export default defineConfig({
  testDir: 'acceptance', testMatch: '**/*.real.ts', timeout: 600_000, workers: 1,
  outputDir: '../roop_3d/.runtime/stage08_acceptance/browser/playwright', reporter: 'list',
  use: { baseURL: 'http://127.0.0.1:8030', channel: 'chrome', headless: true,
    viewport: { width: 1680, height: 1050 }, acceptDownloads: true,
    launchOptions: { args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader-webgl'] } },
});
