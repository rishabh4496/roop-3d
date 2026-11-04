import { expect, test, type CDPSession, type Page, type WebSocketRoute } from '@playwright/test';

/**
 * Stage 18 browser soak, in real Chrome: no leaked WebGL contexts, object URLs, WebSockets, DOM nodes, event listeners or
 * JS heap across repeated mounts and route changes. Numbers come from the in-page resource ledger
 * (`window.__R3D_RESOURCES__`, src/utils/resourceLedger.ts) and Chrome DevTools `Performance.getMetrics` after a forced GC.
 */
const CAPABILITIES = { models: [], job_parameter_bounds: {}, legal_parameter_bounds: {}, rendering_worker_available: false, workstation: { features: {}, read_only_parameters: {}, disabled_parameters: {}, controls: {}, job_controls: {} } };

interface Metrics { heapMB: number; nodes: number; listeners: number; documents: number }
async function metrics(cdp: CDPSession): Promise<Metrics> {
  await cdp.send('HeapProfiler.collectGarbage');
  await cdp.send('HeapProfiler.collectGarbage');
  const { metrics: list } = await cdp.send('Performance.getMetrics');
  const get = (name: string) => list.find(item => item.name === name)?.value ?? 0;
  return { heapMB: get('JSHeapUsedSize') / 1048576, nodes: get('Nodes'), listeners: get('JSEventListeners'), documents: get('Documents') };
}
const ledger = (page: Page) => page.evaluate(() => (window as any).__R3D_RESOURCES__.snapshot());

test('WebGL comparison canvas: 60 mount/unmount cycles leave zero live contexts (and the ledger can see a real leak)', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/harness/leak.html');
  await page.waitForFunction(() => (window as any).__ready === true);
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Performance.enable');

  await page.evaluate(() => (window as any).__cycle(5));                      // warm-up: shader compilation, caches
  const warm = await metrics(cdp);
  await page.evaluate(() => (window as any).__cycle(60));
  const after = await metrics(cdp);
  const snapshot = await ledger(page);

  expect(snapshot.webgl.created, 'every cycle created a real context').toBe(65);
  expect(snapshot.webgl.live, 'every context was released on unmount').toBe(0);
  expect(snapshot.webgl.detached).toBe(0);
  expect(after.nodes - warm.nodes, 'DOM nodes').toBeLessThanOrEqual(40);
  expect(after.listeners - warm.listeners, 'event listeners').toBeLessThanOrEqual(40);
  expect(after.documents - warm.documents).toBeLessThanOrEqual(0);
  expect(after.heapMB - warm.heapMB, 'JS heap after forced GC (MB)').toBeLessThan(6);

  // Validity: a context that is removed from the page but never released must be reported, then clear again.
  await page.evaluate(() => (window as any).__leakOne());
  expect((await ledger(page)).webgl).toMatchObject({ live: 1, detached: 1 });
  await page.evaluate(() => (window as any).__releaseLeak());
  expect((await ledger(page)).webgl).toMatchObject({ live: 0, detached: 0 });
  expect(errors).toEqual([]);
});

test('whole application: 15 laps of every screen keep sockets, object URLs, contexts, nodes, listeners and heap bounded', async ({ page }, info) => {
  const sockets: WebSocketRoute[] = [];
  const errors: string[] = [];
  await page.addInitScript(() => { sessionStorage.setItem('roop3d_skip_linter', 'true'); });
  await page.route('**/api/v1/**', route => {
    const path = new URL(route.request().url()).pathname.replace('/api/v1', '');
    return route.fulfill({ json: path === '/system/capabilities' ? CAPABILITIES : path === '/config' || path === '/health' ? {} : [] });
  });
  await page.routeWebSocket('**/ws/**', ws => { sockets.push(ws); });
  await page.routeWebSocket('**/api/v1/ws/**', ws => { sockets.push(ws); });
  page.on('pageerror', error => errors.push(error.message));

  await page.goto('/?r3d-diag=1#/faceswap');
  await page.waitForSelector('.workstation');
  expect(await page.evaluate(() => !!(window as any).__R3D_RESOURCES__), 'ledger installs from ?r3d-diag=1').toBe(true);
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Performance.enable');
  await page.keyboard.press('Shift+F');                                           // open the floating preview window as well

  const routes = ['#/facemgr', '#/batch', '#/settings', '#/history', '#/gallery', '#/faceswap'];
  const lap = async () => {
    for (const route of routes) {
      await page.evaluate(hash => { location.hash = hash; }, route);
      await expect(page.locator(`[data-screen="${route.slice(2)}"]`)).toBeVisible();
    }
  };
  for (let i = 0; i < 3; i++) await lap();                                       // warm-up laps: lazy chunks, caches
  const baseline = { metrics: await metrics(cdp), ledger: await ledger(page) };
  const samples: Metrics[] = [];
  for (let i = 0; i < 15; i++) { await lap(); if (i % 5 === 4) samples.push(await metrics(cdp)); }
  const final = { metrics: await metrics(cdp), ledger: await ledger(page) };

  info.annotations.push({ type: 'browser soak', description: `heap ${baseline.metrics.heapMB.toFixed(1)} -> ${final.metrics.heapMB.toFixed(1)} MB, nodes ${baseline.metrics.nodes} -> ${final.metrics.nodes}, listeners ${baseline.metrics.listeners} -> ${final.metrics.listeners}, sockets live ${final.ledger.webSockets.live}, object URLs live ${final.ledger.objectUrls.live}, webgl live ${final.ledger.webgl.live}` });
  console.log(`[browser soak] ${JSON.stringify({ baseline: baseline.metrics, samples, final: final.metrics, ledger: final.ledger })}`);
  expect(final.ledger.webSockets.live, 'sockets do not accumulate').toBeLessThanOrEqual(baseline.ledger.webSockets.live);
  expect(final.ledger.webSockets.live).toBeLessThanOrEqual(4);
  expect(final.ledger.objectUrls.live, 'object URLs do not accumulate').toBeLessThanOrEqual(Math.max(2, baseline.ledger.objectUrls.live));
  expect(final.ledger.webgl.detached).toBe(0);
  expect(final.ledger.webgl.live).toBeLessThanOrEqual(Math.max(2, baseline.ledger.webgl.live));
  expect(final.metrics.documents - baseline.metrics.documents).toBeLessThanOrEqual(0);
  expect(final.metrics.nodes - baseline.metrics.nodes, 'DOM nodes').toBeLessThanOrEqual(300);
  expect(final.metrics.listeners - baseline.metrics.listeners, 'event listeners').toBeLessThanOrEqual(150);
  expect(final.metrics.heapMB - baseline.metrics.heapMB, 'JS heap after forced GC (MB)').toBeLessThan(10);
  expect(errors).toEqual([]);
});
