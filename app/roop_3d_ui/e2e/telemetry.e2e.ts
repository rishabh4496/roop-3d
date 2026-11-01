import { expect, test, type Page, type WebSocketRoute } from '@playwright/test';

/**
 * Stage 11 verification gate, in a real browser: a real 30 Hz `/ws/telemetry` stream (frames are pushed from this Node
 * process through page.routeWebSocket, so no backend is needed) must cause ZERO re-renders of the parameter sidebar,
 * timeline, Face Manager, VideoViewport or App; the floating PiP preview and the hotkey remapper must survive
 * route changes and a full page reload.
 *
 * `window.__R3D_RENDER_PROBE__` is the render-count sink read by src/utils/renderProbe.ts.
 */
const CAPABILITIES = { models: [], job_parameter_bounds: {}, legal_parameter_bounds: {}, rendering_worker_available: false, workstation: { features: {}, read_only_parameters: {}, disabled_parameters: {}, controls: {}, job_controls: {} } };
const GUARDED = ['App', 'VideoViewport', 'TimelineWorkstation', 'SettingControl', 'SettingsInspector', 'FaceManager'] as const;

async function prepare(page: Page) {
  const sockets: { events: WebSocketRoute[]; telemetry: WebSocketRoute[] } = { events: [], telemetry: [] };
  await page.addInitScript(() => { (window as any).__R3D_RENDER_PROBE__ = {}; sessionStorage.setItem('roop3d_skip_linter', 'true'); });
  await page.route('**/api/v1/**', route => {
    const path = new URL(route.request().url()).pathname.replace('/api/v1', '');
    const body = path === '/system/capabilities' ? CAPABILITIES : path === '/config' || path === '/health' ? {} : [];
    return route.fulfill({ json: body });
  });
  await page.routeWebSocket('**/api/v1/ws/events', ws => { sockets.events.push(ws); });
  await page.routeWebSocket('**/ws/telemetry', ws => { sockets.telemetry.push(ws); });
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  return { sockets, errors };
}
const probe = (page: Page) => page.evaluate(() => ({ ...(window as any).__R3D_RENDER_PROBE__ }) as Record<string, number>);
/** Wait until the app stops rendering on its own, so the baseline is a true idle. */
async function settle(page: Page) {
  let previous = '';
  for (let round = 0; round < 40; round++) {
    await page.waitForTimeout(150);
    const now = JSON.stringify(await probe(page));
    if (now === previous) return;
    previous = now;
  }
  throw new Error('app never settled');
}
const send = (ws: WebSocketRoute, value: unknown) => ws.send(JSON.stringify(value));

/** Stream `seconds` of telemetry at 30 Hz on every open socket; resolves with the page's rAF intervals while it ran. */
async function burst(page: Page, sockets: { events: WebSocketRoute[]; telemetry: WebSocketRoute[] }, seconds = 3) {
  await page.evaluate(() => {
    const times: number[] = []; (window as any).__frames = times;
    const step = (t: number) => { times.push(t); (window as any).__raf = requestAnimationFrame(step); };
    (window as any).__raf = requestAnimationFrame(step);
  });
  const stages = { STAGE_1_SPARSE_PRESCAN: 'DONE', STAGE_2_ACTIVE_PREPASS: 'DONE', STAGE_3_MAIN_SWAP: 'RUNNING' };
  const total = Math.round(seconds * 30);
  await new Promise<void>(resolve => {
    let tick = 0;
    const timer = setInterval(() => {
      const hardware = { gpus: [{ name: 'RTX 4090', total_vram_mib: 24576, free_vram_mib: 20000 - tick * 20 }], logical_cpu_count: 16 };
      for (const ws of sockets.events) send(ws, { type: 'system.telemetry', hardware });
      for (const ws of sockets.telemetry) {
        send(ws, { type: 'pipeline.status', stage: 'STAGE_3_MAIN_SWAP', stages, stage_progress_pct: Math.min(99, tick), fps: 24 + (tick % 10) / 10, eta_seconds: total - tick, vram_available_gb: 12 - tick * 0.01, skipped_frames_pct: 1.5 });
        send(ws, { type: 'system.telemetry', hardware });
        send(ws, { type: 'job.progress', stage: 'warp', fps: 20 + (tick % 5), stage_timings_ms: { decode: 3, swap: 11 } });
      }
      if (++tick >= total) { clearInterval(timer); resolve(); }
    }, 1000 / 30);
  });
  await page.waitForTimeout(150);                                                              // let the last frame commit
  return page.evaluate(() => {
    cancelAnimationFrame((window as any).__raf);
    const times = (window as any).__frames as number[], intervals = times.slice(1).map((t, i) => t - times[i]).sort((a, b) => a - b);
    return { frames: times.length, median: intervals[Math.floor(intervals.length / 2)] ?? 0, p95: intervals[Math.floor(intervals.length * 0.95)] ?? 0, max: intervals.at(-1) ?? 0 };
  });
}
const deltas = (before: Record<string, number>, after: Record<string, number>) => Object.fromEntries(GUARDED.map(name => [name, (after[name] ?? 0) - (before[name] ?? 0)]));

test('a 30 Hz /ws/telemetry stream re-renders nothing in the parameter sidebar, timeline, Face Manager or App', async ({ page }, info) => {
  const { sockets, errors } = await prepare(page);
  for (const route of ['#/faceswap', '#/facemgr', '#/settings']) {
    await page.goto(`/${route}`);
    await page.waitForSelector('.workstation');
    await expect.poll(() => sockets.events.length + sockets.telemetry.length, { timeout: 10_000 }).toBeGreaterThanOrEqual(3);   // events + run pill + HUD
    await settle(page);
    const before = await probe(page);
    expect(before.SettingControl, 'probes are live in this build').toBeGreaterThan(0);
    const pacing = await burst(page, sockets, 3);
    const after = await probe(page);
    expect(deltas(before, after), `render deltas on ${route}`).toEqual({ App: 0, VideoViewport: 0, TimelineWorkstation: 0, SettingControl: 0, SettingsInspector: 0, FaceManager: 0 });
    console.log(`[pacing ${route}] ${pacing.frames} rAF frames, median ${pacing.median.toFixed(1)} ms, p95 ${pacing.p95.toFixed(1)} ms, max ${pacing.max.toFixed(1)} ms`);
    info.annotations.push({ type: `pacing ${route}`, description: `${pacing.frames} rAF frames, median ${pacing.median.toFixed(1)} ms, p95 ${pacing.p95.toFixed(1)} ms, max ${pacing.max.toFixed(1)} ms` });
    expect(pacing.median, 'main thread keeps up with display refresh during the burst').toBeLessThan(40);
    expect(pacing.max).toBeLessThan(500);
  }
  // The subscribers that SHOULD update did: the run bar and the HUD pill show the last streamed frame.
  await page.goto('/#/faceswap');
  await expect(page.locator('.run-pill')).toBeVisible();
  await settle(page);
  const stream = await burst(page, sockets, 1);
  expect(stream.frames).toBeGreaterThan(10);
  await expect(page.getByTestId('fps')).toHaveText(/^\d+\.\d$/);
  await expect(page.locator('.pipeline-pill-btn')).toContainText('FPS');
  expect(errors).toEqual([]);
});

test('the floating PiP preview stays visible across routes and a reload; remapped hotkeys persist', async ({ page }) => {
  const { errors } = await prepare(page);
  await page.goto('/#/faceswap');
  await page.waitForSelector('.workstation');
  await settle(page);
  const floating = page.getByTestId('floating-window');
  await expect(floating).toHaveCount(0);
  await page.keyboard.press('Shift+F');
  await expect(floating).toBeVisible();
  for (const route of ['#/facemgr', '#/batch', '#/settings', '#/history', '#/gallery', '#/faceswap']) {
    await page.evaluate(hash => { location.hash = hash; }, route);
    await expect(page.locator(`[data-screen="${route.slice(2)}"]`)).toBeVisible();
    await expect(floating).toBeVisible();                                                    // a real visibility check: not inside a hidden / inert screen
    const box = await floating.boundingBox();
    expect(box && box.width >= 220 && box.height >= 150).toBe(true);
  }
  // Move it with the keyboard, then reload: position and open state persist.
  await floating.locator('.floating-window-bar').focus();
  const before = (await floating.boundingBox())!;
  await page.keyboard.press('ArrowLeft'); await page.keyboard.press('ArrowLeft');
  await expect.poll(async () => (await floating.boundingBox())!.x).toBe(before.x - 32);
  await page.reload();
  await expect(page.getByTestId('floating-window')).toBeVisible();
  expect((await page.getByTestId('floating-window').boundingBox())!.x).toBe(before.x - 32);

  // Hotkey remap: change "Toggle split wipe" to X, navigate around, reload.
  await page.evaluate(() => { location.hash = '#/settings'; });
  const field = page.getByLabel('Shortcut for Toggle split wipe');
  await expect(field).toHaveValue('W');
  await field.press('x');
  await expect(field).toHaveValue('X');
  await field.press('d');                                                                  // already used by "Toggle diff map"
  await expect(page.getByRole('alert').filter({ hasText: 'already assigned' })).toBeVisible();
  await expect(field).toHaveValue('X');
  for (const route of ['#/facemgr', '#/faceswap', '#/settings']) await page.evaluate(hash => { location.hash = hash; }, route);
  await expect(field).toHaveValue('X');
  await page.reload();
  await expect(page.getByLabel('Shortcut for Toggle split wipe')).toHaveValue('X');

  // The remapped chord drives the real canvas on the workstation.
  await page.evaluate(() => { location.hash = '#/faceswap'; });
  await expect(page.locator('.compare-host')).toHaveCount(0);
  await page.keyboard.press('w');
  await expect(page.locator('.compare-host')).toHaveCount(0);                              // the old chord is gone
  await page.keyboard.press('x');
  await expect(page.locator('.compare-host')).toHaveCount(1);
  await page.keyboard.press('x');
  await expect(page.locator('.compare-host')).toHaveCount(0);
  expect(errors).toEqual([]);
});
