import fs from 'node:fs';
import path from 'node:path';
import { expect, test, type Page, type WebSocketRoute } from '@playwright/test';
import { boot, SCREENS as MOCK_SCREENS } from './helpers/mockApi';

/**
 * Stage 27 certification, in a real browser:
 *   1. every screen in the router mounts, is reachable by its hash, and shows exactly one at a time;
 *   2. nothing the UI does leaves this machine (every HTTP request and WebSocket targets 127.0.0.1 / localhost, or is data:/blob:);
 *   3. binary JPEG frames pushed over a WebSocket (16-byte header + JPEG) are painted onto the live canvas.
 *
 * The route list is read from src/router/AppRouter.jsx itself, so a screen added there is covered (or this fails) without editing the
 * spec. The result is written to roop_3d/.runtime/certification/ui_coverage.json for app/tests/test_stage27_certification.py.
 */
const routerSource = fs.readFileSync(path.resolve('src/router/AppRouter.jsx'), 'utf8');
const block = routerSource.slice(routerSource.indexOf('export const SCREENS'), routerSource.indexOf('];', routerSource.indexOf('export const SCREENS')));
const ROUTES = [...block.matchAll(/\['([a-z]+)',/g)].map(match => match[1]);
const REPORT = path.resolve('../roop_3d/.runtime/certification/ui_coverage.json');
const LOCAL = new Set(['127.0.0.1', 'localhost']);
const report: Record<string, unknown> = { routes_total: ROUTES.length, routes: ROUTES };

function flush() {
  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.writeFileSync(REPORT, JSON.stringify({ ...report, generated_at: new Date().toISOString() }, null, 2));
}

/**
 * Endpoint-security web filters (Kaspersky on the development machine) inject a script into every HTTP page in transit and that script
 * phones home. Those requests are not the application's: the source and the built bundle reference no such host (see
 * test_standalone_install.py). They are counted and reported, never silently dropped, and every OTHER off-machine request fails the test.
 */
const INJECTED_BY_SECURITY_SOFTWARE = /(^|.)kaspersky-labs.com$/;
const offMachine = (value: string) => {
  const url = new URL(value);
  return !['data:', 'blob:', 'about:'].includes(url.protocol) && !LOCAL.has(url.hostname);
};
const injected = (value: string) => INJECTED_BY_SECURITY_SOFTWARE.test(new URL(value).hostname);

test('the route table is what the browser actually serves', () => {
  expect(ROUTES.length).toBeGreaterThanOrEqual(9);
  expect([...MOCK_SCREENS].sort()).toEqual([...ROUTES].sort());      // the shared e2e helper must not drift from the router
});

test('every screen mounts and is reachable, one at a time, with no page error and no off-machine request', async ({ page }) => {
  const leaked: string[] = [], filtered: string[] = [];
  const note = (value: string) => { if (!offMachine(value)) return; (injected(value) ? filtered : leaked).push(value); };
  page.on('request', request => note(request.url()));
  page.on('websocket', socket => note(socket.url().replace(/^ws/, 'http')));
  const { errors } = await boot(page, { route: '/#/home' });

  const mounted = await page.$$eval('[data-screen]', nodes => nodes.map(node => node.getAttribute('data-screen')));
  expect([...mounted].sort()).toEqual([...ROUTES].sort());            // all screens exist in the DOM (hidden screens stay mounted)

  const visited: string[] = [];
  for (const id of ROUTES) {
    await page.evaluate(hash => { location.hash = hash; }, `#/${id}`);
    await expect(page.locator(`[data-screen="${id}"]`)).toBeVisible();
    const visible = await page.$$eval('[data-screen]', nodes => nodes.filter(node => !(node as HTMLElement).hidden).map(node => node.getAttribute('data-screen')));
    expect(visible, `only #/${id} may be visible`).toEqual([id]);
    visited.push(id);
  }
  await page.waitForTimeout(300);                                       // let late fetches show up in the request log
  expect(errors).toEqual([]);
  expect(leaked, 'requests that left this machine').toEqual([]);
  Object.assign(report, { routes_visited: visited, routes_covered: visited.length, external_requests: leaked, injected_by_security_software: filtered.length, page_errors: errors });
  flush();
});

function header(frame: number): Buffer {
  const buffer = Buffer.alloc(16);
  buffer.writeUInt32LE(frame, 0); buffer.writeUInt32LE(frame * 40, 4);
  buffer.writeUInt16LE(0x11, 14);
  return buffer;
}
async function jpeg(page: Page, color: string): Promise<Buffer> {
  const base64 = await page.evaluate(async fill => {
    const canvas = document.createElement('canvas'); canvas.width = 64; canvas.height = 48;
    const context = canvas.getContext('2d')!; context.fillStyle = fill; context.fillRect(0, 0, 64, 48);
    const blob = await new Promise<Blob>(resolve => canvas.toBlob(value => resolve(value!), 'image/jpeg', 0.95));
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let text = ''; for (const byte of bytes) text += String.fromCharCode(byte);
    return btoa(text);
  }, color);
  return Buffer.from(base64, 'base64');
}
const centre = (page: Page) => page.evaluate(() => {
  const source = document.querySelector<HTMLCanvasElement>('.live-stream-canvas')!;
  const copy = document.createElement('canvas'); copy.width = source.width; copy.height = source.height;
  const context = copy.getContext('2d')!; context.drawImage(source, 0, 0);
  return Array.from(context.getImageData(source.width >> 1, source.height >> 1, 1, 1).data.slice(0, 3));
});
const near = (actual: number[], expected: number[]) => actual.length === 3 && actual.every((value, index) => Math.abs(value - expected[index]) <= 4);

test('binary WebSocket frames are painted on the live canvas, each colour in turn', async ({ page }) => {
  let socket!: WebSocketRoute;
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.routeWebSocket('**/ws/v1/preview/stream', ws => { socket = ws; });
  await page.goto('/harness/livestream.html');
  await expect.poll(() => !!socket).toBe(true);
  const painted: string[] = [];
  for (const [name, color, rgb] of [['red', '#ff0000', [255, 0, 0]], ['green', '#00ff00', [0, 255, 0]], ['blue', '#0000ff', [0, 0, 255]]] as const) {
    const frame = await jpeg(page, color);
    socket.send(Buffer.concat([header(painted.length + 1), frame]));
    await expect.poll(async () => near(await centre(page), [...rgb])).toBe(true);
    painted.push(name);
  }
  expect(errors).toEqual([]);
  Object.assign(report, { binary_frames_painted: painted.length, binary_colours: painted });
  flush();
});
