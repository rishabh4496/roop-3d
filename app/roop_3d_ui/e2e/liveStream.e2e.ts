import { expect, test, type Page, type WebSocketRoute } from '@playwright/test';

/**
 * Defect #3 browser gate: binary JPEG frames pushed over a WebSocket (16-byte header + JPEG) are decoded by the real
 * `createImageBitmap`, drawn on the real canvas through `bitmaprenderer`, with ZERO React renders per frame, and the
 * newest frame always wins when frames arrive faster than the page repaints.
 *
 * The server side is mocked with page.routeWebSocket, so no backend, GPU or model is needed.
 */
function header(frame: number, stage: [number, number, number], flags = 0x11): Buffer {
  const buffer = Buffer.alloc(16);
  buffer.writeUInt32LE(frame, 0); buffer.writeUInt32LE(frame * 40, 4);
  stage.forEach((value, index) => buffer.writeUInt16LE(Math.round(value * 10), 8 + index * 2));
  buffer.writeUInt16LE(flags, 14);
  return buffer;
}

/** A real solid-colour JPEG, produced by the browser's own encoder. */
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

/** The colour of the middle pixel of the live canvas (a bitmaprenderer canvas is read by drawing it elsewhere). */
const centre = (page: Page) => page.evaluate(() => {
  const source = document.querySelector<HTMLCanvasElement>('.live-stream-canvas')!;
  const copy = document.createElement('canvas'); copy.width = source.width; copy.height = source.height;
  const context = copy.getContext('2d')!; context.drawImage(source, 0, 0);
  return Array.from(context.getImageData(source.width >> 1, source.height >> 1, 1, 1).data.slice(0, 3));
});
const isNearColor = (actual: number[] | null, expected: number[]) =>
  actual !== null && actual.length === expected.length && actual.every((value, index) => Math.abs(value - expected[index]) <= 4);

test('binary frames reach the canvas with no React re-renders, and the newest frame wins', async ({ page }) => {
  let socket!: WebSocketRoute;
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.routeWebSocket('**/ws/v1/preview/stream', ws => { socket = ws; });
  await page.goto('/harness/livestream.html');
  await expect.poll(() => !!socket).toBe(true);
  const red = await jpeg(page, '#ff0000'), blue = await jpeg(page, '#0000ff');
  const rendersAtMount = await page.evaluate(() => (window as any).__liveRenders as number);

  // 1. one frame: the canvas appears, shows the colour, and the readout carries the stage latencies from the header
  socket.send(Buffer.concat([header(1, [12.3, 80.5, 40]), red]));
  await expect(page.getByTestId('live-stream')).toHaveAttribute('data-live', 'true');
  await expect.poll(async () => isNearColor(await centre(page), [255, 0, 0])).toBe(true);
  await expect(page.locator('.live-stream-readout')).toContainText('64×48 · detect 12 · swap 81 · blend 40 ms');

  // 2. a burst of 60 alternating frames, the last one blue: whatever was skipped, the final picture is the newest
  for (let frame = 2; frame < 62; frame += 1) socket.send(Buffer.concat([header(frame, [1, 2, 3]), frame % 2 ? red : blue]));
  socket.send(Buffer.concat([header(100, [1, 2, 3]), blue]));
  await expect.poll(async () => isNearColor(await centre(page), [0, 0, 255])).toBe(true);

  // 3. the whole stream cost React nothing
  expect(await page.evaluate(() => (window as any).__liveRenders as number)).toBe(rendersAtMount);

  // 4. a corrupt frame and a wrong-version frame are ignored; the picture stays
  socket.send(Buffer.concat([header(101, [0, 0, 0]), Buffer.from('not a jpeg')]));
  socket.send(Buffer.concat([header(102, [0, 0, 0], 0x12), red]));
  await page.waitForTimeout(300);
  expect(isNearColor(await centre(page), [0, 0, 255])).toBe(true);

  // 5. the stream going quiet hides the canvas again
  await expect(page.getByTestId('live-stream')).toHaveAttribute('data-live', 'false', { timeout: 6000 });
  expect(errors).toEqual([]);
});

test('an unmounted page leaves no open socket behind', async ({ page }) => {
  let closed = false;
  await page.routeWebSocket('**/ws/v1/preview/stream', ws => { ws.onClose(() => { closed = true; }); });
  await page.goto('/harness/livestream.html');
  await page.waitForTimeout(300);
  await page.goto('about:blank');
  await expect.poll(() => closed).toBe(true);
});
