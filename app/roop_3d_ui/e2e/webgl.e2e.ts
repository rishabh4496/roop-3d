import { expect, test, type Page } from '@playwright/test';

/**
 * The unit tests run the shaders against a recording fake context; these specs run the REAL GLSL on a real WebGL2
 * context and read pixels back, checking each inspection mode against the CPU reference in webglModel.js.
 *
 * Fixture (harness/gl.tsx): image A = left half #c80000, right half #00c800; image B = A plus a #ffffff block over the
 * lower-right quarter. Canvas is 640×360 CSS px at devicePixelRatio 1.
 */
const frames = (page: Page, count = 3) => page.evaluate(n => new Promise<void>(resolve => { const step = (left: number) => (left ? requestAnimationFrame(() => step(left - 1)) : resolve()); step(n); }), count);
const set = async (page: Page, props: Record<string, unknown>) => { await page.evaluate(next => (window as any).__setProps(next), props); await frames(page); };
const pixel = (page: Page, x: number, y: number) => page.evaluate(([px, py]) => (window as any).__readPixel(px, py) as number[], [x, y]);
const near = (actual: number[], expected: number[], tolerance = 3) => { expected.forEach((value, i) => expect(Math.abs(actual[i] - value), `channel ${i}: ${actual} vs ${expected}`).toBeLessThanOrEqual(tolerance)); };

test.beforeEach(async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/harness/gl.html');
  await page.waitForFunction(() => (window as any).__ready === true);
  await frames(page);
  expect(await page.evaluate(() => (window as any).__glUnsupported), 'WebGL2 must be available and the shaders must compile').toBeUndefined();
  expect(await page.evaluate(() => !!(window as any).__glReady)).toBe(true);
  expect(errors).toEqual([]);
});

const MODEL = '/src/components/canvas/webglModel.js';   // served by the Vite dev server; imported in-page so the spec uses the exact CPU reference
const CHANGED = [544, 270], UNCHANGED = [100, 100];

test('compiles and links both programs on a real WebGL2 context', async ({ page }) => {
  const info = await page.evaluate(() => { const gl = document.querySelector<HTMLCanvasElement>('[data-testid="webgl-viewport"]')!.getContext('webgl2')!; return { error: gl.getError(), version: gl.getParameter(gl.VERSION) as string, maxTexture: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number }; });
  expect(info.error).toBe(0);                                                                // no GL error after the first frames
  expect(info.version).toContain('WebGL 2');
  expect(info.maxTexture).toBeGreaterThanOrEqual(4096);
});

test('split wipe shows A left of the divider and B right of it, and the divider moves', async ({ page }) => {
  await set(page, { mode: 'wipe', split: 0.5 });
  near(await pixel(page, UNCHANGED[0], UNCHANGED[1]), [200, 0, 0]);
  near(await pixel(page, CHANGED[0], CHANGED[1]), [255, 255, 255]);                           // right of the divider: swapped
  await set(page, { mode: 'wipe', split: 0.9 });
  near(await pixel(page, CHANGED[0], CHANGED[1]), [0, 200, 0]);                               // left of the divider: original
  await set(page, { mode: 'wipe', split: 0.5, vertical: true });
  near(await pixel(page, 544, 100), [0, 200, 0]);                                              // vertical: above the divider = original
  near(await pixel(page, 544, 300), [255, 255, 255]);                                          // below it = swapped
});

test('alpha crossfade / blend mixes the two frames linearly', async ({ page }) => {
  await set(page, { mode: 'blend', mix: 0.5 });
  near(await pixel(page, CHANGED[0], CHANGED[1]), [127.5, 227.5, 127.5]);
  await set(page, { mode: 'blend', mix: 0 });
  near(await pixel(page, CHANGED[0], CHANGED[1]), [0, 200, 0]);
  await set(page, { mode: 'blend', mix: 1 });
  near(await pixel(page, CHANGED[0], CHANGED[1]), [255, 255, 255]);
});

for (const colormap of ['inferno', 'viridis']) {
  test(`diff map applies abs(swapped - original) * gain through the ${colormap} colormap`, async ({ page }) => {
    for (const gain of [0.5, 1]) {
      await set(page, { mode: 'diff', gain, colormap });
      const expected = await page.evaluate(async ([name, g, url]) => {
        const model = await import(/* @vite-ignore */ url as string);
        return { changed: model.colormapColor(name as string, model.diffValue([0, 200, 0], [255, 255, 255], g as number)), unchanged: model.colormapColor(name as string, 0) };
      }, [colormap, gain, MODEL] as const);
      near(await pixel(page, CHANGED[0], CHANGED[1]), expected.changed, 4);
      near(await pixel(page, UNCHANGED[0], UNCHANGED[1]), expected.unchanged, 3);              // unchanged pixels sit at the cold end
    }
    // a higher gain must read hotter, never cooler
    await set(page, { mode: 'diff', gain: 0.25, colormap }); const cool = await pixel(page, CHANGED[0], CHANGED[1]);
    await set(page, { mode: 'diff', gain: 1, colormap }); const hot = await pixel(page, CHANGED[0], CHANGED[1]);
    expect(cool).not.toEqual(hot);
  });
}

test('3.5x loupe magnifies around the pointer; bicubic (Catmull-Rom) differs from bilinear next to a sharp edge', async ({ page }) => {
  const pointer = { x: 320, y: 180, active: true };
  await set(page, { mode: 'loupe', pointer, bicubic: false });
  const bilinear = await pixel(page, 330, 180), outside = await pixel(page, 600, 40);
  await set(page, { mode: 'loupe', pointer, bicubic: true });
  const bicubic = await pixel(page, 330, 180);
  near(bilinear, [0, 200, 0], 2);                                                                // 1.1 texels past the red→green edge: pure green when bilinear
  expect(bicubic[1], 'bicubic overshoots on the bright side of the edge').toBeGreaterThan(bilinear[1] + 3);
  near(outside, [0, 200, 0], 2);                                                                 // outside the lens the swapped frame is shown as-is
  await set(page, { mode: 'loupe', pointer: { ...pointer, active: false } });
  near(await pixel(page, 330, 180), [0, 200, 0], 2);
});

test('draws the yaw/pitch/roll gizmo over the nose bridge only when enabled', async ({ page }) => {
  const face = { id: 'f1', bbox: [0.35, 0.2, 0.65, 0.8], landmarks5: [[0.43, 0.4], [0.57, 0.4], [0.5, 0.52], [0.45, 0.65], [0.55, 0.65]], pose3d: { yaw: 30, pitch: 10, roll: -8 } };
  const blueCount = () => page.evaluate(() => {
    const canvas = document.querySelector<HTMLCanvasElement>('[data-testid="webgl-viewport"]')!, gl = canvas.getContext('webgl2')!;
    const data = new Uint8Array(canvas.width * canvas.height * 4); gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
    let blue = 0, red = 0, green = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 2] > 200 && data[i] < 100 && data[i + 1] < 160) blue++;
      if (data[i] > 220 && data[i + 1] < 90 && data[i + 2] < 90) red++;                          // #ef4444-ish, distinct from the fixture's #c80000
      if (data[i + 1] > 170 && data[i] < 60 && data[i + 2] > 80 && data[i + 2] < 120) green++;   // #22c55e-ish, distinct from the fixture's #00c800
    }
    return { blue, red, green };
  });
  await set(page, { mode: 'wipe', faces: [face], showGizmo: false });
  expect(await blueCount()).toEqual({ blue: 0, red: 0, green: 0 });
  await set(page, { mode: 'wipe', faces: [face], showGizmo: true });
  const drawn = await blueCount();
  expect(drawn.blue).toBeGreaterThan(20); expect(drawn.red).toBeGreaterThan(20); expect(drawn.green).toBeGreaterThan(20);
});

test('renders a native 3840×2160 frame into a 4K backing store', async ({ page }) => {
  await page.goto('/harness/gl.html?w=3840&h=2160');
  await page.waitForFunction(() => (window as any).__ready === true);
  await set(page, { mode: 'diff', width: 1920, height: 1080, pixelRatio: 2, gain: 0.5, colormap: 'inferno' });
  expect(await page.evaluate(() => (window as any).__native)).toEqual({ a: [3840, 2160], b: [3840, 2160] });
  expect(await page.evaluate(() => (window as any).__canvasSize())).toEqual([3840, 2160]);
  expect(await page.evaluate(() => (window as any).__glUnsupported)).toBeUndefined();
  const expected = await page.evaluate(async url => { const m = await import(/* @vite-ignore */ url); return { changed: m.colormapColor('inferno', m.diffValue([0, 200, 0], [255, 255, 255], 0.5)), unchanged: m.colormapColor('inferno', 0) }; }, MODEL);
  near(await pixel(page, 3000, 1500), expected.changed, 4);                                      // lower-right block, in 4K device px
  near(await pixel(page, 200, 200), expected.unchanged, 3);
});
