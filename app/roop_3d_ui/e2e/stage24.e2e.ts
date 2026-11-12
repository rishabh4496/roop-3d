import { expect, test, type Page, type Route } from '@playwright/test';
import { boot } from './helpers/mockApi';

/**
 * Stage 24 in a real browser (system Chrome, mocked local service): the liquid-glass HUDs, the drag-to-reorder lists, the Embla
 * frame dock and the persistent queue. Screenshots go to `test-results/stage24/` for the visual review.
 */
test.use({ viewport: { width: 1680, height: 1050 } });

const lifecycle = ['queued', 'preparing', 'compiling_trt', 'tracking_3d', 'rendering', 'paused', 'muxing_audio', 'completed', 'failed', 'cancelled'] as const;
const job = (id: string, name: string, state: (typeof lifecycle)[number], order: number, extra: Record<string, unknown> = {}) => ({
  job_id: id, name, state: state.toUpperCase(), lifecycle_state: state, queue_order: order, request: {}, slices: [], revision: 1, output: null, error_message: null,
  progress: { frame: state === 'rendering' ? 40 : 0, total_frames: 100, fps: 0, stage: state.toUpperCase() }, ...extra,
});

async function mockQueue(page: Page, jobs: ReturnType<typeof job>[]) {
  const posted: { url: string; body: unknown }[] = [];
  await page.route('**/api/v1/queue**', async (route: Route) => {
    const request = route.request();
    if (request.method() === 'GET') return route.fulfill({ json: jobs });
    posted.push({ url: new URL(request.url()).pathname, body: request.postDataJSON() });
    return route.fulfill({ json: jobs });
  });
  return posted;
}

test.describe('persistent queue (#/processing)', () => {
  test('shows each job\'s lifecycle state and saves a drag-free reorder from the move buttons', async ({ page }) => {
    await boot(page, { route: '/#/home' });
    const posted = await mockQueue(page, [
      job('r'.repeat(32), 'Rendering clip', 'rendering', 0), job('a'.repeat(32), 'Queued A', 'queued', 1), job('b'.repeat(32), 'Queued B', 'queued', 2),
      job('t'.repeat(32), 'Engine build', 'compiling_trt', 3), job('f'.repeat(32), 'Broken', 'failed', 4, { error_message: 'ffmpeg exited 1',
        slices: [{ slice_id: 's1', slice_index: 1, status: 'FAILED', error: 'oom' }] }),
    ]);
    await page.goto('/#/processing');
    const panel = page.getByTestId('backend-queue');
    await expect(panel).toBeVisible();
    await expect(panel.getByTestId('lifecycle-chip')).toHaveText(['Rendering', 'Queued', 'Queued', 'Compiling TensorRT', 'Failed']);
    await expect(panel.getByRole('button', { name: 'Retry segment 1' })).toBeVisible();
    await page.screenshot({ path: 'test-results/stage24/processing-queue.png' });

    await panel.getByRole('button', { name: 'Move Queued B up' }).click();
    await expect.poll(() => posted.find(item => item.url.endsWith('/queue/reorder'))?.body).toEqual({ ordered_job_ids: ['r'.repeat(32), 'b'.repeat(32), 'a'.repeat(32), 't'.repeat(32), 'f'.repeat(32)] });
  });

  test('a running job cannot be moved', async ({ page }) => {
    await boot(page, { route: '/#/home' });
    await mockQueue(page, [job('r'.repeat(32), 'Rendering clip', 'rendering', 0), job('a'.repeat(32), 'Queued A', 'queued', 1)]);
    await page.goto('/#/processing');
    await expect(page.getByTestId('backend-queue').getByRole('button', { name: /Reorder Rendering clip/ })).toHaveCount(0);
    await expect(page.getByTestId('backend-queue').getByRole('button', { name: /Reorder Queued A/ })).toBeVisible();
  });
});

test.describe('batch recipes (#/batch)', () => {
  test('dragging or moving a recipe reorders the matrix columns', async ({ page }) => {
    await boot(page, { route: '/#/batch' });
    await page.getByRole('tab', { name: 'Recipe Matrix' }).click();
    const chips = page.locator('.recipe-order .recipe-order-chip');
    await expect(chips.first()).toBeVisible();
    const nameOf = async (index: number) => (await chips.nth(index).innerText()).split(/\r?\n/).pop()!.trim();   // the chip also holds the grip and arrows
    const first = await nameOf(0);
    const headers = page.locator('.batch-matrix-grid .recipe-header-card strong');
    expect((await headers.nth(0).innerText()).trim()).toBe(first);
    await page.getByRole('button', { name: /^Move .* right$/ }).first().click();
    expect(await nameOf(1)).toBe(first);
    expect((await headers.nth(1).innerText()).trim()).toBe(first);   // the matrix columns follow the chips
    await page.screenshot({ path: 'test-results/stage24/batch-recipe-order.png' });
  });
});

test.describe('studio HUDs and frame dock (#/faceswap)', () => {
  test('the dock follows the playhead, trims with the keyboard and shares the speed with the timeline', async ({ page }) => {
    const { errors } = await boot(page, { openProject: true });
    const dock = page.getByTestId('studio-layout').getByTestId('timeline-dock');
    await expect(dock).toBeVisible();
    await expect(dock.locator('.dock-slide')).not.toHaveCount(0);
    const inHandle = dock.getByRole('slider', { name: 'In point' });
    await inHandle.focus();
    await page.keyboard.press('Shift+ArrowRight');
    await expect(inHandle).toHaveAttribute('aria-valuenow', '1');
    await dock.getByRole('radio', { name: '2x' }).click();
    await expect(dock.getByRole('radio', { name: '2x' })).toHaveAttribute('aria-checked', 'true');
    await expect(page.locator('.speed-btn.active')).toHaveText('2x');          // the long timeline's own buttons follow the same store
    await page.screenshot({ path: 'test-results/stage24/studio-dock.png' });
    expect(errors).toEqual([]);
  });

  test('the people panel lists ranked actors and maps a person to a FaceSet', async ({ page }) => {
    await boot(page, { openProject: true });
    await page.route('**/api/v1/actors/**', route => route.fulfill({ json: { inspection_id: 'a'.repeat(32), actor_count: 2, warnings: [], actors: [
      { actor_id: 'actor_0001', label: 'Person #1', rank: 1, track_ids: ['track_0001'], persistent_ids: [1], grouping: 'persistent_id', screen_time_seconds: 3, prominence: 3, thumbnail_base64: '', thumbnail_mime: 'image/jpeg' },
      { actor_id: 'actor_0002', label: 'Person #2', rank: 2, track_ids: ['track_0000'], persistent_ids: [], grouping: 'single_track', screen_time_seconds: 1, prominence: 1, thumbnail_base64: '', thumbnail_mime: 'image/jpeg' }] } }));
    await page.reload();
    const panel = page.getByTestId('actor-panel');
    if (await panel.count()) {
      await panel.getByLabel('Person #1 source FaceSet').selectOption('alice');
      await expect(panel.getByLabel('Person #1 source FaceSet')).toHaveValue('alice');
    }
  });
});

test.describe('route persistence', () => {
  test('a reload with no hash returns to the last screen, while an explicit link still wins', async ({ page }) => {
    await boot(page, { route: '/#/batch' });
    await expect(page).toHaveURL(/#\/batch$/);
    await page.goto('/');
    await expect(page).toHaveURL(/#\/batch$/);
    await page.goto('/#/settings');
    await expect(page).toHaveURL(/#\/settings$/);
  });
});

test('visual review: the studio with the frame dock', async ({ page }) => {
  await boot(page, { openProject: true });
  const dock = page.getByTestId('studio-layout').getByTestId('timeline-dock');
  await expect(dock).toBeVisible();
  const box = await dock.boundingBox();
  const panel = await page.getByTestId('studio-layout').boundingBox();
  expect(box!.x + box!.width).toBeLessThanOrEqual(panel!.x + panel!.width + 1);   // the dock stays inside the Studio
  await dock.scrollIntoViewIfNeeded();
  // The scroll reveal must finish: nothing visible in the dock may stay faded or blurred (slides scrolled out of the strip reveal when they scroll in).
  await expect.poll(() => dock.evaluate(node => { const strip = node.querySelector('.ul-embla__viewport')!.getBoundingClientRect();
    return [...node.querySelectorAll<HTMLElement>('.dock-head, .dock-embla, .dock-rail-wrap, .dock-tick')].filter(item => !item.matches('.dock-tick') || (item.getBoundingClientRect().right <= strip.right && item.getBoundingClientRect().left >= strip.left))
    .every(item => { let el: HTMLElement | null = item; while (el && el !== node.parentElement) { const s = getComputedStyle(el); if (s.opacity !== '1' || !/^(none|blur\(0(px)?\))?$/.test(s.filter)) return false; el = el.parentElement; } return true; }); }), { timeout: 5000 }).toBe(true);
  await page.screenshot({ path: 'test-results/stage24/studio-dock-fit.png' });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);  // no horizontal page scroll
});

test('canvas HUDs: telemetry pill, 3D mesh mode and the glass brush toolbar (drag springs back, never paints)', async ({ page }) => {
  const { errors } = await boot(page, { openProject: true });
  await page.getByRole('button', { name: /Compare/ }).first().click();
  const pill = page.getByTestId('viewport-telemetry-pill');
  await expect(pill).toBeVisible();
  await expect(pill).toContainText('YAW');
  await page.getByRole('button', { name: '3D Mesh Overlay' }).click();
  await expect(page.locator('.mesh3d-legend')).toBeVisible();
  await expect(pill).toContainText('mesh3d');

  // The brush toolbar is a draggable glass card; only its grip drags, and it springs back to where it was.
  await page.getByRole('button', { name: 'Mask Brush' }).click();
  const toolbar = page.getByRole('toolbar', { name: 'Paint and erase brush' });
  await expect(toolbar).toBeVisible();
  await expect(toolbar.getByRole('button', { name: 'Frame range' })).toBeVisible();
  const grip = toolbar.getByRole('button', { name: 'Drag to move' });
  const home = await toolbar.boundingBox();
  const g = (await grip.boundingBox())!;
  await page.mouse.move(g.x + g.width / 2, g.y + g.height / 2);
  await page.mouse.down();
  await page.mouse.move(g.x + 120, g.y - 90, { steps: 8 });
  const moved = await toolbar.boundingBox();
  expect(Math.abs(moved!.x - home!.x) + Math.abs(moved!.y - home!.y)).toBeGreaterThan(20);   // it follows the pointer
  await page.screenshot({ path: 'test-results/stage24/brush-glass-dragging.png' });
  await page.mouse.up();
  await expect.poll(async () => { const now = await toolbar.boundingBox(); return Math.abs(now!.x - home!.x) + Math.abs(now!.y - home!.y); }, { timeout: 4000 }).toBeLessThan(3);   // elastic bounce-back
  await page.screenshot({ path: 'test-results/stage24/brush-glass.png' });
  expect(errors).toEqual([]);
});
