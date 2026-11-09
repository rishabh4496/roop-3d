import { expect, test, type Page, type Route } from '@playwright/test';
import { boot, FRAME_FACES, mockFrame } from './helpers/mockApi';

/**
 * Target Face Bank: persistent FACE n pills (canvas + bank), click-to-toggle lock, live angle-capture HUD with the 5 x 3 grid,
 * hover inspector, Re-Scan, source binding, and the engine status bar with its switch-failure toast. Real Chrome and React, REST mocked
 * (the real tracker, harvester and endpoints are covered by pytest: tests/test_persistent_target_registry.py, test_target_ui_endpoints.py).
 */
const face = (rank: number, persistent: number, x: number, track: string | null) => ({
  ...FRAME_FACES[0], face_index: rank, persistent_face_id: persistent, track_id: track, cluster_id: persistent + 1, is_target: false,
  bbox: [x, 120, x + 140, 330], bbox_norm: [x / 960, 120 / 540, (x + 140) / 960, 330 / 540],
  kps_5: [[x + 35, 175], [x + 105, 175], [x + 70, 215], [x + 40, 270], [x + 100, 270]],
  pose: { yaw: -10 + rank * 12, pitch: 2, roll: 0 }, pose3d: { yaw: -10 + rank * 12, pitch: -2, roll: 0 },
});
const FACES = [face(0, 0, 90, 'track_0000'), face(1, 1, 420, 'track_0001'), face(2, 2, 740, null)];

const CELLS = [2, 1, 3, 0, 4].flatMap(yaw => [1, 2, 0].map(pitch => [yaw, pitch] as const));      // frontal first, profiles last
const lockPayload = (faceId: number, captured: number, state: 'running' | 'done') => ({
  lock_id: `${faceId}`.repeat(32), target_number: faceId + 1, label: `face ${faceId}`, face_id: faceId, media_path: 'roop_3d/.runtime/media/t.mp4', cluster_id: faceId + 1,
  anchor_frame_index: 0, bank_size: captured, observed_frames: [0], reference_bank: [], track_id: FACES[faceId].track_id,
  coverage: { captured, quality_locked: captured, total: 15, buckets: CELLS.slice(0, captured).map(([yaw, pitch], index) => ({
    bucket: [yaw, pitch], quality: 240, quality_locked: true, has_crop: true, yaw: (yaw - 2) * 30, pitch: (pitch - 1) * 30, roll: 0, frame_index: index * 6,
    source: index === 0 ? 'anchor' : 'auto', match_score: index === 0 ? 1 : 0.55 + (index % 5) * 0.08 })) },
  scan: state === 'running' ? { state, requested: 100, sampled: captured * 6, matched: captured } : { state, requested: 100, sampled: 100, matched: 15, exhaustive: true },
});

async function mockBank(page: Page) {
  const seen = { locks: [] as unknown[], unlocks: [] as string[], rescans: [] as string[] };
  const progress = new Map<number, { captured: number; done: boolean }>();
  await page.route('**/api/v1/media/frame**', async (route: Route) => {
    const url = new URL(route.request().url()), response = mockFrame(url), meta = JSON.parse(response.headers['X-Frame-Meta']);
    if (meta.faces) meta.faces = FACES;
    await route.fulfill({ ...response, headers: { 'X-Frame-Meta': JSON.stringify(meta) } });
  });
  await page.route('**/api/v1/target/**', async (route: Route) => {
    const url = new URL(route.request().url()), path = url.pathname.replace('/api/v1', ''), method = route.request().method();
    if (/\/crop\.jpg$/.test(path)) {
      const [yaw, pitch] = path.split('/').slice(4, 6).map(Number);
      return route.fulfill({ status: 200, contentType: 'image/svg+xml', body: `<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="hsl(${yaw * 50 + pitch * 20},45%,38%)"/><circle cx="32" cy="28" r="14" fill="#f2c9a5"/><text x="32" y="58" font-size="11" text-anchor="middle" fill="#fff">${yaw},${pitch}</text></svg>` });
    }
    if (path === '/target/lock-from-frame') {
      const body = route.request().postDataJSON(); seen.locks.push(body);
      progress.set(body.face_index, { captured: 1, done: false });
      return route.fulfill({ status: 201, json: { ...lockPayload(body.face_index, 1, 'running'), selected_face_index: body.face_index, frame_faces: [] } });
    }
    const id = Number(path.split('/')[2]?.[0]);
    if (method === 'DELETE') { seen.unlocks.push(path.split('/')[2]); progress.delete(id); return route.fulfill({ status: 200, json: { lock_id: path.split('/')[2], face_id: id, unlocked: true } }); }
    if (/\/rescan$/.test(path)) { seen.rescans.push(path.split('/')[2]); progress.set(id, { captured: 1, done: false }); return route.fulfill({ status: 200, json: lockPayload(id, 1, 'running') }); }
    if (/\/match$/.test(path)) return route.fulfill({ status: 200, json: { lock_id: path.split('/')[2], frame_index: 0, target_face_index: null, faces: [] } });
    const state = progress.get(id) ?? { captured: 15, done: true };
    if (!state.done) { state.captured = Math.min(15, state.captured + 3); state.done = state.captured >= 15; }
    return route.fulfill({ status: 200, json: lockPayload(id, state.captured, state.done ? 'done' : 'running') });
  });
  return seen;
}

const engines = (patch: Record<string, unknown> = {}) => ({
  pipeline: { engine_state: 'IDLE', engine_label: null, engine_model: null, engine_precision: null },
  selected: { swapper_model: 'inswapper_128', detector_model: 'scrfd', enhancer_model: 'gpen' },
  active: {
    detector: { stage: 'detector', model_id: 'scrfd', model_path: 'app/models/scrfd_2.5g_kps.onnx', filename: 'scrfd_2.5g_kps.onnx', provider: 'TensorrtExecutionProvider', precision: 'fp16', sha256: 'a'.repeat(64), loaded_at: 1, switches: 0 },
    swapper: { stage: 'swapper', model_id: 'inswapper_128', model_path: 'app/models/inswapper_128.onnx', filename: 'inswapper_128.onnx', provider: 'TensorrtExecutionProvider', precision: 'fp16', sha256: 'b'.repeat(64), loaded_at: 1, switches: 0 },
    enhancer: { stage: 'enhancer', model_id: 'gpen_bfr_512', model_path: 'app/models/gpen_bfr_512.onnx', filename: 'gpen_bfr_512.onnx', provider: 'TensorrtExecutionProvider', precision: 'fp16', sha256: 'c'.repeat(64), loaded_at: 1, switches: 0 },
  },
  precision: { label: 'TensorRT FP16 Mixed', stage: 'swapper', stages: { swapper: 'TensorRT FP16 Mixed' }, policy: 'fp16' }, failures: [], ...patch,
});

test('persistent FACE pills, click-to-lock, live angle capture, hover inspector, re-scan, source binding and unlock', async ({ page }) => {
  const { errors } = await boot(page, { openProject: true });
  const seen = await mockBank(page);
  await page.route('**/api/v1/pipeline/status', route => route.fulfill({ status: 200, json: engines() }));
  await page.reload();
  await page.getByRole('button', { name: 'Projects' }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Verify & open' }).click();

  const bank = page.getByTestId('frame-face-strip');
  const items = bank.locator('.face-bank-item');
  await expect(items).toHaveCount(3);
  await expect(bank.locator('.eyebrow')).toHaveText('TARGET FACE BANK');
  await expect(items.locator('.face-id-pill').first()).toHaveCSS('text-transform', 'uppercase');
  const colours = await items.evaluateAll(nodes => nodes.map(node => (node as HTMLElement).style.getPropertyValue('--face-color')));
  expect(new Set(colours).size).toBe(3);

  // The engine bar: live models from /pipeline/status.
  const engineBar = page.getByTestId('engine-status-bar');
  await expect(engineBar).toContainText('Detector:scrfd');
  await expect(engineBar).toContainText('Swapper:inswapper_128');
  await expect(engineBar).toContainText('Enhancer:gpen_bfr_512• Verified');
  await expect(engineBar).toContainText('Precision:TensorRT FP16 Mixed');

  // Click face 0's card: lock it; the HUD appears inside the card and fills in live.
  await items.nth(0).locator('.frame-face-card').click();
  await expect(items.nth(0)).toHaveClass(/locked/);
  const hud = items.nth(0).getByTestId('angle-hud');
  await expect(hud).toHaveAttribute('data-state', 'capturing');
  await expect(hud).toContainText('Capturing Target Angles…');
  await expect(items.nth(1).getByTestId('angle-hud')).toHaveCount(0);
  await expect.poll(() => hud.locator('[data-captured=true]').count(), { timeout: 15_000 }).toBeGreaterThan(3);
  await page.screenshot({ path: 'test-results/facebank-capturing.png' });
  await expect(hud).toHaveAttribute('data-state', 'complete', { timeout: 20_000 });
  await expect(hud).toContainText('Angles Locked (100% Coverage)');
  await expect(hud.locator('[data-captured=true]')).toHaveCount(15);
  await expect(hud.locator('img.angle-crop').first()).toBeVisible();

  // Hover face 0: every captured crop is listed.
  await items.nth(0).hover();
  const inspector = page.getByTestId('angle-inspector');
  await expect(inspector.locator('figure')).toHaveCount(15);
  await expect(inspector).toContainText('Every crop was assigned to face 0');
  await page.screenshot({ path: 'test-results/facebank-inspector.png' });
  await page.mouse.move(5, 5);
  await expect(inspector).toHaveCount(0);

  // Bind a source identity to face 0.
  await items.nth(0).getByLabel('Source identity for face 0').selectOption('bob');
  await expect(items.nth(0).getByLabel('Source identity for face 0')).toHaveValue('bob');

  // Re-scan.
  await items.nth(0).getByRole('button', { name: 'Re-Scan All Angles' }).click();
  await expect.poll(() => seen.rescans.length).toBe(1);
  await expect(hud).toHaveAttribute('data-state', 'capturing');
  await expect(hud).toHaveAttribute('data-state', 'complete', { timeout: 20_000 });

  // Clicking the locked card again unlocks it; the binding stays visible.
  await items.nth(0).locator('.frame-face-card').click();
  await expect.poll(() => seen.unlocks.length).toBe(1);
  await expect(items.nth(0)).not.toHaveClass(/locked/);
  await expect(items.nth(0).getByTestId('angle-hud')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('clicking a face box on the canvas toggles its lock, and the canvas shows the persistent pills', async ({ page }) => {
  const { errors } = await boot(page, { openProject: true });
  const seen = await mockBank(page);
  await page.route('**/api/v1/pipeline/status', route => route.fulfill({ status: 200, json: engines() }));
  await page.reload();
  await page.getByRole('button', { name: 'Projects' }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Verify & open' }).click();
  const bank = page.getByTestId('frame-face-strip');
  await expect(bank.locator('.face-bank-item')).toHaveCount(3);
  const canvas = page.locator('canvas[aria-label="Detected face bounding boxes"]').first();
  await expect(canvas).toBeVisible();
  const box = (await canvas.boundingBox())!;
  const scale = Math.min(box.width / 960, box.height / 540), offsetX = (box.width - 960 * scale) / 2, offsetY = (box.height - 540 * scale) / 2;
  const at = (x: number, y: number) => ({ x: box.x + offsetX + x * scale, y: box.y + offsetY + y * scale });

  await page.screenshot({ path: 'test-results/facebank-canvas-unlocked.png' });
  const third = at(740 + 70, 225);                         // face 2, a free candidate (no track yet)
  await page.route('**/api/v1/media/detect-frame-faces', route => route.fulfill({ status: 200, json: { inspection_id: 'a'.repeat(32), faces: [{ id: 'track_0002_0', track_id: 'track_0002', frame_index: 0, timestamp: 0,
    bbox: FACES[2].bbox_norm, landmarks5: FACES[2].kps_5.map(([x, y]) => [x / 960, y / 540]), confidence: 0.9, pose: FACES[2].pose, pose3d: FACES[2].pose3d, thumbnail_base64: '' }],
    inspection: { inspection_id: 'a'.repeat(32), target_sha256: 'b'.repeat(64), tracks: [], truncated: false, tracking_method: 'x', warnings: [], scene_cut_frames: [] } } }));
  await page.mouse.click(third.x, third.y);
  await expect.poll(() => seen.locks.length).toBe(1);
  expect((seen.locks[0] as { face_index: number }).face_index).toBe(2);
  await expect(bank.locator('.face-bank-item').nth(2)).toHaveClass(/locked/);
  await expect(bank.locator('.face-bank-item').nth(2).getByTestId('angle-hud')).toBeVisible();
  await page.screenshot({ path: 'test-results/facebank-canvas-locked.png' });

  await page.mouse.click(third.x, third.y);                // again: unlock
  await expect.poll(() => seen.unlocks.length).toBe(1);
  await expect(bank.locator('.face-bank-item').nth(2)).not.toHaveClass(/locked/);
  expect(errors).toEqual([]);
});

test('a model that fails to switch raises an error toast with the exact backend reason', async ({ page }) => {
  const { errors } = await boot(page, { openProject: true });
  let payload = engines();
  await page.route('**/api/v1/pipeline/status', route => route.fulfill({ status: 200, json: payload }));
  await page.reload();
  await page.getByRole('button', { name: 'Projects' }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Verify & open' }).click();
  await expect(page.getByTestId('engine-swapper')).toContainText('inswapper_128');
  payload = engines({ failures: [{ id: 1, stage: 'swapper', model_id: 'hyperswap_1c_256', code: 'MODEL_NOT_INSTALLED', reason: 'Selected swapper has no installed local ONNX weights', at: 1 }] });
  const alert = page.getByRole('alert').filter({ hasText: 'did not switch' });
  await expect(alert).toBeVisible({ timeout: 10_000 });
  await expect(alert).toContainText('Selected swapper has no installed local ONNX weights [MODEL_NOT_INSTALLED]');
  await expect(alert).toContainText('Still using inswapper_128');
  await expect(page.getByTestId('engine-swapper')).toContainText('hyperswap_1c_256 failed');
  await page.screenshot({ path: 'test-results/facebank-switch-failure.png' });
  expect(errors).toEqual([]);
});
