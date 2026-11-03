import { expect, test, type Page, type Route } from '@playwright/test';

const ID = 'a'.repeat(32), TARGET = 'roop_3d/.runtime/media/t.mp4';
const FEATURES = Object.fromEntries(['projects', 'face_inspection', 'queue', 'frame_preview', 'video_playback', 'uploads', 'trim'].map(name => [name, { enabled: true, reason: null }]));
const CAPABILITIES = {
  models: [], job_parameter_bounds: {}, legal_parameter_bounds: {}, inspection_parameter_bounds: {}, enabled_modes: [], rendering_worker_available: true,
  workstation: { features: FEATURES, controls: {}, read_only_parameters: {}, disabled_parameters: {}, job_states: [], job_controls: {}, upload_chunk_bytes: 1048576, max_asset_bytes: 1048576 }
};
const gif = 'R0lGODlhAQABAAAAACw=';
const tinyJpeg = Buffer.from('/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA=', 'base64');
const track = (n: number, x1: number) => ({
  track_id: `track_000${n}`, persistent_id: n, scene_id: 0, landmark_count: 106, first_frame: 0, last_frame: 99, first_timestamp_seconds: 0, last_timestamp_seconds: 3.96,
  bounding_box: { x1, y1: 120, x2: x1 + 160, y2: 360 }, pose_range: { yaw_degrees: [-5, 5], pitch_degrees: [0, 1], roll_degrees: [0, 1], method: 'five_point_geometric_estimate' },
  thumbnail_base64: gif, thumbnail_mime: 'image/gif', sampled_detections: 10
});
const INSPECTION = {
  inspection_id: ID, target_sha256: 'b'.repeat(64), tracks: [track(0, 140), track(1, 620)], truncated: false, tracking_method: 'identity_lock_v1', warnings: [], scene_cut_frames: [60]
};
const sheetTrack = (n: number) => ({
  track_id: `track_000${n}`, persistent_id: n, first_frame: 0, last_frame: 99, segments: [[0, 99, 0, 3.96]], missing_ranges: [],
  visible_frames: 100, swappable_frames: 100, withheld_frames: 0, withheld_reasons: {}, state_frames: { TRACKED: 100 }, confidence_mean: 0.93, confidence_min: 0.71, scene_ids: [0], thumbnails: []
});
const SHEET = {
  inspection_id: ID, target_sha256: 'b'.repeat(64), truncated: false, scene_cut_frames: [60], warnings: [], tracks: [sheetTrack(0), sheetTrack(1)]
};

test.use({ viewport: { width: 1680, height: 1050 } });

async function setupRoutes(page: Page, seenPreviews: any[], previewResponseStatus = 200, previewResponseBody: any = null) {
  await page.addInitScript(() => {
    sessionStorage.setItem('roop3d_skip_linter', 'true');
    localStorage.setItem('roop3d_tour_seen', 'true');
  });

  await page.route('**/api/v1/**', async (route: Route) => {
    const url = new URL(route.request().url()), path = url.pathname.replace('/api/v1', ''), method = route.request().method();
    const json = (body: unknown, code = 200) => route.fulfill({ status: code, json: body });
    if (path === '/system/capabilities') return json(CAPABILITIES);
    if (path === '/config') return json({});
    if (path === '/facesets') return json([{ faceset_id: 'alice', name: 'Alice', item_count: 3 }, { faceset_id: 'bob', name: 'Bob', item_count: 3 }]);
    if (path === '/projects') return json([]);
    if (path === '/models/hub') return json({ models: [], model_states: [], execution_policy: '', download_policy: '', engines: [
      { engine_id: 'reference_onnx', name: 'Roop 3D Reference ONNX', status: 'READY_FOR_INFERENCE', selectable: true, actual_inference_available: true, evidence: 'hash checked' }
    ] });
    if (path.startsWith('/faces/bank')) return json({ version: 1, revision: 1, target_sha256: 'b'.repeat(64), next_rank: 0, clusters: [], aliases: {}, unresolved_tracks: [], inspection_id: ID });
    if (path === '/assets/uploads' && method === 'POST') return json({ upload_id: 'u1', chunk_bytes: 1048576 });
    if (path.startsWith('/assets/uploads/u1') && method === 'PUT') return json({ ok: true });
    if (path === '/assets/uploads/u1/complete' && method === 'POST') return json({ kind: 'video', path: TARGET });
    if (path === '/media/probe') return json({ width: 960, height: 540, duration_seconds: 4, fps_rational: '25/1', declared_frame_count: 100, audio_tracks: [] });
    if (path === '/media/keyframes') return json({ timestamps: [0, 2] });
    if (path === '/media/preview-frame') return route.fulfill({ status: 200, contentType: 'image/jpeg', body: tinyJpeg });
    if (path === '/media/content') return route.fulfill({ status: 204 });
    if (path === `/media/inspections/${ID}` || path === '/media/inspect-faces') return json(INSPECTION);
    if (path === `/identity/inspections/${ID}`) return json(SHEET);
    if (path === '/reconstructions') return json([]);
    return json([]);
  });

  await page.route('**/api/preview', route => {
    const body = route.request().postDataJSON();
    seenPreviews.push(body);
    if (previewResponseStatus === 200) {
      return route.fulfill({
        status: 200,
        contentType: 'image/jpeg',
        body: tinyJpeg,
        headers: {
          'X-Preview-Meta': JSON.stringify({
            media_id: ID,
            frame_index: body.frame_index ?? 0,
            swap_config_hash: String(seenPreviews.length),
            solo_node: body.solo_node ?? 'final',
            width: 960,
            height: 540,
            faces: [],
            applied: [],
            unsupported: [],
            notes: [],
            latency: { total_ms: 15 }
          })
        }
      });
    } else {
      return route.fulfill({
        status: previewResponseStatus,
        json: previewResponseBody ?? { error: { message: `Simulated error ${previewResponseStatus}` } },
      });
    }
  });

  await page.routeWebSocket('**/ws/**', () => undefined);
}

test('audit: fresh target inspect and dropdown assignment drives live swap preview', async ({ page }) => {
  const seenPreviews: any[] = [];
  await setupRoutes(page, seenPreviews);
  await page.goto('/#/faceswap');

  // Upload video via file input
  const fileInput = page.locator('input[type="file"][accept*=".mp4"]');
  await fileInput.setInputFiles({
    name: 'test.mp4',
    mimeType: 'video/mp4',
    buffer: Buffer.from('fake mp4 video content'),
  });

  const inspectBtn = page.getByRole('button', { name: 'Inspect target faces' });
  await expect(inspectBtn).toBeVisible({ timeout: 5000 });
  await inspectBtn.click();

  await expect(page.locator('[data-track-id="track_0000"]')).toBeVisible();

  // Assign track_0000 to "Alice" via the select dropdown
  const select = page.getByLabel('Identity for track_0000');
  await select.selectOption('alice');

  await expect.poll(() => seenPreviews.length, { timeout: 5000 }).toBeGreaterThan(0);
  expect(seenPreviews.at(-1)?.face_mappings).toEqual(expect.arrayContaining([{ track_id: 'track_0000', faceset_id: 'alice' }]));
  await expect(page.locator('img.live-swap-preview')).toBeVisible({ timeout: 5000 });
});

test('audit: classic layout assigns source and triggers live preview', async ({ page }) => {
  const seenPreviews: any[] = [];
  await setupRoutes(page, seenPreviews);
  await page.goto('/#/faceswap');

  // Switch to classic layout
  await page.locator('.layout-toggle').click();

  // Upload video via file input
  const fileInput = page.locator('input[type="file"][accept*=".mp4"]');
  await fileInput.setInputFiles({
    name: 'test.mp4',
    mimeType: 'video/mp4',
    buffer: Buffer.from('fake mp4 video content'),
  });

  // In classic layout, click Inspect target faces
  const inspectBtn = page.getByRole('button', { name: 'Inspect target faces' });
  await expect(inspectBtn).toBeVisible({ timeout: 5000 });
  await inspectBtn.click();

  // In classic layout, each track has: <label className="mapping-choice"><span>Source identity</span><select aria-label="Identity for track_0000" ...>
  const select = page.getByLabel('Identity for track_0000');
  await expect(select).toBeVisible();
  await select.selectOption('alice');

  await expect.poll(() => seenPreviews.length, { timeout: 5000 }).toBeGreaterThan(0);
  expect(seenPreviews.at(-1)?.face_mappings).toEqual(expect.arrayContaining([{ track_id: 'track_0000', faceset_id: 'alice' }]));
  await expect(page.locator('img.live-swap-preview')).toBeVisible({ timeout: 5000 });
});

test('audit: scrubbing playhead requests updated frame_index for live preview', async ({ page }) => {
  const seenPreviews: any[] = [];
  await setupRoutes(page, seenPreviews);
  await page.goto('/#/faceswap');

  const fileInput = page.locator('input[type="file"][accept*=".mp4"]');
  await fileInput.setInputFiles({
    name: 'test.mp4',
    mimeType: 'video/mp4',
    buffer: Buffer.from('fake mp4 video content'),
  });

  const inspectBtn = page.getByRole('button', { name: 'Inspect target faces' });
  await inspectBtn.click();
  const select = page.getByLabel('Identity for track_0000');
  await select.selectOption('alice');

  await expect.poll(() => seenPreviews.length, { timeout: 5000 }).toBeGreaterThan(0);
  expect(seenPreviews.at(-1)?.frame_index).toBe(0);

  // Press right arrow shortcut or step forward to scrub playhead
  await page.keyboard.press('ArrowRight');
  await expect.poll(() => seenPreviews.at(-1)?.frame_index, { timeout: 5000 }).toBeGreaterThan(0);
});

test('audit: solo buffer selection requests intermediate pipeline buffer', async ({ page }) => {
  const seenPreviews: any[] = [];
  await setupRoutes(page, seenPreviews);
  await page.goto('/#/faceswap');

  const fileInput = page.locator('input[type="file"][accept*=".mp4"]');
  await fileInput.setInputFiles({
    name: 'test.mp4',
    mimeType: 'video/mp4',
    buffer: Buffer.from('fake mp4 video content'),
  });

  const inspectBtn = page.getByRole('button', { name: 'Inspect target faces' });
  await inspectBtn.click();
  const select = page.getByLabel('Identity for track_0000');
  await select.selectOption('alice');

  await expect.poll(() => seenPreviews.length, { timeout: 5000 }).toBeGreaterThan(0);

  // Click "Solo Raw Swap" button in the pipeline flow bar
  const swapperBtn = page.getByRole('button', { name: 'Solo Raw Swap' });
  await swapperBtn.click();

  // Next preview request should request solo_node: "swapper"
  await expect.poll(() => seenPreviews.at(-1)?.solo_node, { timeout: 5000 }).toBe('swapper');
});

test('audit: what happens when /api/preview returns 500 error', async ({ page }) => {
  const seenPreviews: any[] = [];
  await setupRoutes(page, seenPreviews, 500, { error: { message: 'Inference model crashed on GPU' } });
  await page.goto('/#/faceswap');

  const fileInput = page.locator('input[type="file"][accept*=".mp4"]');
  await fileInput.setInputFiles({
    name: 'test.mp4',
    mimeType: 'video/mp4',
    buffer: Buffer.from('fake mp4 video content'),
  });

  const inspectBtn = page.getByRole('button', { name: 'Inspect target faces' });
  await expect(inspectBtn).toBeVisible({ timeout: 5000 });
  await inspectBtn.click();

  const select = page.getByLabel('Identity for track_0000');
  await select.selectOption('alice');

  await expect.poll(() => seenPreviews.length, { timeout: 5000 }).toBeGreaterThan(0);

  // When /api/preview fails with 500:
  // LiveSwapStatus shows error
  const status = page.locator('.live-swap-status.error');
  await expect(status).toBeVisible({ timeout: 5000 });
  await expect(status).toContainText('Inference model crashed on GPU');

  // And viewport shows the user-facing retry UI
  const emptyPlaceholder = page.locator('.video-pane.processed .viewport-empty');
  await expect(emptyPlaceholder).toBeVisible();
  await expect(emptyPlaceholder).toContainText('Swap preview failed');
  await expect(emptyPlaceholder).toContainText('Retry preview');
});
