import { expect, test, type Page } from '@playwright/test';
import { boot, FRAME_FACES, mockFrame } from './helpers/mockApi';

/** The bottom dock shows one tool at a time; this opens the given tab before the page loads (it is remembered in localStorage). */
const dockTab = (page: Page, id: string) => page.addInitScript(tab => { try { localStorage.setItem('roop3d-dock-tab', tab); } catch { /* optional */ } }, id);

/**
 * Stage 5: timeline scrubbing, the filmstrip, and on-canvas face selection, in real Chrome with the REST API mocked.
 * (The real decoder, detector and endpoints are exercised by verify_stage5_timeline_and_faces.py and the pytest suite.)
 */

/** Faces away from the mocked project's two tracks: neither is a target yet. */
const NEW_FACES = [
  { ...FRAME_FACES[0], face_index: 0, bbox: [380, 120, 520, 300], bbox_norm: [380 / 960, 120 / 540, 520 / 960, 300 / 540], kps_5: [[415, 175], [485, 175], [450, 215], [420, 262], [480, 262]] },
  { ...FRAME_FACES[1], face_index: 1, bbox: [800, 120, 930, 300], bbox_norm: [800 / 960, 120 / 540, 930 / 960, 300 / 540], kps_5: [[830, 175], [900, 175], [865, 215], [840, 262], [890, 262]] },
];
async function newFacesInFrame(page: import('@playwright/test').Page) {
  await page.route('**/api/v1/media/frame**', async route => {
    const url = new URL(route.request().url());
    const response = mockFrame(url);
    const meta = JSON.parse(response.headers['X-Frame-Meta']);
    if (meta.faces) meta.faces = NEW_FACES;
    await route.fulfill({ ...response, headers: { 'X-Frame-Meta': JSON.stringify(meta) } });
  });
}
const lockResponse = (route: import('@playwright/test').Route) => route.fulfill({ status: 200, json: { inspection_id: 'a'.repeat(32),
  faces: NEW_FACES.map((face, rank) => ({ id: `track_000${rank + 2}_0`, track_id: `track_000${rank + 2}`, frame_index: 0, timestamp: 0, bbox: face.bbox_norm,
    landmarks5: face.kps_5.map(([x, y]) => [x / 960, y / 540]), confidence: face.confidence, pose: face.pose, pose3d: face.pose3d, thumbnail_base64: '' })),
  inspection: { inspection_id: 'a'.repeat(32), target_sha256: 'b'.repeat(64), tracks: [], truncated: false, tracking_method: 'iou_arcface_hungarian', warnings: [], scene_cut_frames: [] } } });

test('loading a target fetches info and the filmstrip, and the filmstrip shows real thumbnails', async ({ page }) => {
  const { seen, errors } = await boot(page, { openProject: true });
  await expect.poll(() => seen.filmstrips.length).toBeGreaterThan(0);
  expect(seen.filmstrips[0]).toMatch(/count=16/);
  await expect(page.locator('.filmstrip-frame-card img.filmstrip-thumb-img')).toHaveCount(16);
  await expect(page.locator('.filmstrip-thumb-skeleton')).toHaveCount(0);
  await expect(page.locator('.timecode-ruler-track .tick-label').first()).toBeVisible();
  expect(errors).toEqual([]);
});

test('the playhead frame is requested without faces while moving and with faces once it rests; the strip lists them', async ({ page }) => {
  await dockTab(page, 'faces');
  const { seen, errors } = await boot(page, { openProject: true });
  await expect(page.locator('img.frame-preview').first()).toBeVisible();
  const strip = page.getByTestId('frame-face-strip');
  await expect(strip.locator('.frame-face-card')).toHaveCount(2);
  expect(seen.frames.some(query => /detect_faces=true/.test(query))).toBe(true);
  expect(seen.frames[0]).not.toMatch(/detect_faces/);                  // the first frame is shown before any detection
  await expect(strip.locator('.frame-face-card').first()).toContainText('face 0');
  await expect(strip.locator('.frame-face-card').first()).toContainText('Y -12° P -3°');
  await expect(page.locator('canvas[aria-label="Detected face bounding boxes"]').first()).toBeVisible();
  expect(errors).toEqual([]);
});

test('dragging the playhead keeps one frame request in flight, skips stale frames and lands on the last one', async ({ page }) => {
  await dockTab(page, 'faces');
  const { errors } = await boot(page, { openProject: true });
  await expect(page.getByTestId('frame-face-strip').locator('.frame-face-card')).toHaveCount(2);
  await page.getByTestId('dock-tab-timeline').click();            // the drag happens on the timeline; the strip is checked again afterwards
  let inflight = 0, peak = 0;
  const wanted: number[] = [];
  await page.route('**/api/v1/media/frame**', async route => {
    const url = new URL(route.request().url());
    inflight++; peak = Math.max(peak, inflight); wanted.push(Number(url.searchParams.get('frame_index')));
    await new Promise(done => setTimeout(done, 40));                  // a slow decode
    inflight--;
    await route.fulfill(mockFrame(url));
  });
  const track = page.locator('.timeline-tracks-inner').first();
  await track.scrollIntoViewIfNeeded();
  const box = (await track.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.1, box.y + 20);
  await page.mouse.down();
  for (let step = 1; step <= 40; step++) await page.mouse.move(box.x + box.width * (0.1 + 0.8 * step / 40), box.y + 20);
  await page.mouse.up();
  await expect.poll(() => wanted.length, { timeout: 8000 }).toBeGreaterThan(1);
  await page.waitForTimeout(600);
  expect(peak).toBe(1);                                               // never a pile of stale requests
  expect(wanted.length).toBeLessThan(40);                              // 40 moves were coalesced
  const moving = wanted.filter((_, index) => index < wanted.length - 1);
  expect(moving.length).toBeGreaterThan(0);
  const last = wanted.at(-1)!;
  expect(last).toBeGreaterThan(70);                                    // ended near 90% of a 100-frame clip
  await page.getByTestId('dock-tab-faces').click();
  await expect(page.getByTestId('frame-face-strip').locator('.frame-face-card')).toHaveCount(2);   // faces came back for the resting frame
  expect(errors).toEqual([]);
});

test('a face that is already a target is selected, never locked twice; Live Pose follows it', async ({ page }) => {
  await dockTab(page, 'faces');
  const { errors } = await boot(page, { openProject: true });
  let locks = 0;
  await page.route('**/api/v1/media/detect-frame-faces', route => { locks++; return lockResponse(route); });
  const strip = page.getByTestId('frame-face-strip');
  await expect(strip.locator('.frame-face-card')).toHaveCount(2);
  await expect(strip.locator('.frame-face-card.locked')).toContainText('face 0');          // the inspection's first track is the initial target
  await strip.locator('.frame-face-card').nth(1).click();
  await expect(strip.locator('.frame-face-card.locked')).toContainText('face 1');
  await expect(page.getByTestId('live-pose')).toHaveText('+25.0° Y · +0.0° P');          // face #2's own pose in this frame, standard frame
  await expect(page.getByTestId('live-pose')).toHaveAttribute('data-source', 'target');
  await page.waitForTimeout(300);
  expect(locks).toBe(0);
  expect(errors).toEqual([]);
});

test('clicking a card of a new face locks it: its track is created, it is mapped to the active source, and Live Pose follows it', async ({ page }) => {
  await dockTab(page, 'faces');
  const { errors } = await boot(page, { openProject: true });
  await newFacesInFrame(page);
  const locks: { path: string; timestamp: number }[] = [];
  await page.route('**/api/v1/media/detect-frame-faces', route => { locks.push(route.request().postDataJSON()); return lockResponse(route); });
  const strip = page.getByTestId('frame-face-strip');
  await page.getByRole('button', { name: 'Select FaceSet Alice' }).click();            // the active source
  await expect(strip.locator('.frame-face-card')).toHaveCount(2);
  await expect(strip.locator('.frame-face-card.locked')).toHaveCount(0);               // nothing in this frame is a target yet
  await strip.locator('.frame-face-card').nth(1).click();
  await expect.poll(() => locks.length).toBe(1);
  expect(locks[0].path).toBe('roop_3d/.runtime/media/t.mp4');
  expect(locks[0].timestamp).toBeGreaterThanOrEqual(0);
  await expect(strip.locator('.frame-face-card.locked')).toHaveCount(1);
  await expect(strip.locator('.frame-face-card.locked')).toContainText('face 1');
  await expect(strip.locator('.frame-face-card.locked')).toContainText('TARGET LOCKED');
  await expect(page.getByTestId('live-pose')).toHaveText('+25.0° Y · +0.0° P');
  await expect(page.getByText(/locked and mapped to Alice/)).toBeVisible();
  await strip.locator('.frame-face-card.locked').click();                              // again: selected, not locked twice
  await page.waitForTimeout(300);
  expect(locks.length).toBe(1);
  expect(errors).toEqual([]);
});

test('clicking a new face box on the canvas locks it (sub-pixel letterbox mapping)', async ({ page }) => {
  await dockTab(page, 'faces');
  const { errors } = await boot(page, { openProject: true });
  await newFacesInFrame(page);
  let locks = 0;
  await page.route('**/api/v1/media/detect-frame-faces', route => { locks++; return lockResponse(route); });
  await expect(page.getByTestId('frame-face-strip').locator('.frame-face-card')).toHaveCount(2);
  const canvas = page.locator('.video-pane.original canvas[aria-label="Detected face bounding boxes"]');
  await canvas.scrollIntoViewIfNeeded();
  const box = (await canvas.boundingBox())!;
  // The 960x540 frame is letterboxed inside the pane: map face #1's centre (450, 210) exactly as the overlay does.
  const scale = Math.min(box.width / 960, box.height / 540);
  const x = box.x + (box.width - 960 * scale) / 2 + 450 * scale, y = box.y + (box.height - 540 * scale) / 2 + 210 * scale;
  await page.mouse.move(x, y);
  await page.mouse.click(x, y);
  await expect.poll(() => locks).toBe(1);
  await expect(page.getByTestId('frame-face-strip').locator('.frame-face-card.locked')).toContainText('face 0');
  expect(errors).toEqual([]);
});

test('Live Frame Swap Preview opens the split-wipe comparison canvas, and a second click returns to side by side', async ({ page }) => {
  const { seen, errors } = await boot(page, { openProject: true });
  const button = page.getByRole('button', { name: /Live Frame Swap Preview/ });
  await expect(button).toBeVisible();
  await button.click();
  await expect(button).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.compare-host')).toBeVisible();
  await expect.poll(() => seen.previews.length).toBeGreaterThan(0);          // the real POST /api/preview path
  await button.click();
  await expect(button).toHaveAttribute('aria-pressed', 'false');
  await expect(page.locator('.compare-host')).toHaveCount(0);
  expect(errors).toEqual([]);
});
