import { expect, test, type Page, type Route } from '@playwright/test';

/**
 * Stage 7 studio workstation in a real browser (system Chrome, mocked API, real React / WebGL / HTML5 drag and drop):
 * four panels + timeline lanes, FaceSet drag onto a target track / a face in the video, click-to-select a face, the
 * inspector's 3D render mode / relighting / engine reaching the live preview and the saved project, identity asset
 * build → 3D view, undo / redo, themes, the classic layout fallback and responsive widths without page overflow.
 */
const ID = 'a'.repeat(32), TARGET = 'roop_3d/.runtime/media/t.mp4';
const FEATURES = Object.fromEntries(['projects', 'face_inspection', 'queue', 'frame_preview', 'video_playback', 'uploads', 'trim'].map(name => [name, { enabled: true, reason: null }]));
const CAPABILITIES = { models: [], job_parameter_bounds: {}, legal_parameter_bounds: {}, inspection_parameter_bounds: {}, enabled_modes: [], rendering_worker_available: true,
  workstation: { features: FEATURES, controls: {}, read_only_parameters: {}, disabled_parameters: {}, job_states: [], job_controls: {}, upload_chunk_bytes: 1048576, max_asset_bytes: 1048576 } };
const gif = 'R0lGODlhAQABAAAAACw=';
const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==';
const tinyJpeg = Buffer.from('/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA=', 'base64');
// Two faces side by side on a 960×540 frame, so a click / drop on the right face must pick track_0001.
const track = (n: number, x1: number) => ({ track_id: `track_000${n}`, persistent_id: n, scene_id: 0, landmark_count: 106, first_frame: 0, last_frame: 99, first_timestamp_seconds: 0, last_timestamp_seconds: 3.96,
  bounding_box: { x1, y1: 120, x2: x1 + 160, y2: 360 }, pose_range: { yaw_degrees: [-5, 5], pitch_degrees: [0, 1], roll_degrees: [0, 1], method: 'five_point_geometric_estimate' }, thumbnail_base64: gif, thumbnail_mime: 'image/gif', sampled_detections: 10 });
const INSPECTION = { inspection_id: ID, target_sha256: 'b'.repeat(64), tracks: [track(0, 140), track(1, 620)], truncated: false, tracking_method: 'identity_lock_v1', warnings: [], scene_cut_frames: [60] };
const sheetTrack = (n: number) => ({ track_id: `track_000${n}`, persistent_id: n, first_frame: 0, last_frame: 99, segments: n ? [[0, 29, 0, 1.16], [50, 99, 2, 3.96]] : [[0, 99, 0, 3.96]], missing_ranges: n ? [[30, 49]] : [],
  visible_frames: 100, swappable_frames: 100, withheld_frames: 0, withheld_reasons: {}, state_frames: { TRACKED: 100 }, confidence_mean: 0.93, confidence_min: 0.71, scene_ids: [0], thumbnails: [] });
const SHEET = { inspection_id: ID, target_sha256: 'b'.repeat(64), truncated: false, scene_cut_frames: [60], warnings: [], tracks: [sheetTrack(0), sheetTrack(1)] };
const project = { project_id: 'p1', name: 'Demo', saved_at: '2026-11-02T00:00:00Z', last_completed_frame: -1, checkpoint_job_id: null, effective_settings: {}, inputs_verified: true,
  request: { target_path: TARGET, inspection_id: ID, face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }], processing: 'local', remote_consent: false, trim_start_seconds: null, trim_end_seconds: null } };
const VIEWS = [{ index: 0, yaw: 0, pitch: 0, roll: 0 }, { index: 1, yaw: -45, pitch: 0, roll: 0 }, { index: 2, yaw: 44, pitch: -15, roll: 0 }];
const status = (built: boolean, id: string) => ({ faceset_id: id, name: id, state: built ? 'built' : 'not_built', build_key: 'k'.repeat(64), note: '', flame: {},
  coverage: { readiness: 'partial_coverage', view_count: 3, usable_count: 3, yaw_range_deg: [-45, 44], yaw_span_deg: 89, pitch_span_deg: 15, yaw_coverage: {}, pitch_coverage: {}, missing: [], requests: ['Add a profile view'], duplicate_groups: [], capabilities: {} },
  asset: built ? { build_key: 'k'.repeat(64), created_at: 'now', algorithm_version: 'identity-mv-2', reconstruction_level: 'multiview', landmark_layout: '5pt', claims: {}, missing_information: ['back of head'], limitations: [],
    statistics: {}, sources_used: 3, sources_total: 3, vertex_count: 4, face_count: 4, texture_size: [1, 1] } : null });
const MESH = { method: 'identity_asset', coordinate_system: 'opengl', vertices: [[0, 0.1, 0], [-0.08, -0.05, 0.02], [0.08, -0.05, 0.02], [0, -0.02, 0.09]], faces: [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
  uvs: [[0.5, 1], [0, 0], [1, 0], [0.5, 0.5]], texture: `data:image/png;base64,${PNG}`, provenance: 'test asset' };

test.use({ viewport: { width: 1680, height: 1050 } });

async function boot(page: Page, options: { width?: number } = {}) {
  const seen = { previews: [] as any[], projects: [] as any[], builds: 0 };
  let built = false;
  await page.addInitScript(() => { sessionStorage.setItem('roop3d_skip_linter', 'true'); localStorage.setItem('roop3d_tour_seen', 'true'); });
  await page.route('**/api/v1/**', async (route: Route) => {
    const url = new URL(route.request().url()), path = url.pathname.replace('/api/v1', ''), method = route.request().method();
    const json = (body: unknown, code = 200) => route.fulfill({ status: code, json: body });
    if (path === '/system/capabilities') return json(CAPABILITIES);
    if (path === '/config') return json({});
    if (path === '/facesets') return json([{ faceset_id: 'alice', name: 'Alice', item_count: 3 }, { faceset_id: 'bob', name: 'Bob', item_count: 3 }]);
    if (path === '/projects' && method === 'POST') { seen.projects.push(route.request().postDataJSON()); return json({ ...project, request: seen.projects.at(-1).request }, 201); }
    if (path === '/projects') return json([project]);
    if (path === '/projects/p1') return json(project);
    if (path === '/models/hub') return json({ models: [], model_states: [], execution_policy: '', download_policy: '', engines: [
      { engine_id: 'reference_onnx', name: 'Roop 3D Reference ONNX', status: 'READY_FOR_INFERENCE', selectable: true, actual_inference_available: true, evidence: 'hash checked' },
      { engine_id: 'experimental_x', name: 'Experimental X', status: 'NOT_INSTALLED', selectable: false, actual_inference_available: false, evidence: 'weights absent' }] });
    if (/^\/facesets\/[^/]+\/views\/\d+\/image$/.test(path) || /^\/facesets\/[^/]+\/thumb$/.test(path)) return route.fulfill({ status: 200, contentType: 'image/jpeg', body: tinyJpeg });
    if (/^\/facesets\/[^/]+\/views$/.test(path)) return json(VIEWS);
    if (/^\/facesets\/[^/]+\/coverage$/.test(path)) return json({ faceset_id: path.split('/')[2], views: [], bins: {} });
    if (/^\/facesets\/[^/]+\/identity-asset$/.test(path)) { if (method === 'POST') { seen.builds += 1; built = true; } return json({ ...status(built, path.split('/')[2]), reused: false }); }
    if (/^\/facesets\/[^/]+\/identity-asset\/mesh$/.test(path)) return built ? json(MESH) : json({ error: { message: 'not built' } }, 404);
    if (path.startsWith('/faces/bank')) return json({ version: 1, revision: 1, target_sha256: 'b'.repeat(64), next_rank: 0, clusters: [], aliases: {}, unresolved_tracks: [], inspection_id: ID });
    if (path === '/media/probe') return json({ width: 960, height: 540, duration_seconds: 4, fps_rational: '25/1', declared_frame_count: 100, audio_tracks: [] });
    if (path === '/media/keyframes') return json({ timestamps: [0, 2] });
    if (path === '/media/preview-frame') return route.fulfill({ status: 200, contentType: 'image/jpeg', body: tinyJpeg });
    if (path === '/media/content') return route.fulfill({ status: 204 });
    if (path === `/media/inspections/${ID}` || path === '/media/inspect-faces') return json(INSPECTION);
    if (path === `/identity/inspections/${ID}`) return json(SHEET);
    if (path === '/reconstructions') return json([]);
    return json([]);
  });
  await page.route('**/api/preview', route => { seen.previews.push(route.request().postDataJSON()); return route.fulfill({ status: 200, contentType: 'image/jpeg', body: tinyJpeg,
    headers: { 'X-Preview-Meta': JSON.stringify({ media_id: ID, frame_index: 0, swap_config_hash: String(seen.previews.length), solo_node: 'final', width: 1, height: 1, faces: [], applied: [], unsupported: [], notes: [], latency: { total_ms: 12 } }) } }); });
  await page.routeWebSocket('**/ws/**', () => undefined);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  if (options.width) await page.setViewportSize({ width: options.width, height: 1000 });
  await page.goto('/#/faceswap');
  await page.getByRole('button', { name: 'Projects' }).click();
  await page.getByRole('button', { name: 'Verify & open' }).click();
  await expect(page.locator('[data-track-id="track_0001"]')).toBeVisible();
  return { seen, errors };
}

/** Client point of the centre of `track`'s face box inside the source overlay canvas (letterboxed 960×540). */
async function facePoint(page: Page, x1: number) {
  const canvas = page.getByLabel('Detected face bounding boxes').first();
  const box = (await canvas.boundingBox())!;
  const scale = Math.min(box.width / 960, box.height / 540), ox = (box.width - 960 * scale) / 2, oy = (box.height - 540 * scale) / 2;
  return { canvas, position: { x: ox + (x1 + 80) * scale, y: oy + 240 * scale } };
}

test('four panels, timeline lanes and a FaceSet dragged onto a target track drive the live preview', async ({ page }) => {
  const { seen, errors } = await boot(page);
  for (const name of ['Target media', 'Source identities', 'Preview', 'Inspector', 'Timeline']) await expect(page.getByRole(name === 'Timeline' ? 'region' : name === 'Preview' ? 'main' : 'complementary', { name, exact: true })).toBeVisible();
  await expect(page.getByLabel('Face track lanes')).toBeVisible();
  await expect(page.locator('.lane-cut')).toHaveCount(1);
  await expect(page.locator('[data-lane-track="track_0001"] .lane-span')).toHaveCount(2);
  await expect(page.locator('[data-lane-track="track_0001"] .lane-gap')).toHaveCount(1);

  // Identity panel: views gallery, coverage indicator, asset status.
  await page.getByRole('button', { name: 'Select FaceSet Bob' }).click();
  await expect(page.getByRole('list', { name: 'Bob views' }).locator('img')).toHaveCount(3);
  await expect(page.locator('.coverage-cell:not(.empty)')).toHaveCount(3);
  await expect(page.locator('.asset-card')).toContainText('partial coverage');

  // Drag Bob onto Face 1's card.
  await page.locator('.fs-row[data-faceset-id="bob"]').dragTo(page.locator('[data-track-id="track_0001"]'));
  await expect(page.getByLabel('Identity for track_0001')).toHaveValue('bob');
  await expect(page.locator('[data-lane-track="track_0001"] .lane-label')).toContainText('Bob');
  await expect.poll(() => seen.previews.at(-1)?.face_mappings).toEqual(expect.arrayContaining([{ track_id: 'track_0001', faceset_id: 'bob' }]));
  await expect(page.locator('img.live-swap-preview')).toBeVisible();

  // Undo / redo the assignment through the global shortcuts.
  await page.locator('body').click({ position: { x: 5, y: 5 } });
  await page.keyboard.press('Control+z');
  await expect(page.getByLabel('Identity for track_0001')).toHaveValue('');
  await page.keyboard.press('Control+y');
  await expect(page.getByLabel('Identity for track_0001')).toHaveValue('bob');

  // Drag onto the lane of Face 0 reassigns it too.
  await page.locator('.fs-row[data-faceset-id="bob"]').dragTo(page.locator('[data-lane-track="track_0000"] .lane-body'));
  await expect(page.getByLabel('Identity for track_0000')).toHaveValue('bob');
  expect(errors).toEqual([]);
  await page.screenshot({ path: 'test-results/studio-assigned.png' });
});

test('clicking a face in the video selects its track; right-click offers extraction; dropping a FaceSet on it assigns', async ({ page }) => {
  const { errors } = await boot(page);
  const right = await facePoint(page, 620);
  await right.canvas.click({ position: right.position });
  await expect(page.locator('.inspector-section').first().locator('summary')).toContainText('Face 1');
  await expect(page.locator('[data-track-id="track_0001"]')).toHaveClass(/active/);
  await expect(page.locator('[data-lane-track="track_0001"]')).toHaveClass(/selected/);
  await page.keyboard.press('Escape');

  const left = await facePoint(page, 140);
  await left.canvas.click({ position: left.position, button: 'right' });
  await expect(page.getByRole('menu', { name: /Use face 1 from this frame/ })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('.inspector-section').first().locator('summary')).toContainText('Face 0');

  await page.locator('.fs-row[data-faceset-id="bob"]').dragTo(right.canvas, { targetPosition: right.position });
  await expect(page.getByLabel('Identity for track_0001')).toHaveValue('bob');
  expect(errors).toEqual([]);
});

test('inspector: 3D render mode, relighting and engine reach the preview and the saved project', async ({ page }) => {
  const { seen, errors } = await boot(page);
  await expect(page.getByLabel('Swap engine')).toHaveValue('reference_onnx');
  await expect(page.getByLabel('Swap engine').locator('option[value="experimental_x"]')).toHaveAttribute('disabled', '');

  await page.getByLabel('3D render mode').selectOption('reenactment_3d');
  await expect.poll(() => seen.previews.at(-1)?.render_mode).toBe('reenactment_3d');
  await page.getByLabel('Occlusion handling').selectOption('robust');
  await expect.poll(() => seen.previews.at(-1)?.occlusion_mode).toBe('robust');
  await page.getByLabel(/3D relighting/).check();
  await expect.poll(() => seen.previews.at(-1)?.relight).toBe(true);
  await expect(page.getByText('Relighting is off')).toHaveCount(0);

  await page.getByRole('button', { name: 'Save project' }).click();
  await expect.poll(() => seen.projects.length).toBe(1);
  const saved = seen.projects[0].request.workstation_parameters;
  expect(saved.render_mode).toBe('reenactment_3d');
  expect(saved.occlusion_mode).toBe('robust');
  expect(saved.relight).toBe(true);

  // Standard mode drops the key again, so the request matches pre-Stage-7 requests exactly.
  await page.getByLabel('3D render mode').selectOption('standard');
  await expect.poll(() => seen.previews.at(-1)?.render_mode ?? 'absent').toBe('absent');
  expect(errors).toEqual([]);
});

test('identity asset: explicit build, then the textured mesh is shown in the 3D viewer and released', async ({ page }) => {
  const { seen, errors } = await boot(page);
  await page.getByRole('button', { name: 'Select FaceSet Alice' }).click();
  await expect(page.locator('.asset-card')).toContainText('Not built yet');
  expect(seen.builds).toBe(0);
  await page.getByRole('button', { name: 'Build 3D identity asset' }).click();
  await expect(page.getByRole('button', { name: 'View 3D asset' })).toBeVisible();
  expect(seen.builds).toBe(1);
  await expect(page.locator('.asset-facts')).toContainText('multiview');
  await page.getByRole('button', { name: 'View 3D asset' }).click();
  await expect(page.getByRole('button', { name: 'Hide 3D asset' })).toBeVisible();
  const viewer = page.locator('.inspector-viewer');
  await expect(viewer.getByRole('button', { name: 'Full 3D Reconstructed' })).toHaveClass(/active/);
  await expect(viewer.locator('canvas')).toBeVisible();
  // Switching FaceSet releases the mesh (it belongs to Alice).
  await page.getByRole('button', { name: 'Select FaceSet Bob' }).click();
  await expect(viewer.getByRole('button', { name: 'Estimated 2.5D' })).toHaveClass(/active/);
  await expect(viewer.getByRole('button', { name: 'Full 3D Reconstructed' })).toBeDisabled();
  expect(errors).toEqual([]);
});

test('themes, the classic layout fallback and responsive widths without horizontal page overflow', async ({ page }) => {
  const { errors } = await boot(page);
  await page.getByRole('button', { name: /Switch to light theme/ }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.screenshot({ path: 'test-results/studio-light.png' });
  await page.getByRole('button', { name: /Switch to dark theme/ }).click();

  await page.getByRole('button', { name: 'Studio layout' }).click();
  await expect(page.getByRole('heading', { name: 'Assign an identity' })).toBeVisible();
  await expect(page.getByTestId('studio-layout')).toHaveCount(0);
  await page.getByRole('button', { name: 'Classic layout' }).click();
  await expect(page.getByTestId('studio-layout')).toBeVisible();

  for (const width of [1920, 1440, 1100, 820, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await expect(page.getByTestId('studio-layout')).toBeVisible();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow, `page overflow at ${width}px`).toBeLessThanOrEqual(0);
    await expect(page.locator('.studio-center .viewport')).toBeVisible();
  }
  expect(errors).toEqual([]);
});
