import { expect, test, type Page, type Route } from '@playwright/test';

/**
 * Identity-lock gate in a real browser (system Chrome, mocked API): a saved project opens an identity inspection;
 * the contact sheet shows thumbnails / visible ranges / confidence / withheld frames for each target track; the user
 * explicitly locks a mapping; the render request then carries the lock id, and a mapping that drifts from the lock, or a
 * lock the server reports stale, BLOCKS queueing instead of silently rendering.
 */
const ID = 'a'.repeat(32), LOCK_ID = 'c'.repeat(32), TARGET = 'roop_3d/.runtime/media/t.mp4';
const FEATURES = Object.fromEntries(['projects', 'face_inspection', 'queue', 'frame_preview', 'video_playback'].map(name => [name, { enabled: true, reason: null }]));
const CAPABILITIES = { models: [], job_parameter_bounds: {}, legal_parameter_bounds: {}, inspection_parameter_bounds: {}, enabled_modes: [], rendering_worker_available: true,
  workstation: { features: FEATURES, controls: {}, read_only_parameters: {}, disabled_parameters: {}, job_states: [], job_controls: {}, upload_chunk_bytes: 1048576, max_asset_bytes: 1048576 } };
const jpeg = 'R0lGODlhAQABAAAAACw=';
const trackRecord = (n: number) => ({ track_id: `track_000${n}`, persistent_id: n, scene_id: 0, landmark_count: 106, first_frame: 0, last_frame: 99, first_timestamp_seconds: 0, last_timestamp_seconds: 3.96,
  bounding_box: { x1: 1, y1: 1, x2: 40, y2: 40 }, pose_range: { yaw_degrees: [-5, 5], pitch_degrees: [0, 1], roll_degrees: [0, 1], method: 'five_point_geometric_estimate', yaw_pitch_state: 'CANDIDATE' }, thumbnail_base64: jpeg, thumbnail_mime: 'image/jpeg', sampled_detections: 90 });
const INSPECTION = { inspection_id: ID, target_sha256: 'b'.repeat(64), detector_model_id: 'det', detector_sha256: 'd'.repeat(64), tracks: [trackRecord(0), trackRecord(1)],
  decoded_frames: 100, sampled_frames: 100, truncated: false, tracking_method: 'identity_lock_v1', identity_continuity_verified: false, warnings: ['Identity lock: faces whose identity is uncertain or ambiguous are never swapped.'], scene_cut_frames: [60] };
const sheetTrack = (n: number, extra: object = {}) => ({ track_id: `track_000${n}`, persistent_id: n, first_frame: 0, last_frame: 99, segments: [[0, 99, 0, 3.96]], missing_ranges: [], visible_frames: 100, swappable_frames: 100,
  withheld_frames: 0, withheld_reasons: {}, state_frames: { TRACKED: 100 }, confidence_mean: 0.97, confidence_min: 0.81, scene_ids: [0], thumbnails: [{ frame: 5, timestamp_seconds: 0.2, jpeg_base64: jpeg }, { frame: 70, timestamp_seconds: 2.8, jpeg_base64: jpeg }], ...extra });
const SHEET = { inspection_id: ID, target_sha256: 'b'.repeat(64), truncated: false, scene_cut_frames: [60], warnings: [],
  tracks: [sheetTrack(0), sheetTrack(1, { segments: [[0, 29, 0, 1.16], [50, 99, 2, 3.96]], missing_ranges: [[30, 49]], visible_frames: 80, swappable_frames: 71, withheld_frames: 9,
    withheld_reasons: { AMBIGUOUS_IDENTITY: 6, REACQUIRING: 3 }, state_frames: { TRACKED: 71, UNCERTAIN: 6, REACQUIRED: 3 }, confidence_mean: 0.88, confidence_min: 0.31 })] };
const project = { project_id: 'p1', name: 'Demo', saved_at: '2026-11-02T00:00:00Z', last_completed_frame: -1, checkpoint_job_id: null, effective_settings: {}, inputs_verified: true,
  request: { target_path: TARGET, inspection_id: ID, face_mappings: [{ track_id: 'track_0001', faceset_id: 'alice' }], processing: 'local', remote_consent: false, trim_start_seconds: null, trim_end_seconds: null } };
const lock = (status = { valid: true, reasons: [] as object[] }) => ({ lock_id: LOCK_ID, revision: 0, digest: 'd'.repeat(64), target_path: TARGET, target_sha256: 'e'.repeat(64), inspection_id: ID,
  assignments: [{ track_id: 'track_0001', faceset_id: 'alice', faceset_sha256: 'f'.repeat(64) }], excluded_tracks: [], status });

async function boot(page: Page) {
  const seen = { posts: [] as { path: string; body: any }[], deletes: 0, jobs: [] as any[], verify: 'valid' as 'valid' | 'stale' };
  let savedRequest = project.request;
  await page.addInitScript(() => { sessionStorage.setItem('roop3d_skip_linter', 'true'); localStorage.setItem('roop3d_tour_seen', 'true'); });
  await page.route('**/api/v1/**', async (route: Route) => {
    const url = new URL(route.request().url()), path = url.pathname.replace('/api/v1', ''), method = route.request().method();
    const json = (body: unknown, status = 200) => route.fulfill({ status, json: body });
    if (method === 'POST') seen.posts.push({ path, body: route.request().postDataJSON?.() ?? null });
    if (path === '/system/capabilities') return json(CAPABILITIES);
    if (path === '/config') return json({});
    if (path === '/facesets') return json([{ faceset_id: 'alice', name: 'Alice', item_count: 1 }, { faceset_id: 'bob', name: 'Bob', item_count: 1 }]);
    if (path === '/projects') return json([project]);
    if (path === '/projects/p1' && method === 'PUT') { savedRequest = route.request().postDataJSON().request; return json({ ...project, request: savedRequest }); }
    if (path === '/projects/p1') return json(project);
    if (/^\/facesets\/[^/]+\/coverage$/.test(path)) return json({ faceset_id: path.split('/')[2], views: [{ yaw: 0, pitch: 0, roll: 0, pose_bin: 'center_center' }], bins: { center_center: 1 } });
    if (path.startsWith('/faces/bank')) return json({ version: 1, revision: 1, target_sha256: 'b'.repeat(64), next_rank: 0, clusters: [], aliases: {}, unresolved_tracks: [], inspection_id: ID });
    if (path === '/media/probe') return json({ width: 960, height: 540, duration_seconds: 4, fps_rational: '25/1', declared_frame_count: 100, audio_tracks: [] });
    if (path === '/media/keyframes') return json({ timestamps: [0] });
    if (path === `/media/inspections/${ID}`) return json(INSPECTION);
    if (path === `/identity/inspections/${ID}`) return json(SHEET);
    if (path === '/identity/locks' && method === 'POST') return json(lock(), 201);
    if (path === `/identity/locks/${LOCK_ID}` && method === 'DELETE') { seen.deletes += 1; return route.fulfill({ status: 204 }); }
    if (path === `/identity/locks/${LOCK_ID}`) return json(seen.verify === 'valid' ? lock()
      : lock({ valid: false, reasons: [{ code: 'FACESET_CHANGED', detail: 'The source FaceSet changed after it was assigned', track_id: 'track_0001', faceset_id: 'alice' } as object] }));
    if (path === '/jobs' && method === 'POST') { seen.jobs.push(route.request().postDataJSON()); return json({ job_id: '1'.repeat(32), state: 'queued', request: seen.jobs.at(-1), progress: { frame: 0, total_frames: null, fps: 0, stage: 'queued' }, output: null }, 201); }
    if (path === '/projects/p1/jobs' && method === 'POST') { seen.jobs.push(savedRequest); return json({ job_id: '1'.repeat(32), state: 'queued', request: savedRequest, progress: { frame: 0, total_frames: null, fps: 0, stage: 'queued' }, output: null }, 201); }
    return json([]);
  });
  const tinyJpeg = Buffer.from('/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA=', 'base64');
  await page.route('**/api/preview', route => route.fulfill({ status: 200, contentType: 'image/jpeg', body: tinyJpeg,
    headers: { 'X-Preview-Meta': JSON.stringify({ media_id: ID, frame_index: 0, swap_config_hash: 'x', solo_node: 'swap', width: 1, height: 1, faces: [], applied: [], unsupported: [], notes: [], latency: {} }) } }));
  await page.routeWebSocket('**/ws/**', () => undefined);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await page.getByRole('button', { name: 'Projects' }).click();
  await page.getByRole('button', { name: 'Verify & open' }).click();
  await expect(page.getByTestId('identity-stats-track_0001')).toBeVisible();
  return { seen, errors };
}

test('contact sheet shows thumbnails, visible ranges, confidence and withheld frames per target track', async ({ page }) => {
  const { errors } = await boot(page);
  const stats = page.getByTestId('identity-stats-track_0001');
  await expect(stats.locator('img')).toHaveCount(2);
  await expect(stats).toContainText('Visible 0:00–0:01, 0:02–0:03 · 80 frames');
  await expect(stats).toContainText('Confidence 0.88 (min 0.31) · reacquired 3×');
  await expect(stats).toContainText('Not detected: frames 30–49');
  await expect(stats).toContainText('9 frames kept original: ambiguous identity ×6, reacquiring ×3');
  await expect(page.getByTestId('identity-stats-track_0000')).toContainText('Confidence 0.97');
  await expect(page.getByLabel('Identity for track_0001')).toHaveValue('alice');
  await expect(page.getByLabel('Identity for track_0000')).toHaveValue('');
  await expect(page.getByTestId('identity-lock')).toContainText('Identity not locked');
  expect(errors).toEqual([]);
  await page.getByTestId('identity-lock').scrollIntoViewIfNeeded();
  await page.screenshot({ path: 'test-results/identity-contact-sheet.png', fullPage: true });
});

test('explicit lock is sent with the render; drift and stale locks block queueing', async ({ page }) => {
  const { seen, errors } = await boot(page);
  const queue = page.getByRole('button', { name: /Queue render/ });

  // Nothing is locked until the user asks: the render carries no lock and no unselected face is mapped.
  await queue.click();
  await expect.poll(() => seen.jobs.length).toBe(1);
  expect(seen.jobs[0].identity_lock_id).toBeUndefined();
  expect(seen.jobs[0].face_mappings).toEqual([{ track_id: 'track_0001', faceset_id: 'alice' }]);

  await page.getByRole('button', { name: 'Lock mapping' }).click();
  await expect(page.getByTestId('identity-lock')).toContainText('Identity locked');
  await expect(page.getByTestId('identity-lock')).toContainText('Face 0001 → Alice');
  expect(seen.posts.find(item => item.path === '/identity/locks')!.body).toEqual({
    target_path: TARGET, inspection_id: ID, assignments: [{ track_id: 'track_0001', faceset_id: 'alice' }] });

  await queue.click();
  await expect.poll(() => seen.jobs.length).toBe(2);
  expect(seen.jobs[1].identity_lock_id).toBe(LOCK_ID);
  expect(seen.jobs[1].face_segments.filter((item: any) => item.role === 'swap').map((item: any) => [item.track_id, item.faceset_id])).toEqual([['track_0001', 'alice']]);

  // Manual correction on screen without re-locking: the render is refused.
  await page.getByLabel('Identity for track_0000').selectOption('bob');
  await expect(page.getByRole('alert')).toContainText('differs from the identity lock');
  await queue.click();
  await expect(page.getByText(/Re-lock or release the identity lock before rendering/)).toBeVisible();
  expect(seen.jobs.length).toBe(2);

  // Re-lock the corrected mapping (replaces the old lock), then release it.
  await page.getByRole('button', { name: 'Re-lock mapping' }).click();
  await expect.poll(() => seen.deletes).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'Release lock' }).click();
  await expect(page.getByTestId('identity-lock')).toContainText('Identity not locked');
  expect(errors).toEqual([]);
});

test('a lock the server reports stale is shown with its reasons and blocks the render', async ({ page }) => {
  const { seen, errors } = await boot(page);
  seen.verify = 'stale';                                     // e.g. the FaceSet file was edited on disk after locking
  await page.getByRole('button', { name: 'Lock mapping' }).click();
  await expect(page.getByTestId('identity-lock')).toContainText('needs attention');
  await expect(page.getByRole('alert')).toContainText('FACESET_CHANGED');
  await expect(page.getByTestId('identity-lock')).toContainText('The source FaceSet changed after it was assigned');
  await page.getByRole('button', { name: /Queue render/ }).click();
  await expect(page.getByText(/Re-lock or release the identity lock before rendering/)).toBeVisible();
  expect(seen.jobs.length).toBe(0);
  expect(errors).toEqual([]);
});
