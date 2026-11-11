import { expect, test, type Page, type Route } from '@playwright/test';
import { MEDIA_INFO, mockFilmstrip, mockFrame } from './helpers/mockApi';

/**
 * The 3D-native workstation features in a real browser (system Chrome, mocked API, real React / WebGL2 / canvas):
 * the eight-stage pipeline node flow bar with measured latencies and 1-8 solo hotkeys, the angular coverage radar with its
 * missing-profile warning, the 6DoF mesh / gaze overlay and the per-face right-click menu.
 */
const ID = 'a'.repeat(32), TARGET = 'roop_3d/.runtime/media/t.mp4';
const FEATURES = Object.fromEntries(['projects', 'face_inspection', 'queue', 'frame_preview', 'video_playback', 'uploads', 'trim'].map(name => [name, { enabled: true, reason: null }]));
const CAPABILITIES = { models: [], job_parameter_bounds: {}, legal_parameter_bounds: {}, inspection_parameter_bounds: {}, enabled_modes: [], rendering_worker_available: true,
  workstation: { features: FEATURES, controls: {}, read_only_parameters: {}, disabled_parameters: {}, job_states: [], job_controls: {}, upload_chunk_bytes: 1048576, max_asset_bytes: 1048576 } };
const gif = 'R0lGODlhAQABAAAAACw=';
const tinyJpeg = Buffer.from('/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAAAAE//9k=', 'base64');
const track = (n: number, x1: number) => ({ track_id: `track_000${n}`, persistent_id: n, scene_id: 0, landmark_count: 106, first_frame: 0, last_frame: 99, first_timestamp_seconds: 0, last_timestamp_seconds: 3.96,
  bounding_box: { x1, y1: 120, x2: x1 + 160, y2: 360 }, pose_range: { yaw_degrees: [-5, 78], pitch_degrees: [0, 1], roll_degrees: [0, 1], method: 'five_point_geometric_estimate' }, thumbnail_base64: gif, thumbnail_mime: 'image/gif', sampled_detections: 4 });
const INSPECTION = { inspection_id: ID, target_sha256: 'b'.repeat(64), tracks: [track(0, 140), track(1, 620)], truncated: false, tracking_method: 'identity_lock_v1', warnings: [], scene_cut_frames: [60] };
const sheetTrack = (n: number) => ({ track_id: `track_000${n}`, persistent_id: n, first_frame: 0, last_frame: 99, segments: [[0, 99, 0, 3.96]], missing_ranges: [], visible_frames: 100, swappable_frames: 100, withheld_frames: 0, withheld_reasons: {}, state_frames: { TRACKED: 100 }, confidence_mean: 0.93, confidence_min: 0.71, scene_ids: [0], thumbnails: [] });
const SHEET = { inspection_id: ID, target_sha256: 'b'.repeat(64), truncated: false, scene_cut_frames: [60], warnings: [], tracks: [sheetTrack(0), sheetTrack(1)] };
const project = { project_id: 'p1', name: 'Demo', saved_at: '2026-11-02T00:00:00Z', last_completed_frame: -1, checkpoint_job_id: null, effective_settings: {}, inputs_verified: true,
  request: { target_path: TARGET, inspection_id: ID, face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }], processing: 'local', remote_consent: false, trim_start_seconds: null, trim_end_seconds: null } };
// Frontal and three-quarter views only: both profile bands are empty, and the target track turns to +78 degrees.
const COVERAGE = { faceset_id: 'alice', views: [{ yaw: 0, pitch: 0, roll: 0, pose_bin: 'front' }, { yaw: -40, pitch: 0, roll: 0, pose_bin: 'left' }, { yaw: 42, pitch: 8, roll: 0, pose_bin: 'right' }], bins: { front: 1, left: 1, right: 1 },
  observed_views: 3, sphere_coverage_count: 3, sphere_coverage_total: 96, sphere_coverage_pct: 3 };
const BACKBONES = { default: '3ddfa_v3_hrn', active: '3ddfa_v3_hrn', backbones: [
  { id: '3ddfa_v3_hrn', label: '3DDFA-V3 Dense 35k + HRN Detail (Photorealistic Default)', available: true, active: true, reason: '' },
  { id: 'flame_smirk', label: 'FLAME 2023 Open + SMIRK Micro-Expressions', available: false, active: false, reason: 'The installed bundle is the bfm35709 family; this backbone needs its own bundle.' },
  { id: 'faceverse_v4', label: 'FaceVerse V4 (52 ARKit Blendshapes)', available: false, active: false, reason: 'Not supported: this build has no FaceVerse topology, exporter or bundle schema.' },
  { id: 'makehuman_legacy', label: 'Legacy MakeHuman CC0 (Debug Only)', available: false, active: false, reason: 'The MakeHuman head is not built.' }] };
const LATENCY = { detection_ms: 9.4, retarget_ms: 3.1, projection_ms: 2.2, swap_ms: 28, total_ms: 71 };

test.use({ viewport: { width: 1680, height: 1050 } });

async function boot(page: Page) {
  const seen = { previews: [] as any[] };
  await page.addInitScript(() => { sessionStorage.setItem('roop3d_skip_linter', 'true'); localStorage.setItem('roop3d_tour_seen', 'true'); localStorage.setItem('roop3d-onboarding-v1', '1'); });
  await page.route('**/api/v1/**', async (route: Route) => {
    const url = new URL(route.request().url()), path = url.pathname.replace('/api/v1', ''), method = route.request().method();
    const json = (body: unknown, code = 200) => route.fulfill({ status: code, json: body });
    if (path === '/system/capabilities') return json(CAPABILITIES);
    if (path === '/config') return json({});
    if (path === '/facesets') return json([{ faceset_id: 'alice', name: 'Alice', item_count: 3 }, { faceset_id: 'bob', name: 'Bob', item_count: 3 }]);
    if (path === '/projects' && method === 'PUT') return json(project);
    if (path === '/projects') return json([project]);
    if (path === '/projects/p1') return json(project);
    if (path === '/jobs') return json([]);
    if (path === '/models/hub') return json({ models: [], model_states: [], execution_policy: '', download_policy: '', engines: [{ engine_id: 'reference_onnx', name: 'Roop 3D Reference ONNX', status: 'READY_FOR_INFERENCE', selectable: true, actual_inference_available: true, evidence: 'hash checked' }] });
    if (/^\/facesets\/[^/]+\/(views\/\d+\/image|thumb)$/.test(path)) return route.fulfill({ status: 200, contentType: 'image/jpeg', body: tinyJpeg });
    if (/^\/facesets\/[^/]+\/views$/.test(path)) return json(COVERAGE.views.map((view, index) => ({ index, ...view })));
    if (/^\/facesets\/[^/]+\/coverage$/.test(path)) return json(COVERAGE);
    if (path === '/faceset/reconstruct-3d' || path === '/faceset/3d-state') return json(COVERAGE);
    if (path === '/reconstruction/backbones') return json(BACKBONES);
    if (/^\/facesets\/[^/]+\/identity-asset/.test(path)) return json({ error: { message: 'not built' } }, 404);
    if (path.startsWith('/faces/bank')) return json({ version: 1, revision: 1, target_sha256: 'b'.repeat(64), next_rank: 0, clusters: [], aliases: {}, unresolved_tracks: [], inspection_id: ID });
    if (path === '/media/probe') return json({ width: 960, height: 540, duration_seconds: 4, fps_rational: '25/1', declared_frame_count: 100, audio_tracks: [] });
    if (path === '/media/keyframes') return json({ timestamps: [0, 2] });
    if (path === '/media/preview-frame') return route.fulfill({ status: 200, contentType: 'image/jpeg', body: tinyJpeg });
    if (path === '/media/frame') return route.fulfill(mockFrame(url));
    if (path === '/media/info') return json(MEDIA_INFO);
    if (path === '/media/filmstrip') return json(mockFilmstrip(url));
    if (path === '/media/content') return route.fulfill({ status: 204 });
    if (path === `/media/inspections/${ID}` || path === '/media/inspect-faces') return json(INSPECTION);
    if (path === `/identity/inspections/${ID}`) return json(SHEET);
    return json([]);
  });
  await page.route('**/api/preview', route => {
    const body = route.request().postDataJSON();
    seen.previews.push(body);
    return route.fulfill({ status: 200, contentType: 'image/jpeg', body: tinyJpeg, headers: { 'X-Preview-Meta': JSON.stringify({ media_id: ID, frame_index: 12, swap_config_hash: String(seen.previews.length),
      solo_node: body.solo_node ?? 'final', width: 1, height: 1, faces: [], applied: [], unsupported: [], notes: [], latency: LATENCY }) } });
  });
  await page.routeWebSocket('**/ws/**', () => undefined);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/#/faceswap');
  await page.getByRole('button', { name: 'Projects' }).click();
  await page.getByRole('button', { name: 'Verify & open' }).click();
  await expect(page.locator('[data-track-id="track_0001"]')).toBeVisible();
  return { seen, errors };
}

test('the pipeline node flow bar shows eight stages with measured latencies and solos a stage with its number key', async ({ page }) => {
  const { seen, errors } = await boot(page);
  const bar = page.getByRole('region', { name: 'Visual execution pipeline' });
  await expect(bar).toBeVisible();                                           // always visible: not inside a collapsed section
  await expect(bar.locator('.pipeline-node')).toHaveCount(8);
  await expect(bar.locator('.node-name')).toHaveText(['Raw Target', '6DoF Mesh & Pose', 'UV Projection', 'SH Relight & Normals', 'Base Swap', 'UltraMax', 'XSeg + Hair Matte', 'Final Composite']);
  await expect.poll(() => seen.previews.length).toBeGreaterThan(0);
  await expect(bar.locator('[data-node="input"] .node-ms')).toHaveText('9.4 ms');
  await expect(bar.locator('[data-node="swap"] .node-ms')).toHaveText('28 ms');
  await expect(bar.locator('[data-node="final"] .node-ms')).toHaveText('Σ 71 ms');
  await expect(bar.locator('[data-node="lighting"] .node-ms')).toHaveText('—');          // the preview does not time this stage: no invented number
  await expect(bar.locator('.pipeline-flow-status')).toContainText('Frame 12 · total 71 ms');

  await page.locator('body').click({ position: { x: 5, y: 5 } });
  await page.keyboard.press('3');
  await expect.poll(() => seen.previews.at(-1)?.solo_node).toBe('source');
  await expect(bar.locator('[data-node="uv"]')).toHaveClass(/solo/);
  await page.keyboard.press('3');                                              // the same key returns to the composite
  await expect.poll(() => seen.previews.at(-1)?.solo_node).toBe('final');
  await bar.getByRole('button', { name: 'Solo XSeg + Hair Occlusion Matte' }).click();       // one click
  await expect.poll(() => seen.previews.at(-1)?.solo_node).toBe('matte');
  await bar.getByRole('button', { name: 'Solo 3D Normal Map' }).click();
  await expect.poll(() => seen.previews.at(-1)?.solo_node).toBe('normal_map');
  await page.screenshot({ path: 'test-results/w3d-flow-bar.png' });
  expect(errors).toEqual([]);
});

test('the coverage radar lists the missing profile angles and marks target poses the FaceSet cannot serve', async ({ page }) => {
  const { errors } = await boot(page);
  await page.getByRole('button', { name: 'Select FaceSet Alice' }).click();
  const radar = page.getByTestId('coverage-section').getByTestId('coverage-radar');   // the studio inspector's radar (the classic one sits in the closed Advanced drawer)
  await expect(radar).toBeVisible();
  await expect(radar.locator('.radar-cell')).toHaveCount(96);
  await expect(radar.locator('.radar-cell.filled')).toHaveCount(3);
  await expect(radar.getByRole('status')).toContainText(/No left profile or right profile view/);
  await expect(radar.locator('.radar-badge')).toHaveClass(/warn/);
  await radar.scrollIntoViewIfNeeded();
  await page.screenshot({ path: 'test-results/w3d-radar.png' });
  expect(errors).toEqual([]);
});

test('the morph panel writes the engine blend and the masking inset sends the hairline offset to the live preview only once touched', async ({ page }) => {
  const { seen, errors } = await boot(page);
  await expect.poll(() => seen.previews.length).toBeGreaterThan(0);
  expect('hairline_offset_px' in seen.previews[0]).toBe(false);
  const inset = page.getByTestId('step-mask').getByRole('slider', { name: 'Forehead hairline inset' });
  await inset.fill('12');
  await expect.poll(() => seen.previews.at(-1)?.hairline_offset_px).toBe(-12);          // the inset is the engine offset with the sign flipped
  await page.getByTestId('morph-section').locator(':scope > summary').click();
  await page.getByRole('radio', { name: /3D Geometry \+ Jaw Reshape Weight/ }).click();                   // an opened saved project starts in reference-retarget pose mode
  const blend = page.getByLabel('Jawline and cranial volume blend');
  await blend.fill('80');
  await expect.poll(() => seen.previews.at(-1)?.pose_strength).toBeCloseTo(0.8);
  expect(errors).toEqual([]);
});

test('the Inspector exposes the backbone, influence mode and hair & neck protection, and only touched controls reach the preview', async ({ page }) => {
  const { seen, errors } = await boot(page);
  await expect.poll(() => seen.previews.length).toBeGreaterThan(0);
  for (const key of ['neck_protection', 'hair_matting', 'preserve_ears_glasses', 'jaw_feather_px', 'skin_texture_retention', 'hairline_offset_px']) expect(key in seen.previews[0]).toBe(false);

  const mask = page.getByTestId('step-mask');
  await mask.scrollIntoViewIfNeeded();
  await expect(mask.getByRole('checkbox', { name: 'Strict Neck & Throat Protection' })).toBeChecked();
  await expect(mask.getByRole('checkbox', { name: 'Hair-Strand Guided Matting' })).toBeChecked();
  await expect(mask.getByRole('checkbox', { name: 'Preserve Target Ears & Eyeglasses' })).toBeChecked();
  await mask.getByRole('checkbox', { name: 'Hair-Strand Guided Matting' }).uncheck();
  await expect.poll(() => seen.previews.at(-1)?.hair_matting).toBe(false);
  await mask.getByRole('slider', { name: 'Jaw-to-neck feather falloff' }).fill('16');
  await expect.poll(() => seen.previews.at(-1)?.jaw_feather_px).toBe(16);
  await expect(mask.getByRole('slider', { name: 'Skin micro-texture and pore retention' })).toBeDisabled();        // the default enhancer is not UltraMax
  await page.screenshot({ path: 'test-results/w3d-protection.png' });

  await page.getByTestId('morph-section').locator(':scope > summary').click();
  const select = page.getByLabel('3D head model backbone');
  await expect(select.locator('option')).toHaveText([/3DDFA-V3 Dense 35k/, /FLAME 2023 Open.*unavailable/, /FaceVerse V4.*unavailable/, /Legacy MakeHuman.*unavailable/]);
  await expect(select.locator('option[value="flame_smirk"]')).toHaveAttribute('disabled', '');
  await expect(page.getByRole('radio', { name: /3D Pose-Guided Neural Swap/ })).toHaveAttribute('aria-checked', 'true');   // an opened saved project starts in reference-retarget pose mode
  await page.getByTestId('morph-section').scrollIntoViewIfNeeded();
  await page.screenshot({ path: 'test-results/w3d-backbone.png' });
  expect(errors).toEqual([]);
});

test('the swap scope selector defaults to the full craniofacial lock, and only a changed scope reaches the live preview', async ({ page }) => {
  const { seen, errors } = await boot(page);
  await expect.poll(() => seen.previews.length).toBeGreaterThan(0);
  expect('swap_scope' in seen.previews[0] || 'skull_blend' in seen.previews[0]).toBe(false);

  const scope = page.getByTestId('swap-scope');
  await scope.scrollIntoViewIfNeeded();
  const options = scope.getByRole('radio');
  await expect(options).toHaveCount(3);
  await expect(scope.getByRole('radio', { name: /Full Craniofacial/ })).toHaveAttribute('aria-checked', 'true');
  await expect(scope.getByRole('slider')).toHaveCount(0);

  await scope.getByRole('radio', { name: /Adaptive Skull Blend/ }).click();
  await expect(scope.getByRole('slider', { name: 'Source skull share' })).toHaveValue('50');
  await expect.poll(() => seen.previews.at(-1)?.swap_scope).toBe('adaptive_skull');
  await scope.getByRole('slider', { name: 'Source skull share' }).fill('80');
  await expect.poll(() => seen.previews.at(-1)?.skull_blend).toBeCloseTo(0.8);
  await expect(scope).toContainText('β source 80% · β target 20%');
  await page.screenshot({ path: 'test-results/w3d-swap-scope.png' });

  await scope.getByRole('radio', { name: /Inner Face/ }).click();
  await expect.poll(() => seen.previews.at(-1)?.swap_scope).toBe('inner_face');
  expect('skull_blend' in (seen.previews.at(-1) ?? {})).toBe(false);
  await scope.getByRole('radio', { name: /Full Craniofacial/ }).click();
  await expect.poll(() => 'swap_scope' in (seen.previews.at(-1) ?? {})).toBe(false);
  expect(errors).toEqual([]);
});

test('the Mesh and Gaze toggles draw a 6DoF overlay and the right-click menu offers the per-face toggles', async ({ page }) => {
  const { errors } = await boot(page);
  const canvas = page.getByLabel('Detected face bounding boxes').first();
  const box = (await canvas.boundingBox())!;
  const scale = Math.min(box.width / 960, box.height / 540), ox = (box.width - 960 * scale) / 2, oy = (box.height - 540 * scale) / 2;
  const faceAt = (x1: number) => ({ x: ox + (x1 + 80) * scale, y: oy + 240 * scale });
  const pixelsDrawn = () => canvas.evaluate((element: HTMLCanvasElement) => { const data = element.getContext('2d')!.getImageData(0, 0, element.width, element.height).data; let n = 0; for (let i = 3; i < data.length; i += 4) if (data[i]) n++; return n; });
  const before = await pixelsDrawn();
  await page.getByRole('button', { name: /^\s*⌗\s*Mesh/ }).click();
  await page.getByRole('button', { name: /Gaze/ }).click();
  await expect.poll(pixelsDrawn).toBeGreaterThan(before);
  await canvas.click({ button: 'right', position: faceAt(620) });
  const menu = page.getByRole('menu');
  await expect(menu).toBeVisible();
  await expect(menu.getByRole('menuitemcheckbox')).toHaveText([/UltraMax enhancer/, /Lock 6DoF pose filter/, /Solo this face mask/]);
  await menu.getByRole('menuitemcheckbox', { name: /Lock 6DoF pose filter/ }).click();
  await expect(menu).toHaveCount(0);
  await page.screenshot({ path: 'test-results/w3d-canvas.png' });
  expect(errors).toEqual([]);
});
