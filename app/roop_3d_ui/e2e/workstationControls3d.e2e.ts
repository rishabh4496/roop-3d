import { createRequire } from 'node:module';
import { expect, test, type Locator, type Page, type Route } from '@playwright/test';
import { MEDIA_INFO, mockFilmstrip, mockFrame } from './helpers/mockApi';

/**
 * Stage 8 in a real browser (system Chrome, real WebGL2 through SwiftShader, mocked API): the viewport's buffer switcher renders five
 * different buffers from one mesh, the SH-relit view follows the light direction, the gizmo dock and the draggable handles write the real
 * expression / light parameters, and the new control panels reach the live preview request only once touched.
 */
const ID = 'a'.repeat(32), TARGET = 'roop_3d/.runtime/media/t.mp4';
const FEATURES = Object.fromEntries(['projects', 'face_inspection', 'queue', 'frame_preview', 'video_playback', 'uploads', 'trim'].map(name => [name, { enabled: true, reason: null }]));
const CAPABILITIES = { models: [], job_parameter_bounds: {}, legal_parameter_bounds: {}, inspection_parameter_bounds: {}, enabled_modes: [], rendering_worker_available: true,
  workstation: { features: FEATURES, controls: {}, read_only_parameters: {}, disabled_parameters: {}, job_states: [], job_controls: {}, upload_chunk_bytes: 1048576, max_asset_bytes: 1048576 } };
const gif = 'R0lGODlhAQABAAAAACw=';
const tinyJpeg = Buffer.from('/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA=', 'base64');
const track = (n: number, x1: number) => ({ track_id: `track_000${n}`, persistent_id: n, scene_id: 0, landmark_count: 106, first_frame: 0, last_frame: 99, first_timestamp_seconds: 0, last_timestamp_seconds: 3.96,
  bounding_box: { x1, y1: 120, x2: x1 + 160, y2: 360 }, pose_range: { yaw_degrees: [-5, 5], pitch_degrees: [0, 1], roll_degrees: [0, 1], method: 'five_point_geometric_estimate' }, thumbnail_base64: gif, thumbnail_mime: 'image/gif', sampled_detections: 4 });
const INSPECTION = { inspection_id: ID, target_sha256: 'b'.repeat(64), tracks: [track(0, 140), track(1, 620)], truncated: false, tracking_method: 'identity_lock_v1', warnings: [], scene_cut_frames: [60] };
const sheetTrack = (n: number) => ({ track_id: `track_000${n}`, persistent_id: n, first_frame: 0, last_frame: 99, segments: [[0, 99, 0, 3.96]], missing_ranges: [], visible_frames: 100, swappable_frames: 100, withheld_frames: 0, withheld_reasons: {}, state_frames: { TRACKED: 100 }, confidence_mean: 0.93, confidence_min: 0.71, scene_ids: [0], thumbnails: [] });
const SHEET = { inspection_id: ID, target_sha256: 'b'.repeat(64), truncated: false, scene_cut_frames: [60], warnings: [], tracks: [sheetTrack(0), sheetTrack(1)] };
const project = { project_id: 'p1', name: 'Demo', saved_at: '2026-11-02T00:00:00Z', last_completed_frame: -1, checkpoint_job_id: null, effective_settings: {}, inputs_verified: true,
  request: { target_path: TARGET, inspection_id: ID, face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }], processing: 'local', remote_consent: false, trim_start_seconds: null, trim_end_seconds: null } };

/** A UV-sphere head (ellipsoid), backend mesh frame (+y down). Grey texture-free, so every buffer is a function of geometry and light only. */
function headMesh() {
  const rows = 28, cols = 40, vertices: number[][] = [], indices: number[][] = [];
  for (let i = 0; i <= rows; i++) for (let j = 0; j <= cols; j++) {
    const phi = (i / rows) * Math.PI, theta = (j / cols) * 2 * Math.PI;
    vertices.push([0.8 * Math.sin(phi) * Math.sin(theta), -Math.cos(phi), 0.9 * Math.sin(phi) * Math.cos(theta)]);
  }
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const a = i * (cols + 1) + j, b = a + cols + 1;
    indices.push([a, b, a + 1], [a + 1, b, b + 1]);
  }
  return { vertices, indices, uvs: [], normals: [], texture_base64: '', reconstruction_level: 'three_tier_facial_surface', limitations: [] };
}
const COVERAGE = { faceset_id: 'alice', views: [{ yaw: 0, pitch: 0, roll: 0, pose_bin: 'front' }], bins: { front: 1 }, observed_views: 1, sphere_coverage_count: 1, sphere_coverage_total: 96, sphere_coverage_pct: 1, mesh_full_3d: headMesh() };
const TOPOLOGIES = { default: 'makehuman_cc0',
  topologies: [{ id: 'makehuman_cc0', title: 'MakeHuman', license_id: 'CC0-1.0', commercial_use: true, default: true, available: true, state: 'ready', detail: null },
    { id: 'flame2023_open', title: 'FLAME 2023 Open', license_id: 'CC-BY-4.0', commercial_use: true, default: false, available: true, state: 'ready', detail: null }],
  semantic_sliders: ['jaw_width', 'cheekbone_prominence', 'chin_projection', 'nose_bridge'].map(key => ({ key, label: key, description: '', unit_mm: 2, minimum: -3, maximum: 3, default: 0 })) };

const require = createRequire(import.meta.url);
const AXE = require.resolve('axe-core/axe.min.js');
test.use({ viewport: { width: 1680, height: 1050 } });

/** WCAG 2.2 AA violations (the same rule set as the Stage 19 gate) inside the elements matching `selector`. */
async function axeViolations(page: Page, selector: string) {
  await page.addScriptTag({ path: AXE });
  return page.evaluate(async target => {
    const result = await (window as any).axe.run(document.querySelector(target), { runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'] } });
    return result.violations.map((item: any) => ({ id: item.id, impact: item.impact, nodes: item.nodes.slice(0, 3).map((node: any) => node.target.join(' ')) }));
  }, selector);
}

async function boot(page: Page) {
  const seen = { previews: [] as any[], meshQueries: [] as string[], configPuts: [] as any[], external: [] as string[] };
  await page.addInitScript(() => { sessionStorage.setItem('roop3d_skip_linter', 'true'); localStorage.setItem('roop3d_tour_seen', 'true'); localStorage.setItem('roop3d-onboarding-v1', '1'); });
  page.on('request', request => { const url = new URL(request.url()); // The workstation must talk to its own origin only. Antivirus web-protection (Kaspersky) injects its own script into every page of this machine's
    // Chrome and calls home from it; those requests are not the app's, so that one injected host is ignored (the static dist scan covers the bundle itself).
    if (!['127.0.0.1', 'localhost'].includes(url.hostname) && !['data:', 'blob:'].includes(url.protocol) && !url.hostname.endsWith('.kaspersky-labs.com')) seen.external.push(request.url()); });
  await page.route('**/api/v1/**', async (route: Route) => {
    const url = new URL(route.request().url()), path = url.pathname.replace('/api/v1', ''), method = route.request().method();
    const json = (body: unknown, code = 200) => route.fulfill({ status: code, json: body });
    if (path === '/system/capabilities') return json(CAPABILITIES);
    if (path === '/config' && method === 'PUT') { seen.configPuts.push(route.request().postDataJSON()); return json(route.request().postDataJSON()); }
    if (path === '/config') return json({ temporal_min_cutoff: 0.15, temporal_beta: 0.04 });
    if (path === '/facesets') return json([{ faceset_id: 'alice', name: 'Alice', item_count: 1 }]);
    if (path === '/projects' && method === 'PUT') return json(project);
    if (path === '/projects') return json([project]);
    if (path === '/projects/p1') return json(project);
    if (path === '/jobs') return json([]);
    if (path === '/models/hub') return json({ models: [], model_states: [], execution_policy: '', download_policy: '', engines: [] });
    if (/^\/facesets\/[^/]+\/(views\/\d+\/image|thumb)$/.test(path)) return route.fulfill({ status: 200, contentType: 'image/jpeg', body: tinyJpeg });
    if (/^\/facesets\/[^/]+\/views$/.test(path)) return json(COVERAGE.views.map((view, index) => ({ index, ...view })));
    if (/^\/facesets\/[^/]+\/coverage$/.test(path) || path === '/faceset/reconstruct-3d' || path === '/faceset/3d-state') return json(COVERAGE);
    if (path === '/morphable-topologies') return json(TOPOLOGIES);
    if (/^\/facesets\/[^/]+\/identity-asset\/mesh$/.test(path)) { seen.meshQueries.push(url.search); return json({ method: 'identity_asset', coordinate_system: 'mesh', vertices: COVERAGE.mesh_full_3d.vertices.map(v => [v[0] * 1.2, v[1], v[2]]), faces: COVERAGE.mesh_full_3d.indices, uvs: null, texture: null, provenance: 'makehuman · semantic sliders applied' }); }
    if (/^\/facesets\/[^/]+\/identity-asset/.test(path)) return json({ error: { message: 'not built' } }, 404);
    if (path === '/reconstruction/backbones') return json({ default: 'x', active: 'x', backbones: [] });
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
    seen.previews.push(route.request().postDataJSON());
    return route.fulfill({ status: 200, contentType: 'image/jpeg', body: tinyJpeg, headers: { 'X-Preview-Meta': JSON.stringify({ media_id: ID, frame_index: 12, swap_config_hash: String(seen.previews.length), solo_node: 'final', width: 1, height: 1, faces: [], applied: [], unsupported: [], notes: [], latency: {} }) } });
  });
  await page.routeWebSocket('**/ws/**', () => undefined);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/#/faceswap');
  await page.getByRole('button', { name: 'Projects' }).click();
  await page.getByRole('button', { name: 'Verify & open' }).click();
  await expect(page.locator('[data-track-id="track_0001"]')).toBeVisible();
  await page.getByRole('button', { name: 'Select FaceSet Alice' }).click();
  await page.locator('[data-track-id="track_0001"]').first().click();
  await expect(page.locator('.pose-canvas-wrapper canvas').first()).toBeVisible();
  return { seen, errors };
}

/**
 * Pixel statistics of the head in the viewport canvas, from a real screenshot. The head sits in the middle of the canvas, so the statistics
 * are taken in a box covering the middle half of its width and height (inside the ellipse, away from rings and gizmo handles), plus three samples
 * and the number of bright pixels on the middle row (the head's width in an unlit, bright buffer).
 */
async function headStats(page: Page, canvas: Locator) {
  const png = (await canvas.screenshot()).toString('base64');
  return page.evaluate(async b64 => {
    const image = new Image(); image.src = `data:image/png;base64,${b64}`; await image.decode();
    const c = document.createElement('canvas'); c.width = image.width; c.height = image.height;
    const ctx = c.getContext('2d')!; ctx.drawImage(image, 0, 0);
    const { data } = ctx.getImageData(0, 0, c.width, c.height);
    const w = c.width, h = c.height, cx = w / 2, cy = h / 2, rx = 0.165 * w, ry = 0.335 * h;
    const px = (x: number, y: number) => { const i = (Math.round(y) * w + Math.round(x)) * 4; return [data[i], data[i + 1], data[i + 2]]; };
    const lum = (p: number[]) => 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    let n = 0, grey = 0, spread = 0, mean = 0, sq = 0, left = 0, nLeft = 0, right = 0, nRight = 0;
    for (let y = Math.round(cy - 0.5 * ry); y <= cy + 0.5 * ry; y++) for (let x = Math.round(cx - 0.5 * rx); x <= cx + 0.5 * rx; x++) {
      const p = px(x, y), l = lum(p), sp = Math.max(...p) - Math.min(...p);
      n++; mean += l; sq += l * l; spread += sp; if (sp < 7) grey++;
      if (x < cx) { left += l; nLeft++; } else { right += l; nRight++; }
    }
    mean /= n;
    const off = cy + 0.25 * ry;   // the orbital rings draw a cross through the exact centre: sample beside it
    let row = 0; for (let x = 0; x < w; x++) if (lum(px(x, off)) > 100) row++;
    return { n, grey: grey / n, spread: spread / n, luminance: mean, std: Math.sqrt(Math.max(0, sq / n - mean * mean)), left: left / nLeft, right: right / nRight,
      centre: lum(px(cx + 0.1 * rx, off)), edge: lum(px(cx + 0.85 * rx, off)), row, canvas: [w, h] };
  }, png);
}

test('the buffer switcher renders wireframe, albedo, normals, depth and SH-relit views from the same mesh', async ({ page }) => {
  const { errors, seen } = await boot(page);
  const switcher = page.getByTestId('viewport-buffers').first();
  await expect(switcher.getByRole('radio')).toHaveText(['Shaded', 'Wireframe Mesh', 'Albedo', 'Surface Normals', 'Z-Depth', 'SH Relit']);
  await expect(switcher.getByRole('radio', { name: 'Shaded' })).toHaveAttribute('aria-checked', 'true');
  const canvas = page.locator('.pose-canvas-wrapper canvas').first();
  const stats: Record<string, Awaited<ReturnType<typeof headStats>>> = {};
  for (const name of ['Shaded', 'Wireframe Mesh', 'Albedo', 'Surface Normals', 'Z-Depth', 'SH Relit']) {
    await switcher.getByRole('radio', { name, exact: true }).click();
    await expect(switcher.getByRole('radio', { name, exact: true })).toHaveAttribute('aria-checked', 'true');
    await page.waitForTimeout(450);                       // let the frame render
    stats[name] = await headStats(page, canvas);
    await canvas.screenshot({ path: `test-results/w3d-stage8-buffer-${name.toLowerCase().replace(/[^a-z]+/g, '-')}.png` });
    console.log(name, JSON.stringify(stats[name]));
    expect(stats[name].n, `${name} samples the head`).toBeGreaterThan(400);
  }
  // Normals are colour (x, y, z -> RGB); depth is a grey ramp with real variation; albedo is flat; wireframe is mostly dark with bright edges.
  expect(stats['Surface Normals'].spread).toBeGreaterThan(40);
  expect(stats['Albedo'].spread).toBeLessThan(stats['Surface Normals'].spread);
  expect(stats['Z-Depth'].spread).toBeLessThan(8);                                  // a grey ramp, not colour
  expect(stats['Z-Depth'].centre - stats['Z-Depth'].edge).toBeGreaterThan(12);   // near = bright: the nose-side centre beats the limb
  expect(stats['Albedo'].std).toBeLessThan(stats['Shaded'].std * 0.6);       // unlit: flatter than the shaded head
  expect(stats['Wireframe Mesh'].luminance).toBeLessThan(stats['Albedo'].luminance * 0.7);
  expect(stats['SH Relit'].right).toBeGreaterThan(stats['SH Relit'].left * 1.05);   // the default key light is from the image right
  await page.screenshot({ path: 'test-results/w3d-stage8-buffers.png' });
  expect(errors).toEqual([]);
  expect(seen.external).toEqual([]);
});

test('the SH-relit buffer follows the light direction set from the gizmo dock, and only a manual light reaches the preview once relighting is on', async ({ page }) => {
  const { seen, errors } = await boot(page);
  const dock = page.getByTestId('viewport-gizmos').first();
  await page.getByTestId('viewport-buffers').first().getByRole('radio', { name: 'SH Relit' }).click();
  await dock.getByRole('button', { name: 'Light direction' }).click();
  const canvas = page.locator('.pose-canvas-wrapper canvas').first();
  await dock.getByRole('slider', { name: 'Light yaw' }).fill('85');
  await dock.getByRole('slider', { name: 'Light pitch' }).fill('0');
  await page.waitForTimeout(450);
  const fromRight = await headStats(page, canvas);
  await dock.getByRole('slider', { name: 'Light yaw' }).fill('-85');
  await page.waitForTimeout(450);
  const fromLeft = await headStats(page, canvas);
  expect(fromRight.right).toBeGreaterThan(fromRight.left * 1.15);
  expect(fromLeft.left).toBeGreaterThan(fromLeft.right * 1.15);
  // Drag the 3D light handle itself (a point on a shell of radius 1.9 around the head) from the right of the face to the left.
  await dock.getByRole('slider', { name: 'Light yaw' }).fill('40');
  await dock.getByRole('slider', { name: 'Light pitch' }).fill('25');
  await canvas.scrollIntoViewIfNeeded();
  const box = (await canvas.boundingBox())!;
  const perPixel = (box.height / 2) / Math.tan((42 / 2) * Math.PI / 180);               // pixels per world unit at distance 1
  const yaw = 40 * Math.PI / 180, pitch = 25 * Math.PI / 180;
  const world = [Math.sin(yaw) * Math.cos(pitch) * 1.9, Math.sin(pitch) * 1.9, Math.cos(yaw) * Math.cos(pitch) * 1.9];
  const depth = 3.8 - world[2];
  const handle = { x: box.x + box.width / 2 + (world[0] / depth) * perPixel, y: box.y + box.height / 2 - (world[1] / depth) * perPixel };
  await page.mouse.move(handle.x, handle.y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2 - 40, { steps: 6 });
  await page.mouse.move(box.x + box.width / 2 - 80, box.y + box.height / 2 - 20, { steps: 8 });
  await page.mouse.up();
  await expect.poll(async () => Number(await dock.getByRole('slider', { name: 'Light yaw' }).inputValue())).toBeLessThan(-10);
  const dragged = Number(await dock.getByRole('slider', { name: 'Light yaw' }).inputValue());
  await expect(dock).toContainText('3D relighting is off');
  expect('lighting_mode' in (seen.previews.at(-1) ?? {})).toBe(false);          // relighting off: the manual light is not sent
  await dock.getByRole('button', { name: 'Turn on 3D relighting' }).click();
  await expect.poll(() => seen.previews.at(-1)?.manual_light_yaw).toBe(dragged);
  expect(seen.previews.at(-1)?.lighting_mode).toBe('manual');
  await page.screenshot({ path: 'test-results/w3d-stage8-light.png' });
  expect(errors).toEqual([]);
});

test('the jaw and gaze gizmos write the selected face\'s expression overrides, by slider and by dragging the 3D handle', async ({ page }) => {
  const { seen, errors } = await boot(page);
  const dock = page.getByTestId('viewport-gizmos').first();
  await dock.getByRole('button', { name: 'Jaw opening' }).click();
  await dock.getByRole('button', { name: 'Eye gaze target' }).click();
  await dock.getByRole('slider', { name: 'Jaw opening angle' }).fill('14');
  await expect.poll(() => seen.previews.at(-1)?.face_expression_overrides?.track_0001?.mouth_open).toBeCloseTo(0.5, 2);
  await dock.getByRole('slider', { name: 'Gaze horizontal' }).fill('0.4');
  await dock.getByRole('slider', { name: 'Gaze vertical' }).fill('-0.3');
  await expect.poll(() => seen.previews.at(-1)?.face_expression_overrides?.track_0001).toMatchObject({ mouth_open: 0.5, gaze_x: 0.4, gaze_y: -0.3 });
  await dock.getByRole('slider', { name: 'Jaw opening angle' }).fill('0');
  await expect.poll(() => 'mouth_open' in (seen.previews.at(-1)?.face_expression_overrides?.track_0001 ?? {})).toBe(false);

  // Drag the chin handle straight down in the viewport: the opening angle must grow from 0 and reach the request.
  const canvas = page.locator('.pose-canvas-wrapper canvas').first();
  await canvas.scrollIntoViewIfNeeded();                                             // the inspector scrolls: use on-screen coordinates
  const box = (await canvas.boundingBox())!;
  const unit = (box.height / 2) / (Math.tan((42 / 2) * Math.PI / 180) * 3.8);          // pixels per world unit at the head's depth
  const chin = { x: box.x + box.width / 2, y: box.y + box.height / 2 + 0.95 * unit };
  await page.mouse.move(chin.x, chin.y);
  await page.mouse.down();
  await page.mouse.move(chin.x, chin.y + 0.15 * unit, { steps: 6 });
  await page.mouse.move(chin.x, chin.y + 0.3 * unit, { steps: 6 });
  await page.mouse.up();
  await expect.poll(() => seen.previews.at(-1)?.face_expression_overrides?.track_0001?.mouth_open ?? 0).toBeGreaterThan(0.05);
  await page.screenshot({ path: 'test-results/w3d-stage8-gizmos.png' });
  expect(errors).toEqual([]);
});

test('the new panels reach the live preview only once touched, and the preference-only toggles never do', async ({ page }) => {
  const { seen, errors } = await boot(page);
  await expect.poll(() => seen.previews.length).toBeGreaterThan(0);
  for (const key of ['face_expression_overrides', 'swap_scope', 'skull_blend', 'lighting_mode']) expect(key in seen.previews[0]).toBe(false);
  const controls = page.getByTestId('workstation-controls-3d');
  await controls.scrollIntoViewIfNeeded();
  await controls.getByTestId('tone-section').locator(':scope > summary').click();
  await controls.getByRole('slider', { name: 'Skin tone preservation versus ambient adaptation' }).fill('25');
  await expect.poll(() => seen.previews.at(-1)?.color_strength).toBeCloseTo(0.25);
  await controls.getByTestId('expression-section').locator(':scope > summary').click();
  await controls.getByRole('slider', { name: 'Expression offset smile' }).fill('0.5');
  await expect.poll(() => seen.previews.at(-1)?.face_expression_overrides?.track_0001?.mouth_smile).toBeCloseTo(0.5);
  await controls.getByTestId('adaptation-section').locator(':scope > summary').click();
  await expect(controls.getByText('saved preference only').filter({ visible: true })).toHaveCount(2);   // both unwired toggles say so
  await controls.getByRole('checkbox', { name: /Speech-lip lock/ }).check();
  await controls.getByRole('checkbox', { name: /3-second scene adaptation/ }).check();
  await page.waitForTimeout(300);
  expect(JSON.stringify(seen.previews.at(-1))).not.toMatch(/lip_lock|speech|scene_adapt|spark|spectre/i);
  await controls.getByRole('slider', { name: 'Temporal Kalman smoothing' }).fill('85');
  // (opening the saved project also PUTs its own settings; only the smoothing change carries the cutoff)
  await expect.poll(() => seen.configPuts.find(body => 'temporal_min_cutoff' in body)?.temporal_min_cutoff, { timeout: 5000 }).toBeLessThan(0.15);
  expect(seen.configPuts.filter(body => 'temporal_min_cutoff' in body)).toHaveLength(1);                // one save after the slider stops, not one per tick
  await page.screenshot({ path: 'test-results/w3d-stage8-panels.png' });
  expect(errors).toEqual([]);
  expect(seen.external).toEqual([]);
});

test('the topology engine and semantic sliders rebuild the identity asset mesh and show it in the viewport', async ({ page }) => {
  const { seen, errors } = await boot(page);
  const topology = page.getByTestId('topology-section');
  await topology.scrollIntoViewIfNeeded();
  await topology.locator(':scope > summary').click();
  await expect(topology.getByRole('radio', { name: /MakeHuman CC0/ })).toHaveAttribute('aria-checked', 'true');
  await expect(topology.getByRole('radio', { name: /FLAME 2023 Open/ })).toBeEnabled();
  expect(seen.meshQueries).toEqual([]);                                             // nothing is fetched until the user changes something
  const canvas = page.locator('.pose-canvas-wrapper canvas').first();
  await page.getByTestId('viewport-buffers').first().getByRole('radio', { name: 'Albedo' }).click();
  await page.waitForTimeout(450);
  const before = await headStats(page, canvas);
  await topology.getByRole('slider', { name: 'Semantic Jawline Width' }).fill('2');
  await expect.poll(() => seen.meshQueries.at(-1)).toBe('?template=makehuman&topology=makehuman_cc0&jaw_width=2');
  await expect(topology.getByTestId('topology-provenance')).toContainText('semantic sliders applied');
  await page.waitForTimeout(450);
  const after = await headStats(page, canvas);
  expect(after.row).toBeGreaterThan(before.row * 1.08);                                // the served mesh is 20% wider: the head covers more pixels
  await topology.getByRole('radio', { name: /FLAME 2023 Open/ }).click();
  await expect.poll(() => seen.meshQueries.at(-1)).toBe('?template=makehuman&topology=flame2023_open&jaw_width=2');
  await page.screenshot({ path: 'test-results/w3d-stage8-topology.png' });
  expect(errors).toEqual([]);
});

test('the buffer switcher, gizmo dock and Stage 8 panels have no WCAG 2.2 AA violations, with every panel and gizmo open', async ({ page }) => {
  const { errors } = await boot(page);
  const dock = page.getByTestId('viewport-gizmos').first();
  for (const name of ['Jaw opening', 'Eye gaze target', 'Light direction']) await dock.getByRole('button', { name }).click();
  await page.getByTestId('viewport-buffers').first().getByRole('radio', { name: 'Z-Depth' }).click();   // shows the buffer note and depth legend
  const controls = page.getByTestId('workstation-controls-3d');
  for (const id of ['topology-section', 'tone-section', 'expression-section', 'adaptation-section']) await controls.getByTestId(id).locator(':scope > summary').click();
  await page.waitForTimeout(300);
  expect(await axeViolations(page, '.pose-geometry-container')).toEqual([]);
  expect(await axeViolations(page, '[data-testid=workstation-controls-3d]')).toEqual([]);
  expect(errors).toEqual([]);
});
