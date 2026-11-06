import { expect, type Page, type Route } from '@playwright/test';

/**
 * Mocked local service for the Stage 19 browser specs. Same approach as the earlier specs (REST mocked per test with
 * `page.route`, WebSockets with `page.routeWebSocket`, everything else real: React, WebGL2, drag and drop, focus), but
 * shared, and it covers the surfaces Stage 19 adds: hardware profile, export presets, FaceSet coverage requests,
 * jobs (with failure codes) and structured API errors.
 */
export const ID = 'a'.repeat(32);
export const TARGET = 'roop_3d/.runtime/media/t.mp4';
const FEATURES = Object.fromEntries(['projects', 'face_inspection', 'queue', 'frame_preview', 'video_playback', 'uploads', 'trim', 'pose_inspection', 'reconstruction_viewer'].map(name => [name, { enabled: true, reason: null }]));
export const CAPABILITIES = {
  models: [{ model_id: 'm1', name: 'gpen_bfr_512.onnx', sha256: '0'.repeat(64), size_bytes: 1, state: 'VERIFIED_LOCAL', family_hint: 'restorer', verified_at: '2026-11-05T00:00:00Z' }],
  job_parameter_bounds: {}, legal_parameter_bounds: {}, inspection_parameter_bounds: {}, enabled_modes: [], rendering_worker_available: true,
  workstation: { features: FEATURES, controls: {}, read_only_parameters: {}, disabled_parameters: {}, job_states: [], job_controls: {}, upload_chunk_bytes: 1048576, max_asset_bytes: 1048576,
    active_hardware_profile: 'auto', resolved_hardware_profile: '12gb' },
};
const gif = 'R0lGODlhAQABAAAAACw=';
const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==';
const tinyJpeg = Buffer.from('/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA=', 'base64');
const track = (n: number, x1: number) => ({ track_id: `track_000${n}`, persistent_id: n, scene_id: 0, landmark_count: 106, first_frame: 0, last_frame: 99, first_timestamp_seconds: 0, last_timestamp_seconds: 3.96,
  bounding_box: { x1, y1: 120, x2: x1 + 160, y2: 360 }, pose_range: { yaw_degrees: [-5, 5], pitch_degrees: [0, 1], roll_degrees: [0, 1], method: 'five_point_geometric_estimate' }, thumbnail_base64: gif, thumbnail_mime: 'image/gif', sampled_detections: 10 });
export const INSPECTION = { inspection_id: ID, target_sha256: 'b'.repeat(64), tracks: [track(0, 140), track(1, 620)], truncated: false, tracking_method: 'identity_lock_v1', warnings: [], scene_cut_frames: [60] };
const sheetTrack = (n: number) => ({ track_id: `track_000${n}`, persistent_id: n, first_frame: 0, last_frame: 99, segments: [[0, 99, 0, 3.96]], missing_ranges: [], visible_frames: 100, swappable_frames: 100, withheld_frames: 0,
  withheld_reasons: {}, state_frames: { TRACKED: 100 }, confidence_mean: 0.93, confidence_min: 0.71, scene_ids: [0], thumbnails: [] });
const SHEET = { inspection_id: ID, target_sha256: 'b'.repeat(64), truncated: false, scene_cut_frames: [60], warnings: [], tracks: [sheetTrack(0), sheetTrack(1)] };
const project = { preview: { timestamp: 0, selected_track: 'track_0000', source_faceset_id: 'alice', viewport: { live_swap: true, view_mode: 'side', compare_mode: 'wipe', show_landmarks: true, show_pose: true, show_overlays: true, grid_mode: 'off', zoom: 1 } },
  project_id: 'p1', name: 'Demo', saved_at: '2026-11-02T00:00:00Z', last_completed_frame: -1, checkpoint_job_id: null, effective_settings: {}, inputs_verified: true,
  request: { target_path: TARGET, inspection_id: ID, face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }], processing: 'local', remote_consent: false, trim_start_seconds: null, trim_end_seconds: null } };
// Three views: frontal, three-quarter left, three-quarter right-down. Stored pitch is the legacy frame (+ = down).
const VIEWS = [{ index: 0, yaw: 0, pitch: 0, roll: 0 }, { index: 1, yaw: -45, pitch: 0, roll: 0 }, { index: 2, yaw: 44, pitch: -15, roll: 0 }];
const status = (id: string) => ({ faceset_id: id, name: id, state: 'not_built', build_key: 'k'.repeat(64), note: '', flame: {},
  coverage: { readiness: 'partial_coverage', view_count: 3, usable_count: 3, yaw_range_deg: [-45, 44], yaw_span_deg: 89, pitch_span_deg: 15, yaw_coverage: {}, pitch_coverage: {}, missing: [],
    requests: ['[recommended] Add a side profile with the nose pointing toward the image LEFT (about -75 deg yaw) to capture nose, jaw and ear depth.', '[optional] Add a portrait looking down (about -20 deg pitch) for vertical coverage.'],
    duplicate_groups: [], capabilities: {} }, asset: null });
export const PROFILES = [
  ['standard_mp4', 'Standard MP4', '.mp4', 8, false], ['high_quality_mp4', 'High-quality MP4', '.mp4', 8, false], ['archival_hevc10', 'High-bitrate archival (HEVC 10-bit)', '.mkv', 10, false],
  ['mov_workflow', 'MOV workflow (ProRes 422 HQ)', '.mov', 10, false], ['lossless_intermediate', 'Lossless intermediate (FFV1)', '.mkv', 8, true], ['image_sequence', 'Image sequence', 'sequence', 8, true],
  ['color_preserving', 'Professional colour-preserving', '.mov', 10, false],
].map(([id, label, container, depth, lossless]) => ({ id, label, description: `${label} description`, container, encoder: 'x', codec_family: 'h264', pix_fmt: null, container_bit_depth: depth, lossless, rgb_native: false,
  hdr_passthrough: false, audio_default: 'copy', limits: [], verification: { state: 'UNKNOWN' } }));

/** Two faces of the mocked 960x540 frame, in native pixels: what GET /media/frame?detect_faces=true returns. */
export const FRAME_FACES = [
  { face_index: 0, track_id: null, bbox: [140, 120, 300, 340], bbox_norm: [140 / 960, 120 / 540, 300 / 960, 340 / 540], kps_5: [[185, 190], [255, 190], [220, 235], [190, 290], [250, 290]],
    pose: { yaw: -12, pitch: 3, roll: 0 }, pose3d: { yaw: -12, pitch: -3, roll: 0 }, pose_cell: 'center', confidence: 0.97 },
  { face_index: 1, track_id: null, bbox: [620, 120, 780, 340], bbox_norm: [620 / 960, 120 / 540, 780 / 960, 340 / 540], kps_5: [[665, 190], [735, 190], [700, 235], [670, 290], [730, 290]],
    pose: { yaw: 25, pitch: 0, roll: 2 }, pose3d: { yaw: 25, pitch: 0, roll: 2 }, pose_cell: 'right', confidence: 0.91 },
];
/** GET /media/frame: a JPEG whose X-Frame-Meta header carries the metadata (and the faces when asked), as the real service answers. */
export function mockFrame(url: URL) {
  const index = Number(url.searchParams.get('frame_index') ?? 0);
  const meta = { frame_index: index, requested_index: index, timestamp_sec: index / 25, width: 960, height: 540, image_width: 960, image_height: 540, total_frames: 100, fps: 25, cache: 'miss', decode_ms: 1,
    ...(url.searchParams.get('detect_faces') === 'true' ? { faces: FRAME_FACES, unsupported: ['embedding_norm', 'track_id'] } : {}) };
  return { status: 200, contentType: 'image/jpeg', body: tinyJpeg, headers: { 'X-Frame-Meta': JSON.stringify(meta) } };
}
export const MEDIA_INFO = { width: 960, height: 540, fps: 25, fps_rational: '25/1', total_frames: 100, duration_sec: 4, rotation_degrees: 0, frame_count_declared: true };
export function mockFilmstrip(url: URL) {
  const count = Number(url.searchParams.get('count') ?? 16);
  return { count, thumb_width: 160, thumb_height: 90, total_frames: 100, duration_sec: 4,
    frames: Array.from({ length: count }, (_, slot) => ({ slot, frame_index: Math.floor((slot + 0.5) / count * 100), requested_frame_index: Math.floor((slot + 0.5) / count * 100), timestamp_sec: ((slot + 0.5) / count) * 4, data_url: `data:image/gif;base64,${gif}` })) };
}

export interface BootOptions {
  jobs?: unknown[];
  /** `POST /media/inspect-faces` answers with this API error instead of the inspection. */
  inspectError?: { status: number; code: string; message: string };
  /** Every API call fails at the network level (the service is stopped). */
  offline?: boolean;
  /** Start on this hash route (default #/faceswap). */
  route?: string;
  /** Do not mark the first-run welcome as seen. */
  firstRun?: boolean;
  width?: number; height?: number;
  openProject?: boolean;
  engines?: unknown[];
  /** Delay `/system/capabilities` (ms) to exercise the still-connecting state. */
  capsDelayMs?: number;
}

export async function boot(page: Page, options: BootOptions = {}) {
  const seen = { previews: [] as any[], previewFrames: [] as string[], frames: [] as string[], filmstrips: [] as string[], projects: [] as any[], queued: [] as any[] };
  await page.addInitScript((firstRun: boolean) => {
    sessionStorage.setItem('roop3d_skip_linter', 'true');
    if (!firstRun) localStorage.setItem('roop3d-onboarding-v1', '1');
  }, !!options.firstRun);
  await page.route('**/api/v1/**', async (route: Route) => {
    if (options.offline) return route.abort('connectionrefused');
    const url = new URL(route.request().url()), path = url.pathname.replace('/api/v1', ''), method = route.request().method();
    const json = (body: unknown, code = 200) => route.fulfill({ status: code, json: body });
    if (path === '/system/capabilities') { if (options.capsDelayMs) await new Promise(done => setTimeout(done, options.capsDelayMs)); return json(CAPABILITIES); }
    if (path === '/config') return json({});
    if (path === '/facesets') return json([{ faceset_id: 'alice', name: 'Alice', item_count: 3 }, { faceset_id: 'bob', name: 'Bob', item_count: 3 }]);
    if (path === '/projects' && method === 'POST') { seen.projects.push(route.request().postDataJSON()); return json({ ...project, request: seen.projects.at(-1).request }, 201); }
    if (path === '/projects/p1' && method === 'PUT') { seen.projects.push(route.request().postDataJSON()); return json({ ...project, request: seen.projects.at(-1).request }); }
    if (path === '/projects/p1/jobs' && method === 'POST') { seen.queued.push(route.request().postDataJSON()); return json({ job_id: 'f'.repeat(32), state: 'queued', request: project.request, progress: { frame: 0, total_frames: 100, fps: 0, stage: 'queued' }, output: null, failure: null }, 201); }
    if (path === '/projects') return json([project]);
    if (path === '/projects/p1') return json(project);
    if (path === '/jobs') return json(options.jobs ?? []);
    if (path === '/media/export-profiles') return json({ profiles: PROFILES, default: 'standard_mp4', processing_bit_depth: 8 });
    if (path === '/models/hub') return json({ models: [], model_states: [], execution_policy: '', download_policy: '', engines: options.engines ?? [
      { engine_id: 'reference_onnx', name: 'Roop 3D Reference ONNX', status: 'LICENSE_RESTRICTED', selectable: true, actual_inference_available: true, evidence: 'Existing production path is preserved.', purpose: 'Video face swap.' },
      { engine_id: 'experimental_x', name: 'Experimental X', status: 'NOT_INSTALLED', selectable: false, actual_inference_available: false, evidence: 'weights absent' }] });
    if (path === '/models/downloads') return json([]);
    if (path === '/system/preflight-lint') return json({ status: 'PASS', summary: 'All checks passed', checks: [{ id: 'vram', name: 'VRAM', status: 'PASS', message: 'Enough video memory.', details: [] }] });
    if (path === '/installer/preflight') return json({ status: 'PASS', ready: true, resolved_hardware_profile: '12gb', checks: [] });
    if (path === '/installer/offline-health') return json({ offline_ready: false, status: 'WARN', message: 'Models needed', core_models: [] });
    if (/^\/facesets\/[^/]+\/views\/\d+\/image$/.test(path) || /^\/facesets\/[^/]+\/thumb$/.test(path)) return route.fulfill({ status: 200, contentType: 'image/jpeg', body: tinyJpeg });
    if (/^\/facesets\/[^/]+\/views$/.test(path)) return json(VIEWS);
    if (/^\/facesets\/[^/]+\/coverage$/.test(path)) return json({ faceset_id: path.split('/')[2], views: [], bins: {} });
    if (/^\/facesets\/[^/]+\/identity-asset$/.test(path)) return json({ ...status(path.split('/')[2]) });
    if (path.startsWith('/faces/bank')) return json({ version: 1, revision: 1, target_sha256: 'b'.repeat(64), next_rank: 0, clusters: [], aliases: {}, unresolved_tracks: [], inspection_id: ID });
    if (path === '/media/probe') return json({ width: 960, height: 540, duration_seconds: 4, fps_rational: '25/1', declared_frame_count: 100, audio_tracks: [] });
    if (path === '/media/keyframes') return json({ timestamps: [0, 2] });
    if (path === '/media/preview-frame') { seen.previewFrames.push(url.pathname + url.search); return route.fulfill({ status: 200, contentType: 'image/jpeg', body: tinyJpeg }); }
    if (path === '/media/frame') { seen.frames.push(url.search); return route.fulfill(mockFrame(url)); }
    if (path === '/media/info') return json(MEDIA_INFO);
    if (path === '/media/filmstrip') { seen.filmstrips.push(url.search); return json(mockFilmstrip(url)); }
    if (path === '/media/content') return route.fulfill({ status: 204 });
    if (path === '/media/inspect-faces' && options.inspectError) return json({ error: { code: options.inspectError.code, message: options.inspectError.message, request_id: 'r1abc' } }, options.inspectError.status);
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
  await page.setViewportSize({ width: options.width ?? 1680, height: options.height ?? 1050 });
  await page.goto(options.route ?? '/#/faceswap');
  if (options.openProject) {
    await page.getByRole('button', { name: 'Projects' }).click();
    await page.getByRole('dialog').getByRole('button', { name: 'Verify & open' }).click();
    await expect(page.locator('[data-track-id="track_0001"]')).toBeVisible();
  }
  return { seen, errors };
}

export const SCREENS = ['home', 'faceswap', 'batch', 'processing', 'facemgr', 'extras', 'gallery', 'history', 'settings'] as const;
export const VIEWPORTS = [
  { name: 'desktop-1680', width: 1680, height: 1050 }, { name: 'laptop-1366', width: 1366, height: 768 },
  { name: 'tablet-landscape-1024', width: 1024, height: 768 }, { name: 'tablet-portrait-820', width: 820, height: 1180 }, { name: 'tablet-portrait-768', width: 768, height: 1024 },
] as const;
