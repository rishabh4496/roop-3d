// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import * as THREE from 'three';
import PoseGeometryViewer, {
  buildMeshAssets, formatPose, legacyPoseToStandard, payloadToReconstruction, poseDirection, poseQuaternion, TEMPLATE_LANDMARK_COUNT,
} from './components/three/PoseGeometryViewer';
import { RENDER_MODES, renderModeInfo } from './components/workstation/renderModes';
import { useFaceSelectionStore } from './useFaceSelectionStore';
import type { Coverage, ReconstructedMesh } from './contracts';

vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: any) => <div data-testid="r3f-canvas">{children}</div>,
  useFrame: vi.fn(),
  useThree: () => ({ camera: { position: { copy: vi.fn() }, up: { copy: vi.fn() }, lookAt: vi.fn() } }),
}));
vi.mock('@react-three/drei', () => ({
  OrbitControls: () => null,
  GizmoHelper: ({ children }: any) => <div>{children}</div>,
  GizmoViewport: () => null,
}));

/** A backend-frame head patch: +x image-right, +y DOWN, +z toward the viewer. Vertex 0 is the nose (largest z), 1 the chin (largest y). */
function payloadMesh(extraVertices = 0): ReconstructedMesh {
  const vertices = [[0, 0, 0.06], [0, 0.1, 0.02], [-0.05, -0.05, 0], [0.05, -0.05, 0]];
  for (let i = 0; i < extraVertices; i++) vertices.push([0.001 * i, 0.001 * i, 0.001 * i]);
  return {
    vertices, uvs: vertices.map((_, i) => [i / vertices.length, 0.5]), indices: [[0, 1, 2], [0, 2, 3]],
    normals: vertices.map(() => [0, 0, 1]), texture_base64: 'iVBORw0KGgo=', reconstruction_level: 'limited_multiview', limitations: ['Template prior beyond the landmarks.'],
  };
}

const payload = (overrides: Partial<Coverage> = {}): Coverage => ({
  faceset_id: 'alice', views: [{ yaw: 0, pitch: 0, roll: 0, pose_bin: 'bin_0' }], bins: {}, status: 'ready',
  observed_views: 5, sphere_coverage_count: 4, sphere_coverage_total: 96, sphere_coverage_pct: 4.17,
  observed_pose_points: [
    { view_index: 0, yaw: 0, pitch: 0, roll: 0, bin_index: 46, position_xyz: [0, 0, 1], reprojection_error_px: 0 },
    { view_index: 1, yaw: 45, pitch: 10, roll: 0, bin_index: 59, position_xyz: [0.7, 0.17, 0.69], reprojection_error_px: 0 },
  ],
  mesh_2_5d: payloadMesh(TEMPLATE_LANDMARK_COUNT), mesh_full_3d: payloadMesh(),
  uv_texture_data_url: 'data:image/png;base64,iVBORw0KGgo=', ...overrides,
});

describe('mesh frame conversion', () => {
  it('turns the backend mesh frame (+y down) into an upright, outward-facing three.js head', () => {
    const recon = payloadToReconstruction(payloadMesh(), 'identity_asset')!;
    expect(recon.coordinate_system).toBe('mesh');
    const { geometry } = buildMeshAssets(recon)!;
    const position = geometry.getAttribute('position');
    const y = [...Array(position.count).keys()].map(i => position.getY(i));
    const z = [...Array(position.count).keys()].map(i => position.getZ(i));
    // The chin (largest backend y) is now the lowest point; the nose is still the point nearest the viewer.
    expect(y.indexOf(Math.min(...y))).toBe(1);
    expect(z.indexOf(Math.max(...z))).toBe(0);
    geometry.dispose();
  });

  it('swaps triangle winding for a mirrored frame so faces stay outward (normals toward +z at the front)', () => {
    const { geometry } = buildMeshAssets(payloadToReconstruction(payloadMesh(), 'identity_asset')!)!;
    const normal = geometry.getAttribute('normal');
    for (let i = 0; i < normal.count; i++) expect(normal.getZ(i)).toBeGreaterThan(0);
    // opengl frames are used as they are, with no swap.
    const opengl = buildMeshAssets({ method: 'flame', coordinate_system: 'opengl', vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces: [[0, 1, 2]], uvs: null, texture: null, provenance: 't' })!;
    expect([...opengl.geometry.getIndex()!.array]).toEqual([0, 1, 2]);
    expect(opengl.geometry.getAttribute('normal').getZ(0)).toBeGreaterThan(0);
  });

  it('centres and fits every mesh to the viewer, keeps one UV per vertex, and drops invalid triangles', () => {
    const mesh = payloadToReconstruction({ ...payloadMesh(), indices: [[0, 1, 2], [0, 2, 99], [1, 2]] as any }, 'tps_2_5d')!;
    const { geometry } = buildMeshAssets(mesh)!;
    expect(geometry.getIndex()!.count).toBe(3);
    expect(geometry.getAttribute('uv').count).toBe(geometry.getAttribute('position').count);
    expect(geometry.boundingSphere!.radius).toBeCloseTo(0.95, 2);
    expect(buildMeshAssets({ ...mesh, faces: [] })).toBeNull();
  });

  it('lifts the 68 template landmarks only when asked, and only from the leading vertices', () => {
    const recon = payloadToReconstruction(payloadMesh(TEMPLATE_LANDMARK_COUNT), 'tps_2_5d')!;
    expect(buildMeshAssets(recon)!.landmarks).toBeNull();
    const withLandmarks = buildMeshAssets(recon, { landmarkCount: TEMPLATE_LANDMARK_COUNT })!;
    expect(withLandmarks.landmarks!.length).toBe(TEMPLATE_LANDMARK_COUNT * 3);
  });

  it('keeps the texture, falls back to the payload UV atlas, and refuses payloads with no geometry', () => {
    expect(payloadToReconstruction(payloadMesh(), 'identity_asset')!.texture).toMatch(/^data:image\/png;base64,/);
    expect(payloadToReconstruction({ ...payloadMesh(), texture_base64: '' }, 'identity_asset', 'data:image/png;base64,AA==')!.texture).toBe('data:image/png;base64,AA==');
    expect(payloadToReconstruction(undefined, 'identity_asset')).toBeNull();
    expect(payloadToReconstruction({ ...payloadMesh(), indices: [] }, 'identity_asset')).toBeNull();
  });
});

describe('pose maths', () => {
  it('converts track poses (pitch + = down) into the standard frame (pitch + = up) exactly once', () => {
    expect(legacyPoseToStandard({ yaw: 10, pitch: 20, roll: 3 })).toEqual({ yaw: 10, pitch: -20, roll: 3 });
    expect(legacyPoseToStandard(null)).toBeNull();
  });

  it('points the head nose at the matching dot of the orientation sphere', () => {
    for (const pose of [{ yaw: 0, pitch: 0 }, { yaw: 40, pitch: 15 }, { yaw: -70, pitch: -25 }, { yaw: 90, pitch: 60 }]) {
      const nose = new THREE.Vector3(0, 0, 1).applyQuaternion(poseQuaternion(pose));
      const dot = poseDirection(pose.yaw, pose.pitch);
      expect(nose.x).toBeCloseTo(dot[0], 5); expect(nose.y).toBeCloseTo(dot[1], 5); expect(nose.z).toBeCloseTo(dot[2], 5);
    }
  });

  it('formats the live pose with explicit signs and never prints a negative zero', () => {
    expect(formatPose({ yaw: 0, pitch: 0 })).toBe('+0.0° Y · +0.0° P');
    expect(formatPose({ yaw: -0.01, pitch: -0 })).toBe('+0.0° Y · +0.0° P');
    expect(formatPose({ yaw: 12.34, pitch: -4.06 })).toBe('+12.3° Y · -4.1° P');
  });
});

describe('render modes', () => {
  it('names the three modes plainly and maps each to the viewer tab it works from', () => {
    expect(RENDER_MODES.map(mode => [mode.id, mode.label])).toEqual([
      ['standard', 'Standard 2D Identity Swap'], ['pose_3d', '3D Facial Warp'], ['reenactment_3d', '3D Expression Transfer'],
    ]);
    expect(RENDER_MODES.map(mode => mode.preview)).toEqual(['estimated_2_5d', 'reconstruction', 'reconstruction']);
    expect(renderModeInfo('unknown').id).toBe('standard');
  });
});

describe('PoseGeometryViewer with a FaceSet reconstruction payload', () => {
  let host: HTMLDivElement, root: Root;
  const render = async (props: Record<string, unknown>) => { await act(async () => { root.render(<PoseGeometryViewer {...props} />); }); };
  const tabs = () => [...host.querySelectorAll<HTMLButtonElement>('.viewer-mode-btn')];
  const activeTab = () => host.querySelector('.viewer-mode-btn.active')?.textContent ?? '';
  const readout = () => [...host.querySelectorAll('.pose-status-readout .status-metric')]
    .map(metric => `${metric.querySelector('.status-label')!.textContent} ${metric.querySelector('strong')!.textContent}`).join(' | ');

  beforeEach(() => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
    useFaceSelectionStore.getState().reset();
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.restoreAllMocks(); });

  it('shows the idle prompt over the reference head when no source is loaded', async () => {
    await render({ coverage: null, track: null, mesh: null });
    expect(host.querySelector('.pose-viewer-overlay.idle')!.textContent).toBe('Load a source face or .fsz faceset to preview 3D reconstruction');
    expect(readout()).toContain('Observed Views 0');
    expect(readout()).toContain('0 / 96 (0%)');
    expect(readout()).toContain('+0.0° Y · +0.0° P');
    expect(tabs()[1].disabled).toBe(true);
  });

  it('reports the loading and error states, and offers a retry', async () => {
    await render({ coverage: null, track: null, mesh: null, reconstruction: { state: 'loading' } });
    expect(host.querySelector('.pose-viewer-overlay.loading')!.textContent).toMatch(/Reconstructing 3D face/);
    const retry = vi.fn();
    await render({ coverage: null, track: null, mesh: null, reconstruction: { state: 'error', message: 'FaceSet archive is missing', retry } });
    const alert = host.querySelector('[role="alert"]')!;
    expect(alert.textContent).toContain('FaceSet archive is missing');
    await act(async () => { alert.querySelector('button')!.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
    expect(retry).toHaveBeenCalledTimes(1);
  });

  it('binds the payload telemetry: observed views, sphere coverage and the live pose', async () => {
    await render({ coverage: payload(), track: null, mesh: null, reconstruction: { state: 'ready' } });
    expect(host.querySelector('.pose-viewer-overlay')).toBeNull();
    expect(readout()).toContain('Observed Views 5');
    expect(readout()).toContain('4 / 96 (4.17%)');
    expect(readout()).toContain('+0.0° Y · +0.0° P');
    expect(host.querySelector('[data-testid="live-pose"]')!.getAttribute('data-source')).toBe('view');
  });

  it('shows the target pose in the telemetry once a target face is selected (track poses converted to the standard frame)', async () => {
    const track: any = { track_id: 'track_0000', pose_range: { yaw_degrees: [10, 30], pitch_degrees: [4, 8], roll_degrees: [0, 0], method: 'x' } };
    await render({ coverage: payload(), track, mesh: null });
    const live = host.querySelector('[data-testid="live-pose"]')!;
    expect(live.textContent).toBe('+20.0° Y · -6.0° P');
    expect(live.getAttribute('data-source')).toBe('target');
  });

  it('opens the fitted reconstruction tab for both 3D render modes', async () => {
    await render({ coverage: payload(), track: null, mesh: null, renderMode: 'standard' });
    expect(activeTab()).toContain('Estimated 2.5D');
    expect(tabs()[1].disabled).toBe(false);
    await render({ coverage: payload(), track: null, mesh: null, renderMode: 'reenactment_3d' });
    expect(activeTab()).toContain('Full 3D Reconstructed');
    expect(tabs()[1].dataset.pipeline).toBe('true');
    expect(tabs()[1].textContent).toContain('render');
    await render({ coverage: payload(), track: null, mesh: null, renderMode: 'pose_3d' });
    expect(activeTab()).toContain('Full 3D Reconstructed');
    expect(tabs()[1].dataset.pipeline).toBe('true');
    // The standard mode has no 3D render step: the tab is labelled as a preview only.
    await render({ coverage: payload(), track: null, mesh: null, renderMode: 'standard' });
    expect(tabs()[0].textContent).toContain('preview');
  });

  it('keeps a manual tab choice until the render mode or source changes', async () => {
    await render({ coverage: payload(), track: null, mesh: null, renderMode: 'pose_3d' });
    await act(async () => { tabs()[0].dispatchEvent(new MouseEvent('click', { bubbles: true })); });
    expect(activeTab()).toContain('Estimated 2.5D');
    await render({ coverage: payload(), track: null, mesh: null, renderMode: 'pose_3d' });
    expect(activeTab()).toContain('Estimated 2.5D');
    await render({ coverage: payload({ faceset_id: 'bob' }), track: null, mesh: null, renderMode: 'pose_3d' });
    expect(activeTab()).toContain('Full 3D Reconstructed');
  });

  it('selects a fitted anatomical surface over the canonical proxy when one is available', async () => {
    const fitted = payload();
    fitted.mesh_full_3d!.reconstruction_level = 'three_tier_facial_surface';
    await render({ coverage: fitted, track: null, mesh: null });
    expect(activeTab()).toContain('Fitted facial surface');
    expect(tabs()[1].title).toContain('Anatomical face');
  });

  it('shows an explicitly chosen identity asset on the Full 3D tab at once', async () => {
    const asset = payloadToReconstruction(payloadMesh(), 'identity_asset')!;
    await render({ coverage: null, track: null, mesh: asset });
    expect(activeTab()).toContain('Full 3D Reconstructed');
    expect(host.querySelector('.pose-viewer-overlay')).toBeNull();
  });

  it('never draws the placeholder crystal head once a source is loading, failed or ready', async () => {
    const crystal = () => host.querySelector('group[name="reference-head"]');
    await render({ coverage: null, track: null, mesh: null });
    expect(crystal()).not.toBeNull(); // idle empty state only
    await render({ coverage: null, track: null, mesh: null, reconstruction: { state: 'loading', sourceId: 'alice' } });
    expect(crystal()).toBeNull();
    await render({ coverage: null, track: null, mesh: null, reconstruction: { state: 'error', message: 'x', sourceId: 'alice' } });
    expect(crystal()).toBeNull();
    await render({ coverage: payload(), track: null, mesh: null, reconstruction: { state: 'ready', sourceId: 'alice' } });
    expect(crystal()).toBeNull();
  });

  it('offers Generate / Refresh 3D Faceset and a create-.fsz slot inside the panel', async () => {
    const refresh = vi.fn();
    await render({ coverage: null, track: null, mesh: null, creator: <button type="button" data-testid="creator">+ Create .fsz from Video/Photos</button>,
      reconstruction: { state: 'idle', sourceId: '', refresh } });
    const generate = host.querySelector<HTMLButtonElement>('[data-testid="generate-3d-faceset"]')!;
    expect(generate.textContent).toContain('Generate / Refresh 3D Faceset');
    expect(generate.disabled).toBe(true); // nothing to reconstruct yet
    expect(host.querySelector('[data-testid="creator"]')!.textContent).toBe('+ Create .fsz from Video/Photos');
    await render({ coverage: payload(), track: null, mesh: null, reconstruction: { state: 'ready', sourceId: 'alice', refresh } });
    await act(async () => { host.querySelector<HTMLButtonElement>('[data-testid="generate-3d-faceset"]')!.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it('holds the vertex / UV buffers, texture and telemetry of a loaded source in the viewer state', async () => {
    const data = payload();
    const full = buildMeshAssets(payloadToReconstruction(data.mesh_full_3d, 'identity_asset', data.uv_texture_data_url))!;
    expect(full.geometry.getAttribute('position').count).toBeGreaterThan(0);
    expect(full.geometry.getAttribute('uv').count).toBe(full.geometry.getAttribute('position').count);
    expect(payloadToReconstruction(data.mesh_full_3d, 'identity_asset', data.uv_texture_data_url)!.texture).toMatch(/^data:image\/png;base64,/);
    await render({ coverage: data, track: null, mesh: null, reconstruction: { state: 'ready', sourceId: 'alice' } });
    expect(readout()).toContain('Observed Views 5');
    expect(readout()).toContain('4 / 96 (4.17%)');
  });

  it('draws a glowing dot for every observed pose point and no target dot without a target', async () => {
    await render({ coverage: payload(), track: null, mesh: null });
    // Each GlowDot is a group of three spheres; the lattice cells are single spheres.
    const dots = host.querySelectorAll('group > mesh > spheregeometry[args="0.035,14,14"]');
    expect(dots.length).toBe(2);
  });
});
