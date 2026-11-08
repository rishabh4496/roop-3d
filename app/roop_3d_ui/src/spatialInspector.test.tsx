// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import SpatialInspector from './components/studio/SpatialInspector';
import { kelvinToRgb } from './utils/kelvinToRgb';
import type { Coverage, Reconstruction, Track } from './contracts';
import { DEFAULT_PARAMETERS } from './components/workstation/pipeline';

// Mock Three.js Canvas and R3F components in JSDOM environment
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: any) => <div data-testid="r3f-canvas">{children}</div>,
  useFrame: vi.fn(),
  useThree: () => ({
    camera: {
      position: { copy: vi.fn() },
      up: { copy: vi.fn() },
      lookAt: vi.fn(),
    },
  }),
}));

vi.mock('@react-three/drei', () => ({
  OrbitControls: () => null,
  GizmoHelper: ({ children }: any) => <div>{children}</div>,
  GizmoViewport: () => null,
}));

describe('Tanner Helland Kelvin to RGB conversion', () => {
  it('converts warm 2700K incandescent light to warm amber hex', () => {
    const rgb = kelvinToRgb(2700);
    expect(rgb.r).toBe(255);
    expect(rgb.g).toBeGreaterThan(140);
    expect(rgb.g).toBeLessThan(180);
    expect(rgb.b).toBeGreaterThan(60);
    expect(rgb.b).toBeLessThan(110);
    expect(rgb.hex).toMatch(/^#[0-9a-f]{6}$/i);
  });

  it('converts neutral 5500K daylight to neutral white-blue', () => {
    const rgb = kelvinToRgb(5500);
    expect(rgb.r).toBe(255);
    expect(rgb.g).toBeGreaterThan(200);
    expect(rgb.b).toBeGreaterThan(190);
  });

  it('converts cool 9000K overcast light to cool bluish tint', () => {
    const rgb = kelvinToRgb(9000);
    expect(rgb.r).toBeLessThan(230);
    expect(rgb.b).toBe(255);
  });
});

describe('SpatialInspector Single-Context 3D Studio', () => {
  let host: HTMLDivElement, root: Root;

  const mockCoverage: Coverage = {
    faceset_id: 'alice',
    views: [
      { yaw: 0, pitch: 0, roll: 0, pose_bin: 'center' },
      { yaw: 20, pitch: 10, roll: 0, pose_bin: 'right' },
    ],
    bins: { center: 1, right: 1 },
  };

  const mockTrack: Track = {
    track_id: 'track_0000',
    first_frame: 0,
    last_frame: 10,
    first_timestamp_seconds: 0,
    last_timestamp_seconds: 0.4,
    thumbnail_base64: '',
    thumbnail_mime: 'image/jpeg',
    landmark_count: 5,
    bounding_box: { x1: 10, y1: 10, x2: 100, y2: 100 },
    pose_range: {
      yaw_degrees: [-10, 20],
      pitch_degrees: [-5, 10],
      roll_degrees: [-2, 2],
      method: 'arcface',
    },
    sampled_detections: 10,
  };

  const mockMesh: Reconstruction = {
    method: 'flame',
    coordinate_system: 'opengl',
    vertices: [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
    ],
    faces: [[0, 1, 2]],
    uvs: [
      [0, 0],
      [1, 0],
      [0, 1],
    ],
    texture: null,
    provenance: 'FLAME v1.3 fit',
  };

  beforeEach(() => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    host.remove();
  });

  it('renders spatial navigation tabs and starts on Head Pose tab', async () => {
    await act(async () => {
      root.render(
        <SpatialInspector
          coverage={mockCoverage}
          track={mockTrack}
          mesh={mockMesh}
          parameters={DEFAULT_PARAMETERS}
        />
      );
    });

    expect(host.textContent).toContain('3D SPATIAL STUDIO');
    expect(host.textContent).toContain('FLAME 3D MESH');

    const tabs = [...host.querySelectorAll<HTMLButtonElement>('.spatial-nav-btn')];
    expect(tabs.length).toBe(5);
    expect(tabs.map(t => t.textContent)).toEqual(['Head Pose', 'Coverage', 'Morph', 'Light Probe', 'FLAME Mesh']);

    // Head Pose tab is active
    expect(tabs[0].classList.contains('active')).toBe(true);
    const posePanel = host.querySelector('#spatial-panel-pose')!;
    expect(posePanel.hasAttribute('hidden')).toBe(false);

    // Coverage, Morph, Light Probe & FLAME panels are hidden but mounted in DOM (preserving WebGL context)
    expect(host.querySelector('#spatial-panel-coverage')!.hasAttribute('hidden')).toBe(true);
    expect(host.querySelector('[data-testid="coverage-radar"]')).toBeTruthy();
    expect(host.querySelector('#spatial-panel-morph')!.hasAttribute('hidden')).toBe(true);
    const lightingPanel = host.querySelector('#spatial-panel-lighting')!;
    expect(lightingPanel.hasAttribute('hidden')).toBe(true);
    const flamePanel = host.querySelector('#spatial-panel-flame')!;
    expect(flamePanel.hasAttribute('hidden')).toBe(true);
  });

  it('switches to Light Probe tab and controls relighting parameters', async () => {
    const setParameters = vi.fn();
    await act(async () => {
      root.render(
        <SpatialInspector
          coverage={mockCoverage}
          track={mockTrack}
          mesh={mockMesh}
          parameters={DEFAULT_PARAMETERS}
          setParameters={setParameters}
        />
      );
    });

    const lightingTab = [...host.querySelectorAll<HTMLButtonElement>('.spatial-nav-btn')].find(
      t => t.textContent?.includes('Light Probe')
    )!;

    await act(async () => {
      lightingTab.click();
    });

    expect(lightingTab.classList.contains('active')).toBe(true);
    const lightingPanel = host.querySelector('#spatial-panel-lighting')!;
    expect(lightingPanel.hasAttribute('hidden')).toBe(false);

    // Switch to manual lighting mode inside LightProbeSphere
    const manualBtn = lightingPanel.querySelector<HTMLButtonElement>('[aria-label="Manual 3D Virtual Key Light"]')!;
    await act(async () => {
      manualBtn.click();
    });
    expect(setParameters).toHaveBeenCalledWith(expect.objectContaining({ lighting_mode: 'manual' }));
  });

  it('switches to FLAME Mesh tab and renders dual-endpoint mesh controls', async () => {
    await act(async () => {
      root.render(
        <SpatialInspector
          coverage={mockCoverage}
          track={mockTrack}
          mesh={mockMesh}
          parameters={DEFAULT_PARAMETERS}
        />
      );
    });

    const flameTab = [...host.querySelectorAll<HTMLButtonElement>('.spatial-nav-btn')].find(
      t => t.textContent?.includes('FLAME Mesh')
    )!;

    await act(async () => {
      flameTab.click();
    });

    expect(flameTab.classList.contains('active')).toBe(true);
    const flamePanel = host.querySelector('#spatial-panel-flame')!;
    expect(flamePanel.hasAttribute('hidden')).toBe(false);

    expect(flamePanel.textContent).toContain('FLAME 3D Topology');
    expect(flamePanel.textContent).toContain('3 Verts · 1 Faces');
    expect(flamePanel.textContent).toContain('FLAME v1.3 fit');
  });
});
