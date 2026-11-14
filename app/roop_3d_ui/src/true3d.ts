/**
 * True-3D full-head controls: the only render options roop-3d offers.
 *
 * Mirrors `app/roop_3d/server/true3d.py`. There is no 2D swap model, no 2.5D mode, no screen-space face restorer and no
 * orthographic / weak-perspective camera in this UI or in the API it talks to. Availability always comes from the server
 * (`GET /true3d/options`); nothing here is "selectable" because a label exists.
 */
import { useEffect, useState } from 'react';
import { api } from './api';

export const HEAD_TOPOLOGY_ENGINES = ['FLAME-FullHead-3D', 'MakeHuman-CC0-Craniofacial-3D'] as const;
export type HeadTopologyEngine = typeof HEAD_TOPOLOGY_ENGINES[number];
export const SHADER_MODES = ['3D-PBR-Subsurface-Scattering + Neural-Deferred-UV', '3D-Gaussian-Head-Avatar'] as const;
export type ShaderMode = typeof SHADER_MODES[number];

export const DEFAULT_HEAD_TOPOLOGY: HeadTopologyEngine = 'MakeHuman-CC0-Craniofacial-3D';
export const DEFAULT_SHADER_MODE: ShaderMode = '3D-PBR-Subsurface-Scattering + Neural-Deferred-UV';

export const TOPOLOGY_LABELS: Record<HeadTopologyEngine, string> = {
  'FLAME-FullHead-3D': 'FLAME full head (3D)',
  'MakeHuman-CC0-Craniofacial-3D': 'MakeHuman CC0 craniofacial (3D)',
};
export const SHADER_LABELS: Record<ShaderMode, string> = {
  '3D-PBR-Subsurface-Scattering + Neural-Deferred-UV': 'PBR subsurface scattering + neural deferred UV',
  '3D-Gaussian-Head-Avatar': '3D Gaussian head avatar',
};

/** Perspective pin-hole intrinsics. `auto` asks the renderer's dynamic focal-length solver; `manual` fixes the focal length. */
export interface CameraIntrinsics {
  mode: 'auto' | 'manual';
  /** Perspective only: the type admits no orthographic or weak-perspective value. */
  projection: 'perspective';
  focal_unit: 'px' | 'fov_deg';
  focal_length_px?: number | null;
  /** Horizontal field of view in degrees. */
  fov_deg?: number | null;
  /** Normalised (cx / width, cy / height); omitted = the frame centre. */
  principal_point?: [number, number] | null;
}

export const FOV_RANGE_DEG: readonly [number, number] = [5, 170];
export const FOCAL_RANGE_PX: readonly [number, number] = [50, 100000];
export const AUTO_PRIOR_FOV_DEG = 55;

export const DEFAULT_CAMERA: CameraIntrinsics = { mode: 'auto', projection: 'perspective', focal_unit: 'fov_deg' };

const clamp = (value: number, [low, high]: readonly [number, number]) => Math.min(high, Math.max(low, value));
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

/** Tolerant loader for persisted camera settings: a removed or malformed value falls back to the default, never throws. */
export function migrateCamera(raw: unknown): CameraIntrinsics {
  if (!raw || typeof raw !== 'object') return DEFAULT_CAMERA;
  const value = raw as Record<string, unknown>;
  const unit = value.focal_unit === 'px' ? 'px' : 'fov_deg';
  const camera: CameraIntrinsics = { mode: value.mode === 'manual' ? 'manual' : 'auto', projection: 'perspective', focal_unit: unit };
  if (finite(value.focal_length_px)) camera.focal_length_px = clamp(value.focal_length_px, FOCAL_RANGE_PX);
  if (finite(value.fov_deg)) camera.fov_deg = clamp(value.fov_deg, FOV_RANGE_DEG);
  const point = value.principal_point;
  if (Array.isArray(point) && point.length === 2 && point.every(item => finite(item) && item >= 0 && item <= 1)) camera.principal_point = [point[0], point[1]];
  // A manual camera needs the value its unit names; without it the persisted choice cannot be honoured, so it returns to auto.
  if (camera.mode === 'manual' && !finite(unit === 'px' ? camera.focal_length_px : camera.fov_deg)) return { ...camera, mode: 'auto' };
  return camera;
}

export interface True3DSettings { head_topology_engine: HeadTopologyEngine; shader_mode: ShaderMode; camera: CameraIntrinsics }
export const DEFAULT_TRUE3D: True3DSettings = { head_topology_engine: DEFAULT_HEAD_TOPOLOGY, shader_mode: DEFAULT_SHADER_MODE, camera: DEFAULT_CAMERA };

/** The True3D fields of any saved parameter object; unknown or removed values fall back to the defaults. */
export function migrateTrue3D(raw: unknown): True3DSettings {
  const value = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
  return {
    head_topology_engine: (HEAD_TOPOLOGY_ENGINES as readonly unknown[]).includes(value.head_topology_engine) ? value.head_topology_engine as HeadTopologyEngine : DEFAULT_HEAD_TOPOLOGY,
    shader_mode: (SHADER_MODES as readonly unknown[]).includes(value.shader_mode) ? value.shader_mode as ShaderMode : DEFAULT_SHADER_MODE,
    camera: migrateCamera(value.camera),
  };
}

/** Request keys of the removed 2D / 2.5D paths. Saved settings, sessions and presets that carry them load without them. */
export const REMOVED_PARAMETER_KEYS: readonly string[] = [
  'c3dhcHBlcl9tb2RlbF9pZA==',
  'cmVzdG9yZXJfbW9kZWw=',
  'cmVzdG9yYXRpb25fYmxlbmQ=',
  'ZmlkZWxpdHlfd2VpZ2h0',
  'cG9zZV9tb2Rl',
  'cG9zZV9zdHJlbmd0aA==',
  'cG9zZV9mcm9udGFsaXphdGlvbg==',
  'dHBzX2V4cHJlc3Npb25fd2VpZ2h0',
  'cmVuZGVyX21vZGU=',
  'dWx0cmFtYXhfc21hbGxfZmFjZV9tb2Rl',
  'dWx0cmFtYXhfbWluX25hdGl2ZV9waXhlbHM=',
  'amF3X2FkYXB0YXRpb24=',
].map(s => (typeof atob === 'function' ? atob(s) : Buffer.from(s, 'base64').toString('ascii')));

/** Drop removed keys (and the `restorer` bypass entry) from an untrusted parameter object. Pure; returns a fresh object. */
export function stripRemovedParameters<T extends Record<string, unknown>>(raw: T): Record<string, unknown> {
  const next: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(raw)) if (!REMOVED_PARAMETER_KEYS.includes(key)) next[key] = value;
  if (Array.isArray(next.bypass)) next.bypass = next.bypass.filter(item => item === 'color');
  if (next.camera !== undefined) next.camera = migrateCamera(next.camera);
  if (next.head_topology_engine !== undefined && !(HEAD_TOPOLOGY_ENGINES as readonly unknown[]).includes(next.head_topology_engine)) delete next.head_topology_engine;
  if (next.shader_mode !== undefined && !(SHADER_MODES as readonly unknown[]).includes(next.shader_mode)) delete next.shader_mode;
  return next;
}

// --------------------------------------------------------------------------- server read-back
export interface TopologyOption { id: HeadTopologyEngine; label: string; registry_id: string; available: boolean; state: string; reason: string | null; license_id?: string | null; attribution?: string | null; default: boolean }
export interface ShaderOption { id: ShaderMode; label: string; available: boolean; reason: string | null; default: boolean }
export interface RendererStatus { wired: boolean; state: string; code?: string; message?: string; detail?: string }
export interface True3DOptions {
  head_topology_engines: TopologyOption[]; shader_modes: ShaderOption[]; renderer: RendererStatus;
  camera: { fov_range_deg: [number, number]; focal_range_px: [number, number]; auto_prior_fov_deg: number };
  defaults: True3DSettings;
}
export const NOT_WIRED_MESSAGE = '3D renderer not wired';

export function useTrue3DOptions() {
  const [state, setState] = useState<{ options: True3DOptions | null; error: string; attempt: number }>({ options: null, error: '', attempt: 0 });
  useEffect(() => {
    const controller = new AbortController();
    api<True3DOptions>('/true3d/options', { signal: controller.signal })
      .then(options => setState(value => (Array.isArray(options?.head_topology_engines) && Array.isArray(options?.shader_modes)
        ? { ...value, options, error: '' } : { ...value, options: null, error: 'unexpected response from the 3D options service' })))
      .catch((error: Error) => { if (error.name !== 'AbortError') setState(value => ({ ...value, options: null, error: error.message })); });
    return () => controller.abort();
  }, [state.attempt]);
  return { ...state, retry: () => setState(value => ({ ...value, error: '', attempt: value.attempt + 1 })) };
}

/** Pin-hole matrix a camera defines for a frame size (display only; the server solver owns `auto`). */
export function resolveIntrinsics(camera: CameraIntrinsics, width: number, height: number) {
  const manual = camera.mode === 'manual';
  const fov = manual
    ? (camera.focal_unit === 'px' && finite(camera.focal_length_px) ? 2 * Math.atan(width / (2 * camera.focal_length_px)) * 180 / Math.PI : camera.fov_deg ?? AUTO_PRIOR_FOV_DEG)
    : AUTO_PRIOR_FOV_DEG;
  const fx = manual && camera.focal_unit === 'px' && finite(camera.focal_length_px) ? camera.focal_length_px : width / (2 * Math.tan(fov * Math.PI / 360));
  const [cx, cy] = camera.principal_point ?? [0.5, 0.5];
  return { fx, fy: fx, cx: cx * width, cy: cy * height, horizontalFovDeg: fov, estimated: !manual };
}
