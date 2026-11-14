import { type CameraIntrinsics, type HeadTopologyEngine, type ShaderMode } from '../../true3d';
/** Seam fusion of the rendered head into the frame (engine default `multi_band`: a four-band Laplacian pyramid). */
export type BlendMode = 'feather' | 'multi_band' | 'poisson' | 'frequency_separated';
/** Pyramid bands only exist in the multi-band and frequency-separated fusions (an unset mode is the multi-band default). */
export const bandedBlend = (mode: BlendMode | undefined): boolean => !mode || mode === 'multi_band' || mode === 'frequency_separated';
export type WorkstationParameters = {
  feather_px: number; color_strength: number;
  erosion_px?: number; bypass: 'color'[];
  color_method?: 'lab' | 'reinhard_lab' | 'histogram' | 'ycbcr' | 'oklab' | 'oklab_wasserstein';
  /** 3D head topology engine. Omitted until the user picks one (server default: MakeHuman CC0 craniofacial). */
  head_topology_engine?: HeadTopologyEngine;
  /** 3D photorealistic shader mode. Omitted until the user picks one (server default: PBR subsurface scattering + neural deferred UV). */
  shader_mode?: ShaderMode;
  /** Perspective camera intrinsics: the dynamic focal-length solver (`auto`) or a fixed focal length / field of view (`manual`). */
  camera?: CameraIntrinsics;
  /** Compositor occlusion strategy. Omitted until the user touches it, so older requests keep the standard path. */
  occlusion_mode?: 'standard' | 'robust';
  /** Robust occlusion only: scale of the XSeg / face-occluder "not face" vote, 0..1 (engine default 1; BiSeNet and manual masks always act in full). */
  xseg_occlusion_strength?: number;
  /** Seam fusion of the rendered head. Omitted until the user picks another (engine default `multi_band`). */
  blend_mode?: BlendMode;
  /** Bands of multi-band / frequency-separated fusion, 4..6 (engine default 4). */
  pyramid_levels?: number;
  /** Stage 17 3D Relighting: off unless enabled; the strength / light controls below only apply while it is on. */
  relight?: boolean;
  relight_strength?: number;
  /** Confidence the semantic face parser must reach before its mask is trusted (0..1, engine default 0.55). Omitted until touched. */
  occlusion_threshold?: number;
  specular_sheen?: number;
  shadow_preservation?: number;
  lighting_mode?: 'auto' | 'manual';
  manual_light_yaw?: number;
  manual_light_pitch?: number;
  /** Ambient fill (the SH band-0 term) of the manual / scene light, 0..1. Omitted until the user touches it. */
  ambient_fill?: number;
  /** Forehead / hairline boundary offset in px (+ grows the swapped region upward, - pulls it down), -32..32. Omitted until touched. */
  hairline_offset_px?: number;
  // Hair / neck / ear protection and skin micro-texture: each is omitted until it differs from the engine default (see studio/protection.ts).
  /** Clamp the matte along the 3D jawline and keep neck / cloth out of the swap (engine default: on). */
  neck_protection?: boolean;
  /** Guided-filter matting of the forehead / hairline instead of a plain Gaussian feather (engine default: on). */
  hair_matting?: boolean;
  /** Keep the target's ears and eyeglasses (engine default: on). */
  preserve_ears_glasses?: boolean;
  /** Inward falloff of the jaw-to-neck clamp, 2..24 px (engine default: 6). */
  jaw_feather_px?: number;
  /** Share of the target's pore / specular residual kept in the render, 0..1 (engine default: 0.25). */
  skin_texture_retention?: number;
  /** Target camera shutter angle in degrees; 180 is the conventional half-frame exposure. */
  sensor_shutter_angle?: number;
  /** Grain variance match strength, 0..100 percent. */
  grain_match_strength?: number;
};
export const DEFAULT_PARAMETERS: WorkstationParameters = {
  feather_px: 5,
  color_strength: .65,
  erosion_px: 0,
  bypass: [],
  relight_strength: 0.8,
  specular_sheen: 0.25,
  shadow_preservation: 0.85,
  lighting_mode: 'auto',
  manual_light_yaw: 35,
  manual_light_pitch: 25,
};
/**
 * The parameters a request carries. A control the current selection cannot act on stays out of it: the XSeg vote and the pyramid bands do
 * nothing outside Robust occlusion / the banded fusions. The workspace keeps the value, so switching back finds it again.
 */
export function wireParameters<T extends Partial<WorkstationParameters>>(parameters: T): T {
  const next: Partial<WorkstationParameters> = { ...parameters };
  if (next.occlusion_mode !== 'robust') delete next.xseg_occlusion_strength;
  if (!bandedBlend(next.blend_mode)) delete next.pyramid_levels;
  return next as T;
}

const WORKSTATION_PARAMETER_KEYS = new Set<keyof WorkstationParameters>([
  'feather_px', 'color_strength', 'erosion_px', 'bypass', 'color_method', 'head_topology_engine', 'shader_mode', 'camera',
  'occlusion_mode', 'xseg_occlusion_strength', 'blend_mode', 'pyramid_levels', 'relight', 'relight_strength',
  'occlusion_threshold', 'specular_sheen', 'shadow_preservation', 'lighting_mode', 'manual_light_yaw',
  'manual_light_pitch', 'ambient_fill', 'hairline_offset_px', 'neck_protection', 'hair_matting', 'preserve_ears_glasses',
  'jaw_feather_px', 'skin_texture_retention', 'sensor_shutter_angle', 'grain_match_strength',
]);

/** Restore persisted parameters only when every key belongs to the current full-head contract. */
export function migrateParameters(raw: unknown): WorkstationParameters {
  const value = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw as Record<string, unknown> : {};
  const unsupported = Object.keys(value).filter(key => !WORKSTATION_PARAMETER_KEYS.has(key as keyof WorkstationParameters));
  if (unsupported.length) throw new Error(`Unsupported workstation parameter keys: ${unsupported.join(', ')}`);
  return { ...DEFAULT_PARAMETERS, ...value } as WorkstationParameters;
}

/** `buffer` is the `solo_node` that `POST /api/preview` returns for this stage. `latencyKeys` are the measured fields of the preview's `latency` object that make up this stage's time. */
export type PipelineNode = { id: string; label: string; buffer: string; note?: string; optional?: boolean; unavailable?: string; latencyKeys?: string[]; short?: string };
/**
 * The eight inspectable stages, in execution order. Hotkeys `1`–`8` solo them in this order (`buffer1`…`buffer8`).
 * Every stage maps to a buffer the server really returns; where the engine has no buffer of exactly that name the note says
 * which real buffer stands in for it.
 */
export const PIPELINE_NODES: PipelineNode[] = [
  { id: 'input', label: 'Raw Target', short: 'Raw Target', buffer: 'detector', latencyKeys: ['detection_ms'], note: "Target frame with the detector's face boxes" },
  { id: 'pose', label: '6DoF Head Pose & Mesh', short: '6DoF Head & Pose', buffer: 'pose', latencyKeys: ['tracking_ms'], note: 'Landmarks, yaw/pitch/roll and the fitted full-head mesh under the perspective camera' },
  { id: 'uv', label: 'Head UV Atlas', short: 'UV Atlas', buffer: 'source', latencyKeys: ['projection_ms'], note: "The source view the head's UV atlas is baked from" },
  { id: 'lighting', label: 'SH 3D Relighting & Normals', short: 'SH Relight & Normals', buffer: 'sh_lighting', note: 'Spherical Harmonics scene illumination transfer and shadow mask. Use the Normals chip for the RGB surface-normal map' },
  { id: 'head', label: '3D Head Render', short: '3D Head', buffer: 'final', latencyKeys: ['render_ms'], note: 'The shaded full-head render (PBR subsurface scattering or Gaussian head avatar) before masking and grading' },
  { id: 'matte', label: 'XSeg + Hair Occlusion Matte', short: 'XSeg + Hair Matte', buffer: 'matte', note: 'Black-and-white blend mask. XSeg and the hair/occluder matte are in it when Occlusion handling is set to Robust' },
  { id: 'color', label: 'Colour Match & Grade', short: 'Colour Match', buffer: 'color', optional: true, note: 'The head after colour transfer to the target frame. Bypass it to see the raw render colour' },
  { id: 'final', label: 'Final Composite', short: 'Final Composite', buffer: 'final', latencyKeys: ['total_ms'], note: 'The composite exactly as a render would write it' },
];
/** Secondary buffers reached from a chip on their parent stage; they are not stages of their own (and have no hotkey). */
export const PIPELINE_ALT_NODES: (PipelineNode & { parent: string })[] = [
  { id: 'normal', parent: 'lighting', label: '3D Normal Map', short: 'Normals', buffer: 'normal_map', note: 'RGB-encoded 3D Surface Normal map (nx, ny, nz)' },
];
export const pipelineNode = (id: string | null): PipelineNode | undefined => PIPELINE_NODES.find(node => node.id === id) ?? PIPELINE_ALT_NODES.find(node => node.id === id);
export const nodeBuffer = (id: string | null): string | null => pipelineNode(id)?.buffer ?? null;

/**
 * Measured milliseconds for one stage from a preview's `latency` object, or null when the engine did not time that stage.
 * Nothing is estimated: the preview times only the stages the renderer reports.
 */
export function stageLatency(node: PipelineNode, latency: Record<string, number> | null | undefined): number | null {
  if (!latency || !node.latencyKeys?.length) return null;
  let total = 0;
  for (const key of node.latencyKeys) {
    const value = latency[key];
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return null;
    total += value;
  }
  return total;
}
export const formatMs = (value: number | null): string => value === null ? '—' : value < 10 ? `${value.toFixed(1)} ms` : `${Math.round(value)} ms`;

export type PreviewResult = { frame_index: number; timestamp: number; notes: string[]; buffers: Record<string, string>; latency?: Record<string, number>; parameters: WorkstationParameters };
