import { stripRemovedParameters, type CameraIntrinsics, type HeadTopologyEngine, type ShaderMode } from '../../true3d';
/** Seam fusion of the rendered head into the frame (engine default `multi_band`: a four-band Laplacian pyramid). */
export type BlendMode = 'feather' | 'multi_band' | 'poisson' | 'frequency_separated';
/** Pyramid bands only exist in the multi-band and frequency-separated fusions (an unset mode is the multi-band default). */
export const bandedBlend = (mode: BlendMode | undefined): boolean => !mode || mode === 'multi_band' || mode === 'frequency_separated';
export type WorkstationParameters = {
  expression_restore_factor?: number;
  face_expression_overrides?: Record<string, FaceEditor>;
  color_method?: 'lab' | 'reinhard_lab' | 'histogram';
  identity_blend_t?: number;
  feather_px: number; color_strength: number;
  swap_weight?: number; erosion_px?: number; bypass: 'color'[];
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
  /** Stage 6B swap scope: which part of the target's head the source geometry replaces. Omitted (= 'full_craniofacial') until the user picks another. */
  swap_scope?: 'full_craniofacial' | 'adaptive_skull' | 'inner_face';
  /** Adaptive scope only: share of the source skull (β source) in the blended skull, 0..1. Omitted otherwise. */
  skull_blend?: number;
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
export type FaceEditor = Partial<Record<'head_pitch' | 'head_yaw' | 'head_roll' | 'gaze_x' | 'gaze_y' | 'blink_left' | 'blink_right' | 'eyebrow_tilt' | 'eyebrow_raise' | 'mouth_smile' | 'mouth_pout' | 'mouth_open' | 'jaw_width' | 'chin_length', number>> & {
  /** Sparse standard ARKit 52 blendshape values, each in [0,1]. */
  arkit_blendshapes?: Record<string, number>;
};
export const DEFAULT_PARAMETERS: WorkstationParameters = {
  feather_px: 5,
  color_strength: .65,
  swap_weight: 1,
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

/**
 * Saved parameters (sessions, projects, jobs, batch presets) from before the purge may carry removed 2D / 2.5D keys. They are dropped, the
 * kept fields are layered over the defaults, and the 3D fields fall back to their defaults when the saved value is no longer valid.
 */
export function migrateParameters(raw: unknown): WorkstationParameters {
  const value = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw as Record<string, unknown> : {};
  return { ...DEFAULT_PARAMETERS, ...stripRemovedParameters(value) } as WorkstationParameters;
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
  { id: 'expression', parent: 'head', label: 'Expression Transfer', short: 'Expression', buffer: 'expression', optional: true, note: 'Target expression residual and per-face 3D edits applied to the head rig' },
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
