export type RestorerModel = 'gpen' | 'codeformer' | 'ultramax' | 'gfpgan';
/** Seam fusion of the swapped face (engine default `multi_band`: a four-band Laplacian pyramid). */
export type BlendMode = 'feather' | 'multi_band' | 'poisson' | 'frequency_separated';
/** Weight file each enhancer loads. UltraMax is the registered CodeFormer fp16 artifact plus Laplacian detail fusion (not a GPEN / Real-ESRGAN cascade). */
export const restorerWeights = (model: RestorerModel | undefined): string => (model === 'ultramax' ? 'codeformer.fp16.onnx' : model === 'codeformer' ? 'CodeFormerv0.1.onnx' : model === 'gfpgan' ? 'GFPGANv1.4.onnx' : 'gpen_bfr_512.onnx');
/** Display names of the four restorers the engine has adapters for (an unset model is the UI's GPEN default). */
export const RESTORER_LABELS: Record<RestorerModel, string> = { gpen: 'GPEN Realistic', codeformer: 'CodeFormer', gfpgan: 'GFPGAN v1.4', ultramax: 'UltraMax' };
/** Only CodeFormer and UltraMax have a runtime fidelity (`w`) input; the engine rejects `fidelity_weight` for GPEN and GFPGAN. */
export const fidelityRestorer = (model: RestorerModel | undefined): boolean => model === 'codeformer' || model === 'ultramax';
/** Pyramid bands only exist in the multi-band and frequency-separated fusions (an unset mode is the multi-band default). */
export const bandedBlend = (mode: BlendMode | undefined): boolean => !mode || mode === 'multi_band' || mode === 'frequency_separated';
/** The restorer stage as these parameters run it: the selected enhancer, or off. */
export const restorerStageLabel = (parameters: Pick<WorkstationParameters, 'restoration_blend' | 'restorer_model' | 'bypass'>): string =>
  parameters.restoration_blend > 0 && !parameters.bypass.includes('restorer') ? RESTORER_LABELS[parameters.restorer_model ?? 'gpen'] : 'Restorer off';
/**
 * Catalog readiness is keyed by the stable model id. UltraMax is its own row (`ultramax`): the catalog resolves it to the registered
 * `codeformer.fp16` artifact through the hash- and trust-verifying provider, the same check `/system/capabilities` makes.
 */
export function restorerAvailable(models: readonly { id?: string; model_id?: string; name: string; state: string; installed?: boolean }[] | undefined, model: RestorerModel | undefined): boolean {
  const id = model === 'gfpgan' ? 'gfpgan_v14' : model === 'gpen' || !model ? 'gpen_bfr_512' : model;
  return !!models?.some(item => item.installed !== false && item.state === 'VERIFIED_LOCAL' &&
    (item.id === id || item.model_id === id || (model !== 'ultramax' && item.name === restorerWeights(model))));
}
/**
 * Does the weight file a stage has loaded belong to this restorer? The `/models/active` read-back names enhancers by registry hash id,
 * so the file decides. CodeFormer and UltraMax load different artifacts (`CodeFormerv0.1` vs `codeformer.fp16`).
 */
export function restorerFileMatches(filename: string, model: RestorerModel | undefined): boolean {
  const name = filename.toLowerCase();
  if (model === 'ultramax') return name.startsWith('codeformer.fp16') || name.startsWith('ultramax');
  if (model === 'codeformer') return name.startsWith('codeformer') && !name.includes('fp16');
  if (model === 'gfpgan') return name.startsWith('gfpgan');
  return name.startsWith('gpen');
}
export type WorkstationParameters = {
  expression_restore_factor?: number;
  face_expression_overrides?: Record<string, FaceEditor>;
  color_method?: 'lab' | 'reinhard_lab' | 'histogram';
  restorer_model?: RestorerModel; fidelity_weight?: number; identity_blend_t?: number;
  restoration_blend: number; feather_px: number; color_strength: number; pose_strength: number; pose_mode?: 'retarget' | 'mesh_blend';
  swap_weight?: number; erosion_px?: number; bypass: ('restorer' | 'color')[];
  /** Pre-swap 3D pose frontalization, 0..1 (Stage 5). Omitted until the user touches it, so older requests keep their identity. */
  pose_frontalization?: number;
  /** TPS expression warp toward the target frame, 0..1 (Stage 5). */
  tps_expression_weight?: number;
  /** Compositor occlusion strategy. Omitted until the user touches it, so older requests keep the standard path. */
  occlusion_mode?: 'standard' | 'robust';
  /** Robust occlusion only: scale of the XSeg / face-occluder "not face" vote, 0..1 (engine default 1; BiSeNet and manual masks always act in full). */
  xseg_occlusion_strength?: number;
  /** Seam fusion of the swapped face. Omitted until the user picks another (engine default `multi_band`). */
  blend_mode?: BlendMode;
  /** Bands of multi-band / frequency-separated fusion, 4..6 (engine default 4). */
  pyramid_levels?: number;
  /** Stage 4 render mode. Omitted (= 'standard') until the user picks a 3D mode, so older requests keep their identity. */
  render_mode?: 'standard' | 'pose_3d' | 'reenactment_3d';
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
  ultramax_small_face_mode?: 'bypass' | 'adaptive' | 'off';
  ultramax_min_native_pixels?: number;
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
  /** Share of the target's pore / specular residual re-injected after UltraMax, 0..1 (engine default: 0.25). */
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
  restoration_blend: 0,
  feather_px: 5,
  color_strength: .65,
  pose_strength: .35,
  pose_mode: 'mesh_blend',
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
 * The parameters a request carries. A control the current selection cannot act on stays out of it: the engine rejects a fidelity input for
 * GPEN / GFPGAN, and the XSeg vote and the pyramid bands do nothing outside Robust occlusion / the banded fusions. The workspace keeps the
 * value, so switching back finds it again.
 */
export function wireParameters<T extends Partial<WorkstationParameters>>(parameters: T): T {
  const next: Partial<WorkstationParameters> = { ...parameters };
  if (next.occlusion_mode !== 'robust') delete next.xseg_occlusion_strength;
  if (!bandedBlend(next.blend_mode)) delete next.pyramid_levels;
  if (!fidelityRestorer(next.restorer_model)) delete next.fidelity_weight;
  return next as T;
}

/** `buffer` is the `solo_node` that `POST /api/preview` returns for this stage. `latencyKeys` are the measured fields of the preview's `latency` object that make up this stage's time. */
export type PipelineNode = { id: string; label: string; buffer: string; note?: string; optional?: boolean; unavailable?: string; latencyKeys?: string[]; short?: string };
/**
 * The eight inspectable stages, in execution order. Hotkeys `1`–`8` solo them in this order (`buffer1`…`buffer8`).
 * Every stage maps to a buffer the server really returns; where the engine has no buffer of exactly that name the note says
 * which real buffer stands in for it.
 */
export const PIPELINE_NODES: PipelineNode[] = [
  { id: 'input', label: 'Raw Target', short: 'Raw Target', buffer: 'detector', latencyKeys: ['detection_ms'], note: 'Target frame with the detector\'s face boxes' },
  { id: 'pose', label: '6DoF 3D Mesh & Pose', short: '6DoF Mesh & Pose', buffer: 'pose', latencyKeys: ['retarget_ms'], note: 'Landmarks, yaw/pitch/roll and the fitted 3D proxy / TPS warp mesh' },
  { id: 'uv', label: 'Canonical UV Projection', short: 'UV Projection', buffer: 'source', latencyKeys: ['projection_ms'], note: 'The pose-matched canonical source view the identity is projected from (the engine exports this view, not the raw UV atlas)' },
  { id: 'lighting', label: 'SH 3D Relighting & Normals', short: 'SH Relight & Normals', buffer: 'sh_lighting', note: 'Spherical Harmonics scene illumination transfer and shadow mask. Use the Normals chip for the RGB surface-normal map' },
  { id: 'swap', label: 'Base 3D Swap', short: 'Base Swap', buffer: 'swapper', latencyKeys: ['swap_ms'], note: 'The swapped face before masking, restoring or grading' },
  { id: 'restorer', label: 'UltraMax High-Res', short: 'UltraMax', buffer: 'restorer', optional: true, note: 'The selected enhancer\'s output (GPEN, CodeFormer, GFPGAN or UltraMax) at its native resolution' },
  { id: 'matte', label: 'XSeg + Hair Occlusion Matte', short: 'XSeg + Hair Matte', buffer: 'matte', note: 'Black-and-white blend mask. XSeg and the hair/occluder matte are in it when Occlusion handling is set to Robust' },
  { id: 'final', label: 'Final Composite', short: 'Final Composite', buffer: 'final', latencyKeys: ['total_ms'], note: 'The composite exactly as a render would write it' },
];
/** Secondary buffers reached from a chip on their parent stage; they are not stages of their own (and have no hotkey). */
export const PIPELINE_ALT_NODES: (PipelineNode & { parent: string })[] = [
  { id: 'expression', parent: 'swap', label: 'Stage 2.5 Micro-Expression Restorer', short: 'Expression', buffer: 'expression', optional: true, note: 'Target expression residual and per-face 3D edits, before UltraMax' },
  { id: 'normal', parent: 'lighting', label: '3D Normal Map', short: 'Normals', buffer: 'normal_map', note: 'RGB-encoded 3D Surface Normal map (nx, ny, nz)' },
];
export const pipelineNode = (id: string | null): PipelineNode | undefined => PIPELINE_NODES.find(node => node.id === id) ?? PIPELINE_ALT_NODES.find(node => node.id === id);
export const nodeBuffer = (id: string | null): string | null => pipelineNode(id)?.buffer ?? null;

/**
 * Measured milliseconds for one stage from a preview's `latency` object, or null when the engine did not time that stage.
 * Nothing is estimated: the preview times detection, retarget, projection, swap and the total only.
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
