import { useEffect, useRef, useState } from 'react';
import { PreviewBusyError, postLivePreview, type LivePreviewMeta } from './api';
import { PreviewURL } from './state';
import type { JobRequest } from './contracts';
import type { WorkstationParameters } from './components/workstation/pipeline';

/** Slider drags coalesce into one request per pause of this length. */
export const LIVE_DEBOUNCE_MS = 150;
/** Dispatched on `window` when the Start button queues a render; in-flight previews are dropped so the server can yield. */
export const JOB_START_EVENT = 'roop3d:job-start';
/** Footer "Render Current Frame": asks the viewport to show the live swap of the frame under the playhead. */
export const RENDER_FRAME_EVENT = 'roop3d:render-frame';

export type LivePayload = Record<string, unknown>;

export function buildLivePayload({ request, time, fps, parameters, maskBase64, restorerAvailable, solo, timestampSelection = false }: {
  request: JobRequest | null; time: number; fps: number; parameters: WorkstationParameters;
  maskBase64?: string | null; restorerAvailable: boolean;
  /** `solo_node` buffer to return instead of the final composite (see PIPELINE_NODES[].buffer). */
  solo?: string | null;
  timestampSelection?: boolean;
}): LivePayload | null {
  if (!request || !request.face_mappings.length) return null;
  const restoring = restorerAvailable && (parameters.restoration_blend > 0 || request.parameter_curves?.tracks.restorer_opacity?.some(key=>key.value>0)) && !parameters.bypass.includes('restorer');
  return {
    engine_id: request.engine_id ?? 'reference_onnx',
    target_path: request.target_path,
    inspection_id: request.inspection_id,
    ...(timestampSelection ? { timestamp: Math.max(0, time) } : { frame_index: Math.max(0, Math.round(time * fps)) }),
    identity_lock_id: request.identity_lock_id ?? undefined,
    face_mappings: request.face_mappings,
    frontalization_weight: parameters.pose_mode === 'mesh_blend' ? parameters.pose_strength : undefined,
    pose_frontalization: parameters.pose_frontalization || undefined,
    tps_expression_weight: parameters.tps_expression_weight || undefined,
    expression_restore_factor: parameters.expression_restore_factor,
    face_expression_overrides: parameters.face_expression_overrides,
    pose_strength: parameters.pose_strength,
    solo_node: solo || undefined,
    occlusion: { erosion_px: parameters.erosion_px ?? 0, feather_px: parameters.feather_px },
    mask_base64: maskBase64 || request.mask_base64 || undefined,
    restorer: restoring ? (parameters.restorer_model === 'codeformer' || parameters.restorer_model === 'ultramax' ? parameters.restorer_model : 'gpen') : 'off',
    identity_blend: request.identity_blend ?? undefined,
    identity_blend_t: parameters.identity_blend_t,
    fidelity_weight: parameters.fidelity_weight,
    restoration_blend: restoring ? parameters.restoration_blend : undefined,
    color_strength: parameters.color_strength,
    color_method: parameters.color_method ?? 'lab',
    ultramax_small_face_mode: parameters.ultramax_small_face_mode,
    ultramax_min_native_pixels: parameters.ultramax_min_native_pixels,
    swap_weight: parameters.swap_weight ?? 1,
    // Opt-in controls: only sent once touched, so earlier previews keep their cache keys.
    hairline_offset_px: parameters.hairline_offset_px || undefined,
    bypass: parameters.bypass,
    // Stage 7: preview what Start would render. Omitted at their defaults so earlier preview cache keys are unchanged.
    render_mode: parameters.render_mode && parameters.render_mode !== 'standard' ? parameters.render_mode : undefined,
    occlusion_mode: parameters.occlusion_mode && parameters.occlusion_mode !== 'standard' ? parameters.occlusion_mode : undefined,
    ...(parameters.relight ? {
      relight: true, relight_strength: parameters.relight_strength, specular_sheen: parameters.specular_sheen, shadow_preservation: parameters.shadow_preservation,
      lighting_mode: parameters.lighting_mode, manual_light_yaw: parameters.manual_light_yaw, manual_light_pitch: parameters.manual_light_pitch,
      ambient_fill: parameters.ambient_fill,
    } : {}),
    face_segments: request.face_segments ?? undefined,
    parameter_curves: request.parameter_curves ?? undefined,
    export: request.export ?? undefined,
  };
}

/** `roop3d_preview_2026-11-01T14-02-03Z.png`: ISO-8601 UTC, with no characters Windows forbids in filenames. */
export const previewFilename = (date: Date = new Date()) =>
  `roop3d_preview_${date.toISOString().replace(/\.\d{3}Z$/, 'Z').replace(/:/g, '-')}.png`;

/** Programmatic blob download. Invoked from a click handler, so no `download` attribute is needed on the control itself. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.rel = 'noopener';
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export type LiveStatus = 'idle' | 'loading' | 'ready' | 'busy' | 'error';
export interface LivePreview {
  status: LiveStatus;
  url: string | null;
  blob: Blob | null;
  meta: LivePreviewMeta | null;
  cached: boolean;
  /** Stage that owns the GPU while `status === 'busy'`. */
  stage: string | null;
  error: string | null;
  /** The shown image belongs to an earlier request; a newer one is pending, busy or failed. */
  stale: boolean;
  retry: () => void;
}

type Snapshot = Omit<LivePreview, 'retry'>;
const IDLE: Snapshot = { status: 'idle', url: null, blob: null, meta: null, cached: false, stage: null, error: null, stale: false };

/**
 * Posts `payload` to `/api/preview` whenever it changes (debounced). A 503 busy answer is not an error:
 * the previous image stays up, a status flag is raised, and the request is re-sent after `retry_after_ms`.
 * Pass `null` to switch the preview off (nothing is requested and the image is released).
 */
export function useLiveSwapPreview(payload: LivePayload | null, debounceMs: number = LIVE_DEBOUNCE_MS): LivePreview {
  const [state, setState] = useState<Snapshot>(IDLE);
  const [nonce, setNonce] = useState(0);
  const lease = useRef(new PreviewURL());
  const key = payload ? JSON.stringify(payload) : '';

  useEffect(() => () => lease.current.clear(), []);
  useEffect(() => {
    const restart = () => setNonce(value => value + 1);
    window.addEventListener(JOB_START_EVENT, restart);
    return () => window.removeEventListener(JOB_START_EVENT, restart);
  }, []);

  useEffect(() => {
    if (!key) {
      lease.current.clear();
      setState(previous => previous.status === 'idle' ? previous : IDLE);
      return;
    }
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    const run = async () => {
      setState(previous => previous.status === 'busy' ? previous : { ...previous, status: 'loading', error: null, stale: previous.url !== null });
      try {
        const result = await postLivePreview(JSON.parse(key), { signal: controller.signal });
        if (!live) return;
        if (result.cancelled) {
          setState(previous => ({ ...previous, status: previous.url !== null ? 'ready' : 'idle', stale: previous.url !== null }));
          return;
        }
        setState({ status: 'ready', url: lease.current.replace(result.blob), blob: result.blob, meta: result.meta,
          cached: result.cached, stage: null, error: null, stale: false });
      } catch (error) {
        if (!live || (error as Error).name === 'AbortError' || (error as Error).name === 'DOMException') return;
        if (error instanceof PreviewBusyError) {
          setState(previous => ({ ...previous, status: 'busy', stage: error.stage, error: null, stale: previous.url !== null }));
          timer = setTimeout(() => void run(), error.retryAfterMs);
        } else {
          setState(previous => ({ ...previous, status: 'error', error: (error as Error).message, stale: previous.url !== null }));
        }
      }
    };
    timer = setTimeout(() => void run(), debounceMs);
    return () => { live = false; clearTimeout(timer); controller.abort(); };
  }, [key, nonce, debounceMs]);

  return { ...state, retry: () => setNonce(value => value + 1) };
}
