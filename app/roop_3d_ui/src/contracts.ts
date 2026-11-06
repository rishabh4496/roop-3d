import type { WorkstationParameters } from './components/workstation/pipeline';
import type { ProjectPreview } from './previewPreferences';
export type Schema = { type?: string; title?: string; description?: string; enum?: (string | number | boolean)[]; const?: unknown; minimum?: number; maximum?: number; properties?: Record<string, Schema> };
export type Config = Record<string, string | number | boolean | string[]>;
export type Feature = { enabled: boolean; reason: string | null };
export interface Capabilities {
  enabled_modes: string[]; rendering_worker_available: boolean; legal_parameter_bounds: Schema;
  inspection_parameter_bounds: Schema; job_parameter_bounds: Schema;
  models: { model_id: string; name: string; state: string; license_state: string; inference_state: string; sha256: string }[];
  workstation: { features: Record<string, Feature>; controls: Record<string, Record<string, string | null>>;
    active_hardware_profile?: string; resolved_hardware_profile?: string;
    read_only_parameters: Record<string, string>; disabled_parameters: Record<string, string>; job_states: string[];
    job_controls: Record<string, string[]>; upload_chunk_bytes: number; max_asset_bytes: number };
}
export type JobState = 'queued' | 'preparing' | 'running' | 'pausing' | 'paused' | 'cancelling' | 'cancelled' | 'succeeded' | 'failed' | 'interrupted';
export type CanonicalJobState = 'QUEUED' | 'PREFLIGHT' | 'PRESCANNING' | 'PREPASSING' | 'RENDERING' | 'MUXING' | 'PAUSED' | 'COMPLETED' | 'FAILED' | 'CANCELLED';

export interface QueueSlice {
  slice_id: string;
  slice_index: number;
  chunk_path: string;
  rendered_chunk_path?: string | null;
  start_seconds: number;
  end_seconds?: number | null;
  duration: number;
  status: 'PENDING' | 'RENDERING' | 'COMPLETED' | 'FAILED';
  error?: string | null;
  retry_count: number;
  frame_count?: number;
}

export interface QueueJob {
  job_id: string;
  name: string;
  state: CanonicalJobState;
  queue_order: number;
  request: Record<string, any>;
  progress: {
    frame: number;
    total_frames: number | null;
    fps: number;
    stage: string;
    stage_latency_ms?: number;
    stage_timings_ms?: Record<string, number>;
  };
  slices: QueueSlice[];
  checkpoint_path?: string | null;
  output?: Record<string, any> | null;
  error_message?: string | null;
  revision: number;
  created_at: string;
  updated_at: string;
}
export interface Mapping { track_id: string; faceset_id: string }
export interface ExportOptions {
  baked_video: boolean; passes: string[]; finishing: { cube: string | null; grain: { size: number; intensity: number; shadows: number; highlights: number; seed: number }; provenance: string };
  /** Render preset (see /media/export-profiles). Omitted = the standard MP4 default. */
  profile?: 'standard_mp4' | 'high_quality_mp4' | 'archival_hevc10' | 'mov_workflow' | 'lossless_intermediate' | 'image_sequence' | 'color_preserving' | null;
  hdr_policy?: 'tonemap_sdr' | 'reject'; subtitles?: 'copy' | 'require' | 'drop'; sequence_format?: 'png' | 'tiff' | 'dpx';
}
export interface JobRequest { engine_id?: string; identity_lock_id?: string | null; identity_blend?: {method: 'slerp' | 'linear'; source_face_ids: string[]; weights?: number[]} | null; target_path: string; inspection_id: string; face_mappings: Mapping[]; processing: 'local'; remote_consent: false; trim_start_seconds: number | null; trim_end_seconds: number | null; mask_base64?: string | null; workstation_parameters?: WorkstationParameters | null; parameter_curves?: { tracks: Record<string, {frame: number; value: number; easing: string}[]> } | null; export?: ExportOptions | null; face_segments?: {cluster_id: string; track_id: string; start_frame: number; end_frame: number | null; role: string; faceset_id: string | null; source_view_index: number | null}[] | null }
export interface Job { project_id?: string | null; effective_settings?: Config | null; job_id: string; state: JobState; request: JobRequest; progress: { frame: number; total_frames: number | null; fps: number; stage: string; stage_latency_ms?: number; stage_timings_ms?: Record<string, number> }; output: { output_path: string; frames: number; passes?: Record<string, string>; pass_manifest_path?: string | null; baked_video?: boolean } | null; failure: { message: string; code?: string; recoverable?: boolean } | null }
export interface Track { track_id: string; first_frame: number; last_frame: number; first_timestamp_seconds: number; last_timestamp_seconds: number; thumbnail_base64: string; thumbnail_mime: string; landmark_count: number; pose_range: { yaw_degrees: [number, number] | null; pitch_degrees: [number, number] | null; roll_degrees: [number, number] | null; method: string }; bounding_box?: { x1: number; y1: number; x2: number; y2: number }; sampled_detections?: number }
export interface DetectedFace {
  id: string;
  frameIndex: number;
  timestamp?: number;
  bbox: [number, number, number, number];
  landmarks5?: [number, number][];
  landmarks68?: [number, number][];
  embedding?: number[];
  clusterId: string;
  /** Legacy frame (pitch + = looking down). Prefer `pose3d` for anything shown to the user. */
  pose: { yaw: number; pitch: number; roll: number };
  /** Standard frame (pitch + = looking up), when the server provided it. */
  pose3d?: { yaw: number; pitch: number; roll: number };
  confidence?: number;
  thumbnailUrl?: string;
}
export interface TargetCluster {
  id: string;
  name?: string;
  faceIds: string[];
  frameCount: number;
  avgPose: { yaw: number; pitch: number; roll: number };
  representativeFaceId: string;
  thumbnailUrl?: string;
  mode: 'swap' | 'ignore' | 'exclude';
  sourceFaceId?: string | null;
  originTrackId?: string;
  startFrame?: number;
  endFrame?: number | null;
  sourceViewIndex?: number | null;
  splitLocked?: boolean;
}
export interface FaceSelectionState {
  detectedFaces: DetectedFace[];
  targetClusters: TargetCluster[];
  assignments: Record<string, string>;
  activeSelection: string | null;
  hoveredFaceId: string | null;
}
export interface Inspection { inspection_id: string; target_sha256: string; tracks: Track[]; warnings: string[]; truncated: boolean; tracking_method?: string; identity_report_sha256?: string | null; scene_cut_frames?: number[] }
export interface IdentitySheetTrack { track_id: string; persistent_id: number; first_frame: number; last_frame: number; segments: [number, number, number, number][]; missing_ranges: [number, number][]; visible_frames: number; swappable_frames: number; withheld_frames: number; withheld_reasons: Record<string, number>; state_frames: Record<string, number>; confidence_mean: number; confidence_min: number; scene_ids: number[]; thumbnails: { frame: number; timestamp_seconds: number; jpeg_base64: string }[] }
export interface IdentitySheet { inspection_id: string; target_sha256: string; truncated: boolean; scene_cut_frames: number[]; warnings: string[]; tracks: IdentitySheetTrack[] }
export interface IdentityLock { lock_id: string; revision: number; digest: string; target_path: string; target_sha256: string; inspection_id: string; assignments: { track_id: string; faceset_id: string; faceset_sha256: string }[]; excluded_tracks: string[]; status: { valid: boolean; reasons: { code: string; detail: string; track_id?: string; faceset_id?: string }[] } }
export interface FaceSet { faceset_id: string; name: string; item_count: number }
/** `GET /facesets/{id}/views`: stored view poses (legacy pitch frame, + = looking down). */
export interface FaceSetView { index: number; yaw: number; pitch: number; roll: number }
/** `GET|POST /facesets/{id}/identity-asset` (Stage 7 server/identity_assets.py). GET never builds. */
export interface IdentityAssetStatus {
  faceset_id: string; name: string; state: 'built' | 'not_built'; build_key: string; reused?: boolean; note: string;
  coverage: { readiness: string; view_count: number; usable_count: number; yaw_range_deg: [number, number] | null; yaw_span_deg: number; pitch_span_deg: number;
    yaw_coverage: Record<string, number[]>; pitch_coverage: Record<string, number[]>; missing: string[]; requests: string[]; duplicate_groups: number[][]; capabilities: Record<string, string> };
  asset: null | { build_key: string; created_at: string; algorithm_version: string; reconstruction_level: string; landmark_layout: string; claims: Record<string, boolean>;
    missing_information: string[]; limitations: string[]; statistics: Record<string, unknown>; sources_used: number; sources_total: number; vertex_count: number; face_count: number; texture_size: [number, number] };
  flame: Record<string, unknown>;
}
export interface Probe { width: number; height: number; duration_seconds: number | null; fps_rational: string | null; declared_frame_count: number | null; audio_tracks: unknown[] }
export interface ReconstructedMesh { vertices: number[][]; uvs: number[][]; indices: number[][]; normals: number[][]; texture_base64: string; reconstruction_level: string; limitations: string[] }
export interface ObservedPosePoint { view_index: number; yaw: number; pitch: number; roll: number; bin_index: number; position_xyz: [number, number, number]; reprojection_error_px: number }
export interface Coverage {
  faceset_id: string; views: { yaw: number; pitch: number; roll: number; pose_bin: string }[]; bins: Record<string, number>;
  status?: string; observed_views?: number; sphere_coverage_count?: number; sphere_coverage_total?: number; sphere_coverage_pct?: number;
  live_pose?: { yaw: number; pitch: number; roll: number }; observed_pose_points?: ObservedPosePoint[];
  mesh_2_5d?: ReconstructedMesh; mesh_full_3d?: ReconstructedMesh; uv_texture_data_url?: string;
}
/** `identity_asset` is the Stage 3 per-FaceSet asset served by `GET /facesets/{id}/identity-asset/mesh` (glTF frame). */
/** State of the `POST /faceset/reconstruct-3d` request that feeds the 3D viewer. */
export interface ReconstructionStatus { state: 'idle' | 'loading' | 'ready' | 'error'; message?: string; retry?: () => void }
/** `mesh` is the backend mesh frame of the `mesh_2_5d` / `mesh_full_3d` payloads (+x image-right, +y down, +z toward the viewer). */
export type ReconstructionFrame = 'opencv' | 'opengl' | 'mesh';
export interface Reconstruction { method: 'flame' | 'tps_2_5d' | 'identity_asset';coordinate_system: ReconstructionFrame; vertices: number[][]; faces: number[][]; uvs: number[][] | null; face_uvs?: number[][] | null; texture: string | null; provenance: string }
export interface Project {
  project_id: string;
  name: string;
  request: JobRequest;
  effective_settings: Config;
  last_completed_frame: number;
  checkpoint_job_id: string | null;
  saved_at: string;
  revision?: number;
  preview?: ProjectPreview;
  identity_assets?: Record<string, string>;
  output_history?: { job_id: string; state: JobState; created_at: string; output: Job['output'] }[];
  inputs_verified?: boolean;
  status?: string;
  verification_reason?: string | null;
  missing_inputs?: string[];
  checkpoint?: {
    job_id: string;
    completed_frame: number;
    completed_segment?: number;
    total_segments?: number | null;
    execution_provider?: string;
    input_hashes?: Record<string, string>;
    faceset_ids?: string[];
    pose_cache?: Record<string, number[]>;
    keyframe_curves?: Record<string, unknown>;
    manual_mask?: string | null;
    has_binary_checkpoint?: boolean;
  } | null;
  target_path?: string;
  face_mappings?: Mapping[];
  trim_start_seconds?: number | null;
  trim_end_seconds?: number | null;
  has_mask?: boolean;
  has_curves?: boolean;
}
export interface Telemetry { gpus: { name: string; total_vram_mib: number; free_vram_mib: number }[]; logical_cpu_count: number }
export interface Event { type: string; sequence?: number; hardware?: Telemetry }

export interface DetectedFaceItem {
  id: string;
  track_id: string;
  frame_index: number;
  timestamp: number;
  bbox: [number, number, number, number];
  landmarks5: [number, number][];
  confidence: number;
  pose: { yaw: number; pitch: number; roll: number };
  pose3d?: { yaw: number; pitch: number; roll: number };
  pose_cell?: string;
  thumbnail_base64: string;
}

export interface FrameFaceDetectionResponse {
  inspection_id: string;
  faces: DetectedFaceItem[];
  inspection: Inspection;
}
