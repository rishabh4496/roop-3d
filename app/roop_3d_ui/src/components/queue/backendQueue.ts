/** Types and rules for the backend-owned persistent queue (`/api/v1/queue`). Pure: no React, no fetch. */

export const LIFECYCLE_STATES = ['queued', 'preparing', 'compiling_trt', 'tracking_3d', 'rendering', 'paused', 'muxing_audio', 'completed', 'failed', 'cancelled'] as const;
export type LifecycleState = (typeof LIFECYCLE_STATES)[number];

export const LIFECYCLE_LABEL: Record<LifecycleState, string> = {
  queued: 'Queued', preparing: 'Preparing', compiling_trt: 'Compiling TensorRT', tracking_3d: 'Tracking 3D', rendering: 'Rendering',
  paused: 'Paused', muxing_audio: 'Muxing audio', completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled',
};
export const TERMINAL_STATES: ReadonlySet<LifecycleState> = new Set(['completed', 'failed', 'cancelled']);

export interface QueueSlice { slice_id: string; slice_index: number; status: string; error?: string | null; retry_count?: number; duration?: number }
export interface QueueJob {
  job_id: string; name: string; state: string; lifecycle_state: LifecycleState; queue_order: number;
  progress: { frame: number; total_frames: number | null; fps: number; stage: string; advisory?: string | null };
  slices: QueueSlice[]; error_message?: string | null; output?: Record<string, unknown> | null; revision: number;
}

export type QueueAction = 'pause' | 'resume' | 'cancel' | 'duplicate' | 'join';

/** Which controls a job offers. The server enforces the same rules; this only decides what to show. */
export function queueActions(job: Pick<QueueJob, 'lifecycle_state' | 'slices' | 'output'>): QueueAction[] {
  const state = job.lifecycle_state;
  const actions: QueueAction[] = [];
  if (state === 'queued' || state === 'preparing' || state === 'compiling_trt' || state === 'tracking_3d' || state === 'rendering') actions.push('pause');
  if (state === 'paused') actions.push('resume');
  if (!TERMINAL_STATES.has(state)) actions.push('cancel');
  actions.push('duplicate');
  const slices = job.slices ?? [];
  if (slices.length > 1 && slices.every(slice => slice.status === 'COMPLETED') && !job.output && (state === 'rendering' || state === 'muxing_audio' || state === 'paused' || state === 'failed')) actions.push('join');
  return actions;
}

export const failedSlices = (job: Pick<QueueJob, 'slices'>) => (job.slices ?? []).filter(slice => slice.status === 'FAILED');

/** Only jobs that have not started can be re-ordered; running and finished jobs keep their place. */
export const isReorderable = (job: Pick<QueueJob, 'lifecycle_state'>) => job.lifecycle_state === 'queued' || job.lifecycle_state === 'paused';

/**
 * The id list to POST after a drag. Jobs that cannot move keep their slots; the movable ones fill the remaining slots in the
 * new relative order. This is what stops a drop from "moving" a running job.
 */
export function reorderPayload(before: readonly QueueJob[], after: readonly QueueJob[]): string[] {
  const movable = after.filter(isReorderable).map(job => job.job_id);
  let next = 0;
  return before.map(job => (isReorderable(job) ? movable[next++] : job.job_id));
}

export function progressPercent(job: Pick<QueueJob, 'progress' | 'lifecycle_state'>): number {
  if (job.lifecycle_state === 'completed') return 100;
  const total = job.progress.total_frames;
  return total && total > 0 ? Math.min(100, Math.max(0, (job.progress.frame / total) * 100)) : 0;
}

/** Stage-by-stage position for the lifecycle strip: done / current / upcoming. Terminal failure states mark the strip failed. */
export function lifecycleSteps(current: LifecycleState): { state: LifecycleState; status: 'done' | 'current' | 'upcoming' }[] {
  const path: LifecycleState[] = ['queued', 'preparing', 'compiling_trt', 'tracking_3d', 'rendering', 'muxing_audio', 'completed'];
  const at = path.indexOf(current);
  return path.map((state, index) => ({ state, status: at === -1 ? 'upcoming' : index < at ? 'done' : index === at ? 'current' : 'upcoming' }));
}
