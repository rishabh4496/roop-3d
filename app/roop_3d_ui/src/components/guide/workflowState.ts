import type { Job } from '../../contracts';
import { WORKFLOW_STEPS, type WorkflowStep, type WorkflowStepId } from './helpContent';

export type StepState = 'done' | 'current' | 'todo';

export interface WorkflowInputs {
  /** A project has been saved (or opened) for this work. */
  projectSaved: boolean;
  target: string;
  faceSetCount: number;
  inspected: boolean;
  /** Number of face tracks that will be swapped (track → FaceSet mappings). */
  mappingCount: number;
  /** A live preview frame has been produced at least once. */
  previewSeen: boolean;
  jobs: Pick<Job, 'state' | 'request'>[];
}

export interface WorkflowProgress {
  steps: Array<WorkflowStep & { state: StepState }>;
  /** First unfinished required step, or null when the whole workflow is done. */
  next: WorkflowStep | null;
  /** 1-based position of `next` among the six required steps (0 when everything is done). */
  nextNumber: number;
  doneCount: number;
  total: number;
}

const REQUIRED: WorkflowStepId[] = ['target', 'source', 'inspect', 'assign', 'preview', 'render'];

/**
 * Where a person is in the local video workflow, derived only from real application state. "project" is
 * recommended but never blocks the next step; the other six run in order and the first unfinished one is "current".
 */
export function workflowProgress(input: WorkflowInputs): WorkflowProgress {
  const done: Record<WorkflowStepId, boolean> = {
    project: input.projectSaved,
    target: !!input.target,
    source: input.faceSetCount > 0,
    inspect: input.inspected,
    assign: input.mappingCount > 0,
    preview: input.previewSeen,
    render: input.jobs.some(job => job.request.target_path === input.target && ['queued', 'preparing', 'running', 'pausing', 'paused', 'succeeded', 'interrupted'].includes(job.state)),
  };
  const nextId = REQUIRED.find(id => !done[id]) ?? null;
  const steps = WORKFLOW_STEPS.map(step => ({ ...step, state: (done[step.id] ? 'done' : step.id === nextId ? 'current' : 'todo') as StepState }));
  return { steps, next: nextId ? WORKFLOW_STEPS.find(step => step.id === nextId) ?? null : null, nextNumber: nextId ? REQUIRED.indexOf(nextId) + 1 : 0, doneCount: REQUIRED.filter(id => done[id]).length, total: REQUIRED.length };
}

/* ---- onboarding persistence (a per-browser convenience, never project state) ---- */
export const ONBOARDING_KEY = 'roop3d-onboarding-v1';
export function onboardingSeen(): boolean { try { return localStorage.getItem(ONBOARDING_KEY) === '1'; } catch { return true; } }
export function markOnboardingSeen(): void { try { localStorage.setItem(ONBOARDING_KEY, '1'); } catch { /* shows again next time */ } }
