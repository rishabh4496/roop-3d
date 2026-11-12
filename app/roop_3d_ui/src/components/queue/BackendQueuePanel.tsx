import { useCallback, useEffect, useRef, useState } from 'react';
import { api, post } from '../../api';
import { DragItems } from '../ui-layouts/DragItems';
import { TimelineAnimation } from '../ui-layouts/TimelineAnimation';
import { cn } from '../../lib/utils';
import {
  LIFECYCLE_LABEL, failedSlices, isReorderable, lifecycleSteps, progressPercent, queueActions, reorderPayload, type QueueAction, type QueueJob,
} from './backendQueue';

export interface BackendQueuePanelProps {
  /** Poll only while the screen is showing. */
  active?: boolean;
  /** Jobs to show instead of asking the server (tests). */
  jobs?: QueueJob[];
  pollMs?: number;
}

const ACTION_LABEL: Record<QueueAction, string> = { pause: 'Pause', resume: 'Resume', cancel: 'Cancel', duplicate: 'Duplicate', join: 'Join clips' };

/**
 * The persistent, backend-owned queue (survives a server or machine restart): lifecycle state per job, drag-to-reorder for jobs that have
 * not started, and cancel / pause / resume / duplicate / retry-slice / join controls. Reordering is saved when the job is dropped.
 */
export default function BackendQueuePanel({ active = true, jobs: injected, pollMs = 4000 }: BackendQueuePanelProps) {
  const [jobs, setJobs] = useState<QueueJob[]>(injected ?? []);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState('');
  const [loaded, setLoaded] = useState(!!injected);
  const dragging = useRef(false);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const next = await api<QueueJob[]>('/queue', { signal });
      if (!dragging.current) setJobs([...next].sort((a, b) => a.queue_order - b.queue_order));
      setError(''); setLoaded(true);
    } catch (reason) {
      if ((reason as Error).name !== 'AbortError') { setError((reason as Error).message); setLoaded(true); }
    }
  }, []);

  useEffect(() => {
    if (injected || !active) return undefined;
    const controller = new AbortController();
    void refresh(controller.signal);
    const timer = window.setInterval(() => void refresh(controller.signal), pollMs);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [active, injected, pollMs, refresh]);

  const act = async (job: QueueJob, action: QueueAction) => {
    setBusy(`${job.job_id}:${action}`); setError('');
    try { await post(`/queue/${job.job_id}/${action}`, {}); await refresh(); } catch (reason) { setError((reason as Error).message); } finally { setBusy(''); }
  };
  const retry = async (job: QueueJob, sliceId: string) => {
    setBusy(`${job.job_id}:retry`); setError('');
    try { await post(`/queue/${job.job_id}/retry`, { slice_index_or_id: sliceId }); await refresh(); } catch (reason) { setError((reason as Error).message); } finally { setBusy(''); }
  };
  const save = async (after: QueueJob[]) => {
    const ids = reorderPayload(jobs, after);
    try { await post('/queue/reorder', { ordered_job_ids: ids }); await refresh(); } catch (reason) { setError((reason as Error).message); await refresh(); }
  };

  return (
    <section className="backend-queue" aria-label="Persistent job queue" data-testid="backend-queue">
      <header className="backend-queue__head">
        <h2>Persistent queue</h2>
        <span className="muted">{jobs.length} {jobs.length === 1 ? 'job' : 'jobs'} · saved on disk, resumes after a restart</span>
        {!injected && <button type="button" className="button small" onClick={() => void refresh()}>Refresh</button>}
      </header>
      {error && <p role="alert" className="inline-warning">{error}</p>}
      {loaded && !jobs.length && !error && <p className="muted empty-note">No batch or queued jobs yet. Jobs from the Batch Matrix appear here.</p>}
      <DragItems
        items={jobs}
        getKey={job => job.job_id}
        getLabel={job => job.name || job.job_id.slice(0, 8)}
        label="Queued jobs. Drag a job that has not started to change the order."
        onReorder={next => { dragging.current = true; setJobs(next); }}
        onCommit={next => { dragging.current = false; void save(next); }}
        renderItem={(job, { controls }) => {
          const steps = lifecycleSteps(job.lifecycle_state);
          const failed = failedSlices(job);
          return (
            <article className={cn('queue-card', `is-${job.lifecycle_state}`)} data-lifecycle={job.lifecycle_state}>
              <div className="queue-card__top">
                {isReorderable(job) ? controls : <span className="ul-drag__controls ul-drag__controls--locked" title="Only queued or paused jobs can be re-ordered" aria-hidden>🔒</span>}
                <strong className="queue-card__name" title={job.name}>{job.name || job.job_id.slice(0, 8)}</strong>
                <span className={cn('queue-chip', `queue-chip--${job.lifecycle_state}`)} data-testid="lifecycle-chip">{LIFECYCLE_LABEL[job.lifecycle_state]}</span>
              </div>
              <ol className="queue-steps" aria-label="Lifecycle">
                {steps.map(({ state, status }, index) => (
                  <TimelineAnimation as="li" key={state} animationNum={index} className={cn('queue-step', `is-${status}`)} aria-current={status === 'current' ? 'step' : undefined}>
                    <span className="queue-step__dot" aria-hidden />{LIFECYCLE_LABEL[state]}
                  </TimelineAnimation>
                ))}
              </ol>
              <progress value={progressPercent(job)} max={100} aria-label={`${job.name || 'Job'} progress`} />
              <small className="muted">{job.progress.frame} / {job.progress.total_frames ?? '?'} frames{job.slices.length ? ` · ${job.slices.filter(slice => slice.status === 'COMPLETED').length}/${job.slices.length} segments` : ''}</small>
              {job.progress.advisory && <p className="advisory-badge">{job.progress.advisory}</p>}
              {job.error_message && <p className="inline-warning">{job.error_message}</p>}
              <div className="queue-card__actions">
                {queueActions(job).map(action => (
                  <button key={action} type="button" className="button small" disabled={!!busy} onClick={() => void act(job, action)}>{ACTION_LABEL[action]}</button>
                ))}
                {failed.map(slice => (
                  <button key={slice.slice_id} type="button" className="button small" disabled={!!busy} onClick={() => void retry(job, slice.slice_id)} title={slice.error ?? 'This segment failed'}>Retry segment {slice.slice_index}</button>
                ))}
              </div>
            </article>
          );
        }}
      />
    </section>
  );
}
