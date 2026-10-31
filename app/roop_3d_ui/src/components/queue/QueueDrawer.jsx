import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { api, post } from '../../api';

/**
 * Normalized queue state definitions.
 */
export const QUEUE_STATES = {
  QUEUED: 'QUEUED',
  ANALYZING: 'ANALYZING',
  PROCESSING: 'PROCESSING',
  COMPLETED: 'COMPLETED',
  FAILED: 'FAILED',
  CANCELLED: 'CANCELLED',
  PAUSED: 'PAUSED',
};

/**
 * Map raw backend job state to user-facing normalized state.
 * @param {string} rawState
 * @returns {string}
 */
export function normalizeJobState(rawState) {
  switch (rawState) {
    case 'queued':
      return QUEUE_STATES.QUEUED;
    case 'preparing':
      return QUEUE_STATES.ANALYZING;
    case 'running':
      return QUEUE_STATES.PROCESSING;
    case 'succeeded':
      return QUEUE_STATES.COMPLETED;
    case 'failed':
      return QUEUE_STATES.FAILED;
    case 'cancelled':
    case 'cancelling':
      return QUEUE_STATES.CANCELLED;
    case 'pausing':
    case 'paused':
    case 'interrupted':
      return QUEUE_STATES.PAUSED;
    default:
      return (rawState || QUEUE_STATES.QUEUED).toUpperCase();
  }
}

/**
 * Robust, persistent queue manager drawer matching roop-ultimate (`QueueDrawer.jsx`).
 * Decoupled from UI state, persists in local storage, supports reordering, duplication,
 * retry on failure, and FFmpeg stream copy concatenation.
 *
 * @param {{
 *   jobs?: Array<any>;
 *   selectedJobId?: string;
 *   onSelectJob?: (job: any) => void;
 *   onRefreshJobs?: () => Promise<void>;
 *   isOpen?: boolean;
 *   onToggleOpen?: () => void;
 * }} props
 */
export default function QueueDrawer({
  jobs = [],
  selectedJobId = '',
  onSelectJob,
  onRefreshJobs,
  isOpen = false,
  onToggleOpen,
}) {
  const [localJobs, setLocalJobs] = useState(jobs);
  const [selectedForConcat, setSelectedForConcat] = useState([]);
  const [isConcatenating, setIsConcatenating] = useState(false);
  const [concatResult, setConcatResult] = useState(null);
  const [actionNotice, setActionNotice] = useState('');

  // Persist jobs to localStorage so closing/refreshing tab does not lose rendering history
  useEffect(() => {
    if (jobs && jobs.length > 0) {
      setLocalJobs(jobs);
      try {
        localStorage.setItem('roop3d_queue_cache', JSON.stringify(jobs));
      } catch {
        // Ignore storage quota
      }
    } else {
      try {
        const cached = localStorage.getItem('roop3d_queue_cache');
        if (cached) {
          const parsed = JSON.parse(cached);
          if (Array.isArray(parsed) && parsed.length > 0) {
            setLocalJobs(parsed);
          }
        }
      } catch {
        // Fallback
      }
    }
  }, [jobs]);

  // Reorder jobs (Move Up / Down)
  const handleMoveJob = useCallback((index, direction) => {
    const targetIdx = index + direction;
    if (targetIdx < 0 || targetIdx >= localJobs.length) return;
    const reordered = [...localJobs];
    const temp = reordered[index];
    reordered[index] = reordered[targetIdx];
    reordered[targetIdx] = temp;
    setLocalJobs(reordered);
    try {
      localStorage.setItem('roop3d_queue_cache', JSON.stringify(reordered));
    } catch {
      // Ignore
    }
  }, [localJobs]);

  // Duplicate Job
  const handleDuplicateJob = async (job) => {
    try {
      setActionNotice(`Duplicating render for ${job.request.target_path.split('/').at(-1)}…`);
      const newRequest = { ...job.request };
      const duplicated = await post('/jobs', newRequest, {
        headers: { 'Idempotency-Key': crypto.randomUUID() },
      });
      if (onRefreshJobs) await onRefreshJobs();
      setActionNotice(`Created duplicate job ${duplicated.job_id.slice(0, 8)}.`);
    } catch (err) {
      setActionNotice(`Duplication failed: ${err.message}`);
    }
  };

  // Retry Failed Job
  const handleRetryJob = async (job) => {
    try {
      setActionNotice(`Retrying job ${job.job_id.slice(0, 8)}…`);
      await post('/jobs', job.request, {
        headers: { 'Idempotency-Key': crypto.randomUUID() },
      });
      if (onRefreshJobs) await onRefreshJobs();
      setActionNotice(`Queued retry for job.`);
    } catch (err) {
      setActionNotice(`Retry failed: ${err.message}`);
    }
  };

  // Job Controls (Pause / Resume / Cancel)
  const handleControl = async (jobId, action) => {
    try {
      await post(`/jobs/${jobId}/${action}`, {});
      if (onRefreshJobs) await onRefreshJobs();
    } catch (err) {
      setActionNotice(`Action ${action} failed: ${err.message}`);
    }
  };

  // Video Segment Concatenation without full re-encode using FFmpeg stream copy
  const handleConcatenate = async () => {
    if (selectedForConcat.length < 2) return;
    try {
      setIsConcatenating(true);
      setActionNotice(`Concatenating ${selectedForConcat.length} segments via FFmpeg stream copy…`);

      // Find output paths of selected jobs
      const segments = selectedForConcat
        .map(id => localJobs.find(j => j.job_id === id)?.output?.output_path)
        .filter(Boolean);

      if (segments.length < 2) {
        throw new Error('Selected jobs must have completed video outputs');
      }

      const res = await post('/media/concat', {
        segment_paths: segments,
        output_name: `master_concat_${Date.now()}.mp4`,
      });

      setConcatResult(res);
      setActionNotice(`Concatenated ${res.segment_count} segments losslessly with zero re-encoding!`);
      setSelectedForConcat([]);
    } catch (err) {
      setActionNotice(`Concatenation error: ${err.message}`);
    } finally {
      setIsConcatenating(false);
    }
  };

  // Summary counts
  const counts = useMemo(() => {
    const total = localJobs.length;
    const active = localJobs.filter(j => ['queued', 'preparing', 'running'].includes(j.state)).length;
    const completed = localJobs.filter(j => j.state === 'succeeded').length;
    const failed = localJobs.filter(j => j.state === 'failed').length;
    return { total, active, completed, failed };
  }, [localJobs]);

  return (
    <aside
      className={`queue-drawer-panel ${isOpen ? 'open' : 'docked'}`}
      role="region"
      aria-label="Persistent Render Queue Manager"
    >
      {/* Drawer Toggle Header */}
      <button
        type="button"
        className="queue-drawer-toggle"
        onClick={onToggleOpen}
        aria-expanded={isOpen}
      >
        <div className="toggle-left">
          <span className="eyebrow">PERSISTENT QUEUE</span>
          <strong>{counts.total} Local Jobs</strong>
          {counts.active > 0 && <span className="active-pill">{counts.active} Processing</span>}
          {counts.failed > 0 && <span className="failed-pill">{counts.failed} Failed</span>}
        </div>
        <div className="toggle-right">
          <span className="drawer-chevron">{isOpen ? '▼' : '▲'}</span>
        </div>
      </button>

      {/* Drawer Body (Visible when open) */}
      {isOpen && (
        <div className="queue-drawer-content">
          {actionNotice && (
            <div className="queue-notice-strip">
              <span>{actionNotice}</span>
              <button type="button" onClick={() => setActionNotice('')}>×</button>
            </div>
          )}

          {/* Concatenation Toolbar (Appears when 2+ completed jobs are checked) */}
          {selectedForConcat.length >= 2 && (
            <div className="concat-toolbar">
              <span>{selectedForConcat.length} video segments selected</span>
              <button
                type="button"
                className="button accent small"
                disabled={isConcatenating}
                onClick={handleConcatenate}
                title="Combine selected clips using FFmpeg stream copy (-c copy)"
              >
                {isConcatenating ? 'Concatenating…' : '⧉ Concat Selected (Stream Copy)'}
              </button>
            </div>
          )}

          {concatResult && (
            <div className="concat-success-banner">
              <span>★ Master output ready: <b>{concatResult.output_path.split('/').at(-1)}</b></span>
              <a
                className="button small primary-soft"
                href={`/api/v1/media/content?path=${encodeURIComponent(concatResult.output_path)}`}
                download
              >
                Download Master Clip ↗
              </a>
            </div>
          )}

          {/* Job Cards List */}
          <div className="queue-items-list" role="list">
            {[...localJobs].reverse().map((job, idx) => {
              const normState = normalizeJobState(job.state);
              const isSelected = selectedJobId === job.job_id;
              const hasOutput = job.state === 'succeeded' && job.output?.output_path;
              const isChecked = selectedForConcat.includes(job.job_id);

              return (
                <div
                  key={job.job_id}
                  className={`queue-job-card ${normState.toLowerCase()} ${isSelected ? 'selected' : ''}`}
                  role="listitem"
                >
                  {/* Concat Checkbox for completed videos */}
                  {hasOutput && (
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={(e) => {
                        e.stopPropagation();
                        setSelectedForConcat(prev =>
                          e.target.checked ? [...prev, job.job_id] : prev.filter(id => id !== job.job_id)
                        );
                      }}
                      title="Select for FFmpeg stream copy concatenation"
                      className="concat-check"
                    />
                  )}

                  {/* Job Reorder Buttons */}
                  <div className="reorder-btns">
                    <button
                      type="button"
                      disabled={idx === 0}
                      onClick={() => handleMoveJob(localJobs.length - 1 - idx, -1)}
                      title="Move Up in Queue"
                    >
                      ▲
                    </button>
                    <button
                      type="button"
                      disabled={idx === localJobs.length - 1}
                      onClick={() => handleMoveJob(localJobs.length - 1 - idx, 1)}
                      title="Move Down in Queue"
                    >
                      ▼
                    </button>
                  </div>

                  {/* Job Details & Selection */}
                  <div
                    className="job-info-block"
                    onClick={() => onSelectJob?.(job)}
                  >
                    <div className="job-title-row">
                      <strong>{job.request.target_path.split('/').at(-1)}</strong>
                      <span className={`state-badge ${normState.toLowerCase()}`}>
                        {normState}
                      </span>
                    </div>

                    <div className="job-meta-row">
                      <span className="job-id-tag">{job.job_id.slice(0, 8)}</span>
                      <span>
                        Frame {job.progress.frame}
                        {job.progress.total_frames ? ` / ${job.progress.total_frames}` : ''}
                      </span>
                      {job.progress.fps > 0 && <span>{job.progress.fps.toFixed(1)} fps</span>}
                    </div>

                    {/* Progress Bar */}
                    <div className="job-progress-track">
                      <div
                        className="job-progress-fill"
                        style={{
                          width: `${
                            job.progress.total_frames
                              ? (job.progress.frame / job.progress.total_frames) * 100
                              : job.state === 'succeeded' ? 100 : 0
                          }%`,
                        }}
                      />
                    </div>

                    {job.failure && <small className="failure-text">{job.failure.message}</small>}
                  </div>

                  {/* Actions (Duplicate, Retry, Cancel, Output Download) */}
                  <div className="job-actions-col">
                    <button
                      type="button"
                      className="button small"
                      onClick={() => handleDuplicateJob(job)}
                      title="Duplicate job parameters into new queued render"
                    >
                      ⧉ Duplicate
                    </button>

                    {job.state === 'failed' && (
                      <button
                        type="button"
                        className="button small accent"
                        onClick={() => handleRetryJob(job)}
                        title="Retry failed render"
                      >
                        ⟲ Retry
                      </button>
                    )}

                    {['running', 'queued'].includes(job.state) && (
                      <button
                        type="button"
                        className="button small"
                        onClick={() => handleControl(job.job_id, 'cancel')}
                        title="Cancel active job"
                      >
                        Cancel
                      </button>
                    )}

                    {hasOutput && (
                      <a
                        className="button small primary-soft"
                        href={`/api/v1/media/content?path=${encodeURIComponent(job.output.output_path)}`}
                        download
                        title="Download rendered output video"
                      >
                        Output ↗
                      </a>
                    )}
                  </div>
                </div>
              );
            })}

            {localJobs.length === 0 && (
              <div className="queue-empty-view">
                <span>⊞</span>
                <p>Render queue is empty.</p>
                <small>Map a target face in the workstation and click "Queue render" to begin.</small>
              </div>
            )}
          </div>
        </div>
      )}
    </aside>
  );
}
