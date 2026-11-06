import React, { useEffect, useState } from 'react';
import { api } from '../../api';

/**
 * Stage 9: shows where a re-run of a stopped job continues from (the last verified finished segment), if one exists.
 */
export default function ResumeNote({ job }) {
  const [info, setInfo] = useState(/** @type {any} */ (null));
  useEffect(() => {
    let live = true;
    api(`/api/v1/jobs/${job.job_id}/checkpoint`).then((value) => { if (live) setInfo(value); }, () => { if (live) setInfo(null); });
    return () => { live = false; };
  }, [job.job_id, job.state]);
  if (!info?.available || !info.completed_segments) return null;
  return (
    <small className="job-hint muted" data-testid="resume-note">
      Resumes at frame {info.resume_frame}{info.total_frames ? ` of ${info.total_frames}` : ''} after {info.completed_segments} finished segment{info.completed_segments === 1 ? '' : 's'}; they are not rendered again.
    </small>
  );
}
