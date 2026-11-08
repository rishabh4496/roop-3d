import { useEffect, useRef, useState } from 'react';
import { api, post } from '../../api';

type SynthesisJob = { job_id: string; status: string; progress: number; scanned: number; phase?: string; error?: string; result?: { faceset_id: string } };

/** Direct action for the measured photos already imported in #/facemgr. */
export default function PhotoSynthesisAction({ images, name, onSaved }: { images: string[]; name: string; onSaved: (id: string) => void | Promise<void> }) {
  const [job, setJob] = useState<SynthesisJob | null>(null), [error, setError] = useState(''), [starting, setStarting] = useState(false);
  const saved = useRef(onSaved);
  useEffect(() => { saved.current = onSaved; }, [onSaved]);
  const running = job?.status === 'synthesizing';
  const jobId = job?.job_id;
  useEffect(() => {
    if (!running || !jobId) return;
    let active = true, polling = false;
    const timer = setInterval(() => {
      if (polling) return;
      polling = true;
      void api<SynthesisJob>(`/faceset/extraction/${jobId}`).then(async value => {
        if (!active) return;
        if (value.status === 'ready' && value.result) await saved.current(value.result.faceset_id);
        if (active) setJob(value);
      }).catch(e => { if (active) { setError(e.message); setJob(previous => previous ? { ...previous, status: 'error' } : null); } })
        .finally(() => { polling = false; });
    }, 700);
    return () => { active = false; clearInterval(timer); };
  }, [running, jobId]);
  if ((!images.length || images.length > 3) && !running) return null;
  return <div className="photo-synthesis-action">
    <button className="button accent" disabled={starting || running} onClick={() => {
      setStarting(true); setError('');
      void post<SynthesisJob>('/faceset/synthesize-photos', { images, name }).then(setJob).catch(e => setError(e.message)).finally(() => setStarting(false));
    }}>Synthesize Full 36-Angle 3D Faceset (.fsz)</button>
    {running && <span role="status"><progress max={1} value={job.progress}/> {job.phase ?? 'Synthesizing'} · {job.scanned} / 36 angles</span>}
    {(error || job?.error) && <p role="alert">{error || job?.error}</p>}
    {job?.status === 'ready' && <span role="status">36-angle faceset saved and activated.</span>}
    <small>33 views on the requested 15° grid, plus three frontal views at (yaw, pitch): (0°, −15°), (0°, +15°), (+7.5°, 0°). Includes UltraMax, ArcFace scores and UV textures.</small>
  </div>;
}
