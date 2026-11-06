import BuilderCoverageSphere from './BuilderCoverageSphere';
import { useEffect, useRef, useState } from 'react';
import { api, post, upload } from '../../api';
import type { Capabilities } from '../../contracts';
import DialogPanel from '../guide/DialogPanel';
import './FacesetBuilder.css';

type Sample = { sample_id: string; bin_index: number; yaw: number; pitch: number; roll: number; score: number; thumbnail_base64: string; occlusion_tier: string; eligible?: boolean };
type Cluster = { rank: number; detections: number; coverage: number; samples: Sample[]; selected_ids: string[] };
type Draft = { job_id: string; status: string; progress: number; scanned: number; error: string | null; clusters: Cluster[] };
export default function FacesetBuilder({ sourceId, onSaved }: { sourceId?: string; onSaved: (id: string) => void | Promise<void> }) {
  const [open, setOpen] = useState(false), [draft, setDraft] = useState<Draft | null>(null);
  const [rank, setRank] = useState(0), [selected, setSelected] = useState<string[]>([]);
  const [name, setName] = useState('New multi-angle faceset'), [path, setPath] = useState('');
  const [editing, setEditing] = useState(false), [busy, setBusy] = useState(false), [error, setError] = useState('');
  const [bin, setBin] = useState<number | null>(null), [uploadProgress, setUploadProgress] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const cluster = draft?.clusters.find(c => c.rank === rank);
  const running = draft?.status === 'extracting';
  const jobId = draft?.job_id;
  useEffect(() => {
    if (!running || !jobId) return;
    let active = true;
    const timer = setInterval(() => { void api<Draft>(`/faceset/extraction/${jobId}`).then(value => {
      if (!active) return;
      setDraft(value);
      if (value.status === 'ready') { const first = value.clusters[0]; setRank(first?.rank ?? 0); setSelected(first?.selected_ids ?? []); }
    }).catch(e => { if (active) { setError(e.message); setDraft(previous => previous ? { ...previous, status: "error" } : previous); clearInterval(timer); } }); }, 700);
    return () => { active = false; clearInterval(timer); };
  }, [running, jobId]);
  async function extract(paths: string[], edit = editing) {
    setError(''); setBusy(true);
    try {
      const value = await post<Draft>('/faceset/extract-from-media', { image_paths: paths, faceset_id: edit ? sourceId : undefined });
      setDraft(value); setEditing(edit); setRank(value.clusters[0]?.rank ?? 0); setSelected(value.status === "ready" ? value.clusters[0]?.selected_ids ?? [] : []); setBin(null);
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  async function files(values: File[]) {
    if (!values.length) return;
    setBusy(true); setError(''); setUploadProgress(0);
    try {
      const caps = await api<Capabilities>('/system/capabilities'); const paths: string[] = [];
      for (let i = 0; i < values.length; i++) {
        const result = await upload(values[i], caps, p => setUploadProgress((i + p) / values.length), new AbortController().signal);
        if (!result.path) throw new Error('Choose reference photos or a video.');
        paths.push(result.path);
      }
      await extract(paths);
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  async function save() {
    setBusy(true); setError('');
    try {
      const value = await post<{ faceset_id: string }>('/faceset/save-fsz', { job_id: draft?.job_id, cluster_rank: rank, selected_ids: selected, name, faceset_id: editing ? sourceId : undefined });
      await onSaved(value.faceset_id); setOpen(false); setDraft(null);
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  const toggle = (sample: Sample) => setSelected(previous => previous.includes(sample.sample_id) ? previous.filter(id => id !== sample.sample_id) : [...previous.filter(id => cluster?.samples.find(s => s.sample_id === id)?.bin_index !== sample.bin_index), sample.sample_id]);
  const covered = new Set(cluster?.samples.filter(s => selected.includes(s.sample_id)).map(s => s.bin_index));
  return <><button type="button" className="button small" onClick={() => setOpen(true)}>Build / Edit .fsz Faceset</button>{open && <div className="fsb-backdrop"><DialogPanel label="Multi-angle Faceset Builder" className="fsb-panel" onClose={() => { if (!busy) setOpen(false); }}>
    <header><h2>Multi-angle .fsz Faceset Builder</h2><button className="button" disabled={busy} onClick={() => setOpen(false)}>Close</button></header>
    <label>Faceset name<input value={name} onChange={e => setName(e.target.value)} /></label>
    <div className="fsb-drop" onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); if (!busy && !running) void files(Array.from(e.dataTransfer.files)); }}>
      <p>Drop a reference video or photo batch to extract head angles</p><button className="button" disabled={busy || running} onClick={() => input.current?.click()}>Choose video / photos</button>
      <input ref={input} hidden type="file" multiple accept="image/*,video/*" onChange={e => void files(Array.from(e.target.files ?? []))}/>
    </div>
    <label>Local video or image directory inside this checkout<input value={path} onChange={e => setPath(e.target.value)} placeholder="roop_3d/.runtime/media/reference.mp4" /></label>
    <button className="button" disabled={busy || running || !path} onClick={() => void extract([path])}>Extract local media</button>
    {sourceId && <button className="button" disabled={busy || running} onClick={() => void extract([], true)}>Edit active faceset / add angles</button>}
    {(busy || running) && <div role="status"><progress max={1} value={running ? draft.progress : uploadProgress}/><span>{running ? `Scanning: ${draft.scanned} sampled frames` : 'Uploading / saving faceset…'}</span></div>}
    {(error || draft?.error) && <p role="alert">{error || draft?.error}</p>}
    {!!draft?.clusters.length && <><label>Person cluster<select value={rank} disabled={running} onChange={e => { const r = Number(e.target.value); setRank(r); setSelected(draft.clusters.find(c => c.rank === r)?.selected_ids ?? []); }}>
      {draft.clusters.map(c => <option key={c.rank} value={c.rank}>Person {c.rank + 1} · {c.detections} detections · {c.coverage}/96 angles</option>)}
    </select></label><h3>Sphere Coverage: {covered.size} / 96</h3><p>Choose a bin to inspect its crops. Empty bins show missing angles. Only visible, clear samples are selected automatically.</p>
    <div className="fsb-maps"><BuilderCoverageSphere covered={covered} selectedBin={bin} onSelect={setBin}/><div className="fsb-grid" aria-label="96 pose bins">{Array.from({ length: 96 }, (_, i) => <button key={i} className={covered.has(i) ? 'covered' : ''} aria-pressed={bin === i} aria-label={`Bin ${i + 1}, yaw ${-82.5 + i % 12 * 15}, pitch ${-52.5 + Math.floor(i / 12) * 15}, ${covered.has(i) ? 'covered' : 'missing'}`} onClick={() => setBin(bin === i ? null : i)}>{i + 1}</button>)}</div></div>
    <button className="button" onClick={() => setBin(null)}>Show all angles</button>
    {['Frontal', 'Left Profile', 'Right Profile', 'Pitch Up', 'Pitch Down'].map(group => {
      const samples = cluster?.samples.filter(s => (bin === null || s.bin_index === bin) && (Math.abs(s.pitch) >= 20 ? (s.pitch > 0 ? 'Pitch Up' : 'Pitch Down') : s.yaw < -25 ? 'Left Profile' : s.yaw > 25 ? 'Right Profile' : 'Frontal') === group) ?? [];
      return <section key={group}><h4>{group} ({samples.length})</h4><div className="fsb-gallery">{samples.map(s => <button key={s.sample_id} disabled={running || busy} aria-pressed={selected.includes(s.sample_id)} onClick={() => toggle(s)}><img src={`data:image/jpeg;base64,${s.thumbnail_base64}`} alt={`${group} crop`}/><span>{s.yaw.toFixed(0)}° / {s.pitch.toFixed(0)}° · quality {s.score.toFixed(2)}</span><small>{s.occlusion_tier}{s.eligible === false ? ' · review occlusion' : ''}</small><strong>{selected.includes(s.sample_id) ? 'Selected — remove' : 'Add angle'}</strong></button>)}</div></section>;
    })}
    <button className="button primary" disabled={busy || running || !selected.length || !name.trim()} onClick={() => void save()}>Save &amp; Activate .fsz Faceset</button></>}
  </DialogPanel></div>}</>;
}
