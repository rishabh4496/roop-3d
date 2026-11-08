import React, { useState, useEffect } from 'react';
import { api, post, mediaURL } from '../api';
import VfxExportPanel, { DEFAULT_EXPORT, FinishingControls } from '../components/export/VfxExportPanel';

export default function ExtrasScreen({ active, jobs, target, exportOptions = DEFAULT_EXPORT, onExportChange = () => {}, meshBlend, relight = false, onQueue, canQueue, previewRequest, timestamp = 0 }) {
  const [capabilities, setCapabilities] = useState(null), [path, setPath] = useState(''), [options, setOptions] = useState(DEFAULT_EXPORT.finishing);
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [preview, setPreview] = useState(''), [output, setOutput] = useState(null);
  useEffect(() => { if (active) void api('/finishing/capabilities').then(setCapabilities).catch(e => setError(e.message)); }, [active]);
  const perform = async action => {
    setBusy(true); setError(''); setPreview(''); setOutput(null);
    try { const value = await post(`/finishing/${action}`, { path: path || target, finishing: options, ...(action === 'preview' ? { timestamp } : {}) }); if (action === 'preview') setPreview(value.image); else setOutput(value); }
    catch (e) { setError(e.message); } finally { setBusy(false); }
  };
  return <main className="screen-page extras-screen"><span className="eyebrow">LOCAL POST-PRODUCTION</span><h1>Media tools</h1>
    <VfxExportPanel value={exportOptions} onChange={onExportChange} capabilities={capabilities?.provenance} jobs={jobs} meshBlend={meshBlend} relight={relight} onQueue={onQueue} canQueue={canQueue} previewRequest={previewRequest} timestamp={timestamp}/>
    <section className="standalone-finishing"><h2>Standalone finishing workstation</h2><p className="muted">Grade an existing local source or rendered video, preserving its audio policy and timing.</p>
      <label>Owned media path<input aria-label="Finishing media path" value={path} placeholder={target || 'roop_3d/.runtime/media/…'} onChange={event => { setPath(event.target.value); setPreview(''); setOutput(null); }}/></label>
      <FinishingControls value={options} onChange={next => { setOptions(next); setPreview(''); setOutput(null); }} capabilities={capabilities?.provenance} disabled={busy}/>
      <div className="export-actions"><button className="button" disabled={busy || !(path || target)} onClick={() => void perform('preview')}>Preview LUT & grain</button><button className="button accent" disabled={busy || !(path || target)} onClick={() => void perform('render')}>Bake finishing to video</button></div>
      {busy && <p role="status">Processing locally…</p>}{preview && <div><img className="finishing-preview" src={preview} alt="LUT and grain finishing preview"/></div>}
      {output && <div><video className="finishing-preview" controls src={mediaURL(output.output_path)}/><div className="export-actions"><a className="button" download href={mediaURL(output.output_path)}>Download finished video</a><a className="button" download href={`/api/v1/finishing/manifest?${new URLSearchParams({ path: output.output_path })}`}>Download render manifest</a><button className="button" onClick={() => void api(`/finishing/verify?${new URLSearchParams({ path: output.output_path })}`).then(result => { if (!result.valid) setError('Output or manifest verification failed'); else setOutput({ ...output, verified: true }); }).catch(e => setError(e.message))}>Verify finished output</button></div>{output.verified && <p role="status">Output integrity verified.</p>}</div>}
      {error && <p role="alert">{error}</p>}
    </section>
  </main>;
}
