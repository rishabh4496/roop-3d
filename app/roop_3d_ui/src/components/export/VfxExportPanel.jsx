import React, { useState } from 'react';
import { api, post, mediaURL } from '../../api';
import { downloadText, exportFCPXML, exportEDL, exportOTIO } from '../../utils/nleExporter';
import { buildZip } from '../facemgr/fszBuilder';
import NumericInput from '../controls/NumericInput';

export const DEFAULT_EXPORT = { baked_video: true, passes: [], finishing: { cube: null, grain: { size: 1, intensity: 0, shadows: 1, highlights: .5, seed: 0 }, provenance: 'none' } };
const PASSES = [['alpha', 'Face-Canvas Alpha Matte'], ['foreground', 'Swapped-Face-Only Foreground (RGBA)'], ['occluder', 'Occluder / XSeg Matte Pass'], ['normal', '3D Normal Map Pass'], ['depth', '3D Depth Map Pass'], ['uv', 'UV Warped Face Pass']];

export function FinishingControls({ value, onChange, capabilities = null, disabled = false }) {
  const [error, setError] = useState(''), [lutName, setLutName] = useState('');
  const changeGrain = (name, next) => onChange({ ...value, grain: { ...value.grain, [name]: next } });
  const loadLut = async file => {
    if (!file) return;
    setError('');
    if (!file.name.toLowerCase().endsWith('.cube') || file.size > 1048576) { setError('Choose a .cube file up to 1 MiB (3D sizes 2–33).'); return; }
    const text = await file.text();
    if (!/^\s*LUT_3D_SIZE\s+(?:[2-9]|[12]\d|3[0-3])\s*(?:#.*)?$/m.test(text)) { setError('Expected a 3D CUBE LUT, size 2–33.'); return; }
    onChange({ ...value, cube: text }); setLutName(file.name);
  };
  return <fieldset disabled={disabled} className="finishing-controls"><legend>Finishing pipeline</legend>
    <label>3D LUT (.cube)<input type="file" accept=".cube" onChange={event => { void loadLut(event.target.files?.[0]).catch(e => setError(e.message)); event.target.value = ''; }}/></label>
    <p className="muted">{value.cube ? (lutName || 'Saved LUT loaded') : 'No LUT loaded'} · Display-referred RGB. Select a LUT matching your footage’s color space.</p>
    {value.cube && <button type="button" className="button" onClick={() => { onChange({ ...value, cube: null }); setLutName(''); }}>Remove LUT</button>}
    <div className="finishing-grid">{[['size', 'Grain Size', .5, 8, .1], ['intensity', 'Grain Intensity', 0, .2, .001], ['shadows', 'Shadow balance', 0, 2, .05], ['highlights', 'Highlight balance', 0, 2, .05], ['seed', 'Grain seed', 0, 2147483647, 1]].map(([name, label, min, max, step]) =>
      <NumericInput key={name} label={label} min={min} max={max} step={step} integer={name === 'seed'} disabled={disabled} value={value.grain[name]} onChange={n => { if (n !== null) changeGrain(name, n); }}/>)}</div>
    <label>Provenance<select aria-label="Provenance" value={value.provenance} onChange={event => onChange({ ...value, provenance: event.target.value })}>
      <option value="none">Standard synthetic-media notice</option><option value="sha256">Embedded recipe hash + SHA-256 manifest</option>
      <option value="hmac-sha256" disabled={!capabilities?.['hmac-sha256']}>Keyed cryptographic manifest (local key required)</option>
      <option value="c2pa" disabled={!capabilities?.c2pa}>C2PA signature (local SDK + ES256 credentials required)</option>
    </select></label>
    <p className="muted">LUT → grain → video encoding → optional signing. Data passes retain their original values. SHA-256 checks integrity; keyed manifests authenticate with your local secret.</p>
    {error && <p role="alert">{error}</p>}
  </fieldset>;
}

export default function VfxExportPanel({ value = DEFAULT_EXPORT, onChange, capabilities = null, jobs = [], meshBlend = false, disabled = false, onQueue, canQueue = false, previewRequest = null, timestamp = 0 }) {
  const [selected, setSelected] = useState(''), [busy, setBusy] = useState(false), [error, setError] = useState(''), [notice, setNotice] = useState('');
  const completed = jobs.filter(job => job.state === 'succeeded' && job.output);
  const job = completed.find(item => item.job_id === selected);
  const perform = async work => { setBusy(true); setError(''); setNotice(''); try { await work(); } catch (e) { setError(e.message); } finally { setBusy(false); } };
  const interchange = format => perform(async () => {
    if (format === 'xml') {
      const xmlData = await api(`/exports/${selected}/timeline?format=xml`);
      downloadText('roop3d.xml', typeof xmlData === 'string' ? xmlData : xmlData.xml || JSON.stringify(xmlData), 'application/xml');
      setNotice('Timeline exported as Premiere/FCP XML (.xml). Relink local media if moved.');
      return;
    }
    const plan = await api(`/exports/${selected}/timeline`);
    if (format === 'fcpxml') downloadText('roop3d.fcpxml', exportFCPXML(plan), 'application/xml');
    else if (format === 'otio') downloadText('roop3d.otio', exportOTIO(plan), 'application/json');
    else {
      const { downloadPass } = await import('../../utils/meshPassExporter');
      downloadPass('roop3d_edl_tracks.zip', buildZip(Object.entries(exportEDL(plan)).map(([name, data]) => ({ name, data }))));
    }
    setNotice('Timeline exported. Relink local media if moved; assign V3 as the V2 track matte in your editor.');
  });
  const togglePass = name => onChange({ ...value, passes: value.passes.includes(name) ? value.passes.filter(item => item !== name) : [...value.passes, name] });
  const valid = value.baked_video || value.passes.length > 0;
  return <section className="vfx-export-panel" aria-label="VFX export and finishing">
    <h2>VFX compositing exports</h2>
    <fieldset disabled={disabled || busy}><legend>Render deliverables</legend>
      <label className="export-check"><input type="checkbox" checked={value.baked_video} onChange={event => onChange({ ...value, baked_video: event.target.checked, finishing: event.target.checked ? value.finishing : DEFAULT_EXPORT.finishing })}/>Final baked video</label>
      {PASSES.map(([id, label]) => <label className="export-check" key={id}><input type="checkbox" checked={value.passes.includes(id)} disabled={['normal', 'depth'].includes(id) && !meshBlend} onChange={() => togglePass(id)}/>{label}</label>)}
      <p className="muted">16-bit PNG sequences + lossless FFV1 video. Face RGB is unblended, with straight alpha supplied separately. Occluders use available semantic/manual masks; XSeg inference is not configured.</p>
      <p className="muted">Normal/depth sequences require mesh blend mode. Geometry and camera are estimated; depth uses canonical camera units. Preserve source timing for pass exports.</p>
      {!valid && <p role="alert">Choose a baked video or at least one pass.</p>}
    </fieldset>
    <FinishingControls value={value.finishing} onChange={finishing => onChange({ ...value, finishing })} capabilities={capabilities} disabled={disabled || !value.baked_video || busy}/>
    {onQueue && <button className="button accent" disabled={disabled || busy || !valid || !canQueue || (value.passes.some(name => ['normal', 'depth'].includes(name)) && !meshBlend)} onClick={onQueue}>Queue export render</button>}
    <button className="button" disabled={disabled || busy || !previewRequest || !meshBlend} onClick={() => void perform(async () => {
      const frame = await post('/workstation/preview', { request: { ...previewRequest, export: { ...value, baked_video: true, passes: ['normal', 'depth'] } }, timestamp });
      const { renderAlignedMeshPasses, passPFM, downloadPass } = await import('../../utils/meshPassExporter');
      const passes = renderAlignedMeshPasses(frame);
      downloadPass(`normal_${frame.frame_index}.pfm`, passPFM(frame.width, frame.height, passes.normal));
      downloadPass(`depth_${frame.frame_index}.pfm`, passPFM(frame.width, frame.height, passes.depth, 1));
      downloadText(`geometry_${frame.frame_index}.json`, JSON.stringify({ ...frame, buffers: undefined, normal: 'signed camera-space XYZ; black background', depth: 'linear estimated camera units; black background', renderer: 'Three.js floating point target; geometric surface only, use face alpha for crop coverage' }, null, 2), 'application/json');
      setNotice(`Three.js normal/depth passes exported for source frame ${frame.frame_index}.`);
    })}>Export current-frame Three.js normal & depth</button>
    <p><a href="#/faceswap">Open workstation canvas to preview finishing</a></p>
    <h2>NLE timeline interchange</h2>
    <label>Completed render<select aria-label="Export render" value={selected} onChange={event => { setSelected(event.target.value); setNotice(''); setError(''); }}><option value="">Select a completed render</option>{completed.map(item => <option key={item.job_id} value={item.job_id}>{item.request.target_path.split(/[\\/]/).at(-1)} · {item.job_id.slice(0, 8)}</option>)}</select></label>
    <div className="export-actions">{[['fcpxml', 'Final Cut Pro XML (.fcpxml)'], ['xml', 'Premiere / FCP 7 XML (.xml)'], ['edl', 'CMX 3600 EDL (.edl)'], ['otio', 'OpenTimelineIO (.otio)']].map(([id, label]) => <button key={id} className="button" disabled={!job || busy} onClick={() => void interchange(id)}>{label}</button>)}</div>
    <p className="muted">V1 source · V2 actual swapped ranges · V3 face alpha. Scene cuts and range boundaries create separate clips. EDL exports one file per track. CFR media is required; missing passes are omitted.</p>
    {job && <div className="export-actions"><a className="button" href={`/api/v1/exports/${job.job_id}/manifest`} download>Render manifest</a>{job.output.pass_manifest_path && <a className="button" href={`/api/v1/exports/${job.job_id}/manifest?passes=true`} download>Pass manifest & sequence paths</a>}{Object.entries(job.output.passes || {}).map(([name, path]) => <a className="button" key={name} href={mediaURL(path)} download>{name} pass</a>)}<button className="button" disabled={busy} onClick={() => void perform(async () => { const result = await api(`/exports/${job.job_id}/verify`); setNotice(JSON.stringify(result)); })}>Verify output integrity</button></div>}
    {notice && <p role="status">{notice}</p>}{error && <p role="alert">{error}</p>}
  </section>;
}
