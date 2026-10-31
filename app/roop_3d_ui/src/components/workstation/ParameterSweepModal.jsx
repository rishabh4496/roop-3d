import React, { useEffect, useRef, useState } from 'react';
import Modal from '../controls/Modal';
import { post } from '../../api';
import { SWEEP_PARAMETERS, sweepAxis } from './pipeline';

/** @param {any} props */
export default function ParameterSweepModal({ request, timestamp, parameters, onApply, onClose, restorerAvailable = false }) {
  const [x, setX] = useState('color_strength'), [y, setY] = useState('feather_px');
  const [xText, setXText] = useState('0.25, 0.50, 0.75'), [yText, setYText] = useState('0, 5, 10');
  const [result, setResult] = useState(null), [busy, setBusy] = useState(false), [error, setError] = useState('');
  const abort = useRef(null), generation = useRef(0);
  const context = JSON.stringify({ request, timestamp, parameters, x, y, xText, yText });
  useEffect(() => { generation.current++; abort.current?.abort(); setBusy(false); setResult(null); setError(''); return () => { generation.current++; abort.current?.abort(); }; }, [context]);
  const generate = async () => {
    const token = ++generation.current;
    try {
      if (!request) throw new Error('Inspect and map a target face before generating a sweep');
      if (x === y) throw new Error('Choose two different parameters');
      const axes = { x: sweepAxis(x, xText), y: sweepAxis(y, yText) };
      const controller = new AbortController(); abort.current = controller;
      setBusy(true); setResult(null); setError('');
      const bypass = parameters.bypass.filter(id => !([x, y].includes('restoration_blend') && id === 'restorer') && !([x, y].includes('color_strength') && id === 'color'));
      // Exactly one batched request for all nine coordinates at the current playhead.
      const response = await post('/workstation/preview', { request, timestamp, parameters: { ...parameters, bypass }, ...axes }, { signal: controller.signal });
      if (token === generation.current) setResult(response);
    } catch (e) { if (token === generation.current && e.name !== 'AbortError') setError(e.message); }
    finally { if (token === generation.current) setBusy(false); }
  };
  const axis = (label, value, change, text, setText) => <div className="sweep-axis"><label>{label}<select aria-label={`${label} parameter`} value={value} onChange={e => { change(e.target.value); setText(SWEEP_PARAMETERS[e.target.value].values.join(', ')); }}>{Object.entries(SWEEP_PARAMETERS).map(([id, field]) => <option key={id} value={id} disabled={id === 'restoration_blend' && !restorerAvailable || id === 'pose_strength' && parameters.pose_mode !== 'mesh_blend'}>{field.label}{id === 'restoration_blend' && !restorerAvailable ? ' · GPEN weights unavailable' : ''}{id === 'pose_strength' && parameters.pose_mode !== 'mesh_blend' ? ' · select mesh blend mode' : ''}</option>)}</select></label><label>Three coordinates<input aria-label={`${label} values`} value={text} onChange={e => setText(e.target.value)} /></label></div>;
  return <Modal title="3×3 Parameter Sweep" close={onClose}><p className="muted">Current playhead: {timestamp.toFixed(3)} s · one local batch · up to 1920 px per cell. Selecting a cell applies its exact settings. Swept stages are enabled for comparison.</p>
    <div className="sweep-axes">{axis('X axis', x, setX, xText, setXText)}{axis('Y axis', y, setY, yText, setYText)}</div>
    <button className="button accent" disabled={busy || !request} onClick={generate}>{busy ? 'Rendering nine previews…' : 'Generate contact sheet'}</button>{busy && <button className="button" onClick={() => { generation.current++; abort.current?.abort(); setBusy(false); }}>Cancel sweep</button>}
    {error && <p role="alert" className="inline-warning">{error}</p>}
    {result && <><p className="muted">Decoded frame {result.frame_index} at {result.timestamp.toFixed(3)} s · X: {SWEEP_PARAMETERS[x].label} → · Y: {SWEEP_PARAMETERS[y].label} ↓</p><div className="sweep-grid">{result.cells.map(cell => <button className="sweep-cell" key={`${cell.row}-${cell.column}`} onClick={() => { onApply(cell.parameters); onClose(); }} aria-label={`Apply ${x} ${cell.coordinates[x]}, ${y} ${cell.coordinates[y]}`}><img src={cell.image} alt={`${SWEEP_PARAMETERS[x].label}: ${cell.coordinates[x]}, ${SWEEP_PARAMETERS[y].label}: ${cell.coordinates[y]}`} /><span>X {cell.coordinates[x]} · Y {cell.coordinates[y]}</span></button>)}</div>{result.notes.map(note => <p className="muted" key={note}>{note}</p>)}</>}
  </Modal>;
}
