import React, { useState } from 'react';
import useHistoryStore from '../../stores/useHistoryStore';

/** @param {any} props */
export default function SnapshotSlots({ onCapture, onRestore, busy = false, captureAvailable = false }) {
  const history = useHistoryStore(), [left, setLeft] = useState('A'), [right, setRight] = useState('B');
  return <div className="snapshot-toolbar" aria-label="A B C D snapshots"><strong>Snapshots</strong>{Object.entries(history.slots).map(([slot, snapshot]) => <button className={`button small ${history.previewSlot === slot ? 'primary-soft' : ''}`} key={slot} disabled={busy || !snapshot && !captureAvailable} aria-label={`${snapshot ? 'View' : 'Capture'} snapshot ${slot}`} aria-pressed={history.previewSlot === slot} title={snapshot ? `Frame ${snapshot.frame} · Shift-click to overwrite` : 'Capture rendered frame and full workspace state'} onClick={event => { if (!snapshot || event.shiftKey) onCapture(slot); else history.selectSlot(slot); }}>{slot}{snapshot ? ' ●' : ' +'}</button>)}
    <label>Left<select aria-label="Left snapshot" value={left} onChange={e => setLeft(e.target.value)}>{Object.keys(history.slots).map(slot => <option key={slot} disabled={!history.slots[slot]}>{slot}</option>)}</select></label><label>Right<select aria-label="Right snapshot" value={right} onChange={e => setRight(e.target.value)}>{Object.keys(history.slots).map(slot => <option key={slot} disabled={!history.slots[slot]}>{slot}</option>)}</select></label><button className="button small" disabled={!history.slots[left] || !history.slots[right] || left === right} onClick={() => history.compare(left, right)}>Split comparison wipe</button>
    <button className="button small" disabled={!history.previewSlot} onClick={() => onRestore(history.slots[history.previewSlot].workspace)}>Restore slot settings</button><button className="button small" onClick={history.live}>Live viewport</button>{busy && <span role="status">Capturing rendered frame…</span>}
  </div>;
}
export function SnapshotViewport() {
  const { slots, previewSlot, comparison } = useHistoryStore(), [split, setSplit] = useState(50);
  if (!comparison && !previewSlot) return null;
  const left = comparison?.[0] ?? previewSlot, right = comparison?.[1];
  return <div className="snapshot-viewport" aria-label={right ? `Snapshot ${left} versus ${right}` : `Snapshot ${left}`}>
    <img src={slots[left].image} alt={`Snapshot ${left}, frame ${slots[left].frame}`}/>{right && <img src={slots[right].image} alt={`Snapshot ${right}, frame ${slots[right].frame}`} style={{ clipPath: `inset(0 0 0 ${split}%)` }}/>}<span className="pane-label">{left} · frame {slots[left].frame}{right && ` / ${right} · frame ${slots[right].frame}`}</span>{right && <><div className="snapshot-wipe-line" style={{ left: `${split}%` }}/><input className="snapshot-wipe-control" aria-label="Snapshot split comparison wipe" type="range" min={0} max={100} value={split} onChange={e => setSplit(Number(e.target.value))}/></>}
  </div>;
}
