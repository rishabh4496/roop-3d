import React from 'react';
import { PIPELINE_NODES } from './pipeline';
import { useHotkeys } from '../controls/Hotkeys';

/** @param {any} props */
export default function PipelineNodeFlowBar({ selected, solo, bypass = [], onSelect, onSolo, onBypass, busy = false, available = true }) {
  const { keys } = useHotkeys();
  return <div className="pipeline-flow" role="region" aria-label="Visual execution pipeline">{PIPELINE_NODES.map((node, index) => <React.Fragment key={node.id}>
    {index > 0 && <span className="pipeline-arrow" aria-hidden="true">→</span>}
    <div className={`pipeline-node ${selected === node.id ? 'selected' : ''} ${solo === node.id ? 'solo' : ''} ${bypass.includes(node.id) ? 'bypassed' : ''}`}>
      <button className="node-select" aria-pressed={selected === node.id} onClick={() => onSelect(node.id)} title={node.unavailable ?? node.note ?? node.label}><small>{String(index + 1).padStart(2, '0')}</small>{node.label}</button>
      <div className="node-actions"><button className="button small" aria-label={`Solo ${node.label}`} aria-pressed={solo === node.id} disabled={!available || !!node.unavailable || busy} title={node.unavailable ?? `${node.note ?? 'Inspect real stage buffer'} · ${keys.solo}`} onClick={() => { onSelect(node.id); onSolo(node.id); }}>Solo ({keys.solo})</button>
        {node.optional && <button className="button small" aria-label={`Bypass ${node.label}`} aria-pressed={bypass.includes(node.id)} disabled={!!node.unavailable || busy} title={node.unavailable ?? `Bypass · ${keys.bypass}`} onClick={() => { onSelect(node.id); onBypass(node.id); }}>B ({keys.bypass})</button>}</div>
    </div></React.Fragment>)}</div>;
}
