import React, { useMemo, useRef } from 'react';
import { AnimatedBeam } from '../ui-layouts';
import { PIPELINE_NODES, PIPELINE_ALT_NODES, formatMs, stageLatency } from './pipeline';
import { useHotkeys } from '../controls/Hotkeys';

/**
 * Visual execution pipeline: eight stage cards in execution order, each a single click (or hotkey `1`-`8`) that solos that
 * stage's real GPU/CPU buffer on the canvas; click it again to return to the composite. Each card shows the stage latency the
 * preview measured, or an em dash when the engine does not time that stage (nothing is estimated).
 * A beam rail (ui-layouts AnimatedBeam) joins each card to the next along the top edge; light travels down it only while a preview
 * frame is rendering, so the motion means "work is flowing", never decoration. Hidden from assistive tech (the cards carry the information).
 *
 * @param {{
 *   selected?: string | null; solo?: string | null; bypass?: string[];
 *   onSelect: (id: string) => void; onSolo: (id: string) => void; onBypass: (id: string) => void;
 *   busy?: boolean; available?: boolean;
 *   latency?: Record<string, number> | null; frame?: number | null; loading?: boolean;
 * }} props
 */
export default function PipelineNodeFlowBar({ selected, solo, bypass = [], onSelect, onSolo, onBypass, busy = false, available = true, latency = null, frame = null, loading = false }) {
  const { keys } = useHotkeys();
  const stage = useRef(null);
  const nodeRefs = useMemo(() => PIPELINE_NODES.map(() => ({ current: null })), []);
  const timings = PIPELINE_NODES.map(node => stageLatency(node, latency));
  // The bar under each card is relative to the slowest *stage* (the final card is the whole-frame total, so it is excluded).
  const stageMax = Math.max(0, ...timings.filter((value, index) => value !== null && PIPELINE_NODES[index].id !== 'final'));
  const total = timings[PIPELINE_NODES.length - 1];
  return <div className="pipeline-flow" data-tour="pipeline-bar" role="region" aria-label="Visual execution pipeline" aria-busy={loading || undefined}>
    <div className="pipeline-flow-stage" ref={stage}>
    <div className="pipeline-flow-track">
    {PIPELINE_NODES.map((node, index) => {
      const ms = timings[index], isFinal = node.id === 'final';
      const disabled = !available || !!node.unavailable || busy;
      const alts = PIPELINE_ALT_NODES.filter(alt => alt.parent === node.id);
      const soloedHere = solo === node.id || alts.some(alt => alt.id === solo);
      return <React.Fragment key={node.id}>
        {index > 0 && <span className="pipeline-arrow" aria-hidden="true">→</span>}
        <div className={`pipeline-node ${selected === node.id ? 'selected' : ''} ${soloedHere ? 'solo' : ''} ${bypass.includes(node.id) ? 'bypassed' : ''}`} data-node={node.id} ref={nodeRefs[index]}>
          <button type="button" className="node-select" aria-label={`Solo ${node.label}`} aria-pressed={solo === node.id} disabled={disabled}
            aria-keyshortcuts={String(index + 1)}
            title={node.unavailable ?? `${node.note ?? 'Inspect the real stage buffer'} · key ${keys[`buffer${index + 1}`] || index + 1}`}
            onClick={() => { onSelect(node.id); onSolo(node.id); }}>
            <small className="node-index">{index + 1}</small>
            <span className="node-name">{node.short ?? node.label}</span>
            <span className={`node-ms ${ms === null ? 'unmeasured' : ''}`} title={ms === null ? 'The preview does not time this stage' : isFinal ? 'Whole-frame time' : 'Measured stage time'}>{isFinal && ms !== null ? `Σ ${formatMs(ms)}` : formatMs(ms)}</span>
            <span className="node-ms-bar" aria-hidden="true"><i style={{ width: ms !== null && !isFinal && stageMax > 0 ? `${Math.max(4, Math.round((ms / stageMax) * 100))}%` : '0%' }}/></span>
          </button>
          {(node.optional || alts.length > 0) && <div className="node-actions">
            {alts.map(alt => <button key={alt.id} type="button" className="button small node-chip" aria-label={`Solo ${alt.label}`} aria-pressed={solo === alt.id} disabled={disabled}
              title={alt.note} onClick={() => { onSelect(node.id); onSolo(alt.id); }}>{alt.short}</button>)}
            {node.optional && <button type="button" className="button small node-chip" aria-label={`Bypass ${node.label}`} aria-pressed={bypass.includes(node.id)} disabled={!!node.unavailable || busy}
              title={node.unavailable ?? `Bypass this stage · ${keys.bypass}`} onClick={() => { onSelect(node.id); onBypass(node.id); }}>Bypass</button>}
          </div>}
        </div>
      </React.Fragment>;
    })}
    </div>
    {PIPELINE_NODES.slice(1).map((node, index) => <AnimatedBeam key={node.id} containerRef={stage} fromRef={nodeRefs[index]} toRef={nodeRefs[index + 1]}
      anchor="top" pathWidth={2} pathOpacity={0.3} pathColor="var(--accent)" gradientStartColor="var(--accent)" gradientStopColor="var(--accent-cyan)"
      duration={1.1} delay={index * 0.14} active={loading && available} className="pipeline-flow-beam" />)}
    </div>
    <p className="pipeline-flow-status" role="status" aria-live="polite">
      {loading ? 'Rendering…' : latency ? `Frame ${frame ?? '—'} · total ${formatMs(total)}${solo ? ` · soloing ${PIPELINE_NODES.find(node => node.id === solo)?.short ?? PIPELINE_ALT_NODES.find(node => node.id === solo)?.short ?? solo}` : ''}` : 'Latencies appear after the first live preview'}
    </p>
  </div>;
}
