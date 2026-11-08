// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import PipelineNodeFlowBar from './components/workstation/PipelineNodeFlowBar';
import { PIPELINE_ALT_NODES, PIPELINE_NODES, formatMs, nodeBuffer, pipelineNode, stageLatency } from './components/workstation/pipeline';

const LATENCY = { detection_ms: 12.4, swap_ms: 31.2, retarget_ms: 4.5, projection_ms: 2.25, total_ms: 88.9 };

it('maps each stage to the measured latency fields, and reports unmeasured stages as null instead of inventing a number', () => {
  const timed = Object.fromEntries(PIPELINE_NODES.map(node => [node.id, stageLatency(node, LATENCY)]));
  expect(timed).toEqual({ input: 12.4, pose: 4.5, uv: 2.25, lighting: null, swap: 31.2, restorer: null, matte: null, final: 88.9 });
  expect(stageLatency(PIPELINE_NODES[0], null)).toBeNull();
  expect(stageLatency(PIPELINE_NODES[0], { detection_ms: Number.NaN })).toBeNull();
  expect(stageLatency(PIPELINE_NODES[0], { detection_ms: -1 })).toBeNull();
  expect(formatMs(null)).toBe('—');
  expect(formatMs(4.56)).toBe('4.6 ms');
  expect(formatMs(88.9)).toBe('89 ms');
});

it('resolves the Normals chip to the normal_map buffer without making it a ninth hotkey stage', () => {
  expect(PIPELINE_NODES).toHaveLength(8);
  expect(PIPELINE_ALT_NODES.map(node => [node.id, node.parent, node.buffer])).toEqual([['normal', 'lighting', 'normal_map']]);
  expect(nodeBuffer('normal')).toBe('normal_map');
  expect(pipelineNode('normal')?.label).toBe('3D Normal Map');
  expect(nodeBuffer('missing')).toBeNull();
  expect(nodeBuffer(null)).toBeNull();
});

it('renders one single-click solo card per stage with its latency, the Normals chip and the bypass toggle', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  const onSolo = vi.fn(), onSelect = vi.fn(), onBypass = vi.fn();
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  try {
    await act(async () => root.render(<PipelineNodeFlowBar selected="final" solo="normal" bypass={[]} onSelect={onSelect} onSolo={onSolo} onBypass={onBypass} latency={LATENCY} frame={42}/>));
    const cards = [...host.querySelectorAll('.pipeline-node')];
    expect(cards).toHaveLength(8);
    expect(cards.map(card => card.querySelector('.node-ms')?.textContent)).toEqual(['12 ms', '4.5 ms', '2.3 ms', '—', '31 ms', '—', '—', 'Σ 89 ms']);
    expect(host.querySelector('[data-node="lighting"]')?.classList.contains('solo')).toBe(true);     // soloing Normals highlights its parent stage
    expect(host.querySelector<HTMLButtonElement>('[aria-label="Solo 3D Normal Map"]')?.getAttribute('aria-pressed')).toBe('true');
    expect(host.querySelector('.pipeline-flow-status')?.textContent).toContain('Frame 42 · total 89 ms');
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Solo UltraMax High-Res"]')!.click());
    expect(onSolo).toHaveBeenCalledWith('restorer'); expect(onSelect).toHaveBeenCalledWith('restorer');
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Solo 3D Normal Map"]')!.click());
    expect(onSolo).toHaveBeenLastCalledWith('normal');
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Bypass UltraMax High-Res"]')!.click());
    expect(onBypass).toHaveBeenCalledWith('restorer');
    expect(host.querySelectorAll('[aria-label^="Bypass"]')).toHaveLength(1);                         // only the optional stage can be bypassed
  } finally { await act(async () => root.unmount()); host.remove(); }
});

it('disables every card until a preview request exists and shows no fake numbers before the first preview', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  try {
    await act(async () => root.render(<PipelineNodeFlowBar solo={null} onSelect={() => {}} onSolo={() => {}} onBypass={() => {}} available={false}/>));
    expect([...host.querySelectorAll<HTMLButtonElement>('.node-select')].every(button => button.disabled)).toBe(true);
    expect([...host.querySelectorAll('.node-ms')].every(node => node.textContent === '—')).toBe(true);
    expect(host.querySelector('.pipeline-flow-status')?.textContent).toBe('Latencies appear after the first live preview');
  } finally { await act(async () => root.unmount()); host.remove(); }
});
