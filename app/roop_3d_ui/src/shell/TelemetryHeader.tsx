import { useRef, type RefObject } from 'react';
import { AnimatedBeam, ConvergingBeams, MotionNumber } from '../components/ui-layouts';
import { skullBlendOf, swapScopeOf } from '../components/studio/swapScope';
import type { WorkstationParameters } from '../components/workstation/pipeline';
import { usePipelineStatus, usePipelineStatusPolling } from '../stores/pipelineStatus';
import { useTrtBuild } from '../stores/pipelineStream';
import { useTelemetry } from '../stores/telemetryStore';
import { cn } from '../lib/utils';

/**
 * Top glass telemetry strip: GPU, precision state, VRAM, live FPS, the active swap mode, and a compact execution graph.
 * Every number is something the server reported (`/ws/telemetry`, `/api/pipeline/status`); a value that has not been reported shows
 * "n/a" or an em dash. Nothing is estimated and no precision or mode is claimed that the engine did not state.
 */
export type GraphNode = 'identity' | 'target' | 'lbs' | 'relight' | 'output';

export const GRAPH_NODES: readonly { id: GraphNode; label: string; hash: string; hint: string }[] = [
  { id: 'identity', label: '.fsz Identity', hash: '#/identity-forge', hint: 'Source FaceSet: open Identity Forge' },
  { id: 'target', label: 'Target Media', hash: '#/studio', hint: 'Target video: open the Studio' },
  { id: 'lbs', label: '3D LBS + Jaw/Cage', hash: '#/studio', hint: 'Mesh, pose and jaw/cage warp' },
  { id: 'relight', label: 'SH Relight + TRT UltraMax', hash: '#/studio', hint: 'Spherical-harmonics relight and the enhancer' },
  { id: 'output', label: 'MP4 Output', hash: '#/queue', hint: 'Rendered output: open the Batch Queue' },
];

/**
 * Which graph node a server stage name belongs to. The engine's stage vocabulary is
 * detection / landmark / warp / restoration / blend (+ encode); anything unrecognised lights nothing rather than guessing.
 */
export function nodeForStage(stage: string | null | undefined): GraphNode | null {
  const name = String(stage ?? '').toLowerCase();
  if (!name || name === 'idle') return null;
  if (/(identity|source|faceset|embed)/.test(name)) return 'identity';
  if (/(detect|landmark|decode|track|inspect|target)/.test(name)) return 'target';
  if (/(warp|mesh|pose|lbs|swap|cage|jaw|reconstruct)/.test(name)) return 'lbs';
  if (/(restor|relight|light|enhanc|ultramax|blend|composit|matte)/.test(name)) return 'relight';
  if (/(encode|output|mux|write|export)/.test(name)) return 'output';
  return null;
}

export function swapModeLabel(parameters: WorkstationParameters): string {
  const scope = swapScopeOf(parameters);
  if (scope === 'full_craniofacial') return '100% FULL CRANIOFACIAL LOCK';
  if (scope === 'adaptive_skull') return `ADAPTIVE SKULL BLEND · ${Math.round(skullBlendOf(parameters) * 100)}% SOURCE`;
  return 'INNER FACE ONLY';
}

const mib = (value: number) => value / 1024;

function VramMeter() {
  const gpu = useTelemetry(state => state.hardware?.gpus?.[0] ?? null, (a, b) => a?.name === b?.name && a?.free_vram_mib === b?.free_vram_mib && a?.total_vram_mib === b?.total_vram_mib);
  if (!gpu || !(gpu.total_vram_mib > 0)) return <div className="thm-cell thm-vram" data-testid="thm-vram"><span className="thm-key">VRAM</span><b className="thm-val muted">n/a</b></div>;
  const total = gpu.total_vram_mib, used = Math.max(0, total - gpu.free_vram_mib);
  const pct = Math.min(100, Math.round((used / total) * 100));
  return (
    <div className="thm-cell thm-vram" data-testid="thm-vram">
      <span className="thm-key">VRAM</span>
      <b className="thm-val"><MotionNumber value={mib(used)} decimals={1} minChars={4} /> / <MotionNumber value={mib(total)} decimals={1} minChars={4} /> GB</b>
      <div className="thm-bar" role="meter" aria-label="GPU memory in use" aria-valuemin={0} aria-valuemax={Math.round(mib(total) * 10) / 10} aria-valuenow={Math.round(mib(used) * 10) / 10} aria-valuetext={`${mib(used).toFixed(1)} of ${mib(total).toFixed(1)} GB in use`}>
        <i style={{ width: `${pct}%` }} data-level={pct > 90 ? 'high' : pct > 75 ? 'warn' : 'ok'} />
      </div>
    </div>
  );
}

function Node({ node, nodeRef, active, className }: { node: (typeof GRAPH_NODES)[number]; nodeRef: RefObject<HTMLAnchorElement | null>; active: boolean; className?: string }) {
  return <a ref={nodeRef} href={node.hash} className={cn('thm-node', active && 'active', className)} data-node={node.id} data-active={active ? 'true' : 'false'} title={node.hint}>{node.label}</a>;
}

/** The compact graph: [.fsz Identity] + [Target Media] -> [3D LBS + Jaw/Cage] -> [SH Relight + TRT UltraMax] -> [MP4 Output]. */
function PipelineMiniMap() {
  const stage = useTelemetry(state => state.progress?.stage ?? state.pipeline?.stage ?? null);
  const running = !!stage && stage !== 'idle';
  const activeNode = running ? nodeForStage(stage) : null;
  const box = useRef<HTMLDivElement>(null);
  const refs = { identity: useRef<HTMLAnchorElement>(null), target: useRef<HTMLAnchorElement>(null), lbs: useRef<HTMLAnchorElement>(null), relight: useRef<HTMLAnchorElement>(null), output: useRef<HTMLAnchorElement>(null) };
  return (
    <div className="thm-map" ref={box} role="group" aria-label={running ? `Pipeline graph, running: ${stage}` : 'Pipeline graph, idle'} data-testid="thm-map">
      <div className="thm-map-inputs">
        <Node node={GRAPH_NODES[0]} nodeRef={refs.identity} active={activeNode === 'identity'} />
        <Node node={GRAPH_NODES[1]} nodeRef={refs.target} active={activeNode === 'target'} />
      </div>
      <Node node={GRAPH_NODES[2]} nodeRef={refs.lbs} active={activeNode === 'lbs'} />
      <Node node={GRAPH_NODES[3]} nodeRef={refs.relight} active={activeNode === 'relight'} />
      <Node node={GRAPH_NODES[4]} nodeRef={refs.output} active={activeNode === 'output'} />
      <ConvergingBeams containerRef={box} sources={[refs.identity, refs.target]} target={refs.lbs} spread={10} stagger={0.2} duration={1.3} pathWidth={1.5} pathColor="var(--accent-indigo)" pathOpacity={0.55} active={running} className="thm-beam" />
      <AnimatedBeam containerRef={box} fromRef={refs.lbs} toRef={refs.relight} duration={1.3} delay={0.4} pathWidth={1.5} pathColor="var(--accent-indigo)" pathOpacity={0.55} active={running} className="thm-beam" />
      <AnimatedBeam containerRef={box} fromRef={refs.relight} toRef={refs.output} duration={1.3} delay={0.8} pathWidth={1.5} pathColor="var(--accent-indigo)" pathOpacity={0.55} active={running} className="thm-beam" />
    </div>
  );
}

export default function TelemetryHeader({ parameters }: { parameters: WorkstationParameters }) {
  usePipelineStatusPolling();
  const gpuName = useTelemetry(state => state.hardware?.gpus?.[0]?.name ?? null);
  const fps = useTelemetry(state => (typeof state.progress?.fps === 'number' ? state.progress.fps : null));
  const precision = usePipelineStatus(state => state.value?.precision?.label ?? null);
  const engineState = usePipelineStatus(state => state.value?.pipeline?.engine_state ?? null);
  const building = !!useTrtBuild();
  const trt = building ? 'TRT ENGINE BUILDING…' : precision ? `${precision}${engineState ? ` · ${engineState}` : ''}` : 'n/a';
  return (
    <section className="telemetry-header" aria-label="Live telemetry" data-testid="telemetry-header">
      <div className="thm-cell thm-gpu" data-testid="thm-gpu"><span className="thm-key">GPU</span><b className="thm-val" title={gpuName ?? undefined}>{gpuName ?? 'n/a'}</b></div>
      <div className="thm-cell thm-trt" data-testid="thm-trt"><span className="thm-key">TensorRT precision</span><b className={cn('thm-val', !precision && !building && 'muted')}>{trt}</b></div>
      <VramMeter />
      <div className="thm-cell thm-fps" data-testid="thm-fps"><span className="thm-key">FPS</span><b className="thm-val"><MotionNumber value={fps ?? Number.NaN} decimals={1} minChars={4} /></b></div>
      <span className="thm-badge" data-testid="thm-swap-mode" title="Swap scope of the current render settings">{swapModeLabel(parameters)}</span>
      <PipelineMiniMap />
    </section>
  );
}
