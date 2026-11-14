import { useRef, type RefObject } from 'react';
import { AnimatedBeam, ConvergingBeams, MotionNumber } from '../components/ui-layouts';
import { usePipelineStatus, usePipelineStatusPolling } from '../stores/pipelineStatus';
import { useTrtBuild } from '../stores/pipelineStream';
import { useTelemetry } from '../stores/telemetryStore';
import { cn } from '../lib/utils';

/**
 * Top glass telemetry strip: GPU, precision state, VRAM, live FPS, and a compact execution graph.
 * Every number is server-reported; unreported values show "n/a" or an em dash.
 */
export type GraphNode = 'identity' | 'target' | 'lbs' | 'relight' | 'output';

export const GRAPH_NODES: readonly { id: GraphNode; label: string; hash: string; hint: string }[] = [
  { id: 'identity', label: '.fsz Identity', hash: '#/identity-forge', hint: 'Source FaceSet: open Identity Forge' },
  { id: 'target', label: 'Target Media', hash: '#/studio', hint: 'Target video: open the Studio' },
  { id: 'lbs', label: '3D mesh fit + pose', hash: '#/studio', hint: 'Volumetric head mesh and pose fitting' },
  { id: 'relight', label: 'SH Relight + PBR Shade', hash: '#/studio', hint: 'Spherical-harmonics relight and 3D shader' },
  { id: 'output', label: 'MP4 Output', hash: '#/queue', hint: 'Rendered output: open the Batch Queue' },
];

/**
 * Which graph node a server stage name belongs to. The engine's stage vocabulary is
 * detection / landmark / mesh fit / render / blend (+ encode); anything unrecognised lights nothing rather than guessing.
 */
export function nodeForStage(stage: string | null | undefined): GraphNode | null {
  const name = String(stage ?? '').toLowerCase();
  if (!name || name === 'idle') return null;
  if (/(identity|source|faceset|embed)/.test(name)) return 'identity';
  if (/(detect|landmark|decode|track|inspect|target)/.test(name)) return 'target';
  if (/(mesh|pose|lbs|reconstruct|fit)/.test(name)) return 'lbs';
  if (/(relight|light|shad|render|blend|composit|matte)/.test(name)) return 'relight';
  if (/(encode|output|mux|write|export)/.test(name)) return 'output';
  return null;
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

/** The compact graph: [.fsz Identity] + [Target Media] -> [3D LBS + Jaw/Cage] -> [SH Relight + PBR Shade] -> [MP4 Output]. */
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

export default function TelemetryHeader() {
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
      <span className="thm-badge" data-testid="thm-pipeline-mode" title="True-3D mesh reconstruction pipeline">PURE 3D HEAD</span>
      <PipelineMiniMap />
    </section>
  );
}
