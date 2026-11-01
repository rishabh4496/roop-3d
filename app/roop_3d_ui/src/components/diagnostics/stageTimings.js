// Seven measured pipeline stages shown by the bottleneck flamegraph. Mirrors roop_3d/core/stage_timing.py.
export const FLAME_STAGES = [
  { id: 'decode', label: 'Decode', hw: 'NVDEC / software decoder', color: '#38bdf8' },
  { id: 'detection', label: 'Detection', hw: 'Face detector + tracker', color: '#818cf8' },
  { id: 'pose_tps', label: '3D Pose / TPS', hw: 'LivePortrait retarget, mesh blend, TPS warp', color: '#a78bfa' },
  { id: 'inswapper', label: 'Inswapper', hw: 'inswapper_128 identity swap', color: '#f43f5e' },
  { id: 'masking', label: 'Masking', hw: 'Face parser, matte, occluders, colour fusion', color: '#fb923c' },
  { id: 'restorer', label: 'Restorer', hw: 'GPEN / CodeFormer', color: '#f59e0b' },
  { id: 'encode', label: 'Encode', hw: 'Frame write / video encoder', color: '#06b6d4' },
];

const finite = value => (typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null);

// Preview captures report engine latency in the engine's own field names.
function fromPreviewLatency(latency) {
  if (!latency || typeof latency !== 'object') return {};
  const out = {};
  const detection = finite(latency.detection_ms ?? latency.detection);
  const swap = finite(latency.swap_ms ?? latency.inswapper);
  const retarget = finite(latency.retarget_ms);
  const projection = finite(latency.projection_ms);
  if (detection !== null) out.detection = detection;
  if (swap !== null) out.inswapper = swap;
  if (retarget !== null || projection !== null) out.pose_tps = (retarget ?? 0) + (projection ?? 0);
  for (const stage of FLAME_STAGES) {
    const direct = finite(latency[stage.id]);
    if (direct !== null && out[stage.id] === undefined) out[stage.id] = direct;
  }
  return out;
}

/**
 * Pick the freshest real measurement. Priority: explicit `timings`, the job's live `stage_timings_ms`, then a preview
 * capture. Unmeasured stages stay `null` — nothing is defaulted or estimated.
 */
export function resolveStageTimings({ timings = null, job = null, latency = null } = {}) {
  let source = 'none';
  let values = {};
  const live = job?.progress?.stage_timings_ms;
  if (timings && Object.keys(timings).length) {
    values = timings;
    source = 'live';
  } else if (live && Object.keys(live).length) {
    values = live;
    source = 'job';
  } else {
    const preview = fromPreviewLatency(latency);
    if (Object.keys(preview).length) {
      values = preview;
      source = 'preview';
    }
  }
  const resolved = {};
  for (const stage of FLAME_STAGES) resolved[stage.id] = finite(values[stage.id]);
  return { values: resolved, source };
}
