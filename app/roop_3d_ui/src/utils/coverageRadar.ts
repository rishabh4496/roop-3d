/**
 * Model for the Angular Coverage Radar: which head angles a `.fsz` FaceSet really contains, on the exact 12 x 8 grid of the
 * backend bin contract (yaw -90..+90 in 15 degree steps, pitch -60..+60 in 15 degree steps). Everything here is pure so it is
 * unit-tested without a DOM. Pitch is the STANDARD frame (+ = looking up), the frame the coverage payload uses.
 */
export const YAW_MIN = -90, YAW_MAX = 90, PITCH_MIN = -60, PITCH_MAX = 60, BIN_DEG = 15;
export const YAW_BINS = 12, PITCH_BINS = 8, CELL_COUNT = YAW_BINS * PITCH_BINS;

export type PoseDeg = { yaw: number; pitch: number };
export type CoverageViews = { views?: { yaw: number; pitch: number }[]; observed_pose_points?: { bin_index: number; yaw?: number; pitch?: number }[] };

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

export const yawBin = (yaw: number) => clamp(Math.floor((clamp(yaw, YAW_MIN, YAW_MAX) - YAW_MIN) / BIN_DEG), 0, YAW_BINS - 1);
export const pitchBin = (pitch: number) => clamp(Math.floor((clamp(pitch, PITCH_MIN, PITCH_MAX) - PITCH_MIN) / BIN_DEG), 0, PITCH_BINS - 1);
export const cellIndex = (pose: PoseDeg) => pitchBin(pose.pitch) * YAW_BINS + yawBin(pose.yaw);
export const cellYawRange = (column: number): [number, number] => [YAW_MIN + column * BIN_DEG, YAW_MIN + (column + 1) * BIN_DEG];
export const cellPitchRange = (row: number): [number, number] => [PITCH_MIN + row * BIN_DEG, PITCH_MIN + (row + 1) * BIN_DEG];
export const inRange = (pose: PoseDeg) => finite(pose.yaw) && finite(pose.pitch) && pose.yaw >= YAW_MIN && pose.yaw <= YAW_MAX && pose.pitch >= PITCH_MIN && pose.pitch <= PITCH_MAX;

/** Samples per cell. The server's own `bin_index` wins when present; otherwise each view is binned by its yaw/pitch. */
export function cellCounts(coverage: CoverageViews | null | undefined): number[] {
  const counts = new Array<number>(CELL_COUNT).fill(0);
  const points = coverage?.observed_pose_points;
  if (Array.isArray(points) && points.length) {
    for (const point of points) if (Number.isInteger(point?.bin_index) && point.bin_index >= 0 && point.bin_index < CELL_COUNT) counts[point.bin_index] += 1;
    return counts;
  }
  for (const view of coverage?.views ?? []) if (finite(view?.yaw) && finite(view?.pitch)) counts[cellIndex(view)] += 1;
  return counts;
}

/**
 * Shirley–Chiu concentric map of the square [-1,1]^2 onto the unit disc. Distance from the centre is max(|yaw|/90, |pitch|/60),
 * so a frontal head sits in the middle and the extreme yaw / pitch cells form the outer ring: a polar view of the yaw x pitch grid.
 * `x` is image-right (+yaw), `y` is screen-down (so +pitch = up is drawn upward).
 */
export function polarPoint(yaw: number, pitch: number): { x: number; y: number } {
  const a = clamp(yaw / YAW_MAX, -1, 1), b = clamp(pitch / PITCH_MAX, -1, 1);
  if (a === 0 && b === 0) return { x: 0, y: 0 };
  let r: number, phi: number;
  if (a * a > b * b) { r = a; phi = (Math.PI / 4) * (b / a); } else { r = b; phi = Math.PI / 2 - (Math.PI / 4) * (a / b); }
  return { x: r * Math.cos(phi), y: -r * Math.sin(phi) };
}

/** Polygon (unit-disc coordinates) of one 15 x 15 degree cell; each edge is subdivided so its curved sides are smooth. */
export function cellPolygon(column: number, row: number, steps = 4): { x: number; y: number }[] {
  const [y0, y1] = cellYawRange(column), [p0, p1] = cellPitchRange(row);
  const edge = (from: PoseDeg, to: PoseDeg) => Array.from({ length: steps }, (_, i) => polarPoint(from.yaw + (to.yaw - from.yaw) * (i / steps), from.pitch + (to.pitch - from.pitch) * (i / steps)));
  return [...edge({ yaw: y0, pitch: p0 }, { yaw: y1, pitch: p0 }), ...edge({ yaw: y1, pitch: p0 }, { yaw: y1, pitch: p1 }),
    ...edge({ yaw: y1, pitch: p1 }, { yaw: y0, pitch: p1 }), ...edge({ yaw: y0, pitch: p1 }, { yaw: y0, pitch: p0 })];
}

/* ---- gaps ------------------------------------------------------------------------------------------------------------ */
export type YawBand = { id: 'left_profile' | 'left_34' | 'front' | 'right_34' | 'right_profile'; label: string; min: number; max: number; profile: boolean };
export const YAW_BANDS: YawBand[] = [
  { id: 'left_profile', label: 'Left profile', min: -90, max: -60, profile: true },
  { id: 'left_34', label: 'Left ¾', min: -60, max: -30, profile: false },
  { id: 'front', label: 'Frontal', min: -30, max: 30, profile: false },
  { id: 'right_34', label: 'Right ¾', min: 30, max: 60, profile: false },
  { id: 'right_profile', label: 'Right profile', min: 60, max: 90, profile: true },
];
export type PitchBand = { id: 'up' | 'level' | 'down'; label: string; min: number; max: number };
export const PITCH_BANDS: PitchBand[] = [
  { id: 'up', label: 'looking up', min: 15, max: 60 }, { id: 'level', label: 'level', min: -15, max: 15 }, { id: 'down', label: 'looking down', min: -60, max: -15 },
];
export type CoverageGap = { id: string; yaw: YawBand; pitch: PitchBand; state: 'missing' | 'sparse' | 'ok'; covered: number; total: number; label: string };

const rangeCells = (yawMin: number, yawMax: number, pitchMin: number, pitchMax: number): number[] => {
  const cells: number[] = [];
  for (let row = pitchBin(pitchMin); row <= pitchBin(pitchMax - 0.001); row++) for (let column = yawBin(yawMin); column <= yawBin(yawMax - 0.001); column++) cells.push(row * YAW_BINS + column);
  return cells;
};

/** Every yaw band x pitch band: `missing` = no sample at all, `sparse` = under half of its cells, `ok` otherwise. */
export function coverageGaps(counts: number[]): CoverageGap[] {
  const gaps: CoverageGap[] = [];
  for (const yaw of YAW_BANDS) for (const pitch of PITCH_BANDS) {
    const cells = rangeCells(yaw.min, yaw.max, pitch.min, pitch.max);
    const covered = cells.filter(index => counts[index] > 0).length;
    const state = covered === 0 ? 'missing' : covered / cells.length < 0.5 ? 'sparse' : 'ok';
    gaps.push({ id: `${yaw.id}:${pitch.id}`, yaw, pitch, state, covered, total: cells.length, label: `${yaw.label}, ${pitch.label}` });
  }
  return gaps;
}

/** The profile bands (|yaw| >= 60) with no sample at level pitch: the angles a turned-head target most often needs. */
export function missingProfiles(counts: number[]): YawBand[] {
  return YAW_BANDS.filter(band => band.profile && coverageGaps(counts).find(gap => gap.yaw.id === band.id && gap.pitch.id === 'level')?.state === 'missing');
}

export type TargetCoverage = { pose: PoseDeg; cell: number; covered: boolean };
/**
 * Which target poses the FaceSet cannot serve. A pose counts as covered when its own cell or any of the 8 neighbouring cells has a
 * sample (a 15 degree grid is coarse: a view 5 degrees across a cell edge is still the nearest reference).
 */
export function targetCoverage(counts: number[], poses: PoseDeg[]): TargetCoverage[] {
  return poses.filter(inRange).map(pose => {
    const cell = cellIndex(pose), column = cell % YAW_BINS, row = Math.floor(cell / YAW_BINS);
    let covered = false;
    for (let dr = -1; dr <= 1 && !covered; dr++) for (let dc = -1; dc <= 1; dc++) {
      const r = row + dr, c = column + dc;
      if (r >= 0 && r < PITCH_BINS && c >= 0 && c < YAW_BINS && counts[r * YAW_BINS + c] > 0) { covered = true; break; }
    }
    return { pose, cell, covered };
  });
}

export type RadarSummary = { total: number; filled: number; percent: number; samples: number; profiles: YawBand[]; gaps: CoverageGap[]; uncoveredTargets: TargetCoverage[]; ready: boolean; message: string };
/** One verdict for the panel and the pre-render warning. */
export function summarizeRadar(counts: number[], targetPoses: PoseDeg[] = []): RadarSummary {
  const filled = counts.filter(count => count > 0).length, samples = counts.reduce((sum, count) => sum + count, 0);
  const gaps = coverageGaps(counts), profiles = missingProfiles(counts);
  const uncoveredTargets = targetCoverage(counts, targetPoses).filter(item => !item.covered);
  const parts: string[] = [];
  if (!samples) parts.push('This FaceSet has no pose samples yet.');
  else {
    if (profiles.length) parts.push(`No ${profiles.map(band => band.label.toLowerCase()).join(' or ')} view.`);
    if (uncoveredTargets.length) {
      const yaws = uncoveredTargets.map(item => item.pose.yaw);
      const low = Math.round(Math.min(...yaws)), high = Math.round(Math.max(...yaws));
      parts.push(`The target turns to ${low === high ? `${low}°` : `${low}°…${high}°`} yaw where the FaceSet has no nearby view (${uncoveredTargets.length} of ${targetPoses.length} sampled poses).`);
    }
  }
  return { total: CELL_COUNT, filled, percent: Math.round((filled / CELL_COUNT) * 100), samples, profiles, gaps, uncoveredTargets, ready: samples > 0 && !profiles.length && !uncoveredTargets.length, message: parts.join(' ') };
}

/* ---- target poses ---------------------------------------------------------------------------------------------------- */
type LegacyPoseRange = { yaw_degrees?: [number, number] | null; pitch_degrees?: [number, number] | null };
const roundPose = (pose: PoseDeg): PoseDeg => ({ yaw: Math.round(pose.yaw), pitch: Math.round(pose.pitch) });

/**
 * Standard-frame poses the target needs a reference for: the five anchor points of the track's measured pose range (centre and the
 * four extremes) plus the poses of the faces on screen. Track and detected-face poses are stored in the LEGACY frame (pitch + = down),
 * so pitch is negated here, once. Poses are de-duplicated at 1 degree.
 */
export function targetPoseSamples(range: LegacyPoseRange | null | undefined, faces: { pose?: { yaw?: number; pitch?: number } | null }[] = []): PoseDeg[] {
  const found = new Map<string, PoseDeg>();
  const add = (yaw: number, legacyPitch: number) => {
    if (!finite(yaw) || !finite(legacyPitch)) return;
    const pose = roundPose({ yaw, pitch: -legacyPitch });
    found.set(`${pose.yaw}:${pose.pitch}`, pose);
  };
  const yaws = range?.yaw_degrees, pitches = range?.pitch_degrees;
  if (yaws && pitches) {
    const yawMid = (yaws[0] + yaws[1]) / 2, pitchMid = (pitches[0] + pitches[1]) / 2;
    add(yawMid, pitchMid); add(yaws[0], pitchMid); add(yaws[1], pitchMid); add(yawMid, pitches[0]); add(yawMid, pitches[1]);
  } else if (yaws) { add(yaws[0], 0); add((yaws[0] + yaws[1]) / 2, 0); add(yaws[1], 0); }
  for (const face of faces) if (face?.pose) add(face.pose.yaw ?? 0, face.pose.pitch ?? 0);
  return [...found.values()];
}
