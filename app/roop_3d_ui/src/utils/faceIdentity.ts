/**
 * Persistent face identities (`face 0`, `face 1`, ...) shared by the canvas boxes, the Target Face Bank cards and the
 * angle grid, so one person keeps one colour and one tag everywhere.
 *
 * The id is the server's zero-based `persistent_face_id`: immutable per discovery slot and never the frame-local
 * left-to-right rank (`face_index`), which changes when people swap places.
 */

/** Distinct on dark video and for the common colour-vision deficiencies; text on every fill is near-black. */
export const FACE_COLORS = ['#38bdf8', '#fbbf24', '#c4b5fd', '#34d399', '#f472b6', '#fb923c', '#2dd4bf', '#a3e635'] as const;

export const faceColor = (id: number | null | undefined): string =>
  id == null || !Number.isInteger(id) || id < 0 ? '#9aa7ab' : FACE_COLORS[id % FACE_COLORS.length];

/** `FACE 0`: the pill text. A face without a persistent id yet (not analysed) falls back to its frame rank, `#1`. */
export const faceTag = (id: number | null | undefined, rank?: number): string =>
  id != null && Number.isInteger(id) && id >= 0 ? `FACE ${id}` : rank != null ? `#${rank + 1}` : 'FACE ?';

/** The lower-case form the API and older labels use (`face 0`). */
export const faceName = (id: number | null | undefined): string => (id != null && id >= 0 ? `face ${id}` : 'face');

/** Pitch rows, top to bottom (standard frame: + pitch = looking up), as the server's pitch bin index. */
export const PITCH_ROWS = [{ bin: 2, label: 'Pitch Up' }, { bin: 1, label: 'Level' }, { bin: 0, label: 'Pitch Down' }] as const;
/** Yaw columns, image left to right (+ yaw = toward image right), as the server's yaw bin index. */
export const YAW_COLUMNS = [
  { bin: 0, label: 'L Profile', short: 'L Prof.' }, { bin: 1, label: 'L Quarter', short: 'L ¼' }, { bin: 2, label: 'Frontal', short: 'Front' },
  { bin: 3, label: 'R Quarter', short: 'R ¼' }, { bin: 4, label: 'R Profile', short: 'R Prof.' },
] as const;

export const ANGLE_CELLS = 15;
export const angleLabel = (yawBin: number, pitchBin: number): string =>
  `${YAW_COLUMNS.find(item => item.bin === yawBin)?.label ?? '?'} · ${PITCH_ROWS.find(item => item.bin === pitchBin)?.label ?? '?'}`;
