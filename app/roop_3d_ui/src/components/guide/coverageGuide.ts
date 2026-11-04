/**
 * Plain-language reading of the FaceSet pose-coverage grid (rows: up / level / down, columns: yaw −75° … +75°).
 * The grid itself comes from the FaceSet's stored views; the "required" columns mirror the backend rule in
 * reconstruction/faceset_quality.py (frontal and both three-quarter views are required, profiles are recommended).
 */
export const YAW_COLUMNS = [
  { id: 'far_left', label: 'left profile (−75°)', need: 'recommended' },
  { id: 'left', label: 'three-quarter left (−45°)', need: 'required' },
  { id: 'front', label: 'frontal (0°)', need: 'required' },
  { id: 'right', label: 'three-quarter right (+45°)', need: 'required' },
  { id: 'far_right', label: 'right profile (+75°)', need: 'recommended' },
] as const;

export interface CoverageReading {
  covered: number; total: number;
  missingRequired: string[]; missingRecommended: string[];
  /** One or two sentences a person can act on. */
  text: string;
}

export function describeCoverage(grid: number[][]): CoverageReading {
  const total = grid.length * (grid[0]?.length ?? 0);
  const covered = grid.flat().filter(count => count > 0).length;
  const hasColumn = (column: number) => grid.some(row => (row[column] ?? 0) > 0);
  const missing = (need: string) => YAW_COLUMNS.map((item, column) => ({ item, column })).filter(({ item, column }) => item.need === need && !hasColumn(column)).map(({ item }) => item.label);
  const missingRequired = missing('required'), missingRecommended = missing('recommended');
  const list = (items: string[]) => (items.length > 1 ? `${items.slice(0, -1).join(', ')} and ${items.at(-1)}` : items[0]);
  let text: string;
  if (!covered) text = 'No views have a measurable head angle yet. Add sharp photos of the person.';
  else if (missingRequired.length) text = `${covered} of ${total} angles covered. Still needed: ${list(missingRequired)}.`;
  else if (missingRecommended.length) text = `${covered} of ${total} angles covered. The required angles are all present. Adding ${list(missingRecommended)} would help when the person turns far to the side.`;
  else text = `${covered} of ${total} angles covered, including both profiles.`;
  return { covered, total, missingRequired, missingRecommended, text };
}

export type RequestLevel = 'blocking' | 'required' | 'recommended' | 'optional' | 'info';
/** The backend prefixes each suggestion with its priority, for example `[required] Add a three-quarter portrait …`. */
export function parseRequest(raw: string): { level: RequestLevel; text: string } {
  const match = raw.match(/^\[(blocking|required|recommended|optional)\]\s*(.*)$/i);
  return match ? { level: match[1].toLowerCase() as RequestLevel, text: match[2] } : { level: 'info', text: raw };
}
