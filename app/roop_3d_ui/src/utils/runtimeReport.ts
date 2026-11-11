/** Types, validation and clipboard helper for the 14-section runtime report (`GET /api/v1/system/runtime-report`). */

export type SectionStatus = 'OK' | 'INFO' | 'WARN' | 'UNAVAILABLE' | 'UNKNOWN';
export interface ReportRow { label: string; value: string }
export interface ReportSection { number: number; key: string; title: string; status: SectionStatus; rows: ReportRow[]; notes: string[] }
export interface RuntimeReport {
  report_id: string; generated_at: string; version: string; trigger: string; overall: string;
  counts: Partial<Record<SectionStatus, number>>; sections: ReportSection[];
  /** Plain-text, box-drawn rendering of the same report: what "Copy" puts on the clipboard. */
  text: string;
}

const STATUSES: readonly SectionStatus[] = ['OK', 'INFO', 'WARN', 'UNAVAILABLE', 'UNKNOWN'];
const str = (value: unknown, fallback = ''): string => (typeof value === 'string' ? value : fallback);

/** Validates the response into a fully-defaulted report, or throws a message the panel can show. Never lets a bad field reach render. */
export function parseRuntimeReport(raw: unknown): RuntimeReport {
  if (!raw || typeof raw !== 'object' || !Array.isArray((raw as { sections?: unknown }).sections)) throw new Error('unexpected response from /system/runtime-report');
  const value = raw as Record<string, unknown>;
  const sections: ReportSection[] = (value.sections as unknown[]).flatMap((item, index) => {
    if (!item || typeof item !== 'object') return [];
    const section = item as Record<string, unknown>;
    const rows = (Array.isArray(section.rows) ? section.rows : []).flatMap((row): ReportRow[] => {
      const entry = row as Record<string, unknown> | null;
      return entry && typeof entry === 'object' ? [{ label: str(entry.label), value: str(entry.value) }] : [];
    });
    const status = (STATUSES as readonly unknown[]).includes(section.status) ? section.status as SectionStatus : 'UNKNOWN';
    return [{ number: typeof section.number === 'number' ? section.number : index + 1, key: str(section.key, `section-${index + 1}`),
      title: str(section.title, `Section ${index + 1}`), status, rows, notes: (Array.isArray(section.notes) ? section.notes : []).filter((note): note is string => typeof note === 'string') }];
  });
  if (!sections.length) throw new Error('the runtime report has no sections');
  return {
    report_id: str(value.report_id), generated_at: str(value.generated_at), version: str(value.version), trigger: str(value.trigger, 'on demand'),
    overall: str(value.overall, 'OK'), counts: (value.counts && typeof value.counts === 'object' ? value.counts : {}) as RuntimeReport['counts'],
    sections, text: str(value.text),
  };
}

/** Sections that need a look (open by default in the panel). */
export const needsAttention = (section: ReportSection): boolean => section.status === 'WARN' || section.status === 'UNAVAILABLE' || section.status === 'UNKNOWN';

/**
 * Puts `text` on the clipboard. Uses the async Clipboard API where the page may (localhost is a secure context) and falls back to a
 * selected, off-screen textarea for browsers or embeds that refuse it. Resolves to whether the copy is believed to have happened.
 */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch { /* fall through to the textarea path */ }
  if (typeof document === 'undefined') return false;
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.style.cssText = 'position:fixed;left:-9999px;top:0;opacity:0';
  document.body.append(area);
  try {
    area.select();
    return typeof document.execCommand === 'function' && document.execCommand('copy');
  } catch { return false; } finally { area.remove(); }
}

/** `GET /api/v1/settings/benchmark_status`: the explicit, bounded thread/pool benchmark (synthetic affine workload, never run at startup). */
export interface BenchmarkResult { execution_threads: number; pool_size: number; samples: number; seconds: number; samples_per_second: number; provider: string }
export interface BenchmarkStatus {
  state: 'IDLE' | 'RUNNING' | 'DONE' | 'FAILED' | 'CANCELLED';
  profile?: 'quick' | 'full'; results: BenchmarkResult[]; recommendation: BenchmarkResult | null; provider?: string; note?: string; error?: string;
}

export function parseBenchmark(raw: unknown): BenchmarkStatus {
  const value = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
  const state = (['IDLE', 'RUNNING', 'DONE', 'FAILED', 'CANCELLED'] as const).find(item => item === value.state) ?? 'IDLE';
  const result = (item: unknown): BenchmarkResult[] => {
    const row = item as Record<string, unknown> | null;
    if (!row || typeof row !== 'object') return [];
    const numbers = ['execution_threads', 'pool_size', 'samples', 'seconds', 'samples_per_second'] as const;
    if (numbers.some(key => typeof row[key] !== 'number' || !Number.isFinite(row[key] as number))) return [];
    return [{ execution_threads: row.execution_threads as number, pool_size: row.pool_size as number, samples: row.samples as number,
      seconds: row.seconds as number, samples_per_second: row.samples_per_second as number, provider: str(row.provider) }];
  };
  return {
    state, profile: value.profile === 'full' ? 'full' : value.profile === 'quick' ? 'quick' : undefined,
    results: (Array.isArray(value.results) ? value.results : []).flatMap(result),
    recommendation: result(value.recommendation)[0] ?? null, provider: str(value.provider) || undefined, note: str(value.note) || undefined,
    error: str(value.error) || undefined,
  };
}
