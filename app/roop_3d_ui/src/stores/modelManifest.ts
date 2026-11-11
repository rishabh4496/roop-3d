import { api, post } from '../api';
import { createAtomicStore, useStoreSelector } from './atomicStore';
import { ACTIVE_TRANSFER, clearTransfer, pipelineStream } from './pipelineStream';

/**
 * Client of the universal model manifest: `GET /models/manifest`, `POST /models/download`, `POST /models/verify-and-repair`
 * and `GET /models/batches/{id}`. Live byte counts arrive separately on `/ws/v1/pipeline/status` (see `pipelineStream`).
 */
export type LicenseBadge = 'open' | 'restricted' | 'unverified';
export type HashStatus = 'verified' | 'mismatch' | 'not_installed';

export interface ManifestModel {
  id: string; name: string; category: string; kind: string;
  /** READY | PROCEDURAL_FALLBACK | MISSING | PARTIAL | CORRUPT | NOT_APPLICABLE */
  state: string;
  installed: boolean; hashStatus: HashStatus; detail: string;
  sizeBytes: number; installedBytes: number; partialBytes: number; sha256: string;
  license: string; licenseBadge: LicenseBadge; licenseUrl: string; licenseNote: string; requiresAcceptance: boolean;
  optional: boolean; mirrorCount: number; acquisition: string; note: string; commercialUse: string; stages: string[];
}
export interface Manifest {
  models: ManifestModel[];
  requiredMissing: string[];
  licenses: Record<LicenseBadge, number>;
  disk: { installedBytes: number; downloadNeededBytes: number; freeBytes: number | null; enoughSpace: boolean | null };
}

const num = (value: unknown, fallback = 0): number => (typeof value === 'number' && Number.isFinite(value) ? value : fallback);
const str = (value: unknown, fallback = ''): string => (typeof value === 'string' ? value : fallback);
const list = (value: unknown): string[] => (Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []);

/** The licence names the badge shows. Anything else is shown as written. */
const BADGE_TEXT: Record<string, string> = { 'CC0-1.0': 'CC0', 'CC-BY-4.0': 'CC-BY-4.0', MIT: 'MIT', 'Apache-2.0': 'Apache-2.0', 'BSD-3-Clause': 'BSD-3-Clause' };
export const licenseText = (model: Pick<ManifestModel, 'license' | 'licenseBadge'>): string =>
  BADGE_TEXT[model.license] ?? (model.licenseBadge === 'unverified' || model.license === 'UNVERIFIED' ? 'UNVERIFIED' : model.license || 'UNKNOWN');

export function parseModel(raw: unknown): ManifestModel | null {
  if (!raw || typeof raw !== 'object') return null;
  const row = raw as Record<string, unknown>;
  const id = str(row.id);
  if (!id) return null;
  const hash = row.hash_status;
  const badge = row.license_badge;
  return {
    id, name: str(row.name, id), category: str(row.category, 'other'), kind: str(row.kind), state: str(row.state, 'MISSING'),
    installed: row.installed === true, detail: str(row.detail),
    hashStatus: hash === 'verified' || hash === 'mismatch' ? hash : 'not_installed',
    sizeBytes: Math.max(0, num(row.size_bytes)), installedBytes: Math.max(0, num(row.installed_bytes)), partialBytes: Math.max(0, num(row.partial_bytes)),
    sha256: str(row.sha256),
    license: str(row.license), licenseBadge: badge === 'open' || badge === 'unverified' ? badge : 'restricted',
    licenseUrl: str(row.license_url), licenseNote: str(row.license_note), requiresAcceptance: row.requires_acceptance === true,
    optional: row.optional === true, mirrorCount: Math.max(0, num(row.mirror_count)), acquisition: str(row.acquisition), note: str(row.note),
    commercialUse: str(row.commercial_use), stages: list(row.required_for_stages),
  };
}

/** Validates the manifest response; throws when it is not a manifest at all (so the caller shows one clear error). */
export function parseManifest(raw: unknown): Manifest {
  if (!raw || typeof raw !== 'object' || !Array.isArray((raw as { models?: unknown }).models)) throw new Error('The local service returned an unexpected model manifest.');
  const body = raw as { models: unknown[]; summary?: Record<string, unknown>; disk?: Record<string, unknown> };
  const models = body.models.map(parseModel).filter((model): model is ManifestModel => model !== null);
  const licenses = (body.summary?.licenses ?? {}) as Record<string, unknown>;
  const disk = body.disk ?? {};
  const requiredMissing = list(body.summary?.required_missing);
  return {
    models,
    // Trust the server's list; derive it only when the summary block is absent altogether.
    requiredMissing: body.summary ? requiredMissing : models.filter(model => !model.optional && !model.installed).map(model => model.id),
    licenses: { open: num(licenses.open), restricted: num(licenses.restricted), unverified: num(licenses.unverified) },
    disk: {
      installedBytes: Math.max(0, num(disk.installed_bytes)), downloadNeededBytes: Math.max(0, num(disk.download_needed_bytes)),
      freeBytes: typeof disk.free_bytes === 'number' ? disk.free_bytes : null, enoughSpace: typeof disk.enough_space === 'boolean' ? disk.enough_space : null,
    },
  };
}

// ------------------------------------------------------------------------------------------------ what "download all required" would do
export interface RequiredPlan {
  /** Fetched without any acknowledgement (open licences). */
  open: ManifestModel[];
  /** Need `accepted_license: true`: non-commercial, research-only or unverified terms. */
  needsAcceptance: ManifestModel[];
  /** Cannot be fetched automatically (no public source, a dependency that is not installed, or a corrupt file waiting for repair). */
  unavailable: ManifestModel[];
  /** Total bytes still to fetch for `open` and (if accepted) `needsAcceptance`. */
  bytesOpen: number;
  bytesRestricted: number;
}

const buildable = (model: ManifestModel): boolean => model.kind === 'compiled' || model.kind === 'bundled';

export function planRequired(manifest: Manifest | null): RequiredPlan {
  const plan: RequiredPlan = { open: [], needsAcceptance: [], unavailable: [], bytesOpen: 0, bytesRestricted: 0 };
  if (!manifest) return plan;
  const byId = new Map(manifest.models.map(model => [model.id, model]));
  for (const id of manifest.requiredMissing) {
    const model = byId.get(id);
    if (!model) continue;
    const reachable = model.mirrorCount > 0 || buildable(model);
    if (!reachable || model.state === 'NOT_APPLICABLE') plan.unavailable.push(model);
    else if (model.requiresAcceptance) { plan.needsAcceptance.push(model); plan.bytesRestricted += Math.max(0, model.sizeBytes - model.installedBytes); }
    else { plan.open.push(model); plan.bytesOpen += Math.max(0, model.sizeBytes - model.installedBytes); }
  }
  return plan;
}

// ------------------------------------------------------------------------------------------------ stores
interface ManifestState { manifest: Manifest | null; loading: boolean; error: string; loadedAt: number }
export const manifestStore = createAtomicStore<ManifestState>({ manifest: null, loading: false, error: '', loadedAt: 0 });
export const useManifest = (): ManifestState => useStoreSelector(manifestStore, state => state, (a, b) => a === b);

let inflight: Promise<void> | null = null;
export function refreshManifest(options: { force?: boolean } = {}): Promise<void> {
  if (inflight && !options.force) return inflight;
  manifestStore.set({ loading: true });
  const run = api<unknown>(`/models/manifest${options.force ? '?refresh=true' : ''}`)
    .then(raw => { manifestStore.set({ manifest: parseManifest(raw), error: '', loading: false, loadedAt: Date.now() }); })
    .catch((error: Error) => { manifestStore.set({ error: error.message || 'The model manifest could not be read.', loading: false }); })
    .finally(() => { if (inflight === run) inflight = null; });
  inflight = run;
  return run;
}

export type BatchPhase = 'idle' | 'starting' | 'running' | 'verifying' | 'done' | 'error';
export interface SkippedModel { id: string; reason: string; detail: string }
interface BatchState {
  phase: BatchPhase; batchId: string; message: string; skipped: SkippedModel[];
  /** Models of the current run, so the UI can show "3 of 7". */
  ids: string[]; failed: string[]; verified: number | null; corrupt: string[];
}
const IDLE_BATCH: BatchState = { phase: 'idle', batchId: '', message: '', skipped: [], ids: [], failed: [], verified: null, corrupt: [] };
export const batchStore = createAtomicStore<BatchState>(IDLE_BATCH);
export const useBatch = (): BatchState => useStoreSelector(batchStore, state => state, (a, b) => a === b);

interface BatchResponse { batch_id?: string; state?: string; entries?: Record<string, { status?: string; error?: string | null }>; skipped?: unknown[]; queued?: unknown[] }
const parseSkipped = (value: unknown): SkippedModel[] => (Array.isArray(value) ? value : []).flatMap(item => {
  const row = item as Record<string, unknown> | null;
  return row && typeof row.id === 'string' ? [{ id: row.id, reason: str(row.reason), detail: str(row.detail) }] : [];
});

const POLL_MS = 1500;
let pollTimer: ReturnType<typeof setTimeout> | null = null;
export const stopBatchPolling = (): void => { if (pollTimer) clearTimeout(pollTimer); pollTimer = null; };

async function pollBatch(batchId: string, then: () => Promise<void>): Promise<void> {
  stopBatchPolling();
  let response: BatchResponse;
  try { response = await api<BatchResponse>(`/models/batches/${encodeURIComponent(batchId)}`); }
  catch (error) {
    // The service may be restarting: keep the batch and try again; the stream's reconnect will tell us when it is back.
    if (batchStore.get().batchId === batchId) pollTimer = setTimeout(() => { void pollBatch(batchId, then); }, POLL_MS * 2);
    batchStore.set({ message: `Waiting for the local service… ${(error as Error).message}` });
    return;
  }
  if (batchStore.get().batchId !== batchId) return;
  const entries = response.entries ?? {};
  const failed = Object.entries(entries).filter(([, row]) => row?.status === 'failed').map(([id]) => id);
  if (response.state === 'running') {
    batchStore.set({ phase: 'running', failed });
    pollTimer = setTimeout(() => { void pollBatch(batchId, then); }, POLL_MS);
    return;
  }
  batchStore.set({ failed });
  await then();
}

/** Start downloading `ids` (or every missing model that needs no acknowledgement when empty). */
export async function startDownload(ids: string[], options: { acceptedLicense?: boolean; verifyAfter?: boolean } = {}): Promise<void> {
  if (batchStore.get().phase === 'starting' || batchStore.get().phase === 'running' || batchStore.get().phase === 'verifying') return;
  batchStore.set({ ...IDLE_BATCH, phase: 'starting', ids, message: 'Starting…' });
  try {
    const response = await post<BatchResponse>('/models/download', { models: ids, accepted_license: options.acceptedLicense === true });
    const batchId = str(response.batch_id);
    const skipped = parseSkipped(response.skipped);
    const queued = list(response.queued);
    if (!batchId || !queued.length) {
      batchStore.set({ phase: 'done', batchId: '', skipped, message: skipped.length ? `${skipped.length} model${skipped.length === 1 ? ' was' : 's were'} not started.` : 'Everything requested is already installed.' });
      await refreshManifest({ force: true });
      return;
    }
    batchStore.set({ phase: 'running', batchId, skipped, ids: queued, message: `Downloading ${queued.length} model${queued.length === 1 ? '' : 's'}…` });
    await pollBatch(batchId, async () => {
      if (options.verifyAfter === false) {
        batchStore.set({ phase: 'done', message: 'Downloads finished.' });
        await refreshManifest({ force: true });
        return;
      }
      await runVerifyAndRepair(queued, { acceptedLicense: options.acceptedLicense, repair: true, keepSkipped: true });
    });
  } catch (error) {
    batchStore.set({ phase: 'error', message: (error as Error).message || 'The download could not start.' });
  }
}

/** Re-hash the given models (all when empty), set corrupt files aside and re-fetch whatever is missing or bad. */
export async function runVerifyAndRepair(ids: string[] = [], options: { acceptedLicense?: boolean; repair?: boolean; keepSkipped?: boolean } = {}): Promise<void> {
  const keep = options.keepSkipped ? batchStore.get().skipped : [];
  batchStore.set({ phase: 'verifying', message: 'Verifying SHA-256 of installed models…', skipped: keep });
  try {
    const report = await post<{ checked?: number; verified?: number; missing?: unknown[]; corrupt?: unknown[]; repair?: BatchResponse | null }>(
      '/models/verify-and-repair', { models: ids, repair: options.repair !== false, accepted_license: options.acceptedLicense === true });
    const corrupt = list(report.corrupt);
    const repairId = str(report.repair?.batch_id);
    if (repairId && list(report.repair?.queued).length) {
      batchStore.set({ phase: 'running', batchId: repairId, ids: list(report.repair?.queued), corrupt, verified: num(report.verified), message: `Repairing ${list(report.repair?.queued).length} model${list(report.repair?.queued).length === 1 ? '' : 's'}…`, skipped: [...keep, ...parseSkipped(report.repair?.skipped)] });
      await pollBatch(repairId, async () => {
        batchStore.set({ phase: 'done', message: 'Repair finished.' });
        await refreshManifest({ force: true });
      });
      return;
    }
    batchStore.set({ phase: 'done', batchId: '', verified: num(report.verified), corrupt, message: `Checked ${num(report.checked)} models: ${num(report.verified)} verified${corrupt.length ? `, ${corrupt.length} repaired or set aside` : ''}.` });
    await refreshManifest({ force: true });
  } catch (error) {
    batchStore.set({ phase: 'error', message: (error as Error).message || 'Verification failed.' });
  }
}

export async function cancelBatch(): Promise<void> {
  const { batchId } = batchStore.get();
  if (!batchId) return;
  try { await api(`/models/batches/${encodeURIComponent(batchId)}`, { method: 'DELETE' }); }
  catch (error) { batchStore.set({ message: (error as Error).message }); }
}

export function resetBatch(): void { stopBatchPolling(); batchStore.set(IDLE_BATCH); }

/** Drop transfer rows that are no longer running once their model shows as installed in the manifest. */
export function pruneFinishedTransfers(manifest: Manifest): void {
  const transfers = pipelineStream.get().transfers;
  for (const model of manifest.models) {
    const transfer = transfers[model.id];
    if (transfer && !ACTIVE_TRANSFER.includes(transfer.status) && model.installed) clearTransfer(model.id);
  }
}

// ------------------------------------------------------------------------------------------------ formatting
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const exponent = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const value = bytes / 1024 ** exponent;
  return `${value >= 100 || exponent === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[exponent]}`;
}
/** The server reports megabits per second; people read MB/s. */
export const formatSpeed = (mbps: number): string => `${(Math.max(0, mbps) / 8).toFixed(1)} MB/s`;
export function formatEta(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return 'ETA …';
  const s = Math.round(seconds);
  if (s < 60) return `ETA ${s}s`;
  if (s < 3600) return `ETA ${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
  return `ETA ${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m`;
}
