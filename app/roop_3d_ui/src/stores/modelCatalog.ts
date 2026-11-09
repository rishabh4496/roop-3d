import { useEffect } from 'react';
import { api, post } from '../api';
import { createAtomicStore, useStoreSelector } from './atomicStore';

/** One row of `GET /api/models`: every registered artifact, installed or not. The UI never filters this list by `installed`/`state`. */
export type CatalogModel = {
  id: string; model_id: string; name: string;
  category: 'swapper' | 'enhancer' | 'detector' | 'occluder' | 'upscaler' | 'other' | string;
  native_resolution: number | null;
  installed: boolean; selectable: boolean; vram_loaded: boolean; trt_cached: boolean;
  download_url: string | null; state: string; sha256: string | null; size_bytes: number | null;
  license_id?: string; license_notice?: string; commercial_use?: string; manual_acceptance_required?: boolean;
  local_format?: string | null; supports_3d_hybrid?: boolean;
  /** A weight present in app/models is verified, approved and never blocked (server models/local_trust.py). */
  verified?: boolean; governance_approved?: boolean | null; blocked?: boolean; local_trust?: boolean;
};
export type DownloadJob = { download_id: string; model_id: string; state: string; bytes_downloaded: number; total_bytes: number; percent: number; error_code: string | null };
export type ImportReport = { status: string; imported?: { name: string; model_id: string }[]; skipped?: { name: string; reason: string }[]; corrupted?: { name: string; file: string; error: string }[] };

/**
 * What the weights are doing right now. `locate` = nothing on disk and no registered public source, so the only way in is a local file.
 * `blocked` = the registry refuses the artifact (licence/provenance); it is still listed, with its reason.
 */
export type ModelStatus = 'trt' | 'ready' | 'disk' | 'download' | 'locate' | 'blocked';
export const STATUS_META: Record<ModelStatus, { label: string; glyph: string; hint: string }> = {
  ready: { label: 'Ready', glyph: '●', hint: 'Weights are on disk and verified.' },
  trt: { label: 'TRT FP16 cached', glyph: '⚡', hint: 'Weights are on disk and a TensorRT engine is cached.' },
  disk: { label: 'On disk', glyph: '▤', hint: 'Weights are on disk but not yet verified against the registry.' },
  download: { label: 'Click to download', glyph: '☁', hint: 'Weights are not installed. A registered source is available.' },
  locate: { label: 'Locate local file', glyph: '📂', hint: 'Weights are not installed and no public source is registered: point Roop 3D at a local copy.' },
  blocked: { label: 'Blocked', glyph: '⛔', hint: 'The registry blocks this artifact.' },
};
export const ACTIVE_DOWNLOAD = ['queued', 'downloading', 'cancelling'];
export const isInstalled = (status: ModelStatus) => status === 'ready' || status === 'trt' || status === 'disk';

export function modelStatus(model: CatalogModel): ModelStatus {
  if (model.state === 'BLOCKED' && !model.installed) return 'blocked';
  if (model.installed) return model.trt_cached ? 'trt' : (model.vram_loaded || model.state === 'VERIFIED_LOCAL') ? 'ready' : 'disk';
  return model.download_url ? 'download' : 'locate';
}

/** Display names in the shape the workstation uses; resolution is shown as a separate badge. */
const DISPLAY_NAMES: Record<string, string> = {
  inswapper_128: 'InSwapper 128', inswapper_128_fp16: 'InSwapper FP16',
  hyperswap_1a_256: 'HyperSwap 1A', hyperswap_1b_256: 'HyperSwap 1B', hyperswap_1c_256: 'HyperSwap 1C',
  realswap_256: 'RealSwap', realswap_512: 'RealSwap', simswap_256: 'SimSwap', simswap_512_unofficial: 'SimSwap (unofficial)',
  ghost_256_unet_1: 'Ghost v1', ghost_256_unet_2: 'Ghost v2', ghost_256_unet_3: 'Ghost v3',
  blendswap_256: 'BlendSwap', uniface_256: 'Uniface',
  gpen_bfr_512: 'GPEN Realistic', gpen_bfr_1024: 'GPEN Realistic', gfpgan_v14: 'GFPGAN v1.4', restoreformer_pp: 'RestoreFormer++', ultramax: 'UltraMax', codeformer: 'CodeFormer',
  scrfd: 'SCRFD 10G', retinaface: 'RetinaFace', yoloface: 'YOLOv8-Face', xseg: 'XSeg Occluder', bisenet: 'BiSeNet Parser',
  realesrgan_x2plus: 'Real-ESRGAN x2plus', realesrgan_x4plus: 'Real-ESRGAN x4plus', '4x_ultrasharp': '4x-UltraSharp',
};
export const modelLabel = (model: CatalogModel) => DISPLAY_NAMES[model.id] ?? model.name;
export const resolutionBadge = (model: CatalogModel) => (model.native_resolution ? `${model.native_resolution}px` : null);
export const modelTitle = (model: CatalogModel) => `${modelLabel(model)}${resolutionBadge(model) ? ` • ${resolutionBadge(model)}` : ''}`;

type CatalogState = {
  models: CatalogModel[]; loaded: boolean; loading: boolean; error: string;
  /** Latest download job per model id. */
  downloads: Record<string, DownloadJob>;
};
export const modelCatalog = createAtomicStore<CatalogState>({ models: [], loaded: false, loading: false, error: '', downloads: {} });

const message = (reason: unknown, fallback: string) => (reason instanceof Error ? reason.message : fallback);
let inflight: Promise<void> | null = null;

export function refreshCatalog(): Promise<void> {
  if (inflight) return inflight;
  modelCatalog.set({ loading: true });
  inflight = api<CatalogModel[]>('/models')
    .then(models => {
      if (!Array.isArray(models)) throw new Error('unexpected response from /models');
      modelCatalog.set({ models, loaded: true, error: '' });
    })
    .catch(reason => modelCatalog.set({ error: message(reason, 'Could not load the model list.') }))
    .finally(() => { modelCatalog.set({ loading: false }); inflight = null; });
  return inflight;
}

/** Re-reads `app/models` on the server, then refreshes the list. */
export async function rescanModels(): Promise<{ models: number; installed: number }> {
  const result = await post<{ models: number; installed: number }>('/models/rescan', {});
  await refreshCatalog();
  return result;
}

let poller: number | undefined;
function pollDownloads() {
  const pending = Object.values(modelCatalog.get().downloads).filter(item => ACTIVE_DOWNLOAD.includes(item.state));
  if (pending.length === 0) { if (poller !== undefined) window.clearInterval(poller); poller = undefined; return; }
  pending.forEach(item => {
    void api<DownloadJob>(`/models/downloads/${item.download_id}`).then(next => {
      modelCatalog.set(state => ({ downloads: { ...state.downloads, [next.model_id]: next } }));
      if (next.state === 'completed') void refreshCatalog();
    }).catch(() => undefined);
  });
}
function watchDownloads() { if (poller === undefined) poller = window.setInterval(pollDownloads, 700); }

export async function startModelDownload(model: Pick<CatalogModel, 'id' | 'manual_acceptance_required'>, acceptedLicense: boolean): Promise<DownloadJob> {
  const job = await post<DownloadJob>(`/models/${encodeURIComponent(model.id)}/downloads`, { accepted_license: model.manual_acceptance_required ? acceptedLicense : false });
  modelCatalog.set(state => ({ downloads: { ...state.downloads, [model.id]: job } }));
  watchDownloads();
  return job;
}

export async function cancelModelDownload(modelId: string): Promise<void> {
  const job = modelCatalog.get().downloads[modelId];
  if (!job) return;
  const next = await api<DownloadJob>(`/models/downloads/${job.download_id}`, { method: 'DELETE' });
  modelCatalog.set(state => ({ downloads: { ...state.downloads, [modelId]: next } }));
}

/**
 * Offline install: `path` is a folder or a .zip holding the model files. The server accepts only files whose SHA-256 and size match a
 * registry record, so an unrecognised .onnx is reported back rather than silently loaded.
 */
export async function importLocalWeights(path: string): Promise<ImportReport> {
  const report = await post<ImportReport>('/installer/models/import', { bundle_path: path.trim() });
  await rescanModels();
  return report;
}

/** Loads the list on first use; every consumer shares one fetch. */
export function useModelCatalog() {
  const state = useStoreSelector(modelCatalog, value => value);
  useEffect(() => {
    if (!modelCatalog.get().loaded && !modelCatalog.get().loading) void refreshCatalog();
    if (Object.values(modelCatalog.get().downloads).some(item => ACTIVE_DOWNLOAD.includes(item.state))) watchDownloads();
  }, []);
  return state;
}
