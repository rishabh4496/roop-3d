import { api, ApiError } from '../api';
import type { Reconstruction } from '../contracts';

/**
 * Loads an identity-asset head as a binary glTF (`.../identity-asset/mesh.glb`) and turns it into the viewer's `Reconstruction`,
 * falling back to the JSON `.../identity-asset/mesh` route when the GLB is unavailable or unreadable.
 *
 * The reader is deliberately small (one mesh, one primitive, float/uint accessors, an optional embedded PNG albedo): it reads
 * exactly what `io/glb_export.py` writes, validates every offset before touching it, and throws a `GlbError` for anything else
 * so the caller can fall back to JSON instead of rendering garbage. glTF is already y-up/right-handed, which is the viewer's own
 * frame, so the result is tagged `coordinate_system: 'opengl'` exactly like the JSON route.
 */
export class GlbError extends Error {
  constructor(message: string) { super(message); this.name = 'GlbError'; }
}

const MAGIC = 0x46546c67;                    // "glTF"
const CHUNK_JSON = 0x4e4f534a, CHUNK_BIN = 0x004e4942;
const COMPONENT = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array } as const;
const WIDTH: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

interface Accessor { bufferView?: number; byteOffset?: number; componentType: number; count: number; type: string; normalized?: boolean }
interface BufferView { buffer?: number; byteOffset?: number; byteLength: number; byteStride?: number }
interface GltfDocument {
  accessors?: Accessor[]; bufferViews?: BufferView[]; images?: { bufferView?: number; mimeType?: string }[]; textures?: { source?: number }[];
  materials?: { pbrMetallicRoughness?: { baseColorTexture?: { index: number } }; normalTexture?: { index: number } }[];
  meshes?: { primitives?: { attributes?: Record<string, number>; indices?: number; targets?: Record<string, number>[]; material?: number; mode?: number }[]; extras?: { targetNames?: unknown } }[];
  extras?: Record<string, unknown>;
}

export interface ParsedGlb {
  mesh: Reconstruction;
  /** Names of the morph targets (the six semantic sliders when `shape_keys=true`). */
  morphTargetNames: string[];
  /** The tangent-space normal map, when the asset has measured relief. */
  normalMap: string | null;
  extras: Record<string, unknown> | null;
}

function containers(buffer: ArrayBuffer): { json: GltfDocument; bin: DataView } {
  if (buffer.byteLength < 20) throw new GlbError('The GLB stream is too short.');
  const head = new DataView(buffer);
  if (head.getUint32(0, true) !== MAGIC) throw new GlbError('Not a GLB file (missing the glTF magic).');
  if (head.getUint32(4, true) !== 2) throw new GlbError(`Unsupported GLB container version ${head.getUint32(4, true)}.`);
  const length = head.getUint32(8, true);
  if (length !== buffer.byteLength) throw new GlbError('The GLB header length does not match the stream (truncated download?).');
  let json: GltfDocument | null = null;
  let bin: DataView | null = null;
  for (let at = 12; at < length;) {
    if (at + 8 > length) throw new GlbError('Truncated GLB chunk header.');
    const size = head.getUint32(at, true), kind = head.getUint32(at + 4, true);
    at += 8;
    if (size % 4 || at + size > length) throw new GlbError('A GLB chunk runs past the end of the file.');
    if (kind === CHUNK_JSON && !json) {
      try { json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, at, size))) as GltfDocument; } catch { throw new GlbError('The GLB JSON chunk is not valid JSON.'); }
    } else if (kind === CHUNK_BIN && !bin) bin = new DataView(buffer, at, size);
    at += size;
  }
  if (!json) throw new GlbError('The GLB has no JSON chunk.');
  return { json, bin: bin ?? new DataView(new ArrayBuffer(0)) };
}

/** Decode an accessor to a plain array of rows (or numbers for SCALAR). Honors `byteStride`; rejects out-of-range reads. */
function readAccessor(json: GltfDocument, bin: DataView, index: number): number[] | number[][] {
  const accessor = json.accessors?.[index];
  if (!accessor || accessor.bufferView === undefined) throw new GlbError(`Accessor ${index} is missing or sparse.`);
  const view = json.bufferViews?.[accessor.bufferView];
  const Component = COMPONENT[accessor.componentType as keyof typeof COMPONENT];
  const width = WIDTH[accessor.type];
  if (!view || !Component || !width) throw new GlbError(`Accessor ${index} uses an unsupported layout.`);
  const bytes = Component.BYTES_PER_ELEMENT;
  const stride = view.byteStride && view.byteStride > 0 ? view.byteStride : width * bytes;
  const start = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const end = start + (accessor.count > 0 ? (accessor.count - 1) * stride + width * bytes : 0);
  const viewEnd = (view.byteOffset ?? 0) + view.byteLength;
  // The accessor must stay inside its own buffer view, and the view inside the binary chunk.
  if (accessor.count < 0 || end > viewEnd || viewEnd > bin.byteLength) throw new GlbError(`Accessor ${index} reads past its buffer view.`);
  const read = (offset: number): number => {
    switch (accessor.componentType) {
      case 5126: return bin.getFloat32(offset, true);
      case 5125: return bin.getUint32(offset, true);
      case 5123: return bin.getUint16(offset, true);
      case 5122: return bin.getInt16(offset, true);
      case 5121: return bin.getUint8(offset);
      default: return bin.getInt8(offset);
    }
  };
  if (width === 1) return Array.from({ length: accessor.count }, (_, i) => read(start + i * stride));
  return Array.from({ length: accessor.count }, (_, i) => Array.from({ length: width }, (_, c) => read(start + i * stride + c * bytes)));
}

function imageDataUrl(json: GltfDocument, bin: DataView, textureIndex: number | undefined): string | null {
  if (textureIndex === undefined) return null;
  const source = json.textures?.[textureIndex]?.source;
  const image = source === undefined ? undefined : json.images?.[source];
  const view = image?.bufferView === undefined ? undefined : json.bufferViews?.[image.bufferView];
  if (!image || !view) return null;
  const start = view.byteOffset ?? 0;
  if (start + view.byteLength > bin.byteLength) throw new GlbError('An embedded image runs past the binary chunk.');
  const bytes = new Uint8Array(bin.buffer, bin.byteOffset + start, view.byteLength);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return `data:${image.mimeType === 'image/jpeg' ? 'image/jpeg' : 'image/png'};base64,${btoa(binary)}`;
}

export function parseGlb(buffer: ArrayBuffer): ParsedGlb {
  const { json, bin } = containers(buffer);
  const primitive = json.meshes?.[0]?.primitives?.[0];
  if (!primitive?.attributes || primitive.indices === undefined) throw new GlbError('The GLB has no indexed mesh.');
  if (primitive.mode !== undefined && primitive.mode !== 4) throw new GlbError('Only triangle meshes are supported.');
  const positionIndex = primitive.attributes.POSITION;
  if (positionIndex === undefined) throw new GlbError('The GLB mesh has no POSITION attribute.');

  const vertices = readAccessor(json, bin, positionIndex) as number[][];
  const flat = readAccessor(json, bin, primitive.indices) as number[];
  if (!vertices.length || flat.length < 3 || flat.length % 3) throw new GlbError('The GLB mesh has no usable triangles.');
  if (!vertices.every(v => v.every(Number.isFinite))) throw new GlbError('The GLB mesh contains NaN or Infinity.');
  const faces: number[][] = [];
  for (let i = 0; i < flat.length; i += 3) {
    const face = [flat[i], flat[i + 1], flat[i + 2]];
    if (face.some(index => !Number.isInteger(index) || index < 0 || index >= vertices.length)) throw new GlbError('A GLB triangle references a missing vertex.');
    faces.push(face);
  }
  const normals = primitive.attributes.NORMAL === undefined ? null : readAccessor(json, bin, primitive.attributes.NORMAL) as number[][];
  const rawUvs = primitive.attributes.TEXCOORD_0 === undefined ? null : readAccessor(json, bin, primitive.attributes.TEXCOORD_0) as number[][];
  // glTF's v origin is the top of the image; the viewer's TextureLoader convention (and the JSON route) has it at the bottom.
  const uvs = rawUvs && rawUvs.length === vertices.length ? rawUvs.map(([u, v]) => [u, 1 - v]) : null;

  const material = json.materials?.[primitive.material ?? 0];
  const texture = imageDataUrl(json, bin, material?.pbrMetallicRoughness?.baseColorTexture?.index);
  const normalMap = imageDataUrl(json, bin, material?.normalTexture?.index);
  const extras = json.extras && typeof json.extras === 'object' ? json.extras : null;
  const names = json.meshes?.[0]?.extras?.targetNames;
  const identity = (extras as { roop3d_identity_asset?: { provenance?: unknown } } | null)?.roop3d_identity_asset;

  return {
    mesh: {
      method: 'identity_asset', coordinate_system: 'opengl', vertices, faces, uvs, texture,
      normals: normals && normals.length === vertices.length ? normals : null,
      provenance: typeof identity?.provenance === 'string' ? identity.provenance : 'identity_asset',
    },
    morphTargetNames: Array.isArray(names) ? names.filter((name): name is string => typeof name === 'string') : [],
    normalMap, extras,
  };
}

// ------------------------------------------------------------------------------------------------ loading with fallback
export type MeshSource = 'glb' | 'json';
export interface LoadedMesh { mesh: Reconstruction; source: MeshSource; morphTargetNames: string[]; normalMap: string | null; bytes: number | null; fallbackReason: string | null }

export const identityMeshPath = (facesetId: string, query: string, format: 'glb' | 'json'): string =>
  `/facesets/${encodeURIComponent(facesetId)}/identity-asset/mesh${format === 'glb' ? '.glb' : ''}${query ? `?${query}` : ''}`;

/** Test seam: replaces `fetch` for the GLB request. */
export interface MeshLoaderDeps { fetchGlb?: (url: string, init: RequestInit) => Promise<Response>; json?: <T>(path: string, init: RequestInit) => Promise<T> }

/**
 * GLB first. Any failure (404, network error, malformed container) is remembered as `fallbackReason` and the JSON route is used
 * instead; only when both fail does the returned promise reject (with the JSON route's error, which is the more specific one).
 * A cancelled request (`AbortError`) is never retried.
 */
export async function loadIdentityMesh(facesetId: string, query: string, init: { signal?: AbortSignal; preferJson?: boolean } = {}, deps: MeshLoaderDeps = {}): Promise<LoadedMesh> {
  const fetchGlb = deps.fetchGlb ?? ((url, options) => fetch(url, options));
  const getJson = deps.json ?? (<T,>(path: string, options: RequestInit) => api<T>(path, options));
  let fallbackReason: string | null = init.preferJson ? 'JSON requested.' : null;
  if (!init.preferJson) {
    try {
      const response = await fetchGlb(`/api/v1${identityMeshPath(facesetId, query, 'glb')}`, { signal: init.signal, headers: { Accept: 'model/gltf-binary' } });
      if (!response.ok) {
        const body = await response.json().catch(() => null) as { error?: { message?: string } } | null;
        throw new ApiError(body?.error?.message ?? `The GLB route answered ${response.status}.`, null, response.status);
      }
      const buffer = await response.arrayBuffer();
      const parsed = parseGlb(buffer);
      return { mesh: parsed.mesh, source: 'glb', morphTargetNames: parsed.morphTargetNames, normalMap: parsed.normalMap, bytes: buffer.byteLength, fallbackReason: null };
    } catch (error) {
      if ((error as Error)?.name === 'AbortError') throw error;
      fallbackReason = (error as Error)?.message || 'The GLB could not be read.';
    }
  }
  const mesh = await getJson<Reconstruction>(identityMeshPath(facesetId, query, 'json'), { signal: init.signal });
  return { mesh, source: 'json', morphTargetNames: [], normalMap: null, bytes: null, fallbackReason };
}
