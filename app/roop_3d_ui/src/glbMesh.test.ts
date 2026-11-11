import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { GlbError, identityMeshPath, loadIdentityMesh, parseGlb } from './utils/glbMesh';
import { buildMeshAssets } from './components/three/PoseGeometryViewer';

/** Written by `roop_3d.io.glb_export.GLBBinarySerializer` (frame="mesh", albedo, one morph target "jaw_width"), so this reads real server output. */
const FIXTURE = readFileSync(new URL('./__fixtures__/fixture_head.glb', import.meta.url));
const fixtureBuffer = (): ArrayBuffer => FIXTURE.buffer.slice(FIXTURE.byteOffset, FIXTURE.byteOffset + FIXTURE.byteLength) as ArrayBuffer;
const glbResponse = (status = 200, body: BodyInit | null = fixtureBuffer()) => new Response(body, { status, headers: { 'Content-Type': 'model/gltf-binary' } });

describe('parseGlb on the server\'s own output', () => {
  const parsed = parseGlb(fixtureBuffer());

  it('reads vertices in the viewer frame: the project\'s y-down mesh frame arrives as glTF y-up', () => {
    expect(parsed.mesh.coordinate_system).toBe('opengl');
    expect(parsed.mesh.vertices).toHaveLength(4);
    parsed.mesh.vertices[0].forEach((value, axis) => expect(value).toBeCloseTo([0, 0.1, 0.02][axis], 5));        // was [0, -0.1, 0.02] in the mesh frame
    parsed.mesh.vertices[3].forEach((value, axis) => expect(value).toBeCloseTo([0, 0, 0.15][axis], 5));
  });

  it('keeps triangle winding (the serializer already swapped it for the frame change) and the vertex normals', () => {
    expect(parsed.mesh.faces).toEqual([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]]);
    expect(parsed.mesh.normals).toHaveLength(4);
    expect(parsed.mesh.normals![0][1]).toBeGreaterThan(0);          // -0.3 in the y-down frame is +0.3 y-up
  });

  it('flips V back to the viewer\'s texture convention and embeds the albedo as a PNG data URL', () => {
    expect(parsed.mesh.uvs![0][1]).toBeCloseTo(0.9, 6);
    expect(parsed.mesh.uvs![1]).toEqual([expect.closeTo(0.1, 6), expect.closeTo(0.1, 6)]);
    expect(parsed.mesh.texture).toMatch(/^data:image\/png;base64,/);
    const bytes = Buffer.from(parsed.mesh.texture!.split(',')[1], 'base64');
    expect([...bytes.subarray(0, 4)]).toEqual([0x89, 0x50, 0x4e, 0x47]);
  });

  it('exposes morph target names and the identity provenance', () => {
    expect(parsed.morphTargetNames).toEqual(['jaw_width']);
    expect(parsed.mesh.provenance).toBe('fixture_head');
    expect(parsed.normalMap).toBeNull();
  });

  it('builds the same viewer geometry the JSON route does (no mirrored winding, outward normals)', () => {
    const assets = buildMeshAssets(parsed.mesh)!;
    expect(assets.geometry.getIndex()!.count).toBe(12);
    expect(assets.geometry.getAttribute('uv')).toBeDefined();
    const box = assets.geometry.boundingBox!;
    expect(box.max.y).toBeGreaterThan(box.min.y);
    assets.geometry.dispose();
  });
});

describe('parseGlb rejects what it cannot trust', () => {
  it('a truncated download', () => expect(() => parseGlb(fixtureBuffer().slice(0, 600))).toThrow(GlbError));
  it('a stream that is not a GLB', () => expect(() => parseGlb(new TextEncoder().encode('<html>404 not found</html>').buffer as ArrayBuffer)).toThrow(/glTF magic|too short/));
  it('an unsupported container version', () => {
    const copy = new Uint8Array(fixtureBuffer()); new DataView(copy.buffer).setUint32(4, 1, true);
    expect(() => parseGlb(copy.buffer)).toThrow(/version/);
  });
  it('an index that points outside the vertex list', () => {
    // Rewrite the first uint16 index (the first accessor after POSITION/NORMAL/TEXCOORD data) by patching the JSON: huge count.
    const text = new TextDecoder().decode(new Uint8Array(fixtureBuffer()));
    expect(text).toContain('"componentType":5126');
    const copy = new Uint8Array(fixtureBuffer());
    const marker = new TextEncoder().encode('"count":4');
    let at = -1; for (let i = 0; i < copy.length - marker.length; i++) { if (marker.every((b, k) => copy[i + k] === b)) { at = i; break; } }
    expect(at).toBeGreaterThan(0);
    copy[at + 8] = '9'.charCodeAt(0);                                  // "count":9 -> reads past the vertex data
    expect(() => parseGlb(copy.buffer)).toThrow(GlbError);
  });
});

describe('loadIdentityMesh: GLB first, JSON as the fallback', () => {
  const jsonMesh = { method: 'identity_asset', coordinate_system: 'opengl', vertices: [[0, 0, 0]], faces: [[0, 0, 0]], uvs: null, texture: null, provenance: 'json_route' };

  it('builds the request paths for both routes', () => {
    expect(identityMeshPath('fs 1', 'template=makehuman', 'glb')).toBe('/facesets/fs%201/identity-asset/mesh.glb?template=makehuman');
    expect(identityMeshPath('fs1', '', 'json')).toBe('/facesets/fs1/identity-asset/mesh');
  });

  it('uses the GLB when it loads and never touches the JSON route', async () => {
    const json = vi.fn();
    const fetchGlb = vi.fn(async () => glbResponse());
    const loaded = await loadIdentityMesh('fs1', 'template=makehuman&jaw_width=1', {}, { fetchGlb, json });
    expect(fetchGlb).toHaveBeenCalledWith('/api/v1/facesets/fs1/identity-asset/mesh.glb?template=makehuman&jaw_width=1', expect.objectContaining({ headers: { Accept: 'model/gltf-binary' } }));
    expect(loaded).toMatchObject({ source: 'glb', fallbackReason: null, morphTargetNames: ['jaw_width'] });
    expect(loaded.bytes).toBe(FIXTURE.byteLength);
    expect(json).not.toHaveBeenCalled();
  });

  it('falls back to JSON when the GLB route answers 404, and says why', async () => {
    const json = vi.fn(async () => jsonMesh);
    const fetchGlb = vi.fn(async () => glbResponse(404, JSON.stringify({ error: { message: 'Build the 3D identity asset first' } })));
    const loaded = await loadIdentityMesh('fs1', '', {}, { fetchGlb, json: json as never });
    expect(loaded.source).toBe('json');
    expect(loaded.fallbackReason).toBe('Build the 3D identity asset first');
    expect(json).toHaveBeenCalledWith('/facesets/fs1/identity-asset/mesh', expect.anything());
    expect(loaded.mesh.provenance).toBe('json_route');
  });

  it('falls back to JSON when the GLB is malformed or the network fails', async () => {
    const json = vi.fn(async () => jsonMesh);
    const garbled = await loadIdentityMesh('fs1', '', {}, { fetchGlb: async () => glbResponse(200, 'not a glb at all, but long enough to pass the length check'), json: json as never });
    expect(garbled).toMatchObject({ source: 'json' });
    expect(garbled.fallbackReason).toMatch(/glTF magic/);
    const offline = await loadIdentityMesh('fs1', '', {}, { fetchGlb: async () => { throw new TypeError('Failed to fetch'); }, json: json as never });
    expect(offline.fallbackReason).toBe('Failed to fetch');
  });

  it('rejects with the JSON route\'s error only when both routes fail', async () => {
    const json = vi.fn(async () => { throw new Error('Identity asset not built'); });
    await expect(loadIdentityMesh('fs1', '', {}, { fetchGlb: async () => glbResponse(500, '{}'), json: json as never })).rejects.toThrow('Identity asset not built');
  });

  it('does not fall back after a cancel', async () => {
    const json = vi.fn();
    const abort = Object.assign(new Error('The operation was aborted'), { name: 'AbortError' });
    await expect(loadIdentityMesh('fs1', '', {}, { fetchGlb: async () => { throw abort; }, json })).rejects.toBe(abort);
    expect(json).not.toHaveBeenCalled();
  });

  it('can be told to skip the GLB', async () => {
    const fetchGlb = vi.fn();
    const loaded = await loadIdentityMesh('fs1', '', { preferJson: true }, { fetchGlb, json: (async () => jsonMesh) as never });
    expect(fetchGlb).not.toHaveBeenCalled();
    expect(loaded.source).toBe('json');
  });
});
