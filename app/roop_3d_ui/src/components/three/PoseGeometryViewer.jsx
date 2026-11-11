import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import GuardedCanvas from './GuardedCanvas';
import WebGLFallbackInspector from './WebGLFallbackInspector';
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import { useFaceSelectionStore } from '../../useFaceSelectionStore';
import { reducedMotion } from '../../utils/motionPrefs';
import { renderModeInfo } from '../workstation/renderModes';
import {
  BUFFER_VIEWS, DEFAULT_SH_LIGHT, DEPTH_FRAGMENT_SHADER, DEPTH_VERTEX_SHADER, JAW_MAX_DEG, SH_FRAGMENT_SHADER, SH_VERTEX_SHADER,
  buildJawRig, eyeAnchors, openJaw, shLightCoefficientsRGB,
} from './viewportTools';
import {
  CAVITY_FRAGMENT_SHADER, CAVITY_VERTEX_SHADER, DEFAULT_FOCAL_MM, FOCAL_RANGE_MM, clampFocal, computeVertexCavity, dollyFactor, environmentToViewerSH, exposeEnvironment,
  horizontalFovDeg, verticalFovDeg,
} from './studioTools';
import { useStudioEnvironments } from './useStudioEnvironments';
import { GazeGizmo, JawGizmo, LightGizmo } from './ViewportGizmos';
import { PrecisionRangeSlider } from '../ui-layouts';

/**
 * @typedef {{
 *   index: number;
 *   position: [number, number, number];
 *   normal: THREE.Vector3;
 *   yawDeg: number;
 *   pitchDeg: number;
 *   sampleCount: number;
 * }} SphereNode
 */

/**
 * Generates an evenly distributed Fibonacci sphere lattice.
 * @param {number} numPoints
 * @param {number} radius
 * @returns {SphereNode[]}
 */
export function generateFibonacciSphere(numPoints = 96, radius = 1.35) {
  const points = [];
  const phi = (1 + Math.sqrt(5)) / 2;
  const goldenAngle = 2 * Math.PI * (1 - 1 / phi);

  for (let i = 0; i < numPoints; i++) {
    const y = 1 - (i / (numPoints - 1)) * 2;
    const radiusAtY = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = goldenAngle * i;

    const x = Math.cos(theta) * radiusAtY;
    const z = Math.sin(theta) * radiusAtY;

    const normal = new THREE.Vector3(x, y, z).normalize();
    const pos = normal.clone().multiplyScalar(radius);

    const yawDeg = (Math.atan2(x, z) * 180) / Math.PI;
    const pitchDeg = (Math.asin(Math.max(-1, Math.min(1, y))) * 180) / Math.PI;

    points.push({
      index: i,
      position: [pos.x, pos.y, pos.z],
      normal,
      yawDeg,
      pitchDeg,
      sampleCount: 0,
    });
  }
  return points;
}

/**
 * Projects a set of [yaw, pitch] views onto the Fibonacci lattice.
 * @param {SphereNode[]} lattice
 * @param {Array<{ yaw: number; pitch: number; roll?: number }>} views
 * @returns {SphereNode[]}
 */
export function projectViewsToLattice(lattice, views = []) {
  const updated = lattice.map(node => ({ ...node, sampleCount: 0 }));
  if (!views || views.length === 0) return updated;

  for (const view of views) {
    const yawRad = (view.yaw * Math.PI) / 180;
    const pitchRad = (view.pitch * Math.PI) / 180;

    const sampleVec = new THREE.Vector3(
      Math.sin(yawRad) * Math.cos(pitchRad),
      Math.sin(pitchRad),
      Math.cos(yawRad) * Math.cos(pitchRad)
    ).normalize();

    let bestIdx = 0;
    let bestDot = -Infinity;

    for (let i = 0; i < updated.length; i++) {
      const dot = sampleVec.dot(updated[i].normal);
      if (dot > bestDot) {
        bestDot = dot;
        bestIdx = i;
      }
    }

    if (bestDot > 0.7) {
      updated[bestIdx].sampleCount += 1;
    }
  }

  return updated;
}

/** Exact 12 yaw x 8 pitch hemisphere grid used by the backend bin contract. */
export function generatePoseGrid(radius = 1.35) {
  const nodes = [];
  for (let pitchBin = 0; pitchBin < 8; pitchBin++) {
    const pitchDeg = -60 + (pitchBin + 0.5) * 15;
    for (let yawBin = 0; yawBin < 12; yawBin++) {
      const yawDeg = -90 + (yawBin + 0.5) * 15;
      const yaw = yawDeg * Math.PI / 180, pitch = pitchDeg * Math.PI / 180;
      const normal = new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)).normalize();
      nodes.push({ index: pitchBin * 12 + yawBin, yawDeg, pitchDeg, normal,
        position: normal.clone().multiplyScalar(radius).toArray(), sampleCount: 0 });
    }
  }
  return nodes;
}

/**
 * Procedural low-poly canonical 3D head geometry centered at [0, 0, 0].
 */
export function createCanonicalHeadGeometry() {
  const geom = new THREE.BufferGeometry();

  // Low-poly canonical head vertices
  // Features: cranium, brow, eye sockets, nose bridge & tip, cheeks, mouth, jaw, chin
  const baseVertices = [
    // Cranium top / back
    [0.0, 0.85, 0.0],     // 0: Crown
    [-0.35, 0.7, 0.25],   // 1: Forehead left
    [0.35, 0.7, 0.25],    // 2: Forehead right
    [0.0, 0.72, 0.45],    // 3: Forehead center
    [-0.45, 0.5, -0.2],   // 4: Temple back left
    [0.45, 0.5, -0.2],    // 5: Temple back right
    [0.0, 0.55, -0.55],   // 6: Occipital back

    // Brow and eyes
    [-0.28, 0.32, 0.52],  // 7: Brow left
    [0.28, 0.32, 0.52],   // 8: Brow right
    [0.0, 0.35, 0.58],    // 9: Glabella (between eyes)
    [-0.24, 0.16, 0.44],  // 10: Eye socket left
    [0.24, 0.16, 0.44],   // 11: Eye socket right
    [-0.48, 0.18, 0.1],   // 12: Cheekbone outer left
    [0.48, 0.18, 0.1],    // 13: Cheekbone outer right

    // Nose
    [0.0, 0.12, 0.68],    // 14: Nose bridge
    [0.0, -0.06, 0.78],   // 15: Nose tip
    [-0.12, -0.08, 0.62], // 16: Nostril left
    [0.12, -0.08, 0.62],  // 17: Nostril right

    // Mouth and cheeks
    [0.0, -0.22, 0.58],   // 18: Upper lip
    [-0.18, -0.32, 0.52], // 19: Mouth corner left
    [0.18, -0.32, 0.52],  // 20: Mouth corner right
    [0.0, -0.38, 0.54],   // 21: Lower lip
    [-0.42, -0.15, 0.25], // 22: Cheek lower left
    [0.42, -0.15, 0.25],  // 23: Cheek lower right

    // Jaw and chin
    [0.0, -0.62, 0.48],   // 24: Chin tip
    [-0.18, -0.6, 0.36],  // 25: Chin left
    [0.18, -0.6, 0.36],   // 26: Chin right
    [-0.38, -0.42, -0.05],// 27: Jaw angle left
    [0.38, -0.42, -0.05], // 28: Jaw angle right
    [0.0, -0.85, -0.1],   // 29: Neck base
  ];

  // Triangles connecting vertices
  const indices = [
    // Forehead / Crown
    0, 1, 3,  0, 3, 2,  0, 4, 1,  0, 2, 5,  0, 6, 4,  0, 5, 6,
    // Brow
    3, 1, 7,  3, 7, 9,  3, 9, 8,  3, 8, 2,  1, 4, 12, 1, 12, 7,  2, 8, 13, 2, 13, 5,
    // Eye sockets
    7, 12, 10, 7, 10, 9, 8, 9, 11, 8, 11, 13, 9, 10, 14, 9, 14, 11,
    // Nose
    14, 10, 16, 14, 16, 15, 14, 15, 17, 14, 17, 11,
    // Cheeks & Mouth
    10, 12, 22, 10, 22, 16, 11, 17, 23, 11, 23, 13,
    15, 16, 18, 15, 18, 17, 16, 22, 19, 16, 19, 18, 17, 18, 20, 17, 20, 23,
    18, 19, 21, 18, 21, 20,
    // Chin & Jaw
    19, 22, 27, 19, 27, 25, 19, 25, 21, 20, 21, 26, 20, 26, 28, 20, 28, 23,
    21, 25, 24, 21, 24, 26,
    // Back / Neck
    4, 6, 27, 6, 5, 28, 27, 6, 29, 6, 28, 29, 25, 27, 29, 28, 26, 29, 25, 29, 24, 26, 24, 29,
  ];

  const positions = new Float32Array(baseVertices.flat());
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  geom.center(); // Center exactly at [0, 0, 0]

  return geom;
}


/**
 * 3D Landmark positions (projecting from surface of canonical head).
 */
const CANONICAL_LANDMARKS = [
  { name: 'Left Eye', pos: [-0.24, 0.16, 0.46] },
  { name: 'Right Eye', pos: [0.24, 0.16, 0.46] },
  { name: 'Nose Tip', pos: [0.0, -0.06, 0.82] },
  { name: 'Mouth Left', pos: [-0.18, -0.32, 0.54] },
  { name: 'Mouth Right', pos: [0.18, -0.32, 0.54] },
  { name: 'Chin', pos: [0.0, -0.62, 0.50] },
];

const SPHERE_RADIUS = 1.35;
const CAMERA_DISTANCE = 3.8;
const FRONTAL_CAMERA = new THREE.Vector3(0, 0, CAMERA_DISTANCE);
const WORLD_UP = new THREE.Vector3(0, 1, 0);
const CYAN = '#22d3ee';
/** Landmark vertices of the canonical template: indices 0..67 of every `mesh_2_5d` (see multiview_fit.identity_template). */
export const TEMPLATE_LANDMARK_COUNT = 68;

/**
 * Per coordinate frame: the sign applied to (x, y, z) to reach three.js (+x right, +y up, +z toward the viewer).
 *  - `mesh`   : the backend mesh frame of `mesh_2_5d` / `mesh_full_3d` (+x image-right, +y DOWN, +z toward the viewer);
 *  - `opencv` : a camera frame (+y down, +z away from the viewer);
 *  - `opengl` : glTF / three.js, used as is.
 */
const FRAME_SIGNS = { mesh: [1, -1, 1], opencv: [1, -1, -1], opengl: [1, 1, 1] };

/**
 * Turns a reconstruction into a three.js geometry fitted to the viewer (centred, bounding radius 0.95).
 * A frame with an odd number of mirrored axes reverses triangle winding, so the winding is swapped to keep outward faces
 * counter-clockwise. Returns null when the mesh has no usable triangles.
 * @param {import('../../contracts').Reconstruction | null | undefined} mesh
 * @param {{ landmarkCount?: number }} [options] `landmarkCount` leading vertices are returned as lifted landmark points.
 * @returns {{ geometry: THREE.BufferGeometry, landmarks: Float32Array | null } | null}
 */
export function buildMeshAssets(mesh, { landmarkCount = 0 } = {}) {
  if (!mesh?.vertices?.length || !mesh?.faces?.length) return null;
  const signs = FRAME_SIGNS[mesh.coordinate_system] ?? FRAME_SIGNS.opengl;
  const mirrored = signs[0] * signs[1] * signs[2] < 0;
  const vertexCount = mesh.vertices.length;
  const faces = mesh.faces.filter(face => face.length === 3 && face.every(index => Number.isInteger(index) && index >= 0 && index < vertexCount));
  if (!faces.length) return null;
  const geometry = new THREE.BufferGeometry();
  const corner = (face, slot) => (mirrored && slot > 0 ? face[3 - slot] : face[slot]);
  let landmarkSource = null;

  if (Array.isArray(mesh.face_uvs)) {
    // Per-corner UVs (imported meshes with seams): vertices are duplicated per triangle corner.
    const positions = [];
    const uvs = [];
    faces.forEach((face, faceIndex) => {
      for (let slot = 0; slot < 3; slot++) {
        const vertexIndex = corner(face, slot);
        const vertex = mesh.vertices[vertexIndex];
        positions.push(vertex[0] * signs[0], vertex[1] * signs[1], vertex[2] * signs[2]);
        const originalSlot = mirrored && slot > 0 ? 3 - slot : slot;
        const uv = mesh.uvs?.[mesh.face_uvs?.[faceIndex]?.[originalSlot] ?? vertexIndex];
        if (uv) uvs.push(uv[0], uv[1]);
      }
    });
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    if (uvs.length === (positions.length / 3) * 2) geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  } else {
    // One vertex per mesh vertex: smooth normals and shared UVs.
    const positions = new Float32Array(vertexCount * 3);
    for (let i = 0; i < vertexCount; i++) {
      const vertex = mesh.vertices[i];
      positions[i * 3] = vertex[0] * signs[0];
      positions[i * 3 + 1] = vertex[1] * signs[1];
      positions[i * 3 + 2] = vertex[2] * signs[2];
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    if (mesh.normals?.length === vertexCount) {
      // Backend normals are welded across UV seams (no creases); reflecting them with the axis signs keeps them outward.
      const normals = new Float32Array(vertexCount * 3);
      for (let i = 0; i < vertexCount; i++) {
        normals[i * 3] = mesh.normals[i][0] * signs[0];
        normals[i * 3 + 1] = mesh.normals[i][1] * signs[1];
        normals[i * 3 + 2] = mesh.normals[i][2] * signs[2];
      }
      geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    }
    if (mesh.uvs?.length === vertexCount) {
      const uvs = new Float32Array(vertexCount * 2);
      for (let i = 0; i < vertexCount; i++) { uvs[i * 2] = mesh.uvs[i][0]; uvs[i * 2 + 1] = mesh.uvs[i][1]; }
      geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    }
    const index = new Uint32Array(faces.length * 3);
    faces.forEach((face, faceIndex) => { for (let slot = 0; slot < 3; slot++) index[faceIndex * 3 + slot] = corner(face, slot); });
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    if (landmarkCount > 0) landmarkSource = Math.min(landmarkCount, vertexCount);
  }

  if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  const center = geometry.boundingBox.getCenter(new THREE.Vector3());
  geometry.translate(-center.x, -center.y, -center.z);
  geometry.computeBoundingSphere();
  const radius = geometry.boundingSphere?.radius ?? 0;
  const scale = radius > 0 ? 0.95 / radius : 1;
  geometry.scale(scale, scale, scale);

  let landmarks = null;
  if (landmarkSource) {
    // Lift the points a hair along the surface normal so they are not buried in (or z-fighting with) the skin.
    const position = geometry.getAttribute('position');
    const normal = geometry.getAttribute('normal');
    landmarks = new Float32Array(landmarkSource * 3);
    for (let i = 0; i < landmarkSource; i++) {
      landmarks[i * 3] = position.getX(i) + normal.getX(i) * 0.012;
      landmarks[i * 3 + 1] = position.getY(i) + normal.getY(i) * 0.012;
      landmarks[i * 3 + 2] = position.getZ(i) + normal.getZ(i) * 0.012;
    }
  }
  geometry.computeBoundingSphere();
  return { geometry, landmarks };
}

/**
 * A `mesh_2_5d` / `mesh_full_3d` payload as a viewer reconstruction (backend mesh frame).
 * @param {import('../../contracts').ReconstructedMesh | null | undefined} value
 * @param {'tps_2_5d' | 'identity_asset'} method
 * @param {string} [fallbackTexture] a `data:image/png` URL used when the mesh carries no texture of its own
 * @returns {import('../../contracts').Reconstruction | null}
 */
export function payloadToReconstruction(value, method, fallbackTexture) {
  if (!value?.vertices?.length || !value?.indices?.length) return null;
  return {
    method, coordinate_system: 'mesh', vertices: value.vertices, faces: value.indices,
    uvs: value.uvs?.length === value.vertices.length ? value.uvs : null,
    normals: value.normals?.length === value.vertices.length ? value.normals : null,
    texture: value.texture_base64 ? `data:image/png;base64,${value.texture_base64}` : fallbackTexture ?? null,
    provenance: value.reconstruction_level,
  };
}

/**
 * Track and detected-face poses are stored in the legacy frame (pitch + = down); coverage points, the sphere and the
 * 3D head use the standard frame (pitch + = up). Convert exactly once, here.
 * @param {{ yaw?: number; pitch?: number; roll?: number } | null | undefined} pose
 */
export function legacyPoseToStandard(pose) {
  return pose ? { yaw: pose.yaw ?? 0, pitch: -(pose.pitch ?? 0), roll: pose.roll ?? 0 } : null;
}

/** `+12.3° Y · -4.0° P`, never `-0.0`. */
export function formatPose(pose) {
  const part = value => {
    const rounded = Math.abs(value) < 0.05 ? 0 : value;
    return `${rounded >= 0 ? '+' : ''}${rounded.toFixed(1)}`;
  };
  return `${part(pose.yaw)}° Y · ${part(pose.pitch)}° P`;
}

/** Direction (unit vector, three.js frame) a head faces at the given standard-frame yaw/pitch. */
export function poseDirection(yawDeg, pitchDeg) {
  const yaw = (yawDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;
  return [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
}

/**
 * Rotation that turns a head facing +z to the given standard-frame pose: its nose then points exactly along
 * `poseDirection(yaw, pitch)`, i.e. at the matching dot on the orientation sphere.
 */
export function poseQuaternion(pose, out = new THREE.Quaternion()) {
  const euler = new THREE.Euler((-(pose?.pitch ?? 0) * Math.PI) / 180, ((pose?.yaw ?? 0) * Math.PI) / 180, (-(pose?.roll ?? 0) * Math.PI) / 180, 'YXZ');
  return out.setFromEuler(euler);
}

/**
 * Unit vector from the head toward a light placed at `yawDeg` (+ = image right) and `pitchDeg` (+ = above), in the viewer frame
 * (+x right, +y up, +z toward the viewer). yaw 0 / pitch 0 is a light at the camera.
 * @returns {[number, number, number]}
 */
export function lightDirection(yawDeg, pitchDeg) {
  const yaw = (yawDeg * Math.PI) / 180, pitch = (pitchDeg * Math.PI) / 180;
  return [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
}

const clamp01 = value => (Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0);

/**
 * The 3D light rig for the head viewer from the relighting controls. `strength` scales the key light, `ambient` is the ambient fill
 * (the SH band-0 term), `specular` is the skin sheen (lower roughness + a stronger specular lobe).
 * @param {{ yaw?: number; pitch?: number; strength?: number; ambient?: number; specular?: number } | null | undefined} lighting
 */
export function lightingRig(lighting) {
  const key = lightDirection(lighting?.yaw ?? 35, lighting?.pitch ?? 25);
  const specular = clamp01(lighting?.specular ?? 0.25);
  return {
    keyPosition: key.map(value => value * 5),
    keyIntensity: 0.4 + 2.6 * clamp01(lighting?.strength ?? 0.8),
    ambientIntensity: 0.1 + 1.1 * clamp01(lighting?.ambient ?? 0.3),
    roughness: 0.85 - 0.5 * specular,
    specularIntensity: 0.2 + 1.6 * specular,
  };
}

/**
 * How the geometry morph controls show in the viewer: `blend` is the source-mesh weight 0..1 (0 = the target-like estimated 2.5D proxy,
 * 1 = the source's 3D mesh) and the target proxy ghost fades out as it rises; `hairline` (-1..1 of the head height) moves the
 * forehead / hairline boundary ring (as a fraction of 8% of the head height).
 * @param {{ blend?: number; hairline?: number } | null | undefined} morph
 */
export function morphRig(morph) {
  const blend = clamp01(morph?.blend ?? 1);
  const hairline = Number.isFinite(morph?.hairline) ? Math.min(1, Math.max(-1, morph.hairline)) : 0;
  return { blend, targetGhostOpacity: (1 - blend) * 0.7, hairlineShift: hairline };
}

/** Releases GPU resources R3F does not own. */
function useDisposable(resource) {
  useEffect(() => () => { resource?.dispose?.(); }, [resource]);
}

/** A texture from a local `data:image/` URL (the only kind the local API produces). */
function useDataTexture(url) {
  const texture = useMemo(() => {
    if (typeof url !== 'string' || !url.startsWith('data:image/')) return null;
    const value = new THREE.TextureLoader().load(url);
    value.colorSpace = THREE.SRGBColorSpace;
    value.anisotropy = 4;
    return value;
  }, [url]);
  useDisposable(texture);
  return texture;
}

/** Shared geometry for the small landmark spheres. */
const LANDMARK_GEOMETRY = new THREE.SphereGeometry(0.016, 8, 8);

/** Idle reference: the canonical head as a faint wireframe that turns slowly (held still under reduced motion). */
function ReferenceHead() {
  const group = useRef(null);
  const geometry = useMemo(() => createCanonicalHeadGeometry(), []);
  useDisposable(geometry);
  useFrame((_, delta) => {
    if (group.current && !reducedMotion()) group.current.rotation.y += delta * 0.35;
  });
  return (
    <group ref={group} name="reference-head">
      <mesh geometry={geometry}>
        <meshBasicMaterial color={CYAN} wireframe transparent opacity={0.28} />
      </mesh>
      {CANONICAL_LANDMARKS.map(landmark => (
        <mesh key={landmark.name} position={landmark.pos} geometry={LANDMARK_GEOMETRY} scale={1.4}>
          <meshBasicMaterial color={CYAN} transparent opacity={0.7} />
        </mesh>
      ))}
    </group>
  );
}

/**
 * Albedo / normals / depth / cavity / SH-relit surface for the buffer switcher; all computed in the browser from the mesh being shown.
 * `shCoefficients` (nine RGB triples in the viewer's SH basis) overrides the manual key light, which is how a studio environment is previewed.
 */
function BufferMaterial({ kind, texture, shLight, shCoefficients = null }) {
  const camera = useThree(state => state.camera);
  const material = useMemo(() => {
    if (kind === 'depth') {
      return new THREE.ShaderMaterial({ vertexShader: DEPTH_VERTEX_SHADER, fragmentShader: DEPTH_FRAGMENT_SHADER, side: THREE.DoubleSide,
        uniforms: { uCenter: { value: CAMERA_DISTANCE }, uRadius: { value: 1.0 } } });
    }
    if (kind === 'cavity') {
      return new THREE.ShaderMaterial({ vertexShader: CAVITY_VERTEX_SHADER, fragmentShader: CAVITY_FRAGMENT_SHADER, side: THREE.DoubleSide });
    }
    if (kind === 'sh') {
      return new THREE.ShaderMaterial({ vertexShader: SH_VERTEX_SHADER, fragmentShader: SH_FRAGMENT_SHADER, side: THREE.DoubleSide,
        uniforms: { uSH: { value: Array.from({ length: 9 }, () => new THREE.Vector3()) }, uMap: { value: null }, uHasMap: { value: 0 }, uFlatColor: { value: new THREE.Color('#c9b8a6') } } });
    }
    return null;
  }, [kind]);
  useDisposable(material);
  useEffect(() => {
    if (kind !== 'sh' || !material) return;
    const rows = shCoefficients ?? shLightCoefficientsRGB(shLight);
    material.uniforms.uSH.value = rows.map(row => new THREE.Vector3(row[0], row[1], row[2]));
    material.uniforms.uMap.value = texture ?? null;
    material.uniforms.uHasMap.value = texture ? 1 : 0;
  }, [kind, material, shLight, shCoefficients, texture]);
  useFrame(() => { if (kind === 'depth' && material) material.uniforms.uCenter.value = camera.position.length(); });
  if (kind === 'wireframe') return <meshBasicMaterial color="#0b1220" side={THREE.DoubleSide} polygonOffset polygonOffsetFactor={1} polygonOffsetUnits={1} />;
  if (kind === 'albedo') return <meshBasicMaterial key={texture ? 'albedo-map' : 'albedo-flat'} map={texture} color={texture ? '#ffffff' : '#c9b8a6'} side={THREE.DoubleSide} />;
  if (kind === 'normals') return <meshNormalMaterial side={THREE.DoubleSide} />;
  return material ? <primitive object={material} attach="material" /> : null;
}

/**
 * The reconstructed head. It follows the target's pose (slerped) so the swap geometry is shown exactly as the pipeline
 * will pose it; the orbit camera moves independently around it.
 */
function ReconstructedHead({ pose, assets, texture, volumetric, wireframe, landmarks, lighting = null, morph = null, targetAssets = null,
  bufferView = 'lit', shLight = DEFAULT_SH_LIGHT, shCoefficients = null, tools = null, gizmos = null }) {
  const group = useRef(null);
  const jawDeg = tools?.jawDeg ?? 0;
  // Cavity buffer: the per-vertex concavity attribute must exist before the first frame that uses the shader, so it is attached while
  // rendering (idempotent, belongs to the geometry and is released with it).
  useMemo(() => {
    const geometry = assets?.geometry;
    if (bufferView !== 'cavity' || !geometry || geometry.getAttribute('aCavity')) return;
    const position = geometry.getAttribute('position');
    geometry.setAttribute('aCavity', new THREE.BufferAttribute(computeVertexCavity(position.array, geometry.getIndex()?.array ?? null), 1));
  }, [assets, bufferView]);
  const rig = useMemo(() => (lighting ? lightingRig(lighting) : null), [lighting]);
  const shape = useMemo(() => (morph ? morphRig(morph) : null), [morph]);
  // Forehead / hairline boundary ring: an ellipse around the head at ~22% below the crown, shifted by the offset control.
  const hairline = useMemo(() => {
    if (!shape) return null;
    const box = assets.geometry.boundingBox;
    if (!box) return null;
    const height = box.max.y - box.min.y, y = box.max.y - 0.22 * height + shape.hairlineShift * 0.08 * height;
    const rx = Math.max(0.05, (box.max.x - box.min.x) / 2), rz = Math.max(0.05, (box.max.z - box.min.z) / 2);
    const fall = Math.max(0.2, 1 - Math.abs((y - (box.max.y + box.min.y) / 2) / (height / 2)) ** 2);   // narrower toward the crown
    const points = Array.from({ length: 65 }, (_, i) => { const angle = (i / 64) * Math.PI * 2; return new THREE.Vector3(Math.sin(angle) * rx * Math.sqrt(fall), y, Math.cos(angle) * rz * Math.sqrt(fall)); });
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [assets, shape]);
  useDisposable(hairline);
  const current = useRef(new THREE.Quaternion());
  const target = useMemo(() => new THREE.Quaternion(), []);

  useEffect(() => { poseQuaternion(pose, target); }, [pose, target]);

  useFrame((_, delta) => {
    if (!group.current) return;
    // Reduce-motion: jump to the pose instead of easing toward it.
    current.current.slerp(target, reducedMotion() ? 1 : Math.min(1, delta * 12));
    group.current.quaternion.copy(current.current);
  });

  // Jaw hinge: the geometry is edited in place from a copy of its rest pose (so opening then closing restores it exactly).
  const rest = useMemo(() => {
    const position = assets.geometry.getAttribute('position'), normal = assets.geometry.getAttribute('normal');
    return { position: Float32Array.from(position.array), normal: normal ? Float32Array.from(normal.array) : null, landmarks: assets.landmarks ?? null };
  }, [assets]);
  const jawRig = useMemo(() => buildJawRig(rest.position), [rest]);
  const eyes = useMemo(() => eyeAnchors(rest.position, rest.landmarks), [rest]);
  useEffect(() => {
    const position = assets.geometry.getAttribute('position');
    if (!jawRig || !position) return;
    openJaw(rest.position, position.array, jawRig, jawDeg);
    position.needsUpdate = true;
    const normal = assets.geometry.getAttribute('normal');
    if (jawDeg === 0 && normal && rest.normal) { normal.array.set(rest.normal); normal.needsUpdate = true; } else assets.geometry.computeVertexNormals();
  }, [assets, rest, jawRig, jawDeg]);

  const landmarkPoints = useMemo(() => {
    if (!assets?.landmarks) return [];
    let source = assets.landmarks;
    if (jawRig && jawDeg !== 0) { source = new Float32Array(assets.landmarks.length); openJaw(assets.landmarks, source, jawRig, jawDeg); }
    const points = [];
    for (let i = 0; i < source.length; i += 3) points.push([source[i], source[i + 1], source[i + 2]]);
    return points;
  }, [assets, jawRig, jawDeg]);

  return (
    <group ref={group}>
      <mesh geometry={assets.geometry} name={`head-${bufferView}`}>
        {bufferView !== 'lit' ? (
          <BufferMaterial kind={bufferView} texture={texture} shLight={shLight} shCoefficients={shCoefficients} />
        ) : rig ? (
          <meshPhysicalMaterial
            key={texture ? 'textured-lit' : 'flat-lit'}
            map={texture}
            color={texture ? '#ffffff' : '#9fd8c8'}
            roughness={rig.roughness}
            metalness={0}
            specularIntensity={rig.specularIntensity}
            side={THREE.DoubleSide}
          />
        ) : (
          <meshStandardMaterial
            key={texture ? 'textured' : 'flat'}
            map={texture}
            color={texture ? '#ffffff' : '#9fd8c8'}
            roughness={volumetric ? 0.62 : 0.7}
            metalness={0}
            side={THREE.DoubleSide}
          />
        )}
      </mesh>
      {targetAssets && shape && shape.targetGhostOpacity > 0.01 && (
        <mesh geometry={targetAssets.geometry} name="target-geometry-ghost">
          <meshBasicMaterial color="#fbbf24" wireframe transparent opacity={shape.targetGhostOpacity} depthWrite={false} />
        </mesh>
      )}
      {hairline && (
        <lineLoop geometry={hairline} name="hairline-boundary">
          <lineBasicMaterial color="#fbbf24" transparent opacity={0.9} depthTest={false} />
        </lineLoop>
      )}
      {(wireframe || bufferView === 'wireframe') && (
        <mesh geometry={assets.geometry} name="wireframe-overlay">
          <meshBasicMaterial color={CYAN} wireframe transparent opacity={bufferView === 'wireframe' ? 0.9 : 0.22} depthWrite={false}
            polygonOffset polygonOffsetFactor={-1} polygonOffsetUnits={-1} />
        </mesh>
      )}
      {landmarks && landmarkPoints.map((point, index) => (
        <mesh key={index} position={point} geometry={LANDMARK_GEOMETRY}>
          <meshBasicMaterial color={CYAN} />
        </mesh>
      ))}
      {tools && gizmos?.jaw && jawRig && <JawGizmo rig={jawRig} theta={jawDeg} onChange={tools.onJaw} enabled={!!tools.trackId} />}
      {tools && gizmos?.gaze && eyes && <GazeGizmo eyes={eyes} gaze={tools.gaze} onChange={tools.onGaze} enabled={!!tools.trackId} />}
    </group>
  );
}

/** An arrow from the key light toward the head: the in-viewer light gizmo (amber while the light is manually overridden). */
function LightArrow({ position, manual }) {
  const arrow = useMemo(() => {
    const from = new THREE.Vector3(...position).normalize().multiplyScalar(1.9);
    return new THREE.ArrowHelper(from.clone().negate().normalize(), from, 0.85, manual ? 0xf59e0b : 0x38bdf8, 0.22, 0.12);
  }, [position, manual]);
  useEffect(() => () => { arrow.dispose(); }, [arrow]);
  return <primitive object={arrow} name="key-light-arrow" />;
}

const ringPoints = (radius, y, segments = 96) => {
  const points = [];
  for (let i = 0; i < segments; i++) {
    const angle = (i / segments) * Math.PI * 2;
    points.push(new THREE.Vector3(Math.cos(angle) * radius, y, Math.sin(angle) * radius));
  }
  return points;
};

/** Geodesic guide rings: the three great circles plus latitude rings every 30 degrees of pitch. */
function OrbitalRings({ radius }) {
  const rings = useMemo(() => {
    const great = new THREE.BufferGeometry().setFromPoints(ringPoints(radius, 0));
    const latitudes = [-60, -30, 30, 60].map(degrees => {
      const rad = (degrees * Math.PI) / 180;
      return new THREE.BufferGeometry().setFromPoints(ringPoints(radius * Math.cos(rad), radius * Math.sin(rad)));
    });
    return { great, latitudes };
  }, [radius]);
  useEffect(() => () => { rings.great.dispose(); rings.latitudes.forEach(geometry => geometry.dispose()); }, [rings]);
  return (
    <group>
      <lineLoop geometry={rings.great}><lineBasicMaterial color="#38bdf8" transparent opacity={0.5} /></lineLoop>
      <lineLoop geometry={rings.great} rotation={[Math.PI / 2, 0, 0]}><lineBasicMaterial color="#38bdf8" transparent opacity={0.32} /></lineLoop>
      <lineLoop geometry={rings.great} rotation={[0, 0, Math.PI / 2]}><lineBasicMaterial color="#38bdf8" transparent opacity={0.32} /></lineLoop>
      {rings.latitudes.map((geometry, index) => (
        <lineLoop key={index} geometry={geometry}><lineBasicMaterial color="#38bdf8" transparent opacity={0.16} /></lineLoop>
      ))}
    </group>
  );
}

/** A glowing dot on the orientation sphere. */
function GlowDot({ position, color = CYAN, size = 0.035 }) {
  return (
    <group position={position}>
      <mesh><sphereGeometry args={[size, 14, 14]} /><meshBasicMaterial color={color} /></mesh>
      <mesh><sphereGeometry args={[size * 2.1, 14, 14]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} blending={THREE.AdditiveBlending} depthWrite={false} /></mesh>
      <mesh><sphereGeometry args={[size * 3.4, 14, 14]} />
        <meshBasicMaterial color={color} transparent opacity={0.1} blending={THREE.AdditiveBlending} depthWrite={false} /></mesh>
    </group>
  );
}

/**
 * Interactive FaceSet Orientation Sphere (heatmap of observed angles).
 * `observations` are the backend `observed_pose_points`: each glowing dot sits at that view's exact `position_xyz`.
 */
export function FaceSetOrientationSphere({ lattice, observations = [], showRings = true, radius = SPHERE_RADIUS, targetPose = null }) {
  return (
    <group>
      {showRings && <OrbitalRings radius={radius} />}

      {/* Exact 96-cell hemisphere lattice */}
      {lattice.map(node => {
        let color = '#334155'; // Slate: Unobserved blind spot
        let opacity = 0.35;
        let size = 0.025;

        if (node.sampleCount >= 3) {
          color = '#10b981'; // Emerald Green: Well-observed
          opacity = 0.95;
          size = 0.045;
        } else if (node.sampleCount >= 1) {
          color = '#f59e0b'; // Amber: Sparse angle
          opacity = 0.85;
          size = 0.035;
        }

        return (
          <mesh key={node.index} position={node.position}>
            <sphereGeometry args={[size, 10, 10]} />
            <meshBasicMaterial color={color} transparent opacity={opacity} />
          </mesh>
        );
      })}
      {/* Measured view directions, separate from their containing coverage cells. */}
      {observations.map((point, index) => (
        <GlowDot key={`observed-${point.view_index ?? index}`} position={point.position_xyz.map(value => value * radius)} />
      ))}
      {/* Where the target frame points the head. */}
      {targetPose && <GlowDot position={poseDirection(targetPose.yaw, targetPose.pitch).map(value => value * radius)} color="#f472b6" size={0.04} />}
    </group>
  );
}

/**
 * Reads the orbit camera: reports the head pose it currently shows (a trackball: dragging right turns the head right) and
 * moves a marker on the sphere to that pose. The marker is mutated in place, so orbiting never re-renders React.
 */
function ViewTracker({ onPose, radius, showMarker }) {
  const { camera } = useThree();
  const marker = useRef(null);
  const last = useRef('');
  useFrame(() => {
    const p = camera.position;
    const length = p.length() || 1;
    const yaw = (-Math.atan2(p.x, p.z) * 180) / Math.PI;
    const pitch = (-Math.asin(Math.max(-1, Math.min(1, p.y / length))) * 180) / Math.PI;
    if (marker.current) {
      marker.current.position.set(-p.x / length, -p.y / length, p.z / length).multiplyScalar(radius);
      marker.current.visible = showMarker;
    }
    const key = `${yaw.toFixed(1)}|${pitch.toFixed(1)}`;
    if (key !== last.current) { last.current = key; onPose({ yaw, pitch, roll: 0 }); }
  });
  return (
    <mesh ref={marker}>
      <ringGeometry args={[0.04, 0.055, 24]} />
      <meshBasicMaterial color="#ffffff" transparent opacity={0.75} side={THREE.DoubleSide} depthTest={false} />
    </mesh>
  );
}

/**
 * Physical-camera preview: sets the lens of the orbit camera. `focalMm` null leaves the viewer's own 42 degree camera alone.
 * Otherwise the vertical field of view is that of a 36 x 24 mm sensor behind a lens of `focalMm`, and the camera steps back or
 * forward (`factor`) so the head keeps its size in frame, which is what makes a long lens flatten the face and a wide one stretch it.
 */
const BASE_FOV = 42;
function PhysicalCameraRig({ focalMm, factor }) {
  const camera = useThree(state => state.camera);
  const invalidate = useThree(state => state.invalidate);
  const applied = useRef(1);
  useEffect(() => {
    const fov = focalMm == null ? BASE_FOV : verticalFovDeg(focalMm);
    if (camera.isPerspectiveCamera) {
      camera.fov = fov;
      camera.far = 50 * Math.max(1, factor);
      // Keep the viewing direction; only the distance changes with the lens.
      camera.position.multiplyScalar(factor / applied.current);
      applied.current = factor;
      camera.updateProjectionMatrix();
      invalidate();
    }
  }, [camera, focalMm, factor, invalidate]);
  return null;
}

/** Camera snaps to the frontal view (+0.0° Y · +0.0° P) whenever `signal` changes, and on the Reset Camera button. */
function CameraRig({ controlsRef, resetSignal, factor = 1 }) {
  const { camera } = useThree();
  useEffect(() => {
    camera.position.copy(FRONTAL_CAMERA);
    if (factor !== 1) camera.position.multiplyScalar(factor);
    camera.up.copy(WORLD_UP);
    camera.lookAt(0, 0, 0);
    const controls = controlsRef.current;
    if (controls) { controls.target.set(0, 0, 0); controls.update(); }
  }, [resetSignal, camera, controlsRef]);     // eslint-disable-line react-hooks/exhaustive-deps -- a lens change dollies by itself; only a reset snaps
  return null;
}

/** Imperative live-pose readout: orbiting updates this one node, not the whole viewer. */
function LivePoseReadout({ targetPose, registerView }) {
  const [view, setView] = useState({ yaw: 0, pitch: 0, roll: 0 });
  useEffect(() => { registerView(setView); return () => registerView(null); }, [registerView]);
  const shown = targetPose ?? view;
  return (
    <strong data-testid="live-pose" data-source={targetPose ? 'target' : 'view'}
      title={targetPose ? 'Pose of the selected target face' : 'Pose of the head as currently viewed (drag to rotate)'}>
      {formatPose(shown)}
    </strong>
  );
}

const IDLE_MESSAGE = 'Load a source face or .fsz faceset to preview 3D reconstruction';

function FaceCameraFocus({ resetCamera }) {
  const focus = useFaceSelectionStore(state => state.poseFocus);
  // The head already follows the selected face's pose, so syncing to it means looking at the head from the front.
  useEffect(() => { if (focus) resetCamera(); }, [focus, resetCamera]);
  return null;
}

/**
 * Main WebGL Pose & Geometry Viewer Component.
 *
 * `coverage` is the `POST /faceset/reconstruct-3d` payload (`mesh_2_5d`, `mesh_full_3d`, `observed_pose_points`, the
 * telemetry counts). `mesh` is an explicitly chosen reconstruction (a built identity asset, an imported artifact) and
 * overrides `coverage.mesh_full_3d`. `renderMode` is the workstation render mode: it selects the tab showing the
 * geometry that mode works from. `reconstruction` is the request state (idle | loading | ready | error).
 * @param {any} props
 */
export default function PoseGeometryViewer({
  coverage = null,
  track = null,
  mesh = null,
  renderMode,
  reconstruction = null,
  creator = null,
  lighting = null,
  morph = null,
  tools = null,
}) {
  const hostRef = useRef(null);
  // Buffer switcher + gizmos. `tools` (jaw / gaze / light callbacks bound to the real parameters) is supplied by the inspector.
  const [bufferView, setBufferView] = useState('lit');
  const [gizmos, setGizmos] = useState({ jaw: false, gaze: false, light: false });
  const toggleGizmo = key => setGizmos(value => ({ ...value, [key]: !value[key] }));
  const lightSource = tools?.light ?? lighting ?? null;
  const shLight = useMemo(
    () => (lightSource ? { yaw: lightSource.yaw, pitch: lightSource.pitch, strength: lightSource.strength, ambient: lightSource.ambient } : DEFAULT_SH_LIGHT),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the four scalars are the whole input
    [lightSource?.yaw, lightSource?.pitch, lightSource?.strength, lightSource?.ambient]);
  const controlsRef = useRef(null);
  const [viewMode, setViewMode] = useState('estimated_2_5d'); // 'estimated_2_5d' | 'reconstruction'
  const viewSetter = useRef(null);
  const registerView = useCallback(setter => { viewSetter.current = setter; }, []);
  const reportView = useCallback(pose => viewSetter.current?.(pose), []);

  const [resetSignal, setResetSignal] = useState(0);
  // A subtle topology overlay on the single-view estimate; off by default on the textured head so the albedo reads clearly.
  const [wireframeByTab, setWireframeByTab] = useState({ estimated_2_5d: true, reconstruction: false });
  const wireframe = wireframeByTab[viewMode];
  const toggleWireframe = () => setWireframeByTab(value => ({ ...value, [viewMode]: !value[viewMode] }));
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [showRings, setShowRings] = useState(true);

  // Studio lighting (five CC0 probes) and the physical-camera lens: both are previews in this viewport only.
  const [studioOpen, setStudioOpen] = useState(false);
  const studio = useStudioEnvironments(studioOpen);
  const [environmentId, setEnvironmentId] = useState('');                       // '' = the manual key light
  const environment = studio.environments.find(item => item.id === environmentId) ?? null;
  const environmentSH = useMemo(() => (environment ? exposeEnvironment(environmentToViewerSH(environment.sh)) : null), [environment]);
  const [focalMm, setFocalMm] = useState(null);                                 // null = the viewer's own camera
  const lensFactor = useMemo(() => (focalMm == null ? 1 : dollyFactor(BASE_FOV, verticalFovDeg(focalMm))), [focalMm]);
  const chooseEnvironment = id => { setEnvironmentId(id); if (id) setBufferView(view => (view === 'lit' ? 'sh' : view)); };

  // Target pose: the selected face, else the selected track, else the first target cluster. Standard frame.
  const activeSelection = useFaceSelectionStore(s => s.activeSelection);
  const detectedFaces = useFaceSelectionStore(s => s.detectedFaces);
  const targetClusters = useFaceSelectionStore(s => s.targetClusters);
  const targetPose = useMemo(() => {
    const face = activeSelection ? detectedFaces.find(f => f.id === activeSelection) : null;
    if (face?.pose) return legacyPoseToStandard(face.pose);
    const range = track?.pose_range;
    if (range) {
      const mid = pair => (pair ? (pair[0] + pair[1]) / 2 : 0);
      return legacyPoseToStandard({ yaw: mid(range.yaw_degrees), pitch: mid(range.pitch_degrees), roll: mid(range.roll_degrees) });
    }
    return targetClusters.length > 0 ? legacyPoseToStandard(targetClusters[0].avgPose) : null;
  }, [activeSelection, detectedFaces, track, targetClusters]);

  // The two reconstructions of the payload, in the backend mesh frame.
  const estimatedMesh = useMemo(() => payloadToReconstruction(coverage?.mesh_2_5d, 'tps_2_5d'), [coverage]);
  const payloadFullMesh = useMemo(() => payloadToReconstruction(coverage?.mesh_full_3d, 'identity_asset', coverage?.uv_texture_data_url), [coverage]);
  const fullMesh = mesh ?? payloadFullMesh;
  const denseFace = fullMesh?.provenance === 'three_tier_facial_surface';
  const displayedMesh = viewMode === 'estimated_2_5d' ? (estimatedMesh ?? fullMesh) : fullMesh;
  const volumetric = viewMode === 'reconstruction' && !!fullMesh;

  const assets = useMemo(() => buildMeshAssets(displayedMesh, { landmarkCount: viewMode === 'estimated_2_5d' && displayedMesh === estimatedMesh ? TEMPLATE_LANDMARK_COUNT : 0 }),
    [displayedMesh, viewMode, estimatedMesh]);
  useDisposable(assets?.geometry);
  // Live relighting preview: key light direction, ambient fill and specular sheen drive the viewer's lights and skin material.
  const lightRig = useMemo(() => (lighting ? lightingRig(lighting) : null), [lighting]);
  // The target-like (estimated 2.5D) proxy, shown as a ghost while the geometry blend is below 100% source mesh.
  const targetAssets = useMemo(() => (morph && estimatedMesh ? buildMeshAssets(estimatedMesh) : null), [morph, estimatedMesh]);
  useDisposable(targetAssets?.geometry);
  const texture = useDataTexture(assets && typeof displayedMesh?.texture === 'string' && assets.geometry.getAttribute('uv') ? displayedMesh.texture : null);

  // Render mode picks the tab (and so does a newly supplied explicit mesh, or a new source arriving).
  const modeInfo = useMemo(() => renderModeInfo(renderMode), [renderMode]);
  const intent = mesh || denseFace ? 'reconstruction' : renderMode === undefined ? 'estimated_2_5d' : modeInfo.preview;
  const hasFullMesh = !!fullMesh;
  const sourceKey = coverage?.faceset_id;
  useEffect(() => {
    setViewMode(intent === 'reconstruction' && hasFullMesh ? 'reconstruction' : 'estimated_2_5d');
  }, [intent, hasFullMesh, sourceKey, mesh]);

  // Observed pose points -> counts per cell of the exact backend 12x8 grid.
  const baseLattice = useMemo(() => generatePoseGrid(SPHERE_RADIUS), []);
  const observations = useMemo(() => [...(coverage?.observed_pose_points ?? []), ...(coverage?.generated_pose_points ?? [])].filter(point => Array.isArray(point?.position_xyz)), [coverage]);
  const lattice = useMemo(() => {
    const counts = new Array(96).fill(0);
    if (observations.length) {
      for (const point of observations) if (point.bin_index >= 0 && point.bin_index < 96) counts[point.bin_index] += 1;
    } else {
      for (const view of coverage?.views ?? []) {
        const yawBin = Math.min(11, Math.max(0, Math.floor((Math.min(90, Math.max(-90, view.yaw)) + 90) / 15)));
        const pitchBin = Math.min(7, Math.max(0, Math.floor((Math.min(60, Math.max(-60, view.pitch)) + 60) / 15)));
        counts[pitchBin * 12 + yawBin] += 1;
      }
    }
    return baseLattice.map(node => ({ ...node, sampleCount: counts[node.index] }));
  }, [baseLattice, observations, coverage]);

  // Telemetry comes straight from the payload; the source FaceSet is the only thing it describes.
  const stats = useMemo(() => {
    const totalNodes = coverage?.sphere_coverage_total ?? 96;
    const observedNodes = coverage?.sphere_coverage_count ?? lattice.filter(n => n.sampleCount > 0).length;
    const coveragePct = coverage?.sphere_coverage_pct ?? Math.round((observedNodes / totalNodes) * 100);
    const totalSamples = coverage?.observed_views ?? coverage?.views?.length ?? 0;
    return { totalNodes, observedNodes, coveragePct, totalSamples };
  }, [lattice, coverage]);

  const state = reconstruction?.state ?? (estimatedMesh || fullMesh ? 'ready' : 'idle');
  const hasGeometry = !!assets;
  // A source face / FaceSet is active (its reconstruction is loading, failed or done) or a mesh was supplied explicitly.
  const sourceLoaded = state !== 'idle' || !!coverage || !!mesh;
  const canRefresh = !!reconstruction?.refresh && !!reconstruction?.sourceId;
  const overlay = state === 'loading' ? { kind: 'loading', text: 'Reconstructing 3D face from the FaceSet…' }
    : state === 'error' ? { kind: 'error', text: reconstruction?.message || '3D reconstruction failed.' }
    : !hasGeometry ? { kind: 'idle', text: IDLE_MESSAGE } : null;

  const resetCamera = useCallback(() => setResetSignal(value => value + 1), []);
  const info = displayedMesh?.provenance ? `${volumetric ? (denseFace ? 'Facial surface' : 'Full 3D') : 'Estimated 2.5D'} · ${String(displayedMesh.provenance).replaceAll('_', ' ')}` : '';
  const limitation = (viewMode === 'reconstruction' ? coverage?.mesh_full_3d : coverage?.mesh_2_5d)?.limitations?.[0] ?? '';

  const tab = (id, label, disabled, title) => (
    <button
      type="button"
      className={`viewer-mode-btn ${viewMode === id ? 'active' : ''}`}
      onClick={() => setViewMode(id)}
      disabled={disabled}
      title={title}
      aria-pressed={viewMode === id}
      data-pipeline={renderMode !== undefined && modeInfo.preview === id ? 'true' : undefined}
    >
      {label}
      {renderMode !== undefined && modeInfo.preview === id && <span className="viewer-pipeline-tag" aria-label={`Render mode ${modeInfo.label}: ${modeInfo.previewNote}`}>{modeInfo.previewNote === 'preview only' ? 'preview' : 'render'}</span>}
    </button>
  );

  return (
    <div className="pose-geometry-container" ref={hostRef}>
      {/* Top Controls Overlay */}
      <div className="pose-viewer-toolbar">
        <div className="pose-mode-toggle">
          {tab('estimated_2_5d', 'Estimated 2.5D', false, 'Single-view estimate: the FaceSet\'s best frontal photo draped over a depth-prior face mesh')}
          {tab('reconstruction', denseFace ? 'Fitted facial surface' : 'Full 3D Reconstructed', !fullMesh,
            fullMesh ? (denseFace ? 'Anatomical face fitted from observed photographs' : `Legacy preview: ${fullMesh.provenance ?? fullMesh.method}`) : 'Load a FaceSet to reconstruct its facial surface')}
        </div>
      </div>
      <div className="pose-viewer-options" role="group" aria-label="3D viewer overlays">
        <button type="button" className="viewer-toggle" aria-pressed={wireframe} onClick={toggleWireframe} disabled={!hasGeometry}>Wireframe</button>
        <button type="button" className="viewer-toggle" aria-pressed={showLandmarks} onClick={() => setShowLandmarks(value => !value)} disabled={!hasGeometry || viewMode !== 'estimated_2_5d'}
          title="68 landmark points of the estimated face mesh">Landmarks</button>
        <button type="button" className="viewer-toggle" aria-pressed={showRings} onClick={() => setShowRings(value => !value)}
          title="Orbital rings and the pose coverage sphere">Coverage rings</button>
        <button
          type="button"
          className="camera-reset-btn"
          onClick={resetCamera}
          title="Reset 3D camera to the frontal view (+0.0° Y · +0.0° P)"
        >
          ⟲ Reset Camera
        </button>
      </div>

      <div className="viewport-buffers" role="radiogroup" aria-label="Viewport buffer" data-testid="viewport-buffers">
        {BUFFER_VIEWS.map(view => (
          <button key={view.id} type="button" role="radio" aria-checked={bufferView === view.id} className={`viewer-toggle ${bufferView === view.id ? 'active' : ''}`}
            title={view.hint} disabled={!hasGeometry} onClick={() => setBufferView(view.id)}>{view.label}</button>
        ))}
      </div>
      {bufferView !== 'lit' && <p className="viewport-buffer-note muted" data-testid="viewport-buffer-note">{BUFFER_VIEWS.find(view => view.id === bufferView)?.hint}
        {bufferView === 'depth' && <span className="depth-legend" aria-hidden="true"><i /> near → far</span>}</p>}
      <details className="viewport-studio" open={studioOpen} onToggle={event => setStudioOpen(event.currentTarget.open)} data-testid="viewport-studio">
        <summary>Studio lighting &amp; lens <small className="muted">preview in this viewport</small></summary>
        <div className="viewport-buffers" role="radiogroup" aria-label="Studio lighting environment" data-testid="studio-environments">
          <button type="button" role="radio" aria-checked={!environmentId} className={`viewer-toggle ${!environmentId ? 'active' : ''}`} disabled={!hasGeometry}
            title="The manual key light and ambient fill" onClick={() => chooseEnvironment('')}>Manual light</button>
          {studio.environments.map(item => (
            <button key={item.id} type="button" role="radio" aria-checked={environmentId === item.id} className={`viewer-toggle ${environmentId === item.id ? 'active' : ''}`} disabled={!hasGeometry}
              title={`${item.name} · ${Math.round(item.kelvin)} K · ${item.license} · reduced from the Poly Haven HDRI ${item.id}`} onClick={() => chooseEnvironment(item.id)}>{item.name}</button>
          ))}
        </div>
        {studio.state === 'loading' && <small className="muted" role="status">Loading the lighting environments…</small>}
        {studio.state === 'error' && <small className="identity-warn" role="alert">{studio.error} <button type="button" className="button small" onClick={studio.retry}>Retry</button></small>}
        {environment && <small className="muted" data-testid="studio-environment-note">{environment.name}: {Math.round(environment.kelvin)} K, {environment.license}. Shown as the SH Relit buffer, auto-exposed so the five can be compared.</small>}
        <label className="inspector-field gizmo-field">Focal length
          <output data-testid="focal-readout">{focalMm == null ? 'viewer default' : `${Math.round(focalMm)} mm · ${verticalFovDeg(focalMm).toFixed(1)}° V · ${horizontalFovDeg(focalMm).toFixed(1)}° H`}</output>
          <input type="range" aria-label="Focal length in millimetres" min={FOCAL_RANGE_MM[0]} max={FOCAL_RANGE_MM[1]} step={1} value={focalMm ?? DEFAULT_FOCAL_MM} disabled={!hasGeometry}
            aria-valuetext={focalMm == null ? 'viewer default camera' : `${Math.round(focalMm)} millimetres`} onChange={event => setFocalMm(clampFocal(Number(event.target.value)))} />
          <span className="protection-scale" aria-hidden="true"><small>{FOCAL_RANGE_MM[0]} mm wide</small><small>{FOCAL_RANGE_MM[1]} mm tele</small></span>
        </label>
        <div className="morph-actions"><button type="button" className="button small" disabled={focalMm == null} onClick={() => setFocalMm(null)}>Reset lens</button></div>
        <small className="muted">Previews the look of the lens (36 × 24 mm sensor, head kept the same size) and of the lighting. The physical-camera matcher and the probes are not part of the render pipeline yet, so neither changes an output.</small>
      </details>
      {tools && (
        <div className="viewport-gizmo-dock" role="group" aria-label="3D gizmos" data-testid="viewport-gizmos">
          <div className="gizmo-toggles">
            <button type="button" className="viewer-toggle" aria-pressed={gizmos.jaw} disabled={!hasGeometry} onClick={() => toggleGizmo('jaw')}
              title="Drag the chin handle to open the jaw (θ jaw). Edits the selected face's mouth-open value.">Jaw opening</button>
            <button type="button" className="viewer-toggle" aria-pressed={gizmos.gaze} disabled={!hasGeometry} onClick={() => toggleGizmo('gaze')}
              title="Drag the gaze target in front of the eyes (θ eyes). Edits the selected face's eyeball gaze.">Eye gaze target</button>
            <button type="button" className="viewer-toggle" aria-pressed={gizmos.light} disabled={!hasGeometry} onClick={() => toggleGizmo('light')}
              title="Drag the light handle around the head. Sets the manual key light direction.">Light direction</button>
          </div>
          {(gizmos.jaw || gizmos.gaze) && !tools.trackId && <small className="muted" role="status">Select a target face to edit jaw opening and eye gaze.</small>}
          {gizmos.jaw && (
            <PrecisionRangeSlider className="inspector-field gizmo-field" label="Jaw opening θ" ariaLabel="Jaw opening angle" unit="°" min={0} max={JAW_MAX_DEG} step={1}
              value={Math.round(tools.jawDeg)} defaultValue={0} disabled={!tools.trackId} onChange={tools.onJaw} />)}
          {gizmos.gaze && (<>
            <PrecisionRangeSlider className="inspector-field gizmo-field" label="Gaze horizontal" min={-1} max={1} step={0.01}
              value={tools.gaze.x} defaultValue={0} disabled={!tools.trackId} onChange={value => tools.onGaze(value, tools.gaze.y)} />
            <PrecisionRangeSlider className="inspector-field gizmo-field" label="Gaze vertical" min={-1} max={1} step={0.01}
              value={tools.gaze.y} defaultValue={0} disabled={!tools.trackId} onChange={value => tools.onGaze(tools.gaze.x, value)} />
          </>)}
          {gizmos.light && (<>
            <PrecisionRangeSlider className="inspector-field gizmo-field" label="Light yaw" unit="°" min={-180} max={180} step={1}
              value={Math.round(tools.light.yaw)} defaultValue={0} onChange={value => tools.onLight(value, tools.light.pitch)} />
            <PrecisionRangeSlider className="inspector-field gizmo-field" label="Light pitch" unit="°" min={-90} max={90} step={1}
              value={Math.round(tools.light.pitch)} defaultValue={0} onChange={value => tools.onLight(tools.light.yaw, value)} />
            {!tools.relightOn && <small className="muted" role="status">3D relighting is off: this sets the manual light only. {tools.onRelight && <button type="button" className="button small" onClick={() => tools.onRelight(true)}>Turn on 3D relighting</button>}</small>}
          </>)}
        </div>
      )}

      {(reconstruction?.refresh || creator) && (
        <div className="pose-viewer-actions" role="group" aria-label="3D FaceSet actions">
          {reconstruction?.refresh && (
            <button type="button" className="viewer-action-btn primary" onClick={reconstruction.refresh}
              disabled={!canRefresh || state === 'loading'} data-testid="generate-3d-faceset"
              title={canRefresh ? 'Rebuild the textured 3D head from the active source / .fsz (cached until the archive changes)' : 'Load a source face or .fsz faceset first'}>
              ⚡ {state === 'loading' ? 'Generating…' : 'Generate / Refresh 3D Faceset'}
            </button>
          )}
          {creator}
        </div>
      )}

      {/* R3F Canvas */}
      <div className="pose-canvas-wrapper">
        <GuardedCanvas
          label="3D viewport"
          resetKey={coverage?.faceset_id ?? ''}
          fallback={({ reason, retry, use3D }) => <WebGLFallbackInspector mesh={displayedMesh} reason={reason} onRetry={retry} onUse3D={use3D} />}
          camera={{ position: [0, 0, CAMERA_DISTANCE], fov: 42, near: 0.05, far: 50 }}
          dpr={[1, 2]}
          gl={{ antialias: true, alpha: true }}
          onCreated={({ gl }) => {
            gl.setClearColor(0x000000, 0);
          }}
        >
          {/* Lighting */}
          {lightRig ? (
            <>
              <ambientLight intensity={lightRig.ambientIntensity} name="ambient-fill" />
              <directionalLight position={lightRig.keyPosition} intensity={lightRig.keyIntensity} name="key-light" />
              <LightArrow position={lightRig.keyPosition} manual={lighting.manual} />
            </>
          ) : (
            <>
              <ambientLight intensity={0.7} />
              <directionalLight position={[4, 5, 5]} intensity={1.8} />
              <directionalLight position={[-4, -2, -3]} intensity={0.5} color="#38bdf8" />
              <pointLight position={[0, 3, 2]} intensity={0.8} />
            </>
          )}

          {tools && gizmos.light && <LightGizmo light={tools.light} onChange={tools.onLight} enabled />}
          {assets ? (
            <ReconstructedHead pose={targetPose} assets={assets} texture={texture} volumetric={volumetric}
              bufferView={bufferView} shLight={shLight} shCoefficients={environmentSH} tools={tools} gizmos={gizmos}
              wireframe={wireframe} landmarks={showLandmarks && viewMode === 'estimated_2_5d'}
              lighting={lighting} morph={morph} targetAssets={morph && displayedMesh !== estimatedMesh ? targetAssets : null} />
          ) : !sourceLoaded ? (
            // Empty-state hint only: once a source exists (loading, error or ready) the crystal is never drawn.
            <ReferenceHead />
          ) : null}

          {/* FaceSet orientation heatmap, orbital rings and the glowing observed-pose dots */}
          {(observations.length > 0 || coverage) && (
            <FaceSetOrientationSphere lattice={lattice} observations={observations} showRings={showRings} targetPose={targetPose} />
          )}
          {!coverage && showRings && <OrbitalRings radius={SPHERE_RADIUS} />}

          {/* OrbitControls: drag to rotate, scroll to zoom */}
          <OrbitControls
            ref={controlsRef}
            makeDefault
            enablePan={false}
            rotateSpeed={0.6}
            enableDamping
            dampingFactor={0.08}
            minDistance={1.8 * lensFactor}
            maxDistance={8 * lensFactor}
          />
          <PhysicalCameraRig focalMm={focalMm} factor={lensFactor} />
          <CameraRig controlsRef={controlsRef} resetSignal={resetSignal} factor={lensFactor} />
          <ViewTracker onPose={reportView} radius={SPHERE_RADIUS} showMarker={showRings && !targetPose && (observations.length > 0 || !!coverage)} />

          {/* Camera Orientation Gizmo: follows the camera */}
          <GizmoHelper alignment="bottom-right" margin={[40, 40]}>
            <GizmoViewport
              axisColors={['#f43f5e', '#10b981', '#3b82f6']}
              labelColor="#ffffff"
              hideNegativeAxes
            />
          </GizmoHelper>
        </GuardedCanvas>
        {overlay && (
          <div className={`pose-viewer-overlay ${overlay.kind}`} role={overlay.kind === 'error' ? 'alert' : 'status'}>
            <span>{overlay.text}</span>
            {overlay.kind === 'error' && reconstruction?.retry && <button type="button" className="camera-reset-btn" onClick={reconstruction.retry}>Retry</button>}
          </div>
        )}
        <FaceCameraFocus resetCamera={resetCamera} />
      </div>
      {info && <div className="pose-viewer-caption" title={limitation}>{info}{limitation ? ' · template prior beyond the observed landmarks' : ''}</div>}

      {/* Status Readout Bar */}
      <div className="pose-status-readout">
        <div className="status-metric">
          <span className="status-label">Observed Views</span>
          <strong>{stats.totalSamples}</strong>
          {!!coverage?.generated_views && <small>{coverage.generated_views} generated references</small>}
        </div>
        <div className="status-divider" />
        <div className="status-metric">
          <span className="status-label">Sphere Coverage</span>
          <strong>{stats.observedNodes} / {stats.totalNodes} ({stats.coveragePct}%)</strong>
        </div>
        <div className="status-divider" />
        <div className="status-metric">
          <span className="status-label">Live Pose</span>
          <LivePoseReadout targetPose={targetPose} registerView={registerView} />
        </div>
      </div>
    </div>
  );
}
