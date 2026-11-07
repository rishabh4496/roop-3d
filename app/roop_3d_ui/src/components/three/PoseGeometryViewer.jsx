import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import { useFaceSelectionStore } from '../../useFaceSelectionStore';
import { reducedMotion } from '../../utils/motionPrefs';
import { renderModeInfo } from '../workstation/renderModes';

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
 * The reconstructed head. It follows the target's pose (slerped) so the swap geometry is shown exactly as the pipeline
 * will pose it; the orbit camera moves independently around it.
 */
function ReconstructedHead({ pose, assets, texture, volumetric, wireframe, landmarks }) {
  const group = useRef(null);
  const current = useRef(new THREE.Quaternion());
  const target = useMemo(() => new THREE.Quaternion(), []);

  useEffect(() => { poseQuaternion(pose, target); }, [pose, target]);

  useFrame((_, delta) => {
    if (!group.current) return;
    // Reduce-motion: jump to the pose instead of easing toward it.
    current.current.slerp(target, reducedMotion() ? 1 : Math.min(1, delta * 12));
    group.current.quaternion.copy(current.current);
  });

  const landmarkPoints = useMemo(() => {
    if (!assets?.landmarks) return [];
    const points = [];
    for (let i = 0; i < assets.landmarks.length; i += 3) points.push([assets.landmarks[i], assets.landmarks[i + 1], assets.landmarks[i + 2]]);
    return points;
  }, [assets]);

  return (
    <group ref={group}>
      <mesh geometry={assets.geometry}>
        <meshStandardMaterial
          key={texture ? 'textured' : 'flat'}
          map={texture}
          color={texture ? '#ffffff' : '#9fd8c8'}
          roughness={volumetric ? 0.62 : 0.7}
          metalness={0}
          side={THREE.DoubleSide}
        />
      </mesh>
      {wireframe && (
        <mesh geometry={assets.geometry}>
          <meshBasicMaterial color={CYAN} wireframe transparent opacity={0.22} depthWrite={false}
            polygonOffset polygonOffsetFactor={-1} polygonOffsetUnits={-1} />
        </mesh>
      )}
      {landmarks && landmarkPoints.map((point, index) => (
        <mesh key={index} position={point} geometry={LANDMARK_GEOMETRY}>
          <meshBasicMaterial color={CYAN} />
        </mesh>
      ))}
    </group>
  );
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

/** Camera snaps to the frontal view (+0.0° Y · +0.0° P) whenever `signal` changes, and on the Reset Camera button. */
function CameraRig({ controlsRef, resetSignal }) {
  const { camera } = useThree();
  useEffect(() => {
    camera.position.copy(FRONTAL_CAMERA);
    camera.up.copy(WORLD_UP);
    camera.lookAt(0, 0, 0);
    const controls = controlsRef.current;
    if (controls) { controls.target.set(0, 0, 0); controls.update(); }
  }, [resetSignal, camera, controlsRef]);
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
}) {
  const hostRef = useRef(null);
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
  const displayedMesh = viewMode === 'estimated_2_5d' ? (estimatedMesh ?? fullMesh) : fullMesh;
  const volumetric = viewMode === 'reconstruction' && !!fullMesh;

  const assets = useMemo(() => buildMeshAssets(displayedMesh, { landmarkCount: viewMode === 'estimated_2_5d' && displayedMesh === estimatedMesh ? TEMPLATE_LANDMARK_COUNT : 0 }),
    [displayedMesh, viewMode, estimatedMesh]);
  useDisposable(assets?.geometry);
  const texture = useDataTexture(assets && typeof displayedMesh?.texture === 'string' && assets.geometry.getAttribute('uv') ? displayedMesh.texture : null);

  // Render mode picks the tab (and so does a newly supplied explicit mesh, or a new source arriving).
  const modeInfo = useMemo(() => renderModeInfo(renderMode), [renderMode]);
  const intent = renderMode === undefined ? (mesh ? 'reconstruction' : 'estimated_2_5d') : (mesh ? 'reconstruction' : modeInfo.preview);
  const hasFullMesh = !!fullMesh;
  const sourceKey = coverage?.faceset_id;
  useEffect(() => {
    setViewMode(intent === 'reconstruction' && hasFullMesh ? 'reconstruction' : 'estimated_2_5d');
  }, [intent, hasFullMesh, sourceKey, mesh]);

  // Observed pose points -> counts per cell of the exact backend 12x8 grid.
  const baseLattice = useMemo(() => generatePoseGrid(SPHERE_RADIUS), []);
  const observations = useMemo(() => (Array.isArray(coverage?.observed_pose_points) ? coverage.observed_pose_points.filter(point => Array.isArray(point?.position_xyz)) : []), [coverage]);
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
  const info = displayedMesh?.provenance ? `${volumetric ? 'Full 3D' : 'Estimated 2.5D'} · ${String(displayedMesh.provenance).replaceAll('_', ' ')}` : '';
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
          {tab('reconstruction', 'Full 3D Reconstructed', !fullMesh,
            fullMesh ? `Stitched multi-view head: ${fullMesh.provenance ?? fullMesh.method}` : 'No full 3D head yet: load a FaceSet or build a 3D identity asset')}
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
        <Canvas
          camera={{ position: [0, 0, CAMERA_DISTANCE], fov: 42, near: 0.05, far: 50 }}
          dpr={[1, 2]}
          gl={{ antialias: true, alpha: true }}
          onCreated={({ gl }) => {
            gl.setClearColor(0x000000, 0);
          }}
        >
          {/* Lighting */}
          <ambientLight intensity={0.7} />
          <directionalLight position={[4, 5, 5]} intensity={1.8} />
          <directionalLight position={[-4, -2, -3]} intensity={0.5} color="#38bdf8" />
          <pointLight position={[0, 3, 2]} intensity={0.8} />

          {assets ? (
            <ReconstructedHead pose={targetPose} assets={assets} texture={texture} volumetric={volumetric}
              wireframe={wireframe} landmarks={showLandmarks && viewMode === 'estimated_2_5d'} />
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
            minDistance={1.8}
            maxDistance={8}
          />
          <CameraRig controlsRef={controlsRef} resetSignal={resetSignal} />
          <ViewTracker onPose={reportView} radius={SPHERE_RADIUS} showMarker={showRings && !targetPose && (observations.length > 0 || !!coverage)} />

          {/* Camera Orientation Gizmo: follows the camera */}
          <GizmoHelper alignment="bottom-right" margin={[40, 40]}>
            <GizmoViewport
              axisColors={['#f43f5e', '#10b981', '#3b82f6']}
              labelColor="#ffffff"
              hideNegativeAxes
            />
          </GizmoHelper>
        </Canvas>
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
