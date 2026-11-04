import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import { useFaceSelectionStore } from '../../useFaceSelectionStore';
import { reducedMotion } from '../../utils/motionPrefs';

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

/**
 * Canonical 3D Head Mesh Component with Slerp Interpolation and Landmark Dots.
 */
function CanonicalHeadMesh({ pose, mode, mesh }) {
  const groupRef = useRef(null);
  const currentQuat = useRef(new THREE.Quaternion());
  const targetQuat = useMemo(() => new THREE.Quaternion(), []);

  const headGeom = useMemo(() => createCanonicalHeadGeometry(), []);

  // Update target quaternion whenever pose angles change
  useEffect(() => {
    const yaw = pose?.yaw ?? 0;
    const pitch = pose?.pitch ?? 0;
    const roll = pose?.roll ?? 0;

    const yawRad = (yaw * Math.PI) / 180;
    const pitchRad = (pitch * Math.PI) / 180;
    const rollRad = (roll * Math.PI) / 180;

    const euler = new THREE.Euler(pitchRad, yawRad, -rollRad, 'YXZ');
    targetQuat.setFromEuler(euler);
  }, [pose, targetQuat]);

  // Smooth slerp interpolation on every render frame
  useFrame((_, delta) => {
    if (groupRef.current) {
      // Reduce-motion: jump to the pose instead of easing toward it.
      currentQuat.current.slerp(targetQuat, reducedMotion() ? 1 : Math.min(1, delta * 12));
      groupRef.current.quaternion.copy(currentQuat.current);
    }
  });

  // If a reconstructed mesh exists and mode is 'reconstruction'
  const customMeshGeom = useMemo(() => {
    if (!mesh || mode !== 'reconstruction') return null;
    const positions = [];
    const uvs = [];
    for (let faceIndex = 0; faceIndex < mesh.faces.length; faceIndex++) {
      const face = mesh.faces[faceIndex];
      face.forEach((vertexIndex, index) => {
        const vertex = mesh.vertices[vertexIndex];
        positions.push(
          vertex[0],
          mesh.coordinate_system === 'opencv' ? -vertex[1] : vertex[1],
          mesh.coordinate_system === 'opencv' ? -vertex[2] : vertex[2]
        );
        const uv = mesh.uvs?.[mesh.face_uvs?.[faceIndex]?.[index] ?? vertexIndex];
        if (uv) uvs.push(uv[0], uv[1]);
      });
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    if (uvs.length === (positions.length / 3) * 2) {
      g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    }
    g.computeVertexNormals();
    g.center();
    // Receipts are in metres (identity assets) or arbitrary units (imported meshes): fit every mesh to the viewer.
    g.computeBoundingSphere();
    const radius = g.boundingSphere?.radius ?? 0;
    if (radius > 0) g.scale(0.95 / radius, 0.95 / radius, 0.95 / radius);
    return g;
  }, [mesh, mode]);
  // The observed atlas (identity assets: unseen texels are grey). Only data: URLs from the local API are loaded.
  const customTexture = useMemo(() => {
    if (!customMeshGeom || typeof mesh?.texture !== 'string' || !mesh.texture.startsWith('data:image/') || !customMeshGeom.getAttribute('uv')) return null;
    const texture = new THREE.TextureLoader().load(mesh.texture);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  }, [customMeshGeom, mesh]);
  // GPU resources created here are not owned by R3F: release them when replaced or unmounted.
  useEffect(() => () => customMeshGeom?.dispose(), [customMeshGeom]);
  useEffect(() => () => customTexture?.dispose(), [customTexture]);
  useEffect(() => () => headGeom.dispose(), [headGeom]);

  return (
    <group ref={groupRef} position={[0, 0, 0]}>
      {customMeshGeom ? (
        <mesh geometry={customMeshGeom}>
          <meshStandardMaterial
            key={customTexture ? 'textured' : 'flat'}
            map={customTexture}
            color={customTexture ? '#ffffff' : '#91e6c9'}
            roughness={0.65}
            metalness={0.2}
            side={THREE.DoubleSide}
          />
        </mesh>
      ) : (
        <>
          {/* Solid canonical low-poly head */}
          <mesh geometry={headGeom}>
            <meshStandardMaterial
              color="#1e293b"
              roughness={0.5}
              metalness={0.3}
              flatShading={true}
            />
          </mesh>

          {/* Elegant contour wireframe */}
          <mesh geometry={headGeom}>
            <meshBasicMaterial
              color="#38bdf8"
              wireframe={true}
              transparent={true}
              opacity={0.35}
            />
          </mesh>

          {/* 3D ArcFace landmark points projecting from mesh surface */}
          {CANONICAL_LANDMARKS.map((lm, idx) => (
            <group key={idx} position={lm.pos}>
              {/* Glowing Landmark Dot */}
              <mesh>
                <sphereGeometry args={[0.028, 12, 12]} />
                <meshBasicMaterial color="#00f5d4" />
              </mesh>
              {/* Outer halo */}
              <mesh>
                <sphereGeometry args={[0.048, 12, 12]} />
                <meshBasicMaterial color="#00f5d4" transparent opacity={0.3} />
              </mesh>
            </group>
          ))}
        </>
      )}

      {/* Origin axes indicator */}
      <axesHelper args={[0.6]} />
    </group>
  );
}

/**
 * Interactive FaceSet Orientation Sphere (Heatmap of observed angles).
 */
export function FaceSetOrientationSphere({ lattice }) {
  // Wireframe guide circles
  const circleGeom = useMemo(() => {
    const pts = [];
    const segs = 64;
    for (let i = 0; i <= segs; i++) {
      const a = (i / segs) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(a) * 1.35, 0, Math.sin(a) * 1.35));
    }
    const g = new THREE.BufferGeometry();
    g.setFromPoints(pts);
    return g;
  }, []);

  return (
    <group>
      {/* Great circle equatorial and meridian guides */}
      <line geometry={circleGeom}>
        <lineBasicMaterial color="#334155" transparent opacity={0.4} />
      </line>
      <line geometry={circleGeom} rotation={[Math.PI / 2, 0, 0]}>
        <lineBasicMaterial color="#334155" transparent opacity={0.4} />
      </line>
      <line geometry={circleGeom} rotation={[0, 0, Math.PI / 2]}>
        <lineBasicMaterial color="#334155" transparent opacity={0.4} />
      </line>

      {/* Discrete sampling nodes from Fibonacci lattice */}
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
    </group>
  );
}

/**
 * Main WebGL Pose & Geometry Viewer Component.
 * @param {{
 *   coverage?: import('../../contracts').Coverage | null;
 *   track?: import('../../contracts').Track | null;
 *   mesh?: import('../../contracts').Reconstruction | null;
 * }} props
 */
function FaceCameraFocus({ controlsRef }) {
  const { camera } = useThree();
  const focus = useFaceSelectionStore(state => state.poseFocus);
  useEffect(() => {
    if (!focus) return;
    const face = useFaceSelectionStore.getState().detectedFaces.find(face => face.id === focus.faceId);
    if (!face) return;
    const pose = face.pose, quaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(pose.pitch * Math.PI / 180, pose.yaw * Math.PI / 180, -pose.roll * Math.PI / 180, 'YXZ'));
    camera.position.copy(new THREE.Vector3(0, 0, 3).applyQuaternion(quaternion));
    camera.up.copy(new THREE.Vector3(0, 1, 0).applyQuaternion(quaternion)); camera.lookAt(0, 0, 0);
    controlsRef.current?.target.set(0, 0, 0); controlsRef.current?.update();
  }, [focus, camera, controlsRef]);
  return null;
}

/** @param {any} props */
export default function PoseGeometryViewer({
  coverage = null,
  track = null,
  mesh = null,
}) {
  const hostRef = useRef(null);
  const controlsRef = useRef(null);

  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [viewMode, setViewMode] = useState(mesh ? 'reconstruction' : 'estimated_2_5d'); // 'estimated_2_5d' | 'reconstruction'
  // A newly supplied mesh is shown at once; removing it returns to the estimate (there is nothing else to show).
  useEffect(() => { setViewMode(mesh ? 'reconstruction' : 'estimated_2_5d'); }, [mesh]);
  const [webglError, setWebglError] = useState('');

  // Active pose from useFaceSelectionStore
  const activeSelection = useFaceSelectionStore(s => s.activeSelection);
  const detectedFaces = useFaceSelectionStore(s => s.detectedFaces);
  const targetClusters = useFaceSelectionStore(s => s.targetClusters);

  // Derive pose: selected face -> track pose range -> cluster avgPose -> default [0, 0, 0]
  const currentPose = useMemo(() => {
    if (activeSelection) {
      const face = detectedFaces.find(f => f.id === activeSelection);
      if (face?.pose) return face.pose;
    }
    if (track?.pose_range) {
      const yawAvg = track.pose_range.yaw_degrees ? (track.pose_range.yaw_degrees[0] + track.pose_range.yaw_degrees[1]) / 2 : 0;
      const pitchAvg = track.pose_range.pitch_degrees ? (track.pose_range.pitch_degrees[0] + track.pose_range.pitch_degrees[1]) / 2 : 0;
      const rollAvg = track.pose_range.roll_degrees ? (track.pose_range.roll_degrees[0] + track.pose_range.roll_degrees[1]) / 2 : 0;
      return { yaw: yawAvg, pitch: pitchAvg, roll: rollAvg };
    }
    if (targetClusters.length > 0) {
      return targetClusters[0].avgPose;
    }
    return { yaw: 0, pitch: 0, roll: 0 };
  }, [activeSelection, detectedFaces, track, targetClusters]);

  // Aggregate observed sample views: coverage views OR video track faces
  const rawViews = useMemo(() => {
    if (coverage?.views && coverage.views.length > 0) {
      return coverage.views;
    }
    // Fallback: extract from video track detections
    if (detectedFaces && detectedFaces.length > 0) {
      return detectedFaces.map(f => ({
        yaw: f.pose?.yaw ?? 0,
        pitch: f.pose?.pitch ?? 0,
        roll: f.pose?.roll ?? 0,
        pose_bin: 'track_view',
      }));
    }
    return [];
  }, [coverage, detectedFaces]);

  // Fibonacci lattice nodes populated with view coverage
  const baseLattice = useMemo(() => generateFibonacciSphere(96, 1.35), []);
  const lattice = useMemo(() => projectViewsToLattice(baseLattice, rawViews), [baseLattice, rawViews]);

  // Coverage statistics
  const stats = useMemo(() => {
    const totalNodes = lattice.length;
    const observedNodes = lattice.filter(n => n.sampleCount > 0).length;
    const coveragePct = Math.round((observedNodes / totalNodes) * 100);
    const totalSamples = rawViews.length;
    return { totalNodes, observedNodes, coveragePct, totalSamples };
  }, [lattice, rawViews]);

  // ResizeObserver for robust layout adaptation
  const updateSize = useCallback(() => {
    if (hostRef.current) {
      const { clientWidth, clientHeight } = hostRef.current;
      setDimensions({ width: clientWidth, height: clientHeight });
    }
  }, []);

  useEffect(() => {
    updateSize();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(updateSize);
    if (hostRef.current) ro.observe(hostRef.current);
    return () => ro.disconnect();
  }, [updateSize]);

  // Auto-reset camera view
  const handleResetCamera = () => {
    if (controlsRef.current) {
      controlsRef.current.reset();
      controlsRef.current.target.set(0, 0, 0);
    }
  };

  return (
    <div className="pose-geometry-container" ref={hostRef}>
      {/* Top Controls Overlay */}
      <div className="pose-viewer-toolbar">
        <div className="pose-mode-toggle">
          <button
            type="button"
            className={`viewer-mode-btn ${viewMode === 'estimated_2_5d' ? 'active' : ''}`}
            onClick={() => setViewMode('estimated_2_5d')}
            title="Estimated 2.5D Canonical Head & Orientation Sphere"
          >
            Estimated 2.5D
          </button>
          <button
            type="button"
            className={`viewer-mode-btn ${viewMode === 'reconstruction' ? 'active' : ''}`}
            onClick={() => setViewMode('reconstruction')}
            disabled={!mesh}
            title={mesh ? `Reconstructed mesh: ${mesh.provenance ?? mesh.method}` : 'No reconstruction loaded: build a 3D identity asset or choose a reconstruction artifact'}
          >
            Full 3D Reconstructed
          </button>
        </div>

        <button
          type="button"
          className="camera-reset-btn"
          onClick={handleResetCamera}
          title="Reset 3D camera to default viewpoint"
        >
          ⟲ Reset Camera
        </button>
      </div>

      {/* R3F Canvas */}
      <div className="pose-canvas-wrapper">
        {webglError ? (
          <div className="empty-caption">{webglError}</div>
        ) : (
          <Canvas
            camera={{ position: [2.0, 1.2, 2.6], fov: 42, near: 0.05, far: 50 }}
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

            {/* Canonical Head & Landmarks */}
            <CanonicalHeadMesh
              pose={currentPose}
              mode={viewMode}
              mesh={mesh}
            />

            {/* FaceSet Orientation Heatmap Sphere */}
            {viewMode === 'estimated_2_5d' && (
              <FaceSetOrientationSphere lattice={lattice} />
            )}

            {/* OrbitControls */}
            <OrbitControls
              ref={controlsRef}
              makeDefault
              enableDamping
              dampingFactor={0.08}
              minDistance={1.2}
              maxDistance={7}
            />
            <FaceCameraFocus controlsRef={controlsRef}/>

            {/* Camera Orientation Gizmo */}
            <GizmoHelper alignment="bottom-right" margin={[40, 40]}>
              <GizmoViewport
                axisColors={['#f43f5e', '#10b981', '#3b82f6']}
                labelColor="#ffffff"
              />
            </GizmoHelper>
          </Canvas>
        )}
      </div>

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
          <strong>
            {currentPose.yaw >= 0 ? '+' : ''}{currentPose.yaw.toFixed(1)}° Y ·{' '}
            {currentPose.pitch >= 0 ? '+' : ''}{currentPose.pitch.toFixed(1)}° P
          </strong>
        </div>
      </div>
    </div>
  );
}
