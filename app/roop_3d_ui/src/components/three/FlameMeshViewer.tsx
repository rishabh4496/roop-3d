import React, { useState, useMemo, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import GuardedCanvas from './GuardedCanvas';
import WebGLFallbackInspector from './WebGLFallbackInspector';
import type { Reconstruction } from '../../contracts';
import { reducedMotion } from '../../utils/motionPrefs';

export interface FlameMeshViewerProps {
  mesh: Reconstruction | null;
  poseStrength?: number;
  initialWireframe?: boolean;
}

function FlameScene({
  mesh,
  poseStrength = 1.0,
  wireframe = false,
  solid = true,
  showTexture = true,
}: {
  mesh: Reconstruction | null;
  poseStrength: number;
  wireframe: boolean;
  solid: boolean;
  showTexture: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  const { geometry, texture } = useMemo(() => {
    if (!mesh || !mesh.vertices || !mesh.faces) {
      // Default canonical low-poly placeholder if no external mesh is loaded
      const placeholder = new THREE.SphereGeometry(1, 24, 24);
      return { geometry: placeholder, texture: null };
    }

    const positions: number[] = [];
    const uvs: number[] = [];
    const isOpencv = mesh.coordinate_system === 'opencv';

    for (let faceIdx = 0; faceIdx < mesh.faces.length; faceIdx++) {
      const face = mesh.faces[faceIdx];
      face.forEach((vertexIdx, pointIdx) => {
        const v = mesh.vertices[vertexIdx];
        if (!v) return;

        // Dual-endpoint blend: flat camera projection (z=0) vs full 3D depth
        const x = v[0];
        const y = isOpencv ? -v[1] : v[1];
        const z3d = isOpencv ? -v[2] : v[2];
        const z = z3d * poseStrength; // 0.0 = flat plane projection, 1.0 = full 3D

        positions.push(x, y, z);

        const uv = mesh.uvs?.[mesh.face_uvs?.[faceIdx]?.[pointIdx] ?? vertexIdx];
        if (uv) {
          uvs.push(uv[0], uv[1]);
        }
      });
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    if (uvs.length === (positions.length / 3) * 2) {
      g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    }
    g.computeVertexNormals();
    g.center();
    g.computeBoundingSphere();
    const radius = g.boundingSphere?.radius ?? 0;
    if (radius > 0) {
      g.scale(1.2 / radius, 1.2 / radius, 1.2 / radius);
    }

    let tex: THREE.Texture | null = null;
    if (mesh.texture && typeof mesh.texture === 'string' && mesh.texture.startsWith('data:image/')) {
      tex = new THREE.TextureLoader().load(mesh.texture);
      tex.colorSpace = THREE.SRGBColorSpace;
    }

    return { geometry: g, texture: tex };
  }, [mesh, poseStrength]);

  useEffect(() => {
    return () => {
      geometry?.dispose();
      texture?.dispose();
    };
  }, [geometry, texture]);

  // Subtle idle turntable rotation (skipped under the reduce-motion preference; the camera stays user-driven)
  useFrame((_, delta) => {
    if (meshRef.current && !reducedMotion()) {
      meshRef.current.rotation.y += delta * 0.15;
    }
  });

  return (
    <group>
      <ambientLight intensity={0.6} />
      <directionalLight position={[2, 4, 3]} intensity={1.2} />
      <directionalLight position={[-2, -2, -2]} intensity={0.4} color="#38bdf8" />

      {solid && (
        <mesh ref={meshRef} geometry={geometry}>
          <meshStandardMaterial
            map={showTexture && texture ? texture : undefined}
            color={showTexture && texture ? '#ffffff' : '#1e293b'}
            roughness={0.4}
            metalness={0.25}
            side={THREE.DoubleSide}
          />
        </mesh>
      )}

      {wireframe && (
        <mesh geometry={geometry}>
          <meshBasicMaterial
            wireframe
            color="#10b981"
            transparent
            opacity={0.7}
          />
        </mesh>
      )}
    </group>
  );
}

export default function FlameMeshViewer({
  mesh,
  poseStrength = 1.0,
  initialWireframe = false,
}: FlameMeshViewerProps) {
  const [wireframe, setWireframe] = useState(initialWireframe);
  const [solid, setSolid] = useState(true);
  const [showTexture, setShowTexture] = useState(true);
  const [blend, setBlend] = useState(poseStrength);

  const vertexCount = mesh?.vertices?.length ?? 0;
  const faceCount = mesh?.faces?.length ?? 0;
  const isFlame = mesh?.method === 'flame';

  return (
    <div className="flame-mesh-viewer" role="region" aria-label="FLAME 3D Mesh Inspector">
      <div className="flame-viewer-toolbar">
        <div className="flame-status-badges">
          <span className={`mesh-type-badge ${isFlame ? 'flame' : 'standard'}`}>
            {isFlame ? 'FLAME 3D Topology' : mesh ? '2.5D Surface Mesh' : 'Canonical Sphere Proxy'}
          </span>
          <span className="mesh-stat-pill">
            {vertexCount} Verts · {faceCount} Faces
          </span>
        </div>

        <div className="flame-view-toggles">
          <button
            type="button"
            className={`flame-toggle-btn ${solid ? 'active' : ''}`}
            onClick={() => setSolid(!solid)}
            aria-pressed={solid}
            title="Toggle solid shaded surface"
          >
            Solid
          </button>
          <button
            type="button"
            className={`flame-toggle-btn ${wireframe ? 'active' : ''}`}
            onClick={() => setWireframe(!wireframe)}
            aria-pressed={wireframe}
            title="Toggle wireframe topology overlay"
          >
            Wireframe
          </button>
          {mesh?.texture && (
            <button
              type="button"
              className={`flame-toggle-btn ${showTexture ? 'active' : ''}`}
              onClick={() => setShowTexture(!showTexture)}
              aria-pressed={showTexture}
              title="Toggle UV surface texture"
            >
              Texture
            </button>
          )}
        </div>
      </div>

      <div className="flame-canvas-container" style={{ width: '100%', height: '260px', position: 'relative' }}>
        <GuardedCanvas label="FLAME mesh viewer" camera={{ position: [0, 0, 2.8], fov: 45 }}
          fallback={({ reason, retry, use3D }) => <WebGLFallbackInspector mesh={mesh} reason={reason} onRetry={retry} onUse3D={use3D} label="Lightweight FLAME inspector" />}>
          <FlameScene
            mesh={mesh}
            poseStrength={blend}
            wireframe={wireframe}
            solid={solid}
            showTexture={showTexture}
          />
          <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
          <GizmoHelper alignment="bottom-right" margin={[40, 40]}>
            <GizmoViewport axisColors={['#ef4444', '#10b981', '#3b82f6']} labelColor="#ffffff" />
          </GizmoHelper>
        </GuardedCanvas>
      </div>

      <div className="flame-blend-slider">
        <label className="flame-slider-label">
          <span>Dual-Endpoint Blend ({Math.round(blend * 100)}% 3D)</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={blend}
            onChange={e => setBlend(parseFloat(e.target.value))}
            aria-label="3D Mesh Projection Blend (0% Flat to 100% 3D)"
          />
        </label>
        <span className="flame-provenance">
          {mesh?.provenance ? `Provenance: ${mesh.provenance}` : 'Zero-copy local topology'}
        </span>
      </div>
    </div>
  );
}
