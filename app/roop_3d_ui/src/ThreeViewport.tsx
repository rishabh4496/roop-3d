import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { Coverage, Reconstruction, Track } from './contracts';

export function ThreeViewport({ coverage, track, mesh }: { coverage: Coverage | null; track: Track | null; mesh: Reconstruction | null }) {
  const host = useRef<HTMLDivElement>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!host.current) return;
    const container = host.current;
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); }
    catch { setError('WebGL is unavailable. Observed pose bins remain listed below.'); return; }
    setError('');
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, .01, 100);
    camera.position.set(2, 1.1, 2.8);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 1.3; controls.maxDistance = 7;
    const axes = new THREE.AxesHelper(1.35);
    const mean = (range: [number, number] | null | undefined) => range ? (range[0] + range[1]) * Math.PI / 360 : 0;
    axes.rotation.set(mean(track?.pose_range.pitch_degrees), mean(track?.pose_range.yaw_degrees), -mean(track?.pose_range.roll_degrees));
    scene.add(axes);
    const resources: { dispose(): void }[] = [];
    const textures: THREE.Texture[] = [];
    if (mesh) {
      const positions: number[] = [], uvs: number[] = [];
      for (let faceIndex = 0; faceIndex < mesh.faces.length; faceIndex++) {
        const face = mesh.faces[faceIndex];
        face.forEach((vertexIndex, index) => {
          const vertex = mesh.vertices[vertexIndex];
          positions.push(vertex[0], mesh.coordinate_system === 'opencv' ? -vertex[1] : vertex[1], mesh.coordinate_system === 'opencv' ? -vertex[2] : vertex[2]);
          const uv = mesh.uvs?.[mesh.face_uvs?.[faceIndex]?.[index] ?? vertexIndex];
          if (uv) uvs.push(uv[0], uv[1]);
        });
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      if (uvs.length === positions.length / 3 * 2) geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
      geometry.computeVertexNormals(); geometry.computeBoundingSphere();
      const material = new THREE.MeshStandardMaterial({ color: 0xadc4c4, side: THREE.DoubleSide, roughness: .85 });
      if (mesh.texture && geometry.getAttribute('uv')) { const texture = new THREE.TextureLoader().load(mesh.texture); texture.colorSpace = THREE.SRGBColorSpace; material.map = texture; textures.push(texture); }
      const object = new THREE.Mesh(geometry, material);
      const sphere = geometry.boundingSphere!;
      object.scale.setScalar(1 / Math.max(sphere.radius, .0001));
      object.position.copy(sphere.center).multiplyScalar(-object.scale.x);
      scene.add(object, new THREE.HemisphereLight(0xffffff, 0x334455, 3));
      const light = new THREE.DirectionalLight(0xffffff, 3); light.position.set(2, 3, 4); scene.add(light);
      resources.push(geometry, material);
    } else {
      const sphere = new THREE.SphereGeometry(1, 24, 12, 0, Math.PI, 0, Math.PI);
      const material = new THREE.MeshBasicMaterial({ color: 0x527a7c, wireframe: true, transparent: true, opacity: .18 });
      scene.add(new THREE.Mesh(sphere, material)); resources.push(sphere, material);
      const points = (coverage?.views ?? []).flatMap(view => { const yaw = view.yaw * Math.PI / 180, pitch = view.pitch * Math.PI / 180; return [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)]; });
      const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
      const pointMaterial = new THREE.PointsMaterial({ color: 0x6bdfc0, size: .075 });
      scene.add(new THREE.Points(geometry, pointMaterial)); resources.push(geometry, pointMaterial);
    }
    const resize = () => { const width = container.clientWidth, height = container.clientHeight; renderer.setSize(width, height); camera.aspect = width / Math.max(height, 1); camera.updateProjectionMatrix(); };
    const observer = new ResizeObserver(resize); observer.observe(container); resize();
    let animation = 0;
    const render = () => { controls.update(); renderer.render(scene, camera); animation = requestAnimationFrame(render); };
    render();
    return () => { cancelAnimationFrame(animation); observer.disconnect(); controls.dispose(); axes.geometry.dispose(); (axes.material as THREE.Material).dispose(); resources.forEach(item => item.dispose()); textures.forEach(texture => texture.dispose()); renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove(); };
  }, [coverage, track, mesh]);
  return <div className="three-host" ref={host} aria-label={mesh ? 'Imported reconstruction viewer' : 'Estimated pose axes and observed source coverage'}>{error && <p className="empty-caption">{error}</p>}</div>;
}
export function PoseCoverageInspector({ coverage, track, mesh }: { coverage: Coverage | null; track: Track | null; mesh: Reconstruction | null }) {
  return <section className="pose-inspector"><div className="section-title"><span>POSE & GEOMETRY</span><span className="tiny-badge">{mesh?.method === 'flame' ? 'FLAME MESH' : 'ESTIMATED 2.5D'}</span></div>
    <ThreeViewport coverage={coverage} track={track} mesh={mesh} />
    <p className="muted compact">{mesh ? `Imported ${mesh.method === 'flame' ? 'FLAME 3D' : 'approximate 2.5D'} mesh · user supplied provenance` : 'Orientation axes + observed FaceSet directions. No reconstructed head mesh loaded.'}</p>
    {mesh?.texture && <details><summary>UV texture</summary><img className="uv-texture" src={mesh.texture} alt="Actual reconstruction UV texture" /></details>}
    <div className="pose-legend"><span><i className="dot mint"/>Observed views {coverage?.views.length ?? 0}</span><span><i className="dot dim"/>Unobserved</span></div>
    {coverage && <div className="pose-bins">{Object.entries(coverage.bins).map(([bin, count]) => <span key={bin}>{bin.replaceAll('_', ' ')} <strong>{count}</strong></span>)}</div>}
  </section>;
}
