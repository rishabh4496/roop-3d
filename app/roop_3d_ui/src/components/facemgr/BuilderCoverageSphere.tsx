import { OrbitControls } from '@react-three/drei';
import GuardedCanvas from '../three/GuardedCanvas';

/** The same 12 x 8 frontal hemisphere used by Stage 3, with selectable missing bins. */
export default function BuilderCoverageSphere({ covered, selectedBin, onSelect }: { covered: Set<number>; selectedBin: number | null; onSelect: (index: number) => void }) {
  return <div className="fsb-sphere" aria-label="Rotatable 3D sphere coverage; use the adjacent buttons for keyboard selection">
    <GuardedCanvas label="Coverage sphere" camera={{ position: [0, 0, 3.4], fov: 42 }} dpr={[1, 1.5]}
      fallback={({ reason, retry }) => <p className="muted" role="status">The 3D sphere is not drawing ({reason}) Pick missing angles with the buttons next to it.{retry && <> <button type="button" className="button small" onClick={retry}>Restart 3D view</button></>}</p>}>
      <ambientLight intensity={1.2}/><directionalLight position={[3, 4, 5]} intensity={2}/>
      <mesh><sphereGeometry args={[.98, 24, 16]}/><meshStandardMaterial color="#172d3c" roughness={.8}/></mesh>
      <mesh><sphereGeometry args={[.99, 24, 16]}/><meshBasicMaterial color="#476074" wireframe transparent opacity={.25}/></mesh>
      {Array.from({ length: 96 }, (_, i) => {
        const yaw = (-82.5 + i % 12 * 15) * Math.PI / 180;
        const pitch = (-52.5 + Math.floor(i / 12) * 15) * Math.PI / 180;
        return <mesh key={i} position={[Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)]} onClick={event => { event.stopPropagation(); onSelect(i); }}>
          <sphereGeometry args={[selectedBin === i ? .055 : .035, 10, 8]}/>
          <meshBasicMaterial color={selectedBin === i ? '#fff' : covered.has(i) ? '#55e8ab' : '#738698'}/>
        </mesh>;
      })}
      <OrbitControls enablePan={false} minDistance={2} maxDistance={5}/>
    </GuardedCanvas>
  </div>;
}
