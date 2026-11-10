/**
 * Interactive 3D gizmos for the head viewport: jaw opening (theta_jaw), eye-gaze target (theta_eyes) and key-light direction.
 * Each is a draggable handle that reports a value through a callback; the inspector maps those onto the real workstation
 * parameters (`face_expression_overrides.mouth_open`, `.gaze_x/.gaze_y`, `manual_light_yaw/pitch`). Keyboard users get the same
 * values from the sliders in the viewport's gizmo dock, because canvas handles cannot take focus.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import { chinAt, gazeToPoint, jawAngleFromPoint, lightAngles, lightVector, pointToGaze, GAZE_DEPTH, GAZE_HALF_EXTENT } from './viewportTools';

const AMBER = '#f59e0b';
const SKY = '#38bdf8';
const MINT = '#34d399';
const HANDLE = new THREE.SphereGeometry(0.06, 16, 16);
/** Invisible, larger pick target so a handle is easy to grab at any zoom. */
const HIT = new THREE.SphereGeometry(0.15, 8, 8);
const LIGHT_SHELL_RADIUS = 1.9;

/** Pointer capture plus "orbit controls off while a handle is held", restored even when the gizmo unmounts mid-drag. */
function useHandleDrag() {
  const controls = useThree(state => state.controls);
  const gl = useThree(state => state.gl);
  const held = useRef(false);
  useEffect(() => () => { if (held.current && controls) controls.enabled = true; gl.domElement.style.cursor = ''; }, [controls, gl]);
  return useMemo(() => ({
    begin(event) {
      event.stopPropagation();
      event.target.setPointerCapture?.(event.pointerId);
      held.current = true;
      if (controls) controls.enabled = false;
      gl.domElement.style.cursor = 'grabbing';
    },
    end(event) {
      event.target.releasePointerCapture?.(event.pointerId);
      held.current = false;
      if (controls) controls.enabled = true;
      gl.domElement.style.cursor = '';
    },
    hover(on) { if (!held.current) gl.domElement.style.cursor = on ? 'grab' : ''; },
    get active() { return held.current; },
  }), [controls, gl]);
}

/** A draggable handle: the visible dot plus an invisible pick sphere, both under one group that carries the pointer handlers. */
function Handle({ position, colour, enabled, name, drag, scale = 1, onMove }) {
  const [hot, setHot] = useState(false);
  return (
    <group position={position} name={name}
      onPointerOver={() => { if (enabled) { setHot(true); drag.hover(true); } }} onPointerOut={() => { setHot(false); drag.hover(false); }}
      onPointerDown={event => { if (enabled) drag.begin(event); }} onPointerMove={onMove} onPointerUp={drag.end} onPointerCancel={drag.end}>
      <mesh geometry={HANDLE} scale={hot ? scale * 1.35 : scale} renderOrder={20}>
        <meshBasicMaterial color={colour} depthTest={false} transparent opacity={enabled ? 1 : 0.55} />
      </mesh>
      <mesh geometry={HIT} renderOrder={21}><meshBasicMaterial transparent opacity={0} depthWrite={false} depthTest={false} /></mesh>
    </group>
  );
}

const lineGeometry = points => new THREE.BufferGeometry().setFromPoints(points.map(point => new THREE.Vector3(...point)));

function useGeometry(points) {
  const geometry = useMemo(() => lineGeometry(points), [points]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return geometry;
}

/** The ray of a pointer event in the local frame of `object` (the head group, which turns with the target pose). */
function localRay(event, object) {
  const inverse = new THREE.Matrix4().copy(object.matrixWorld).invert();
  return event.ray.clone().applyMatrix4(inverse);
}

/** Lower-jaw hinge: drag the chin handle down and back. Lives inside the head group. */
export function JawGizmo({ rig, theta, onChange, enabled }) {
  const drag = useHandleDrag();
  const root = useRef(null);
  const chin = chinAt(rig, theta);
  const arc = useMemo(() => {
    const steps = Math.max(2, Math.ceil(Math.abs(theta) / 2));
    return Array.from({ length: steps + 1 }, (_, i) => { const [y, z] = chinAt(rig, (theta * i) / steps); return [0, y, z]; });
  }, [rig, theta]);
  const spoke = useMemo(() => [[0, rig.hinge[0], rig.hinge[1]], [0, chin[0], chin[1]]], [rig, chin[0], chin[1]]); // eslint-disable-line react-hooks/exhaustive-deps
  const arcGeometry = useGeometry(arc);
  const spokeGeometry = useGeometry(spoke);
  const plane = useMemo(() => new THREE.Plane(new THREE.Vector3(1, 0, 0), 0), []);
  const move = event => {
    if (!drag.active || !root.current) return;
    const ray = localRay(event, root.current.parent);
    const hit = ray.intersectPlane(plane, new THREE.Vector3());
    if (hit) onChange(jawAngleFromPoint(rig, [hit.y, hit.z]));
  };
  return (
    <group ref={root} name="jaw-gizmo" visible>
      <lineSegments geometry={spokeGeometry} renderOrder={19}><lineBasicMaterial color={AMBER} transparent opacity={0.8} depthTest={false} /></lineSegments>
      {theta > 0.5 && <line geometry={arcGeometry} renderOrder={19}><lineBasicMaterial color={AMBER} depthTest={false} /></line>}
      <Handle position={[0, chin[0], chin[1]]} colour={enabled ? AMBER : '#64748b'} enabled={enabled} name="jaw-handle" drag={drag} onMove={move} />
    </group>
  );
}

/** Eye-gaze target: a point on a plane in front of the eyes (head-local), with a sight line from each eye. Lives inside the head group. */
export function GazeGizmo({ eyes, gaze, onChange, enabled }) {
  const drag = useHandleDrag();
  const root = useRef(null);
  const origin = useMemo(() => [(eyes.left[0] + eyes.right[0]) / 2, (eyes.left[1] + eyes.right[1]) / 2, (eyes.left[2] + eyes.right[2]) / 2], [eyes]);
  const point = gazeToPoint(gaze, origin);
  const sight = useMemo(() => [eyes.left, point, eyes.right, point], [eyes, point[0], point[1], point[2]]); // eslint-disable-line react-hooks/exhaustive-deps
  const frame = useMemo(() => {
    const z = origin[2] + GAZE_DEPTH, e = GAZE_HALF_EXTENT;
    return [[origin[0] - e, origin[1] - e, z], [origin[0] + e, origin[1] - e, z], [origin[0] + e, origin[1] + e, z], [origin[0] - e, origin[1] + e, z], [origin[0] - e, origin[1] - e, z]];
  }, [origin]);
  const sightGeometry = useGeometry(sight);
  const frameGeometry = useGeometry(frame);
  const plane = useMemo(() => new THREE.Plane(new THREE.Vector3(0, 0, 1), -(origin[2] + GAZE_DEPTH)), [origin]);
  const move = event => {
    if (!drag.active || !root.current) return;
    const hit = localRay(event, root.current.parent).intersectPlane(plane, new THREE.Vector3());
    if (!hit) return;
    const next = pointToGaze([hit.x, hit.y, hit.z], origin);
    onChange(next.x, next.y);
  };
  return (
    <group ref={root} name="gaze-gizmo">
      <lineSegments geometry={sightGeometry} renderOrder={19}><lineBasicMaterial color={MINT} transparent opacity={0.75} depthTest={false} /></lineSegments>
      <lineLoop geometry={frameGeometry} renderOrder={18}><lineBasicMaterial color={MINT} transparent opacity={0.28} depthTest={false} /></lineLoop>
      <Handle position={point} colour={enabled ? MINT : '#64748b'} enabled={enabled} name="gaze-handle" drag={drag} onMove={move} />
    </group>
  );
}

/** Key-light direction: a handle on a shell around the head, in the world frame (the light does not turn with the head). */
export function LightGizmo({ light, onChange, enabled }) {
  const drag = useHandleDrag();
  const position = lightVector(light.yaw, light.pitch).map(value => value * LIGHT_SHELL_RADIUS);
  const beam = useMemo(() => [position, [0, 0, 0]], [position[0], position[1], position[2]]); // eslint-disable-line react-hooks/exhaustive-deps
  const beamGeometry = useGeometry(beam);
  const shell = useMemo(() => new THREE.Sphere(new THREE.Vector3(), LIGHT_SHELL_RADIUS), []);
  const colour = enabled ? (light.manual ? AMBER : SKY) : '#64748b';
  const move = event => {
    if (!drag.active) return;
    // The pointer keeps moving after it leaves the shell: fall back to the shell point closest to the ray.
    let hit = event.ray.intersectSphere(shell, new THREE.Vector3());
    if (!hit) {
      const near = event.ray.closestPointToPoint(shell.center, new THREE.Vector3());
      hit = near.lengthSq() < 1e-9 ? new THREE.Vector3(0, 0, 1) : near.normalize().multiplyScalar(LIGHT_SHELL_RADIUS);
    }
    const angles = lightAngles([hit.x, hit.y, hit.z]);
    onChange(angles.yaw, angles.pitch);
  };
  return (
    <group name="light-gizmo">
      <lineSegments geometry={beamGeometry} renderOrder={19}><lineBasicMaterial color={colour} transparent opacity={0.6} depthTest={false} /></lineSegments>
      <Handle position={position} colour={colour} enabled={enabled} name="light-handle" drag={drag} scale={1.15} onMove={move} />
    </group>
  );
}
