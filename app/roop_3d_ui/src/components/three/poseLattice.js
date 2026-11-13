import * as THREE from 'three';

/**
 * Generates an evenly distributed Fibonacci sphere lattice.
 * @param {number} numPoints
 * @param {number} radius
 * @returns {{index:number,position:[number,number,number],normal:THREE.Vector3,yawDeg:number,pitchDeg:number,sampleCount:number}[]}
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
    points.push({
      index: i,
      position: [pos.x, pos.y, pos.z],
      normal,
      yawDeg: (Math.atan2(x, z) * 180) / Math.PI,
      pitchDeg: (Math.asin(Math.max(-1, Math.min(1, y))) * 180) / Math.PI,
      sampleCount: 0,
    });
  }
  return points;
}

/**
 * Projects a set of [yaw, pitch] views onto the Fibonacci lattice.
 * @param {ReturnType<typeof generateFibonacciSphere>} lattice
 * @param {Array<{ yaw: number; pitch: number; roll?: number }>} views
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

    if (bestDot > 0.7) updated[bestIdx].sampleCount += 1;
  }
  return updated;
}
