import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
  generateFibonacciSphere,
  projectViewsToLattice,
  createCanonicalHeadGeometry,
} from './components/three/PoseGeometryViewer';

describe('Fibonacci Sphere Lattice & View Coverage Heatmap', () => {
  it('generates an evenly distributed Fibonacci sphere lattice', () => {
    const numPoints = 96;
    const radius = 1.35;
    const lattice = generateFibonacciSphere(numPoints, radius);

    expect(lattice).toHaveLength(numPoints);

    // Verify all nodes lie on sphere of radius 1.35
    for (const node of lattice) {
      const v = new THREE.Vector3(...node.position);
      expect(v.length()).toBeCloseTo(radius, 4);
      expect(node.normal.length()).toBeCloseTo(1.0, 4);
      expect(node.sampleCount).toBe(0);
    }

    // Top node has y close to radius, bottom node has y close to -radius
    expect(lattice[0].position[1]).toBeCloseTo(radius, 1);
    expect(lattice[lattice.length - 1].position[1]).toBeCloseTo(-radius, 1);
  });

  it('projects [yaw, pitch] view angles to appropriate lattice nodes and marks coverage', () => {
    const lattice = generateFibonacciSphere(96, 1.35);

    // Frontal view (yaw 0, pitch 0)
    const views = [
      { yaw: 0, pitch: 0, roll: 0, pose_bin: 'frontal' },
      { yaw: 1, pitch: -1, roll: 0, pose_bin: 'frontal' },
      { yaw: -1, pitch: 2, roll: 0, pose_bin: 'frontal' },
      { yaw: 0.5, pitch: 0.5, roll: 0, pose_bin: 'frontal' }, // 4 samples around frontal -> Emerald Green (>3)
      { yaw: 45, pitch: 15, roll: 0, pose_bin: 'angled' },    // 1 sample -> Amber (1)
    ];

    const mapped = projectViewsToLattice(lattice, views);

    // At least one node has >= 3 samples (Emerald Green)
    const wellObserved = mapped.filter(n => n.sampleCount >= 3);
    expect(wellObserved.length).toBeGreaterThanOrEqual(1);

    // At least one node has 1 sample (Amber)
    const sparse = mapped.filter(n => n.sampleCount === 1);
    expect(sparse.length).toBeGreaterThanOrEqual(1);

    // Unobserved nodes (Slate)
    const unobserved = mapped.filter(n => n.sampleCount === 0);
    expect(unobserved.length).toBeGreaterThan(80);

    // Coverage stats
    const observedCount = mapped.filter(n => n.sampleCount > 0).length;
    const coveragePct = Math.round((observedCount / mapped.length) * 100);
    expect(coveragePct).toBeGreaterThan(0);
    expect(coveragePct).toBeLessThan(50);
  });
});

describe('Canonical 3D Head Mesh Geometry', () => {
  it('creates a low-poly canonical head geometry centered at origin [0, 0, 0]', () => {
    const geom = createCanonicalHeadGeometry();

    expect(geom.getAttribute('position')).toBeDefined();
    expect(geom.getIndex()).toBeDefined();
    expect(geom.getAttribute('normal')).toBeDefined();

    geom.computeBoundingBox();
    const bb = geom.boundingBox!;
    const center = new THREE.Vector3();
    bb.getCenter(center);

    // Centered at [0, 0, 0]
    expect(center.x).toBeCloseTo(0, 2);
    expect(center.y).toBeCloseTo(0, 2);
    expect(center.z).toBeCloseTo(0, 2);

    // Proportions
    const size = new THREE.Vector3();
    bb.getSize(size);
    expect(size.y).toBeGreaterThan(size.x); // Height is greater than width
    expect(size.z).toBeGreaterThan(0.5);   // Nose depth
  });
});
