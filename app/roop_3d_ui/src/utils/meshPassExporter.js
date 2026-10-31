import * as THREE from 'three';

/** Render the actual frame projection, never the orbit/inspection camera.
 * Background alpha=0. Depth is linear estimated camera Z, not physical metres.
 * Floating-point PFM retains raw camera units and signed normal components.
 */
export function renderAlignedMeshPasses({ width, height, aligned_geometry }) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width * height > 3840 * 2160 || !aligned_geometry?.length) throw new Error('Frame-aligned geometry is required (up to 4K)');
  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
  const target = new THREE.WebGLRenderTarget(width, height, { type: THREE.FloatType, format: THREE.RGBAFormat, depthBuffer: true });
  const scene = new THREE.Scene(), camera = new THREE.Camera(), objects = [];
  const material = new THREE.RawShaderMaterial({ uniforms: { depthPass: { value: false } }, side: THREE.DoubleSide,
    vertexShader: `precision highp float; attribute vec3 position; attribute vec3 normal; varying vec3 n; varying float z;
      void main(){n=normal; z=position.z; float nearZ=0.0001; float farZ=2048.0;
        gl_Position=vec4(position.xy,((farZ+nearZ)/(farZ-nearZ))*z-2.0*farZ*nearZ/(farZ-nearZ),z);}`,
    fragmentShader: `precision highp float; uniform bool depthPass; varying vec3 n; varying float z;
      void main(){gl_FragColor=depthPass?vec4(z,z,z,1.0):vec4(normalize(n),1.0);}` });
  try {
    if (!renderer.extensions.has('EXT_color_buffer_float')) throw new Error('Floating-point WebGL render targets are unavailable');
    renderer.setSize(width, height, false); renderer.setClearColor(0, 0); renderer.toneMapping = THREE.NoToneMapping;
    for (const mesh of aligned_geometry) {
      const positions = [], normals = [];
      for (const face of mesh.faces) {
        const v = face.map(index => new THREE.Vector3(...mesh.camera_vertices[index]));
        const n = new THREE.Vector3().subVectors(v[1], v[0]).cross(new THREE.Vector3().subVectors(v[2], v[0])).normalize();
        for (const i of face) {
          const [x, y] = mesh.pixels[i], z = mesh.camera_vertices[i][2];
          if (![x, y, z].every(Number.isFinite) || z <= 0 || z >= 2048) throw new Error('Invalid aligned geometry or depth range');
          positions.push((2 * x / width - 1) * z, (1 - 2 * y / height) * z, z); normals.push(n.x, n.y, n.z);
        }
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
      const object = new THREE.Mesh(geometry, material); object.frustumCulled = false; scene.add(object); objects.push(object);
    }
    renderer.setRenderTarget(target);
    const output = {};
    for (const name of ['normal', 'depth']) {
      material.uniforms.depthPass.value = name === 'depth';
      renderer.clear(); renderer.render(scene, camera);
      const values = new Float32Array(width * height * 4);
      renderer.readRenderTargetPixels(target, 0, 0, width, height, values);
      if (!values.some((value, index) => index % 4 === 3 && value > 0)) throw new Error('Aligned mesh produced no visible pass pixels');
      output[name] = values;
    }
    return { width, height, ...output };
  } finally { for (const object of objects) object.geometry.dispose(); material.dispose(); target.dispose(); renderer.dispose(); }
}

export function passPFM(width, height, rgba, channels = 3) {
  const values = new Float32Array(width * height * channels);
  for (let i = 0; i < width * height; i++) for (let c = 0; c < channels; c++) values[i * channels + c] = rgba[i * 4 + c];
  // WebGL and PFM both store the bottom row first. Explicit little-endian floats.
  const bytes = new Uint8Array(values.length * 4), view = new DataView(bytes.buffer);
  values.forEach((value, i) => view.setFloat32(i * 4, value, true));
  return new Blob([`${channels === 3 ? 'PF' : 'Pf'}\n${width} ${height}\n-1.0\n`, bytes], { type: 'application/octet-stream' });
}

export function downloadPass(filename, blob) {
  const url = URL.createObjectURL(blob), anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
