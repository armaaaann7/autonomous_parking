import * as THREE from 'three';

/** Gridded asphalt ground: painted on a canvas so there is no external texture fetch and no
 *  z-fighting geometry. A 1 m minor / 5 m major grid doubles as a scale reference. */
export function buildGround(scene, bounds) {
  const [xmin, xmax, ymin, ymax] = bounds;
  const pad = 60;
  const W = (xmax - xmin) + pad, H = (ymax - ymin) + pad;
  const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
  const ppm = 24;
  const cv = document.createElement('canvas');
  cv.width = Math.min(4096, Math.round(W * ppm)); cv.height = Math.min(4096, Math.round(H * ppm));
  const ctx = cv.getContext('2d');
  const sx = cv.width / W, sy = cv.height / H;
  ctx.fillStyle = '#2a2d31'; ctx.fillRect(0, 0, cv.width, cv.height);
  for (let i = 0; i < 9000; i++) {   // asphalt grain
    const v = 30 + Math.random() * 28;
    ctx.fillStyle = `rgba(${v},${v},${v + 2},0.35)`;
    ctx.fillRect(Math.random() * cv.width, Math.random() * cv.height, 2, 2);
  }
  const line = (x0, y0, x1, y1, col, w) => { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); };
  const toPx = (x, y) => [(x - (cx - W / 2)) * sx, (y - (cy - H / 2)) * sy];
  for (let x = Math.ceil(cx - W / 2); x <= cx + W / 2; x++) {
    const major = x % 5 === 0; const [px] = toPx(x, 0);
    line(px, 0, px, cv.height, major ? 'rgba(120,160,200,.30)' : 'rgba(120,160,200,.10)', major ? 1.6 : 1);
  }
  for (let y = Math.ceil(cy - H / 2); y <= cy + H / 2; y++) {
    const major = y % 5 === 0; const py = ((cy + H / 2) - y) * sy;   // sim +y is three -z (no mirroring)
    line(0, py, cv.width, py, major ? 'rgba(120,160,200,.30)' : 'rgba(120,160,200,.10)', major ? 1.6 : 1);
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.MeshStandardMaterial({ map: tex, roughness: .95, metalness: 0 }));
  ground.rotation.x = -Math.PI / 2; ground.position.set(cx, 0, -cy); ground.receiveShadow = true;
  scene.add(ground);
  return ground;
}

/** White bay lines around the target slot (sim coords -> three: x, z=y). */
export function slotMarking(slot, car, color) {
  const [sx, sy, th] = slot;
  const g = new THREE.Group();
  const L = car.length + 0.6, W = car.width + 0.6;   // = the env bay (car + 0.6 m each way); was +0.8, which drew the bay 0.2 m too long
  const shape = new THREE.Mesh(new THREE.PlaneGeometry(L, W), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.18, depthWrite: false }));
  shape.rotation.x = -Math.PI / 2; shape.position.y = 0.012; g.add(shape);
  const edge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(L, 0.01, W)), new THREE.LineBasicMaterial({ color }));
  edge.position.y = 0.02; g.add(edge);
  g.position.set(sx, 0, -sy); g.rotation.y = th;
  return g;
}

export function addLights(renderer, scene) {
  scene.add(new THREE.HemisphereLight(0xbfd8ff, 0x23262b, 0.55));
  const sun = new THREE.DirectionalLight(0xfff2dd, 2.2);
  sun.position.set(14, 26, -10); sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -30, right: 30, top: 30, bottom: -30, near: 1, far: 90 });
  sun.shadow.bias = -0.0005;
  scene.add(sun);
}
