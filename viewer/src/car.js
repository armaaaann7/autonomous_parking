import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { CONFIG } from './config.js';

let _template = null;

/** Loads the real glTF car once, normalises it (centre on origin, long axis = +X, scaled to the
 *  sim's car length) and returns a template. Falls back to a procedural sedan if the file is
 *  missing so the viewer never shows a blank scene. */
export async function loadCarTemplate(length, width) {
  if (_template) return _template;
  let root;
  try {
    const gltf = await new GLTFLoader().loadAsync(CONFIG.carModelUrl);
    root = gltf.scene;
    root.updateMatrixWorld(true);
    let box = new THREE.Box3().setFromObject(root);
    const size = box.getSize(new THREE.Vector3());
    const wrap = new THREE.Group();
    wrap.add(root);
    if (size.z > size.x) { root.rotation.y = Math.PI / 2; }   // long axis -> X
    wrap.updateMatrixWorld(true);
    box = new THREE.Box3().setFromObject(wrap);
    const c = box.getCenter(new THREE.Vector3());
    root.position.sub(new THREE.Vector3(c.x, box.min.y, c.z));
    wrap.updateMatrixWorld(true);
    box = new THREE.Box3().setFromObject(wrap);
    const sx = box.max.x - box.min.x, sz = box.max.z - box.min.z;
    // 'inside': uniform scale so the visible car never exceeds the sim's collision box (what you
    // see is never bigger than what is collided). 'length': match length only (can overhang).
    const s = CONFIG.fitMode === 'length' ? length / sx : Math.min(length / sx, width / sz);
    wrap.scale.setScalar(s);
    if (CONFIG.modelForwardSign < 0) wrap.rotation.y = Math.PI;
    wrap.userData.fit = { modelLength: sx * s, modelWidth: sz * s, scale: s };
    rigWheels(wrap);
    wrap.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    root = wrap;
    root.userData.real = true;
  } catch (e) {
    console.warn('car model failed to load, using procedural fallback:', e);
    root = proceduralSedan(length, width);
    root.userData.real = false;
  }
  _template = root;
  return root;
}

/** Wraps each of the 4 wheel nodes in steer -> spin pivots at the hub, preserving the model's own
 *  transforms, so front wheels can steer (about Y) and all wheels can roll (about the Z axle). */
function rigWheels(wrap) {
  wrap.updateMatrixWorld(true);
  const found = {};
  wrap.traverse(o => { const m = /^Wheel(Front|Rear)(L|R)$/.exec(o.name || ''); if (m) found[m[1][0] + m[2]] = o; });
  let rSum = 0, n = 0;
  for (const [key, node] of Object.entries(found)) {
    const bb = new THREE.Box3().setFromObject(node);
    rSum += (bb.max.y - bb.min.y) / 2; n++;
    const hub = wrap.worldToLocal(bb.getCenter(new THREE.Vector3()));
    const steer = new THREE.Group(); steer.name = 'steer_' + key; steer.position.copy(hub);
    const spin = new THREE.Group(); spin.name = 'spin_' + key; steer.add(spin);
    wrap.add(steer); wrap.updateMatrixWorld(true);
    spin.attach(node);
  }
  wrap.userData.wheelRadius = n ? rSum / n : 0.33;   // world metres (after scaling)
  wrap.userData.wheelsRigged = n;
}

/** Wheel pivots of an instantiated car (null entries for a model without named wheels). */
export function getWheelRig(carGroup) {
  const g = n => carGroup.getObjectByName(n) || null;
  const keys = ['FL', 'FR', 'RL', 'RR'];
  const tpl = carGroup.children[0];
  return { steer: keys.filter(k => k[0] === 'F').map(k => g('steer_' + k)).filter(Boolean),
           spin: keys.map(k => g('spin_' + k)).filter(Boolean),
           radius: (tpl && tpl.userData.wheelRadius) || 0.33 };
}

export function instantiateCar(template, tint = null) {
  const g = new THREE.Group();
  const car = template.clone(true);
  if (tint != null) {
    car.traverse(o => {
      if (o.isMesh && o.material && /body|paint|color/i.test(o.material.name || '')) {
        o.material = o.material.clone(); o.material.color.setHex(tint);
      }
    });
  }
  g.add(car);
  return g;
}

function proceduralSedan(L, W) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(L, 0.6, W), new THREE.MeshStandardMaterial({ color: 0x8899aa, metalness: .6, roughness: .35 }));
  body.position.y = 0.55; g.add(body);
  const cab = new THREE.Mesh(new THREE.BoxGeometry(L * 0.5, 0.5, W * 0.9), new THREE.MeshStandardMaterial({ color: 0x223344, metalness: .3, roughness: .1 }));
  cab.position.set(-L * 0.05, 1.05, 0); g.add(cab);
  return g;
}
