import * as THREE from 'three';
import { CONFIG } from './config.js';

/** Ghosted collision volume: translucent box + wireframe, slightly larger than the car. */
export function makeHitbox(length, width, height = 1.5, pad = CONFIG.hitboxPad, color = CONFIG.colors.hitbox) {
  const g = new THREE.Group();
  const geo = new THREE.BoxGeometry(length + 2 * pad, height, width + 2 * pad);
  g.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.16, depthWrite: false })));
  g.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color })));
  g.position.y = height / 2;
  g.userData.isHitbox = true;
  return g;
}

export function flashHitbox(hitbox, on) {
  hitbox.children[0].material.opacity = on ? 0.6 : 0.16;
}

/** Steering arrow at the front axle (points where the front wheels point, length ~ |steer|),
 *  throttle/brake arrow at the rear (forward+green = throttle, backward+red = brake). */
export function makeActionVectors(car) {
  const front = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(car.length / 2, 0.5, 0), 1.5, CONFIG.colors.ego, 0.4, 0.28);
  const rear = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(-car.length / 2, 0.5, 0), 1.0, 0x3fb950, 0.4, 0.28);
  const g = new THREE.Group(); g.add(front, rear);
  g.userData = { front, rear, car };
  return g;
}

/** action = [steer, throttle_brake] in [-1, 1]. Steer is mapped to a wheel angle. The z-sign
 *  matches the sim->three mapping (z = y, rotation.y = -theta): positive steer = CCW in the sim. */
export function updateActionVectors(g, action, v = 0, reverseGear = false) {
  const { front, rear } = g.userData;
  const steer = action[0] * CONFIG.maxSteerRad;
  front.setDirection(new THREE.Vector3(Math.cos(steer), 0, -Math.sin(steer)).normalize());
  front.setLength(0.8 + 1.6 * Math.abs(action[0]), 0.4, 0.28);
  const thb = action[1];
  // Old semantics (no reverse gear): negative = brake. With a reverse gear the sign is relative to the
  // direction of travel: opposing the current motion = braking, otherwise it is throttle in that direction.
  const braking = reverseGear ? ((thb < 0 && v > 0.05) || (thb > 0 && v < -0.05)) : thb < 0;
  const dir = reverseGear ? (braking ? -Math.sign(v || 1) : Math.sign(thb)) : (thb >= 0 ? 1 : -1);
  rear.setDirection(new THREE.Vector3(dir, 0, 0));
  const len = Math.max(0.05, Math.abs(thb) * 2.2);
  rear.setLength(len, Math.min(0.4, len * 0.6), 0.28);
  rear.setColor(braking ? 0xf85149 : 0x3fb950);
  rear.visible = Math.abs(thb) > 0.02;
}
