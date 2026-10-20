import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { CONFIG } from './config.js';
import { loadCarTemplate, instantiateCar, getWheelRig } from './car.js';
import { buildGround, slotMarking, addLights } from './world.js';
import { makeHitbox, flashHitbox, makeActionVectors, updateActionVectors } from './diagnostics.js';

const $ = id => document.getElementById(id);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
scene.fog = new THREE.Fog(0x0d1117, 45, 110);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
addLights(renderer, scene);

const camera = new THREE.PerspectiveCamera(48, innerWidth / innerHeight, 0.1, 400);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true; orbit.maxPolarAngle = Math.PI / 2 - 0.02;
camera.position.set(10, 8, 10);

let S = null;          // current episode state
let playing = true, t = 0, last = performance.now();

async function loadEpisode(url) {
  const data = await (await fetch(url, { cache: 'no-store' })).json();
  if (S) scene.remove(S.root);
  const root = new THREE.Group(); scene.add(root);
  const car = data.car;
  const template = await loadCarTemplate(car.length, car.width);

  buildGroundOnce(root, data.bounds);
  root.add(slotMarking(data.slot, car, CONFIG.colors.slot));
  const slotGroup = root.children[root.children.length - 1];

  const hitboxes = [];
  const palette = [0x9aa5b1, 0xc0392b, 0x2c3e50, 0xd5d8dc, 0x1e8449, 0xb9770e];
  data.obstacles.forEach((o, i) => {
    const [x0, y0, x1, y1] = o; const w = x1 - x0, h = y1 - y0;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const isCar = Math.max(w, h) > 3.2 && Math.max(w, h) < 5.5 && Math.min(w, h) > 1.4 && Math.min(w, h) < 2.4;
    if (isCar) {
      const parked = instantiateCar(template, palette[i % palette.length]);
      parked.position.set(cx, 0, -cy);
      parked.rotation.y = w > h ? 0 : -Math.PI / 2;      // long side along x, or along y
      root.add(parked);
      const hb = makeHitbox(Math.max(w, h), Math.min(w, h), 1.5, 0);   // exact AABB the sim collides with
      hb.position.set(cx, 0.75, -cy); hb.rotation.y = w > h ? 0 : -Math.PI / 2;
      root.add(hb); hitboxes.push(hb);
    } else {
      const wall = new THREE.Mesh(new THREE.BoxGeometry(w, 1.2, h), new THREE.MeshStandardMaterial({ color: 0x59606a, roughness: .9 }));
      wall.position.set(cx, 0.6, -cy); wall.castShadow = wall.receiveShadow = true; root.add(wall);
      const hb = makeHitbox(w, h, 1.2, 0); hb.position.set(cx, 0.6, -cy); root.add(hb); hitboxes.push(hb);
    }
  });

  const ego = instantiateCar(template, CONFIG.colors.ego);
  const egoHit = makeHitbox(car.length, car.width, 1.5);
  const vectors = makeActionVectors(car);
  ego.add(egoHit, vectors);
  root.add(ego); hitboxes.push(egoHit);

  const pts = data.frames.map(f => new THREE.Vector3(f.x, 0.04, -f.y));
  const trail = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: CONFIG.colors.trail }));
  root.add(trail);

  const rig = getWheelRig(ego);
  const cum = [0];                                  // signed distance travelled, for wheel roll
  for (let i = 1; i < data.frames.length; i++) cum.push(cum[i - 1] + data.frames[i].v * data.dt);
  S = { rig, cum, data, root, ego, egoHit, hitboxes, vectors, trail, slotGroup, N: data.frames.length - 1 };
  const r = data.result;
  $('title').textContent = `${data.controller || 'unknown controller'} · ${data.kind} · seed ${data.seed}`;
  $('result').innerHTML = r.success ? '<span style="color:var(--ok)">✔ parked</span>'
    : r.collision ? '<span style="color:var(--bad)">✖ collision</span>' : '<span style="color:var(--bad)">✖ timeout</span>';
  $('result').innerHTML += ` <span style="color:var(--dim);font-weight:400">pos err ${r.pos_error.toFixed(2)} m · hdg ${r.heading_error_deg.toFixed(1)}°</span>`;
  window.__S = S; window.__camera = camera; window.__orbit = orbit;   // debug handles
  t = 0; playing = true; $('play').textContent = '⏸ Pause';
  applyOptions(); apply(0);
  const t0 = new URLSearchParams(location.search).get('t'); if (t0 !== null) { t = parseFloat(t0); playing = false; $('play').textContent = '▶ Play'; apply(t); }
  const [sx, sy] = data.slot; orbit.target.set(sx, 0, -sy);
  camera.position.set(sx + 9, 8, -sy + 9);
}

let groundBuilt = false;
function buildGroundOnce(root, bounds) { buildGround(root, bounds); }

function lerpFrame(a, b, k) {
  let dth = b.theta - a.theta; dth = Math.atan2(Math.sin(dth), Math.cos(dth));
  return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k, theta: a.theta + dth * k,
    v: a.v + (b.v - a.v) * k, delta: a.delta + (b.delta - a.delta) * k, action: [a.action[0] + (b.action[0] - a.action[0]) * k, a.action[1] + (b.action[1] - a.action[1]) * k] };
}

function apply(time) {
  const i = Math.min(S.N - 1, Math.max(0, Math.floor(time))), k = Math.min(1, time - i);
  const A = S.data.frames[i], B = S.data.frames[Math.min(S.N, i + 1)];
  const f = S.N === 0 ? A : lerpFrame(A, B, k);
  S.ego.position.set(f.x, 0, -f.y); S.ego.rotation.y = f.theta;
  const dist = S.cum[i] + (S.cum[Math.min(S.N, i + 1)] - S.cum[i]) * k;
  S.rig.steer.forEach(w => { w.rotation.y = f.delta; });       // actual (rate-limited) wheel angle
  S.rig.spin.forEach(w => { w.rotation.z = -dist / S.rig.radius; });
  // the action in frame i+1 is the command that PRODUCED that frame, so show it while moving i -> i+1
  const act = S.N === 0 ? A.action : B.action;
  updateActionVectors(S.vectors, act, f.v, !!S.data.reverse);
  flashHitbox(S.egoHit, B.collision && k > 0.5);
  const idx = Math.round(time);
  const F = S.data.frames[Math.min(S.N, idx)];
  $('v-step').textContent = `${idx} / ${S.N}`;
  $('v-speed').textContent = `${f.v.toFixed(2)} m/s`;
  $('v-steer').textContent = act[0].toFixed(2);
  const brk = S.data.reverse ? ((act[1] < 0 && f.v > 0.05) || (act[1] > 0 && f.v < -0.05)) : act[1] < 0;
  $('v-thb').textContent = act[1].toFixed(2) + (brk ? ' (brake)' : (f.v < -0.05 ? ' (reverse)' : ''));
  $('v-shield').textContent = F.shield ? 'ACTIVE' : 'off';
  $('v-shield').style.color = F.shield ? 'var(--bad)' : '';
  bar($('b-steer'), act[0]); bar($('b-thb'), act[1], true);
  $('scrub').value = Math.round((time / Math.max(1, S.N)) * 1000);
}
function bar(el, v, color) {
  const w = Math.abs(v) * 50; el.style.left = (v >= 0 ? 50 : 50 - w) + '%'; el.style.width = w + '%';
  if (color) el.style.background = v >= 0 ? 'var(--ok)' : 'var(--bad)';
}

function applyOptions() {
  S.hitboxes.forEach(h => h.visible = $('o-hit').checked);
  S.vectors.visible = $('o-act').checked;
  S.trail.visible = $('o-trail').checked;
  S.slotGroup.visible = $('o-slot').checked;
}
['o-hit', 'o-act', 'o-trail', 'o-slot'].forEach(id => $(id).addEventListener('change', () => S && applyOptions()));

$('play').onclick = () => { playing = !playing; $('play').textContent = playing ? '⏸ Pause' : '▶ Play'; last = performance.now(); if (playing && t >= S.N) t = 0; };
$('replay').onclick = () => { t = 0; playing = true; $('play').textContent = '⏸ Pause'; last = performance.now(); };
$('scrub').oninput = e => { t = (e.target.value / 1000) * S.N; playing = false; $('play').textContent = '▶ Play'; apply(t); };
$('cam').onchange = () => { orbit.enabled = $('cam').value === 'orbit'; };
CONFIG.episodes.forEach((e, i) => { const o = document.createElement('option'); o.value = e.url; o.textContent = e.label; $('traj').appendChild(o); });
$('traj').onchange = () => loadEpisode($('traj').value);
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });

function cameraRig() {
  const mode = $('cam').value;
  if (mode === 'chase') {
    const p = S.ego.position, th = S.ego.rotation.y;
    camera.position.lerp(new THREE.Vector3(p.x - Math.cos(th) * 9, 4.5, p.z + Math.sin(th) * 9), 0.1);
    camera.lookAt(p.x, 0.8, p.z);
  } else if (mode === 'top') {
    const [sx, sy] = S.data.slot;
    camera.position.lerp(new THREE.Vector3(sx, 24, -sy + 0.01), 0.1); camera.lookAt(sx, 0, -sy);
  } else orbit.update();
}

function tick(now) {
  const dt = (now - last) / 1000; last = now;
  if (S) {
    if (playing) {
      t += dt / S.data.dt * parseFloat($('rate').value) / 1;   // real time: 1 sim step = dt seconds
      if (t >= S.N) { t = S.N; playing = false; $('play').textContent = '▶ Play'; }
      apply(t);
    }
    cameraRig();
  }
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

const qs = new URLSearchParams(location.search);
try {
  await loadEpisode(qs.get('file') || CONFIG.episodes[0].url);
  $('loading').classList.add('done');
  window.__ready = true;
  requestAnimationFrame(tick);
} catch (e) {
  $('msg').innerHTML = `Could not load the episode.<br><br><code>${e}</code><br><br>Serve this folder over HTTP: <code>python3 -m http.server 8000</code> then open <code>http://localhost:8000</code>.`;
  console.error(e);
}
