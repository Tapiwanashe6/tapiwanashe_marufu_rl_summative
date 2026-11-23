// MediFlight AI — Three.js mission viewer.
//
// The viewer contains NO simulation logic. It loads a replay recorded from the
// Python MediFlightEnv (schema 2.0, environment/rendering.py) and reconstructs
// the state at any instant from the last recorded snapshot plus the
// environment's own timestamped events (event sourcing). Drone positions between
// take-off and arrival are interpolated for display only; times, battery
// endpoints, deliveries, expiries and weather all come from the environment.
//
// Everything else on screen — buildings, traffic, pedestrians, clouds, cameras —
// is decoration in world/*.js. It is seeded (identical on every load) and never
// reads or writes the replay state except to *display* it.
//
// Transport: `ReplaySource` fetches a file. A live source (e.g. Server-Sent
// Events streaming the same records from Python) can implement the same
// `load()` contract without changing anything below it.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { buildCity, LOT_Y } from './world/city.js';
import { buildLife } from './world/life.js';
import { createDrone, GEAR_DROP } from './world/drone.js';
import { Sky, WEATHER } from './world/sky.js';
import { CameraRig, MODES, MODE_LABEL, MODE_HINT } from './world/cameras.js';
import { Minimap } from './world/minimap.js';

const $ = (id) => document.getElementById(id);
const S = 2.2;                    // scene units per km
const PAD_H = LOT_Y + 0.08 + GEAR_DROP, HOVER_H = 2.6, CRUISE_H = 4.6;
const SIM_MIN_PER_SEC = 4;        // at 1x, one second of playback = 4 simulated minutes
const PKG_COLOR = { blood: 0xe5484d, emergency_kit: 0xf76b15, vaccine: 0x3e9bff, prescription: 0xb48cff, medicine: 0x30c48d };
const PKG_CSS = { blood: '#e5484d', emergency_kit: '#f76b15', vaccine: '#3e9bff', prescription: '#b48cff', medicine: '#30c48d' };
const URG_COLOR = { 3: 0xff5d5d, 2: 0xffb224, 1: 0x5fd4a0 };
const KIND_ICON = { hospital: '✚', pharmacy: '℞', home: '⌂', depot: '◆' };
const KIND_TAG = { hospital: 'HOSP', pharmacy: 'PHARM', home: 'HOME', depot: 'DEPOT' };

// ------------------------------------------------------------------ source
class ReplaySource {
  constructor(url) { this.url = url; }
  async load() {
    const r = await fetch(this.url, { cache: 'no-store' });
    if (!r.ok) throw new Error(`HTTP ${r.status} for ${this.url}`);
    const doc = await r.json();
    if (doc.schema_version !== '2.0') throw new Error(`Unsupported replay schema ${doc.schema_version}`);
    return doc;
  }
}

// ------------------------------------------------------------------ model
class Episode {
  constructor(doc) {
    this.doc = doc;
    this.nodes = doc.header.map.nodes;
    this.sites = this.nodes.slice(1);
    this.coldLimit = Object.fromEntries(doc.header.map.package_types.map((p) => [p.name, p.cold_chain_min]));
    this.critical = new Set(doc.header.map.package_types.filter((p) => p.urgency === 3).map((p) => p.name));
    this.steps = doc.steps;
    this.end = this.steps.length ? this.steps.at(-1).t_end : 0;
    this.flights = this.steps.map((s) => {
      const take = s.events.find((e) => e.type === 'takeoff');
      if (!take) return null;
      const arrive = s.events.find((e) => e.type === 'arrive');
      const crash = s.events.find((e) => e.type === 'crash');
      return { t0: take.t, t1: (arrive || crash).t, from: take.origin, to: take.destination,
               crash: !!crash, frac: crash ? crash.fraction : 1 };
    });
    this.allEvents = [...doc.header.events.map((e) => ({ ...e, step: -1 })),
                      ...this.steps.flatMap((s) => s.events.map((e) => ({ ...e, step: s.step })))];
  }
  stepIndexAt(t) {
    let lo = 0, hi = this.steps.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (this.steps[m].t_end <= t) lo = m + 1; else hi = m; }
    return lo;
  }
  stateBefore(i) { return i === 0 ? this.doc.header.state : this.steps[i - 1].state; }

  /** Full display state at time t, rebuilt from the previous snapshot + events up to t. */
  stateAt(t) {
    const i = this.stepIndexAt(t);
    const base = this.stateBefore(Math.min(i, this.steps.length));
    const st = {
      i, t: Math.min(t, this.end), battery: base.battery, weather: base.weather, location: base.location,
      orders: base.orders.map((o) => (o ? { ...o, coldAt: base.time } : null)),
      payload: [...base.payload], stats: { ...base.stats }, flight: null, charging: false, dropping: null,
      cumulative: i === 0 ? 0 : this.steps[i - 1].cumulative_reward,
      last: i === 0 ? null : this.steps[i - 1],
    };
    if (i >= this.steps.length) return st;
    const step = this.steps[i];
    for (const e of step.events) {
      if (e.t > t) break;
      switch (e.type) {
        case 'weather': st.weather = e.weather; break;
        case 'order_created': st.orders[e.site] = { ...e.order, coldAt: e.t }; break;
        case 'order_expired': st.orders[e.site] = null; st.stats.expired += 1; break;
        case 'load': if (st.orders[e.site]) { st.orders[e.site].status = 'onboard'; st.orders[e.site].loadedAt = e.t; }
                     st.payload.push(e.site); break;
        case 'deliver':
          st.orders[e.site] = null; st.payload = st.payload.filter((p) => p !== e.site);
          if (e.on_time) { st.stats.delivered_on_time += 1; if (this.critical.has(e.package)) st.stats.critical_on_time += 1; }
          else st.stats.delivered_late += 1;
          st.dropping = { site: e.site, t0: e.t, pkg: e.package };
          break;
        case 'spoiled': st.orders[e.site] = null; st.payload = st.payload.filter((p) => p !== e.site); st.stats.spoiled += 1; break;
        case 'crash': st.payload = []; st.crashed = e; break;
        case 'order_rejected': st.stats.rejected_busy += 1; break;
        default: break;
      }
    }
    const after = step.state;
    const f = this.flights[i];
    if (f) {
      const u = THREE.MathUtils.clamp((t - f.t0) / Math.max(1e-6, f.t1 - f.t0), 0, 1);
      st.battery = THREE.MathUtils.lerp(base.battery, after.battery, u);
      st.flight = { ...f, u };
      if (t >= f.t1) st.location = f.crash ? null : f.to;
    } else if (step.action_name === 'charge') {
      const u = THREE.MathUtils.clamp((t - step.t_start) / Math.max(1e-6, step.t_end - step.t_start), 0, 1);
      st.battery = THREE.MathUtils.lerp(base.battery, after.battery, u);
      st.charging = true;
    } else if (step.action_name === 'hold' || !step.valid) {
      const u = THREE.MathUtils.clamp((t - step.t_start) / Math.max(1e-6, step.t_end - step.t_start), 0, 1);
      st.battery = THREE.MathUtils.lerp(base.battery, after.battery, u);
    }
    st.current = step;
    return st;
  }
}

// ------------------------------------------------------------------ renderer & scene
const container = $('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
container.appendChild(renderer.domElement);
const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.inset = '0';
labelRenderer.domElement.style.pointerEvents = 'none';
container.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
const camera = new THREE.PerspectiveCamera(60, 1, 0.02, 900);
camera.position.set(6, 6, 10);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI / 2.1;
controls.minDistance = 3;
controls.maxDistance = 120;
controls.enabled = false;
const sky = new Sky(scene, renderer);

const nodePos = (n) => new THREE.Vector3(n.x * S, 0, -n.y * S);
let city = null, life = null, rig = null, siteFx = [];
const minimap = new Minimap($('minimap'), S);

function addLabel(text, kind, pos, height) {
  const el = document.createElement('div');
  el.className = 'site-label';
  el.innerHTML = `<span class="tag ${kind}">${KIND_TAG[kind]}</span>${text}`;
  const obj = new CSS2DObject(el);
  obj.position.set(pos.x, height, pos.z);
  scene.add(obj);
}

function buildWorld(nodes) {
  city = buildCity(scene, nodes, nodePos);
  life = buildLife(scene, city);
  siteFx = city.siteFx;
  nodes.forEach((n) => addLabel(n.kind === 'depot' ? 'Medical Distribution Center' : n.name, n.kind, nodePos(n), n.kind === 'hospital' ? 3.2 : 1.9));
  rig = new CameraRig(camera, renderer.domElement, controls, city);
}

// Drone
const drone = createDrone(scene);
const cargo = drone.cargo;
const tether = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineBasicMaterial({ color: 0xf2f2f2 }));
tether.visible = false; scene.add(tether);
const dropBox = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.11, 0.14), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6 }));
dropBox.visible = false; dropBox.castShadow = true; scene.add(dropBox);

const routeMat = new THREE.LineDashedMaterial({ color: 0x3ee0b5, dashSize: 0.5, gapSize: 0.3, transparent: true, opacity: 0.9, toneMapped: false });
const route = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(1, 0, 0)]), routeMat);
route.visible = false; scene.add(route);
const TRAIL_N = 140;
const trailGeom = new THREE.BufferGeometry();
trailGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_N * 3), 3));
const trail = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 }));
trail.frustumCulled = false;
scene.add(trail);
let trailPts = [];

const bursts = [];
function burst(pos, color) {
  const m = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.32, 40), new THREE.MeshBasicMaterial({ color, transparent: true, side: THREE.DoubleSide, toneMapped: false }));
  m.rotation.x = -Math.PI / 2; m.position.set(pos.x, LOT_Y + 0.1, pos.z); scene.add(m); bursts.push({ m, age: 0 });
}

// ------------------------------------------------------------------ playback state
let ep = null, simT = 0, playing = false, speed = 1, lastFxT = 0, lastEventCount = -1;
const motion = { yaw: 0, roll: 0, pitch: 0, prev: new THREE.Vector3(), vel: new THREE.Vector3(), yawRate: 0 };

function droneTransform(st) {
  const nodes = ep.nodes;
  const restH = (id) => (id === 0 ? PAD_H : HOVER_H);
  if (st.flight) {
    const f = st.flight, a = nodePos(nodes[f.from]), b = nodePos(nodes[f.to]);
    if (f.from === 0) a.z += 0.6;
    if (f.to === 0) b.z += 0.6;
    const u = f.u * f.frac;
    const pos = a.clone().lerp(b, u);
    const climb = THREE.MathUtils.smoothstep(f.u, 0, 0.14), descend = 1 - THREE.MathUtils.smoothstep(f.u, 0.86, 1);
    let h = THREE.MathUtils.lerp(restH(f.from), CRUISE_H, climb);
    if (f.u > 0.5) h = THREE.MathUtils.lerp(restH(f.to), CRUISE_H, descend);
    if (f.crash && f.u >= 1) h = LOT_Y + 0.12;
    pos.y = h;
    const yaw = Math.atan2(b.x - a.x, b.z - a.z);
    return { pos, yaw, dest: f.crash ? null : f.to, from: f.from };
  }
  const loc = st.location ?? 0;
  const pos = nodePos(nodes[loc]); pos.y = restH(loc);
  if (loc === 0) pos.z += 0.6;
  return { pos, yaw: motion.yaw, dest: null };
}

function fmtClock(t) { const h = Math.floor(t / 60), m = Math.floor(t % 60); return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`; }
const siteName = (i) => ep.sites[i]?.name ?? '?';
const pretty = (s) => s.replaceAll('_', ' ');

function describeAction(step) {
  if (!step) return '—';
  const [verb, target] = step.action_name.split(':');
  return { load: `Load → ${target}`, fly: `Fly → ${target}`, hold: 'Hold 5 min', charge: 'Charge',
           return_to_depot: 'Return to depot' }[verb] ?? step.action_name;
}

function eventText(e) {
  switch (e.type) {
    case 'order_created': return [`New ${pretty(e.order.type)} request · ${siteName(e.site)} (P${e.order.urgency})`, 'info'];
    case 'order_rejected': return [`${siteName(e.site)} busy: ${pretty(e.package)} sent by road`, ''];
    case 'load': return [`Loaded ${pretty(e.package)} for ${siteName(e.site)}`, ''];
    case 'takeoff': return [`Take-off → ${ep.nodes[e.destination].name} (${e.weather}, est. ${e.expected_energy.toFixed(0)}% battery)`, ''];
    case 'arrive': return [`Arrived ${ep.nodes[e.node].name} · battery ${e.battery.toFixed(0)}%`, ''];
    case 'deliver': return [e.on_time ? `Delivered ${pretty(e.package)} to ${siteName(e.site)} on time`
                                      : `Delivered ${pretty(e.package)} to ${siteName(e.site)} ${e.lateness.toFixed(0)} min late`, e.on_time ? 'good' : 'bad'];
    case 'order_expired': return [`Deadline missed · ${siteName(e.site)}`, 'bad'];
    case 'order_late': return [`Running late · ${siteName(e.site)}`, 'bad'];
    case 'spoiled': return [`Cold chain broken · ${siteName(e.site)}`, 'bad'];
    case 'charge': return [`Charged ${e.battery_before.toFixed(0)}% → ${e.battery_after.toFixed(0)}%`, 'info'];
    case 'weather': return [`Weather now ${e.weather.toUpperCase()}`, e.weather === 'storm' ? 'bad' : 'info'];
    case 'crash': return [`MISSION FAILURE: ${e.reason} · ${e.packages_lost} package(s) lost`, 'bad'];
    case 'mission_end': return [e.at_depot ? 'Shift complete · drone recovered at depot' : 'Shift complete · drone NOT at depot', e.at_depot ? 'good' : 'bad'];
    case 'hold': return ['Holding position', ''];
    case 'invalid_action': return [`Invalid command rejected: ${e.name}`, 'bad'];
    default: return [e.type, ''];
  }
}

let toastTimer;
function toast(text, cls) {
  const t = $('toast'); t.textContent = text; t.className = `show ${cls}`;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.className = ''; }, 1800);
}

function fireEffects(fromT, toT) {
  if (toT <= fromT || toT - fromT > 30) return;  // seeks do not replay effects
  for (const e of ep.allEvents) {
    if (e.t <= fromT || e.t > toT) continue;
    if (e.type === 'deliver') { burst(siteFx[e.site + 1].p, e.on_time ? 0x46d39a : 0xffb224); toast(eventText(e)[0], e.on_time ? 'good' : 'bad'); }
    else if (e.type === 'order_expired' || e.type === 'spoiled') { burst(siteFx[e.site + 1].p, 0xff5d5d); toast(eventText(e)[0], 'bad'); }
    else if (e.type === 'crash') { toast('MISSION FAILURE — battery exhausted', 'bad'); }
    else if (e.type === 'order_created') { const fx = siteFx[e.site + 1]; if (fx) fx.popT = 0; }
    else if (e.type === 'weather') { toast(`Weather: ${e.weather.toUpperCase()}`, e.weather === 'storm' ? 'bad' : ''); }
  }
}

function updateScene(st, dt, now) {
  const tf = droneTransform(st);
  const g = drone.group;
  const onPad = !st.flight && (st.location ?? 0) === 0;
  g.position.copy(tf.pos);
  if (!st.crashed && !onPad) g.position.y += Math.sin(now * 0.004) * 0.03;

  // attitude: smooth yaw, bank into turns, pitch with speed (display only)
  let dyaw = tf.yaw - motion.yaw; dyaw = ((dyaw + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;
  const yawStep = dyaw * (1 - Math.exp(-dt * 5));
  motion.yaw += yawStep;
  motion.yawRate = THREE.MathUtils.lerp(motion.yawRate, dt > 0 ? yawStep / dt : 0, 0.2);
  const disp = g.position.clone().sub(motion.prev);
  if (disp.length() > 3 || dt <= 0) disp.set(0, 0, 0);            // seek / episode change
  motion.vel.lerp(disp.divideScalar(Math.max(dt, 1e-3)), 0.15);
  motion.prev.copy(g.position);
  const fwd = new THREE.Vector3(Math.sin(motion.yaw), 0, Math.cos(motion.yaw));
  const fspeed = motion.vel.dot(fwd);
  const gust = st.weather === 'storm' ? Math.sin(now * 0.011) * 0.05 + Math.sin(now * 0.023) * 0.03 : st.weather === 'windy' ? Math.sin(now * 0.008) * 0.025 : 0;
  motion.pitch = THREE.MathUtils.lerp(motion.pitch, THREE.MathUtils.clamp(fspeed * 0.045, -0.2, 0.28), 0.1);
  motion.roll = THREE.MathUtils.lerp(motion.roll, THREE.MathUtils.clamp(-motion.yawRate * 0.35, -0.45, 0.45) + gust, 0.1);
  g.rotation.set(motion.pitch, motion.yaw, motion.roll);
  if (st.crashed && (!st.flight || st.flight.u >= 1)) { g.rotation.z = 0.9; g.rotation.x = 0.3; }

  const flying = !!st.flight && st.flight.u < 1 && !st.crashed;
  const dropping = st.dropping && st.t - st.dropping.t0 < 1.0 && st.t >= st.dropping.t0;
  const spin = st.crashed ? 0 : flying ? 42 : onPad ? (st.charging ? 0 : 6) : 30;
  const groundY = city ? Math.max(LOT_Y, city.heightAt(g.position.x, g.position.z)) : 0;
  const altitude = g.position.y - groundY;
  const spotlight = dropping ? 1 : flying ? 0.35 + 0.65 * THREE.MathUtils.smoothstep(4.0 - altitude, 0, 2.5) : onPad ? 0 : 0.8;
  drone.update(dt, now, { spin, crashed: !!st.crashed, darkness: envState.darkness, spotlight, coneLength: Math.max(0.5, altitude) });

  // cargo boxes
  const key = st.payload.map((s) => st.orders[s]?.type ?? '').join(',');
  if (cargo.userData.key !== key) {
    cargo.clear();
    st.payload.forEach((s, k) => {
      const o = st.orders[s]; if (!o) return;
      const b = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.07, 0.1), new THREE.MeshStandardMaterial({ color: PKG_COLOR[o.type], roughness: 0.6 }));
      b.position.set((k - (st.payload.length - 1) / 2) * 0.11, 0, 0); b.castShadow = true; cargo.add(b);
    });
    cargo.userData.key = key;
  }
  // tether drop: winch line from the cargo bay to the drop pad
  if (dropping) {
    const u = (st.t - st.dropping.t0) / 1.0;
    const top = g.position.clone(); top.y -= 0.12;
    const padY = siteFx[st.dropping.site + 1]?.padY ?? LOT_Y;
    const bottom = top.clone(); bottom.y = THREE.MathUtils.lerp(top.y, padY + 0.06, Math.min(1, u * 1.6));
    bottom.x += Math.sin(now * 0.006) * 0.03 * (1 - Math.min(1, u * 1.6));
    tether.geometry.setFromPoints([top, bottom]); tether.visible = true;
    dropBox.position.copy(bottom); dropBox.material.color.setHex(PKG_COLOR[st.dropping.pkg]); dropBox.visible = true;
  } else { tether.visible = false; dropBox.visible = false; }

  // route + trail
  if (flying && !st.flight.crash) {
    const b = nodePos(ep.nodes[st.flight.to]); if (st.flight.to === 0) b.z += 0.6; b.y = LOT_Y + 0.05;
    const a = g.position.clone(); a.y = LOT_Y + 0.05;
    route.geometry.setFromPoints([a, b]); route.computeLineDistances(); route.visible = true;
  } else route.visible = false;
  const last = trailPts.at(-1);
  if (!last || last.distanceTo(g.position) > 0.25) { trailPts.push(g.position.clone()); if (trailPts.length > TRAIL_N) trailPts.shift(); }
  const arr = trailGeom.attributes.position.array;
  for (let k = 0; k < TRAIL_N; k++) { const p = trailPts[Math.min(k, trailPts.length - 1)] ?? g.position; arr[k * 3] = p.x; arr[k * 3 + 1] = p.y; arr[k * 3 + 2] = p.z; }
  trailGeom.attributes.position.needsUpdate = true;
  trailGeom.setDrawRange(0, trailPts.length);

  // site markers: urgency ring + halo + light beam for pending requests, spinning ring for cargo on board
  const destination = flying ? st.flight.to - 1 : -1;
  st.orders.forEach((o, s) => {
    const fx = siteFx[s + 1]; if (!fx) return;
    const pending = o && o.status === 'pending', onboard = o && o.status === 'onboard';
    fx.ring.visible = fx.halo.visible = fx.shaft.visible = !!pending || (onboard && s === destination);
    fx.onboard.visible = !!onboard;
    if (fx.ring.visible) {
      const color = pending ? URG_COLOR[o.urgency] : PKG_COLOR[o.type];
      for (const m of [fx.ring, fx.halo, fx.shaft]) m.material.color.setHex(color);
      const tight = o.deadline - st.t < 15;
      const pulse = tight ? 0.5 + 0.5 * Math.sin(now * 0.012) : 0.5 + 0.5 * Math.sin(now * 0.003 + s);
      fx.popT = Math.min(1, fx.popT + dt * 2);
      const pop = THREE.MathUtils.smoothstep(fx.popT, 0, 1);
      fx.ring.scale.setScalar((1 + 0.12 * pulse) * (0.3 + 0.7 * pop));
      fx.halo.material.opacity = 0.35 + 0.35 * pulse;
      fx.shaft.material.opacity = (s === destination ? 0.42 : 0.16 + 0.08 * pulse) * (0.7 + 0.6 * envState.darkness) * pop;
      fx.shaft.scale.set(1, pop, 1); fx.shaft.position.y = fx.padY + 9 * pop;
    }
    if (onboard) { fx.onboard.rotation.y += dt * 1.6; fx.onboard.material.color.setHex(PKG_COLOR[o.type]); }
  });
  const dfx = city.depotFx;
  dfx.chargeRing.material.opacity = st.charging ? 0.55 + 0.4 * Math.sin(now * 0.008) : 0;
  dfx.strip.color.setHex(st.charging ? (Math.floor(now / 300) % 2 ? 0x3ee0b5 : 0x9ffff0) : 0x1d6b58);
  for (const c of dfx.cables) {
    c.visible = !!st.charging;
    if (c.visible) {
      const a = c.userData.anchor, b = g.position.clone().add(new THREE.Vector3(Math.sign(a.x - g.position.x) * 0.12, 0.02, 0));
      const mid = a.clone().lerp(b, 0.5); mid.y -= 0.18;
      c.geometry.setFromPoints(new THREE.QuadraticBezierCurve3(a, mid, b).getPoints(12));
    }
  }

  for (let k = bursts.length - 1; k >= 0; k--) {
    const b = bursts[k]; b.age += dt; b.m.scale.setScalar(1 + b.age * 9); b.m.material.opacity = Math.max(0, 1 - b.age / 0.9);
    if (b.age > 0.9) { scene.remove(b.m); bursts.splice(k, 1); }
  }
  return { fwd, flying, dropping };
}

function updatePanel(st) {
  const nSteps = ep.steps.length;
  $('step').textContent = `Step ${Math.min(st.i + 1, nSteps)} of ${nSteps}`;
  $('clock').textContent = fmtClock(st.t);
  $('shift-bar').style.width = `${Math.min(100, (100 * st.t) / ep.doc.header.config.horizon_min)}%`;
  $('weather').textContent = st.weather;
  $('weather-note').textContent = WEATHER[st.weather].note;
  $('weather-card').className = `overlay weather ${st.weather}`;
  const b = Math.max(0, st.battery);
  $('battery').textContent = `${b.toFixed(0)}%`;
  $('battery-bar').style.strokeDashoffset = String(100 - Math.min(100, b));
  $('battery-bar').style.stroke = b < 20 ? 'var(--bad)' : b < 40 ? 'var(--warn)' : 'var(--good)';
  const tf = st.flight && st.flight.u < 1;
  $('location').textContent = tf ? 'In flight' : (st.crashed ? 'Crash site' : ep.nodes[st.location ?? 0].name);
  $('destination').textContent = tf ? ep.nodes[st.flight.to].name : (st.charging ? 'Charging on pad' : '—');
  if (st.crashed) { $('flight-status').textContent = 'Mission failure'; $('route').textContent = 'Battery exhausted — drone lost'; }
  else if (tf) { $('flight-status').textContent = `Flying to ${ep.nodes[st.flight.to].name}`; $('route').textContent = `From ${ep.nodes[st.flight.from].name} · ${(100 * st.flight.u).toFixed(0)}% of the leg`; }
  else if (st.charging) { $('flight-status').textContent = 'Charging'; $('route').textContent = 'On the depot charging pad'; }
  else if (st.dropping && st.t - st.dropping.t0 < 1) { $('flight-status').textContent = 'Lowering package'; $('route').textContent = `${pretty(st.dropping.pkg)} for ${siteName(st.dropping.site)}`; }
  else { $('flight-status').textContent = (st.location ?? 0) === 0 ? 'At the depot' : `At ${ep.nodes[st.location].name}`; $('route').textContent = st.t >= ep.end ? 'Shift complete' : (st.current ? `Next: ${describeAction(st.current)}` : ''); }

  const kg = st.payload.reduce((a, s) => a + (st.orders[s]?.weight ?? 0), 0);
  $('payload-kg').textContent = `· ${kg.toFixed(2)} of 2.5 kg · ${st.payload.length} of 3 slots`;
  $('payload').innerHTML = st.payload.length ? st.payload.map((s) => {
    const o = st.orders[s]; if (!o) return '';
    const lim = ep.coldLimit[o.type];
    const cold = lim ? `<span class="cold">❄ ${Math.max(0, lim - (st.t - (o.loadedAt ?? (o.coldAt - (lim - (o.cold_chain_left ?? lim)))))).toFixed(0)}m</span>` : '';
    return `<span class="pkg"><i style="background:${PKG_CSS[o.type]}"></i>${pretty(o.type)} → ${siteName(s)} ${cold}</span>`;
  }).join('') : '<span class="empty">empty</span>';

  const s = st.stats;
  $('k-ontime').textContent = s.delivered_on_time;
  $('k-late').textContent = s.delivered_late;
  $('k-missed').textContent = s.expired + s.spoiled + (s.lost_in_crash || 0);
  $('k-critical').textContent = `${s.critical_on_time}/${s.critical_requests}`;
  $('reward-total').textContent = (st.cumulative ?? 0).toFixed(1);
  $('last-action').textContent = describeAction(st.last);
  const pol = st.last?.policy;
  const conf = pol?.kind === 'probability' ? ` · confidence ${(100 * pol.value).toFixed(0)}%`
    : pol?.kind === 'q_values' ? ` · Q ${pol.value.toFixed(1)}` : '';
  $('last-reward').textContent = st.last ? `Reward ${st.last.reward >= 0 ? '+' : ''}${st.last.reward.toFixed(2)}${conf}` : '\u00a0';
  $('episode-state').textContent = st.t >= ep.end ? (ep.doc.summary.crashed ? 'Failed' : 'Complete') : 'In progress';

  const active = st.orders.map((o, i) => (o ? { ...o, site: i } : null)).filter(Boolean)
    .sort((a, b) => (b.status === 'onboard') - (a.status === 'onboard') || b.urgency - a.urgency || a.deadline - b.deadline);
  $('pending-count').textContent = st.t >= ep.end ? `${active.length} unserved at shift end` : `${active.length} active`;
  $('orders').innerHTML = active.length ? active.map((o) => {
    const due = o.deadline - st.t;
    return `<div class="order u${o.urgency} ${o.status}"><span class="icon">${KIND_ICON[ep.sites[o.site].kind]}</span>
      <div><b>${siteName(o.site)}</b><small><i style="color:${PKG_CSS[o.type]}">●</i> ${pretty(o.type)} · ${o.weight.toFixed(2)} kg</small></div>
      <div class="due ${due < 15 ? 'tight' : ''}">${due >= 0 ? `${due.toFixed(0)} min` : 'Late'}<small>${o.status === 'onboard' ? 'On board' : `P${o.urgency}`}</small></div></div>`;
  }).join('') : '<p class="empty">No active requests</p>';

  const seen = ep.allEvents.filter((e) => e.t <= st.t && !['hold'].includes(e.type));
  if (seen.length !== lastEventCount) {
    lastEventCount = seen.length;
    $('last-event').textContent = seen.length ? eventText(seen.at(-1))[0] : '\u00a0';
    $('events').innerHTML = seen.slice(-40).reverse().map((e) => {
      const [txt, cls] = eventText(e);
      return `<li class="${cls}"><time>${fmtClock(e.t)}</time><span>${txt}</span></li>`;
    }).join('');
  }
  $('seek').value = String(Math.round((1000 * st.t) / Math.max(1e-6, ep.end)));
  $('progress-label').textContent = `${st.t.toFixed(1)} / ${ep.end.toFixed(1)} min`;
  renderDeliveries(st);
}

// On-time deliveries per 30-minute block (solid = played, hatched = still to come).
let chartKey = '';
function renderDeliveries(st) {
  const H = ep.doc.header.config.horizon_min, BIN = 30, n = Math.ceil(H / BIN);
  const all = Array(n).fill(0);
  for (const e of ep.allEvents) if (e.type === 'deliver' && e.on_time) all[Math.min(n - 1, Math.floor(e.t / BIN))] += 1;
  const cur = Math.min(n - 1, Math.floor(st.t / BIN));
  const shown = all.map((c, i) => (i < cur ? c : i === cur ? ep.allEvents.filter((e) => e.type === 'deliver' && e.on_time && e.t <= st.t && Math.floor(e.t / BIN) === i).length : 0));
  const key = `${cur}|${shown.join(',')}|${st.t >= ep.end}`;
  if (key === chartKey) return; chartKey = key;
  const max = Math.max(1, ...all), done = st.t >= ep.end;
  $('hourly').innerHTML = all.map((c, i) => {
    const future = !done && i > cur, current = !done && i === cur;
    const v = future ? c : shown[i];
    const h = 18 + 82 * (future ? 0.55 : v / max);
    const label = `${Math.floor((i * BIN) / 60)}:${String((i * BIN) % 60).padStart(2, '0')}`;
    return `<div class="pill ${future ? 'future' : ''} ${current ? 'current' : ''}" title="${label}–${Math.floor(((i + 1) * BIN) / 60)}:${String(((i + 1) * BIN) % 60).padStart(2, '0')}: ${future ? 'not played yet' : `${v} on time`}">
      ${current ? `<em>${v}</em>` : ''}<i style="height:${h}%"></i><span>${label}</span></div>`;
  }).join('');
  $('hourly-total').textContent = `${shown.reduce((a, b) => a + b, 0)} on time · per 30 min`;
}

// ------------------------------------------------------------------ loading
let sceneBuilt = false;
async function loadEpisode(file) {
  const doc = await new ReplaySource(`episodes/${file}`).load();
  ep = new Episode(doc);
  if (!sceneBuilt) { buildWorld(ep.nodes); sceneBuilt = true; }
  const md = doc.header.metadata;
  $('policy-label').textContent = md.label || md.policy;
  $('episode-meta').textContent = `Seed ${md.seed} · ${ep.steps.length} decisions · final return ${doc.summary.return.toFixed(1)} · ` +
    `${doc.summary.delivered_on_time} on time of ${doc.summary.requests_total} requests${doc.summary.crashed ? ' · CRASHED' : ''}`;
  $('policy-kind').textContent = md.kind === 'agent' ? 'Trained agent' : 'Baseline';
  $('policy-kind').className = `badge ${md.kind === 'agent' ? 'agent' : ''}`;
  chartKey = '';
  trailPts = []; minimap.resetTrail(); lastEventCount = -1; simT = 0; lastFxT = 0; setPlaying(false);
  const st = ep.stateAt(0);
  updatePanel(st);
  const startT = Number(new URLSearchParams(location.search).get('t'));
  if (startT > 0 && file === new URLSearchParams(location.search).get('file')) { simT = Math.min(startT, ep.end); lastFxT = simT; }
  else history.replaceState(null, '', `?file=${encodeURIComponent(file)}`);
  // place the drone and camera at the start position instead of flying in from the origin
  const st0 = ep.stateAt(simT), tf = droneTransform(st0);
  if (!st0.flight) tf.yaw = Math.PI;             // parked: nose towards the hangar, camera over the pad
  motion.yaw = tf.yaw; motion.prev.copy(tf.pos); drone.group.position.copy(tf.pos);
  sky.snap(st0.weather);
  if (firstLoad) {
    firstLoad = false;
    const want = new URLSearchParams(location.search).get('cam');
    rig.setMode(MODES.includes(want) ? want : 'chase', { dronePos: tf.pos });
    if (rig.mode === 'orbit') camera.position.copy(rig.orbitTo);
    else if (rig.mode === 'chase') camera.position.copy(tf.pos).add(new THREE.Vector3(-Math.sin(tf.yaw) * 2, 0.8, -Math.cos(tf.yaw) * 2));
    rig.look.copy(tf.pos);
    rig.blend = rig.mode === 'orbit' ? 1 : 0.6;
    updateCamUI();
  }
}
let firstLoad = true;

async function init() {
  let index;
  try {
    const r = await fetch('episodes/index.json', { cache: 'no-store' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    index = await r.json();
  } catch (err) {
    $('policy-label').textContent = 'No replays found';
    $('episode-meta').textContent = 'Run: uv run python -m visualization.export_events';
    throw err;
  }
  const sel = $('episode');
  sel.innerHTML = index.episodes.map((e) =>
    `<option value="${e.file}">${e.kind === 'agent' ? '★ ' : ''}${e.label} · seed ${e.seed} · return ${e.return.toFixed(1)}</option>`).join('');
  const wanted = new URLSearchParams(location.search).get('file');
  const first = index.episodes.find((e) => e.file === wanted) ? wanted : index.episodes[0].file;
  sel.value = first;
  sel.onchange = () => loadEpisode(sel.value);
  await loadEpisode(first);
}

// ------------------------------------------------------------------ controls
function setPlaying(p) { playing = p; $('play').textContent = p ? '❚❚' : '▶'; $('play').setAttribute('aria-label', p ? 'Pause' : 'Play'); }
$('play').onclick = () => { if (ep && simT >= ep.end) { simT = 0; lastFxT = 0; trailPts = []; } setPlaying(!playing); };
$('reset').onclick = () => { simT = 0; lastFxT = 0; trailPts = []; minimap.resetTrail(); setPlaying(false); };
$('next').onclick = () => { setPlaying(false); const i = ep.stepIndexAt(simT); simT = i < ep.steps.length ? ep.steps[i].t_end : ep.end; lastFxT = simT; };
$('prev').onclick = () => { setPlaying(false); const i = ep.stepIndexAt(simT - 1e-6); simT = i > 0 ? ep.steps[i - 1].t_start : 0; lastFxT = simT; };
$('speed').onchange = (e) => { speed = Number(e.target.value); };
$('seek').oninput = (e) => { setPlaying(false); simT = (Number(e.target.value) / 1000) * ep.end; lastFxT = simT; trailPts = []; minimap.resetTrail(); };
$('labels').onchange = (e) => { document.body.classList.toggle('hide-labels', !e.target.checked); };
document.body.classList.toggle('hide-labels', !$('labels').checked);

// Light / dark theme (per-viewer preference; the 3D scene is unaffected).
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  $('theme').setAttribute('aria-label', t === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
  try { localStorage.setItem('mediflight-theme', t); } catch { /* storage unavailable */ }
}
{
  let saved = new URLSearchParams(location.search).get('theme');
  try { saved = saved || localStorage.getItem('mediflight-theme'); } catch { /* ignore */ }
  applyTheme(saved === 'dark' || saved === 'light' ? saved : (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
}
$('theme').onclick = () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
function updateCamUI() {
  $('cam').textContent = `${MODE_LABEL[rig.mode]}  ·  C`;
  $('cam-mode').textContent = MODE_LABEL[rig.mode];
  $('cam-hint').textContent = MODE_HINT[rig.mode];
  document.body.dataset.cam = rig.mode;
}
const camCtx = () => ({ dronePos: drone.group.position.clone() });
$('cam').onclick = () => { if (rig) { rig.next(camCtx()); updateCamUI(); } };
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT' && e.target.type !== 'checkbox' && e.target.type !== 'range') return;
  if (e.code === 'Space') { e.preventDefault(); $('play').click(); }
  if (e.code === 'ArrowRight') $('next').click();
  if (e.code === 'ArrowLeft') $('prev').click();
  if (e.code === 'KeyC' && rig) { rig.next(camCtx()); updateCamUI(); }
});

function resize() {
  const w = container.clientWidth, h = container.clientHeight;
  camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h); labelRenderer.setSize(w, h);
}
new ResizeObserver(resize).observe(container);
resize();

// ------------------------------------------------------------------ frame loop
let envState = { darkness: 0, wind: 0.15, wet: 0, rain: 0 };
let prevNow = performance.now(), fpsAcc = 0, fpsN = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (now - prevNow) / 1000); prevNow = now;
  if (ep && city) {
    if (playing) {
      simT = Math.min(ep.end, simT + dt * speed * SIM_MIN_PER_SEC);
      fireEffects(lastFxT, simT); lastFxT = simT;
      if (simT >= ep.end) setPlaying(false);
    }
    const st = ep.stateAt(simT);
    const view = updateScene(st, dt, now);
    rig.update(dt, { dronePos: drone.group.position, yaw: motion.yaw, roll: motion.roll, t: now / 1000,
      follow: $('follow').checked, flying: view.flying, dropping: view.dropping });
    const [focus, extent] = rig.shadowFocus(drone.group.position);
    sky.setShadowExtent(extent);
    envState = sky.update(dt, now, st.weather, camera, focus);
    city.setWeather(envState, now);
    life.update(dt, now / 1000, envState);
    updatePanel(st);
    const camDir = new THREE.Vector3(); camera.getWorldDirection(camDir);
    minimap.draw({ ep, st, dronePos: drone.group.position, droneYaw: motion.yaw, camPos: camera.position, camDir, now });
  }
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) { $('fps').textContent = `${Math.round(fpsN / fpsAcc)} FPS`; fpsAcc = 0; fpsN = 0; }
}
requestAnimationFrame(frame);
init();
