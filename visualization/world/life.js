// Ambient city life: swaying trees, street lamps, traffic signals, cars and
// pedestrians. Visual only — it runs on its own wall-clock time and never reads
// or changes the replay state. Every group is a handful of InstancedMeshes, so
// hundreds of agents cost only a few draw calls.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { CURB, LINES, ROAD_EXT, ROAD_HALF, RING, CENTERS } from './city.js';
import { glowTex, mulberry32 } from './textures.js';

const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _e = new THREE.Euler(), _p = new THREE.Vector3(), _s = new THREE.Vector3();

function colored(geo, hex) {        // bake a vertex colour so merged parts keep their own tint
  const c = new THREE.Color(hex), n = geo.attributes.position.count, arr = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { arr[i * 3] = c.r; arr[i * 3 + 1] = c.g; arr[i * 3 + 2] = c.b; }
  geo.setAttribute('color', new THREE.BufferAttribute(arr, 3));
  return geo;
}
const part = (geo, hex, x, y, z) => colored(geo.translate(x, y, z), hex);

export function buildLife(scene, city) {
  const rng = mulberry32(4242);
  const glow = glowTex();
  const additive = (color, size, opacity = 1) => new THREE.PointsMaterial({ color, size, map: glow, transparent: true, opacity,
    depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true, toneMapped: false });

  // =============================================================== trees
  const trees = city.treeSpots.map((t) => ({ ...t, yaw: rng() * 6.28, phase: rng() * 6.28, tint: 0.8 + rng() * 0.35 }));
  const conifers = trees.filter((t) => t.kind === 'conifer'), broad = trees.filter((t) => t.kind !== 'conifer');
  const trunk = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.022, 0.034, 0.34, 6).translate(0, 0.17, 0),
    new THREE.MeshStandardMaterial({ color: 0x6b4f35, roughness: 1 }), trees.length);
  const broadGeo = new THREE.IcosahedronGeometry(0.24, 1).scale(1, 1.1, 1).translate(0, 0.5, 0);
  const crownB = new THREE.InstancedMesh(broadGeo, new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, flatShading: true }), broad.length);
  const crownC = new THREE.InstancedMesh(new THREE.ConeGeometry(0.22, 0.7, 8).translate(0, 0.6, 0),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, flatShading: true }), conifers.length);
  broad.forEach((t, i) => crownB.setColorAt(i, new THREE.Color(0x4f8a3f).multiplyScalar(t.tint).offsetHSL((rng() - 0.5) * 0.04, 0, 0)));
  conifers.forEach((t, i) => crownC.setColorAt(i, new THREE.Color(0x356b3c).multiplyScalar(t.tint)));
  for (const m of [trunk, crownB, crownC]) { m.castShadow = true; m.receiveShadow = true; scene.add(m); }

  function updateTrees(t, env) {
    const amp = 0.015 + env.wind * 0.12;
    let bi = 0, ci = 0;
    trees.forEach((tr, i) => {
      const sway = amp * (Math.sin(t * (1.3 + env.wind) + tr.phase) * 0.7 + Math.sin(t * 2.9 + tr.phase * 1.7) * 0.3);
      _e.set(sway * 0.35, tr.yaw, -Math.abs(sway) - env.wind * 0.04, 'XYZ');
      _q.setFromEuler(_e); _p.set(tr.x, tr.kind === 'street' ? CURB : 0.05, tr.z); _s.setScalar(tr.s);
      _m.compose(_p, _q, _s);
      trunk.setMatrixAt(i, _m);
      if (tr.kind === 'conifer') crownC.setMatrixAt(ci++, _m); else crownB.setMatrixAt(bi++, _m);
    });
    trunk.instanceMatrix.needsUpdate = crownB.instanceMatrix.needsUpdate = crownC.instanceMatrix.needsUpdate = true;
  }

  // =============================================================== street lamps
  const lamps = city.lampSpots;
  const poleGeo = mergeGeometries([
    new THREE.CylinderGeometry(0.012, 0.016, 0.62, 6).translate(0, 0.31, 0),
    new THREE.BoxGeometry(0.24, 0.012, 0.012).translate(0.12, 0.61, 0),
    new THREE.BoxGeometry(0.08, 0.025, 0.045).translate(0.23, 0.6, 0),
  ]);
  const poles = new THREE.InstancedMesh(poleGeo, new THREE.MeshStandardMaterial({ color: 0x3a4248, roughness: 0.5, metalness: 0.6 }), lamps.length);
  const lensMat = new THREE.MeshBasicMaterial({ color: 0xffe2a8, toneMapped: false });
  const lenses = new THREE.InstancedMesh(new THREE.BoxGeometry(0.07, 0.006, 0.035).translate(0.23, 0.585, 0), lensMat, lamps.length);
  const lampGlowPos = new Float32Array(lamps.length * 3);
  const poolGeo = new THREE.PlaneGeometry(1.1, 1.1).rotateX(-Math.PI / 2);
  const poolMat = new THREE.MeshBasicMaterial({ color: 0xffd28a, map: glow, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false });
  const pools = new THREE.InstancedMesh(poolGeo, poolMat, lamps.length);
  lamps.forEach((l, i) => {
    const yaw = Math.atan2(-l.dz, l.dx);
    _m.compose(_p.set(l.x, CURB, l.z), _q.setFromEuler(_e.set(0, yaw, 0)), _s.setScalar(1));
    poles.setMatrixAt(i, _m); lenses.setMatrixAt(i, _m);
    const hx = l.x + l.dx * 0.23, hz = l.z + l.dz * 0.23;
    lampGlowPos.set([hx, CURB + 0.58, hz], i * 3);
    pools.setMatrixAt(i, _m.makeTranslation(hx, 0.02, hz));
  });
  poles.castShadow = true;
  const lampGlowGeo = new THREE.BufferGeometry(); lampGlowGeo.setAttribute('position', new THREE.BufferAttribute(lampGlowPos, 3));
  const lampGlow = new THREE.Points(lampGlowGeo, additive(0xffd9a0, 0.7, 0));
  pools.renderOrder = 1;
  scene.add(poles, lenses, lampGlow, pools);

  // =============================================================== traffic signals
  const PERIOD = 10;
  const intersections = [];
  LINES.forEach((a, i) => LINES.forEach((b, j) => intersections.push({ x: a, z: b, off: ((i + j) * 1.3) % PERIOD })));
  const phase = (inter, axis, t) => {          // 0 green, 1 amber, 2 red for traffic moving along `axis`
    const u = (t + inter.off) % PERIOD;
    if (axis === 'x') return u < 4.2 ? 0 : u < 5 ? 1 : 2;
    return u >= 5 && u < 9.2 ? 0 : u >= 9.2 ? 1 : 2;
  };
  const heads = [];                            // one signal head per corner, facing the traffic that stops there
  for (const it of intersections) {
    heads.push({ it, axis: 'x', x: it.x - 0.66, z: it.z + 0.66, yaw: -Math.PI / 2 });   // faces traffic arriving from -x
    heads.push({ it, axis: 'x', x: it.x + 0.66, z: it.z - 0.66, yaw: Math.PI / 2 });
    heads.push({ it, axis: 'z', x: it.x - 0.66, z: it.z - 0.66, yaw: Math.PI });
    heads.push({ it, axis: 'z', x: it.x + 0.66, z: it.z + 0.66, yaw: 0 });
  }
  const sigPole = new THREE.InstancedMesh(mergeGeometries([
    new THREE.CylinderGeometry(0.01, 0.012, 0.42, 6).translate(0, 0.21, 0),
    new THREE.BoxGeometry(0.045, 0.12, 0.035).translate(0, 0.42, 0),
  ]), new THREE.MeshStandardMaterial({ color: 0x2a3034, roughness: 0.6, metalness: 0.4 }), heads.length);
  const bulbGeo = new THREE.SphereGeometry(0.013, 8, 6);
  const bulbs = [0, 1, 2].map(() => new THREE.InstancedMesh(bulbGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false }), heads.length));
  const BULB_ON = [new THREE.Color(0x35ff8a), new THREE.Color(0xffb020), new THREE.Color(0xff3030)];
  const BULB_OFF = new THREE.Color(0x1c2226);
  heads.forEach((h, i) => {
    _m.compose(_p.set(h.x, CURB, h.z), _q.setFromEuler(_e.set(0, h.yaw, 0)), _s.setScalar(1));
    sigPole.setMatrixAt(i, _m);
    [0.455, 0.42, 0.385].forEach((y, k) => {  // red top, amber middle, green bottom
      const lm = new THREE.Matrix4().compose(_p.set(0, y, 0.019), new THREE.Quaternion(), _s.setScalar(1));
      bulbs[k].setMatrixAt(i, _m.clone().multiply(lm));
      bulbs[k].setColorAt(i, BULB_OFF);
    });
  });
  scene.add(sigPole, ...bulbs);
  let lastSig = -1;
  function updateSignals(t) {
    const tick = Math.floor(t * 4);
    if (tick === lastSig) return; lastSig = tick;
    heads.forEach((h, i) => {
      const s = phase(h.it, h.axis, t);
      bulbs[0].setColorAt(i, s === 2 ? BULB_ON[2] : BULB_OFF);
      bulbs[1].setColorAt(i, s === 1 ? BULB_ON[1] : BULB_OFF);
      bulbs[2].setColorAt(i, s === 0 ? BULB_ON[0] : BULB_OFF);
    });
    bulbs.forEach((b) => { b.instanceColor.needsUpdate = true; });
  }
  // =============================================================== cars
  const LANE = 0.23;
  const carGeo = mergeGeometries([
    part(new THREE.BoxGeometry(0.26, 0.1, 0.62), 0xffffff, 0, 0.09, 0),
    part(new THREE.BoxGeometry(0.23, 0.085, 0.32), 0x1a2228, 0, 0.18, -0.03),
    ...[[-1, -1], [-1, 1], [1, -1], [1, 1]].map(([sx, sz]) => part(new THREE.CylinderGeometry(0.045, 0.045, 0.04, 10).rotateZ(Math.PI / 2), 0x111111, sx * 0.12, 0.045, sz * 0.2)),
  ]);
  const vanGeo = mergeGeometries([
    part(new THREE.BoxGeometry(0.28, 0.2, 0.68), 0xffffff, 0, 0.14, -0.02),
    part(new THREE.BoxGeometry(0.26, 0.09, 0.12), 0x1a2228, 0, 0.18, 0.29),
    part(new THREE.BoxGeometry(0.285, 0.035, 0.5), 0xd8282f, 0, 0.13, -0.06),
    ...[[-1, -1], [-1, 1], [1, -1], [1, 1]].map(([sx, sz]) => part(new THREE.CylinderGeometry(0.045, 0.045, 0.04, 10).rotateZ(Math.PI / 2), 0x111111, sx * 0.13, 0.045, sz * 0.22)),
  ]);
  const paints = [0xf2f2f0, 0xc9ced2, 0x23272b, 0x4a5157, 0x2f5f9e, 0xa8282c, 0xd8c9a8, 0x2e6b4f, 0xf2c230];
  const lanes = [];
  for (const axis of ['x', 'z']) for (const line of LINES) for (const dir of [1, -1]) lanes.push({ axis, line, dir, cars: [] });
  const cars = [];
  lanes.forEach((lane) => {
    const n = 2 + Math.floor(rng() * 3);
    for (let k = 0; k < n; k++) {
      const car = { lane, s: -ROAD_EXT + rng() * 2 * ROAD_EXT, v: 0, vmax: 0.9 + rng() * 0.5, amb: false, paint: paints[Math.floor(rng() * paints.length)] };
      lane.cars.push(car); cars.push(car);
    }
  });
  for (let k = 0; k < 4; k++) { const c = cars[Math.floor(rng() * cars.length)]; c.amb = true; c.vmax = 1.9; c.paint = 0xffffff; }
  const sedans = cars.filter((c) => !c.amb), vans = cars.filter((c) => c.amb);
  const carMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.35, metalness: 0.45 });
  const sedanMesh = new THREE.InstancedMesh(carGeo, carMat, sedans.length);
  const vanMesh = new THREE.InstancedMesh(vanGeo, carMat, vans.length);
  sedans.forEach((c, i) => sedanMesh.setColorAt(i, new THREE.Color(c.paint)));
  vans.forEach((c, i) => vanMesh.setColorAt(i, new THREE.Color(c.paint)));
  sedanMesh.castShadow = vanMesh.castShadow = true;
  const headGeo = mergeGeometries([new THREE.BoxGeometry(0.05, 0.025, 0.01).translate(-0.085, 0.1, 0.311), new THREE.BoxGeometry(0.05, 0.025, 0.01).translate(0.085, 0.1, 0.311)]);
  const tailGeo = mergeGeometries([new THREE.BoxGeometry(0.05, 0.022, 0.01).translate(-0.09, 0.11, -0.311), new THREE.BoxGeometry(0.05, 0.022, 0.01).translate(0.09, 0.11, -0.311)]);
  const headMat = new THREE.MeshBasicMaterial({ color: 0xfff4d6, toneMapped: false });
  const tailMat = new THREE.MeshBasicMaterial({ color: 0xff2a2a, toneMapped: false });
  const heads3 = new THREE.InstancedMesh(headGeo, headMat, cars.length), tails = new THREE.InstancedMesh(tailGeo, tailMat, cars.length);
  const barMesh = new THREE.InstancedMesh(mergeGeometries([
    colored(new THREE.BoxGeometry(0.1, 0.025, 0.035).translate(-0.05, 0.255, 0.18), 0xffffff),
    colored(new THREE.BoxGeometry(0.1, 0.025, 0.035).translate(0.05, 0.255, 0.18), 0xffffff)]),
    new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: false }), vans.length);
  vans.forEach((_, i) => barMesh.setColorAt(i, new THREE.Color(0xff2020)));
  const headGlowPos = new Float32Array(cars.length * 3), tailGlowPos = new Float32Array(cars.length * 3), sirenPos = new Float32Array(vans.length * 3);
  const mkPoints = (arr, mat) => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(arr, 3)); return new THREE.Points(g, mat); };
  const headGlow = mkPoints(headGlowPos, additive(0xfff1cc, 0.5, 0)), tailGlow = mkPoints(tailGlowPos, additive(0xff3030, 0.3, 0));
  const sirenGlow = mkPoints(sirenPos, additive(0xff3030, 0.9, 0.9));
  scene.add(sedanMesh, vanMesh, heads3, tails, barMesh, headGlow, tailGlow, sirenGlow);

  const STOP = 1.16;          // car centre stops this far before an intersection centre
  function updateCars(dt, t, env) {
    for (const lane of lanes) {
      lane.cars.sort((a, b) => (b.s - a.s) * lane.dir);      // front-most first
      lane.cars.forEach((c, k) => {
        let target = c.vmax * (env.wet > 0.5 ? 0.8 : 1);
        if (!c.amb) {
          for (const L of LINES) {
            const dist = (L - lane.dir * STOP - c.s) * lane.dir;
            if (dist < -0.05 || dist > 2.6) continue;
            const inter = { off: ((LINES.indexOf(lane.axis === 'x' ? L : lane.line) + LINES.indexOf(lane.axis === 'x' ? lane.line : L)) * 1.3) % PERIOD };
            const s = phase(inter, lane.axis, t);
            if (s === 2 || (s === 1 && dist > 0.4)) target = Math.min(target, Math.max(0, dist) * 1.5);
            break;
          }
        }
        const lead = lane.cars[k - 1];
        if (lead) {
          const gap = (lead.s - c.s) * lane.dir - 0.72;
          if (gap >= 0 && gap < 1.4) target = Math.min(target, Math.max(0, gap - 0.1) * 1.6);
        }
        c.v += THREE.MathUtils.clamp(target - c.v, -2.6 * dt, 0.9 * dt);
        c.s += lane.dir * c.v * dt;
        if (c.s * lane.dir > ROAD_EXT) c.s = -lane.dir * ROAD_EXT;
      });
    }
    let si = 0, vi = 0;
    cars.forEach((c, i) => {
      const { axis, line, dir } = c.lane;
      let x, z, yaw;
      if (axis === 'x') { x = c.s; z = line + dir * LANE; yaw = dir > 0 ? Math.PI / 2 : -Math.PI / 2; }
      else { z = c.s; x = line - dir * LANE; yaw = dir > 0 ? 0 : Math.PI; }
      _m.compose(_p.set(x, 0.012, z), _q.setFromEuler(_e.set(0, yaw, 0)), _s.setScalar(1));
      if (c.amb) { vanMesh.setMatrixAt(vi, _m); barMesh.setMatrixAt(vi, _m); } else sedanMesh.setMatrixAt(si++, _m);
      heads3.setMatrixAt(i, _m); tails.setMatrixAt(i, _m);
      const fx = Math.sin(yaw), fz = Math.cos(yaw);
      headGlowPos.set([x + fx * 0.36, 0.1, z + fz * 0.36], i * 3);
      tailGlowPos.set([x - fx * 0.33, 0.11, z - fz * 0.33], i * 3);
      if (c.amb) {
        const red = Math.floor(t * 6 + vi) % 2 === 0;
        barMesh.setColorAt(vi, red ? new THREE.Color(0xff2020) : new THREE.Color(0x2050ff));
        sirenPos.set([x + fx * 0.18, 0.28, z + fz * 0.18], vi * 3);
        vi++;
      }
    });
    sirenGlow.material.color.setHex(Math.floor(t * 6) % 2 ? 0x2050ff : 0xff2020);
    for (const m of [sedanMesh, vanMesh, heads3, tails, barMesh]) m.instanceMatrix.needsUpdate = true;
    barMesh.instanceColor.needsUpdate = true;
    headGlow.geometry.attributes.position.needsUpdate = tailGlow.geometry.attributes.position.needsUpdate = true;
    sirenGlow.geometry.attributes.position.needsUpdate = true;
  }

  // =============================================================== pedestrians
  const peds = [];
  for (let k = 0; k < 170; k++) {
    const cx = CENTERS[Math.floor(rng() * CENTERS.length)], cz = CENTERS[Math.floor(rng() * CENTERS.length)];
    if (Math.hypot(cx, cz) > 22 && rng() < 0.6) continue;
    peds.push({ cx, cz, a: RING + (rng() - 0.5) * 0.18, s: rng() * 8 * RING, dir: rng() < 0.5 ? 1 : -1, speed: 0.16 + rng() * 0.12, ph: rng() * 6.28 });
  }
  const shirts = [0x2f5f9e, 0xa8282c, 0xe2e2dc, 0x2e6b4f, 0xd89a2b, 0x5a4a8a, 0x3a3f44, 0xc96f8a];
  const skins = [0xf1c9a5, 0xd9a07a, 0xa86f4c, 0x6b4430, 0xe8b894];
  const torso = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.03, 0.038, 0.11, 8).translate(0, 0.155, 0),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 }), peds.length);
  const head = new THREE.InstancedMesh(new THREE.SphereGeometry(0.024, 10, 8).translate(0, 0.235, 0),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8 }), peds.length);
  const legs = new THREE.InstancedMesh(new THREE.BoxGeometry(0.022, 0.1, 0.024).translate(0, -0.05, 0),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 }), peds.length * 2);
  const umbrellas = new THREE.InstancedMesh(mergeGeometries([
    colored(new THREE.ConeGeometry(0.1, 0.045, 10).translate(0, 0.33, 0), 0xffffff),
    colored(new THREE.CylinderGeometry(0.003, 0.003, 0.12, 4).translate(0, 0.27, 0), 0x222222)]),
    new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6 }), peds.length);
  peds.forEach((p, i) => {
    torso.setColorAt(i, new THREE.Color(shirts[Math.floor(rng() * shirts.length)]));
    head.setColorAt(i, new THREE.Color(skins[Math.floor(rng() * skins.length)]));
    const trousers = new THREE.Color([0x2a3340, 0x3d3d3d, 0x5a4a3a, 0x1f2a44][Math.floor(rng() * 4)]);
    legs.setColorAt(i * 2, trousers); legs.setColorAt(i * 2 + 1, trousers);
    umbrellas.setColorAt(i, new THREE.Color([0x1b1f24, 0x2f5f9e, 0xa8282c, 0x2e6b4f, 0xf2c230][Math.floor(rng() * 5)]));
  });
  scene.add(torso, head, legs, umbrellas);
  const _leg = new THREE.Matrix4();
  function updatePeds(dt, t, env) {
    const hurry = 1 + env.rain * 0.6;
    peds.forEach((p, i) => {
      p.s = (p.s + p.dir * p.speed * hurry * dt + 8 * p.a * 100) % (8 * p.a);
      const a = p.a, side = Math.floor(p.s / (2 * a)), u = (p.s % (2 * a)) - a;
      let x, z, hx, hz;
      if (side === 0) { x = u; z = -a; hx = 1; hz = 0; }
      else if (side === 1) { x = a; z = u; hx = 0; hz = 1; }
      else if (side === 2) { x = -u; z = a; hx = -1; hz = 0; }
      else { x = -a; z = -u; hx = 0; hz = -1; }
      const yaw = Math.atan2(hx * p.dir, hz * p.dir);
      p.ph += dt * p.speed * hurry * 55;
      const bob = Math.abs(Math.sin(p.ph)) * 0.006;
      const qYaw = new THREE.Quaternion().setFromEuler(_e.set(0, yaw, 0));
      _m.compose(_p.set(p.cx + x, CURB + bob, p.cz + z), qYaw, _s.setScalar(1));
      torso.setMatrixAt(i, _m); head.setMatrixAt(i, _m);
      const sw = Math.sin(p.ph) * 0.55;
      for (const [k, side2] of [[0, -1], [1, 1]]) {
        _leg.compose(_p.set(side2 * 0.016, 0.1, 0), _q.setFromEuler(_e.set(sw * side2, 0, 0)), _s.setScalar(1));
        legs.setMatrixAt(i * 2 + k, _m.clone().multiply(_leg));
      }
      _s.setScalar(Math.max(0.0001, env.rain));
      umbrellas.setMatrixAt(i, new THREE.Matrix4().compose(_p.set(p.cx + x, CURB + bob, p.cz + z), qYaw, _s));
    });
    for (const m of [torso, head, legs, umbrellas]) m.instanceMatrix.needsUpdate = true;
  }

  // =============================================================== frame update
  function update(dt, t, env) {
    updateTrees(t, env);
    updateSignals(t);
    updateCars(dt, t, env);
    updatePeds(dt, t, env);
    const dark = env.darkness;
    lampGlow.material.opacity = 0.12 + 0.88 * dark;
    lampGlow.material.size = 0.5 + 0.6 * dark;
    lensMat.color.setScalar(0.55 + 0.45 * dark).multiply(new THREE.Color(0xffe2a8));
    poolMat.opacity = 0.05 + 0.55 * dark;
    headGlow.material.opacity = 0.15 + 0.85 * dark;
    tailGlow.material.opacity = 0.2 + 0.8 * dark;
  }
  return { update };
}
