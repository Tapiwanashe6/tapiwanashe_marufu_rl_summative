// Static town: road grid, sidewalks, procedural buildings, site landmarks and the
// per-site markers (drop pads, urgency rings, light beams). Purely visual: site
// positions come from the replay's map (2.2 scene units per km); everything else
// is decoration placed by a seeded RNG, so the town is identical on every load.
//
// Geometry that shares a material is merged into one mesh, so the whole town is
// a few dozen draw calls.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import * as TX from './textures.js';

export const STOREY = 0.46, BAY = 0.45;
export const ROAD_HALF = 0.5, CURB = 0.045, LOT_Y = 0.055;
export const BLOCK = 6.6, INNER = 2.45, RING = 2.625;
export const LINES = [-23.1, -16.5, -9.9, -3.3, 3.3, 9.9, 16.5, 23.1];   // road centre lines (x and z)
export const CENTERS = [-19.8, -13.2, -6.6, 0, 6.6, 13.2, 19.8];        // block centres
export const ROAD_EXT = 42.4;                                          // roads continue into the countryside
const SEG_CENTERS = [-39.6, -33, -26.4, ...CENTERS, 26.4, 33, 39.6];
const FILL_SQUARES = [-36.3, -29.7, 29.7, 36.3];

class HeightMap {
  constructor(min = -46, max = 46, res = 0.25) {
    this.min = min; this.res = res; this.n = Math.ceil((max - min) / res);
    this.h = new Float32Array(this.n * this.n);
  }
  fill(cx, cz, w, d, h) {
    const i0 = Math.max(0, Math.floor((cx - w / 2 - this.min) / this.res)), i1 = Math.min(this.n - 1, Math.floor((cx + w / 2 - this.min) / this.res));
    const j0 = Math.max(0, Math.floor((cz - d / 2 - this.min) / this.res)), j1 = Math.min(this.n - 1, Math.floor((cz + d / 2 - this.min) / this.res));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) this.h[j * this.n + i] = Math.max(this.h[j * this.n + i], h);
  }
  at(x, z) {
    const i = Math.floor((x - this.min) / this.res), j = Math.floor((z - this.min) / this.res);
    return i < 0 || j < 0 || i >= this.n || j >= this.n ? 0 : this.h[j * this.n + i];
  }
  around(x, z, r) {
    let m = 0;
    for (const [dx, dz] of [[0, 0], [r, 0], [-r, 0], [0, r], [0, -r], [r, r], [-r, -r], [r, -r], [-r, r]]) m = Math.max(m, this.at(x + dx, z + dz));
    return m;
  }
}

// ---------------------------------------------------------------- geometry helpers
function scaleUV(g, su, sv, ou = 0, ov = 0) {
  const uv = g.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * su + ou, uv.getY(i) * sv + ov);
  return g;
}

/** Box whose side faces tile a facade texture (4 bays x 4 storeys per tile). */
function facadeBox(w, h, d, ou = 0, ov = 0) {
  const g = new THREE.BoxGeometry(w, h, d);
  const uv = g.attributes.uv, dims = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]];
  for (let f = 0; f < 6; f++) for (let k = 0; k < 4; k++) {
    const i = f * 4 + k;
    uv.setXY(i, uv.getX(i) * dims[f][0] / (4 * BAY) + ou, uv.getY(i) * dims[f][1] / (4 * STOREY) + ov);
  }
  return g;
}

/** Gabled roof: returns [slopes, gables] with the ridge along local x. */
function gableRoof(w, d, hr) {
  const x = w / 2, z = d / 2;
  const slopes = new THREE.BufferGeometry();
  slopes.setAttribute('position', new THREE.Float32BufferAttribute([
    -x, 0, z, x, 0, z, x, hr, 0, -x, hr, 0,        // front slope
    x, 0, -z, -x, 0, -z, -x, hr, 0, x, hr, 0,      // back slope
  ], 3));
  const sl = Math.hypot(z, hr) * 3;
  slopes.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, w * 3, 0, w * 3, sl, 0, sl, 0, 0, w * 3, 0, w * 3, sl, 0, sl], 2));
  slopes.setIndex([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
  slopes.computeVertexNormals();
  const gables = new THREE.BufferGeometry();
  gables.setAttribute('position', new THREE.Float32BufferAttribute([x, 0, z, x, 0, -z, x, hr, 0, -x, 0, -z, -x, 0, z, -x, hr, 0], 3));
  gables.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0.5, 0.5, 0, 0, 1, 0, 0.5, 0.5], 2));
  gables.setIndex([0, 1, 2, 3, 4, 5]);
  gables.computeVertexNormals();
  return [slopes, gables];
}

const rectsOverlap = (a, b) => Math.abs(a.cx - b.cx) * 2 < a.w + b.w && Math.abs(a.cz - b.cz) * 2 < a.d + b.d;

// ---------------------------------------------------------------- city
export function buildCity(scene, nodes, nodePos) {
  const rng = TX.mulberry32(1337);
  const hm = new HeightMap();
  const buckets = new Map();
  const add = (key, g, m) => { if (m) g.applyMatrix4(m); if (!buckets.has(key)) buckets.set(key, []); buckets.get(key).push(g); };
  const M = (x, y, z, ry = 0) => new THREE.Matrix4().makeRotationY(ry).setPosition(x, y, z);

  // ---- materials
  const std = (o) => new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0.02, ...o });
  const fac = {};
  for (const [k, o] of Object.entries({ glass: { roughness: 0.32, metalness: 0.35 }, concrete: {}, brick: { roughness: 0.92 },
    hospital: { roughness: 0.5 }, shop: { roughness: 0.6 }, house: { roughness: 0.9 } })) {
    const t = TX.facade(k, rng);
    fac[k] = std({ map: t.map, emissiveMap: t.emissive, emissive: 0xffffff, emissiveIntensity: 0.15, ...o });
  }
  const roadMat = std({ map: TX.roadTex(rng), roughness: 0.9 });
  const asphaltMat = std({ map: TX.asphaltTex(rng), roughness: 0.9 });
  const crossMat = std({ map: TX.crosswalkTex(rng), roughness: 0.9 });
  const wetMats = [roadMat, asphaltMat, crossMat];
  const sidewalkMat = std({ map: TX.sidewalkTex(rng), roughness: 0.88 });
  const plazaMat = std({ map: TX.plazaTex(rng), roughness: 0.85 });
  const grassTexture = TX.grassTex(rng);
  const lawnMat = std({ map: grassTexture, roughness: 1 });
  const mats = {
    ...fac,
    roof: std({ map: TX.roofTex(rng), roughness: 0.95 }),
    tile: std({ map: TX.tileRoofTex(rng), roughness: 0.8 }),
    trim: std({ color: 0xd9d6cf, roughness: 0.8 }),
    darkTrim: std({ color: 0x4a5258, roughness: 0.6, metalness: 0.3 }),
    metal: std({ color: 0xa9b1b5, roughness: 0.45, metalness: 0.6 }),
    white: std({ color: 0xf3f4f2, roughness: 0.6 }),
    wood: std({ color: 0x7a5a3c, roughness: 0.9 }),
    door: std({ color: 0x5a3e2a, roughness: 0.7 }),
    green: std({ color: 0x1f9e6e, roughness: 0.6 }),
    red: std({ color: 0xc8343a, roughness: 0.6 }),
    blue: std({ color: 0x2f6eb5, roughness: 0.5, metalness: 0.2 }),
  };

  // ---- ground
  const ground = new THREE.Mesh(new THREE.CircleGeometry(160, 72), std({ map: grassTexture, roughness: 1 }));
  ground.material.map = grassTexture.clone(); ground.material.map.repeat.set(70, 70); ground.material.map.needsUpdate = true;
  ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);
  const hillMat = std({ color: 0x5d8550, roughness: 1 });
  for (let i = 0; i < 14; i++) {
    const a = (i / 14) * Math.PI * 2 + rng() * 0.3, r = 95 + rng() * 40;
    const hill = new THREE.Mesh(new THREE.SphereGeometry(18 + rng() * 16, 20, 12), hillMat);
    hill.scale.y = 0.18 + rng() * 0.2; hill.position.set(Math.cos(a) * r, -2, Math.sin(a) * r); hill.receiveShadow = true; scene.add(hill);
  }

  // ---- roads (instanced: segments, intersections, crosswalks)
  const segGeo = scaleUV(new THREE.PlaneGeometry(BLOCK - 2 * ROAD_HALF, 2 * ROAD_HALF).rotateX(-Math.PI / 2), (BLOCK - 1) / 2, 1);
  const segs = [];
  for (const line of LINES) for (const c of SEG_CENTERS) { segs.push(M(c, 0.012, line)); segs.push(M(line, 0.012, c, Math.PI / 2)); }
  scene.add(instanced(segGeo, roadMat, segs, { receive: true }));
  const sqGeo = scaleUV(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), 1, 1);
  const squares = [];
  for (const a of LINES) for (const b of LINES) squares.push(M(a, 0.012, b));
  for (const line of LINES) for (const f of FILL_SQUARES) { squares.push(M(f, 0.012, line)); squares.push(M(line, 0.012, f)); }
  scene.add(instanced(sqGeo, asphaltMat, squares, { receive: true }));
  const cwGeo = new THREE.PlaneGeometry(0.3, 2 * ROAD_HALF).rotateX(-Math.PI / 2);
  const cws = [];
  for (const a of LINES) for (const b of LINES) for (const s of [-1, 1]) {
    cws.push(M(a + s * 0.68, 0.014, b));                // across the x-road approaching along x
    cws.push(M(a, 0.014, b + s * 0.68, Math.PI / 2));
  }
  scene.add(instanced(cwGeo, crossMat, cws, { receive: true }));

  // ---- sites: compute plots first so decoration keeps clear of them
  const plots = nodes.map((n, i) => sitePlot(n, i, nodePos));
  const exclusions = plots.flatMap((p) => [p.building, { cx: p.pad.x, cz: p.pad.z, w: p.pad.r * 2 + 0.3, d: p.pad.r * 2 + 0.3 }]);
  const blocked = (r) => exclusions.some((e) => rectsOverlap(e, r));
  const nearPad = (x, z, r) => plots.some((p) => Math.hypot(p.pad.x - x, p.pad.z - z) < p.pad.r + r);

  // ---- blocks: sidewalk ring + lot surface (instanced)
  const walkGeo = scaleUV(new THREE.BoxGeometry(BLOCK - 2 * ROAD_HALF, CURB, BLOCK - 2 * ROAD_HALF), 5.6, 5.6);
  const lotGeo = scaleUV(new THREE.BoxGeometry(2 * INNER, 0.012, 2 * INNER), 2.5, 2.5);
  const walks = [], lawns = [], plazas = [];
  const treeSpots = [], lampSpots = [], parkBlocks = [];
  const siteBlock = (cx, cz) => plots.find((p) => Math.abs(p.block.x - cx) < 0.1 && Math.abs(p.block.z - cz) < 0.1);
  for (const cx of CENTERS) for (const cz of CENTERS) {
    walks.push(M(cx, CURB / 2, cz));
    const dist = Math.hypot(cx, cz), site = siteBlock(cx, cz);
    const park = !site && dist > 6 && rng() < 0.1;
    const paved = site ? site.kind !== 'home' : (!park && dist < 12) || (!park && dist < 17 && rng() < 0.5);
    (paved ? plazas : lawns).push(M(cx, CURB + 0.006, cz));
    if (park) parkBlocks.push([cx, cz]);
    // street trees and lamps on the sidewalk ring
    for (const [sx, sz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const along = [sz, sx];                          // direction along this side
      for (const t of [-1.6, 1.6]) {
        const x = cx + sx * 2.6 + along[0] * t, z = cz + sz * 2.6 + along[1] * t;
        if (!nearPad(x, z, 0.7) && rng() < 0.8) treeSpots.push({ x, z, s: 0.8 + rng() * 0.35, kind: 'street' });
      }
      const lx = cx + sx * 2.74, lz = cz + sz * 2.74;
      if (!nearPad(lx, lz, 0.6)) lampSpots.push({ x: lx, z: lz, dx: sx, dz: sz });
    }
    if (park) {
      for (let k = 0; k < 16; k++) treeSpots.push({ x: cx + (rng() - 0.5) * 4.4, z: cz + (rng() - 0.5) * 4.4, s: 0.9 + rng() * 0.8, kind: rng() < 0.3 ? 'conifer' : 'park' });
      add('trim', new THREE.BoxGeometry(4.6, 0.01, 0.35), M(cx, LOT_Y + 0.004, cz));
      add('trim', new THREE.BoxGeometry(0.35, 0.01, 4.6), M(cx, LOT_Y + 0.004, cz));
      continue;
    }
    if (site) continue;
    generateLots(cx, cz, dist);
  }
  scene.add(instanced(walkGeo, sidewalkMat, walks, { receive: true }));
  scene.add(instanced(lotGeo, lawnMat, lawns, { receive: true }));
  scene.add(instanced(lotGeo, plazaMat, plazas, { receive: true }));

  function generateLots(cx, cz, dist) {
    // Downtown blocks sometimes hold one large tower; otherwise four lots.
    if (dist < 10 && rng() < 0.3) {
      const r = { cx, cz, w: 4.5, d: 4.5 };
      if (!blocked(r)) return office(cx, cz, 4.05, 4.05, 6 + Math.floor(rng() * 3), true);
    }
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      const x = cx + sx * 1.3, z = cz + sz * 1.3;
      if (blocked({ cx: x, cz: z, w: 2.3, d: 2.3 })) continue;
      if (dist < 10) office(x, z, snap(1.4 + rng() * 0.8), snap(1.4 + rng() * 0.8), 4 + Math.floor(rng() * 4), false);
      else if (dist < 17) midrise(x, z, snap(1.4 + rng() * 0.8), snap(1.4 + rng() * 0.8), 2 + Math.floor(rng() * 3));
      else if (rng() < 0.85) house(x, z, rng() < 0.5 ? 1.35 : 1.8, 1.35, sz > 0 ? 0 : Math.PI, 'house');
      else treeSpots.push({ x, z, s: 1 + rng() * 0.5, kind: 'park' });
    }
  }
  function snap(v) { return Math.max(BAY * 3, Math.round(v / BAY) * BAY); }

  // ---- building kits
  function roofKit(x, z, w, d, h, equipment = 2) {
    add('roof', scaleUV(new THREE.BoxGeometry(w - 0.04, 0.03, d - 0.04), w, d), M(x, h + 0.015, z));
    const t = 0.05, ph = 0.09;
    add('trim', new THREE.BoxGeometry(w, ph, t), M(x, h + ph / 2, z + d / 2 - t / 2));
    add('trim', new THREE.BoxGeometry(w, ph, t), M(x, h + ph / 2, z - d / 2 + t / 2));
    add('trim', new THREE.BoxGeometry(t, ph, d), M(x + w / 2 - t / 2, h + ph / 2, z));
    add('trim', new THREE.BoxGeometry(t, ph, d), M(x - w / 2 + t / 2, h + ph / 2, z));
    for (let k = 0; k < equipment; k++) {
      const ux = x + (rng() - 0.5) * (w - 0.6), uz = z + (rng() - 0.5) * (d - 0.6);
      const uw = 0.2 + rng() * 0.25, ud = 0.18 + rng() * 0.2, uh = 0.12 + rng() * 0.1;
      add('metal', new THREE.BoxGeometry(uw, uh, ud), M(ux, h + uh / 2 + 0.03, uz));
      add('darkTrim', new THREE.CylinderGeometry(0.06, 0.06, 0.02, 12), M(ux, h + uh + 0.04, uz));
    }
  }

  function office(x, z, w, d, floors, tower) {
    const style = rng() < 0.6 ? 'glass' : 'concrete';
    const ou = Math.floor(rng() * 4) * 0.25, ov = Math.floor(rng() * 4) * 0.25;
    if (tower && rng() < 0.6) {                       // podium + setback tower
      const ph = 2 * STOREY;
      add('concrete', facadeBox(w, ph, d, ou, ov), M(x, LOT_Y + ph / 2, z));
      roofKit(x, z, w, d, LOT_Y + ph, 1);
      const tw = snap(w * 0.6), td = snap(d * 0.6), th = floors * STOREY;
      add(style, facadeBox(tw, th, td, ou, ov), M(x, LOT_Y + ph + th / 2, z));
      roofKit(x, z, tw, td, LOT_Y + ph + th, 2);
      add('metal', new THREE.CylinderGeometry(0.015, 0.02, 0.6, 6), M(x + tw * 0.25, LOT_Y + ph + th + 0.3, z));
      hm.fill(x, z, w, d, LOT_Y + ph); hm.fill(x, z, tw, td, LOT_Y + ph + th + 0.6);
      return;
    }
    const h = floors * STOREY;
    add(style, facadeBox(w, h, d, ou, ov), M(x, LOT_Y + h / 2, z));
    roofKit(x, z, w, d, LOT_Y + h, 1 + Math.floor(rng() * 3));
    hm.fill(x, z, w, d, LOT_Y + h + 0.25);
  }

  function midrise(x, z, w, d, floors) {
    const style = rng() < 0.65 ? 'brick' : 'concrete';
    const h = floors * STOREY, ou = Math.floor(rng() * 4) * 0.25;
    add(style, facadeBox(w, h, d, ou, 0), M(x, LOT_Y + h / 2, z));
    roofKit(x, z, w, d, LOT_Y + h, 1 + Math.floor(rng() * 2));
    if (style === 'brick' && rng() < 0.5) {           // water tank
      const tx = x + (rng() - 0.5) * (w - 0.5), tz = z + (rng() - 0.5) * (d - 0.5);
      add('wood', new THREE.CylinderGeometry(0.13, 0.13, 0.22, 12), M(tx, LOT_Y + h + 0.25, tz));
      add('darkTrim', new THREE.ConeGeometry(0.15, 0.1, 12), M(tx, LOT_Y + h + 0.41, tz));
      add('darkTrim', new THREE.BoxGeometry(0.2, 0.14, 0.2), M(tx, LOT_Y + h + 0.07, tz));
    }
    hm.fill(x, z, w, d, LOT_Y + h + 0.3);
  }

  function house(x, z, w, d, yaw, wall, porch = true) {
    const m = M(x, LOT_Y, z, yaw), h = 2 * STOREY;
    add(wall, facadeBox(w, h, d, Math.floor(rng() * 4) * 0.25, 0), m.clone().multiply(M(0, h / 2, 0)));
    const [slopes, gables] = gableRoof(w + 0.16, d + 0.16, 0.42);
    add('tile', slopes, m.clone().multiply(M(0, h, 0)));
    add(wall, gables, m.clone().multiply(M(0, h, 0)));
    add('brick', facadeBox(0.14, 0.4, 0.14), m.clone().multiply(M(w * 0.28, h + 0.32, -d * 0.18)));
    add('door', new THREE.BoxGeometry(0.16, 0.28, 0.02), m.clone().multiply(M(0, 0.14, d / 2 + 0.005)));
    if (porch) {
      add('trim', new THREE.BoxGeometry(0.7, 0.03, 0.35), m.clone().multiply(M(0, 0.015, d / 2 + 0.17)));
      add('white', new THREE.BoxGeometry(0.025, 0.3, 0.025), m.clone().multiply(M(-0.32, 0.18, d / 2 + 0.32)));
      add('white', new THREE.BoxGeometry(0.025, 0.3, 0.025), m.clone().multiply(M(0.32, 0.18, d / 2 + 0.32)));
      add('tile', new THREE.BoxGeometry(0.8, 0.03, 0.42), m.clone().multiply(M(0, 0.34, d / 2 + 0.18)));
    }
    const fw = Math.abs(Math.sin(yaw)) > 0.5 ? d : w, fd = Math.abs(Math.sin(yaw)) > 0.5 ? w : d;
    hm.fill(x, z, fw + 0.2, fd + 0.2, LOT_Y + h + 0.45);
    treeSpots.push({ x: x + (rng() < 0.5 ? -1 : 1) * (w / 2 + 0.3), z: z - Math.cos(yaw) * 0.2, s: 0.8 + rng() * 0.4, kind: 'park' });
  }

  // ---- site landmarks and markers
  const siteFx = [], signs = [];
  let depotFx = null;
  const padTex = TX.dropPadTex();
  const padMat = std({ map: padTex, roughness: 0.7, transparent: true });
  const glow = TX.glowTex(), beam = TX.beamTex();
  const fxMat = (color, map, opacity = 1) => new THREE.MeshBasicMaterial({ color, ...(map ? { map } : {}), transparent: true, opacity, depthWrite: false,
    blending: THREE.AdditiveBlending, toneMapped: false, side: THREE.DoubleSide });

  plots.forEach((plot, i) => {
    const n = nodes[i], b = plot.building, f = plot.front;
    if (n.kind === 'depot') { depotFx = buildDepot(plot); siteFx.push(null); return; }
    const padY = LOT_Y + 0.02;
    const pad = new THREE.Mesh(new THREE.CircleGeometry(plot.pad.r, 40).rotateX(-Math.PI / 2), padMat);
    pad.position.set(plot.pad.x, padY, plot.pad.z); pad.receiveShadow = true; scene.add(pad);
    if (n.kind === 'hospital') buildHospital(b, f);
    else if (n.kind === 'pharmacy') buildPharmacy(b, f);
    else buildHome(b, f);

    const ring = new THREE.Mesh(new THREE.RingGeometry(plot.pad.r + 0.04, plot.pad.r + 0.16, 56).rotateX(-Math.PI / 2), fxMat(0xffffff));
    ring.position.set(plot.pad.x, padY + 0.01, plot.pad.z);
    const halo = new THREE.Mesh(new THREE.PlaneGeometry(3.2, 3.2).rotateX(-Math.PI / 2), fxMat(0xffffff, glow, 0.7));
    halo.position.set(plot.pad.x, padY + 0.005, plot.pad.z);
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.16, 18, 20, 1, true), fxMat(0xffffff, beam, 0.3));
    shaft.position.set(plot.pad.x, padY + 9, plot.pad.z);
    const onboard = new THREE.Mesh(new THREE.TorusGeometry(plot.pad.r * 0.7, 0.035, 8, 48).rotateX(-Math.PI / 2), fxMat(0x3ee0b5));
    onboard.position.set(plot.pad.x, padY + 0.12, plot.pad.z);
    for (const o of [ring, halo, shaft, onboard]) { o.visible = false; o.renderOrder = 2; scene.add(o); }
    siteFx.push({ p: new THREE.Vector3(plot.pad.x, 0, plot.pad.z), padY, ring, halo, shaft, onboard, popT: 1 });
  });

  function facing(b, f) {           // returns matrix at building centre with local +z = front
    return M(b.cx, LOT_Y, b.cz, Math.atan2(f.x, f.z));
  }
  function frontDims(b, f) { return Math.abs(f.x) > 0.5 ? [b.d, b.w] : [b.w, b.d]; }   // [width across front, depth]

  function buildHospital(b, f) {
    let [w, d] = frontDims(b, f);
    w = Math.min(snap(w - 0.2), 3.6); d = Math.min(snap(d - 0.2), 2.7);
    const m = facing(b, f), h = 5 * STOREY;
    add('hospital', facadeBox(w, h, d), m.clone().multiply(M(0, h / 2, 0)));
    roofKit(b.cx, b.cz, Math.abs(f.x) > 0.5 ? d : w, Math.abs(f.x) > 0.5 ? w : d, LOT_Y + h, 0);
    // rooftop helipad
    const hp = new THREE.Mesh(new THREE.BoxGeometry(1.15, 0.04, 1.15), [mats.darkTrim, mats.darkTrim,
      std({ map: TX.helipadTex(), roughness: 0.7 }), mats.darkTrim, mats.darkTrim, mats.darkTrim]);
    hp.applyMatrix4(m.clone().multiply(M(-w * 0.15, h + 0.08, 0))); hp.receiveShadow = true; scene.add(hp);
    add('metal', new THREE.BoxGeometry(0.3, 0.2, 0.3), m.clone().multiply(M(w * 0.34, h + 0.1, -d * 0.2)));
    // emergency entrance canopy
    add('white', new THREE.BoxGeometry(1.1, 0.04, 0.35), m.clone().multiply(M(-w * 0.2, 0.36, d / 2 + 0.17)));
    add('metal', new THREE.BoxGeometry(0.025, 0.34, 0.025), m.clone().multiply(M(-w * 0.2 - 0.5, 0.17, d / 2 + 0.32)));
    add('metal', new THREE.BoxGeometry(0.025, 0.34, 0.025), m.clone().multiply(M(-w * 0.2 + 0.5, 0.17, d / 2 + 0.32)));
    sign(TX.textTex('EMERGENCY', '#ffffff', '#c8343a'), 1.0, 0.16, m.clone().multiply(M(-w * 0.2, 0.43, d / 2 + 0.36)), 0.9, 'hospital');
    sign(TX.crossSignTex('#e5484d'), 0.62, 0.62, m.clone().multiply(M(0, h - 0.5, d / 2 + 0.012)), 1.2, 'hospital');
    hm.fill(b.cx, b.cz, b.w, b.d, LOT_Y + h + 0.3);
  }

  function buildPharmacy(b, f) {
    let [w, d] = frontDims(b, f);
    w = Math.min(snap(w - 0.3), 2.7); d = Math.min(snap(d - 0.2), 1.8);
    const m = facing(b, f), h = 2 * STOREY;
    add('shop', facadeBox(w, h, d), m.clone().multiply(M(0, h / 2, 0)));
    roofKit(b.cx, b.cz, Math.abs(f.x) > 0.5 ? d : w, Math.abs(f.x) > 0.5 ? w : d, LOT_Y + h, 1);
    const front = new THREE.Mesh(new THREE.PlaneGeometry(w * 0.8, 0.3),
      new THREE.MeshStandardMaterial({ color: 0xbfe8d8, emissive: 0xfff1d0, emissiveIntensity: 0.5, roughness: 0.2, metalness: 0.3 }));
    front.applyMatrix4(m.clone().multiply(M(0, 0.2, d / 2 + 0.006))); scene.add(front);
    signs.push({ mat: front.material, base: 0.5, kind: 'storefront' });
    const awning = new THREE.Mesh(new THREE.BoxGeometry(w * 0.86, 0.03, 0.28), mats.green);
    awning.applyMatrix4(m.clone().multiply(M(0, 0.4, d / 2 + 0.14)).multiply(new THREE.Matrix4().makeRotationX(0.25)));
    awning.castShadow = true; scene.add(awning);
    sign(TX.textTex('PHARMACY', '#ffffff', '#1f9e6e'), w * 0.7, 0.13, m.clone().multiply(M(0, 0.56, d / 2 + 0.012)), 0.8, 'pharmacy');
    // projecting blade sign with an illuminated green cross (both faces)
    const blade = m.clone().multiply(M(w / 2 - 0.1, 0.72, d / 2 + 0.22)).multiply(new THREE.Matrix4().makeRotationY(Math.PI / 2));
    add('darkTrim', new THREE.BoxGeometry(0.4, 0.02, 0.02), m.clone().multiply(M(w / 2 - 0.1, 0.92, d / 2 + 0.2)).multiply(new THREE.Matrix4().makeRotationY(Math.PI / 2)));
    sign(TX.crossSignTex('#1fbf7a'), 0.34, 0.34, blade, 1.3, 'pharmacy', true);
    hm.fill(b.cx, b.cz, b.w, b.d, LOT_Y + h + 0.3);
  }

  function buildHome(b, f) {
    const [w, d] = frontDims(b, f);
    house(b.cx, b.cz, Math.min(snap(w - 0.4), 1.8), Math.min(snap(d - 0.5), 1.35), Math.atan2(f.x, f.z), 'house');
    treeSpots.push({ x: b.cx - f.z * 1.2 + f.x * 0.3, z: b.cz + f.x * 1.2 + f.z * 0.3, s: 1.1, kind: 'park' });
    // picket fence along the front edge
    const fm = facing(b, f);
    for (let k = -4; k <= 4; k++) add('white', new THREE.BoxGeometry(0.02, 0.12, 0.02), fm.clone().multiply(M(k * 0.24, 0.06, Math.min(b.d, b.w) / 2)));
  }

  function sign(texture, w, h, matrix, intensity, kind, doubleSided = false) {
    const mat = new THREE.MeshStandardMaterial({ map: texture, emissiveMap: texture, emissive: 0xffffff, emissiveIntensity: intensity,
      roughness: 0.4, side: doubleSided ? THREE.DoubleSide : THREE.FrontSide });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
    mesh.applyMatrix4(matrix); scene.add(mesh);
    signs.push({ mat, base: intensity, kind });
  }

  function buildDepot(plot) {
    const b = plot.building, px = plot.pad.x, pz = plot.pad.z, r = plot.pad.r;
    // hangar / charging bay
    const w = Math.min(b.w - 0.2, 4.4), d = Math.min(b.d - 0.1, 1.7), h = 1.05;
    add('white', new THREE.BoxGeometry(w, h, d), M(b.cx, LOT_Y + h / 2, b.cz));
    add('blue', new THREE.BoxGeometry(w + 0.06, 0.08, d + 0.06), M(b.cx, LOT_Y + h + 0.04, b.cz));
    add('roof', scaleUV(new THREE.BoxGeometry(w - 0.04, 0.02, d - 0.04), w, d), M(b.cx, LOT_Y + h + 0.09, b.cz));
    for (const dx of [-1.2, 0, 1.2]) {
      const door = new THREE.Mesh(new THREE.PlaneGeometry(0.95, 0.72),
        new THREE.MeshStandardMaterial({ color: 0x9fd8ff, emissive: 0x4aa8ff, emissiveIntensity: 0.55, roughness: 0.3 }));
      door.position.set(b.cx + dx, LOT_Y + 0.36, b.cz + d / 2 + 0.006); scene.add(door);
      signs.push({ mat: door.material, base: 0.55, kind: 'depot' });
    }
    sign(TX.textTex('MEDIFLIGHT · DISTRIBUTION', '#ffffff', '#2f6eb5', 768, 96), 2.6, 0.2, M(b.cx, LOT_Y + h - 0.16, b.cz + d / 2 + 0.012), 0.7, 'depot');
    add('metal', new THREE.BoxGeometry(0.5, 0.18, 0.4), M(b.cx - 1.4, LOT_Y + h + 0.18, b.cz));
    add('metal', new THREE.BoxGeometry(0.4, 0.18, 0.4), M(b.cx + 1.4, LOT_Y + h + 0.18, b.cz));
    hm.fill(b.cx, b.cz, w, d, LOT_Y + h + 0.3);
    // glowing H pad
    const t = TX.depotPadTex();
    for (const m of [t.map, t.emissive]) { m.center.set(0.5, 0.5); m.rotation = Math.PI / 2; }
    const padMesh = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 0.08, 56),
      [std({ color: 0x2b3a44 }), std({ map: t.map, emissiveMap: t.emissive, emissive: 0xffffff, emissiveIntensity: 0.6, roughness: 0.6 }), std({ color: 0x2b3a44 })]);
    padMesh.position.set(px, LOT_Y + 0.04, pz); padMesh.receiveShadow = true; scene.add(padMesh);
    signs.push({ mat: padMesh.material[1], base: 0.6, kind: 'depot' });
    // charging gantry: two cantilever arms with light strips and cables
    const strip = new THREE.MeshBasicMaterial({ color: 0x3ee0b5, toneMapped: false });
    const cables = [];
    for (const s of [-1, 1]) {
      const gx = px + s * (r + 0.25);
      add('darkTrim', new THREE.BoxGeometry(0.08, 1.15, 0.08), M(gx, LOT_Y + 0.575, pz));
      add('darkTrim', new THREE.BoxGeometry(0.75, 0.07, 0.1), M(gx - s * 0.33, LOT_Y + 1.12, pz));
      const st = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.02, 0.03), strip);
      st.position.set(gx - s * 0.36, LOT_Y + 1.08, pz + 0.055); scene.add(st);
      const cable = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
        new THREE.LineBasicMaterial({ color: 0x1b2126 }));
      cable.userData.anchor = new THREE.Vector3(gx - s * 0.66, LOT_Y + 1.08, pz); cable.visible = false;
      scene.add(cable); cables.push(cable);
    }
    const chargeRing = new THREE.Mesh(new THREE.TorusGeometry(r - 0.08, 0.05, 10, 72).rotateX(-Math.PI / 2), fxMat(0x3ee0b5));
    chargeRing.material.opacity = 0; chargeRing.position.set(px, LOT_Y + 0.1, pz); scene.add(chargeRing);
    return { chargeRing, strip, cables, padTop: LOT_Y + 0.08 };
  }

  // ---- merge buckets into meshes
  for (const [key, list] of buckets) {
    const mesh = new THREE.Mesh(mergeGeometries(list, false), mats[key]);
    mesh.castShadow = true; mesh.receiveShadow = true; scene.add(mesh);
  }

  // ---- outskirts forest (keeps clear of roads)
  const onRoad = (x, z) => LINES.some((l) => (Math.abs(z - l) < 1.4 && Math.abs(x) < ROAD_EXT + 1) || (Math.abs(x - l) < 1.4 && Math.abs(z) < ROAD_EXT + 1));
  for (let i = 0; i < 520; i++) {
    const a = rng() * Math.PI * 2, r = 25.5 + Math.pow(rng(), 0.7) * 50;
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    if (onRoad(x, z)) continue;
    treeSpots.push({ x, z, s: 0.9 + rng() * 1.1, kind: rng() < 0.55 ? 'conifer' : 'park' });
  }

  // ---- runtime updates (visual only)
  const wetBase = wetMats.map((m) => m.color.clone());
  function setWeather(env, now) {
    for (const [k, m] of wetMats.entries()) {
      m.roughness = THREE.MathUtils.lerp(0.9, 0.18, env.wet);
      m.metalness = THREE.MathUtils.lerp(0.0, 0.35, env.wet);
      m.color.copy(wetBase[k]).multiplyScalar(1 - 0.35 * env.wet);
    }
    sidewalkMat.roughness = THREE.MathUtils.lerp(0.88, 0.4, env.wet);
    for (const k of ['glass', 'concrete', 'brick', 'hospital', 'shop', 'house']) fac[k].emissiveIntensity = 0.12 + 1.3 * env.darkness;
    const t = now / 1000;
    for (const s of signs) {
      const pulse = s.kind === 'pharmacy' ? 0.82 + 0.18 * Math.sin(t * 2.4) : 1;
      s.mat.emissiveIntensity = s.base * (0.85 + 0.9 * env.darkness) * pulse;
    }
  }

  return {
    plots, siteFx, depotFx, treeSpots, lampSpots, heightAt: (x, z) => hm.at(x, z),
    heightAround: (x, z, r) => hm.around(x, z, r), setWeather,
    isSolid: (x, z) => hm.at(x, z) > 0.15,
  };
}

function instanced(geo, mat, matrices, { cast = false, receive = false } = {}) {
  const im = new THREE.InstancedMesh(geo, mat, matrices.length);
  matrices.forEach((m, i) => im.setMatrixAt(i, m));
  im.castShadow = cast; im.receiveShadow = receive;
  return im;
}

/** Plot for a site: pad at the environment coordinate, building in the rest of its block. */
function sitePlot(n, i, nodePos) {
  const p = nodePos(n);
  const block = { x: Math.round(p.x / BLOCK) * BLOCK, z: Math.round(p.z / BLOCK) * BLOCK };
  const pad = { x: p.x, z: p.z + (n.kind === 'depot' ? 0.6 : 0), r: n.kind === 'depot' ? 1.15 : 0.6 };
  const dx = pad.x - block.x, dz = pad.z - block.z;
  const alongZ = Math.abs(dz) >= Math.abs(dx) - 1e-6;
  const side = alongZ ? (dz > 0.01 ? -1 : dz < -0.01 ? 1 : -1) : (dx > 0 ? -1 : 1);   // building side of the block
  let x0 = block.x - INNER, x1 = block.x + INNER, z0 = block.z - INNER, z1 = block.z + INNER;
  const gap = pad.r + 0.15;
  if (alongZ) { if (side < 0) z1 = Math.min(z1, pad.z - gap); else z0 = Math.max(z0, pad.z + gap); }
  else { if (side < 0) x1 = Math.min(x1, pad.x - gap); else x0 = Math.max(x0, pad.x + gap); }
  const building = { cx: (x0 + x1) / 2, cz: (z0 + z1) / 2, w: x1 - x0, d: z1 - z0 };
  const front = alongZ ? { x: 0, z: -side } : { x: -side, z: 0 };   // building faces its pad
  return { kind: n.kind, block, pad, building, front, index: i };
}
