// Procedural canvas textures. Everything is generated from a seeded RNG, so the
// town looks identical on every load and no image assets are needed.
import * as THREE from 'three';

export function mulberry32(a) {
  return function rand() {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function canvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return [c, c.getContext('2d')];
}

function tex(c, { srgb = true, repeat = true } = {}) {
  const t = new THREE.CanvasTexture(c);
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

function speckle(g, w, h, rng, amount) {
  const img = g.getImageData(0, 0, w, h), d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (rng() - 0.5) * amount;
    d[i] += n; d[i + 1] += n; d[i + 2] += n;
  }
  g.putImageData(img, 0, 0);
}

// ------------------------------------------------------------------ facades
// One tile = 4 bays x 4 storeys (64 px each). Buildings scale their UVs so a
// bay is 0.45 scene units wide and a storey 0.46 units tall.
const FACADE = {
  glass:    { wall: '#2b3945', frame: '#93a6b2', top: '#6a8ea8', bot: '#2a4256', win: [3, 4, 58, 50], band: '#38464f', lit: 0.28, noise: 6 },
  concrete: { wall: '#d3cec3', frame: '#f2efe8', top: '#566b7b', bot: '#26343f', win: [13, 12, 38, 36], sill: '#b4aea2', lit: 0.3, noise: 16 },
  brick:    { wall: '#8b4a38', frame: '#eee4d4', top: '#4e606c', bot: '#202d36', win: [17, 10, 30, 40], sill: '#cdbfa8', lit: 0.34, noise: 10, brick: true },
  hospital: { wall: '#eef2f3', frame: '#cbd6db', top: '#9fd0ee', bot: '#3d6d8f', win: [5, 16, 54, 28], band: '#d7e1e5', lit: 0.55, noise: 6 },
  shop:     { wall: '#f1f3ef', frame: '#ffffff', top: '#7fa7a0', bot: '#2d4a46', win: [10, 12, 44, 36], sill: '#d8dcd6', lit: 0.4, noise: 8 },
  house:    { wall: '#ece0c8', frame: '#ffffff', top: '#62798a', bot: '#2c3c48', win: [20, 14, 24, 30], shutter: '#5b7a6a', lit: 0.3, noise: 8, siding: true },
};

export function facade(style, rng) {
  const N = 256, cell = 64, P = FACADE[style];
  const [c, g] = canvas(N, N);
  const [ce, ge] = canvas(N, N);
  ge.fillStyle = '#000'; ge.fillRect(0, 0, N, N);
  g.fillStyle = P.wall; g.fillRect(0, 0, N, N);
  if (P.brick) {
    g.fillStyle = '#a7907c';
    for (let y = 0; y < N; y += 8) {
      g.fillRect(0, y, N, 1);
      for (let x = (y / 8) % 2 ? 8 : 0; x < N; x += 16) g.fillRect(x, y, 1, 8);
    }
  }
  if (P.siding) { g.fillStyle = 'rgba(0,0,0,0.07)'; for (let y = 0; y < N; y += 6) g.fillRect(0, y, N, 1); }
  for (let by = 0; by < 4; by++) {
    for (let bx = 0; bx < 4; bx++) {
      const x = bx * cell, y = by * cell, [wx, wy, ww, wh] = P.win;
      if (P.band) { g.fillStyle = P.band; g.fillRect(x, y + cell - 7, cell, 7); }
      g.fillStyle = P.frame; g.fillRect(x + wx - 2, y + wy - 2, ww + 4, wh + 4);
      const grad = g.createLinearGradient(0, y + wy, 0, y + wy + wh);
      grad.addColorStop(0, P.top); grad.addColorStop(1, P.bot);
      g.fillStyle = grad; g.fillRect(x + wx, y + wy, ww, wh);
      if (rng() < P.lit) {
        const warm = ['#ffd592', '#ffe4b0', '#fff0d2', '#d4e8ff'][Math.floor(rng() * 4)];
        g.globalAlpha = 0.5; g.fillStyle = warm; g.fillRect(x + wx, y + wy, ww, wh); g.globalAlpha = 1;
        ge.globalAlpha = 0.55 + 0.45 * rng(); ge.fillStyle = warm; ge.fillRect(x + wx, y + wy, ww, wh); ge.globalAlpha = 1;
      }
      g.fillStyle = 'rgba(255,255,255,0.18)'; g.fillRect(x + wx + 2, y + wy + 2, ww * 0.35, 2);   // glint
      g.fillStyle = P.frame; g.fillRect(x + wx + ww / 2 - 1, y + wy, 2, wh);
      if (P.sill) { g.fillStyle = P.sill; g.fillRect(x + wx - 4, y + wy + wh + 2, ww + 8, 4); }
      if (P.shutter) { g.fillStyle = P.shutter; g.fillRect(x + wx - 9, y + wy, 6, wh); g.fillRect(x + wx + ww + 3, y + wy, 6, wh); }
    }
  }
  speckle(g, N, N, rng, P.noise);
  return { map: tex(c), emissive: tex(ce) };
}

// ------------------------------------------------------------------ ground
export function roadTex(rng) {          // u: 2 units along the road, v: 1 unit across
  const [c, g] = canvas(256, 128);
  g.fillStyle = '#3b3f43'; g.fillRect(0, 0, 256, 128);
  speckle(g, 256, 128, rng, 26);
  g.fillStyle = 'rgba(0,0,0,0.16)'; g.fillRect(0, 24, 256, 14); g.fillRect(0, 90, 256, 14);   // tyre wear
  g.fillStyle = 'rgba(240,240,232,0.85)'; g.fillRect(0, 6, 256, 3); g.fillRect(0, 119, 256, 3);
  g.fillStyle = '#e9dfb8'; g.fillRect(0, 62, 112, 4);
  return tex(c);
}

export function asphaltTex(rng) {
  const [c, g] = canvas(128, 128);
  g.fillStyle = '#3b3f43'; g.fillRect(0, 0, 128, 128);
  speckle(g, 128, 128, rng, 26);
  return tex(c);
}

export function crosswalkTex(rng) {     // u: along travel (0.3 units), v: across the road (1 unit)
  const [c, g] = canvas(64, 256);
  g.fillStyle = '#3b3f43'; g.fillRect(0, 0, 64, 256);
  speckle(g, 64, 256, rng, 26);
  g.fillStyle = 'rgba(244,244,236,0.92)';
  for (let k = 0; k < 7; k++) g.fillRect(4, 14 + k * 34, 56, 20);
  return tex(c);
}

export function sidewalkTex(rng) {      // 1 unit, 4x4 slabs
  const [c, g] = canvas(256, 256);
  g.fillStyle = '#b8b5ad'; g.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    const v = Math.floor(rng() * 14) - 7;
    g.fillStyle = `rgb(${184 + v},${181 + v},${173 + v})`; g.fillRect(i * 64 + 2, j * 64 + 2, 60, 60);
  }
  speckle(g, 256, 256, rng, 18);
  return tex(c);
}

export function plazaTex(rng) {
  const [c, g] = canvas(256, 256);
  g.fillStyle = '#9d9184'; g.fillRect(0, 0, 256, 256);
  for (let y = 0; y < 256; y += 16) for (let x = (y / 16) % 2 ? -16 : 0; x < 256; x += 32) {
    const v = Math.floor(rng() * 22) - 11;
    g.fillStyle = `rgb(${196 + v},${182 + v},${162 + v})`; g.fillRect(x + 1, y + 1, 30, 14);
  }
  speckle(g, 256, 256, rng, 14);
  return tex(c);
}

export function grassTex(rng) {
  const [c, g] = canvas(256, 256);
  g.fillStyle = '#5b8a4a'; g.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 5000; i++) {
    const v = rng();
    g.fillStyle = v < 0.33 ? '#4e7d3f' : v < 0.66 ? '#6a9a54' : '#5f8f48';
    g.fillRect(rng() * 256, rng() * 256, 1 + rng() * 2, 2 + rng() * 3);
  }
  return tex(c);
}

export function roofTex(rng) {
  const [c, g] = canvas(128, 128);
  g.fillStyle = '#8a8c89'; g.fillRect(0, 0, 128, 128);
  speckle(g, 128, 128, rng, 44);
  return tex(c);
}

export function tileRoofTex(rng) {
  const [c, g] = canvas(128, 128);
  g.fillStyle = '#9c4a33'; g.fillRect(0, 0, 128, 128);
  for (let y = 0; y < 128; y += 12) {
    g.fillStyle = 'rgba(0,0,0,0.25)'; g.fillRect(0, y, 128, 2);
    for (let x = (y / 12) % 2 ? 8 : 0; x < 128; x += 16) { g.fillStyle = 'rgba(0,0,0,0.12)'; g.fillRect(x, y, 1, 12); }
  }
  speckle(g, 128, 128, rng, 16);
  return tex(c);
}

// ------------------------------------------------------------------ markings & signs
export function depotPadTex() {
  const [c, g] = canvas(512, 512), [ce, ge] = canvas(512, 512);
  g.fillStyle = '#26323a'; g.fillRect(0, 0, 512, 512);
  ge.fillStyle = '#000'; ge.fillRect(0, 0, 512, 512);
  for (const ctx of [g, ge]) {
    ctx.strokeStyle = '#3ee0b5'; ctx.lineWidth = 18; ctx.beginPath(); ctx.arc(256, 256, 226, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([26, 18]); ctx.lineWidth = 6; ctx.beginPath(); ctx.arc(256, 256, 190, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#f4fbff';
    ctx.fillRect(176, 140, 40, 232); ctx.fillRect(296, 140, 40, 232); ctx.fillRect(176, 236, 160, 40);
  }
  return { map: tex(c, { repeat: false }), emissive: tex(ce, { repeat: false }) };
}

export function dropPadTex() {
  const [c, g] = canvas(256, 256);
  g.fillStyle = '#39434a'; g.beginPath(); g.arc(128, 128, 126, 0, Math.PI * 2); g.fill();
  g.strokeStyle = '#f2f2ea'; g.lineWidth = 8; g.beginPath(); g.arc(128, 128, 100, 0, Math.PI * 2); g.stroke();
  g.strokeStyle = '#ffc34d'; g.lineWidth = 6; g.setLineDash([16, 12]); g.beginPath(); g.arc(128, 128, 118, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
  g.fillStyle = '#f2f2ea'; g.fillRect(118, 70, 20, 116); g.fillRect(70, 118, 116, 20);
  return tex(c, { repeat: false });
}

export function helipadTex() {
  const [c, g] = canvas(256, 256);
  g.fillStyle = '#2c3136'; g.fillRect(0, 0, 256, 256);
  g.strokeStyle = '#ffd34d'; g.lineWidth = 6; g.strokeRect(10, 10, 236, 236);
  g.strokeStyle = '#f4f4f0'; g.lineWidth = 10; g.beginPath(); g.arc(128, 128, 92, 0, Math.PI * 2); g.stroke();
  g.fillStyle = '#e5484d'; g.fillRect(84, 72, 22, 112); g.fillRect(150, 72, 22, 112); g.fillRect(84, 117, 88, 22);
  return tex(c, { repeat: false });
}

export function crossSignTex(color) {
  const [c, g] = canvas(128, 128);
  g.fillStyle = '#ffffff'; g.beginPath(); g.roundRect(4, 4, 120, 120, 18); g.fill();
  g.fillStyle = color; g.fillRect(48, 20, 32, 88); g.fillRect(20, 48, 88, 32);
  return tex(c, { repeat: false });
}

export function textTex(text, fg, bg, w = 512, h = 96) {
  const [c, g] = canvas(w, h);
  g.fillStyle = bg; g.fillRect(0, 0, w, h);
  g.fillStyle = fg; g.font = `800 ${Math.round(h * 0.56)}px system-ui, -apple-system, sans-serif`;
  g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(text, w / 2, h / 2 + 2);
  return tex(c, { repeat: false });
}

// ------------------------------------------------------------------ effects (linear, not sRGB)
export function glowTex() {
  const [c, g] = canvas(128, 128);
  const r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(0.25, 'rgba(255,255,255,0.55)'); r.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = r; g.fillRect(0, 0, 128, 128);
  return tex(c, { srgb: false, repeat: false });
}

export function beamTex() {             // canvas top = cylinder top: fade out upwards
  const [c, g] = canvas(8, 256);
  const l = g.createLinearGradient(0, 0, 0, 256);
  l.addColorStop(0, 'rgba(255,255,255,0)'); l.addColorStop(0.7, 'rgba(255,255,255,0.35)'); l.addColorStop(1, 'rgba(255,255,255,1)');
  g.fillStyle = l; g.fillRect(0, 0, 8, 256);
  return tex(c, { srgb: false, repeat: false });
}

export function cloudTex(rng) {
  const [c, g] = canvas(256, 256);
  for (let i = 0; i < 14; i++) {
    const x = 60 + rng() * 136, y = 80 + rng() * 96, r = 40 + rng() * 50;
    const grad = g.createRadialGradient(x, y, 0, x, y, r);
    grad.addColorStop(0, 'rgba(255,255,255,0.55)'); grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad; g.fillRect(0, 0, 256, 256);
  }
  return tex(c, { repeat: false });
}

export function rotorBlurTex() {
  const [c, g] = canvas(128, 128);
  const r = g.createRadialGradient(64, 64, 8, 64, 64, 64);
  r.addColorStop(0, 'rgba(30,40,46,0)'); r.addColorStop(0.2, 'rgba(30,40,46,0.55)');
  r.addColorStop(0.92, 'rgba(30,40,46,0.35)'); r.addColorStop(1, 'rgba(30,40,46,0)');
  g.fillStyle = r; g.fillRect(0, 0, 128, 128);
  g.strokeStyle = 'rgba(220,230,235,0.25)'; g.lineWidth = 2;
  for (let k = 0; k < 3; k++) { g.beginPath(); g.arc(64, 64, 30 + k * 11, k, k + 1.2); g.stroke(); }
  return tex(c, { repeat: false });
}
