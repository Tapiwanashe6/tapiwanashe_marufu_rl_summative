// Tactical radar minimap (2D canvas). Reads the same reconstructed replay state
// as the 3D scene (st = ep.stateAt(simT)), drawn in scene coordinates
// (2.2 units per km), north up.
import { LINES, ROAD_EXT } from './city.js';

const KIND = { hospital: '#ff5d73', pharmacy: '#30c48d', home: '#ffb224', depot: '#4aa8ff' };
const URG = { 3: '#ff5d5d', 2: '#ffb224', 1: '#5fd4a0' };
const PKG = { blood: '#e5484d', emergency_kit: '#f76b15', vaccine: '#3e9bff', prescription: '#b48cff', medicine: '#30c48d' };

export class Minimap {
  constructor(canvas, unitsPerKm) {
    this.c = canvas; this.g = canvas.getContext('2d'); this.upk = unitsPerKm;
    this.range = 25;                      // scene units from centre to edge
    this.trail = [];
  }

  resize() {
    const dpr = Math.min(devicePixelRatio, 2), r = this.c.getBoundingClientRect();
    if (this.c.width !== Math.round(r.width * dpr)) { this.c.width = Math.round(r.width * dpr); this.c.height = Math.round(r.height * dpr); }
    this.dpr = dpr; this.size = r.width;
  }

  draw({ ep, st, dronePos, droneYaw, camPos, camDir, now }) {
    this.resize();
    const g = this.g, S = this.size, R = S / 2 - 4, k = R / this.range, dpr = this.dpr;
    const X = (x) => S / 2 + x * k, Z = (z) => S / 2 + z * k;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, S, S);
    g.save();
    g.beginPath(); g.arc(S / 2, S / 2, R, 0, Math.PI * 2); g.clip();
    g.fillStyle = 'rgba(6,18,24,0.82)'; g.fillRect(0, 0, S, S);

    // range rings + sweep
    g.strokeStyle = 'rgba(62,224,181,0.10)'; g.lineWidth = 1;
    for (const rr of [0.33, 0.66]) { g.beginPath(); g.arc(S / 2, S / 2, R * rr, 0, Math.PI * 2); g.stroke(); }
    const sweep = (now / 1000) * 1.2 % (Math.PI * 2);
    const grad = g.createConicGradient(sweep, S / 2, S / 2);
    grad.addColorStop(0, 'rgba(62,224,181,0.18)'); grad.addColorStop(0.08, 'rgba(62,224,181,0)'); grad.addColorStop(1, 'rgba(62,224,181,0)');
    g.fillStyle = grad; g.fillRect(0, 0, S, S);

    // roads
    g.strokeStyle = 'rgba(150,180,190,0.28)'; g.lineWidth = Math.max(1.5, 1.0 * k);
    for (const l of LINES) {
      g.beginPath(); g.moveTo(X(-ROAD_EXT), Z(l)); g.lineTo(X(ROAD_EXT), Z(l)); g.stroke();
      g.beginPath(); g.moveTo(X(l), Z(-ROAD_EXT)); g.lineTo(X(l), Z(ROAD_EXT)); g.stroke();
    }

    // flight path + trail
    const last = this.trail.at(-1);
    if (!last || Math.hypot(last[0] - dronePos.x, last[1] - dronePos.z) > 0.3) { this.trail.push([dronePos.x, dronePos.z]); if (this.trail.length > 80) this.trail.shift(); }
    if (this.trail.length > 1) {
      g.strokeStyle = 'rgba(255,255,255,0.22)'; g.lineWidth = 1.2; g.beginPath();
      this.trail.forEach(([x, z], i) => (i ? g.lineTo(X(x), Z(z)) : g.moveTo(X(x), Z(z)))); g.stroke();
    }
    if (st.flight && !st.flight.crash && st.flight.u < 1) {
      const to = ep.nodes[st.flight.to];
      g.setLineDash([5, 4]); g.strokeStyle = '#3ee0b5'; g.lineWidth = 1.6;
      g.beginPath(); g.moveTo(X(dronePos.x), Z(dronePos.z)); g.lineTo(X(to.x * this.upk), Z(-to.y * this.upk)); g.stroke();
      g.setLineDash([]);
    }

    // sites + urgency rings
    const pulse = 0.5 + 0.5 * Math.sin(now / 160);
    ep.nodes.forEach((n, idx) => {
      const x = X(n.x * this.upk), z = Z(-n.y * this.upk);
      const o = idx > 0 ? st.orders[idx - 1] : null;
      if (o) {
        const tight = o.deadline - st.t < 15;
        g.strokeStyle = o.status === 'onboard' ? PKG[o.type] : URG[o.urgency];
        g.lineWidth = o.status === 'onboard' ? 1.6 : 2.2;
        if (o.status === 'onboard') g.setLineDash([3, 3]);
        g.beginPath(); g.arc(x, z, 8 + (tight && o.status !== 'onboard' ? pulse * 3 : 0), 0, Math.PI * 2); g.stroke();
        g.setLineDash([]);
      }
      g.fillStyle = KIND[n.kind];
      if (n.kind === 'depot') { g.fillRect(x - 4.5, z - 4.5, 9, 9); }
      else if (n.kind === 'hospital') { g.fillRect(x - 1.5, z - 4.5, 3, 9); g.fillRect(x - 4.5, z - 1.5, 9, 3); }
      else if (n.kind === 'pharmacy') { g.beginPath(); g.arc(x, z, 3.6, 0, Math.PI * 2); g.fill(); }
      else { g.beginPath(); g.moveTo(x, z - 4.5); g.lineTo(x + 4, z + 3.5); g.lineTo(x - 4, z + 3.5); g.closePath(); g.fill(); }
    });

    // camera wedge (where you are looking from)
    if (camPos) {
      const cx = X(camPos.x), cz = Z(camPos.z), a = Math.atan2(camDir.z, camDir.x);
      g.fillStyle = 'rgba(255,255,255,0.12)'; g.beginPath(); g.moveTo(cx, cz);
      g.arc(cx, cz, 26, a - 0.5, a + 0.5); g.closePath(); g.fill();
      g.fillStyle = 'rgba(255,255,255,0.7)'; g.beginPath(); g.arc(cx, cz, 2.2, 0, Math.PI * 2); g.fill();
    }

    // drone
    const dx = X(dronePos.x), dz = Z(dronePos.z), h = Math.atan2(Math.cos(droneYaw), Math.sin(droneYaw));
    g.save(); g.translate(dx, dz); g.rotate(h);
    g.fillStyle = st.crashed ? '#ff5d5d' : '#ffffff'; g.strokeStyle = '#3ee0b5'; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(8, 0); g.lineTo(-5, 5); g.lineTo(-2.5, 0); g.lineTo(-5, -5); g.closePath(); g.fill(); g.stroke();
    g.restore();
    g.restore();

    // bezel, north marker, scale bar
    g.strokeStyle = 'rgba(62,224,181,0.45)'; g.lineWidth = 1.5;
    g.beginPath(); g.arc(S / 2, S / 2, R, 0, Math.PI * 2); g.stroke();
    g.fillStyle = '#9fb6bd'; g.font = '600 10px system-ui, sans-serif'; g.textAlign = 'center';
    g.fillText('N', S / 2, 14);
    const bar = 2 * this.upk * k;   // 2 km
    g.strokeStyle = '#9fb6bd'; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(S - 16 - bar, S - 16); g.lineTo(S - 16, S - 16); g.stroke();
    g.fillText('2 km', S - 16 - bar / 2, S - 21);
  }

  resetTrail() { this.trail = []; }
}
