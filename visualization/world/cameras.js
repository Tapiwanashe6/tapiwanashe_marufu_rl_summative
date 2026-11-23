// Camera rig with four modes: chase (default), belly/FPV, street-level walker
// and free orbit. Cameras only *observe* the replay; they never change it.
import * as THREE from 'three';
import { LINES } from './city.js';

export const MODES = ['chase', 'fpv', 'street', 'orbit'];
export const MODE_LABEL = { chase: 'Chase cam', fpv: 'Belly cam', street: 'Street level', orbit: 'Free orbit' };
export const MODE_HINT = {
  chase: 'Scroll to zoom · C to switch camera',
  fpv: 'Looking down from under the drone · C to switch camera',
  street: 'WASD walk · Shift run · drag to look · "Follow drone" tracks it · C to switch',
  orbit: 'Drag to rotate · right-drag to pan · scroll to zoom · C to switch',
};
const FOV = { chase: 60, fpv: 78, street: 68, orbit: 50 };
const EYE = 0.24;

export class CameraRig {
  constructor(camera, dom, controls, city) {
    this.camera = camera; this.controls = controls; this.city = city;
    this.mode = 'chase'; this.blend = 1; this.zoom = 1;
    this.look = new THREE.Vector3();
    this.walker = { pos: new THREE.Vector3(1.5, EYE, 3.4), yaw: Math.PI, pitch: 0.25, placed: false };
    this.keys = new Set();
    this.fpvPitch = 0.55;

    addEventListener('keydown', (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      const k = e.code;
      if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ShiftLeft', 'ShiftRight'].includes(k)) this.keys.add(k);
    });
    addEventListener('keyup', (e) => this.keys.delete(e.code));
    addEventListener('blur', () => this.keys.clear());

    let drag = null;
    dom.addEventListener('pointerdown', (e) => { if (this.mode === 'street') { drag = { x: e.clientX, y: e.clientY }; dom.setPointerCapture(e.pointerId); } });
    dom.addEventListener('pointermove', (e) => {
      if (!drag || this.mode !== 'street') return;
      this.walker.yaw -= (e.clientX - drag.x) * 0.0045;
      this.walker.pitch = THREE.MathUtils.clamp(this.walker.pitch - (e.clientY - drag.y) * 0.0045, -1.2, 1.35);
      drag = { x: e.clientX, y: e.clientY };
      this.manualLook = performance.now();
    });
    dom.addEventListener('pointerup', () => { drag = null; });
    dom.addEventListener('wheel', (e) => {
      if (this.mode === 'chase') { this.zoom = THREE.MathUtils.clamp(this.zoom * (e.deltaY > 0 ? 1.1 : 0.9), 0.55, 3.5); e.preventDefault(); }
    }, { passive: false });
  }

  setMode(mode, ctx) {
    if (!MODES.includes(mode)) return;
    this.mode = mode; this.blend = 0;
    this.controls.enabled = false;
    if (mode === 'street' && ctx) this.placeWalker(ctx.dronePos);
    if (mode === 'orbit' && ctx) {
      this.controls.target.set(ctx.dronePos.x, 0, ctx.dronePos.z);
      this.orbitFrom = this.camera.position.clone();
      this.orbitTo = ctx.dronePos.clone().add(new THREE.Vector3(11, 13, 15));
    }
  }

  next(ctx) { this.setMode(MODES[(MODES.indexOf(this.mode) + 1) % MODES.length], ctx); }

  /** Put the walker on open pavement a few units from a point, looking at it. */
  placeWalker(p) {
    const w = this.walker;
    const onRoad = (x, z) => LINES.some((l) => Math.abs(x - l) < 0.62 || Math.abs(z - l) < 0.62);
    for (let r = 2.6; r < 10; r += 0.35) {
      for (let a = 0.3; a < Math.PI * 2 + 0.3; a += Math.PI / 10) {
        const x = p.x + Math.cos(a) * r, z = p.z + Math.sin(a) * r;
        if (onRoad(x, z) || [[0, 0], [0.25, 0], [-0.25, 0], [0, 0.25], [0, -0.25]].some(([dx, dz]) => this.city.isSolid(x + dx, z + dz))) continue;
        const dx = p.x - x, dz = p.z - z;
        w.pos.set(x, EYE, z); w.yaw = Math.atan2(-dx, -dz);
        w.pitch = THREE.MathUtils.clamp(Math.atan2(p.y - EYE, Math.hypot(dx, dz)), 0.1, 1.0); w.placed = true; return;
      }
    }
  }

  update(dt, ctx) {
    const cam = this.camera, k = this.blend < 1 ? 1 - Math.exp(-dt * 5) : null;
    this.blend = Math.min(1, this.blend + dt * 1.4);
    cam.fov += (FOV[this.mode] - cam.fov) * (1 - Math.exp(-dt * 4));
    cam.near = this.mode === 'orbit' ? 0.1 : 0.02;
    cam.updateProjectionMatrix();
    const fwd = new THREE.Vector3(Math.sin(ctx.yaw), 0, Math.cos(ctx.yaw));

    if (this.mode === 'chase') {
      const dist = 1.9 * this.zoom, up = 0.62 * this.zoom + 0.15;
      const want = ctx.dronePos.clone().addScaledVector(fwd, -dist);
      want.y += up;
      want.x += Math.sin(ctx.t * 0.7) * 0.04; want.y += Math.sin(ctx.t * 1.1) * 0.03;
      want.y = Math.max(want.y, this.city.heightAround(want.x, want.z, 0.3) + 0.35, 0.3);
      cam.position.lerp(want, k ?? 1 - Math.exp(-dt * 3.5));
      const target = ctx.dronePos.clone().addScaledVector(fwd, 1.2); target.y -= 0.15;
      this.look.lerp(target, k ?? 1 - Math.exp(-dt * 6));
      cam.up.set(0, 1, 0); cam.lookAt(this.look);
      cam.rotateZ(-ctx.roll * 0.45);
    } else if (this.mode === 'fpv') {
      const want = ctx.dronePos.clone().addScaledVector(fwd, 0.12); want.y -= 0.2;
      const pitchTarget = ctx.dropping ? 1.35 : ctx.flying ? 0.5 : 0.85;
      this.fpvPitch += (pitchTarget - this.fpvPitch) * (1 - Math.exp(-dt * 2.5));
      cam.position.lerp(want, k ?? 1);
      const dir = fwd.clone().multiplyScalar(Math.cos(this.fpvPitch)); dir.y = -Math.sin(this.fpvPitch);
      const target = cam.position.clone().add(dir);
      if (k !== null) this.look.lerp(target, k); else this.look.copy(target);
      cam.up.set(0, 1, 0); cam.lookAt(this.look);
      cam.rotateZ(-ctx.roll * 0.8);
    } else if (this.mode === 'street') {
      const w = this.walker;
      const speed = (this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? 3.2 : 1.1) * dt;
      const f = new THREE.Vector3(-Math.sin(w.yaw), 0, -Math.cos(w.yaw)), rgt = new THREE.Vector3(-f.z, 0, f.x);
      const move = new THREE.Vector3();
      if (this.keys.has('KeyW')) move.add(f); if (this.keys.has('KeyS')) move.sub(f);
      if (this.keys.has('KeyD')) move.add(rgt); if (this.keys.has('KeyA')) move.sub(rgt);
      if (move.lengthSq() > 0) {
        move.normalize().multiplyScalar(speed);
        const nx = w.pos.x + move.x, nz = w.pos.z + move.z;
        if (!this.city.isSolid(nx, w.pos.z)) w.pos.x = nx;       // slide along walls
        if (!this.city.isSolid(w.pos.x, nz)) w.pos.z = nz;
        const r = Math.hypot(w.pos.x, w.pos.z); if (r > 60) w.pos.multiplyScalar(60 / r);
      }
      if (ctx.follow && (!this.manualLook || performance.now() - this.manualLook > 1500)) {
        const to = ctx.dronePos.clone().sub(w.pos);
        const yawT = Math.atan2(-to.x, -to.z), pitchT = Math.atan2(to.y - EYE, Math.hypot(to.x, to.z));
        let dy = yawT - w.yaw; dy = ((dy + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;
        w.yaw += dy * (1 - Math.exp(-dt * 3)); w.pitch += (pitchT - w.pitch) * (1 - Math.exp(-dt * 3));
      }
      const bob = move.lengthSq() > 0 ? Math.sin(ctx.t * 9) * 0.006 : 0;
      const want = new THREE.Vector3(w.pos.x, EYE + 0.045 + bob, w.pos.z);
      cam.position.lerp(want, k ?? 1);
      const dir = new THREE.Vector3(-Math.sin(w.yaw) * Math.cos(w.pitch), Math.sin(w.pitch), -Math.cos(w.yaw) * Math.cos(w.pitch));
      const target = cam.position.clone().add(dir);
      if (k !== null) this.look.lerp(target, k); else this.look.copy(target);
      cam.up.set(0, 1, 0); cam.lookAt(this.look);
    } else {
      if (this.blend < 1 && this.orbitTo) {
        cam.position.lerp(this.orbitTo, 1 - Math.exp(-dt * 4));
        cam.up.set(0, 1, 0); cam.lookAt(this.controls.target);
      } else {
        this.controls.enabled = true;
        if (ctx.follow) this.controls.target.lerp(new THREE.Vector3(ctx.dronePos.x, 0, ctx.dronePos.z), 0.06);
        this.controls.update();
      }
    }
  }

  /** Where the sun's shadow frustum should be centred and how large it should be. */
  shadowFocus(dronePos) {
    if (this.mode === 'orbit') return [this.controls.target, 34];
    if (this.mode === 'street') return [this.camera.position, 18];
    return [dronePos, 16];
  }
}
