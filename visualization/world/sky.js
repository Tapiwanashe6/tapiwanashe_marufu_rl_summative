// Sky dome, sun and fill light, cloud layers, rain and lightning. The weather
// *state* comes from the replay (st.weather); this module only renders it and
// blends smoothly between calm, windy and storm.
import * as THREE from 'three';
import { cloudTex, mulberry32 } from './textures.js';

export const WEATHER = {
  calm:  { top: 0x3f86d0, horizon: 0xcfe4ef, ground: 0x8fa58a, fog: [70, 210], sun: 3.0, hemi: 0.9, env: 0.55, dark: 0.0, wind: 0.15, wet: 0, rain: 0, cloud: 0.35, cloudColor: 0xffffff, cloudY: 26, exposure: 1.0, note: 'Nominal flight energy' },
  windy: { top: 0x5f86a8, horizon: 0xb7c7cf, ground: 0x7e8f80, fog: [50, 150], sun: 1.7, hemi: 0.75, env: 0.45, dark: 0.35, wind: 0.6, wet: 0.15, rain: 0, cloud: 0.7, cloudColor: 0xe1e6e9, cloudY: 22, exposure: 0.95, note: 'Energy ×1.3 · speed ×0.85' },
  storm: { top: 0x2c353e, horizon: 0x56626b, ground: 0x353d38, fog: [22, 85], sun: 0.2, hemi: 0.4, env: 0.22, dark: 0.92, wind: 1.0, wet: 1, rain: 1, cloud: 1.0, cloudColor: 0x56606a, cloudY: 15, exposure: 0.86, note: 'Energy ×1.8 · speed ×0.70' },
};
const SUN_DIR = new THREE.Vector3(-0.55, 0.62, 0.42).normalize();

export class Sky {
  constructor(scene, renderer) {
    this.scene = scene; this.renderer = renderer;
    const rng = mulberry32(99);
    const w = WEATHER.calm;
    this.cur = { top: new THREE.Color(w.top), horizon: new THREE.Color(w.horizon), ground: new THREE.Color(w.ground),
      cloudColor: new THREE.Color(w.cloudColor), fogNear: w.fog[0], fogFar: w.fog[1], sun: w.sun, hemi: w.hemi, env: w.env,
      dark: 0, wind: w.wind, wet: 0, rain: 0, cloud: w.cloud, cloudY: w.cloudY, exposure: 1 };
    this.flash = 0; this.nextBolt = 4;

    // dome
    this.domeMat = new THREE.ShaderMaterial({
      uniforms: { top: { value: this.cur.top }, horizon: { value: this.cur.horizon }, ground: { value: this.cur.ground },
        sunDir: { value: SUN_DIR }, sunStrength: { value: 1 }, flash: { value: 0 } },
      vertexShader: `varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.); }`,
      fragmentShader: `uniform vec3 top, horizon, ground, sunDir; uniform float sunStrength, flash; varying vec3 vDir;
        void main(){ float h = vDir.y; vec3 col = h > 0. ? mix(horizon, top, pow(clamp(h,0.,1.), 0.55)) : mix(horizon, ground, clamp(-h*6., 0., 1.));
          float s = max(dot(vDir, sunDir), 0.); col += vec3(1.,0.93,0.8) * (pow(s, 900.) * 3. + pow(s, 12.) * 0.25) * sunStrength;
          col = mix(col, vec3(0.85,0.9,1.0), flash); gl_FragColor = vec4(col, 1.);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
      side: THREE.BackSide, depthWrite: false,
    });
    this.dome = new THREE.Mesh(new THREE.SphereGeometry(380, 40, 20), this.domeMat);
    this.dome.renderOrder = -10; this.dome.frustumCulled = false; scene.add(this.dome);
    scene.background = null;
    scene.fog = new THREE.Fog(this.cur.horizon.clone(), w.fog[0], w.fog[1]);

    // lights
    this.hemi = new THREE.HemisphereLight(0xdff1ff, 0x4a5a40, w.hemi); scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xfff0d6, w.sun);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0004; this.sun.shadow.normalBias = 0.02;
    this.sun.shadow.camera.near = 1; this.sun.shadow.camera.far = 140;
    scene.add(this.sun, this.sun.target);
    this.setShadowExtent(30);

    // clouds: soft sprite puffs in clusters
    const ctex = cloudTex(rng);
    this.cloudMat = new THREE.SpriteMaterial({ map: ctex, color: 0xffffff, transparent: true, opacity: 0.8, depthWrite: false, fog: true });
    this.clusters = [];
    for (let i = 0; i < 26; i++) {
      const g = new THREE.Group();
      const n = 4 + Math.floor(rng() * 5);
      for (let k = 0; k < n; k++) {
        const s = new THREE.Sprite(this.cloudMat);
        const sc = 9 + rng() * 10; s.scale.set(sc, sc * 0.55, 1);
        s.position.set((rng() - 0.5) * 14, (rng() - 0.5) * 2, (rng() - 0.5) * 8); g.add(s);
      }
      g.position.set((rng() - 0.5) * 200, 0, (rng() - 0.5) * 200);
      g.userData = { rank: rng(), dy: rng() * 4 };
      scene.add(g); this.clusters.push(g);
    }

    // rain: wind-angled streaks around the camera
    this.RAIN_N = 2600; this.rng = rng;
    this.rainPos = new Float32Array(this.RAIN_N * 6);
    this.drops = new Float32Array(this.RAIN_N * 3);
    for (let i = 0; i < this.RAIN_N; i++) { this.drops[i * 3] = (rng() - 0.5) * 50; this.drops[i * 3 + 1] = rng() * 24; this.drops[i * 3 + 2] = (rng() - 0.5) * 50; }
    const rg = new THREE.BufferGeometry(); rg.setAttribute('position', new THREE.BufferAttribute(this.rainPos, 3));
    this.rain = new THREE.LineSegments(rg, new THREE.LineBasicMaterial({ color: 0xaecbe0, transparent: true, opacity: 0, depthWrite: false }));
    this.rain.frustumCulled = false; scene.add(this.rain);

    // lightning bolt
    this.bolt = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xeaf2ff, transparent: true, opacity: 0, toneMapped: false }));
    this.bolt.frustumCulled = false; scene.add(this.bolt);
  }

  /** Jump straight to a weather state (episode load / seek) instead of blending. */
  snap(weather) {
    const w = WEATHER[weather] ?? WEATHER.calm, c = this.cur;
    c.top.set(w.top); c.horizon.set(w.horizon); c.ground.set(w.ground); c.cloudColor.set(w.cloudColor);
    for (const key of ['sun', 'hemi', 'env', 'dark', 'wind', 'wet', 'rain', 'cloud', 'cloudY', 'exposure']) c[key] = w[key];
    c.fogNear = w.fog[0]; c.fogFar = w.fog[1];
  }

  setShadowExtent(r) {
    const c = this.sun.shadow.camera;
    if (c.right === r) return;
    Object.assign(c, { left: -r, right: r, top: r, bottom: -r }); c.updateProjectionMatrix();
  }

  /** Returns the blended visual environment used by the rest of the scene. */
  update(dt, now, weather, camera, focus) {
    const w = WEATHER[weather] ?? WEATHER.calm, c = this.cur, k = 1 - Math.exp(-dt * 1.2);
    c.top.lerp(new THREE.Color(w.top), k); c.horizon.lerp(new THREE.Color(w.horizon), k); c.ground.lerp(new THREE.Color(w.ground), k);
    c.cloudColor.lerp(new THREE.Color(w.cloudColor), k);
    for (const key of ['sun', 'hemi', 'env', 'dark', 'wind', 'wet', 'rain', 'cloud', 'cloudY', 'exposure']) c[key] += (w[key] - c[key]) * k;
    c.fogNear += (w.fog[0] - c.fogNear) * k; c.fogFar += (w.fog[1] - c.fogFar) * k;
    const t = now / 1000;

    // lightning
    this.flash = Math.max(0, this.flash - dt * 4);
    this.bolt.material.opacity = Math.max(0, this.bolt.material.opacity - dt * 6);
    if (c.rain > 0.6) {
      this.nextBolt -= dt;
      if (this.nextBolt <= 0) {
        this.nextBolt = 3 + this.rng() * 6; this.flash = 0.9;
        const a = this.rng() * Math.PI * 2, r = 40 + this.rng() * 40;
        const pts = []; let x = camera.position.x + Math.cos(a) * r, z = camera.position.z + Math.sin(a) * r;
        for (let y = c.cloudY + 4; y > 0; y -= 2.5) { pts.push(new THREE.Vector3(x, y, z)); x += (this.rng() - 0.5) * 3; z += (this.rng() - 0.5) * 3; }
        this.bolt.geometry.setFromPoints(pts); this.bolt.material.opacity = 1;
      }
    }

    this.dome.position.copy(camera.position);
    this.domeMat.uniforms.sunStrength.value = Math.max(0, 1 - c.dark * 1.1);
    this.domeMat.uniforms.flash.value = this.flash * 0.6;
    this.scene.fog.color.copy(c.horizon).lerp(new THREE.Color(0xdde8ff), this.flash * 0.4);
    this.scene.fog.near = c.fogNear; this.scene.fog.far = c.fogFar;
    this.hemi.intensity = c.hemi + this.flash * 2.2;
    this.hemi.groundColor.copy(c.ground);
    this.sun.intensity = c.sun;
    this.scene.environmentIntensity = c.env;
    this.renderer.toneMappingExposure = c.exposure;

    // the sun (and its shadow frustum) follows the point of interest
    this.sun.target.position.copy(focus);
    this.sun.position.copy(focus).addScaledVector(SUN_DIR, 70);

    // clouds
    const drift = 0.6 + c.wind * 5;
    this.cloudMat.color.copy(c.cloudColor);
    this.cloudMat.opacity = 0.55 + 0.35 * c.cloud;
    for (const g of this.clusters) {
      g.visible = g.userData.rank < c.cloud;
      g.position.x += dt * drift; if (g.position.x > 110) g.position.x -= 220;
      g.position.y = c.cloudY + g.userData.dy;
    }

    // rain
    this.rain.material.opacity = 0.55 * c.rain;
    this.rain.visible = c.rain > 0.02;
    if (this.rain.visible) {
      const vx = 4 + c.wind * 7, vy = -26, len = 0.03;
      const cx = camera.position.x, cz = camera.position.z, d = this.drops, p = this.rainPos;
      for (let i = 0; i < this.RAIN_N; i++) {
        d[i * 3] += vx * dt; d[i * 3 + 1] += vy * dt;
        if (d[i * 3 + 1] < 0) { d[i * 3 + 1] += 24; d[i * 3] = (this.rng() - 0.5) * 50; d[i * 3 + 2] = (this.rng() - 0.5) * 50; }
        if (d[i * 3] > 25) d[i * 3] -= 50;
        const x = cx + d[i * 3], y = d[i * 3 + 1], z = cz + d[i * 3 + 2];
        p[i * 6] = x; p[i * 6 + 1] = y; p[i * 6 + 2] = z;
        p[i * 6 + 3] = x - vx * len; p[i * 6 + 4] = y - vy * len; p[i * 6 + 5] = z;
      }
      this.rain.geometry.attributes.position.needsUpdate = true;
    }
    return { darkness: Math.max(0, c.dark - this.flash * 0.5), wind: c.wind, wet: c.wet, rain: c.rain, t };
  }
}
