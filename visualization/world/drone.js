// Detailed quad-rotor delivery drone. Local +z is the nose, +y is up.
import * as THREE from 'three';
import { crossSignTex, glowTex, rotorBlurTex } from './textures.js';

export const GEAR_DROP = 0.17;      // distance from the drone origin to the bottom of the skids

export function createDrone(scene) {
  const group = new THREE.Group();
  group.rotation.order = 'YXZ';      // yaw, then pitch, then roll in the body frame
  const std = (o) => new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.2, ...o });
  const white = std({ color: 0xf4f7f8, roughness: 0.3 });
  const carbon = std({ color: 0x20272c, roughness: 0.5, metalness: 0.5 });
  const metal = std({ color: 0x9aa6ac, roughness: 0.35, metalness: 0.8 });
  const add = (geo, mat, x = 0, y = 0, z = 0, parent = group) => {
    const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = true; parent.add(m); return m;
  };

  // fuselage
  const body = add(new THREE.CapsuleGeometry(0.1, 0.24, 8, 18).rotateX(Math.PI / 2), white);
  body.scale.set(1.25, 0.72, 1);
  add(new THREE.SphereGeometry(0.085, 18, 10, 0, Math.PI * 2, 0, Math.PI / 2), std({ color: 0x0f1a22, roughness: 0.08, metalness: 0.6 }), 0, 0.035, 0.08);
  add(new THREE.BoxGeometry(0.262, 0.018, 0.3), std({ color: 0x1fc3a2, emissive: 0x0c6b58, emissiveIntensity: 0.6 }), 0, -0.01, 0);
  const decal = new THREE.Mesh(new THREE.PlaneGeometry(0.11, 0.11).rotateX(-Math.PI / 2),
    std({ map: crossSignTex('#e5484d'), transparent: true, roughness: 0.5 }));
  decal.position.set(0, 0.073, -0.07); group.add(decal);

  // arms, motors, propellers
  const ARM = 0.36, rotors = [];
  const blurTex = rotorBlurTex();
  for (const [sx, sz] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
    const ang = Math.atan2(sx, sz);
    const arm = add(new THREE.CylinderGeometry(0.016, 0.02, ARM, 8).rotateX(Math.PI / 2), carbon);
    arm.position.set(sx * ARM / 2 * Math.SQRT1_2, 0.01, sz * ARM / 2 * Math.SQRT1_2); arm.rotation.y = ang;
    const mx = sx * ARM * Math.SQRT1_2, mz = sz * ARM * Math.SQRT1_2;
    add(new THREE.CylinderGeometry(0.034, 0.03, 0.055, 14), metal, mx, 0.03, mz);
    add(new THREE.CylinderGeometry(0.012, 0.012, 0.02, 8), carbon, mx, 0.067, mz);
    const prop = new THREE.Group(); prop.position.set(mx, 0.075, mz); group.add(prop);
    const blades = add(new THREE.BoxGeometry(0.34, 0.004, 0.03), carbon, 0, 0, 0, prop);
    const blur = new THREE.Mesh(new THREE.CircleGeometry(0.18, 36).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ map: blurTex, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }));
    blur.position.set(mx, 0.078, mz); group.add(blur);
    rotors.push({ prop, blades, blur, dir: sx * sz });
  }

  // landing skids
  for (const sx of [-1, 1]) {
    add(new THREE.CylinderGeometry(0.008, 0.008, 0.34, 6).rotateX(Math.PI / 2), carbon, sx * 0.1, -GEAR_DROP + 0.008, 0);
    for (const sz of [-1, 1]) {
      const strut = add(new THREE.CylinderGeometry(0.007, 0.007, 0.13, 6), carbon, sx * 0.085, -0.1, sz * 0.09);
      strut.rotation.z = sx * 0.25;
    }
  }

  // cargo bay (packages are added by the viewer)
  const cargo = new THREE.Group(); cargo.position.y = -0.1; group.add(cargo);

  // navigation lights: red port (left), green starboard (right), white tail strobe, red beacon
  const glow = glowTex();
  const light = (color, x, y, z, size) => {
    const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.014, 8, 6), new THREE.MeshBasicMaterial({ color, toneMapped: false }));
    bulb.position.set(x, y, z); group.add(bulb);
    const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, color, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }));
    halo.scale.setScalar(size); halo.position.copy(bulb.position); group.add(halo);
    return { bulb, halo };
  };
  const port = light(0xff2a2a, ARM * Math.SQRT1_2 + 0.03, 0.03, ARM * Math.SQRT1_2, 0.16);
  const starboard = light(0x2aff6a, -ARM * Math.SQRT1_2 - 0.03, 0.03, ARM * Math.SQRT1_2, 0.16);
  const strobe = light(0xffffff, 0, 0.0, -0.22, 0.32);
  const beacon = light(0xff3b3b, 0, 0.08, -0.02, 0.22);

  // downward delivery spotlight + visible light cone
  const spot = new THREE.SpotLight(0xfff2d8, 0, 14, 0.42, 0.65, 1.2);
  spot.position.set(0, -0.08, 0.05); group.add(spot);
  const spotTarget = new THREE.Object3D(); spotTarget.position.set(0, -6, 0.6); group.add(spotTarget); spot.target = spotTarget;
  const coneMat = new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(0xfff0cc) }, uOpacity: { value: 0 }, uLen: { value: 4 } },
    vertexShader: `varying float vY; varying vec3 vN; varying vec3 vV;
      void main(){ vY = position.y; vec4 mv = modelViewMatrix * vec4(position,1.); vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
    fragmentShader: `uniform vec3 uColor; uniform float uOpacity; uniform float uLen; varying float vY; varying vec3 vN; varying vec3 vV;
      void main(){ float fade = clamp(1.0 + vY/uLen, 0.0, 1.0); float rim = pow(abs(dot(vN, vV)), 1.5);
        gl_FragColor = vec4(uColor, uOpacity * fade * fade * rim);
        #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
  });
  const cone = new THREE.Mesh(new THREE.ConeGeometry(1.0, 4, 32, 1, true).translate(0, -2, 0), coneMat);
  cone.position.set(0, -0.08, 0); cone.renderOrder = 3; group.add(cone);

  scene.add(group);

  const state = { spin: 0 };
  /** spin: rotor rad/s; flying: in flight; crashed; darkness 0..1; spotlight 0..1 */
  function update(dt, now, { spin, crashed, darkness, spotlight, coneLength }) {
    state.spin = THREE.MathUtils.lerp(state.spin, spin, 1 - Math.exp(-dt * 3));
    const blurAmt = THREE.MathUtils.smoothstep(state.spin, 6, 30);
    for (const r of rotors) {
      r.prop.rotation.y += r.dir * state.spin * dt;
      r.blur.rotation.y = r.prop.rotation.y * 0.1;
      r.blur.material.opacity = 0.85 * blurAmt;
      r.blades.visible = blurAmt < 0.95;
    }
    const t = now / 1000, on = !crashed;
    const strobeOn = on && (t % 1.2 < 0.06 || (t % 1.2 > 0.16 && t % 1.2 < 0.22));
    strobe.bulb.visible = strobe.halo.visible = strobeOn;
    beacon.halo.material.opacity = on ? 0.4 + 0.6 * Math.max(0, Math.sin(t * 5)) : 0;
    port.halo.visible = starboard.halo.visible = on;
    const s = on ? spotlight : 0;
    spot.intensity = s * (6 + 18 * darkness);
    coneMat.uniforms.uOpacity.value = s * (0.06 + 0.16 * darkness);
    cone.scale.set(coneLength * 0.42, coneLength / 4, coneLength * 0.42);
    cone.visible = s > 0.02;
  }

  return { group, cargo, update };
}
