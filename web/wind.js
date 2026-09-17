import * as THREE from "three";

import { SIM_Y_TO_WORLD_Z } from "./boat.js";

export function createWind(scene) {
  const windGroup = new THREE.Group();
  scene.add(windGroup);

  const state = {
    dir: new THREE.Vector3(1, 0, 0),
    speedMs: 0.0,
  };

  initWindWisps(windGroup);

  function setFromPayload(wind) {
    if (!wind) return;
    const dirDeg = (wind.dir_deg || 0) % 360;
    const dirRad = (dirDeg * Math.PI) / 180;
    state.dir.set(Math.cos(dirRad), 0, SIM_Y_TO_WORLD_Z * Math.sin(dirRad)).normalize();
    state.speedMs = wind.speed || 0;
  }

  function advect(dt, t) {
    const advectSpeed = THREE.MathUtils.clamp(state.speedMs, 1.0, 15.0);
    const flowDir = state.dir.clone().normalize();
    const moveVec = flowDir.clone().multiplyScalar(advectSpeed * dt * 4.0);

    const resetRadius = 260;
    const spawnRadius = 220;

    windGroup.children.forEach((obj) => {
      const line = /** @type {THREE.Line} */ (obj);
      line.position.addScaledVector(
        moveVec,
        line.userData.speedFactor || 1.0,
      );
      const dist = Math.hypot(line.position.x, line.position.z);
      if (dist > resetRadius) {
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * spawnRadius;
        line.position.x = Math.cos(angle) * radius;
        line.position.z = Math.sin(angle) * radius;
      }
      line.position.y =
        1.8 +
        (line.userData.heightJitter || 0) +
        0.15 * Math.sin(t * 1.3 + line.position.x * 0.02);

      const baseDir = new THREE.Vector3(1, 0, 0);
      const quat = new THREE.Quaternion().setFromUnitVectors(
        baseDir,
        flowDir,
      );
      line.quaternion.copy(quat);
    });
  }

  return {
    setFromPayload,
    advect,
    getDir: () => state.dir.clone(),
    getSpeed: () => state.speedMs,
  };
}

function initWindWisps(windGroup) {
  const wispCount = 160;
  const areaRadius = 220;
  const wispLength = 4.0;

  const wispMaterial = new THREE.LineBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.35,
  });

  for (let i = 0; i < wispCount; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.random() * areaRadius;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const y = 1.6 + Math.random() * 1.0;

    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array([
      -wispLength / 2,
      0,
      0,
      wispLength / 2,
      0,
      0,
    ]);
    geom.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3),
    );

    const line = new THREE.Line(geom, wispMaterial);
    line.position.set(x, y, z);
    line.userData.speedFactor = 0.3 + Math.random() * 0.7;
    line.userData.heightJitter = (Math.random() - 0.5) * 0.4;
    windGroup.add(line);
  }
}

