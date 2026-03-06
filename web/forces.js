import * as THREE from "three";

/** Scale factor: N -> m arrow length. 50 N ~ 1 m. */
const FORCE_SCALE = 0.02;
/** Max arrow length in meters. */
const MAX_ARROW_LENGTH = 3.0;
/** Min arrow length to show (hide tiny forces). */
const MIN_ARROW_LENGTH = 0.05;

const COMPONENT_COLORS = {
  hull: 0x888888,
  keel: 0x4488ff,
  rudder: 0xff8844,
  sail: 0x88ffff,
  total: 0xffff44,
};

const COMPONENT_ORDER = ["hull", "keel", "rudder", "sail", "total"];

/**
 * Create a force arrow (line + cone) for a component.
 * @param {number} color - Hex color
 * @returns {{ group: THREE.Group, setArrow: (dir: THREE.Vector3, length: number) => void, setVisible: (v: boolean) => void }}
 */
function createArrow(color) {
  const group = new THREE.Group();

  const lineGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(1, 0, 0),
  ]);
  const lineMat = new THREE.LineBasicMaterial({ color });
  const line = new THREE.Line(lineGeom, lineMat);
  group.add(line);

  const coneGeom = new THREE.ConeGeometry(0.08, 0.2, 8);
  const coneMat = new THREE.MeshBasicMaterial({ color });
  const cone = new THREE.Mesh(coneGeom, coneMat);
  cone.position.x = 1;
  cone.rotation.z = -Math.PI / 2;
  group.add(cone);

  const dir = new THREE.Vector3(1, 0, 0);
  const coneY = new THREE.Vector3(0, 1, 0);

  function setArrow(direction, length) {
    if (length < 1e-6) return;
    dir.copy(direction).normalize();
    const shaftLength = Math.max(0, length - 0.2);
    line.geometry.dispose();
    line.geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(shaftLength * dir.x, 0, shaftLength * dir.z),
    ]);
    cone.position.set(shaftLength * dir.x, 0, shaftLength * dir.z);
    cone.quaternion.setFromUnitVectors(coneY, dir.clone().set(dir.x, 0, dir.z).normalize());
  }

  function setVisible(v) {
    group.visible = v;
  }

  return { group, setArrow, setVisible };
}

/**
 * Create force arrows group as child of boatGroup.
 * @param {THREE.Group} boatGroup
 * @returns {{ group: THREE.Group, update: (forces: object, visible: boolean) => void }}
 */
export function createForceArrows(boatGroup) {
  const group = new THREE.Group();
  group.position.set(0, 0.3, 0);
  group.visible = false;

  const arrows = {};
  for (const name of COMPONENT_ORDER) {
    const color = COMPONENT_COLORS[name];
    const arrow = createArrow(color);
    arrows[name] = arrow;
    group.add(arrow.group);
  }

  function update(forces, visible) {
    group.visible = visible && forces && Object.keys(forces).length > 0;
    if (!group.visible || !forces) return;

    for (const name of COMPONENT_ORDER) {
      const comp = forces[name];
      const arrow = arrows[name];
      if (!comp || typeof comp.fx !== "number" || typeof comp.fy !== "number") {
        arrow.setVisible(false);
        continue;
      }

      const fx = comp.fx;
      const fy = comp.fy;
      const magnitude = Math.sqrt(fx * fx + fy * fy);
      let length = magnitude * FORCE_SCALE;
      if (length < MIN_ARROW_LENGTH) {
        arrow.setVisible(false);
        continue;
      }
      length = Math.min(length, MAX_ARROW_LENGTH);
      arrow.setVisible(true);

      // Boat frame: +x forward, +y starboard. Three.js boat: +X forward, +Z starboard.
      const dir = new THREE.Vector3(fx, 0, fy).normalize();
      arrow.setArrow(dir, length);
    }
  }

  boatGroup.add(group);
  return { group, update };
}
