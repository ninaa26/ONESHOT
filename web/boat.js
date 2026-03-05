import * as THREE from "three";
import { setSpeed, setHeading, setSailAngle, setRudderAngle } from "./hud.js";

export const WORLD_SCALE = 1.0;

export function createBoat(scene) {
  const boatGroup = new THREE.Group();
  scene.add(boatGroup);

  // Hull: extruded triangle (bow +X, stern -X)
  const hullShape = new THREE.Shape();
  hullShape.moveTo(1.6, 0.0);
  hullShape.lineTo(-1.0, 0.6);
  hullShape.lineTo(-1.0, -0.6);
  hullShape.closePath();

  const hullGeom = new THREE.ExtrudeGeometry(hullShape, {
    depth: 0.4,
    bevelEnabled: false,
  });
  hullGeom.rotateX(-Math.PI / 2);
  hullGeom.translate(0, 0.0, 0);

  const hullMat = new THREE.MeshStandardMaterial({
    color: 0xe6f1ff,
    metalness: 0.1,
    roughness: 0.35,
  });
  const hull = new THREE.Mesh(hullGeom, hullMat);
  hull.castShadow = true;
  hull.receiveShadow = true;
  boatGroup.add(hull);

  // Rudder
  const rudderGroup = new THREE.Group();
  rudderGroup.position.set(-1.0, 0.25, 0.0);
  boatGroup.add(rudderGroup);

  const rudderGeom = new THREE.BoxGeometry(0.5, 0.5, 0.04);
  const rudderMat = new THREE.MeshStandardMaterial({
    color: 0x9ca3af,
    metalness: 0.1,
    roughness: 0.5,
  });
  const rudderMesh = new THREE.Mesh(rudderGeom, rudderMat);
  rudderMesh.castShadow = true;
  rudderMesh.receiveShadow = true;
  rudderMesh.position.set(-0.25, 0.0, 0.0);
  rudderGroup.add(rudderMesh);

  // Mast
  const mastGeom = new THREE.CylinderGeometry(0.03, 0.05, 2.6, 12);
  const mastMat = new THREE.MeshStandardMaterial({
    color: 0xd5dde8,
    metalness: 0.05,
    roughness: 0.4,
  });
  const mast = new THREE.Mesh(mastGeom, mastMat);
  mast.position.set(0.5, 1.3, 0.0);
  boatGroup.add(mast);

  // Sail group (boom + sail) anchored at mast
  const sailGroup = new THREE.Group();
  sailGroup.position.set(0.5, 1.3, 0.0);
  boatGroup.add(sailGroup);

  const boomGeom = new THREE.CylinderGeometry(0.025, 0.035, 1.6, 10);
  const boomMat = new THREE.MeshStandardMaterial({
    color: 0xf1f5ff,
    metalness: 0.2,
    roughness: 0.35,
  });
  const boom = new THREE.Mesh(boomGeom, boomMat);
  boom.rotation.z = Math.PI / 2;
  boom.position.set(-0.8, 0, 0);
  sailGroup.add(boom);

  const sailGeom = new THREE.BufferGeometry();
  const sailVertices = new Float32Array([
    0.0,
    0.0,
    0.0, // tack at mast
    -1.6,
    0.0,
    0.0, // clew
    0.0,
    2.2,
    0.0, // head
  ]);
  sailGeom.setAttribute(
    "position",
    new THREE.BufferAttribute(sailVertices, 3),
  );
  sailGeom.setIndex([0, 1, 2]);
  sailGeom.computeVertexNormals();

  const sailMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.9,
  });
  const sailMesh = new THREE.Mesh(sailGeom, sailMat);
  sailMesh.castShadow = true;
  sailGroup.add(sailMesh);

  // Wake / trace
  const tracePoints = [];
  const maxTracePoints = 200;
  const traceGeom = new THREE.BufferGeometry();
  const traceMat = new THREE.LineBasicMaterial({
    color: 0x88e5ff,
    linewidth: 2,
  });
  const traceLine = new THREE.Line(traceGeom, traceMat);
  scene.add(traceLine);

  function updateTrace() {
    const p = new THREE.Vector3();
    boatGroup.getWorldPosition(p);
    tracePoints.push(p.clone());
    if (tracePoints.length > maxTracePoints) {
      tracePoints.shift();
    }

    const positions = new Float32Array(tracePoints.length * 3);
    for (let i = 0; i < tracePoints.length; i++) {
      positions[i * 3 + 0] = tracePoints[i].x;
      positions[i * 3 + 1] = 0.01;
      positions[i * 3 + 2] = tracePoints[i].z;
    }
    traceGeom.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3),
    );
    traceGeom.setDrawRange(0, tracePoints.length);
  }

  return {
    boatGroup,
    sailGroup,
    rudderGroup,
    updateTrace,
  };
}

export function updateBoatFromState(
  msg,
  boatGroup,
  sailGroup,
  rudderGroup,
  sailDeg,
  rudderDeg,
) {
  const boat = msg.boat;
  const pos = boat.position;
  const heading = boat.heading;
  const velBody = boat.velocity_body;
  const wind = msg.wind;

  boatGroup.position.set(
    pos.x * WORLD_SCALE,
    0,
    pos.y * WORLD_SCALE,
  );

  const yawWorld = Math.atan2(heading.sin, heading.cos);
  const yawVis = -yawWorld;
  boatGroup.rotation.set(0, yawVis, 0);

  const speed = Math.sqrt(
    velBody.u * velBody.u + velBody.v * velBody.v,
  );
  setSpeed(speed);
  setHeading(heading.deg);
  setSailAngle(sailDeg);
  setRudderAngle(rudderDeg);

  sailGroup.rotation.y = THREE.MathUtils.degToRad(-sailDeg);
  rudderGroup.rotation.y = THREE.MathUtils.degToRad(rudderDeg);

  return wind || null;
}

