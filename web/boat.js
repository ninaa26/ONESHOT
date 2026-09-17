import * as THREE from "three";
import {
  setSpeed,
  setHeading,
  setSailAngle,
  setRudderAngle,
  setSailForce,
} from "./hud.js";

export const WORLD_SCALE = 1.0;

// How the simulator's ground plane is laid into the scene.
//
// The sim works in a right-handed plane: +x east, +y north, angles counting
// counter-clockwise. The scene is y-up, so that plane has to go somewhere, and
// it goes into world x and z. Sending sim +y to world +z reverses orientation
// -- looking down on it, a counter-clockwise turn reads clockwise -- which is
// why the yaw used to be negated on the way in. It also meant that a top-down
// camera could be north-up or east-right but never both.
//
// Sending sim +y to world -z instead preserves orientation, so the overhead
// view is a chart: north up, east right, and headings that match the picture.
// Everything that puts a simulator quantity into the scene has to agree, so
// grep for this constant: the boat's position and yaw, its sail and rudder
// angles, the force arrows and the wind all go through it.
export const SIM_Y_TO_WORLD_Z = -1;

const SURFACE_ANIM_RESPONSE = 15.0; // larger = snappier easing
const RUDDER_MAX_RATE_RAD_S = THREE.MathUtils.degToRad(520.0);

// Which rig each sail model is drawn as. The ORC models are soft-sail VPP
// envelopes, so they get cloth, and `orc_w_jib` is a sloop and carries a jib as
// well. `basic` is a symmetric section at a geometric angle of attack and
// `hybrid` is an analytic wing/parachute, so both are drawn as a rigid wing --
// which is what WPI's boat actually flies.
const RIG_BY_SAIL_MODEL = {
  basic: "wing",
  hybrid: "wing",
  orc_main: "main",
  orc_w_jib: "sloop",
};
const DEFAULT_RIG = "main";

// Drawn if a config gives nothing to draw from. Every shipped boat states its
// hull, so these only ever cover a half-written config.
const FALLBACK = {
  hull: { L: 1.5, B: 0.6, T: 0.1 },
  keel: { area: 0.12, span: 0.7, x_pos: 0.0 },
  rudder: { area: 0.05, span: 0.45, x_pos: -0.5 },
  sail: { area: 1.5, span: 2.5, x_pos: 0.1 },
};

const MATERIALS = {
  hull: new THREE.MeshStandardMaterial({ color: 0xe6f1ff, metalness: 0.1, roughness: 0.35 }),
  foil: new THREE.MeshStandardMaterial({ color: 0x9ca3af, metalness: 0.1, roughness: 0.5 }),
  spar: new THREE.MeshStandardMaterial({ color: 0xd5dde8, metalness: 0.05, roughness: 0.4 }),
  cloth: new THREE.MeshStandardMaterial({
    color: 0xffffff,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.9,
  }),
  wing: new THREE.MeshStandardMaterial({ color: 0xdbe6f5, metalness: 0.25, roughness: 0.3 }),
};

function num(value, fallback) {
  return typeof value === "number" && isFinite(value) && value > 0 ? value : fallback;
}

// Half-thickness of a symmetric NACA 00xx section at chord fraction `xc`.
function nacaHalfThickness(xc, thickness) {
  return (
    5 *
    thickness *
    (0.2969 * Math.sqrt(xc) -
      0.126 * xc -
      0.3516 * xc * xc +
      0.2843 * xc * xc * xc -
      0.1015 * xc * xc * xc * xc)
  );
}

// A symmetric foil section, drawn in the shape plane (x aft, y lateral) and
// then laid upright so the extrusion depth becomes the span. Used for the keel,
// the rudder and the rigid wing: all three are symmetric sections, differing
// only in chord, span and which way they hang.
function foilGeometry(chord, span, thickness) {
  const shape = new THREE.Shape();
  const steps = 16;
  shape.moveTo(0, 0);
  for (let i = 1; i <= steps; i++) {
    const xc = i / steps;
    shape.lineTo(-xc * chord, nacaHalfThickness(xc, thickness) * chord);
  }
  for (let i = steps - 1; i >= 0; i--) {
    const xc = i / steps;
    shape.lineTo(-xc * chord, -nacaHalfThickness(xc, thickness) * chord);
  }
  shape.closePath();

  const geom = new THREE.ExtrudeGeometry(shape, { depth: span, bevelEnabled: false });
  geom.rotateX(-Math.PI / 2);
  return geom;
}

function triangleGeometry(points) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(points), 3));
  geom.setIndex([0, 1, 2]);
  geom.computeVertexNormals();
  return geom;
}

function replaceGeometry(mesh, geom) {
  mesh.geometry.dispose();
  mesh.geometry = geom;
}

export function createBoat(scene) {
  const boatGroup = new THREE.Group();
  scene.add(boatGroup);

  // Every mesh is rebuilt by setGeometry; these are placeholders so the groups
  // and the animation wiring exist before the first boat arrives.
  const hull = new THREE.Mesh(new THREE.BufferGeometry(), MATERIALS.hull);
  hull.castShadow = true;
  hull.receiveShadow = true;
  boatGroup.add(hull);

  const keel = new THREE.Mesh(new THREE.BufferGeometry(), MATERIALS.foil);
  keel.castShadow = true;
  boatGroup.add(keel);

  const rudderGroup = new THREE.Group();
  rudderGroup.userData.targetYaw = 0.0;
  boatGroup.add(rudderGroup);
  const rudder = new THREE.Mesh(new THREE.BufferGeometry(), MATERIALS.foil);
  rudder.castShadow = true;
  rudderGroup.add(rudder);

  const mast = new THREE.Mesh(new THREE.BufferGeometry(), MATERIALS.spar);
  boatGroup.add(mast);

  const sailGroup = new THREE.Group();
  sailGroup.userData.targetYaw = 0.0;
  boatGroup.add(sailGroup);

  const boom = new THREE.Mesh(new THREE.BufferGeometry(), MATERIALS.spar);
  sailGroup.add(boom);
  const mainSail = new THREE.Mesh(new THREE.BufferGeometry(), MATERIALS.cloth);
  mainSail.castShadow = true;
  sailGroup.add(mainSail);
  const wing = new THREE.Mesh(new THREE.BufferGeometry(), MATERIALS.wing);
  wing.castShadow = true;
  sailGroup.add(wing);

  // The jib pivots about its tack at the stemhead, not about the mast, so it
  // gets its own group. The simulator carries one sail angle, so it swings with
  // the main rather than being sheeted separately.
  const jibGroup = new THREE.Group();
  boatGroup.add(jibGroup);
  const jib = new THREE.Mesh(new THREE.BufferGeometry(), MATERIALS.cloth);
  jib.castShadow = true;
  jibGroup.add(jib);

  let rig = DEFAULT_RIG;

  // Build the boat at the size its config says it is. Lengths are metres in the
  // simulator's frame: +x forward of the centre of rotation, y up from the
  // waterline, so the hull floats at y = 0 and the foils hang below it.
  function setGeometry(geometry) {
    const g = geometry || {};
    const hullCfg = { ...FALLBACK.hull, ...(g.hull || {}) };
    const L = num(hullCfg.L, FALLBACK.hull.L);
    const B = num(hullCfg.B, FALLBACK.hull.B);
    const T = num(hullCfg.T, FALLBACK.hull.T);
    // Enough topside to read as a boat rather than a raft.
    const freeboard = Math.max(0.09, 0.9 * T);

    // Hull: a waterline-plan triangle, bow forward, extruded through the draft
    // and the freeboard and dropped so the waterline sits at y = 0.
    const plan = new THREE.Shape();
    plan.moveTo(L / 2, 0.0);
    plan.lineTo(-L / 2, B / 2);
    plan.lineTo(-L / 2, -B / 2);
    plan.closePath();
    const hullGeom = new THREE.ExtrudeGeometry(plan, { depth: T + freeboard, bevelEnabled: false });
    hullGeom.rotateX(-Math.PI / 2);
    hullGeom.translate(0, -T, 0);
    replaceGeometry(hull, hullGeom);

    // Foils: chord follows from the area the sim actually integrates over, so a
    // big keel looks big. Both hang from the hull bottom.
    const keelCfg = { ...FALLBACK.keel, ...(g.keel || {}) };
    const keelSpan = num(keelCfg.span, FALLBACK.keel.span);
    const keelChord = num(keelCfg.area, FALLBACK.keel.area) / keelSpan;
    replaceGeometry(keel, foilGeometry(keelChord, keelSpan, 0.12));
    keel.position.set(num(keelCfg.x_pos, 0.0) || keelCfg.x_pos || 0.0, -T - keelSpan, 0.0);
    keel.geometry.translate(keelChord / 2, 0, 0); // hang it about its mid-chord

    const rudderCfg = { ...FALLBACK.rudder, ...(g.rudder || {}) };
    const rudderSpan = num(rudderCfg.span, FALLBACK.rudder.span);
    const rudderChord = num(rudderCfg.area, FALLBACK.rudder.area) / rudderSpan;
    replaceGeometry(rudder, foilGeometry(rudderChord, rudderSpan, 0.12));
    rudder.position.set(0, -T - rudderSpan, 0);
    rudderGroup.position.set(rudderCfg.x_pos || 0.0, 0, 0);

    // Rig: `span` is the luff for a soft sail and the wing's height. A triangle
    // of that luff and this area needs a foot of 2A/b.
    const sailCfg = { ...FALLBACK.sail, ...(g.sail || {}) };
    const luff = num(sailCfg.span, FALLBACK.sail.span);
    const sailArea = num(sailCfg.area, FALLBACK.sail.area);
    const foot = (2.0 * sailArea) / luff;
    const mastX = sailCfg.x_pos || 0.0;
    const deck = freeboard;

    const mastGeom = new THREE.CylinderGeometry(0.012, 0.02, luff + 0.1, 12);
    replaceGeometry(mast, mastGeom);
    mast.position.set(mastX, deck + (luff + 0.1) / 2, 0.0);

    sailGroup.position.set(mastX, deck, 0.0);

    const boomGeom = new THREE.CylinderGeometry(0.012, 0.016, foot, 10);
    boomGeom.rotateZ(Math.PI / 2);
    boomGeom.translate(-foot / 2, 0, 0);
    replaceGeometry(boom, boomGeom);

    replaceGeometry(
      mainSail,
      triangleGeometry([0, 0, 0, -foot, 0, 0, 0, luff, 0]),
    );

    // The wing carries the same area on the same height, so its chord is A/b.
    const wingGeom = foilGeometry(sailArea / luff, luff, 0.18);
    replaceGeometry(wing, wingGeom);
    wing.position.set(0, 0, 0);

    // A jib on the fore-triangle: tack at the stemhead, head up the forestay.
    const jibLuff = 0.75 * luff;
    const jibFoot = 0.55 * foot;
    jibGroup.position.set(L / 2 - 0.02, deck, 0.0);
    replaceGeometry(
      jib,
      triangleGeometry([
        0, 0, 0,
        -jibFoot, 0.12 * jibLuff, 0,
        -(L / 2 - mastX), jibLuff, 0,
      ]),
    );

    setRig(rig);
  }

  // Which sails are on deck. The sim sends one sail model per build and the rig
  // follows it: cloth for the ORC envelopes, a wing for the section and
  // analytic models, and a jib only for the sloop.
  function setRig(sailModel) {
    rig = RIG_BY_SAIL_MODEL[sailModel] || rig || DEFAULT_RIG;
    const isWing = rig === "wing";
    mainSail.visible = !isWing;
    boom.visible = !isWing;
    wing.visible = isWing;
    jib.visible = rig === "sloop";
  }

  setGeometry(null);

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

  function animateControlSurfaces(dt) {
    const safeDt = Math.max(0.0, dt);
    const alpha = 1.0 - Math.exp(-SURFACE_ANIM_RESPONSE * safeDt);

    // Sail should follow sim state immediately (no frontend smoothing).
    sailGroup.rotation.y = sailGroup.userData.targetYaw;
    jibGroup.rotation.y = sailGroup.userData.targetYaw;

    const rudderErr = rudderGroup.userData.targetYaw - rudderGroup.rotation.y;
    const rudderStep = THREE.MathUtils.clamp(
      rudderErr * alpha,
      -RUDDER_MAX_RATE_RAD_S * safeDt,
      RUDDER_MAX_RATE_RAD_S * safeDt,
    );
    rudderGroup.rotation.y += rudderStep;
  }

  return {
    boatGroup,
    sailGroup,
    rudderGroup,
    jibGroup,
    updateTrace,
    animateControlSurfaces,
    setRig,
    setGeometry,
  };
}

export function updateBoatFromState(
  msg,
  boatGroup,
  sailGroup,
  rudderGroup,
) {
  const boat = msg.boat;
  const pos = boat.position;
  const heading = boat.heading;
  const velBody = boat.velocity_body;
  const wind = msg.wind;
  const sailForce = msg.sail_force;
  const sailState = msg.sail;
  const rudderState = msg.rudder;

  boatGroup.position.set(
    pos.x * WORLD_SCALE,
    0,
    SIM_Y_TO_WORLD_Z * pos.y * WORLD_SCALE,
  );

  const yawWorld = Math.atan2(heading.sin, heading.cos);
  // Orientation is preserved now, so the yaw goes in as it comes.
  const yawVis = -SIM_Y_TO_WORLD_Z * yawWorld;
  boatGroup.rotation.set(0, yawVis, 0);

  const speed = Math.sqrt(
    velBody.u * velBody.u + velBody.v * velBody.v,
  );
  setSpeed(speed);
  setHeading(heading.deg);
  // Prefer actual sail angle from simulation if provided; fallback to local helm value.
  const sailAngleDeg =
    sailState && typeof sailState.angle_deg === "number"
      ? sailState.angle_deg
      : 0.0;
  const rudderAngleDeg =
    rudderState && typeof rudderState.angle_deg === "number"
      ? rudderState.angle_deg
      : 0.0;
  setSailAngle(sailAngleDeg);
  setRudderAngle(rudderAngleDeg);
  if (sailForce) {
    setSailForce(sailForce.fx, sailForce.fy);
  }

  // Deflections are about the same axis as the yaw, so they carry the same sign.
  sailGroup.userData.targetYaw = THREE.MathUtils.degToRad(-SIM_Y_TO_WORLD_Z * sailAngleDeg);
  rudderGroup.userData.targetYaw = THREE.MathUtils.degToRad(-SIM_Y_TO_WORLD_Z * rudderAngleDeg);

  return {
    wind: wind || null,
    sailAngleDeg,
    rudderAngleDeg,
  };
}

