import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { createScene, handleResize } from "./scene.js";
import { createWind } from "./wind.js";
import { createBoat, updateBoatFromState } from "./boat.js";
import { createForceArrows } from "./forces.js";
import { setConnectionStatus, setControlMode, updateForcesChart } from "./hud.js";

// --- Scene & environment ------------------------------------------------

const app = document.getElementById("app");
const { scene, camera, renderer } = createScene(app);

// --- Wind ---------------------------------------------------------------

const wind = createWind(scene);

// --- Boat ---------------------------------------------------------------

const { boatGroup, sailGroup, rudderGroup, updateTrace, animateControlSurfaces } = createBoat(scene);
const waypointMarker = new THREE.Mesh(
  new THREE.SphereGeometry(0.28, 20, 20),
  new THREE.MeshStandardMaterial({
    color: 0xffff66,
    emissive: 0xffdd33,
    emissiveIntensity: 1.8,
    roughness: 0.2,
    metalness: 0.0,
  }),
);
waypointMarker.position.y = 0.4;
waypointMarker.visible = false;
scene.add(waypointMarker);

const { update: updateForceArrows } = createForceArrows(boatGroup);
let showForces = false;
let lastForces = null;

// Follow camera: stays directly behind the boat, boat faces forward in view
const FOLLOW_DISTANCE = 15;
const FOLLOW_HEIGHT = 12;

const boatPos = new THREE.Vector3();
const forward = new THREE.Vector3();

let cameraFollowMode = true;
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

function updateFollowCamera() {
  boatGroup.getWorldPosition(boatPos);
  // Boat bow is along local +X; getWorldDirection returns Z. Use quaternion to get X axis.
  forward.set(1, 0, 0);
  forward.applyQuaternion(boatGroup.quaternion);
  forward.y = 0;
  forward.normalize();

  camera.position.copy(boatPos).addScaledVector(forward, -FOLLOW_DISTANCE);
  camera.position.y = FOLLOW_HEIGHT;

  camera.lookAt(boatPos.x, 0, boatPos.z);
}

// --- simulation state mapping -----------------------------------------

/**
 * Update the boat model from a state message payload.
 * @param {any} msg
 */
// Legacy updateBoatFromState is now handled by boat.js and wind.js.

// --- WebSocket client --------------------------------------------------

let socket = null;
let reconnectDelayMs = 1000;

function makeWsUrl() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.hostname || "127.0.0.1";
  const port = 8765; // default server port
  return `${proto}://${host}:${port}/sim`;
}

  function connectWebSocket() {
  const url = makeWsUrl();
  try {
    socket = new WebSocket(url);
  } catch (err) {
    setConnectionStatus("error");
    return;
  }

  setConnectionStatus("connecting…");

  socket.onopen = () => {
    reconnectDelayMs = 1000;
    setConnectionStatus("connected");
  };

  socket.onclose = () => {
    setConnectionStatus("disconnected");
    socket = null;
    const delay = reconnectDelayMs;
    reconnectDelayMs = Math.min(reconnectDelayMs * 2, 10000);
    window.setTimeout(connectWebSocket, delay);
  };

  socket.onerror = () => {
    setConnectionStatus("error");
  };

  socket.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg && msg.type === "state") {
        const stateVis = updateBoatFromState(
          msg,
          boatGroup,
          sailGroup,
          rudderGroup,
        );
        // NOTE: Do not overwrite command targets from server state.
        // Mixing "what the boat is doing" with "what the user commands"
        // causes twitch/feedback oscillations.
        if (stateVis && typeof stateVis.sailAngleDeg === "number") {
          sailStateDeg = stateVis.sailAngleDeg;
        }
        if (stateVis && typeof stateVis.rudderAngleDeg === "number") {
          rudderStateDeg = stateVis.rudderAngleDeg;
        }
        if (stateVis && stateVis.wind) {
          wind.setFromPayload(stateVis.wind);
        }
        if (msg.forces) {
          lastForces = msg.forces;
          updateForceArrows(msg.forces, showForces);
          updateForcesChart(msg.forces);
        }
        if (msg.control && typeof msg.control.mode === "string") {
          setControlMode(msg.control.mode);
        }
        if (msg.waypoint) {
          waypointMarker.visible = true;
          waypointMarker.position.set(
            msg.waypoint.x,
            0.4,
            msg.waypoint.y,
          );
        }
      }
    } catch {
      // ignore malformed messages
    }
  };
}

connectWebSocket();

// --- local helm controls ----------------------------------------------

// Commanded control targets (what we send to the server).
let rudderCmdDeg = 0.0;
// Sheet-limit command from centerline (deg, 0..SAIL_MAX).
let sailCmdDeg = 0.0;

// Measured / reported by server (for UI/debug only).
let rudderStateDeg = 0.0;
let sailStateDeg = 0.0;

const RUDDER_MAX_DEG = 35.0;
const SAIL_MAX_DEG = 90.0; // total travel ±90° (180° span)
const RUDDER_RATE_DEG = 80.0; // deg/s
const RUDDER_CENTER_RATE_DEG = 220.0; // deg/s (snap back when no key pressed)
const SAIL_RATE_DEG = 90.0; // deg/s

let keyLeft = false;
let keyRight = false;
let keyUp = false;
let keyDown = false;

window.addEventListener("keydown", (ev) => {
  switch (ev.code) {
    case "ArrowLeft":
    case "KeyA":
      keyLeft = true;
      break;
    case "ArrowRight":
    case "KeyD":
      keyRight = true;
      break;
    case "ArrowUp":
    case "KeyW":
      keyUp = true;
      break;
    case "ArrowDown":
    case "KeyS":
      keyDown = true;
      break;
    default:
      break;
  }
});

window.addEventListener("keyup", (ev) => {
  switch (ev.code) {
    case "ArrowLeft":
    case "KeyA":
      keyLeft = false;
      break;
    case "ArrowRight":
    case "KeyD":
      keyRight = false;
      break;
    case "ArrowUp":
    case "KeyW":
      keyUp = false;
      break;
    case "ArrowDown":
    case "KeyS":
      keyDown = false;
      break;
    case "KeyF":
      showForces = !showForces;
      if (lastForces) {
        updateForceArrows(lastForces, showForces);
      }
      ev.preventDefault();
      break;
    case "KeyC":
      cameraFollowMode = !cameraFollowMode;
      ev.preventDefault();
      break;
    default:
      break;
  }
});

// Forces section toggle
const forcesSection = document.getElementById("forces-section");
const forcesToggle = document.getElementById("forces-toggle");
if (forcesSection && forcesToggle) {
  forcesToggle.addEventListener("click", () => {
    forcesSection.classList.toggle("collapsed");
    forcesToggle.textContent = forcesSection.classList.contains("collapsed")
      ? "Forces (Fx, Fy) ▶"
      : "Forces (Fx, Fy) ▼";
  });
}

let lastSentRudderDeg = rudderCmdDeg;
let lastSentSailDeg = sailCmdDeg;

function maybeSendControls() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;

  const changedRudder =
    Math.abs(rudderCmdDeg - lastSentRudderDeg) > 0.1;
  const changedSail = Math.abs(sailCmdDeg - lastSentSailDeg) > 0.1;

  if (!changedRudder && !changedSail) return;

  const payload = {
    type: "control",
    rudder_deg: rudderCmdDeg,
    sail_deg: sailCmdDeg,
  };

  socket.send(JSON.stringify(payload));
  lastSentRudderDeg = rudderCmdDeg;
  lastSentSailDeg = sailCmdDeg;
}

// --- animation loop ----------------------------------------------------

  function onWindowResize() {
  handleResize(camera, renderer);
}

window.addEventListener("resize", onWindowResize);

let lastFrameTime = performance.now();

function animate(now) {
  requestAnimationFrame(animate);
  const dt = (now - lastFrameTime) / 1000.0;
  lastFrameTime = now;

  // integrate local helm controls
  if (keyLeft) {
    rudderCmdDeg = Math.min(
      RUDDER_MAX_DEG,
      rudderCmdDeg + RUDDER_RATE_DEG * dt,
    );
  }
  if (keyRight) {
    rudderCmdDeg = Math.max(
      -RUDDER_MAX_DEG,
      rudderCmdDeg - RUDDER_RATE_DEG * dt,
    );
  }
  if (!keyLeft && !keyRight) {
    // Snap back toward center when no rudder key is held.
    const step = RUDDER_CENTER_RATE_DEG * dt;
    if (rudderCmdDeg > 0) {
      rudderCmdDeg = Math.max(0, rudderCmdDeg - step);
    } else if (rudderCmdDeg < 0) {
      rudderCmdDeg = Math.min(0, rudderCmdDeg + step);
    }
  }
  // Up/down adjust sheet limit (let out / pull in).
  if (keyUp) {
    sailCmdDeg = Math.min(SAIL_MAX_DEG, sailCmdDeg + SAIL_RATE_DEG * dt);
  }
  if (keyDown) {
    sailCmdDeg = Math.max(0.0, sailCmdDeg - SAIL_RATE_DEG * dt);
  }

  maybeSendControls();

  // Visually rotate sail around mast based on last known sail angle from HUD.
  // (Angle is updated in boat.js using data from the server.)

  // subtle bobbing while idle to keep the scene alive
  const t = now / 1000.0;
  boatGroup.position.y = 0.02 * Math.sin(t * 1.4);

  // Advect wind wisps and update wake
  wind.advect(dt, t);
  animateControlSurfaces(dt);
  updateTrace();

  if (cameraFollowMode) {
    updateFollowCamera();
  } else {
    boatGroup.getWorldPosition(boatPos);
    controls.target.copy(boatPos);
    controls.update();
  }

  renderer.render(scene, camera);
}

requestAnimationFrame(animate);

