import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { createScene, handleResize } from "./scene.js";
import { createWind } from "./wind.js";
import { createBoat, updateBoatFromState } from "./boat.js";
import { createForceArrows } from "./forces.js";
import { setConnectionStatus, updateForcesChart } from "./hud.js";

// --- Scene & environment ------------------------------------------------

const app = document.getElementById("app");
const { scene, camera, renderer } = createScene(app);

// --- Wind ---------------------------------------------------------------

const wind = createWind(scene);

// --- Boat ---------------------------------------------------------------

const { boatGroup, sailGroup, rudderGroup, updateTrace } = createBoat(scene);

const { update: updateForceArrows } = createForceArrows(boatGroup);
let showForces = false;
let lastForces = null;

// Camera & controls
const cameraOffset = new THREE.Vector3(-10, 6, 14);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

function updateCameraTarget() {
  const target = new THREE.Vector3();
  boatGroup.getWorldPosition(target);
  // Keep a fixed offset from the boat for the initial view
  if (!updateCameraTarget.hasInitialized) {
    camera.position.copy(target).add(cameraOffset);
    updateCameraTarget.hasInitialized = true;
  }
  // Always orbit around the boat's current position
  controls.target.copy(target);
}

updateCameraTarget();

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
        const windPayload = updateBoatFromState(
          msg,
          boatGroup,
          sailGroup,
          rudderGroup,
          sailDeg,
          rudderDeg,
        );
        if (windPayload) {
          wind.setFromPayload(windPayload);
        }
        if (msg.forces) {
          lastForces = msg.forces;
          updateForceArrows(msg.forces, showForces);
          updateForcesChart(msg.forces);
        }
      }
    } catch {
      // ignore malformed messages
    }
  };
}

connectWebSocket();

// --- local helm controls ----------------------------------------------

let rudderDeg = 0.0;
let sailDeg = 0.0;

const RUDDER_MAX_DEG = 35.0;
const SAIL_MAX_DEG = 90.0; // total travel ±90° (180° span)
const RUDDER_RATE_DEG = 80.0; // deg/s
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

let lastSentRudderDeg = rudderDeg;
let lastSentSailDeg = sailDeg;

function maybeSendControls() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;

  const changedRudder =
    Math.abs(rudderDeg - lastSentRudderDeg) > 0.1;
  const changedSail = Math.abs(sailDeg - lastSentSailDeg) > 0.1;

  if (!changedRudder && !changedSail) return;

  const payload = {
    type: "control",
    rudder_deg: rudderDeg,
    sail_deg: sailDeg,
  };

  socket.send(JSON.stringify(payload));
  lastSentRudderDeg = rudderDeg;
  lastSentSailDeg = sailDeg;
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
    rudderDeg = Math.min(
      RUDDER_MAX_DEG,
      rudderDeg + RUDDER_RATE_DEG * dt,
    );
  }
  if (keyRight) {
    rudderDeg = Math.max(
      -RUDDER_MAX_DEG,
      rudderDeg - RUDDER_RATE_DEG * dt,
    );
  }
  if (keyUp) {
    sailDeg = Math.max(-SAIL_MAX_DEG, sailDeg - SAIL_RATE_DEG * dt);
  }
  if (keyDown) {
    sailDeg = Math.min(SAIL_MAX_DEG, sailDeg + SAIL_RATE_DEG * dt);
  }

  maybeSendControls();

  // Visually rotate sail around mast based on local sail angle (deg from centerline)
  // Positive sailDeg now rotates the sail to starboard (right) when
  // looking in the boat's forward direction.
  sailGroup.rotation.y = THREE.MathUtils.degToRad(-sailDeg);

  // Visually rotate rudder around its hinge at the stern
  rudderGroup.rotation.y = THREE.MathUtils.degToRad(-rudderDeg);

  // subtle bobbing while idle to keep the scene alive
  const t = now / 1000.0;
  boatGroup.position.y = 0.02 * Math.sin(t * 1.4);

  // Advect wind wisps and update wake
  wind.advect(dt, t);
  updateTrace();
  updateCameraTarget();
  controls.update();
  renderer.render(scene, camera);
}

requestAnimationFrame(animate);

