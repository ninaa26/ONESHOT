import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { createScene, handleResize } from "./scene.js";
import { createWind } from "./wind.js";
import { SIM_Y_TO_WORLD_Z, createBoat, updateBoatFromState } from "./boat.js";
import { createForceArrows } from "./forces.js";
import { createRadar } from "./radar.js";
import { setBoat, setConnectionStatus, setControlMode, setWind, updateForcesChart } from "./hud.js";
import { createShipyard } from "./shipyard.js";

// --- Scene & environment ------------------------------------------------

const app = document.getElementById("app");
const { scene, camera, renderer } = createScene(app);

// --- Wind ---------------------------------------------------------------

const wind = createWind(scene);

// --- Boat ---------------------------------------------------------------

const { boatGroup, sailGroup, rudderGroup, updateTrace, animateControlSurfaces, setRig, setGeometry } =
  createBoat(scene);
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
const radar = createRadar(document.getElementById("radar"));
let showForces = false;
let lastForces = null;

// Chase camera: looks straight down on the boat, so what is ahead of it is on
// screen rather than hidden behind its own sail. The view is north-up, not
// heading-up: the boat turns within a world that stays put, which is what makes
// a course readable. `C` drops out of it into free orbit.
const EYE_HEIGHT_MIN = 8;
const EYE_HEIGHT_MAX = 220;
const EYE_HEIGHT_DEFAULT = 22;
const EYE_ZOOM_STEP = 1.12;

const boatPos = new THREE.Vector3();

let cameraFollowMode = true;
let eyeHeight = EYE_HEIGHT_DEFAULT;
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

function updateFollowCamera() {
  boatGroup.getWorldPosition(boatPos);
  camera.position.set(boatPos.x, eyeHeight, boatPos.z);
  // Looking straight down, the default up vector is the direction of view and
  // lookAt has nothing to orient against, so screen-up is given as a world
  // direction. Sim +y is north and maps to world -z, so this is north-up -- and
  // because the embedding no longer reverses orientation, east is to the right.
  camera.up.set(0, 0, SIM_Y_TO_WORLD_Z);
  camera.lookAt(boatPos.x, 0, boatPos.z);
}

// OrbitControls' own zoom cannot help while the camera is being placed every
// frame, so the wheel changes how high the eye sits instead.
renderer.domElement.addEventListener(
  "wheel",
  (event) => {
    if (!cameraFollowMode) return;
    event.preventDefault();
    const factor = event.deltaY > 0 ? EYE_ZOOM_STEP : 1 / EYE_ZOOM_STEP;
    eyeHeight = Math.min(EYE_HEIGHT_MAX, Math.max(EYE_HEIGHT_MIN, eyeHeight * factor));
  },
  { passive: false },
);

// --- simulation state mapping -----------------------------------------

/**
 * Update the boat model from a state message payload.
 * @param {any} msg
 */
// Legacy updateBoatFromState is now handled by boat.js and wind.js.

// --- Shipyard (pre-sail selection) --------------------------------------

// The server sends a catalog on connect and waits for a `setup` before it
// starts simulating. The shipyard collects that choice; Esc while sailing
// drops the connection so the next one can pick again.
const shipyard = createShipyard({ onLaunch: sendSetup });
// A training stream (live_vis) sends state with no catalog first; once we
// see that, the shipyard has nothing to offer and stays out of the way.
let streamOnly = false;

function sendSetup(build) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    shipyard.showError("Not connected to the sim backend yet.");
    return;
  }
  socket.send(JSON.stringify({ type: "setup", ...build }));
}

function returnToShipyard() {
  if (streamOnly || shipyard.isOpen()) return;
  shipyard.setStatus("Reconnecting…");
  shipyard.show();
  setBoat("—");
  // Each connection sails one build, so a new pick means a new connection.
  reconnectDelayMs = 200;
  if (socket) {
    socket.close();
  }
}

// --- WebSocket client --------------------------------------------------

let socket = null;
let reconnectDelayMs = 1000;

// The sim backend is a separate process on its own port, so this page cannot
// infer it from where it was served. 8765 is what the README starts it on and
// stays the default, so nothing changes for the usual workflow.
const DEFAULT_WS_PORT = 8765;

// `?port=` points the page at a backend on another port, which is what lets two
// checkouts be previewed at once without editing this file. `?host=` reaches a
// backend on another machine, and `?ws=` replaces the whole URL for one behind a
// proxy. A port that is not a number falls back rather than building a URL that
// can only fail to connect.
function makeWsUrl() {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("ws");
  if (override) {
    return override;
  }
  const requested = params.get("port");
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const host = params.get("host") || window.location.hostname || "127.0.0.1";
  const port = /^[0-9]+$/.test(requested || "") ? requested : DEFAULT_WS_PORT;
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
    if (shipyard.isOpen()) {
      shipyard.setStatus("Connected. Waiting for the catalog…");
    }
  };

  socket.onclose = () => {
    setConnectionStatus("disconnected");
    socket = null;
    if (shipyard.isOpen()) {
      shipyard.setStatus("Backend disconnected. Retrying…");
    }
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
      if (!msg) return;
      if (msg.type === "catalog") {
        streamOnly = false;
        // Tell the server we'll be sending a setup, so it waits instead of
        // assuming an older page and sailing its defaults.
        socket.send(JSON.stringify({ type: "hello" }));
        shipyard.setCatalog(msg);
        return;
      }
      if (msg.type === "ready") {
        const build = { boat: msg.boat, parts: msg.parts || {}, helm: msg.helm };
        setBoat(shipyard.describe(build));
        // Draw this boat at its own size, with the rig its sail model describes,
        // rather than one fixed hull with one fixed sail.
        setGeometry(shipyard.geometryFor(build.boat));
        setRig(build.parts.sail);
        if (typeof msg.control_mode === "string") {
          setControlMode(msg.control_mode);
        }
        // Fresh boat, fresh track, and its own config's wind again.
        radar.reset();
        windCmdMs = null;
        lastSentWindMs = null;
        windDirCmdDeg = null;
        lastSentWindDirDeg = null;
        // Fresh boat, fresh helm: don't carry stale commands into it.
        rudderCmdDeg = 0;
        sailCmdDeg = 0;
        lastSentRudderDeg = 0;
        lastSentSailDeg = 0;
        shipyard.hide();
        return;
      }
      if (msg.type === "error") {
        shipyard.showError(String(msg.message || "The backend rejected that build."));
        if (!shipyard.isOpen()) shipyard.show();
        return;
      }
      if (msg.type === "state") {
        if (shipyard.isOpen() && !shipyard.hasCatalog()) {
          // No catalog ever came: this backend just streams (e.g. training).
          streamOnly = true;
          shipyard.hide();
        }
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
          // Adopt whatever the boat is actually sailing in as the starting
          // point, so the first keypress nudges from there instead of jumping.
          if (windCmdMs === null && typeof stateVis.wind.speed === "number") {
            windCmdMs = stateVis.wind.speed;
            syncWindSliders();
          }
          if (windDirCmdDeg === null && typeof stateVis.wind.dir_deg === "number") {
            windDirCmdDeg = stateVis.wind.dir_deg;
            syncWindSliders();
          }
          setWind(stateVis.wind.speed, stateVis.wind.dir_deg, windCmdMs, windDirCmdDeg);
        }
        if (msg.forces) {
          lastForces = msg.forces;
          updateForceArrows(msg.forces, showForces);
          updateForcesChart(msg.forces);
        }
        if (msg.control && typeof msg.control.mode === "string") {
          setControlMode(msg.control.mode);
        }
        // The plan view works in sim coordinates directly, so it needs the
        // message rather than anything the scene derived from it.
        if (msg.boat && msg.boat.position) {
          radar.update({
            x: msg.boat.position.x,
            y: msg.boat.position.y,
            headingRad: Math.atan2(msg.boat.heading.sin, msg.boat.heading.cos),
            waypoint: msg.waypoint || null,
            wind: msg.wind
              ? { speed: msg.wind.speed, dirDeg: msg.wind.dir_deg }
              : null,
          });
        }
        if (msg.waypoint) {
          waypointMarker.visible = true;
          waypointMarker.position.set(
            msg.waypoint.x,
            0.4,
            SIM_Y_TO_WORLD_Z * msg.waypoint.y,
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

let lastSentRudderDeg = rudderCmdDeg;
let lastSentSailDeg = sailCmdDeg;

// Wind the helm can change. It starts as null so the boat sails whatever its
// config asked for; the first state message adopts that as the starting point,
// and only a keypress after that makes this a command the server is sent.
const WIND_STEP_MS = 0.5;
const WIND_MIN_MS = 0.0;
const WIND_MAX_MS = 25.0;
const WIND_DIR_STEP_DEG = 5.0;
let windCmdMs = null;
let lastSentWindMs = null;
let windDirCmdDeg = null;
let lastSentWindDirDeg = null;

function nudgeWind(deltaMs) {
  if (windCmdMs === null) return;
  windCmdMs = Math.min(WIND_MAX_MS, Math.max(WIND_MIN_MS, windCmdMs + deltaMs));
  syncWindSliders();
}

function nudgeWindDir(deltaDeg) {
  if (windDirCmdDeg === null) return;
  // Direction wraps rather than clamping: there is no end of the compass.
  windDirCmdDeg = ((windDirCmdDeg + deltaDeg) % 360 + 360) % 360;
  syncWindSliders();
}

// The sliders and the keys are two handles on one command, so each writes the
// command and then both are made to agree with it.
const windSpeedSlider = document.getElementById("wind-speed-slider");
const windDirSlider = document.getElementById("wind-dir-slider");

function syncWindSliders() {
  if (windSpeedSlider && windCmdMs !== null) {
    windSpeedSlider.value = String(windCmdMs);
  }
  if (windDirSlider && windDirCmdDeg !== null) {
    windDirSlider.value = String(Math.round(windDirCmdDeg));
  }
}

if (windSpeedSlider) {
  windSpeedSlider.addEventListener("input", () => {
    windCmdMs = Number(windSpeedSlider.value);
  });
}
if (windDirSlider) {
  windDirSlider.addEventListener("input", () => {
    windDirCmdDeg = Number(windDirSlider.value);
  });
}

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
  // A focused slider owns the keys it actually uses -- the arrows and the ends
  // of its range -- because steering the boat from those at the same time would
  // fight whoever is dragging it. It does not own the rest: swallowing every
  // key meant that clicking a wind slider silently stopped `[`, `]`, `,` and
  // `.` from working, so the wind stopped changing and everything downstream of
  // it looked frozen.
  const tag = ev.target && ev.target.tagName;
  const isField = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  const OWNED_BY_FIELD = new Set([
    "ArrowLeft",
    "ArrowRight",
    "ArrowUp",
    "ArrowDown",
    "Home",
    "End",
    "PageUp",
    "PageDown",
  ]);
  if (isField && OWNED_BY_FIELD.has(ev.code)) return;
  if (shipyard.handleKey(ev)) return;
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
    // Wind strength. Stepped rather than held: it is a condition being set,
    // not a control being flown, so it should stay where it is put.
    case "BracketLeft":
      nudgeWind(-WIND_STEP_MS);
      break;
    case "BracketRight":
      nudgeWind(WIND_STEP_MS);
      break;
    // Wind direction, counter-clockwise and clockwise. Same reasoning: a
    // condition to set, not a control to hold.
    case "Comma":
      nudgeWindDir(WIND_DIR_STEP_DEG);
      break;
    case "Period":
      nudgeWindDir(-WIND_DIR_STEP_DEG);
      break;
    default:
      break;
  }
});

window.addEventListener("keyup", (ev) => {
  if (shipyard.isOpen()) {
    // A key that opened the shipyard must not keep steering behind it.
    keyLeft = keyRight = keyUp = keyDown = false;
    return;
  }
  switch (ev.code) {
    case "Escape":
      returnToShipyard();
      ev.preventDefault();
      break;
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
      if (!cameraFollowMode) {
        // Hand a usable orientation back to OrbitControls on the way out.
        camera.up.set(0, 1, 0);
        controls.target.copy(boatPos);
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

function maybeSendControls() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  // Before `ready` the server is waiting on a setup; a control message now
  // would make it sail the CLI defaults instead of what the shipyard picks.
  if (shipyard.isOpen() && !streamOnly) return;

  const changedRudder =
    Math.abs(rudderCmdDeg - lastSentRudderDeg) > 0.1;
  const changedSail = Math.abs(sailCmdDeg - lastSentSailDeg) > 0.1;
  const changedWind =
    windCmdMs !== null &&
    (lastSentWindMs === null || Math.abs(windCmdMs - lastSentWindMs) > 0.01);
  const changedWindDir =
    windDirCmdDeg !== null &&
    (lastSentWindDirDeg === null ||
      Math.abs(windDirCmdDeg - lastSentWindDirDeg) > 0.01);

  if (!changedRudder && !changedSail && !changedWind && !changedWindDir) return;

  const payload = {
    type: "control",
    rudder_deg: rudderCmdDeg,
    sail_deg: sailCmdDeg,
  };
  // Omitted unless the helm has actually moved it: the server reads a missing
  // field as "no change", so sending it always would pin the wind to whatever
  // the page last saw and stop the config from ever setting it.
  if (changedWind) {
    payload.wind_speed = windCmdMs;
  }
  if (changedWindDir) {
    payload.wind_dir_deg = windDirCmdDeg;
  }

  socket.send(JSON.stringify(payload));
  lastSentRudderDeg = rudderCmdDeg;
  lastSentSailDeg = sailCmdDeg;
  if (changedWind) {
    lastSentWindMs = windCmdMs;
  }
  if (changedWindDir) {
    lastSentWindDirDeg = windDirCmdDeg;
  }
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

