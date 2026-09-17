/**
 * SailBench test bench — loop.
 *
 * An open stretch of water and the waypoint task policies are trained on. Three
 * drivers can hold the helm, all scored identically:
 *
 *   manual    — arrow keys, for a human baseline
 *   autopilot — `autopilot.js`, a hand-written controller, the thing to beat
 *   backend   — `web_runner` over a WebSocket, which is how a trained
 *               checkpoint gets tested here
 *
 * In manual and autopilot modes the browser integrates the boat with the ported
 * models (`web/physics/models.js`, a port of `SailboatHub`). In backend mode
 * Python owns the physics and this page mirrors it.
 *
 * To test a different controller, give it `act(observation) -> [rudder, sheet]`
 * and swap it in beside `createAutopilot` below.
 */

import {
  RAD, DEG, clamp, boatConfig, CONFIG_NAMES, solveAt, stepSim, rudderServoStep,
} from "../physics/models.js";
import { polarAt } from "../physics/analysis.js";
import { createRenderer, drawMinimap, drawWindRose, WORLD_HALF } from "./world.js";
import { createHud } from "./hud.js";
import {
  TASK, decodeAction, encodeAction, observation, sampleWaypoint, vmgToWaypoint,
  newEpisode, stepEpisode, newScoreboard, recordEpisode,
} from "./task.js";
import { createAutopilot } from "./autopilot.js";
import { createLink } from "./link.js";
import { liveryFor } from "./boats.js";

// Manual helm rates, matched to web/main.js so the two UIs feel the same.
const RUDDER_RATE_DEG = 80;
const RUDDER_CENTER_RATE_DEG = 220;
const SAIL_RATE_DEG = 90;

const WAKE_POINTS = 1200;
const WAKE_EVERY_S = 0.12;

const canvas = document.getElementById("stage");
const minimap = document.getElementById("minimap");
const rose = document.getElementById("rose");

const renderer = createRenderer(canvas);
renderer.setZoom(16);

// --- state --------------------------------------------------------------

const game = {
  config: "flingo_floty.yaml",
  cfgs: null,
  livery: null,
  wind: { speed: 5, dirDeg: 90 },
  /** [x, y, cos psi, sin psi, u, v, r] — the simulator's own state vector. */
  arr: [0, 0, 1, 0, 0, 0, 0],
  frames: null,
  /** Normalised [rudder, sheet], exactly the action a policy emits. */
  action: [0, -0.3],
  sheetDeg: 30,
  rudderCmdDeg: 0,
  servoDeg: 0,
  wake: [],
  wakeClock: 0,
  episode: null,
  boards: { manual: newScoreboard(), autopilot: newScoreboard(), backend: newScoreboard() },
  mode: "manual",
  paused: false,
  options: { showForces: false, showWind: true, showTrail: true, showMap: true, autoNext: true },
  keys: { left: false, right: false, up: false, down: false },
  /** Latest backend frame, when one is attached. */
  remote: null,
};

let autopilot = null;

function loadBoat(name) {
  game.config = name;
  game.cfgs = boatConfig(name);
  game.cfgs.sail.wind_speed = game.wind.speed;
  game.cfgs.sail.wind_dir_deg = game.wind.dirDeg;
  game.livery = liveryFor(name);
  autopilot = createAutopilot(name);
  document.title = `SailBench — ${game.livery.name}`;
}

function boatState() {
  const [x, y, c, s, u, v, r] = game.arr;
  return { x, y, psi: Math.atan2(s, c), u, v, r };
}

/** Start a fresh episode. `waypoint` overrides the sampled goal. */
function newRun(waypoint = null) {
  const start = [game.arr[0], game.arr[1]];
  const goal = waypoint || sampleWaypoint(start, game.wind.dirDeg);
  game.episode = newEpisode(goal, start);
  if (autopilot) autopilot.reset();
  game.wake = [];
  game.wakeClock = 0;
  hud.hideBanner();
}

/** Put the boat back at the origin, close hauled with steerage way on. */
function resetBoat() {
  const heading = (game.wind.dirDeg + 180 + 45) * RAD;
  game.arr = [0, 0, Math.cos(heading), Math.sin(heading), 0.4, 0, 0];
  game.frames = null;
  game.servoDeg = 0;
  game.rudderCmdDeg = 0;
  game.sheetDeg = 30;
  game.action = encodeAction(0, 30);
}

// --- HUD ----------------------------------------------------------------

const MODES = [
  ["manual", "manual", "you steer with the arrow keys"],
  ["autopilot", "autopilot", "the hand-written baseline controller in autopilot.js"],
  ["backend", "backend", "mirror web_runner — run a trained policy with --policy-model"],
];

const hud = createHud({
  onToggle: (key) => {
    game.options[key] = !game.options[key];
    hud.setToggles(game.options);
    hud.setMapVisible(game.options.showMap);
    canvas.focus({ preventScroll: true });
  },
  onWind: ({ speed, dirDeg }) => {
    if (speed !== undefined) game.wind.speed = speed;
    if (dirDeg !== undefined) game.wind.dirDeg = dirDeg;
    game.cfgs.sail.wind_speed = game.wind.speed;
    game.cfgs.sail.wind_dir_deg = game.wind.dirDeg;
  },
  onBoat: (name) => {
    loadBoat(name);
    resetBoat();
    newRun();
    canvas.focus({ preventScroll: true });
  },
  onMode: (mode) => setMode(mode),
  configs: CONFIG_NAMES.map((name) => [name, liveryFor(name).name]),
  modes: MODES,
});

// --- backend link -------------------------------------------------------

const link = createLink({
  onState: (msg) => { game.remote = msg; },
  onStatus: (status, detail) => {
    if (game.mode !== "backend") return;
    if (status === "connected") hud.banner("backend attached", detail, 1600);
    else if (status === "error") hud.banner("no backend", detail, 2400);
  },
});

function setMode(mode) {
  if (mode === game.mode) return;
  game.mode = mode;
  hud.setMode(mode);

  if (mode === "backend") {
    hud.banner("backend mode", "start web_runner, optionally with --policy-model", 2600);
    link.connect();
  } else {
    link.disconnect();
    game.remote = null;
    resetBoat();
  }
  if (autopilot) autopilot.reset();
  newRun(game.episode ? game.episode.waypoint : null);
  canvas.focus({ preventScroll: true });
}

loadBoat(game.config);
hud.setToggles(game.options);
hud.setWind(game.wind.speed, game.wind.dirDeg);
hud.setBoat(game.config);
hud.setMode(game.mode);
resetBoat();
newRun();
hud.banner("test bench", "pick a driver · click the water to place a waypoint", 2600);

// --- input --------------------------------------------------------------

const KEYMAP = {
  ArrowLeft: "left", KeyA: "left",
  ArrowRight: "right", KeyD: "right",
  ArrowUp: "up", KeyW: "up",
  ArrowDown: "down", KeyS: "down",
};

window.addEventListener("keydown", (ev) => {
  if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLSelectElement) return;

  const dir = KEYMAP[ev.code];
  if (dir) {
    game.keys[dir] = true;
    ev.preventDefault();
    return;
  }

  switch (ev.code) {
    case "KeyF":
    case "KeyT":
    case "KeyM":
    case "KeyQ":
    case "KeyO": {
      // W and S are helm keys, so wind gets Q and auto-next gets O.
      const option = {
        KeyF: "showForces", KeyT: "showTrail", KeyM: "showMap",
        KeyQ: "showWind", KeyO: "autoNext",
      }[ev.code];
      game.options[option] = !game.options[option];
      hud.setToggles(game.options);
      hud.setMapVisible(game.options.showMap);
      break;
    }
    case "KeyN":
      newRun();
      break;
    case "KeyR":
      resetBoat();
      newRun();
      break;
    case "Digit1": setMode("manual"); break;
    case "Digit2": setMode("autopilot"); break;
    case "Digit3": setMode("backend"); break;
    case "Space":
      game.paused = !game.paused;
      if (game.paused) hud.banner("paused", "space to resume", 0);
      else hud.hideBanner();
      ev.preventDefault();
      break;
    case "Equal":
    case "NumpadAdd":
      renderer.setZoom(renderer.zoom * 1.25);
      break;
    case "Minus":
    case "NumpadSubtract":
      renderer.setZoom(renderer.zoom / 1.25);
      break;
    default:
      break;
  }
});

window.addEventListener("keyup", (ev) => {
  const dir = KEYMAP[ev.code];
  if (dir) game.keys[dir] = false;
});

// Click the water to drop a waypoint there and start a fresh episode.
canvas.addEventListener("pointerdown", (ev) => {
  canvas.focus({ preventScroll: true });
  const world = renderer.screenToWorld(ev.offsetX, ev.offsetY);
  newRun([
    clamp(world[0], -WORLD_HALF, WORLD_HALF),
    clamp(world[1], -WORLD_HALF, WORLD_HALF),
  ]);
});

canvas.addEventListener("wheel", (ev) => {
  renderer.setZoom(renderer.zoom * (ev.deltaY < 0 ? 1.12 : 1 / 1.12));
  ev.preventDefault();
}, { passive: false });

window.addEventListener("resize", () => {
  renderer.resize();
  renderer.seedStreaks();
});
canvas.setAttribute("tabindex", "0");
canvas.focus({ preventScroll: true });

// --- drivers ------------------------------------------------------------

/** Manual helm: integrate key holds into a commanded rudder and sheet. */
function manualAction(dt) {
  if (game.keys.left) {
    game.rudderCmdDeg = Math.min(TASK.maxRudderDeg, game.rudderCmdDeg + RUDDER_RATE_DEG * dt);
  }
  if (game.keys.right) {
    game.rudderCmdDeg = Math.max(-TASK.maxRudderDeg, game.rudderCmdDeg - RUDDER_RATE_DEG * dt);
  }
  if (!game.keys.left && !game.keys.right) {
    const step = RUDDER_CENTER_RATE_DEG * dt;
    game.rudderCmdDeg = game.rudderCmdDeg > 0
      ? Math.max(0, game.rudderCmdDeg - step)
      : Math.min(0, game.rudderCmdDeg + step);
  }
  if (game.keys.up) game.sheetDeg = Math.min(TASK.maxSailDeg, game.sheetDeg + SAIL_RATE_DEG * dt);
  if (game.keys.down) game.sheetDeg = Math.max(0, game.sheetDeg - SAIL_RATE_DEG * dt);
  return encodeAction(game.rudderCmdDeg, game.sheetDeg);
}

// --- stepping -----------------------------------------------------------

/** One simulator step, driven by whichever controller holds the helm. */
function stepPhysics(dt) {
  const state = boatState();
  const obs = observation(state, game.episode.waypoint, game.wind, game.action);

  const action = game.mode === "autopilot" ? autopilot.act(obs, dt) : manualAction(dt);
  game.action = action;

  const { rudderDeg, sheetDeg } = decodeAction(action);
  game.sheetDeg = sheetDeg;
  if (game.mode === "autopilot") game.rudderCmdDeg = rudderDeg;

  // The hub puts the commanded rudder through its servo every step, so we do too.
  game.servoDeg = rudderServoStep(game.servoDeg, rudderDeg, dt, game.cfgs.rudder);

  const out = stepSim(game.arr, dt, game.cfgs, {
    sheetRad: sheetDeg * RAD,
    rudderRad: game.servoDeg * RAD,
  }, game.frames);

  // The models can run away if the state is driven somewhere the real sim never
  // goes; recover rather than rendering NaN geometry.
  if (!out.arr.every(Number.isFinite)
    || Math.hypot(out.arr[4], out.arr[5]) > 40
    || Math.abs(out.arr[6]) > 40) {
    hud.banner("diverged", "state ran away — boat reset", 1800);
    resetBoat();
    return;
  }

  const moved = Math.hypot(out.arr[0] - game.arr[0], out.arr[1] - game.arr[1]);
  game.arr = out.arr;
  game.frames = out.frames;

  // Water boundary: bleed off speed rather than letting the boat escape.
  for (const i of [0, 1]) {
    if (Math.abs(game.arr[i]) > WORLD_HALF) {
      game.arr[i] = Math.sign(game.arr[i]) * WORLD_HALF;
      game.arr[4] *= 0.4;
      game.arr[5] *= 0.4;
      game.frames = null;
    }
  }

  advanceEpisode(boatState(), dt, moved);
}

/** Mirror one backend frame into the bench's own state. */
function stepBackend(dt) {
  const msg = game.remote;
  if (!msg) return;

  const { position, heading, velocity_body: vb } = msg.boat;
  const prev = [game.arr[0], game.arr[1]];
  game.arr = [position.x, position.y, heading.cos, heading.sin, vb.u, vb.v, vb.r];

  if (msg.wind) {
    game.wind.speed = msg.wind.speed;
    game.wind.dirDeg = msg.wind.dir_deg;
    game.cfgs.sail.wind_speed = msg.wind.speed;
    game.cfgs.sail.wind_dir_deg = msg.wind.dir_deg;
    hud.setWind(game.wind.speed, game.wind.dirDeg);
  }
  if (msg.sail) game.sheetDeg = Math.abs(msg.sail.angle_deg);
  if (msg.rudder) game.servoDeg = msg.rudder.angle_deg;
  game.action = encodeAction(game.servoDeg, game.sheetDeg);

  // When a policy is driving, the backend owns the goal.
  if (msg.waypoint) {
    const goal = [msg.waypoint.x, msg.waypoint.y];
    const moved = Math.hypot(
      goal[0] - game.episode.waypoint[0],
      goal[1] - game.episode.waypoint[1],
    );
    if (moved > 0.5) {
      newRun(goal);
      return;
    }
  }

  advanceEpisode(boatState(), dt, Math.hypot(game.arr[0] - prev[0], game.arr[1] - prev[1]));
}

function advanceEpisode(state, dt, moved) {
  const twa = ((((game.wind.dirDeg + 180 - state.psi * DEG) % 360) + 540) % 360) - 180;
  const before = game.episode.result;
  const after = stepEpisode(game.episode, state, dt, twa, moved);

  if (before === "running" && after !== "running") {
    recordEpisode(game.boards[game.mode], game.episode);
    const label = { success: "reached", failure: "lost it", timeout: "timed out" }[after];
    const detail = `${game.episode.t.toFixed(1)} s · ${game.episode.steps} steps · `
      + `${game.episode.tacks} tacks`;

    if (game.options.autoNext) {
      // Roll straight into the next episode so a controller can be left running
      // and the scoreboard becomes a real sample rather than a single run.
      hud.banner(label, `${detail} · next episode`, 1400);
      window.setTimeout(() => {
        if (game.episode && game.episode.done) newRun();
      }, 900);
    } else {
      hud.banner(label, `${detail} · N for the next one`, 0);
    }
  }
}

// --- frame --------------------------------------------------------------

/** Steady-state speed the polar predicts for this TWA, or null. */
function polarTarget(twaAbs, speed) {
  const polar = polarAt(game.config, game.wind.speed);
  if (!polar) return null;
  const list = polar.twa_deg;
  let i = 0;
  while (i < list.length - 2 && list[i + 1] < twaAbs) i += 1;
  const span = list[i + 1] - list[i] || 1;
  const f = clamp((twaAbs - list[i]) / span, 0, 1);
  const target = polar.speed[i] + f * (polar.speed[i + 1] - polar.speed[i]);
  return { target, ratio: target > 1e-6 ? speed / target : 0 };
}

let last = performance.now();
let accumulator = 0;

function frame(now) {
  requestAnimationFrame(frame);

  const wall = Math.min((now - last) / 1000, 0.1);
  last = now;

  const simDt = Number(game.cfgs.simulation.dt ?? 0.02);
  if (!game.paused) {
    if (game.mode === "backend") {
      stepBackend(wall);
    } else {
      accumulator += wall;
      let guard = 0;
      while (accumulator >= simDt && guard < 12) {
        stepPhysics(simDt);
        accumulator -= simDt;
        guard += 1;
      }
      if (guard >= 12) accumulator = 0;
    }
  }

  const state = boatState();
  const speed = Math.hypot(state.u, state.v);

  if (!game.paused) {
    game.wakeClock += wall;
    if (game.wakeClock >= WAKE_EVERY_S) {
      game.wakeClock = 0;
      game.wake.push([state.x, state.y]);
      if (game.wake.length > WAKE_POINTS) game.wake.shift();
    }
  }

  const solved = solveAt(state, state.psi, game.cfgs, {
    sheetRad: game.sheetDeg * RAD,
    rudderRad: game.servoDeg * RAD,
  });

  const headingDeg = state.psi * DEG;
  const twa = ((((game.wind.dirDeg + 180 - headingDeg) % 360) + 540) % 360) - 180;
  const distance = Math.hypot(
    game.episode.waypoint[0] - state.x,
    game.episode.waypoint[1] - state.y,
  );

  const boat = { ...state, sheetDeg: game.sheetDeg, rudderDeg: game.servoDeg };

  const scene = {
    boat,
    cfgs: game.cfgs,
    livery: game.livery,
    solved,
    episode: game.episode,
    wind: game.wind,
    wake: game.wake,
    options: game.options,
    awaDeg: solved.sail.awaDeg,
    headingDeg,
    twa,
    speed,
    distance,
    vmg: vmgToWaypoint(state, game.episode.waypoint),
    target: polarTarget(Math.abs(twa), speed),
    observation: observation(state, game.episode.waypoint, game.wind, game.action),
    action: game.action,
    mode: game.mode,
    board: game.boards[game.mode],
  };

  if (!game.paused) renderer.follow(boat, wall);
  renderer.draw(scene, game.paused ? 0 : wall);
  hud.update(scene);
  drawWindRose(rose, scene, 32);
  if (game.options.showMap) drawMinimap(minimap, scene);
}

requestAnimationFrame(frame);

/**
 * Console handle, for scripting an evaluation without touching the UI.
 *
 *   sailbench.setMode("autopilot")
 *   sailbench.game.boards.autopilot          // running tally
 *   sailbench.newRun([25, 10])               // place a waypoint
 *   sailbench.setController(myController)    // anything with act(obs, dt)
 */
window.sailbench = {
  game,
  renderer,
  link,
  setMode,
  newRun,
  resetBoat,
  setController(controller) {
    autopilot = controller;
    if (autopilot.reset) autopilot.reset();
  },
};
