/**
 * Browser port of the waypoint task that policies are trained and evaluated on.
 *
 * Mirrors `sailbench/rl/envs/waypoint_env.py` — the same observation vector, the
 * same action mapping, the same termination rules — so a controller driven here
 * sees exactly what it would see inside the Gymnasium env. The dynamics come
 * from `web/physics/models.js`, which is the port of `SailboatHub`.
 *
 * Keep this in step with `WaypointEnvConfig` when the env changes.
 */

import { RAD, clamp } from "../shared/physics/models.js";

/** Defaults copied from `WaypointEnvConfig`. */
export const TASK = {
  maxEpisodeSteps: 1000,
  maxRudderDeg: 35.0,
  maxSailDeg: 85.0,
  spawnRadiusM: 10.0,
  waypointMinRadiusM: 12.0,
  waypointMaxRadiusM: 25.0,
  successRadiusM: 1.5,
  failRadiusM: 80.0,
  speedScale: 6.0,
  yawRateScale: 2.0,
  windSpeedScale: 15.0,
  upwindWaypointBias: 0.0,
  upwindHalfAngleDeg: 35.0,
};

/**
 * `WaypointEnv.step`'s action mapping.
 *
 * @param {[number, number]} action normalised [rudder, sheet] in [-1, 1]
 * @returns {{rudderDeg: number, sheetDeg: number}} what the hub is commanded
 */
export function decodeAction(action) {
  const rudder = clamp(action[0], -1, 1);
  const sail = clamp(action[1], -1, 1);
  return {
    rudderDeg: rudder * TASK.maxRudderDeg,
    // [-1, 1] -> [0, 1], so the sheet limit spans [0, maxSailDeg].
    sheetDeg: 0.5 * (sail + 1) * TASK.maxSailDeg,
  };
}

/** Inverse of `decodeAction`, for showing a manual helm as a policy action. */
export function encodeAction(rudderDeg, sheetDeg) {
  return [
    clamp(rudderDeg / TASK.maxRudderDeg, -1, 1),
    clamp((sheetDeg / TASK.maxSailDeg) * 2 - 1, -1, 1),
  ];
}

/** Waypoint offset in the boat frame, as `_relative_waypoint_boat_frame`. */
export function relativeWaypoint(state, waypoint) {
  const dxWorld = waypoint[0] - state.x;
  const dyWorld = waypoint[1] - state.y;
  const c = Math.cos(state.psi);
  const s = Math.sin(state.psi);
  return [c * dxWorld + s * dyWorld, -s * dxWorld + c * dyWorld];
}

/** `_velocity_made_good_to_waypoint`. */
export function vmgToWaypoint(state, waypoint) {
  const dx = waypoint[0] - state.x;
  const dy = waypoint[1] - state.y;
  const distance = Math.max(Math.hypot(dx, dy), 1e-9);
  const c = Math.cos(state.psi);
  const s = Math.sin(state.psi);
  const vx = c * state.u - s * state.v;
  const vy = s * state.u + c * state.v;
  return (vx * dx + vy * dy) / distance;
}

/**
 * The 13-element observation `WaypointEnv._get_observation` builds.
 *
 * @param {{x,y,psi,u,v,r}} state boat state, psi in radians
 * @param {[number, number]} waypoint world position
 * @param {{speed: number, dirDeg: number}} wind
 * @param {[number, number]} lastAction the previous normalised action
 * @returns {number[]} length 13, every element already clipped to its range
 */
export function observation(state, waypoint, wind, lastAction) {
  const rel = relativeWaypoint(state, waypoint);
  const distance = Math.max(Math.hypot(waypoint[0] - state.x, waypoint[1] - state.y), 1e-9);
  const relDirX = rel[0] / distance;
  const relDirY = rel[1] / distance;

  const windRad = wind.dirDeg * RAD;
  const c = Math.cos(state.psi);
  const s = Math.sin(state.psi);
  const windWorldX = Math.cos(windRad);
  const windWorldY = Math.sin(windRad);
  const windBoatX = c * windWorldX + s * windWorldY;
  const windBoatY = -s * windWorldX + c * windWorldY;

  return [
    clamp(rel[0] / TASK.waypointMaxRadiusM, -1, 1),
    clamp(rel[1] / TASK.waypointMaxRadiusM, -1, 1),
    clamp(distance / TASK.waypointMaxRadiusM, 0, 1),
    clamp(relDirX, -1, 1),
    clamp(relDirY, -1, 1),
    clamp(state.u / TASK.speedScale, -1, 1),
    clamp(state.v / TASK.speedScale, -1, 1),
    clamp(state.r / TASK.yawRateScale, -1, 1),
    clamp(windBoatX, -1, 1),
    clamp(windBoatY, -1, 1),
    clamp(wind.speed / TASK.windSpeedScale, 0, 1),
    lastAction[0],
    lastAction[1],
  ];
}

export const OBSERVATION_LABELS = [
  "rel_wp_x", "rel_wp_y", "distance", "rel_dir_x", "rel_dir_y",
  "u", "v", "r", "wind_boat_x", "wind_boat_y", "wind_speed",
  "prev_rudder", "prev_sheet",
];

/** Sample a waypoint the way `_sample_waypoint` does. */
export function sampleWaypoint(from, windDirDeg, rand = Math.random) {
  let theta = (rand() * 2 - 1) * Math.PI;
  const bias = clamp(TASK.upwindWaypointBias, 0, 1);
  if (rand() < bias) {
    const upwind = windDirDeg * RAD + Math.PI;
    const half = TASK.upwindHalfAngleDeg * RAD;
    theta = upwind - half + rand() * 2 * half;
  }
  const radius = TASK.waypointMinRadiusM
    + rand() * (TASK.waypointMaxRadiusM - TASK.waypointMinRadiusM);
  return [from[0] + radius * Math.cos(theta), from[1] + radius * Math.sin(theta)];
}

/**
 * Episode bookkeeping: termination, plus the numbers you actually want when
 * judging a controller.
 */
export function newEpisode(waypoint, start) {
  return {
    waypoint,
    start: [start[0], start[1]],
    steps: 0,
    t: 0,
    pathLength: 0,
    closest: Infinity,
    straightLine: Math.hypot(waypoint[0] - start[0], waypoint[1] - start[1]),
    tacks: 0,
    gybes: 0,
    lastTwaSign: 0,
    done: false,
    /** "running" | "success" | "failure" | "timeout" */
    result: "running",
  };
}

/**
 * Advance the episode by one simulator step.
 *
 * @returns {"running"|"success"|"failure"|"timeout"} the state after this step
 */
export function stepEpisode(episode, state, dt, twa, moved) {
  if (episode.done) return episode.result;

  episode.steps += 1;
  episode.t += dt;
  episode.pathLength += moved;

  const distance = Math.hypot(episode.waypoint[0] - state.x, episode.waypoint[1] - state.y);
  episode.closest = Math.min(episode.closest, distance);

  // Count manoeuvres: the wind crossing the bow is a tack, crossing the stern a gybe.
  const sign = Math.sign(twa);
  if (sign !== 0 && episode.lastTwaSign !== 0 && sign !== episode.lastTwaSign) {
    if (Math.abs(twa) < 90) episode.tacks += 1;
    else episode.gybes += 1;
  }
  if (sign !== 0) episode.lastTwaSign = sign;

  if (distance <= TASK.successRadiusM) {
    episode.done = true;
    episode.result = "success";
  } else if (distance >= TASK.failRadiusM) {
    episode.done = true;
    episode.result = "failure";
  } else if (episode.steps >= TASK.maxEpisodeSteps) {
    episode.done = true;
    episode.result = "timeout";
  }
  return episode.result;
}

/** Running tally across episodes, per control mode. */
export function newScoreboard() {
  return { episodes: 0, success: 0, failure: 0, timeout: 0, totalTime: 0, totalEfficiency: 0 };
}

export function recordEpisode(board, episode) {
  board.episodes += 1;
  board[episode.result] = (board[episode.result] || 0) + 1;
  if (episode.result === "success") {
    board.totalTime += episode.t;
    board.totalEfficiency += episode.pathLength > 1e-6
      ? episode.straightLine / episode.pathLength
      : 0;
  }
  return board;
}

export function scoreSummary(board) {
  const wins = board.success || 0;
  return {
    episodes: board.episodes,
    successRate: board.episodes ? wins / board.episodes : 0,
    meanTime: wins ? board.totalTime / wins : NaN,
    meanEfficiency: wins ? board.totalEfficiency / wins : NaN,
  };
}
