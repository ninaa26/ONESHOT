/**
 * Shared lab state.
 *
 * Sections never talk to each other directly: the rail writes here, sections
 * subscribe. Keeps each `sections/*.js` module self-contained and independently
 * removable, the same way `boat.js` / `wind.js` / `hud.js` stay independent in
 * the 3D viewer.
 */

import { boatConfig } from "./models.js";

const listeners = new Set();

/** @type {{config: string, cfgs: object, windSpeed: number, windDirDeg: number, live: object|null}} */
export const state = {
  config: "basic_sailbot.yaml",
  cfgs: boatConfig("basic_sailbot.yaml"),
  windSpeed: 5,
  windDirDeg: 90,
  /** Latest state message from web_runner, or null when detached. */
  live: null,
};

/**
 * Subscribe to state changes.
 * @param {(state: object, changed: string[]) => void} fn
 * @returns {() => void} unsubscribe
 */
export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Apply a patch and notify subscribers. */
export function update(patch) {
  const changed = [];
  for (const [k, v] of Object.entries(patch)) {
    if (state[k] !== v) {
      state[k] = v;
      changed.push(k);
    }
  }
  if (patch.config) {
    state.cfgs = boatConfig(state.config);
    state.cfgs.sail.wind_speed = state.windSpeed;
    state.cfgs.sail.wind_dir_deg = state.windDirDeg;
    if (!changed.includes("cfgs")) changed.push("cfgs");
  }
  if (patch.windSpeed !== undefined) state.cfgs.sail.wind_speed = state.windSpeed;
  if (patch.windDirDeg !== undefined) state.cfgs.sail.wind_dir_deg = state.windDirDeg;
  if (!changed.length) return;
  for (const fn of listeners) fn(state, changed);
}

/** Convenience: has any of `keys` changed in this notification? */
export const touched = (changed, ...keys) => keys.some((k) => changed.includes(k));

// Wind is a lab-level control, so seed the config with it up front.
state.cfgs.sail.wind_speed = state.windSpeed;
state.cfgs.sail.wind_dir_deg = state.windDirDeg;
