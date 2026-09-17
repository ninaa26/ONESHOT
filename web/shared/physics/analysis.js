/**
 * Derived numbers from the baked polar sweeps.
 *
 * Kept out of `models.js` on purpose: this is analysis of simulator output,
 * not a port of simulator physics.
 */

import { DATA } from "./models.js";

/** All polars generated for a config, or an empty list. */
export const polarsFor = (config) => DATA.polars[config] || [];

/** The polar closest to a target wind speed. */
export function polarAt(config, windSpeed) {
  const list = polarsFor(config);
  if (!list.length) return null;
  return list.reduce((best, p) => (
    Math.abs(p.wind_speed - windSpeed) < Math.abs(best.wind_speed - windSpeed) ? p : best
  ), list[0]);
}

/**
 * Targets a sailor would read off a polar.
 *
 * `noGoDeg` is reported as the closest angle where the boat still makes 25% of
 * its top speed — the sweep never returns exactly zero, so a hard "speed > 0"
 * threshold would claim the boat sails dead upwind.
 */
export function polarStats(polar) {
  if (!polar) return null;
  const { twa_deg: twa, speed, vmg, best_sheet_deg: sheet } = polar;

  const top = Math.max(...speed);
  const topIdx = speed.indexOf(top);

  let upIdx = 0;
  let downIdx = 0;
  vmg.forEach((value, i) => {
    if (value > vmg[upIdx]) upIdx = i;
    if (value < vmg[downIdx]) downIdx = i;
  });

  const threshold = 0.25 * top;
  let noGoIdx = speed.findIndex((s) => s >= threshold);
  if (noGoIdx < 0) noGoIdx = 0;

  return {
    windSpeed: polar.wind_speed,
    top,
    topTwa: twa[topIdx],
    topSheet: sheet[topIdx],
    upwind: { twa: twa[upIdx], speed: speed[upIdx], vmg: vmg[upIdx], sheet: sheet[upIdx] },
    downwind: { twa: twa[downIdx], speed: speed[downIdx], vmg: -vmg[downIdx], sheet: sheet[downIdx] },
    noGoDeg: twa[noGoIdx],
    threshold,
    leewayMax: Math.max(...polar.leeway_deg.map(Math.abs)),
  };
}
