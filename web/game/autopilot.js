/**
 * Baseline waypoint controller — the thing a trained policy has to beat.
 *
 * It takes the same 13-element observation `WaypointEnv` hands a policy and
 * returns the same normalised `[rudder, sheet]` action, so swapping it for a
 * model is a one-line change in `main.js`. Nothing here is learned: it is
 * ordinary sailing logic, written to be read and argued with.
 *
 * Strategy:
 *   1. Work out the bearing to the waypoint and the true wind angle, both
 *      measured off the bow, from the observation alone.
 *   2. If laying the mark would mean sailing inside the no-go zone, beat on the
 *      favoured tack. Same at the other end: a dead run is bear-away territory,
 *      because the sail cannot be set square.
 *   3. Tacking is a committed manoeuvre, not a heading change. A small boat
 *      with a small rudder parks itself head to wind if you ask for the new
 *      tack and let a PD loop sort it out, so there is an explicit state for
 *      it: keep the sheet in, hold the helm over, and only call it done once
 *      the wind is properly on the new side. If the boat stalls part-way
 *      through, bear away and rebuild speed before trying again.
 *   4. Otherwise steer with a PD loop on the heading error, damped by yaw rate.
 *   5. Trim the sheet from the boat's own polar when one has been generated,
 *      and from a simple half-the-wind-angle rule when it has not.
 */

import { DEG, RAD, clamp } from "../shared/physics/models.js";
import { polarAt } from "../shared/physics/analysis.js";
import { TASK, encodeAction } from "./task.js";

const DEFAULTS = {
  /** Closest angle to the wind the controller will ask for. */
  noGoDeg: 45,
  /** Furthest off the wind before the sail stops drawing usefully. */
  deadRunDeg: 165,
  /** Extra bearing the other tack must gain before it is worth tacking. */
  tackHysteresisDeg: 25,
  /** Do not start a tack slower than this — there is not enough way on. */
  minTackSpeed: 0.5,
  /** Below this, bear away and build speed before trying to point. */
  minPointSpeed: 0.45,
  /** Speed to reach before going back to pointing. */
  resumePointSpeed: 0.85,
  /** Angle to bear away to while building speed. */
  buildTwaDeg: 85,
  /** Give up on a tack after this long and bear away to rebuild speed. */
  tackTimeoutS: 9,
  /** How long to hold a bear-away recovery. */
  recoverS: 4,
  /** Heading PD gains: normalised rudder per radian, and per rad/s. */
  kp: 1.1,
  kd: 0.3,
  /** Rudder used through a tack. */
  tackRudder: 0.85,
};

/** Wrap to (-pi, pi]. */
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

/**
 * @param {string} config boat config name, used to look up its polar
 * @param {object} tuning overrides for the constants above
 */
export function createAutopilot(config, tuning = {}) {
  const k = { ...DEFAULTS, ...tuning };

  let tackSign = 0; // +1 wind to port, -1 to starboard, 0 = not beating
  let building = false; // bearing away to get the boat moving
  let phase = "sail"; // "sail" | "tack" | "recover"
  let phaseTime = 0;
  let tackTarget = 0;
  let lastSheetDeg = 30;

  /** Best sheet limit for this true wind angle, from the generated polar. */
  function sheetFromPolar(twaAbsDeg, windSpeed) {
    const polar = polarAt(config, windSpeed);
    if (!polar) return null;
    const list = polar.twa_deg;
    let i = 0;
    while (i < list.length - 2 && list[i + 1] < twaAbsDeg) i += 1;
    const span = list[i + 1] - list[i] || 1;
    const f = clamp((twaAbsDeg - list[i]) / span, 0, 1);
    return polar.best_sheet_deg[i] + f * (polar.best_sheet_deg[i + 1] - polar.best_sheet_deg[i]);
  }

  /** Sheet trim for a target wind angle, eased gradually like a real winch. */
  function trimFor(twaAbsDeg, windSpeed) {
    const fromPolar = sheetFromPolar(twaAbsDeg, windSpeed);
    const wanted = fromPolar ?? clamp(twaAbsDeg * 0.5 - 4, 8, TASK.maxSailDeg);
    lastSheetDeg += clamp(wanted - lastSheetDeg, -6, 6);
    return clamp(lastSheetDeg, 0, TASK.maxSailDeg);
  }

  return {
    name: "baseline",

    reset() {
      tackSign = 0;
      building = false;
      phase = "sail";
      phaseTime = 0;
      tackTarget = 0;
      lastSheetDeg = 30;
    },

    /**
     * @param {number[]} obs the 13-element observation
     * @param {number} dt seconds since the last call, for the phase timers
     * @returns {[number, number]} normalised [rudder, sheet]
     */
    act(obs, dt = 0.02) {
      // Undo the observation's normalisation. Bearings are measured off the
      // bow and increase to port, matching the simulator's own convention.
      const bearingWp = Math.atan2(obs[4], obs[3]);
      const windSpeed = obs[10] * TASK.windSpeedScale;
      const yawRate = obs[7] * TASK.yawRateScale;
      const surge = obs[5] * TASK.speedScale;

      // wind_boat points the way the wind blows TO, so the breeze is astern of it.
      const twa = Math.atan2(-obs[9], -obs[8]);

      const noGo = k.noGoDeg * RAD;
      const deadRun = k.deadRunDeg * RAD;
      phaseTime += dt;

      // --- committed manoeuvres ------------------------------------------

      if (phase === "tack") {
        const through = Math.sign(twa) === tackTarget && Math.abs(twa) > noGo * 0.8;
        if (through) {
          phase = "sail";
          phaseTime = 0;
          tackSign = tackTarget;
        } else if (phaseTime > k.tackTimeoutS) {
          // Stuck head to wind. Bear away onto the old tack and try later.
          phase = "recover";
          phaseTime = 0;
          tackSign = -tackTarget;
        } else {
          // Helm hard over toward the new tack, sheet kept in.
          return [
            clamp(tackTarget * k.tackRudder, -1, 1),
            encodeAction(0, trimFor(k.noGoDeg, windSpeed))[1],
          ];
        }
      }

      if (phase === "recover") {
        if (phaseTime > k.recoverS && surge > k.minTackSpeed) {
          phase = "sail";
          phaseTime = 0;
        } else {
          // Bear away to about a beam reach on the current tack and build speed.
          const target = tackSign * 90 * RAD;
          const delta = wrap(twa - target);
          return [
            clamp(-k.kp * delta + k.kd * yawRate, -1, 1),
            encodeAction(0, trimFor(90, windSpeed))[1],
          ];
        }
      }

      // --- pick a target wind angle --------------------------------------

      const twaIfWeLay = wrap(twa - bearingWp);
      let targetTwa = twaIfWeLay;

      // A stationary boat cannot point. With no flow over the keel and rudder
      // there is nothing to steer with either, so bear away to a reach, get
      // moving, and only then start working upwind. Without this the boat
      // luffs slowly head to wind and sits there for the whole episode.
      if (building && surge > k.resumePointSpeed) building = false;
      else if (!building && surge < k.minPointSpeed) building = true;

      if (building) {
        const side = tackSign || Math.sign(twa) || 1;
        const target = side * k.buildTwaDeg * RAD;
        const delta = wrap(twa - target);
        return [
          clamp(-k.kp * delta + k.kd * yawRate, -1, 1),
          encodeAction(0, trimFor(k.buildTwaDeg, windSpeed))[1],
        ];
      }

      if (Math.abs(twaIfWeLay) < noGo) {
        // Cannot lay it: beat. Favour the tack that points closer to the mark,
        // but stay on the current one until the other is clearly better.
        const wanted = Math.sign(twaIfWeLay) || Math.sign(twa) || 1;
        if (tackSign === 0) {
          tackSign = Math.sign(twa) || wanted;
        } else if (wanted !== tackSign
          && Math.abs(bearingWp) * DEG > k.tackHysteresisDeg
          && surge > k.minTackSpeed) {
          phase = "tack";
          phaseTime = 0;
          tackTarget = wanted;
          return [
            clamp(tackTarget * k.tackRudder, -1, 1),
            encodeAction(0, trimFor(k.noGoDeg, windSpeed))[1],
          ];
        }
        targetTwa = tackSign * noGo;
      } else if (Math.abs(twaIfWeLay) > deadRun) {
        // Dead downwind: heat it up so the sail keeps drawing.
        targetTwa = Math.sign(twaIfWeLay || 1) * deadRun;
        tackSign = 0;
      } else {
        tackSign = 0;
      }

      // Turning by delta changes the true wind angle by -delta.
      const delta = wrap(twa - targetTwa);
      // Positive rudder turns to starboard, positive delta asks for port.
      const rudder = clamp(-k.kp * delta + k.kd * yawRate, -1, 1);
      const sheetDeg = trimFor(Math.abs(targetTwa) * DEG, windSpeed);

      return [rudder, encodeAction(0, sheetDeg)[1]];
    },
  };
}
