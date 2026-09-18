/**
 * JavaScript mirror of the SailBench physics core.
 *
 * Every function here is a line-for-line port of the Python model it names, so
 * the page can evaluate the real models interactively instead of showing
 * pre-baked curves. The only external input is the NeuralFoil CL/CD table in
 * `data.js`, sampled from the exact same `Foil.cl_cd` path the simulator uses.
 *
 * Ported from:
 *   sailbench/dynamics/basic_hull_model.py   BasicHullModel.compute
 *   sailbench/foils/basic_sail.py            BasicSail.compute
 *   sailbench/foils/basic_keel.py            BasicKeel.compute
 *   sailbench/foils/basic_rudder.py          BasicRudder.compute
 *   sailbench/sim/sailboat_hub.py            SailboatHub._resolve_sail_angle_from_sheet,
 *                                            _update_dynamic_frames, _forces, step
 *
 * `web/physics/index.html` validates this port against forces recorded from the
 * Python core (see the "Mirror check" section).
 */

import { PHYSICS_DATA } from "./data.js";

export const DEG = 180 / Math.PI;
export const RAD = Math.PI / 180;

export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
export const hypot = Math.hypot;

/** Rotate a 2-vector by `theta` (the tf_tree child -> parent direction). */
export function rot([x, y], theta) {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [c * x - s * y, s * x + c * y];
}

/** Wrap degrees to (-180, 180]. */
export function wrap180(deg) {
  return ((((deg + 180) % 360) + 360) % 360) - 180;
}

// --- foil coefficients --------------------------------------------------

const ALPHA = PHYSICS_DATA.foil.alpha_deg;
const A0 = ALPHA[0];
const A_STEP = ALPHA[1] - ALPHA[0];

/**
 * NACA0012 CL/CD from the NeuralFoil table, mirroring `Foil.cl_cd`:
 * alpha is clipped to the config's [alpha_min, alpha_max] before lookup.
 *
 * @param {number} alphaRad angle of attack
 * @param {object} cfg component config (supplies res / alpha limits)
 */
export function clCd(alphaRad, cfg) {
  const re = String(Math.round(Number((cfg.res && cfg.res[0]) ?? cfg.re ?? 1e5)));
  const table = PHYSICS_DATA.foil.re[re] || PHYSICS_DATA.foil.re["100000"];
  const aMin = cfg.alpha_min ?? -20;
  const aMax = cfg.alpha_max ?? 20;
  const a = clamp(alphaRad * DEG, aMin, aMax);

  const t = (a - A0) / A_STEP;
  const i = clamp(Math.floor(t), 0, ALPHA.length - 2);
  const f = clamp(t - i, 0, 1);
  return {
    cl: table.cl[i] + f * (table.cl[i + 1] - table.cl[i]),
    cd: table.cd[i] + f * (table.cd[i + 1] - table.cd[i]),
    alphaDeg: a,
  };
}

/** Whole-sweep accessor for plotting the raw polars. */
export function foilCurve(reKey) {
  const table = PHYSICS_DATA.foil.re[String(reKey)];
  return { alpha: ALPHA, cl: table.cl, cd: table.cd };
}

// --- hull ---------------------------------------------------------------

/**
 * `BasicHullModel.compute` — quadratic drag with coefficients derived from
 * hull geometry rather than tuned per-axis constants.
 */
export function hullCoeffs(cfg) {
  const L = Number(cfg.L);
  const B = Number(cfg.B);
  const T = Number(cfg.T);
  const rho = Number(cfg.rho ?? 1000.0);

  const S = 1.7 * L * (B + T); // wetted-surface estimate (Denny's formula)
  const aSide = L * T; // lateral plane

  return {
    L, B, T, rho, S, aSide,
    kU: 0.5 * rho * S * 0.004, // 0.004 = flat-plate friction coefficient
    kV: 0.5 * rho * aSide, // bluff-body sway, Cd = 1 implied
    kR: (1 / 8) * rho * T * L ** 4,
  };
}

export function hullForce(state, cfg) {
  const { kU, kV, kR, ...rest } = hullCoeffs(cfg);
  const { u, v, r } = state;
  return {
    fx: -kU * u * Math.abs(u),
    fy: -kV * v * Math.abs(v),
    mz: -kR * r * Math.abs(r),
    kU, kV, kR, ...rest,
  };
}

// --- sail ---------------------------------------------------------------

/**
 * `BasicSail.compute` with every intermediate exposed for the UI.
 *
 * @param {{u:number,v:number}} state body-frame velocity
 * @param {number} headingRad boat heading (world)
 * @param {number} sailRad resolved sail angle relative to the boat
 * @param {object} cfg sail config
 */
export function sailForce(state, headingRad, sailRad, cfg) {
  const windSpeed = Number(cfg.wind_speed ?? 0);
  const windRad = Number(cfg.wind_dir_deg ?? 0) * RAD;
  const windWorld = [windSpeed * Math.cos(windRad), windSpeed * Math.sin(windRad)];

  const vWorld = rot([state.u, state.v], headingRad);
  const awWorld = [windWorld[0] - vWorld[0], windWorld[1] - vWorld[1]];
  const awSail = rot(awWorld, -(headingRad + sailRad));
  const awBoat = rot(awWorld, -headingRad);
  const speed = hypot(awSail[0], awSail[1]);

  const out = {
    windWorld, vWorld, awWorld, awBoat, awSail,
    awSpeed: speed,
    awaDeg: DEG * Math.atan2(-awBoat[1], -awBoat[0]),
    aoaRad: 0, clRaw: 0, cl: 0, cd: 0, luffScale: 1,
    q: 0, lift: 0, drag: 0, fx: 0, fy: 0, mz: 0, luffing: false,
  };
  if (speed < 1e-6) return out;

  const aoa = Math.atan2(awSail[1], awSail[0]);
  const { cl: clRaw, cd } = clCd(aoa, cfg);

  // Luffing ramp: near-centreline flow flaps the sail and kills lift authority.
  const aoaDegAbs = Math.abs(aoa * DEG);
  const luffDeg = Number(cfg.luff_deg ?? 7.5);
  const luffRamp = Math.max(Number(cfg.luff_ramp_deg ?? 4.0), 1e-6);
  const luffScale = clamp((aoaDegAbs - luffDeg) / luffRamp, 0, 1);
  const cl = clRaw * luffScale;

  const rho = Number(cfg.air_density ?? 1.225);
  const q = 0.5 * rho * speed * speed;
  const area = Number(cfg.area ?? 1);
  const lift = cl * q * area;
  const drag = cd * q * area;

  const fSail = rot([drag, lift], aoa); // fluid -> sail
  const fBoat = rot(fSail, sailRad); // sail -> boat

  return {
    ...out,
    aoaRad: aoa, aoaDeg: aoa * DEG,
    clRaw, cl, cd, luffScale,
    luffing: luffScale < 1,
    q, lift, drag, area, rho,
    fSail, fx: fBoat[0], fy: fBoat[1],
  };
}

/** The luff ramp on its own, for plotting. */
export function luffScale(aoaDeg, cfg) {
  const luffDeg = Number(cfg.luff_deg ?? 7.5);
  const luffRamp = Math.max(Number(cfg.luff_ramp_deg ?? 4.0), 1e-6);
  return clamp((Math.abs(aoaDeg) - luffDeg) / luffRamp, 0, 1);
}

/**
 * `SailboatHub._resolve_sail_angle_from_sheet` — the sail weather-vanes to the
 * apparent wind and the sheet acts as a purely geometric stop.
 */
export function resolveSheet(state, headingRad, sheetLimitRad, cfg, lastSailRad = 0) {
  const windSpeed = Number(cfg.wind_speed ?? 0);
  const windRad = Number(cfg.wind_dir_deg ?? 0) * RAD;
  const windWorld = [windSpeed * Math.cos(windRad), windSpeed * Math.sin(windRad)];
  const vWorld = rot([state.u, state.v], headingRad);
  const awWorld = [windWorld[0] - vWorld[0], windWorld[1] - vWorld[1]];
  const awBoat = rot(awWorld, -headingRad);
  const awa = Math.atan2(-awBoat[1], -awBoat[0]);

  const sheetLimit = clamp(Math.abs(sheetLimitRad), 0, 0.5 * Math.PI);
  if (sheetLimit <= 0) {
    return { sailRad: 0, awa, freeMag: 0, boomMag: 0, sheetLimit, sheeted: true, awBoat };
  }

  const freeMag = clamp(Math.abs(awa), 0, 0.5 * Math.PI);
  const boomMag = Math.min(freeMag, sheetLimit);

  let windSide = Math.sign(awBoat[1]);
  if (windSide === 0) {
    windSide = Math.sign(lastSailRad) || -1;
  }

  return {
    sailRad: -windSide * boomMag,
    awa, freeMag, boomMag, sheetLimit, windSide, awBoat,
    sheeted: boomMag >= sheetLimit - 1e-12 && freeMag > sheetLimit,
  };
}

// --- keel ---------------------------------------------------------------

/**
 * `BasicKeel.compute` — keel is fixed on the centreline; AoA is the leeway angle.
 *
 * `fluid` is the frozen fluid-frame direction from the transform tree. The
 * Python model reads CL/CD from the *current* state but rotates the result
 * through whatever the tree last stored, so the caller supplies it.
 */
export function keelForce(state, cfg, fluid) {
  const { u, v } = state;
  const localTrack = Math.atan2(v, u);
  const { cl, cd } = clCd(localTrack, cfg);

  const rho = Number(cfg.water_density ?? 1000.0);
  const speed = hypot(u, v);
  const q = 0.5 * rho * speed * speed;
  const area = Number(cfg.area ?? 1);
  const lift = cl * q * area;
  const drag = cd * q * area;

  const [cF, sF] = fluid || fluidFrame(state);
  return {
    fx: cF * drag - sF * lift,
    fy: sF * drag + cF * lift,
    mz: 0,
    leewayDeg: localTrack * DEG,
    cl, cd, q, lift, drag, speed, area,
  };
}

/** Fluid frame direction for a state: opposite the velocity, as the hub sets it. */
export function fluidFrame({ u, v }) {
  const speed = hypot(u, v);
  return speed > 1e-6 ? [-u / speed, -v / speed] : [1, 0];
}

// --- rudder -------------------------------------------------------------

/** `BasicRudder.compute` — local inflow at the blade, including yaw rate. */
export function rudderForce(state, rudderRad, cfg) {
  const xPos = Number(cfg.x_pos ?? 0);
  const yPos = Number(cfg.y_pos ?? 0);
  const uLocal = state.u - state.r * yPos;
  const vLocal = state.v + state.r * xPos;
  const speed = hypot(uLocal, vLocal);

  const base = {
    uLocal, vLocal, speed, xPos, yPos,
    aoaRaw: 0, aoaDeg: 0, clamped: false,
    cl: 0, cd: 0, clClamped: false, cdClamped: false,
    lift: 0, drag: 0, fx: 0, fy: 0, mz: 0,
  };
  if (speed < 1e-6) return base;

  const vRudder = rot([uLocal, vLocal], -rudderRad);
  const aoaRaw = -Math.atan2(vRudder[1], vRudder[0]);

  const aoaLimit = Number(cfg.aoa_limit_deg ?? 25.0) * RAD;
  const aoa = clamp(aoaRaw, -aoaLimit, aoaLimit);

  const raw = clCd(aoa, cfg);
  const clMax = Number(cfg.cl_max ?? 1.0);
  const cdMax = Number(cfg.cd_max ?? 1.2);
  const cl = clamp(raw.cl, -clMax, clMax);
  const cd = clamp(raw.cd, 0, cdMax);

  const rho = Number(cfg.water_density ?? 1000.0);
  const q = 0.5 * rho * speed * speed;
  const area = Number(cfg.area ?? 1);
  const eff = Number(cfg.effectiveness ?? 0.25);
  const drag = cd * q * area * eff;
  const lift = cl * q * area * eff;

  const fRudder = rot([drag, lift], aoa); // fluid -> rudder
  const fBoat = rot(fRudder, rudderRad); // rudder -> boat

  return {
    ...base,
    aoaRaw, aoa, aoaDeg: aoa * DEG,
    clamped: Math.abs(aoaRaw) > aoaLimit + 1e-12,
    cl, cd, clRaw: raw.cl, cdRaw: raw.cd,
    clClamped: Math.abs(raw.cl) > clMax + 1e-9,
    cdClamped: raw.cd > cdMax + 1e-9,
    q, lift, drag, eff, area,
    fx: fBoat[0], fy: fBoat[1], mz: 0,
  };
}

/**
 * `SailboatHub._update_dynamic_frames` rudder servo: deadband, exponential
 * return-to-centre when released, rate limit otherwise.
 *
 * @returns {number} the new servo angle in degrees
 */
export function rudderServoStep(currentDeg, commandDeg, dt, cfg) {
  const deadband = Number(cfg.deadband_deg ?? 1.5);
  const centerTau = Number(cfg.center_tau_s ?? 0.6);
  const maxRate = Number(cfg.max_rate_deg_s ?? 120.0);

  let cmd = commandDeg;
  if (Math.abs(cmd) <= deadband) cmd = 0.0;

  if (centerTau > 0 && cmd === 0) {
    const alpha = clamp(dt / centerTau, 0, 1);
    return (1 - alpha) * currentDeg;
  }
  const maxStep = maxRate * dt;
  return currentDeg + clamp(cmd - currentDeg, -maxStep, maxStep);
}

// --- assembly -----------------------------------------------------------

/**
 * The transform-tree state `SailboatHub` carries *between* steps.
 *
 * `_update_dynamic_frames` runs once per step, after integration, so the boat,
 * fluid, sail and rudder frames are all held fixed while RK4 evaluates its four
 * stages. Freezing them here is what keeps this port's stepping faithful — and
 * stable, since a sail angle that re-resolves inside every stage feeds back on
 * itself.
 */
export function freezeFrames(state, headingRad, cfgs, { sheetRad, rudderRad, lastSailRad = 0 }) {
  const sheet = resolveSheet(state, headingRad, sheetRad, cfgs.sail, lastSailRad);
  return {
    headingRad,
    sailRad: sheet.sailRad,
    rudderRad,
    fluid: fluidFrame(state),
    sheet,
  };
}

/**
 * `SailboatHub._forces` — sum component forces and moments about the CG.
 *
 * @param {{u:number,v:number,r:number}} state body-frame velocities
 * @param {object} frames frozen transform tree from `freezeFrames`
 * @returns per-component forces plus totals and the resolved sail angle
 */
export function allForces(state, frames, cfgs) {
  const { headingRad, sailRad, rudderRad, fluid, sheet } = frames;

  const hull = hullForce(state, cfgs.hull);
  const keel = keelForce(state, cfgs.keel, fluid);
  const sail = sailForce(state, headingRad, sailRad, cfgs.sail);
  const rudder = rudderForce(state, rudderRad, cfgs.rudder);

  const parts = [
    ["hull", hull, cfgs.hull],
    ["keel", keel, cfgs.keel],
    ["sail", sail, cfgs.sail],
    ["rudder", rudder, cfgs.rudder],
  ];

  let fx = 0;
  let fy = 0;
  let mz = 0;
  const moments = {};
  for (const [name, force, cfg] of parts) {
    const xPos = Number(cfg.x_pos ?? 0);
    const yPos = Number(cfg.y_pos ?? 0);
    const m = xPos * force.fy - yPos * force.fx + (force.mz || 0);
    moments[name] = m;
    fx += force.fx;
    fy += force.fy;
    mz += m;
  }

  return {
    hull, keel, sail, rudder, sheet, sailRad, moments,
    total: { fx, fy, mz },
  };
}

/**
 * Body-frame state derivative, mirroring `SailboatHub.step.dynamics`.
 *
 * Only the velocities vary across an RK4 stage; the frames come from the
 * previous step, exactly as they do in Python.
 */
export function derivative(arr, frames, cfgs) {
  const [, , c, s, u, v, r] = arr;
  const res = allForces({ u, v, r }, frames, cfgs);
  const { fx, fy, mz } = res.total;
  const m = Number(cfgs.boat.mass ?? cfgs.boat.m ?? 27.0);
  const iz = Number(cfgs.boat.inertia_z ?? cfgs.boat.Iz ?? 25.0);

  return [
    u * c - v * s,
    u * s + v * c,
    -r * s,
    r * c,
    fx / m + r * v,
    fy / m - r * u,
    mz / iz,
  ];
}

/** One RK4 step over frozen frames, mirroring `sailbench/solvers/rk4.py`. */
export function rk4Step(arr, dt, cfgs, frames) {
  const f = (a) => derivative(a, frames, cfgs);
  const add = (a, b, k) => a.map((val, i) => val + k * b[i]);
  const k1 = f(arr);
  const k2 = f(add(arr, k1, dt / 2));
  const k3 = f(add(arr, k2, dt / 2));
  const k4 = f(add(arr, k3, dt));
  const out = arr.map((val, i) => val + (dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])) / 6);
  // normalise the heading pair, as State.from_array does
  const n = hypot(out[2], out[3]) || 1;
  out[2] /= n;
  out[3] /= n;
  return out;
}

/**
 * One whole `SailboatHub.step`: integrate over the frames carried in, then
 * rebuild them from the new state.
 *
 * @returns {{arr: number[], frames: object}} next state and next frozen frames
 */
export function stepSim(arr, dt, cfgs, { sheetRad, rudderRad }, frames) {
  const heading = () => Math.atan2(arr[3], arr[2]);
  const current = frames || freezeFrames(
    { u: arr[4], v: arr[5], r: arr[6] },
    heading(),
    cfgs,
    { sheetRad, rudderRad },
  );

  const next = rk4Step(arr, dt, cfgs, current);
  const nextFrames = freezeFrames(
    { u: next[4], v: next[5], r: next[6] },
    Math.atan2(next[3], next[2]),
    cfgs,
    { sheetRad, rudderRad, lastSailRad: current.sailRad },
  );
  return { arr: next, frames: nextFrames };
}

/** Instantaneous solve: freeze the frames at this very state, then evaluate. */
export function solveAt(state, headingRad, cfgs, ctrl) {
  return allForces(state, freezeFrames(state, headingRad, cfgs, ctrl), cfgs);
}

// --- config helpers -----------------------------------------------------

/** Merge a raw YAML config into the shape the model functions expect. */
export function boatConfig(name, overrides = {}) {
  const raw = PHYSICS_DATA.configs[name];
  const clone = JSON.parse(JSON.stringify(raw));
  return {
    name,
    simulation: clone.simulation,
    boat: clone.boat,
    // NOTE: BasicHullModel reads p["rho"], not the config's rho_water, so the
    // hull always falls back to 1000 kg/m3. Mirrored here on purpose.
    hull: clone.hull,
    keel: clone.keel,
    sail: { ...clone.sail, ...overrides.sail },
    rudder: clone.rudder,
    raw,
  };
}

export const CONFIG_NAMES = Object.keys(PHYSICS_DATA.configs);
export const DATA = PHYSICS_DATA;
