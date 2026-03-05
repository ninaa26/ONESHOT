// HUD utilities for SailBench browser UI

/** @type {HTMLSpanElement | null} */
const hudSpeedEl = document.getElementById("hud-speed");
/** @type {HTMLSpanElement | null} */
const hudHeadingEl = document.getElementById("hud-heading");
/** @type {HTMLSpanElement | null} */
const hudConnEl = document.getElementById("hud-conn");
/** @type {HTMLSpanElement | null} */
const hudSailAngleEl = document.getElementById("hud-sail-angle");
/** @type {HTMLSpanElement | null} */
const hudRudderAngleEl = document.getElementById("hud-rudder-angle");

export function setSpeed(valueMs) {
  if (!hudSpeedEl) return;
  hudSpeedEl.textContent = `${valueMs.toFixed(2)} m/s`;
}

export function setHeading(deg) {
  if (!hudHeadingEl) return;
  const hdg = ((deg % 360) + 360) % 360;
  hudHeadingEl.textContent = `${hdg.toFixed(1)}°`;
}

export function setConnectionStatus(text) {
  if (!hudConnEl) return;
  hudConnEl.textContent = text;
}

export function setSailAngle(deg) {
  if (!hudSailAngleEl) return;
  hudSailAngleEl.textContent = `${deg.toFixed(1)}°`;
}

export function setRudderAngle(deg) {
  if (!hudRudderAngleEl) return;
  hudRudderAngleEl.textContent = `${deg.toFixed(1)}°`;
}

