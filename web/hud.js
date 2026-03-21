// HUD utilities for SailBench browser UI

/** @type {HTMLSpanElement | null} */
const hudSpeedEl = document.getElementById("hud-speed");
/** @type {HTMLSpanElement | null} */
const hudHeadingEl = document.getElementById("hud-heading");
/** @type {HTMLSpanElement | null} */
const hudConnEl = document.getElementById("hud-conn");
/** @type {HTMLSpanElement | null} */
const hudControlModeEl = document.getElementById("hud-control-mode");
/** @type {HTMLSpanElement | null} */
const hudSailAngleEl = document.getElementById("hud-sail-angle");
/** @type {HTMLSpanElement | null} */
const hudRudderAngleEl = document.getElementById("hud-rudder-angle");
/** @type {HTMLSpanElement | null} */
const hudSailForceEl = document.getElementById("hud-sail-force");

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

export function setControlMode(mode) {
  if (!hudControlModeEl) return;
  hudControlModeEl.textContent = mode === "rl" ? "RL AUTOPILOT" : "MANUAL";
}

export function setSailAngle(deg) {
  if (!hudSailAngleEl) return;
  hudSailAngleEl.textContent = `${deg.toFixed(1)}°`;
}

export function setRudderAngle(deg) {
  if (!hudRudderAngleEl) return;
  hudRudderAngleEl.textContent = `${deg.toFixed(1)}°`;
}

export function setSailForce(fx, fy) {
  if (!hudSailForceEl) return;
  hudSailForceEl.textContent = `Fx ${fx.toFixed(1)} N, Fy ${fy.toFixed(1)} N`;
}

/** @type {HTMLDivElement | null} */
const hudForcesChartEl = document.getElementById("hud-forces-chart");

/** Max absolute value for bar scale (N). */
const FORCES_CHART_MAX = 80;

/**
 * Update the forces bar chart from per-component forces.
 * @param {object} forces - { hull: {fx, fy}, keel: {...}, rudder: {...}, sail: {...}, total: {...} }
 */
export function updateForcesChart(forces) {
  if (!hudForcesChartEl || !forces) return;

  const names = ["hull", "keel", "rudder", "sail", "total"];
  const colors = {
    hull: "#888",
    keel: "#48f",
    rudder: "#f84",
    sail: "#8ff",
    total: "#ff4",
  };

  let html = "";
  for (const name of names) {
    const comp = forces[name];
    if (!comp || typeof comp.fx !== "number" || typeof comp.fy !== "number") {
      continue;
    }
    const fx = comp.fx;
    const fy = comp.fy;
    const fxPct = Math.min(100, Math.max(0, 50 + (fx / FORCES_CHART_MAX) * 50));
    const fyPct = Math.min(100, Math.max(0, 50 + (fy / FORCES_CHART_MAX) * 50));
    const color = colors[name] || "#fff";
    html += `
      <div class="forces-chart-row">
        <span class="forces-chart-label">${name}</span>
        <div class="forces-chart-bars">
          <div class="forces-chart-bar-wrap" title="Fx">
            <div class="forces-chart-bar" style="width:${fxPct}%;background:${color}"></div>
          </div>
          <span class="forces-chart-val">${fx.toFixed(0)}</span>
          <div class="forces-chart-bar-wrap" title="Fy">
            <div class="forces-chart-bar" style="width:${fyPct}%;background:${color}"></div>
          </div>
          <span class="forces-chart-val">${fy.toFixed(0)}</span>
        </div>
      </div>
    `;
  }
  hudForcesChartEl.innerHTML = html || "<span class='hud-label'>No forces</span>";
}

