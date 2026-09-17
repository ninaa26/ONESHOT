/**
 * Sections 05 and 06 — rudder authority and the helm servo.
 *
 * The rudder chart evaluates `BasicRudder` across the helm range at three yaw
 * rates; the servo figure integrates the hub's deadband / rate-limit /
 * auto-centre logic in real time.
 */

import { lineChart } from "../charts.js";
import { h, slider, readout, uval, fmt, mount, pycode } from "../ui.js";
import { RAD, DEG, rudderForce, rudderServoStep } from "../models.js";
import { state, subscribe, touched } from "../state.js";

const RUDDER_SRC = `# local inflow at the blade, not at the CG
>>u_local = state.u - state.r * y_pos
>>v_local = state.v + state.r * x_pos

v_local_rudder = tf_tree.vector_to_frame(v_local_boat, "boat", "rudder")
aoa = -np.arctan2(v_local_rudder[1], v_local_rudder[0])

>>aoa = np.clip(aoa, -aoa_limit, aoa_limit)   # stalled-blade behaviour
cl, cd = self.cl_cd(aoa, re=self.get_reynolds())
>>cl = np.clip(cl, -cl_max, cl_max)
>>cd = np.clip(cd, 0.0, cd_max)

drag = cd * q * area * effectiveness
lift = cl * q * area * effectiveness`;

const SERVO_SRC = `if abs(cmd_deg) <= deadband_deg:
    cmd_deg = 0.0

if center_tau_s > 0.0 and cmd_deg == 0.0:
>>  alpha = np.clip(dt / center_tau_s, 0.0, 1.0)
>>  self._rudder_angle_deg *= (1.0 - alpha)      # let go -> centre
else:
>>  max_step = max_rate_deg_s * dt
>>  err = cmd_deg - self._rudder_angle_deg
>>  self._rudder_angle_deg += np.clip(err, -max_step, max_step)`;

const YAW_RATES = [
  [0, "var(--c-rudder)"],
  [0.6, "var(--c-sail)"],
  [-0.6, "var(--warn)"],
];

/** Section 05 — side force across the helm range. */
function rudder() {
  let speed = 1.5;
  let eff = null;
  let aoaLimit = null;

  const chart = lineChart({
    width: 540,
    height: 290,
    xDomain: [-35, 35],
    yDomain: [-60, 60],
    xLabel: "rudder angle δ (°)",
    yLabel: "side force Fy (N)",
    xFmt: (v) => String(Math.round(v)),
    yFmt: (v) => String(Math.round(v)),
  });

  const ro = readout([
    { sep: "inflow at the blade" },
    { key: "arm", label: "x_pos (moment arm)" },
    { key: "ulocal", label: "u − r·y_pos" },
    { key: "vlocal", label: "v + r·x_pos" },
    { key: "vmag", label: "|V| at blade" },
    { sep: "angle of attack" },
    { key: "aoaraw", label: "raw α" },
    { key: "aoa", label: "after clamp", hl: true },
    { sep: "coefficients" },
    { key: "cl", label: "CL (clamped)" },
    { key: "cd", label: "CD (clamped)" },
    { sep: "force" },
    { key: "lift", label: "lift · effectiveness" },
    { key: "fy", label: "Fy in boat frame" },
    { key: "mz", label: "yaw moment", hl: true },
  ]);

  const sSpeed = slider({
    label: "Boat speed u", min: 0.2, max: 4, step: 0.05, value: speed, unit: " m/s",
    fmt: (v) => v.toFixed(2), onInput: (v) => { speed = v; render(); },
  });
  const sEff = slider({
    label: "effectiveness", hint: "force scale", min: 0.05, max: 1, step: 0.05,
    value: 0.2, fmt: (v) => v.toFixed(2),
    onInput: (v) => { eff = v; render(); },
  });
  const sLimit = slider({
    label: "aoa_limit_deg", min: 5, max: 60, step: 1, value: 35, unit: "°",
    onInput: (v) => { aoaLimit = v; render(); },
  });

  function syncFromConfig() {
    eff = Number(state.cfgs.rudder.effectiveness ?? 0.25);
    aoaLimit = Number(state.cfgs.rudder.aoa_limit_deg ?? 25);
    sEff.set(eff);
    sLimit.set(aoaLimit);
  }

  function render() {
    const cfg = { ...state.cfgs.rudder, effectiveness: eff, aoa_limit_deg: aoaLimit };
    const series = [];
    let peak = 1;

    for (const [r, color] of YAW_RATES) {
      const pts = [];
      const clampedPts = [];
      for (let d = -35; d <= 35; d += 0.5) {
        const f = rudderForce({ u: speed, v: 0, r }, d * RAD, cfg);
        pts.push([d, f.fy]);
        if (f.clamped) clampedPts.push([d, f.fy]);
        peak = Math.max(peak, Math.abs(f.fy));
      }
      series.push({ points: pts, color, width: r === 0 ? 2.4 : 1.7, dash: r === 0 ? null : "5 3" });
      if (clampedPts.length) {
        series.push({ points: [], color: "var(--bad)", dots: clampedPts.filter((_, i) => i % 6 === 0), dotR: 1.7 });
      }
    }

    chart.setYDomain([-peak * 1.15, peak * 1.15]);
    chart.setSeries(series);

    boardBits();
  }

  /** Marker + readout for the current boat state; cheap enough for the helm loop. */
  function boardBits() {
    const cfg = { ...state.cfgs.rudder, effectiveness: eff, aoa_limit_deg: aoaLimit };
    const board = state.board;
    const markers = [];
    if (board) {
      const f = rudderForce(board, board.rudderDeg * RAD, cfg);
      markers.push({ type: "dot", x: board.rudderDeg, y: f.fy, color: "var(--accent)", label: "board" });

      ro.set("arm", uval(Number(cfg.x_pos ?? 0), 2, " m"));
      ro.set("ulocal", uval(f.uLocal, 3, " m/s"));
      ro.set("vlocal", uval(f.vLocal, 3, " m/s"));
      ro.set("vmag", uval(f.speed, 3, " m/s"));
      ro.set("aoaraw", uval((f.aoaRaw ?? 0) * DEG, 2, "°"));
      ro.set("aoa", `${fmt((f.aoa ?? 0) * DEG, 2)}°${f.clamped ? ' <span class="u" style="color:var(--bad)">clamped</span>' : ""}`);
      ro.set("cl", `${fmt(f.cl, 3)}${f.clClamped ? ' <span class="u" style="color:var(--bad)">cl_max</span>' : ""}`);
      ro.set("cd", `${fmt(f.cd, 3)}${f.cdClamped ? ' <span class="u" style="color:var(--bad)">cd_max</span>' : ""}`);
      ro.set("lift", uval(f.lift, 2, " N"));
      ro.set("fy", uval(f.fy, 2, " N"));
      ro.set("mz", uval(Number(cfg.x_pos ?? 0) * f.fy - Number(cfg.y_pos ?? 0) * f.fx, 2, " N·m"));
    }
    chart.setMarkers(markers);
  }

  syncFromConfig();
  render();
  mount("rudder-fig", chart.svg);
  mount("rudder-controls", sSpeed.node, sEff.node, sLimit.node);
  mount("rudder-readout", ro.node);
  pycode(document.getElementById("rudder-code"), RUDDER_SRC);

  return {
    render: (reset) => {
      if (reset) syncFromConfig();
      render();
    },
    boardBits,
  };
}

/** Section 06 — live servo response. */
function servo() {
  const WINDOW = 10; // seconds on screen
  const DT = 1 / 60;

  const chart = lineChart({
    width: 540,
    height: 280,
    xDomain: [0, WINDOW],
    yDomain: [-40, 40],
    xLabel: "time (s)",
    yLabel: "rudder angle (°)",
    xFmt: (v) => v.toFixed(0),
    yFmt: (v) => String(Math.round(v)),
  });

  let t = 0;
  let angle = 0;
  let mode = "script"; // script | manual
  let running = false;
  const cmdTrace = [];
  const actTrace = [];
  let manualCmd = 0;

  let deadband = 1.5;
  let tau = 0.6;
  let rate = 120;

  const sDead = slider({
    label: "deadband_deg", min: 0, max: 10, step: 0.1, value: 1.5, unit: "°",
    fmt: (v) => v.toFixed(1), onInput: (v) => { deadband = v; },
  });
  const sTau = slider({
    label: "center_tau_s", hint: "release time constant", min: 0, max: 3, step: 0.05,
    value: 0.6, unit: " s", fmt: (v) => v.toFixed(2), onInput: (v) => { tau = v; },
  });
  const sRate = slider({
    label: "max_rate_deg_s", min: 10, max: 400, step: 5, value: 120, unit: " °/s",
    onInput: (v) => { rate = v; },
  });

  /** Scripted helm profile that exercises every branch of the servo. */
  function scriptedCommand(time) {
    if (time < 0.6) return 0;
    if (time < 2.6) return 30;
    if (time < 4.4) return 0; // release -> exponential centring
    if (time < 6.0) return 1.0 * Math.sin(time * 9); // inside the deadband
    if (time < 8.0) return -25;
    return 0;
  }

  const statusEl = document.getElementById("servo-status");

  function step() {
    if (!running) return;
    const cfg = { deadband_deg: deadband, center_tau_s: tau, max_rate_deg_s: rate };
    const cmd = mode === "manual" ? manualCmd : scriptedCommand(t);
    angle = rudderServoStep(angle, cmd, DT, cfg);
    cmdTrace.push([t, cmd]);
    actTrace.push([t, angle]);
    t += DT;

    if (mode === "script" && t > WINDOW) {
      t = 0;
      cmdTrace.length = 0;
      actTrace.length = 0;
    }
    if (mode === "manual") {
      while (cmdTrace.length && cmdTrace[0][0] < t - WINDOW) {
        cmdTrace.shift();
        actTrace.shift();
      }
    }
    draw();
    requestAnimationFrame(step);
  }

  function draw() {
    const shift = mode === "manual" ? Math.max(0, t - WINDOW) : 0;
    const off = (pts) => pts.map(([x, y]) => [x - shift, y]);
    const cfgRateLine = [];
    chart.setSeries([
      { points: off(cmdTrace), color: "var(--text-faint)", width: 1.4, dash: "4 3" },
      { points: off(actTrace), color: "var(--c-rudder)", width: 2.3 },
      { points: cfgRateLine, color: "var(--warn)" },
    ]);
    chart.setBands([
      { x: [0, WINDOW], color: "transparent", opacity: 0 },
    ]);
    chart.setMarkers([
      { type: "dot", x: (actTrace.length ? actTrace[actTrace.length - 1][0] : 0) - shift, y: angle, color: "var(--c-rudder)", label: `${angle.toFixed(1)}°` },
    ]);
    if (statusEl) {
      statusEl.textContent = running
        ? `${mode === "manual" ? "you have the helm" : "scripted"} · ${angle.toFixed(1)}°`
        : "paused";
    }
  }

  function reset() {
    t = 0;
    angle = 0;
    cmdTrace.length = 0;
    actTrace.length = 0;
    draw();
  }

  const btnPlay = h("button", {
    class: "chip on", text: "pause",
    onclick: () => {
      running = !running;
      btnPlay.textContent = running ? "pause" : "play";
      btnPlay.classList.toggle("on", running);
      if (running) requestAnimationFrame(step);
      else draw();
    },
  });
  const btnMode = h("button", {
    class: "chip", text: "take the helm",
    onclick: () => {
      mode = mode === "script" ? "manual" : "script";
      btnMode.textContent = mode === "manual" ? "back to script" : "take the helm";
      btnMode.classList.toggle("on", mode === "manual");
      reset();
    },
  });
  const btnReset = h("button", { class: "chip", text: "reset", onclick: reset });

  // Arrow keys drive the helm while the figure is hovered or focused.
  let hovering = false;
  const host = document.getElementById("servo-fig");
  if (host) {
    host.addEventListener("pointerenter", () => { hovering = true; });
    host.addEventListener("pointerleave", () => { hovering = false; manualCmd = 0; });
  }
  window.addEventListener("keydown", (ev) => {
    if (!hovering || mode !== "manual") return;
    if (ev.key === "ArrowLeft") { manualCmd = 35; ev.preventDefault(); }
    if (ev.key === "ArrowRight") { manualCmd = -35; ev.preventDefault(); }
  });
  window.addEventListener("keyup", (ev) => {
    if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") manualCmd = 0;
  });

  function syncFromConfig() {
    const cfg = state.cfgs.rudder;
    deadband = Number(cfg.deadband_deg ?? 1.5);
    tau = Number(cfg.center_tau_s ?? 0.6);
    rate = Number(cfg.max_rate_deg_s ?? 120);
    sDead.set(deadband);
    sTau.set(tau);
    sRate.set(rate);
  }

  syncFromConfig();
  mount("servo-fig", chart.svg);
  mount("servo-controls", sDead.node, sTau.node, sRate.node);
  mount("servo-buttons", btnPlay, btnMode, btnReset);
  pycode(document.getElementById("servo-code"), SERVO_SRC);

  running = true;
  requestAnimationFrame(step);

  return syncFromConfig;
}

export function init() {
  const { render: renderRudder, boardBits: updateRudderBoard } = rudder();
  const syncServo = servo();

  subscribe((_s, changed) => {
    if (touched(changed, "config", "cfgs")) {
      renderRudder(true);
      syncServo();
    } else if (touched(changed, "board")) {
      updateRudderBoard();
    }
  });
}
