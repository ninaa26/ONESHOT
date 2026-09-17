/**
 * Section 12 — live telemetry.
 *
 * Attaches to the same `ws://<host>:8765/sim` endpoint the 3D viewer uses
 * (see `web/main.js`), mirrors the running boat, and cross-checks the forces
 * the backend reports against the in-page models.
 */

import { lineChart } from "../charts.js";
import { h, readout, uval, fmt, mount } from "../ui.js";
import { RAD, solveAt } from "../../shared/physics/models.js";
import { state, update } from "../state.js";

const PORT = 8765;
const WINDOW = 30; // seconds of trace

/** Same URL rule as web/main.js so both views follow the same backend. */
function wsUrl() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.hostname || "127.0.0.1";
  return `${proto}://${host}:${PORT}/sim`;
}

export function init() {
  let socket = null;
  let autoReconnect = false;
  let reconnectDelay = 1000;
  let t0 = null;

  const speedTrace = [];
  const sailTrace = [];
  const totalTrace = [];

  const chart = lineChart({
    width: 540,
    height: 260,
    xDomain: [0, WINDOW],
    yDomain: [0, 3],
    xLabel: "seconds",
    yLabel: "m/s  ·  N ÷ 50",
    xFmt: (v) => v.toFixed(0),
    yFmt: (v) => v.toFixed(1),
  });

  const ro = readout([
    { sep: "backend" },
    { key: "mode", label: "control mode" },
    { key: "t", label: "sim time" },
    { key: "pos", label: "position" },
    { key: "heading", label: "heading" },
    { sep: "velocity" },
    { key: "u", label: "surge u" },
    { key: "v", label: "sway v" },
    { key: "r", label: "yaw rate r" },
    { key: "speed", label: "speed", hl: true },
    { sep: "controls" },
    { key: "sail", label: "sail angle" },
    { key: "rudder", label: "rudder angle" },
    { key: "wind", label: "wind" },
    { sep: "forces reported by python (N)" },
    { key: "fhull", label: "hull" },
    { key: "fkeel", label: "keel" },
    { key: "fsail", label: "sail" },
    { key: "frudder", label: "rudder" },
    { key: "ftotal", label: "Σ", hl: true },
    { sep: "cross-check" },
    { key: "residual", label: "|Δ| vs in-page models" },
    { key: "waypoint", label: "waypoint" },
  ]);

  const pulse = document.getElementById("live-pulse");
  const statusEl = document.getElementById("live-status");

  function setStatus(text, cls = "") {
    if (statusEl) statusEl.textContent = text;
    if (pulse) pulse.className = `pulse ${cls}`;
  }

  function handle(msg) {
    if (!msg || msg.type !== "state") return;
    if (t0 === null) t0 = msg.t;
    const t = msg.t - t0;

    const { velocity_body: vb, heading, position } = msg.boat;
    const speed = Math.hypot(vb.u, vb.v);
    const forces = msg.forces || {};

    ro.set("mode", (msg.control && msg.control.mode) || "manual");
    ro.set("t", uval(msg.t, 1, " s"));
    ro.set("pos", `${fmt(position.x, 1)}, ${fmt(position.y, 1)}<span class="u">m</span>`);
    ro.set("heading", uval(heading.deg, 1, "°"));
    ro.set("u", uval(vb.u, 3, " m/s"));
    ro.set("v", uval(vb.v, 3, " m/s"));
    ro.set("r", uval(vb.r, 3, " rad/s"));
    ro.set("speed", uval(speed, 3, " m/s"));
    ro.set("sail", msg.sail ? uval(msg.sail.angle_deg, 1, "°") : "—");
    ro.set("rudder", msg.rudder ? uval(msg.rudder.angle_deg, 1, "°") : "—");
    ro.set("wind", msg.wind ? `${fmt(msg.wind.speed, 1)} m/s → ${fmt(msg.wind.dir_deg, 0)}°` : "—");
    ro.set("waypoint", msg.waypoint ? `${fmt(msg.waypoint.x, 1)}, ${fmt(msg.waypoint.y, 1)}` : "—");

    for (const key of ["hull", "keel", "sail", "rudder", "total"]) {
      const f = forces[key];
      ro.set(`f${key}`, f ? `${fmt(f.fx, 1)}, ${fmt(f.fy, 1)}` : "—");
    }

    // Cross-check: feed the backend's own state through the in-page models.
    if (msg.wind && msg.sail && forces.total) {
      const cfgs = state.cfgs;
      const prevSpeed = cfgs.sail.wind_speed;
      const prevDir = cfgs.sail.wind_dir_deg;
      cfgs.sail.wind_speed = msg.wind.speed;
      cfgs.sail.wind_dir_deg = msg.wind.dir_deg;
      const mirror = solveAt(
        { u: vb.u, v: vb.v, r: vb.r },
        heading.deg * RAD,
        cfgs,
        {
          // The backend reports the resolved sail angle, so drive the mirror
          // with a sheet limit that reproduces it exactly.
          sheetRad: Math.abs(msg.sail.angle_deg) * RAD,
          rudderRad: (msg.rudder ? msg.rudder.angle_deg : 0) * RAD,
        },
      );
      const d = Math.hypot(
        mirror.total.fx - forces.total.fx,
        mirror.total.fy - forces.total.fy,
      );
      const mag = Math.hypot(forces.total.fx, forces.total.fy);
      ro.set("residual", `${fmt(d, 2)} N <span class="u">(${fmt((d / Math.max(mag, 1e-6)) * 100, 2)}%)</span>`);
      cfgs.sail.wind_speed = prevSpeed;
      cfgs.sail.wind_dir_deg = prevDir;
    }

    speedTrace.push([t, speed]);
    sailTrace.push([t, forces.sail ? Math.hypot(forces.sail.fx, forces.sail.fy) / 50 : 0]);
    totalTrace.push([t, forces.total ? Math.hypot(forces.total.fx, forces.total.fy) / 50 : 0]);
    while (speedTrace.length && speedTrace[0][0] < t - WINDOW) {
      speedTrace.shift();
      sailTrace.shift();
      totalTrace.shift();
    }

    const shift = Math.max(0, t - WINDOW);
    const off = (pts) => pts.map(([x, y]) => [x - shift, y]);
    const peak = Math.max(3, ...speedTrace.map((p) => p[1]), ...totalTrace.map((p) => p[1]));
    chart.setYDomain([0, peak * 1.1]);
    chart.setSeries([
      { points: off(speedTrace), color: "var(--accent)", width: 2 },
      { points: off(sailTrace), color: "var(--c-sail)", width: 1.4, dash: "4 3" },
      { points: off(totalTrace), color: "var(--c-total)", width: 1.4, opacity: 0.8 },
    ]);

    update({ live: msg });
  }

  function connect() {
    if (socket) return;
    setStatus("connecting…");
    try {
      socket = new WebSocket(wsUrl());
    } catch {
      setStatus("unavailable", "err");
      socket = null;
      return;
    }
    socket.onopen = () => {
      reconnectDelay = 1000;
      setStatus(`connected · ${wsUrl()}`, "on");
    };
    socket.onmessage = (ev) => {
      try {
        handle(JSON.parse(ev.data));
      } catch {
        // ignore malformed frames, same as web/main.js
      }
    };
    socket.onerror = () => setStatus("no backend on :8765", "err");
    socket.onclose = () => {
      socket = null;
      t0 = null;
      setStatus("disconnected");
      update({ live: null });
      if (autoReconnect) {
        window.setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 10000);
      }
    };
  }

  function disconnect() {
    autoReconnect = false;
    if (socket) socket.close();
  }

  const btnConnect = h("button", {
    class: "chip", text: "connect",
    onclick: () => {
      autoReconnect = true;
      connect();
    },
  });
  const btnDisconnect = h("button", { class: "chip", text: "disconnect", onclick: disconnect });
  const btnCopy = h("button", {
    class: "chip", text: "copy backend command",
    onclick: () => {
      const cmd = `uv run python -m sailbench.sim.web_runner --config ${state.config} --fps 60`;
      navigator.clipboard?.writeText(cmd);
      btnCopy.textContent = "copied ✓";
      window.setTimeout(() => { btnCopy.textContent = "copy backend command"; }, 1600);
    },
  });

  mount("live-fig", chart.svg);
  mount("live-readout", ro.node);
  mount("live-buttons", btnConnect, btnDisconnect, btnCopy);
  setStatus("disconnected");

  // One quiet attempt on load: if the sim is already running, just attach.
  connect();
}
