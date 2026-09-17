/**
 * Sections 02 and 03 — geometric sheeting and the luff ramp.
 *
 * Both figures evaluate the ported sail model directly, so moving a slider
 * shows exactly what `BasicSail` / `_resolve_sail_angle_from_sheet` would do.
 */

import { lineChart } from "../charts.js";
import { slider, mount, pycode, chips } from "../ui.js";
import { RAD, clCd, luffScale, clamp } from "../models.js";
import { state, subscribe, touched } from "../state.js";

const SHEET_SRC = `def _resolve_sail_angle_from_sheet(self, state, sheet_limit_rad):
    awa = np.arctan2(-apparent_wind_boat[1], -apparent_wind_boat[0])

    sheet_limit = np.clip(abs(sheet_limit_rad), 0.0, 0.5 * np.pi)
    if sheet_limit <= 0.0:
        return 0.0
>>  free_mag = np.clip(abs(awa), 0.0, 0.5 * np.pi)   # weather-vane
>>  boom_mag = min(free_mag, sheet_limit)            # sheet is a stop

    wind_side = np.sign(apparent_wind_boat[1])
    if wind_side == 0.0:                             # hysteresis
        wind_side = np.sign(self.last_sail_angle_rad) or -1.0

    return float(-wind_side * boom_mag)`;

const LUFF_SRC = `aoa_deg_abs = float(np.degrees(np.abs(aoa_rad)))
luff_deg = float(self.p.get("luff_deg", 7.5))
luff_ramp_deg = max(float(self.p.get("luff_ramp_deg", 4.0)), 1e-6)
>>cl_scale = np.clip((aoa_deg_abs - luff_deg) / luff_ramp_deg, 0.0, 1.0)
>>cl *= cl_scale     # drag is deliberately left alone`;

/** Section 02 — boom angle as a function of apparent wind angle. */
function sheeting() {
  let sheetDeg = 25;

  const chart = lineChart({
    width: 560,
    height: 300,
    xDomain: [-180, 180],
    yDomain: [-95, 95],
    xLabel: "apparent wind angle (°)   ·   0 = dead ahead",
    yLabel: "boom angle (°)",
    xTicks: 8,
    xFmt: (v) => `${Math.round(v)}`,
    yFmt: (v) => `${Math.round(v)}`,
  });

  const sSheet = slider({
    label: "Sheet limit", hint: "control action", min: 0, max: 90, step: 1,
    value: sheetDeg, unit: "°",
    onInput: (v) => { sheetDeg = v; render(); },
  });

  function render() {
    const free = [];
    const resolved = [];
    for (let awa = -180; awa <= 180; awa += 1) {
      const sign = Math.sign(awa) || 1;
      const freeMag = clamp(Math.abs(awa), 0, 90);
      free.push([awa, sign * freeMag]);
      resolved.push([awa, sign * Math.min(freeMag, sheetDeg)]);
    }

    chart.setBands([
      { x: [-180, -sheetDeg], color: "var(--warn)", opacity: 0.05 },
      { x: [sheetDeg, 180], color: "var(--warn)", opacity: 0.05, label: "sheet loaded" },
      { x: [-sheetDeg, sheetDeg], color: "var(--text-faint)", opacity: 0.05, label: "sail free" },
    ]);

    chart.setSeries([
      { points: free, color: "var(--text-faint)", width: 1.3, dash: "4 4" },
      { points: resolved, color: "var(--c-sail)", width: 2.4 },
      {
        points: [[-180, sheetDeg], [180, sheetDeg]],
        color: "var(--warn)", width: 1, dash: "2 4", opacity: 0.8,
      },
      {
        points: [[-180, -sheetDeg], [180, -sheetDeg]],
        color: "var(--warn)", width: 1, dash: "2 4", opacity: 0.8,
      },
      {
        // What the old rigid-command behaviour looked like.
        points: [[-180, -sheetDeg], [0, -sheetDeg], [0, sheetDeg], [180, sheetDeg]],
        color: "var(--bad)", width: 1.2, dash: "1 4", opacity: 0.55,
      },
    ]);

    boardBits();
  }

  /** Just the live-boat dot; safe to call at helm frame rate. */
  function boardBits() {
    const board = state.board;
    if (!board || !board.forces || !board.forces.sail) {
      chart.setMarkers([]);
      return;
    }
    const awa = board.forces.sail.awaDeg;
    const sign = Math.sign(awa) || 1;
    chart.setMarkers([{
      type: "dot",
      x: awa,
      y: sign * Math.min(clamp(Math.abs(awa), 0, 90), sheetDeg),
      color: "var(--accent)",
      label: board.sailing ? "you" : "force board",
    }]);
  }

  render();
  mount("sheet-fig", chart.svg);
  mount("sheet-controls", sSheet.node);
  pycode(document.getElementById("sheet-code"), SHEET_SRC);
  return { render, boardBits };
}

/** Section 03 — the luff ramp and its effect on the real CL curve. */
function luffing() {
  const sail = state.cfgs.sail;
  let luffDeg = Number(sail.luff_deg ?? 7.5);
  let rampDeg = Number(sail.luff_ramp_deg ?? 4.0);
  let span = 40;

  const rampChart = lineChart({
    width: 520,
    height: 250,
    xDomain: [0, 30],
    yDomain: [-0.04, 1.08],
    xLabel: "|angle of attack| (°)",
    yLabel: "cl_scale",
    yFmt: (v) => v.toFixed(1),
  });

  const clChart = lineChart({
    width: 520,
    height: 250,
    xDomain: [-span, span],
    yDomain: [-1.4, 1.4],
    xLabel: "angle of attack α (°)",
    yLabel: "CL",
    yFmt: (v) => v.toFixed(1),
  });

  const sLuff = slider({
    label: "luff_deg", hint: "dead band", min: 0, max: 30, step: 0.5,
    value: luffDeg, unit: "°", fmt: (v) => v.toFixed(1),
    onInput: (v) => { luffDeg = v; render(); },
  });
  const sRamp = slider({
    label: "luff_ramp_deg", hint: "fade width", min: 0.5, max: 20, step: 0.5,
    value: rampDeg, unit: "°", fmt: (v) => v.toFixed(1),
    onInput: (v) => { rampDeg = v; render(); },
  });
  const spanChips = chips([["40", "±40°"], ["180", "full ±180°"]], "40", (id) => {
    span = Number(id);
    render(true);
  });

  function render(rebuild = false) {
    const cfg = { ...state.cfgs.sail, luff_deg: luffDeg, luff_ramp_deg: rampDeg };

    const ramp = [];
    for (let a = 0; a <= 30; a += 0.25) ramp.push([a, luffScale(a, cfg)]);
    rampChart.setBands([
      { x: [0, luffDeg], color: "var(--bad)", opacity: 0.08, label: "no lift" },
      { x: [luffDeg, luffDeg + rampDeg], color: "var(--warn)", opacity: 0.08, label: "fade" },
      { x: [luffDeg + rampDeg, 30], color: "var(--new)", opacity: 0.06, label: "full lift" },
    ]);
    rampChart.setSeries([
      { points: ramp, color: "var(--c-sail)", width: 2.4, fillTo: 0, fillOpacity: 0.09 },
      {
        // The hybrid sail's old hard cut-off, for contrast.
        points: [[0, 0], [5, 0], [5, 1], [30, 1]],
        color: "var(--bad)", width: 1.3, dash: "3 3", opacity: 0.7,
      },
    ]);
    rampChart.setMarkers([
      { type: "vline", x: luffDeg, color: "var(--warn)", label: `${luffDeg.toFixed(1)}°` },
    ]);

    if (rebuild) {
      clChart.setYDomain([-1.4, 1.4]);
    }
    const step = span / 240;
    const raw = [];
    const scaled = [];
    const hard = [];
    for (let a = -span; a <= span; a += step) {
      const { cl } = clCd(a * RAD, cfg);
      raw.push([a, cl]);
      scaled.push([a, cl * luffScale(a, cfg)]);
      hard.push([a, Math.abs(a) < 5 ? 0 : cl]);
    }
    clChart.setSeries([
      { points: raw, color: "var(--text-faint)", width: 1.3, dash: "4 3" },
      { points: hard, color: "var(--bad)", width: 1.1, opacity: 0.55 },
      { points: scaled, color: "var(--c-sail)", width: 2.3 },
    ]);
    clChart.setBands([
      { x: [-luffDeg, luffDeg], color: "var(--bad)", opacity: 0.07 },
    ]);

    boardBits();

    const caption = document.getElementById("luff-caption");
    if (caption) {
      const cfgName = state.config.replace(".yaml", "");
      const declared = state.cfgs.sail.luff_deg !== undefined;
      caption.innerHTML = declared
        ? `<b>${cfgName}</b> sets luff_deg = ${state.cfgs.sail.luff_deg}°, luff_ramp_deg = ${state.cfgs.sail.luff_ramp_deg}°.`
        : `<b>${cfgName}</b> declares neither key, so the model's defaults (7.5° / 4.0°) apply.`;
    }
  }

  /** Live-boat dots on both luff figures. */
  function boardBits() {
    const board = state.board;
    if (!board || !board.forces || !Number.isFinite(board.forces.sail.aoaDeg)) return;
    const label = board.sailing ? "you" : "force board";
    clChart.setMarkers([{
      type: "dot",
      x: clamp(board.forces.sail.aoaDeg, -span, span),
      y: board.forces.sail.cl,
      color: "var(--accent)",
      label,
    }]);
    rampChart.setMarkers([
      { type: "vline", x: luffDeg, color: "var(--warn)" },
      {
        type: "dot",
        x: Math.min(Math.abs(board.forces.sail.aoaDeg), 30),
        y: board.forces.sail.luffScale,
        color: "var(--accent)",
        label,
      },
    ]);
  }

  const reBadge = document.getElementById("luff-re");
  if (reBadge) reBadge.textContent = `Re ${Number(state.cfgs.sail.res[0]).toExponential(0)}`;

  render(true);
  mount("luff-ramp-fig", rampChart.svg);
  mount("luff-cl-fig", clChart.svg);
  mount("luff-controls", sLuff.node, sRamp.node, spanChips.node);
  pycode(document.getElementById("luff-code"), LUFF_SRC);

  return {
    render: (reset = false) => {
      if (reset) {
        luffDeg = Number(state.cfgs.sail.luff_deg ?? 7.5);
        rampDeg = Number(state.cfgs.sail.luff_ramp_deg ?? 4.0);
        sLuff.set(luffDeg);
        sRamp.set(rampDeg);
        if (reBadge) reBadge.textContent = `Re ${Number(state.cfgs.sail.res[0]).toExponential(0)}`;
      }
      render(reset);
    },
    boardBits,
  };
}

export function init() {
  const sheet = sheeting();
  const luff = luffing();

  subscribe((_s, changed) => {
    if (touched(changed, "config", "cfgs")) {
      sheet.render();
      luff.render(true);
    } else if (touched(changed, "board")) {
      // Helm frames only move the markers — the curves are unchanged.
      sheet.boardBits();
      luff.boardBits();
    }
  });
}
