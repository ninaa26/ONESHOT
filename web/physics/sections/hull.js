/**
 * Section 04 — hull resistance derived from hull geometry.
 *
 * L / B / T drive the three drag coefficients directly, so the sliders here
 * show what changing the boat's dimensions does without any re-tuning.
 */

import { lineChart } from "../charts.js";
import { slider, readout, uval, fmt, mount, pycode } from "../ui.js";
import { hullCoeffs, hullForce } from "../../shared/physics/models.js";
import { state, subscribe, touched } from "../state.js";

const HULL_SRC = `s = 1.7 * l * (b + t)      # wetted surface
aside = l * t              # lateral plane

>>k_u = 0.5 * rho * s * 0.004        # skin friction
>>k_v = 0.5 * rho * aside            # bluff-body sway
>>k_r = (1.0 / 8.0) * rho * t * (l**4)   # yaw damping

fx = -k_u * u * abs(u)
fy = -k_v * v * abs(v)
mz = -k_r * r * abs(r)`;

export function init() {
  let dims = null; // { L, B, T } working copy

  const chart = lineChart({
    width: 540,
    height: 290,
    xDomain: [0, 2],
    yDomain: [-90, 0],
    xLabel: "u, v (m/s)   ·   r (rad/s)",
    yLabel: "resisting force (N) / moment (N·m)",
    yFmt: (v) => String(Math.round(v)),
  });

  const ro = readout([
    { sep: "geometry" },
    { key: "S", label: "wetted surface  1.7·L·(B+T)" },
    { key: "aside", label: "lateral plane  L·T" },
    { sep: "coefficients" },
    { key: "ku", label: "k_u  surge", hl: true },
    { key: "kv", label: "k_v  sway", hl: true },
    { key: "kr", label: "k_r  yaw", hl: true },
    { sep: "at 1 m/s and 1 rad/s" },
    { key: "fx", label: "Fx" },
    { key: "fy", label: "Fy" },
    { key: "mz", label: "Mz" },
    { key: "ratio", label: "sway : surge drag" },
  ]);

  const sL = slider({
    label: "L", hint: "waterline length", min: 0.4, max: 4, step: 0.05, value: 1.5, unit: " m",
    fmt: (v) => v.toFixed(2), onInput: (v) => { dims.L = v; render(); },
  });
  const sB = slider({
    label: "B", hint: "beam", min: 0.1, max: 1.6, step: 0.02, value: 0.6, unit: " m",
    fmt: (v) => v.toFixed(2), onInput: (v) => { dims.B = v; render(); },
  });
  const sT = slider({
    label: "T", hint: "draft", min: 0.02, max: 0.6, step: 0.005, value: 0.05, unit: " m",
    fmt: (v) => v.toFixed(3), onInput: (v) => { dims.T = v; render(); },
  });

  function syncFromConfig() {
    const hull = state.cfgs.hull;
    dims = { L: Number(hull.L), B: Number(hull.B), T: Number(hull.T) };
    sL.set(dims.L);
    sB.set(dims.B);
    sT.set(dims.T);
  }

  /** Just the "where the boat is right now" dots — cheap enough for the helm loop. */
  function markers() {
    const k = hullCoeffs({ ...state.cfgs.hull, ...dims });
    const board = state.board;
    chart.setMarkers(board ? [
      { type: "dot", x: Math.abs(board.u), y: -k.kU * board.u * board.u, color: "var(--c-hull)" },
      { type: "dot", x: Math.abs(board.v), y: -k.kV * board.v * board.v, color: "var(--c-keel)" },
    ] : []);
  }

  function render() {
    const cfg = { ...state.cfgs.hull, ...dims };
    const k = hullCoeffs(cfg);

    const surge = [];
    const sway = [];
    const yaw = [];
    for (let x = 0; x <= 2; x += 0.02) {
      surge.push([x, -k.kU * x * x]);
      sway.push([x, -k.kV * x * x]);
      yaw.push([x, -k.kR * x * x]);
    }

    const worst = Math.max(k.kU, k.kV, k.kR) * 4;
    chart.setYDomain([-Math.max(worst, 10), 0]);
    chart.setSeries([
      { points: surge, color: "var(--c-hull)", width: 2.2 },
      { points: sway, color: "var(--c-keel)", width: 2.2 },
      { points: yaw, color: "var(--c-rudder)", width: 2.2, dash: "5 3" },
    ]);

    markers();

    const f = hullForce({ u: 1, v: 1, r: 1 }, cfg);
    ro.set("S", uval(k.S, 3, " m²"));
    ro.set("aside", uval(k.aSide, 4, " m²"));
    ro.set("ku", uval(k.kU, 2, " N·s²/m²"));
    ro.set("kv", uval(k.kV, 1, " N·s²/m²"));
    ro.set("kr", uval(k.kR, 1, " N·m·s²"));
    ro.set("fx", uval(f.fx, 2, " N"));
    ro.set("fy", uval(f.fy, 1, " N"));
    ro.set("mz", uval(f.mz, 1, " N·m"));
    ro.set("ratio", `${fmt(k.kV / k.kU, 1)}×`);
  }

  syncFromConfig();
  render();
  mount("hull-fig", chart.svg);
  mount("hull-controls", sL.node, sB.node, sT.node);
  mount("hull-readout", ro.node);
  pycode(document.getElementById("hull-code"), HULL_SRC);

  subscribe((_s, changed) => {
    if (touched(changed, "config", "cfgs")) {
      syncFromConfig();
      render();
    } else if (touched(changed, "board")) {
      markers();
    }
  });
}
