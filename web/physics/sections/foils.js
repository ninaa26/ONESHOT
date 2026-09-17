/**
 * Sections 07 and 08 — the keel, and the NeuralFoil polars behind every foil.
 */

import { lineChart } from "../charts.js";
import { slider, readout, uval, fmt, mount, chips } from "../ui.js";
import { RAD, clCd, keelForce, foilCurve } from "../models.js";
import { state, subscribe, touched } from "../state.js";

/** Section 07 — keel side force against leeway angle. */
function keel() {
  let speed = 1.5;
  let area = null;
  let cursor = 3;

  const chart = lineChart({
    width: 540,
    height: 290,
    xDomain: [-20, 20],
    yDomain: [-2000, 2000],
    xLabel: "leeway angle β = atan2(v, u)  (°)",
    yLabel: "force (N)",
    xFmt: (v) => String(Math.round(v)),
    yFmt: (v) => String(Math.round(v)),
  });

  const ro = readout([
    { sep: "at the cursor" },
    { key: "beta", label: "leeway β" },
    { key: "cl", label: "CL" },
    { key: "cd", label: "CD" },
    { key: "ld", label: "L/D", hl: true },
    { sep: "force" },
    { key: "q", label: "dynamic pressure q" },
    { key: "lift", label: "lift" },
    { key: "drag", label: "drag" },
    { key: "fx", label: "Fx (boat frame)" },
    { key: "fy", label: "Fy (boat frame)", hl: true },
    { sep: "context" },
    { key: "sail", label: "sail side force now" },
    { key: "balance", label: "keel / sail balance" },
  ]);

  const sSpeed = slider({
    label: "Boat speed", min: 0.2, max: 4, step: 0.05, value: speed, unit: " m/s",
    fmt: (v) => v.toFixed(2), onInput: (v) => { speed = v; render(); },
  });
  const sArea = slider({
    label: "Keel area", min: 0.02, max: 1.2, step: 0.01, value: 0.75, unit: " m²",
    fmt: (v) => v.toFixed(2), onInput: (v) => { area = v; render(); },
  });
  const sCursor = slider({
    label: "Cursor β", min: -20, max: 20, step: 0.25, value: cursor, unit: "°",
    fmt: (v) => v.toFixed(2), onInput: (v) => { cursor = v; render(); },
  });

  function syncFromConfig() {
    area = Number(state.cfgs.keel.area);
    sArea.set(area);
  }

  function render() {
    const cfg = { ...state.cfgs.keel, area };
    const fy = [];
    const fx = [];
    const ld = [];
    let peak = 1;
    let maxLd = 1;

    for (let b = -20; b <= 20; b += 0.2) {
      const u = speed * Math.cos(b * RAD);
      const v = speed * Math.sin(b * RAD);
      const f = keelForce({ u, v, r: 0 }, cfg);
      fy.push([b, f.fy]);
      fx.push([b, f.fx]);
      const ratio = f.cd > 1e-9 ? f.cl / f.cd : 0;
      ld.push([b, ratio]);
      peak = Math.max(peak, Math.abs(f.fy), Math.abs(f.fx));
      maxLd = Math.max(maxLd, Math.abs(ratio));
    }

    chart.setYDomain([-peak * 1.1, peak * 1.1]);
    const ldScale = (peak * 0.85) / maxLd;
    chart.setSeries([
      { points: fy, color: "var(--c-keel)", width: 2.4 },
      { points: fx, color: "var(--c-hull)", width: 1.9 },
      { points: ld.map(([b, r]) => [b, r * ldScale]), color: "var(--warn)", width: 1.4, dash: "4 3", opacity: 0.85 },
    ]);

    const u = speed * Math.cos(cursor * RAD);
    const v = speed * Math.sin(cursor * RAD);
    const f = keelForce({ u, v, r: 0 }, cfg);
    chart.setMarkers([
      { type: "vline", x: cursor, color: "var(--accent)" },
      { type: "dot", x: cursor, y: f.fy, color: "var(--c-keel)" },
      { type: "text", x: -19.5, y: peak * 0.95, color: "var(--warn)", label: `L/D ×${ldScale.toFixed(0)} (scaled)`, size: 9.5 },
    ]);

    ro.set("beta", uval(cursor, 2, "°"));
    ro.set("cl", fmt(f.cl, 4));
    ro.set("cd", fmt(f.cd, 4));
    ro.set("ld", fmt(f.cd > 1e-9 ? f.cl / f.cd : 0, 1));
    ro.set("q", uval(f.q, 0, " Pa"));
    ro.set("lift", uval(f.lift, 1, " N"));
    ro.set("drag", uval(f.drag, 1, " N"));
    ro.set("fx", uval(f.fx, 1, " N"));
    ro.set("fy", uval(f.fy, 1, " N"));

    boardBits();
  }

  /** Only the rows that depend on the live boat; safe to call every frame. */
  function boardBits() {
    const board = state.board;
    if (!board || !board.forces) return;
    const sailFy = board.forces.sail.fy;
    ro.set("sail", uval(sailFy, 1, " N"));
    ro.set("balance", Math.abs(sailFy) > 1e-6
      ? `${fmt(-board.forces.keel.fy / sailFy, 2)}× opposing`
      : "—");
  }

  syncFromConfig();
  render();
  mount("keel-fig", chart.svg);
  mount("keel-controls", sSpeed.node, sArea.node, sCursor.node);
  mount("keel-readout", ro.node);

  return {
    render: (reset) => {
      if (reset) syncFromConfig();
      render();
    },
    boardBits,
  };
}

/** Section 08 — raw NACA0012 polars at both Reynolds numbers in play. */
function foils() {
  let metric = "cl";
  let span = 30;
  let cursor = 8;

  const chart = lineChart({
    width: 540,
    height: 300,
    xDomain: [-30, 30],
    yDomain: [-1.4, 1.4],
    xLabel: "angle of attack α (°)",
    yLabel: "CL",
    xFmt: (v) => String(Math.round(v)),
    yFmt: (v) => v.toFixed(2),
  });

  const ro = readout([
    { key: "alpha", label: "α" },
    { sep: "Re 1e5 — sail" },
    { key: "cl1", label: "CL" },
    { key: "cd1", label: "CD" },
    { key: "ld1", label: "L/D", hl: true },
    { sep: "Re 5e5 — keel & rudder" },
    { key: "cl5", label: "CL" },
    { key: "cd5", label: "CD" },
    { key: "ld5", label: "L/D", hl: true },
    { sep: "stall" },
    { key: "peak1", label: "CL peak @ Re 1e5" },
    { key: "peak5", label: "CL peak @ Re 5e5" },
  ]);

  const metricChips = chips(
    [["cl", "CL"], ["cd", "CD"], ["ld", "L/D"]],
    "cl",
    (id) => { metric = id; render(true); },
  );
  const spanChips = chips(
    [["30", "±30°"], ["90", "±90°"], ["180", "±180°"]],
    "30",
    (id) => { span = Number(id); render(true); },
  );
  const sCursor = slider({
    label: "Cursor α", min: -30, max: 30, step: 0.5, value: cursor, unit: "°",
    fmt: (v) => v.toFixed(1), onInput: (v) => { cursor = v; render(); },
  });

  const curves = {
    100000: foilCurve(100000),
    500000: foilCurve(500000),
  };

  function valueAt(curve, i) {
    if (metric === "cl") return curve.cl[i];
    if (metric === "cd") return curve.cd[i];
    return curve.cd[i] > 1e-9 ? curve.cl[i] / curve.cd[i] : 0;
  }

  function render(rebuild = false) {
    if (rebuild) {
      sCursor.input.min = String(-span);
      sCursor.input.max = String(span);
      if (Math.abs(cursor) > span) {
        cursor = 0;
        sCursor.set(0);
      }
    }

    const series = [];
    let lo = 0;
    let hi = 0;
    for (const [re, color] of [[100000, "var(--c-sail)"], [500000, "var(--c-keel)"]]) {
      const curve = curves[re];
      const pts = [];
      curve.alpha.forEach((a, i) => {
        if (Math.abs(a) > span) return;
        const val = valueAt(curve, i);
        pts.push([a, val]);
        lo = Math.min(lo, val);
        hi = Math.max(hi, val);
      });
      series.push({ points: pts, color, width: 2.1 });
    }

    const padY = (hi - lo) * 0.1 || 0.2;
    chart.setYDomain([lo - padY, hi + padY]);
    chart.setSeries(series);

    // Highlight the region the configs actually allow (alpha_min/alpha_max).
    const aMax = Number(state.cfgs.sail.alpha_max ?? 20);
    chart.setBands(aMax < span ? [
      { x: [aMax, span], color: "var(--bad)", opacity: 0.08, label: "clipped" },
      { x: [-span, -aMax], color: "var(--bad)", opacity: 0.08 },
    ] : []);

    const at = (re) => clCd(cursor * RAD, { res: [re], alpha_min: -179, alpha_max: 179 });
    const a1 = at(1e5);
    const a5 = at(5e5);
    chart.setMarkers([
      { type: "vline", x: cursor, color: "var(--accent)", label: `${cursor.toFixed(1)}°` },
      { type: "dot", x: cursor, y: metric === "cl" ? a1.cl : metric === "cd" ? a1.cd : a1.cl / Math.max(a1.cd, 1e-9), color: "var(--c-sail)" },
      { type: "dot", x: cursor, y: metric === "cl" ? a5.cl : metric === "cd" ? a5.cd : a5.cl / Math.max(a5.cd, 1e-9), color: "var(--c-keel)" },
    ]);

    ro.set("alpha", uval(cursor, 1, "°"));
    ro.set("cl1", fmt(a1.cl, 4));
    ro.set("cd1", fmt(a1.cd, 4));
    ro.set("ld1", fmt(a1.cl / Math.max(a1.cd, 1e-9), 1));
    ro.set("cl5", fmt(a5.cl, 4));
    ro.set("cd5", fmt(a5.cd, 4));
    ro.set("ld5", fmt(a5.cl / Math.max(a5.cd, 1e-9), 1));

    for (const [re, key] of [[100000, "peak1"], [500000, "peak5"]]) {
      const curve = curves[re];
      let best = { cl: 0, a: 0 };
      curve.alpha.forEach((a, i) => {
        if (a > 0 && a < 40 && curve.cl[i] > best.cl) best = { cl: curve.cl[i], a };
      });
      ro.set(key, `${fmt(best.cl, 3)} <span class="u">at ${best.a}°</span>`);
    }
  }

  const badges = document.getElementById("foil-badges");
  if (badges) {
    badges.innerHTML = '<span class="badge commit">Re 1e5 · sail</span> '
      + '<span class="badge commit">Re 5e5 · keel &amp; rudder</span>';
  }

  render(true);
  mount("foil-fig", chart.svg);
  mount("foil-controls", metricChips.node, spanChips.node, sCursor.node);
  mount("foil-readout", ro.node);
  return () => render(true);
}

export function init() {
  const { render: renderKeel, boardBits: updateKeelBoard } = keel();
  const renderFoils = foils();

  subscribe((_s, changed) => {
    if (touched(changed, "config", "cfgs")) {
      renderKeel(true);
      renderFoils();
    } else if (touched(changed, "board")) {
      updateKeelBoard();
    }
  });
}
