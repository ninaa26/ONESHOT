/**
 * Section 00 — hero stats, force pipeline and transform tree.
 *
 * The pipeline diagram mirrors whatever the force board (section 01) last
 * solved, so the architecture picture always carries real numbers.
 */

import { el } from "../charts.js";
import { h, fmt, mount } from "../ui.js";
import { DATA } from "../../shared/physics/models.js";
import { polarAt, polarStats } from "../../shared/physics/analysis.js";
import { state, subscribe, touched } from "../state.js";

const COMPONENTS = [
  ["hull", "BasicHullModel", "var(--c-hull)"],
  ["keel", "BasicKeel", "var(--c-keel)"],
  ["sail", "BasicSail", "var(--c-sail)"],
  ["rudder", "BasicRudder", "var(--c-rudder)"],
];

/** Summary numbers for the selected boat, pulled from the baked polar sweep. */
function heroStats() {
  const polars = DATA.polars[state.config];
  const host = document.getElementById("hero-stats");
  if (!host) return;

  const cards = [
    ["models rewritten", "4", "sail · hull · rudder · servo"],
  ];

  if (polars && polars.length) {
    const stats = polarStats(polarAt(state.config, 5));
    cards.push(
      ["top speed", `${stats.top.toFixed(2)}<small>m/s</small>`, `at ${stats.windSpeed} m/s true, ${stats.topTwa}° TWA`],
      ["best upwind VMG", `${stats.upwind.twa}<small>°</small>`, `${stats.upwind.vmg.toFixed(2)} m/s made good`],
      ["best downwind VMG", `${stats.downwind.twa}<small>°</small>`, `${stats.downwind.vmg.toFixed(2)} m/s made good`],
    );
  } else {
    cards.push(["polar sweep", "—", "not generated for this boat"]);
  }

  cards.push(["foil model", "NACA0012", "NeuralFoil, live in-page"]);

  host.replaceChildren(...cards.map(([k, v, d]) => h("div", { class: "stat" }, [
    h("div", { class: "k", text: k }),
    h("div", { class: "v", html: v }),
    h("div", { class: "k", html: d, style: "margin-top:4px;text-transform:none;letter-spacing:0" }),
  ])));
}

/** Boxes-and-arrows view of one simulation step, with live force values. */
function pipeline() {
  const W = 560;
  const H = 300;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });

  const box = (x, y, w, hh, { title, sub, color = "var(--line-strong)", fill = "rgba(255,255,255,0.03)" }) => {
    const g = el("g");
    g.append(el("rect", { x, y, width: w, height: hh, rx: 8, fill, stroke: color, "stroke-width": 1 }));
    g.append(el("text", {
      x: x + w / 2, y: y + 16, "text-anchor": "middle", fill: color,
      "font-size": 10.5, "font-family": "var(--mono)",
    }, title));
    const val = el("text", {
      x: x + w / 2, y: y + hh - 9, "text-anchor": "middle", fill: "var(--text)",
      "font-size": 10, "font-family": "var(--mono)",
    }, sub);
    g.append(val);
    svg.append(g);
    return val;
  };

  const arrow = (x0, y0, x1, y1, dashed = false) => {
    svg.append(el("path", {
      d: `M${x0} ${y0}L${x1} ${y1}`,
      stroke: "rgba(140,180,230,0.32)", "stroke-width": 1,
      "stroke-dasharray": dashed ? "3 3" : null,
      "marker-end": "url(#pipe-arrow)",
    }));
  };

  const defs = el("defs");
  const marker = el("marker", {
    id: "pipe-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5,
    markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse",
  });
  marker.append(el("path", { d: "M0 0 L10 5 L0 10 z", fill: "rgba(140,180,230,0.45)" }));
  defs.append(marker);
  svg.append(defs);

  const stateVal = box(14, 124, 96, 52, { title: "State", sub: "u, v, r", color: "var(--accent)" });

  const compVals = {};
  COMPONENTS.forEach(([key, label, color], i) => {
    const y = 12 + i * 70;
    compVals[key] = box(160, y, 150, 52, { title: label, sub: "—", color });
    arrow(112, 150, 156, y + 26);
    arrow(312, y + 26, 356, 150);
  });

  const sumVal = box(358, 124, 96, 52, { title: "Σ F, Σ M", sub: "—", color: "var(--c-total)" });
  arrow(456, 150, 492, 150);
  const rk4Val = box(492, 124, 56, 52, { title: "RK4", sub: "dt", color: "var(--text-dim)" });

  // feedback loop
  svg.append(el("path", {
    d: `M520 176 L520 262 L62 262 L62 178`,
    fill: "none", stroke: "rgba(140,180,230,0.25)", "stroke-width": 1,
    "stroke-dasharray": "4 4", "marker-end": "url(#pipe-arrow)",
  }));
  svg.append(el("text", {
    x: 291, y: 276, "text-anchor": "middle", fill: "var(--text-faint)",
    "font-size": 9.5, "font-family": "var(--mono)",
  }, "integrate → next state"));

  function render() {
    const board = state.board;
    if (!board) return;
    stateVal.textContent = `${fmt(board.u, 2)}, ${fmt(board.v, 2)}, ${fmt(board.r, 2)}`;
    for (const [key] of COMPONENTS) {
      const f = board.forces[key];
      compVals[key].textContent = `${fmt(f.fx, 1)}, ${fmt(f.fy, 1)} N`;
    }
    const t = board.forces.total;
    sumVal.textContent = `${fmt(t.fx, 0)}, ${fmt(t.fy, 0)}, ${fmt(t.mz, 0)}`;
    rk4Val.textContent = `${fmt(Number(state.cfgs.simulation.dt) * 1000, 0)} ms`;
  }

  render();
  return { svg, render };
}

/** The tf_tree, with the two frames that move every step called out. */
function tfTree() {
  const W = 360;
  const H = 300;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });
  const nodes = [
    { id: "world", x: 150, y: 20, color: "var(--text-dim)", note: "inertial" },
    { id: "boat", x: 150, y: 80, color: "var(--accent)", note: "ψ" },
    { id: "fluid", x: 34, y: 160, color: "var(--c-hull)", note: "−V̂" },
    { id: "keel", x: 122, y: 160, color: "var(--c-keel)", note: "fixed" },
    { id: "sail", x: 210, y: 160, color: "var(--c-sail)", note: "δs" },
    { id: "rudder", x: 298, y: 160, color: "var(--c-rudder)", note: "δr" },
  ];
  const links = [["world", "boat"], ["boat", "fluid"], ["boat", "keel"], ["boat", "sail"], ["boat", "rudder"]];
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));

  for (const [a, b] of links) {
    const na = byId[a];
    const nb = byId[b];
    svg.append(el("path", {
      d: `M${na.x} ${na.y + 13}C${na.x} ${na.y + 40} ${nb.x} ${nb.y - 40} ${nb.x} ${nb.y - 13}`,
      stroke: "rgba(140,180,230,0.25)", "stroke-width": 1, fill: "none",
    }));
  }

  const valueRefs = {};
  for (const n of nodes) {
    svg.append(el("circle", {
      cx: n.x, cy: n.y, r: 5, fill: "#05070c", stroke: n.color, "stroke-width": 1.6,
    }));
    svg.append(el("text", {
      x: n.x, y: n.y - 12, "text-anchor": "middle", fill: n.color,
      "font-size": 10, "font-family": "var(--mono)",
    }, n.id));
    valueRefs[n.id] = el("text", {
      x: n.x, y: n.y + 22, "text-anchor": "middle", fill: "var(--text-faint)",
      "font-size": 9.5, "font-family": "var(--mono)",
    }, n.note);
    svg.append(valueRefs[n.id]);
  }

  svg.append(el("text", {
    x: 8, y: 220, fill: "var(--text-faint)", "font-size": 9.5, "font-family": "var(--mono)",
  }, "rebuilt every step:"));
  svg.append(el("text", {
    x: 8, y: 236, fill: "var(--text-dim)", "font-size": 9.5, "font-family": "var(--mono)",
  }, "boat ← state · fluid ← −V̂ · sail ← sheet"));
  svg.append(el("text", {
    x: 8, y: 252, fill: "var(--text-dim)", "font-size": 9.5, "font-family": "var(--mono)",
  }, "rudder ← servo"));
  svg.append(el("text", {
    x: 8, y: 276, fill: "var(--text-faint)", "font-size": 9.5, "font-family": "var(--mono)",
  }, "vector_to_frame() rotates only — no translation"));

  function render() {
    const board = state.board;
    if (!board) return;
    valueRefs.boat.textContent = `${fmt(board.headingDeg, 0)}°`;
    valueRefs.sail.textContent = `${fmt(board.sailDeg, 1)}°`;
    valueRefs.rudder.textContent = `${fmt(board.rudderDeg, 1)}°`;
    valueRefs.fluid.textContent = `${fmt(-board.leewayDeg, 1)}°`;
  }

  render();
  return { svg, render };
}

export function init() {
  const pipe = pipeline();
  const tree = tfTree();
  mount("pipeline-fig", pipe.svg);
  mount("tftree-fig", tree.svg);
  heroStats();

  const badge = document.getElementById("pipe-config");
  const setBadge = () => { if (badge) badge.textContent = state.config.replace(".yaml", ""); };
  setBadge();

  subscribe((_s, changed) => {
    if (touched(changed, "board")) {
      pipe.render();
      tree.render();
    }
    if (touched(changed, "config", "cfgs")) {
      heroStats();
      setBadge();
    }
  });
}
