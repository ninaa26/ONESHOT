/**
 * Section 09 — Flingo Floty.
 *
 * The newest hull in `configs/`. Everything here is derived from the YAML and
 * from the models, so the comparison stays honest if the config changes.
 */

import { el } from "../charts.js";
import { h, spec, mount, table } from "../ui.js";
import { DATA, hullCoeffs, boatConfig } from "../models.js";
import { polarAt, polarStats } from "../analysis.js";
import { update } from "../state.js";

const SUBJECT = "flingo_floty.yaml";
const BASELINE = "basic_sailbot.yaml";

const SECTIONS = ["boat", "hull", "sail", "keel", "rudder"];
const SKIP = new Set(["model_type", "alpha_min", "alpha_max"]);

/** Plan-view outline for one boat, in metres, bow at +x. */
function outline(cfg) {
  const L = Number(cfg.hull.L);
  const B = Number(cfg.hull.B);
  const bow = 0.6 * L;
  const stern = -0.4 * L;
  const half = B / 2;
  return [
    [bow, 0], [0.32 * L, half * 0.78], [stern + 0.12 * L, half], [stern, half * 0.82],
    [stern, -half * 0.82], [stern + 0.12 * L, -half], [0.32 * L, -half * 0.78],
  ];
}

/** Overlaid, to-scale plan views of the two boats. */
function planView() {
  const W = 560;
  const H = 320;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });

  const subject = boatConfig(SUBJECT);
  const baseline = boatConfig(BASELINE);
  const maxL = Math.max(Number(subject.hull.L), Number(baseline.hull.L));
  const ppm = (W * 0.62) / maxL;
  const cx = W * 0.44;

  const draw = (cfg, y, color, opacity, name) => {
    const g = el("g");
    const P = (bx, by) => [cx + bx * ppm, y - by * ppm];

    g.append(el("path", {
      d: `${outline(cfg).map((p, i) => {
        const [px, py] = P(p[0], p[1]);
        return `${i ? "L" : "M"}${px.toFixed(1)} ${py.toFixed(1)}`;
      }).join("")}Z`,
      fill: color, "fill-opacity": 0.07, stroke: color, "stroke-width": 1.5,
      "stroke-linejoin": "round", opacity,
    }));

    // Keel / rudder blades, chord scaled from planform area.
    const blade = (x0, x1, color, width, alpha) => {
      const [ax, ay] = P(x0, 0);
      const [bx, by] = P(x1, 0);
      g.append(el("line", {
        x1: ax, y1: ay, x2: bx, y2: by,
        stroke: color, "stroke-width": width, "stroke-linecap": "round", opacity: alpha,
      }));
    };

    const keelChord = Math.sqrt(Number(cfg.keel.area)) * 0.75;
    const kx = Number(cfg.keel.x_pos ?? 0);
    blade(kx + keelChord / 2, kx - keelChord / 2, "var(--c-keel)", 5, 0.6 * opacity);

    const rudChord = Math.sqrt(Number(cfg.rudder.area)) * 0.9;
    const rx = Number(cfg.rudder.x_pos ?? 0);
    blade(rx, rx - rudChord, "var(--c-rudder)", 4, opacity);

    // Mast + boom, boom length from sail area.
    const boom = Math.sqrt(Number(cfg.sail.area)) * 0.95;
    const sx = Number(cfg.sail.x_pos ?? 0);
    const [mx, my] = P(sx, 0);
    g.append(el("line", {
      x1: mx, y1: my, x2: P(sx - boom, 0)[0], y2: my,
      stroke: "var(--c-sail)", "stroke-width": 3, "stroke-linecap": "round", opacity,
    }));
    g.append(el("circle", { cx: mx, cy: my, r: 3.5, fill: "var(--c-sail)", opacity }));
    g.append(el("circle", {
      cx: cx, cy: y, r: 3, fill: "none", stroke: "#fff", "stroke-width": 1, opacity: 0.6 * opacity,
    }));

    // Length dimension line.
    const L = Number(cfg.hull.L);
    const [ax] = P(0.6 * L, 0);
    const [bx] = P(-0.4 * L, 0);
    const dimY = y + Number(cfg.hull.B) / 2 * ppm + 20;
    g.append(el("path", {
      d: `M${bx} ${dimY}L${ax} ${dimY}M${bx} ${dimY - 4}L${bx} ${dimY + 4}M${ax} ${dimY - 4}L${ax} ${dimY + 4}`,
      stroke: "rgba(140,180,230,0.35)", "stroke-width": 1, fill: "none",
    }));
    g.append(el("text", {
      x: (ax + bx) / 2, y: dimY + 14, "text-anchor": "middle", fill: "var(--text-faint)",
      "font-size": 9.5, "font-family": "var(--mono)",
    }, `L = ${L} m`));

    g.append(el("text", {
      x: 8, y: y - 4, fill: color, "font-size": 10.5, "font-family": "var(--mono)",
    }, name));
    g.append(el("text", {
      x: 8, y: y + 10, fill: "var(--text-faint)", "font-size": 9.5, "font-family": "var(--mono)",
    }, `sail ${cfg.sail.area} m² · keel ${cfg.keel.area} m²`));

    svg.append(g);
  };

  draw(baseline, 104, "#8c9cb3", 0.75, "basic_sailbot");
  draw(subject, 232, "#5ee0ff", 1, "flingo_floty");

  svg.append(el("text", {
    x: W - 8, y: H - 6, "text-anchor": "end", fill: "var(--text-faint)",
    "font-size": 9, "font-family": "var(--mono)",
  }, "same scale · appendage chords scaled from planform area"));

  return svg;
}

/** Everything that differs between the two configs, as a table. */
function diffRows() {
  const a = DATA.configs[SUBJECT];
  const b = DATA.configs[BASELINE];
  const rows = [];

  for (const section of SECTIONS) {
    const keys = new Set([...Object.keys(a[section] || {}), ...Object.keys(b[section] || {})]);
    for (const key of [...keys].sort()) {
      if (SKIP.has(key)) continue;
      const av = a[section]?.[key];
      const bv = b[section]?.[key];
      const same = String(av) === String(bv);
      let delta = "";
      if (!same && Number.isFinite(Number(av)) && Number.isFinite(Number(bv)) && Number(bv) !== 0) {
        const pct = ((Number(av) - Number(bv)) / Math.abs(Number(bv))) * 100;
        delta = `<span class="diff-tag ${pct > 0 ? "add" : "del"}">${pct > 0 ? "+" : ""}${pct.toFixed(0)}%</span>`;
      } else if (av === undefined) {
        delta = '<span class="diff-tag del">absent</span>';
      } else if (bv === undefined) {
        delta = '<span class="diff-tag add">only here</span>';
      }
      rows.push([
        { html: `<span class="dim">${section}.</span>${key}`, cls: same ? "dim" : "" },
        { html: av === undefined ? "—" : String(av), cls: same ? "dim" : "" },
        { html: bv === undefined ? "—" : String(bv), cls: "dim" },
        { html: delta },
      ]);
    }
  }
  return rows;
}

export function init() {
  const subject = boatConfig(SUBJECT);
  const baseline = boatConfig(BASELINE);
  const kS = hullCoeffs(subject.hull);
  const kB = hullCoeffs(baseline.hull);

  const statsS = polarStats(polarAt(SUBJECT, 5));
  const statsB = polarStats(polarAt(BASELINE, 5));

  const pct = (x, y) => `${x > y ? "+" : ""}${(((x - y) / Math.abs(y)) * 100).toFixed(0)}%`;
  const izS = Number(subject.boat.inertia_z);
  const izB = Number(baseline.boat.inertia_z);
  const sailPerKg = Number(subject.sail.area) / Number(subject.boat.mass);

  const blurb = document.getElementById("flingo-blurb");
  if (blurb) {
    blurb.innerHTML = "The newest hull in <code>configs/</code>, added in "
      + "<code>9c6442c</code>. Short waterline, deep draft, a fraction of the sail and keel "
      + "area of <code>basic_sailbot</code>, and a tenth of its yaw inertia — a small, "
      + "twitchy boat that turns out to be the fastest thing in the repo.";
  }

  mount("flingo-fig", planView());

  mount(
    "flingo-specs",
    spec("mass", `${subject.boat.mass}<small>kg</small>`, "same as every config"),
    spec("yaw inertia", `${izS}<small>kg·m²</small>`, `${pct(izS, izB)} vs basic_sailbot`, "down"),
    spec("waterline L", `${subject.hull.L}<small>m</small>`, `${pct(Number(subject.hull.L), Number(baseline.hull.L))}`, "down"),
    spec("beam B", `${subject.hull.B}<small>m</small>`, `${pct(Number(subject.hull.B), Number(baseline.hull.B))}`, "down"),
    spec("draft T", `${subject.hull.T}<small>m</small>`, `${pct(Number(subject.hull.T), Number(baseline.hull.T))}`, "up"),
    spec("sail area", `${subject.sail.area}<small>m²</small>`, `${sailPerKg.toFixed(3)} m²/kg`, "down"),
    spec("keel area", `${subject.keel.area}<small>m²</small>`, `NACA0010 section`, "down"),
    spec("rudder area", `${subject.rudder.area}<small>m²</small>`, `arm ${subject.rudder.x_pos} m`, "down"),
    statsS ? spec("top speed", `${statsS.top.toFixed(2)}<small>m/s</small>`, `${pct(statsS.top, statsB.top)} vs basic_sailbot`, "up") : null,
    statsS ? spec("upwind VMG", `${statsS.upwind.twa}<small>°</small>`, `${statsS.upwind.vmg.toFixed(2)} m/s made good`, "up") : null,
  );

  mount("flingo-notes", h("div", {
    class: "note info",
    html: "<b>It inherits the sail defaults.</b> <code>flingo_floty.yaml</code> declares no "
      + "<code>luff_deg</code> or <code>luff_ramp_deg</code>, so <code>BasicSail</code> falls back "
      + "to 7.5° / 4.0° — a much narrower luff dead band than the 14° "
      + "<code>basic_sailbot</code> asks for. It starts making lift far closer to head-to-wind.",
  }));

  table(
    document.getElementById("flingo-table"),
    ["parameter", "flingo_floty", "basic_sailbot", "Δ"],
    diffRows(),
  );

  // --- derived implications, all computed from the config above ---
  const yawTau = izS / kS.kR;
  const yawTauB = izB / kB.kR;
  const rudderArmRatio = Math.abs(Number(subject.rudder.x_pos)) / Number(subject.hull.L);
  const sailOffset = Number(subject.sail.x_pos);

  mount("flingo-implications", h("ul", { class: "tight" }, [
    h("li", {
      html: `<strong>Spins on a dime.</strong> Yaw inertia is ${izS} kg·m² against `
        + `${kS.kR.toFixed(1)} N·m·s² of hull yaw damping — a characteristic time of `
        + `${yawTau.toFixed(3)} s, versus ${yawTauB.toFixed(3)} s for basic_sailbot. `
        + "A given rudder moment changes its heading roughly "
        + `${(izB / izS).toFixed(0)}× faster.`,
    }),
    h("li", {
      html: `<strong>Deep hull, stiff in sway.</strong> Draft ${subject.hull.T} m over a `
        + `${subject.hull.L} m waterline gives k_v = ${kS.kV.toFixed(1)} N·s²/m² against `
        + `${kB.kV.toFixed(1)} for basic_sailbot — ${pct(kS.kV, kB.kV)} more resistance to `
        + "side-slip from the hull alone, which is how it gets away with a keel "
        + `${(Number(subject.keel.area) / Number(baseline.keel.area) * 100).toFixed(0)}% the size.`,
    }),
    h("li", {
      html: `<strong>Low drag is why it's fast.</strong> Surge coefficient k_u = `
        + `${kS.kU.toFixed(2)} vs ${kB.kU.toFixed(2)} — ${pct(kS.kU, kB.kU)}. Less sail area, `
        + "but much less to drag through the water, and the polar bears that out: "
        + (statsS ? `${statsS.top.toFixed(2)} m/s top speed vs ${statsB.top.toFixed(2)}.` : "see the polar section."),
    }),
    h("li", {
      html: `<strong>Long rudder arm, tiny blade.</strong> The rudder sits `
        + `${Math.abs(Number(subject.rudder.x_pos))} m aft — ${(rudderArmRatio * 100).toFixed(0)}% `
        + `of the waterline — with only ${subject.rudder.area} m² of area. Plenty of leverage, `
        + "very little force, so it is sensitive to the <code>effectiveness</code> scale factor.",
    }),
    h("li", {
      html: `<strong>The mast is ${sailOffset} m forward of the CG.</strong> Every other config puts `
        + "it at 0.0. A forward sail force generates a bow-down yaw moment "
        + "(<code>mz = x·Fy</code>), so the boat carries a built-in turn-away-from-the-wind bias "
        + "the rudder has to trim out.",
    }),
    statsS ? h("li", {
      html: `<strong>It points absurdly high.</strong> The sweep has it making way at `
        + `${statsS.noGoDeg.toFixed(0)}° TWA. No real sailboat does that — the model has no heel, `
        + "no righting moment and no added resistance in waves, so the penalty for pinching is "
        + "far too small. Treat the very top of its polar as optimistic.",
    }) : null,
  ]));

  // Deep-linking: make the section's own boat selectable in one click.
  const head = document.querySelector("#flingo .sec-head");
  if (head) {
    head.append(h("div", { class: "btn-row", style: "margin-top:12px" }, [
      h("button", {
        class: "chip",
        text: "load flingo_floty into every figure ↑",
        onclick: () => {
          update({ config: SUBJECT });
          const select = document.getElementById("boat-select");
          if (select) select.value = SUBJECT;
          document.getElementById("board").scrollIntoView({ behavior: "smooth" });
        },
      }),
      h("button", {
        class: "chip",
        text: "back to basic_sailbot",
        onclick: () => {
          update({ config: BASELINE });
          const select = document.getElementById("boat-select");
          if (select) select.value = BASELINE;
        },
      }),
    ]));
  }
}
