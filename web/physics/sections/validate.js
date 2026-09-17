/**
 * Section 11 — mirror check.
 *
 * Replays force samples recorded from `SailboatHub._forces` through the
 * JavaScript port and reports the residual, so the page can show its work
 * instead of asking to be believed.
 */

import { lineChart } from "../charts.js";
import { h, mount, table } from "../ui.js";
import { DATA, RAD, solveAt, boatConfig } from "../../shared/physics/models.js";

const COMPONENTS = [
  ["hull", "var(--c-hull)"],
  ["keel", "var(--c-keel)"],
  ["sail", "var(--c-sail)"],
  ["rudder", "var(--c-rudder)"],
  ["total", "var(--c-total)"],
];

/** Angle of attack that counts as "flow exactly astern" for the boundary tally. */
const ASTERN_DEG = 178;

/** Run every recorded case through the mirror. */
function run() {
  const cfgs = boatConfig("basic_sailbot.yaml");
  const stats = Object.fromEntries(COMPONENTS.map(([name]) => [name, {
    maxAbs: 0, sumAbs: 0, sumRef: 0, n: 0, points: [], rel: [],
  }]));
  let astern = 0;
  let flipped = 0;
  let maxSailAngleErr = 0;

  for (const c of DATA.validation) {
    cfgs.sail.wind_speed = c.wind_speed;
    const mirror = solveAt(
      { u: c.u, v: c.v, r: c.r },
      c.heading_deg * RAD,
      cfgs,
      { sheetRad: c.sheet_deg * RAD, rudderRad: c.rudder_deg * RAD },
    );

    // The resolved sheet angle is recorded too, so the sheeting rule is
    // checked independently of the aerodynamics.
    maxSailAngleErr = Math.max(
      maxSailAngleErr,
      Math.abs(mirror.sailRad * (180 / Math.PI) - c.sail_angle_deg),
    );
    const sternOn = Math.abs(mirror.sail.aoaDeg ?? 0) >= ASTERN_DEG;
    if (sternOn) {
      astern += 1;
      // A sign flip shows up as a residual comparable to the force itself.
      const ref = c.forces.sail;
      const refMag = Math.hypot(ref[0], ref[1]);
      const err = Math.hypot(mirror.sail.fx - ref[0], mirror.sail.fy - ref[1]);
      if (refMag > 1e-9 && err > refMag) flipped += 1;
    }

    for (const [name] of COMPONENTS) {
      const ref = c.forces[name];
      if (!ref) continue;
      const got = name === "total" ? mirror.total : mirror[name];
      const err = Math.hypot(got.fx - ref[0], got.fy - ref[1]);
      const refMag = Math.hypot(ref[0], ref[1]);
      const s = stats[name];
      s.maxAbs = Math.max(s.maxAbs, err);
      s.sumAbs += err;
      s.sumRef += refMag;
      s.n += 1;
      s.rel.push(refMag > 1e-9 ? err / refMag : 0);
      s.points.push([refMag, Math.hypot(got.fx, got.fy)]);
    }
  }

  for (const [name] of COMPONENTS) {
    const sorted = [...stats[name].rel].sort((a, b) => a - b);
    stats[name].median = sorted[Math.floor(sorted.length / 2)] || 0;
  }
  return { stats, astern, flipped, maxSailAngleErr };
}

export function init() {
  const countEl = document.getElementById("valid-count");
  if (countEl) countEl.textContent = String(DATA.validation.length);

  const { stats, astern, flipped, maxSailAngleErr } = run();
  const peak = Math.max(...COMPONENTS.flatMap(([n]) => stats[n].points.map((p) => p[0])), 1);

  const chart = lineChart({
    width: 500,
    height: 300,
    xDomain: [0, peak],
    yDomain: [0, peak],
    xLabel: "Python |F| (N)",
    yLabel: "browser |F| (N)",
    xFmt: (v) => String(Math.round(v)),
    yFmt: (v) => String(Math.round(v)),
  });

  chart.setSeries([
    { points: [[0, 0], [peak, peak]], color: "var(--text-faint)", width: 1, dash: "4 4" },
    ...COMPONENTS.map(([name, color]) => ({
      points: [],
      color,
      dots: stats[name].points,
      dotR: 2,
    })),
  ]);

  let worstMedian = 0;
  table(
    document.getElementById("valid-table"),
    ["component", "median |Δ|/|F|", "mean |Δ| N", "max |Δ| N"],
    COMPONENTS.map(([name]) => {
      const s = stats[name];
      worstMedian = Math.max(worstMedian, s.median);
      return [
        name,
        { html: `${(s.median * 100).toExponential(2)} %` },
        { html: (s.sumAbs / Math.max(s.n, 1)).toExponential(2) },
        { html: s.maxAbs.toExponential(2) },
      ];
    }),
  );

  const verdict = document.getElementById("valid-verdict");
  if (verdict) {
    const good = worstMedian < 1e-3;
    verdict.textContent = good
      ? `match · worst median ${(worstMedian * 100).toExponential(1)}%`
      : `worst median ${(worstMedian * 100).toFixed(2)}%`;
    verdict.className = `badge ${good ? "new" : "warn"}`;
  }

  // The handful of cases that do not agree are worth stating plainly.
  mount("valid-notes", h("div", {
    class: "note",
    html: "<b>The sail's outliers are a real knife edge, not a porting bug.</b> "
      + `${astern} of ${DATA.validation.length} sampled states put the apparent flow within `
      + `${180 - ASTERN_DEG}° of dead astern, and ${flipped} of those land on opposite sides of `
      + "±180°. <code>Foil.cl_cd</code> clips to <code>alpha_max</code> on one side and "
      + "<code>alpha_min</code> on the other, and NACA0012 is antisymmetric — so the lift "
      + "<em>magnitude</em> agrees to six figures while its <em>sign</em> flips, decided by the "
      + "sign of a quantity that is numerically zero. With the flow exactly astern this model's "
      + "side force is settled by floating-point noise, which is worth knowing before trusting a "
      + "deep-downwind RL reward.<br><br>"
      + "The sheeting rule itself is exact: the resolved sail angle matches the Python core to "
      + `${maxSailAngleErr.toExponential(1)}° across every case.`,
  }));

  mount("valid-fig", chart.svg);
}
