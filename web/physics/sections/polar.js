/**
 * Section 10 — VPP polars.
 *
 * Unlike every other figure on the page, these points are not computed in the
 * browser: each one is a 20 s, heading-held run of the real simulator, baked by
 * `scripts/gen_physics_ui_data.py`.
 */

import { polarChart, lineChart } from "../charts.js";
import { h, spec, mount, chips, table } from "../ui.js";
import { DATA } from "../models.js";
import { polarsFor, polarAt, polarStats } from "../analysis.js";
import { state, subscribe, touched } from "../state.js";

const WIND_COLORS = ["#3c7fb1", "var(--c-sail)", "#a8e8ff"];

export function init() {
  let selected = 5;

  const chart = polarChart({ size: 460, rMax: 2, rings: 4 });
  const vmgChart = lineChart({
    width: 460,
    height: 250,
    xDomain: [0, 180],
    yDomain: [-2, 2.4],
    xLabel: "true wind angle (°)",
    yLabel: "m/s",
    xTicks: 6,
    xFmt: (v) => String(Math.round(v)),
    yFmt: (v) => v.toFixed(1),
  });

  const windChips = h("div");

  function render() {
    const list = polarsFor(state.config);
    const badge = document.getElementById("polar-config-badge");
    if (badge) badge.textContent = state.config.replace(".yaml", "");

    if (!list.length) {
      chart.clear();
      mount("polar-targets", h("div", {
        class: "note",
        html: `<b>No sweep for ${state.config}.</b> Add it to the <code>plan</code> list in `
          + "<code>scripts/gen_physics_ui_data.py</code> and regenerate to get a polar here.",
      }));
      vmgChart.setSeries([]);
      table(document.getElementById("polar-table"), ["TWA"], []);
      mount("polar-controls");
      return;
    }

    if (!list.some((p) => p.wind_speed === selected)) selected = list[0].wind_speed;

    const main = polarAt(state.config, selected);
    const stats = polarStats(main);
    const globalMax = Math.max(...list.flatMap((p) => p.speed));

    chart.setMax(Math.ceil(globalMax * 1.15 * 2) / 2);
    chart.clear();
    chart.addWedge(stats.noGoDeg, {
      color: "var(--bad)",
      label: `no-go ±${stats.noGoDeg.toFixed(0)}°`,
    });

    list.forEach((p, i) => {
      const isMain = p.wind_speed === selected;
      chart.addCurve(p.twa_deg, p.speed, {
        color: isMain ? "var(--c-sail)" : WIND_COLORS[i % WIND_COLORS.length],
        width: isMain ? 2.4 : 1.2,
        opacity: isMain ? 1 : 0.45,
        dash: isMain ? null : "4 3",
        fill: isMain,
      });
    });

    chart.addRay(stats.upwind.twa, stats.upwind.speed, {
      color: "var(--new)",
      label: `VMG ${stats.upwind.twa}°`,
    });
    chart.addRay(stats.downwind.twa, stats.downwind.speed, {
      color: "var(--warn)",
      label: `VMG ${stats.downwind.twa}°`,
    });
    chart.addDot(stats.topTwa, stats.top, { color: "var(--c-total)", label: `${stats.top.toFixed(2)} m/s` });
    chart.addWindGlyph();

    // --- targets ---
    mount(
      "polar-targets",
      spec("upwind VMG", `${stats.upwind.twa}<small>°</small>`, `${stats.upwind.vmg.toFixed(2)} m/s made good`, "up"),
      spec("downwind VMG", `${stats.downwind.twa}<small>°</small>`, `${stats.downwind.vmg.toFixed(2)} m/s made good`, "up"),
      spec("top speed", `${stats.top.toFixed(2)}`, `m/s at ${stats.topTwa}° TWA`),
      spec("no-go", `±${stats.noGoDeg.toFixed(0)}<small>°</small>`, `below ${stats.threshold.toFixed(2)} m/s`, "down"),
      spec("best sheet", `${stats.topSheet}<small>°</small>`, "at top speed"),
      spec("max leeway", `${stats.leewayMax.toFixed(1)}<small>°</small>`, "across the sweep"),
    );

    // --- VMG / sheet chart ---
    const sheetScale = 2.2 / 90;
    vmgChart.setYDomain([Math.min(-0.2, Math.min(...main.vmg) * 1.15), Math.max(2.4, stats.top * 1.15)]);
    vmgChart.setSeries([
      { points: main.twa_deg.map((t, i) => [t, main.speed[i]]), color: "var(--c-sail)", width: 2.2 },
      { points: main.twa_deg.map((t, i) => [t, main.vmg[i]]), color: "var(--new)", width: 1.8, dash: "4 3" },
      {
        points: main.twa_deg.map((t, i) => [t, main.best_sheet_deg[i] * sheetScale]),
        color: "var(--warn)", width: 1.3, opacity: 0.8,
      },
    ]);
    vmgChart.setBands([{ x: [0, stats.noGoDeg], color: "var(--bad)", opacity: 0.09, label: "no-go" }]);
    vmgChart.setMarkers([
      { type: "vline", x: stats.upwind.twa, color: "var(--new)", label: "best VMG" },
      { type: "vline", x: stats.downwind.twa, color: "var(--warn)" },
    ]);

    // --- table ---
    table(
      document.getElementById("polar-table"),
      ["TWA °", "speed m/s", "VMG m/s", "sheet °", "leeway °", "sail side force N"],
      main.twa_deg.map((t, i) => [
        { html: `${t}`, cls: "" },
        { html: main.speed[i].toFixed(3), cls: i === main.speed.indexOf(stats.top) ? "" : "" },
        { html: main.vmg[i].toFixed(3) },
        { html: String(main.best_sheet_deg[i]), cls: "dim" },
        { html: main.leeway_deg[i].toFixed(2), cls: "dim" },
        { html: main.side_force[i].toFixed(1), cls: "dim" },
      ]),
    );

    // --- wind-speed switch ---
    const picker = chips(
      list.map((p) => [String(p.wind_speed), `${p.wind_speed} m/s`]),
      String(selected),
      (id) => { selected = Number(id); render(); },
    );
    windChips.replaceChildren(picker.node);
    mount(
      "polar-controls",
      windChips,
      h("div", {
        class: "fig-cap",
        html: `${main.twa_deg.length} TWA × 17 sheet angles = `
          + `${main.twa_deg.length * 17} simulator runs · dt ${DATA.meta.sweep.dt} s · `
          + "12 s warm-up, 8 s average",
      }),
    );
  }

  render();
  mount("polar-fig", chart.svg);
  mount("polar-vmg-fig", vmgChart.svg);

  subscribe((_s, changed) => {
    if (touched(changed, "config")) render();
  });
}
