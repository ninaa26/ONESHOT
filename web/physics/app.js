/**
 * SailBench Physics Lab — entry point.
 *
 * Mirrors the structure of `web/main.js`: this module owns the page-level
 * wiring (rail controls, scroll-spy, section bootstrap) and each
 * `sections/*.js` module owns one part of the page. Sections communicate only
 * through `state.js`, so any of them can be dropped without touching the rest.
 */

import { slider, h } from "./ui.js";
import { DATA, CONFIG_NAMES } from "../shared/physics/models.js";
import { state, update } from "./state.js";

import * as overview from "./sections/overview.js";
import * as board from "./sections/board.js";
import * as sail from "./sections/sail.js";
import * as hull from "./sections/hull.js";
import * as steering from "./sections/steering.js";
import * as foils from "./sections/foils.js";
import * as flingo from "./sections/flingo.js";
import * as polar from "./sections/polar.js";
import * as validate from "./sections/validate.js";
import * as live from "./sections/live.js";

/** Section modules, in page order. Add one here and it is wired up. */
const SECTIONS = [
  ["board", board],
  ["overview", overview],
  ["sail", sail],
  ["hull", hull],
  ["steering", steering],
  ["foils", foils],
  ["flingo", flingo],
  ["polar", polar],
  ["validate", validate],
  ["live", live],
];

function railControls() {
  const select = document.getElementById("boat-select");
  if (select) {
    for (const name of CONFIG_NAMES) {
      select.append(h("option", { value: name, text: name.replace(".yaml", "") }));
    }
    select.value = state.config;
    select.addEventListener("change", (ev) => update({ config: ev.target.value }));
  }

  const windSpeed = slider({
    label: "True wind", min: 0, max: 15, step: 0.5, value: state.windSpeed, unit: " m/s",
    fmt: (v) => v.toFixed(1),
    onInput: (v) => update({ windSpeed: v }),
  });
  const windDir = slider({
    label: "Wind direction", hint: "blows toward", min: -180, max: 180, step: 5,
    value: state.windDirDeg, unit: "°",
    onInput: (v) => update({ windDirDeg: v }),
  });
  const host = document.getElementById("rail-wind");
  if (host) host.replaceChildren(windSpeed.node, windDir.node);
}

/** Highlight the nav entry for whichever section is in view. */
function scrollSpy() {
  const links = [...document.querySelectorAll("#toc a")];
  const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const visible = new Map();

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      visible.set(entry.target.id, entry.isIntersecting ? entry.intersectionRatio : 0);
    }
    let bestId = null;
    let bestRatio = 0;
    for (const [id, ratio] of visible) {
      if (ratio > bestRatio) {
        bestRatio = ratio;
        bestId = id;
      }
    }
    for (const [id, link] of byId) link.classList.toggle("active", id === bestId);
  }, { threshold: [0, 0.12, 0.35, 0.6], rootMargin: "-8% 0px -55% 0px" });

  for (const id of byId.keys()) {
    const node = document.getElementById(id);
    if (node) observer.observe(node);
  }
}

function meta() {
  const stamp = DATA.meta.generated_utc.replace("T", " ").replace("Z", " UTC");
  for (const id of ["meta-commit", "meta-commit-2"]) {
    const node = document.getElementById(id);
    if (node) node.textContent = DATA.meta.commit;
  }
  const gen = document.getElementById("meta-generated");
  if (gen) gen.textContent = stamp;
}

function boot() {
  meta();
  railControls();

  for (const [name, mod] of SECTIONS) {
    try {
      mod.init();
    } catch (err) {
      // One broken section must not take the whole page down.
      console.error(`[physics-lab] section "${name}" failed to initialise`, err);
    }
  }

  scrollSpy();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
