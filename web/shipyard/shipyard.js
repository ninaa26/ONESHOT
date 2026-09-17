// Shipyard: the pick-your-boat screen shown before sailing.
//
// The server sends a catalog on connect (boats, part models, trained policies);
// this renders it as a Mario-Kart-style selection grid driven by the arrow keys,
// and hands the chosen build back to main.js to send as a `setup` message.

const STORAGE_KEY = "sailbench.shipyard.selection";

/** Row order on screen. Row 0 is the boat cards, the rest are parts, then helm. */
const PART_ROWS = ["sail", "keel", "rudder", "hull"];
const ROWS = ["boat", ...PART_ROWS, "helm"];

const PART_LABELS = {
  sail: "Sail",
  keel: "Keel",
  rudder: "Rudder",
  hull: "Hull",
  helm: "Helm",
};

/**
 * @param {{ onLaunch: (selection: object) => void }} opts
 */
export function createShipyard({ onLaunch }) {
  const root = document.getElementById("shipyard");
  const statusEl = document.getElementById("shipyard-status");
  const errorEl = document.getElementById("shipyard-error");
  const gridEl = document.getElementById("shipyard-grid");
  const boatsEl = document.getElementById("shipyard-boats");
  const statsEl = document.getElementById("shipyard-stats");
  const partsEl = document.getElementById("shipyard-parts");
  const launchEl = document.getElementById("shipyard-launch");

  /** @type {object | null} */
  let catalog = null;
  let open = true;
  let row = 0;
  /** @type {{ boat: string, parts: Record<string, string>, helm: string }} */
  let selection = { boat: "", parts: {}, helm: "manual" };
  /** Largest value of each stat across boats, for the bar scale. */
  let statMax = {};

  // --- selection helpers -------------------------------------------------

  function boatById(id) {
    return catalog.boats.find((b) => b.id === id) || null;
  }

  function currentBoat() {
    return boatById(selection.boat) || catalog.boats[0];
  }

  /** Options for a row as [{id, name, blurb, disabled, reason}]. */
  function rowOptions(name) {
    if (name === "boat") {
      return catalog.boats.map((b) => ({ id: b.id, name: b.name, disabled: false }));
    }
    if (name === "helm") {
      return [
        { id: "manual", name: "Manual", blurb: "You steer: ←/→ rudder, ↑/↓ sheet", disabled: false },
        ...catalog.policies.map((p) => ({
          id: p.id,
          name: p.name,
          blurb: "Trained PPO policy sails to waypoints",
          disabled: false,
        })),
      ];
    }
    const boat = currentBoat();
    const availability = (boat.available && boat.available[name]) || {};
    return catalog.parts[name].map((o) => {
      const reason = availability[o.id];
      return { ...o, disabled: typeof reason === "string", reason };
    });
  }

  function rowValue(name) {
    if (name === "boat") return selection.boat;
    if (name === "helm") return selection.helm;
    return selection.parts[name];
  }

  function setRowValue(name, id) {
    if (name === "boat") {
      // A boat comes rigged the way its config says; swap parts after picking it.
      selection.boat = id;
      selection.parts = { ...currentBoat().defaults };
    } else if (name === "helm") {
      selection.helm = id;
    } else {
      selection.parts[name] = id;
    }
  }

  /** Keep each part on a model the current boat can build, else its default. */
  function reconcileParts() {
    const boat = currentBoat();
    for (const part of PART_ROWS) {
      const options = rowOptions(part);
      const chosen = options.find((o) => o.id === selection.parts[part]);
      if (!chosen || chosen.disabled) {
        selection.parts[part] = boat.defaults[part];
      }
    }
  }

  function stepRow(name, dir) {
    const options = rowOptions(name);
    if (options.length === 0) return;
    let idx = options.findIndex((o) => o.id === rowValue(name));
    if (idx < 0) idx = 0;
    // Skip greyed-out options; give up after one full lap.
    for (let n = 0; n < options.length; n += 1) {
      idx = (idx + dir + options.length) % options.length;
      if (!options[idx].disabled) {
        setRowValue(name, options[idx].id);
        return;
      }
    }
  }

  // --- persistence ------------------------------------------------------

  function loadStored() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  function store() {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(selection));
    } catch {
      // private mode or blocked storage: the pick just isn't remembered
    }
  }

  /**
   * Keep what is already picked across a reconnect; otherwise start from the
   * last launch when it still fits the catalog, else the CLI defaults.
   */
  function initialSelection() {
    const prior = selection.boat ? selection : loadStored();
    const boat = prior && boatById(prior.boat) ? prior.boat : catalog.default_boat;
    selection = {
      boat,
      parts: prior && prior.parts ? { ...prior.parts } : {},
      helm: "manual",
    };
    const helmIds = rowOptions("helm").map((o) => o.id);
    const wantHelm = prior && prior.helm ? prior.helm : catalog.default_helm;
    selection.helm = helmIds.includes(wantHelm) ? wantHelm : "manual";
    reconcileParts();
  }

  // --- rendering --------------------------------------------------------

  function boatSvg(boat) {
    // A silhouette scaled by the boat's own numbers: hull length by L, sail by area.
    const stat = (label, fallback) => {
      const s = boat.stats.find((x) => x.label === label);
      return s ? s.value : fallback;
    };
    const lFrac = Math.min(1, stat("Length", 1.5) / (statMax.Length || 1.5));
    const sailFrac = Math.min(1, stat("Sail area", 2) / (statMax["Sail area"] || 2));
    const hullW = 60 + 60 * lFrac;
    const sailH = 40 + 50 * sailFrac;
    const x0 = 80 - hullW / 2;
    const x1 = 80 + hullW / 2;
    return `
      <svg viewBox="0 0 160 130" class="boat-silhouette" aria-hidden="true">
        <line x1="80" y1="${112 - sailH}" x2="80" y2="112" class="mast" />
        <polygon points="80,${112 - sailH} ${80 + hullW * 0.42},108 80,108" class="sail" />
        <polygon points="${x0},108 ${x1},108 ${x1 - hullW * 0.12},122 ${x0 + hullW * 0.1},122" class="hull" />
      </svg>`;
  }

  function renderBoats() {
    const options = rowOptions("boat");
    boatsEl.innerHTML = "";
    for (const o of options) {
      const boat = boatById(o.id);
      const card = document.createElement("button");
      card.type = "button";
      card.className = "boat-card";
      card.dataset.id = o.id;
      if (o.id === selection.boat) card.classList.add("selected");
      card.innerHTML = `${boatSvg(boat)}<div class="boat-name">${escapeHtml(boat.name)}</div><div class="boat-file">${escapeHtml(boat.id)}</div>`;
      card.addEventListener("click", () => {
        row = 0;
        setRowValue("boat", o.id);
        render();
      });
      boatsEl.appendChild(card);
    }
    boatsEl.parentElement.classList.toggle("active", row === 0);
  }

  function renderStats() {
    const boat = currentBoat();
    statsEl.innerHTML = "";
    for (const s of boat.stats) {
      const max = statMax[s.label] || s.value || 1;
      const pct = Math.max(4, Math.round((100 * s.value) / max));
      const el = document.createElement("div");
      el.className = "stat";
      el.innerHTML = `
        <span class="stat-label">${escapeHtml(s.label)}</span>
        <span class="stat-bar"><span class="stat-fill" style="width:${pct}%"></span></span>
        <span class="stat-value">${formatStat(s.value)} ${escapeHtml(s.unit)}</span>`;
      statsEl.appendChild(el);
    }
  }

  function renderParts() {
    partsEl.innerHTML = "";
    for (const name of [...PART_ROWS, "helm"]) {
      const rowIdx = ROWS.indexOf(name);
      const options = rowOptions(name);
      const value = rowValue(name);
      const line = document.createElement("div");
      line.className = "part-row";
      if (rowIdx === row) line.classList.add("active");

      const label = document.createElement("div");
      label.className = "part-label";
      label.textContent = PART_LABELS[name];
      line.appendChild(label);

      const chips = document.createElement("div");
      chips.className = "part-options";
      for (const o of options) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        if (o.id === value) chip.classList.add("selected");
        if (o.disabled) {
          chip.classList.add("disabled");
          chip.title = o.reason || "Not available on this boat";
        } else if (o.blurb) {
          chip.title = o.blurb;
        }
        chip.textContent = o.name;
        chip.addEventListener("click", () => {
          if (o.disabled) return;
          row = rowIdx;
          setRowValue(name, o.id);
          render();
        });
        chips.appendChild(chip);
      }
      line.appendChild(chips);

      const blurb = document.createElement("div");
      blurb.className = "part-blurb";
      const chosen = options.find((o) => o.id === value);
      blurb.textContent = chosen && chosen.blurb ? chosen.blurb : "";
      line.appendChild(blurb);

      partsEl.appendChild(line);
    }
  }

  function render() {
    if (!catalog) return;
    renderBoats();
    renderStats();
    renderParts();
  }

  // --- public API -------------------------------------------------------

  function setCatalog(payload) {
    catalog = payload;
    statMax = {};
    for (const b of catalog.boats) {
      for (const s of b.stats) {
        statMax[s.label] = Math.max(statMax[s.label] || 0, s.value);
      }
    }
    initialSelection();
    gridEl.hidden = false;
    statusEl.hidden = true;
    errorEl.hidden = true;
    render();
    show();
  }

  /** Message shown while there is no catalog yet (connecting, or an old server). */
  function setStatus(text) {
    statusEl.textContent = text;
    statusEl.hidden = false;
  }

  function showError(text) {
    errorEl.textContent = text;
    errorEl.hidden = false;
    launchEl.disabled = false;
    launchEl.textContent = "Set sail  ⏎";
  }

  function show() {
    open = true;
    root.classList.remove("hidden");
    launchEl.disabled = false;
    launchEl.textContent = "Set sail  ⏎";
  }

  function hide() {
    open = false;
    root.classList.add("hidden");
  }

  function launch() {
    if (!catalog) return;
    store();
    launchEl.disabled = true;
    launchEl.textContent = "Rigging…";
    errorEl.hidden = true;
    onLaunch({
      boat: selection.boat,
      parts: { ...selection.parts },
      helm: selection.helm,
    });
  }

  /**
   * Keyboard navigation. Returns true when the key was consumed, so main.js
   * does not also treat it as a helm input.
   * @param {KeyboardEvent} ev
   */
  function handleKey(ev) {
    if (!open) return false;
    if (!catalog) return true; // swallow keys while waiting; nothing to steer yet
    switch (ev.code || ev.key) {
      case "ArrowUp":
      case "KeyW":
        row = (row - 1 + ROWS.length) % ROWS.length;
        break;
      case "ArrowDown":
      case "KeyS":
        row = (row + 1) % ROWS.length;
        break;
      case "ArrowLeft":
      case "KeyA":
        stepRow(ROWS[row], -1);
        break;
      case "ArrowRight":
      case "KeyD":
        stepRow(ROWS[row], 1);
        break;
      case "Enter":
      case "Space":
        ev.preventDefault();
        if (!launchEl.disabled) launch();
        return true;
      default:
        return false;
    }
    ev.preventDefault();
    render();
    return true;
  }

  launchEl.addEventListener("click", () => {
    if (!launchEl.disabled) launch();
  });

  /** Human-readable summary of a build, for the HUD. */
  function describe(build) {
    if (!catalog) return "";
    const boat = boatById(build.boat);
    const name = boat ? boat.name : build.boat;
    const parts = PART_ROWS.map((p) => {
      const opt = catalog.parts[p].find((o) => o.id === build.parts[p]);
      return opt ? opt.name : build.parts[p];
    });
    return `${name} · ${parts.join(" / ")}`;
  }

  return {
    setCatalog,
    setStatus,
    showError,
    show,
    hide,
    handleKey,
    describe,
    isOpen: () => open,
    hasCatalog: () => catalog !== null,
    // This app does present a boat picker, so the sim should greet the
    // server and let it hold the catalog open until a build is chosen.
    wantsCatalog: () => true,
  };
}

function formatStat(v) {
  if (v >= 100) return v.toFixed(0);
  if (v >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
