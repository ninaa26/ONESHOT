/**
 * HUD for the test bench: instruments, wind, episode/scoreboard, and the
 * observation-to-action inspector.
 *
 * Owns every DOM write so `main.js` stays a loop. Same split as the 3D viewer,
 * where `hud.js` owns the readouts and `main.js` owns the loop.
 */

import { OBSERVATION_LABELS, scoreSummary } from "./task.js";

const $ = (id) => document.getElementById(id);

const TELLS = [
  ["sheeted", "SHEETED", "the sheet is loaded — the boom is on its stop"],
  ["luffing", "LUFFING", "flow is inside the luff dead band, lift is scaled down"],
  ["rudder", "RUDDER CLAMPED", "rudder angle of attack hit aoa_limit_deg"],
  ["keel", "KEEL STALLED", "leeway is past the keel section's stall angle"],
  ["irons", "IN IRONS", "head to wind with no drive"],
];

const RESULT_COLORS = {
  running: "var(--accent)",
  success: "var(--good)",
  failure: "var(--bad)",
  timeout: "var(--warn)",
};

/** Name the point of sail from the true wind angle. */
export function pointOfSail(twaAbs) {
  if (twaAbs < 32) return "in irons";
  if (twaAbs < 55) return "close hauled";
  if (twaAbs < 80) return "close reach";
  if (twaAbs < 105) return "beam reach";
  if (twaAbs < 150) return "broad reach";
  return "running";
}

export function createHud({ onToggle, onWind, onBoat, onMode, configs, modes }) {
  const els = {
    speed: $("i-speed"),
    heading: $("i-heading"),
    twa: $("i-twa"),
    pos: $("i-pos"),
    vmg: $("i-vmg"),
    sheet: $("i-sheet"),
    rudder: $("i-rudder"),
    target: $("i-target"),
    targetFill: $("i-target-fill"),
    tells: $("i-tells"),
    wTrue: $("w-true"),
    wApp: $("w-app"),
    wDir: $("w-dir"),
    eMode: $("e-mode"),
    eStep: $("e-step"),
    eTime: $("e-time"),
    eDist: $("e-dist"),
    eResult: $("e-result"),
    sEpisodes: $("s-episodes"),
    sRate: $("s-rate"),
    sTime: $("s-time"),
    sEff: $("s-eff"),
    sTacks: $("s-tacks"),
    sGybes: $("s-gybes"),
    banner: $("banner"),
  };

  // --- tells ---
  const tellNodes = {};
  for (const [key, label, title] of TELLS) {
    const node = document.createElement("span");
    node.className = "tell";
    node.textContent = label;
    node.title = title;
    tellNodes[key] = node;
    els.tells.append(node);
  }

  // --- driver modes ---
  const modeNodes = {};
  const modeHost = $("modes");
  for (const [key, label, title] of modes) {
    const btn = document.createElement("button");
    btn.className = "mode";
    btn.textContent = label;
    btn.title = title;
    btn.addEventListener("click", () => onMode(key));
    modeNodes[key] = btn;
    modeHost.append(btn);
  }

  // --- toggle buttons ---
  const TOGGLES = [
    ["showForces", "forces", "F"],
    ["showWind", "wind", "Q"],
    ["showTrail", "trail", "T"],
    ["showMap", "map", "M"],
    ["autoNext", "auto-next", "O"],
  ];
  const toggleNodes = {};
  const toggleHost = $("toggles");
  for (const [key, label, hotkey] of TOGGLES) {
    const btn = document.createElement("button");
    btn.className = "toggle";
    btn.textContent = `${label} (${hotkey})`;
    btn.addEventListener("click", () => onToggle(key));
    toggleNodes[key] = btn;
    toggleHost.append(btn);
  }

  // --- boat selector ---
  const boatSelect = document.createElement("select");
  boatSelect.className = "boat-select";
  boatSelect.setAttribute("aria-label", "Boat");
  for (const [value, label] of configs) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    boatSelect.append(option);
  }
  boatSelect.addEventListener("change", () => onBoat(boatSelect.value));
  const boatRow = document.createElement("label");
  boatRow.className = "boat-row";
  const boatLabel = document.createElement("span");
  boatLabel.textContent = "boat";
  boatRow.append(boatLabel, boatSelect);
  toggleHost.after(boatRow);

  // --- observation / action inspector ---
  function makeBarRow(host, label, color) {
    const row = document.createElement("div");
    row.className = "obs-row";
    const name = document.createElement("span");
    name.className = "obs-name";
    name.textContent = label;
    const track = document.createElement("div");
    track.className = "obs-track";
    const fill = document.createElement("div");
    fill.className = "obs-fill";
    if (color) fill.style.background = color;
    track.append(fill);
    const value = document.createElement("b");
    value.className = "obs-val";
    value.textContent = "0.00";
    row.append(name, track, value);
    host.append(row);
    return { fill, value };
  }

  const obsHost = $("obs-list");
  const obsRows = OBSERVATION_LABELS.map((label, i) => makeBarRow(obsHost, `${i} ${label}`));

  const actHost = $("act-list");
  const actRows = [
    makeBarRow(actHost, "→ rudder", "#ff9459"),
    makeBarRow(actHost, "→ sheet", "#5ee0ff"),
  ];

  /** Bar for a value in [-1, 1] (signed, grows from the centre) or [0, 1]. */
  function setBar(row, v, signed = true) {
    const clamped = Math.max(-1, Math.min(1, Number.isFinite(v) ? v : 0));
    if (signed) {
      const half = Math.abs(clamped) * 50;
      row.fill.style.left = clamped >= 0 ? "50%" : `${(50 - half).toFixed(2)}%`;
      row.fill.style.width = `${half.toFixed(2)}%`;
    } else {
      row.fill.style.left = "0%";
      row.fill.style.width = `${(Math.abs(clamped) * 100).toFixed(2)}%`;
    }
    row.value.textContent = clamped.toFixed(2);
  }

  // --- wind controls ---
  const windSpeed = $("w-speed");
  const windDir = $("w-dirdeg");
  windSpeed.addEventListener("input", () => {
    $("w-speed-val").textContent = `${Number(windSpeed.value).toFixed(1)} m/s`;
    onWind({ speed: Number(windSpeed.value) });
  });
  windDir.addEventListener("input", () => {
    $("w-dir-val").textContent = `${windDir.value}°`;
    onWind({ dirDeg: Number(windDir.value) });
  });
  for (const control of [windSpeed, windDir]) {
    control.addEventListener("change", () => $("stage").focus({ preventScroll: true }));
  }

  let bannerTimer = 0;

  return {
    setToggles(options) {
      for (const [key] of TOGGLES) {
        toggleNodes[key].classList.toggle("on", Boolean(options[key]));
      }
    },

    setMode(mode) {
      for (const [key, node] of Object.entries(modeNodes)) {
        node.classList.toggle("on", key === mode);
      }
    },

    setBoat(config) {
      boatSelect.value = config;
    },

    setWind(speed, dirDeg) {
      windSpeed.value = String(speed);
      windDir.value = String(dirDeg);
      $("w-speed-val").textContent = `${speed.toFixed(1)} m/s`;
      $("w-dir-val").textContent = `${Math.round(dirDeg)}°`;
    },

    /** One frame of instrument updates. */
    update(scene) {
      const {
        boat, wind, solved, episode, headingDeg, twa, speed, target,
        observation, action, mode, board,
      } = scene;

      els.speed.textContent = speed.toFixed(2);
      els.heading.textContent = `${((headingDeg % 360) + 360) % 360 | 0}°`;
      // Bearings increase counter-clockwise, so a positive TWA is wind to port.
      els.twa.textContent = `${Math.abs(twa).toFixed(0)}° ${twa >= 0 ? "port" : "stbd"}`;
      els.pos.textContent = pointOfSail(Math.abs(twa));
      els.vmg.textContent = `${scene.vmg.toFixed(2)} m/s`;
      els.sheet.textContent = `${boat.sheetDeg.toFixed(0)}°`;
      els.rudder.textContent = `${boat.rudderDeg.toFixed(1)}°`;

      if (target) {
        els.target.textContent = `${(target.ratio * 100).toFixed(0)}%`;
        els.targetFill.style.width = `${Math.max(0, Math.min(100, target.ratio * 100)).toFixed(1)}%`;
        els.targetFill.style.background = target.ratio > 0.92
          ? "var(--good)"
          : target.ratio > 0.7 ? "var(--accent)" : "var(--warn)";
      } else {
        els.target.textContent = "no polar";
        els.targetFill.style.width = "0%";
      }

      els.wTrue.textContent = `${wind.speed.toFixed(1)} m/s`;
      els.wApp.textContent = `${solved.sail.awSpeed.toFixed(1)} m/s @ ${solved.sail.awaDeg.toFixed(0)}°`;
      els.wDir.textContent = `${Math.round(((wind.dirDeg % 360) + 360) % 360)}°`;

      els.eMode.textContent = mode;
      els.eStep.textContent = String(episode.steps);
      els.eTime.textContent = `${episode.t.toFixed(1)} s`;
      els.eDist.textContent = `${scene.distance.toFixed(1)} m`;
      els.eResult.textContent = episode.result;
      els.eResult.style.color = RESULT_COLORS[episode.result] || "var(--text)";
      els.sTacks.textContent = String(episode.tacks);
      els.sGybes.textContent = String(episode.gybes);

      const summary = scoreSummary(board);
      els.sEpisodes.textContent = String(summary.episodes);
      els.sRate.textContent = summary.episodes
        ? `${(summary.successRate * 100).toFixed(0)}%`
        : "—";
      els.sTime.textContent = Number.isFinite(summary.meanTime)
        ? `${summary.meanTime.toFixed(1)} s`
        : "—";
      els.sEff.textContent = Number.isFinite(summary.meanEfficiency)
        ? `${(summary.meanEfficiency * 100).toFixed(0)}%`
        : "—";

      // Elements 2 and 10 (distance, wind speed) are the only unsigned ones.
      observation.forEach((v, i) => setBar(obsRows[i], v, i !== 2 && i !== 10));
      setBar(actRows[0], action[0]);
      setBar(actRows[1], action[1]);

      const tells = {
        sheeted: solved.sheet.sheeted,
        luffing: solved.sail.luffScale < 0.999,
        rudder: solved.rudder.clamped,
        keel: Math.abs(solved.keel.leewayDeg) > 11,
        irons: Math.abs(twa) < 32 && speed < 0.45,
      };
      for (const [key, node] of Object.entries(tellNodes)) {
        node.classList.toggle("on", Boolean(tells[key]));
      }
    },

    /** Big centred message; `ms = 0` keeps it up until the next call. */
    banner(headline, sub = "", ms = 1800) {
      els.banner.innerHTML = `<div class="headline">${headline}</div>`
        + (sub ? `<div class="sub">${sub}</div>` : "");
      els.banner.classList.add("show");
      window.clearTimeout(bannerTimer);
      if (ms > 0) {
        bannerTimer = window.setTimeout(() => els.banner.classList.remove("show"), ms);
      }
    },

    hideBanner() {
      window.clearTimeout(bannerTimer);
      els.banner.classList.remove("show");
    },

    setMapVisible(visible) {
      $("panel-map").style.display = visible ? "" : "none";
    },
  };
}
