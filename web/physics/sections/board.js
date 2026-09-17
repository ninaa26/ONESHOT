/**
 * Section 01 — the force board, and the helm.
 *
 * Two modes over the same plan view:
 *   inspect — pin the state with sliders and read every intermediate value
 *   sail    — arrow keys drive the boat, integrated in-browser with the ported
 *             models (same RK4 step, same servo, same sheeting rules as
 *             `SailboatHub.step`), no backend required
 *
 * Either way the solution is published to `state.board`, so every other figure
 * on the page tracks whatever the boat is doing right now.
 */

import { el, arrowPath } from "../charts.js";
import { h, slider, readout, uval, fmt, mount, chips } from "../ui.js";
import {
  DEG, RAD, solveAt, stepSim, rudderServoStep, clamp,
} from "../../shared/physics/models.js";
import { polarAt } from "../../shared/physics/analysis.js";
import { state, subscribe, update, touched } from "../state.js";

const SIZE = 560;
const CX = SIZE / 2;
const CY = SIZE / 2;

/** Boat-frame (x fwd, y starboard) metres -> screen pixels, bow up. */
const toScreen = (bx, by, ppm) => [CX + by * ppm, CY - bx * ppm];

const COLORS = {
  hull: "var(--c-hull)",
  keel: "var(--c-keel)",
  sail: "var(--c-sail)",
  rudder: "var(--c-rudder)",
  total: "var(--c-total)",
};

// Helm rates, matched to web/main.js so the two UIs feel the same.
const RUDDER_MAX_DEG = 35;
const SAIL_MAX_DEG = 90;
const RUDDER_RATE_DEG = 80;
const RUDDER_CENTER_RATE_DEG = 220;
const SAIL_RATE_DEG = 90;
const PUBLISH_EVERY = 6; // frames between state.board updates while sailing

export function init() {
  const board = {
    headingDeg: 135, // wind blows toward 90 deg, so this is a beat
    u: 1.2,
    v: -0.05,
    r: 0,
    sheetDeg: 25,
    rudderDeg: 0,
    showForces: "all",
  };

  // Helm/world state, only meaningful in sail mode.
  const helm = {
    on: false,
    x: 0,
    y: 0,
    servoDeg: 0,
    rudderCmdDeg: 0,
    wake: [],
    distance: 0,
    frameCount: 0,
    /** Frozen transform tree carried between steps, as SailboatHub does. */
    frames: null,
    keys: { left: false, right: false, up: false, down: false },
  };

  const svg = el("svg", {
    viewBox: `0 0 ${SIZE} ${SIZE}`,
    tabindex: "0",
    style: "cursor:grab;touch-action:none;outline:none",
  });
  const gStage = el("g");
  const gWake = el("g");
  const gBoat = el("g");
  const gFlow = el("g");
  const gForce = el("g");
  const gLabel = el("g");
  svg.append(gStage, gWake, gBoat, gFlow, gForce, gLabel);

  const ro = readout([
    { sep: "state" },
    { key: "heading", label: "heading ψ" },
    { key: "twa", label: "true wind angle" },
    { key: "speed", label: "boat speed" },
    { key: "vmg", label: "VMG upwind" },
    { key: "leeway", label: "leeway β" },
    { sep: "apparent wind" },
    { key: "aws", label: "AWS at sail" },
    { key: "awa", label: "AWA" },
    { key: "aoa", label: "sail AoA α" },
    { sep: "sail" },
    { key: "sheetstate", label: "sheet" },
    { key: "boom", label: "boom angle" },
    { key: "luff", label: "luff scale" },
    { key: "clcd", label: "CL / CD" },
    { key: "liftdrag", label: "lift / drag" },
    { sep: "forces — boat frame (N)" },
    { key: "fhull", label: "hull" },
    { key: "fkeel", label: "keel" },
    { key: "fsail", label: "sail" },
    { key: "frudder", label: "rudder" },
    { key: "ftotal", label: "Σ", hl: true },
    { sep: "moments about CG (N·m)" },
    { key: "mkeel", label: "keel" },
    { key: "msail", label: "sail" },
    { key: "mrudder", label: "rudder" },
    { key: "mhull", label: "hull damping" },
    { key: "mtotal", label: "Σ", hl: true },
    { sep: "acceleration" },
    { key: "acc", label: "du/dt, dv/dt" },
    { key: "yawacc", label: "dr/dt" },
  ]);

  // --- tells: the updated models, lit up as they engage ------------------

  const TELLS = [
    ["sheeted", "SHEETED", "the sheet is loaded — boom is on the geometric stop"],
    ["luffing", "LUFFING", "apparent flow is inside the luff dead band — lift is scaled down"],
    ["rudderstall", "RUDDER CLAMPED", "rudder angle of attack hit aoa_limit_deg"],
    ["keelstall", "KEEL STALLED", "leeway is past the section's stall angle"],
    ["servo", "SERVO MOVING", "helm is rate-limited or self-centring"],
    ["diverged", "RESET", "the state ran away and the helm was stopped"],
  ];
  const tellNodes = {};
  const tellRow = h("div", { class: "btn-row" }, TELLS.map(([key, label, title]) => {
    const node = h("span", { class: "tell", title, text: label });
    tellNodes[key] = node;
    return node;
  }));

  const gauge = h("div", { class: "gauge" }, [
    h("div", { class: "gauge-head" }, [
      h("span", { class: "ctl-label", text: "vs polar target" }),
      h("span", { class: "ctl-val", html: "—" }),
    ]),
    h("div", { class: "gauge-track" }, h("div", { class: "gauge-fill" })),
  ]);
  const gaugeVal = gauge.querySelector(".ctl-val");
  const gaugeFill = gauge.querySelector(".gauge-fill");

  // --- controls ---------------------------------------------------------

  const sHeading = slider({
    label: "Heading ψ", hint: "world", min: -180, max: 180, step: 1,
    value: board.headingDeg, unit: "°",
    onInput: (v) => { board.headingDeg = v; if (!helm.on) render(); },
  });
  const sU = slider({
    label: "Surge u", min: -0.5, max: 4, step: 0.01, value: board.u, unit: " m/s",
    fmt: (v) => v.toFixed(2),
    onInput: (v) => { board.u = v; if (!helm.on) render(); },
  });
  const sV = slider({
    label: "Sway v", hint: "leeway", min: -1, max: 1, step: 0.01, value: board.v, unit: " m/s",
    fmt: (v) => v.toFixed(2),
    onInput: (v) => { board.v = v; if (!helm.on) render(); },
  });
  const sR = slider({
    label: "Yaw rate r", min: -1.5, max: 1.5, step: 0.01, value: board.r, unit: " rad/s",
    fmt: (v) => v.toFixed(2),
    onInput: (v) => { board.r = v; if (!helm.on) render(); },
  });
  const sSheet = slider({
    label: "Sheet limit", hint: "↑ / ↓ while sailing", min: 0, max: SAIL_MAX_DEG, step: 1,
    value: board.sheetDeg, unit: "°",
    onInput: (v) => { board.sheetDeg = v; if (!helm.on) render(); },
  });
  const sRudder = slider({
    label: "Rudder", hint: "← / → while sailing", min: -RUDDER_MAX_DEG, max: RUDDER_MAX_DEG,
    step: 0.5, value: board.rudderDeg, unit: "°", fmt: (v) => v.toFixed(1),
    onInput: (v) => { board.rudderDeg = v; if (!helm.on) render(); },
  });

  const filter = chips(
    [["all", "all forces"], ["sail", "sail only"], ["hydro", "hydro only"], ["none", "hide"]],
    "all",
    (id) => { board.showForces = id; render(); },
  );

  const btnSail = h("button", { class: "chip", text: "▶ sail it", onclick: () => setHelm(!helm.on) });
  const btnSettle = h("button", { class: "chip", text: "settle ⟳", onclick: () => settle() });
  const btnTrim = h("button", { class: "chip", text: "best trim", onclick: () => bestTrim() });

  const stateSliders = h("div", {}, [sHeading.node, sU.node, sV.node]);

  // --- helm loop --------------------------------------------------------

  /** Put the boat back on a sane close-hauled starting state. */
  function resetHelmState() {
    helm.x = 0;
    helm.y = 0;
    helm.wake = [];
    helm.distance = 0;
    helm.rudderCmdDeg = 0;
    helm.servoDeg = 0;
    helm.frames = null;
    board.u = 1.0;
    board.v = 0;
    board.r = 0;
    board.rudderDeg = 0;
    sU.set(board.u.toFixed(2));
    sV.set(board.v.toFixed(2));
    sR.set(board.r.toFixed(2));
    sRudder.set(board.rudderDeg.toFixed(1));
    render();
  }

  function setHelm(on) {
    helm.on = on;
    btnSail.textContent = on ? "■ stop" : "▶ sail it";
    btnSail.classList.toggle("on", on);
    stateSliders.style.opacity = on ? "0.45" : "1";
    stateSliders.style.pointerEvents = on ? "none" : "auto";
    btnSettle.disabled = on;
    btnTrim.disabled = on;
    if (on) {
      helm.x = 0;
      helm.y = 0;
      helm.wake = [];
      helm.distance = 0;
      helm.servoDeg = board.rudderDeg;
      helm.rudderCmdDeg = 0;
      helm.diverged = false;
      helm.frames = null;
      helm.last = performance.now();
      svg.focus({ preventScroll: true });
      requestAnimationFrame(frame);
    }
    render();
  }

  function frame(now) {
    if (!helm.on) return;
    const dt = Math.min((now - helm.last) / 1000, 0.05);
    helm.last = now;

    // Integrate the helm commands exactly as web/main.js does.
    if (helm.keys.left) {
      helm.rudderCmdDeg = Math.min(RUDDER_MAX_DEG, helm.rudderCmdDeg + RUDDER_RATE_DEG * dt);
    }
    if (helm.keys.right) {
      helm.rudderCmdDeg = Math.max(-RUDDER_MAX_DEG, helm.rudderCmdDeg - RUDDER_RATE_DEG * dt);
    }
    if (!helm.keys.left && !helm.keys.right) {
      const step = RUDDER_CENTER_RATE_DEG * dt;
      helm.rudderCmdDeg = helm.rudderCmdDeg > 0
        ? Math.max(0, helm.rudderCmdDeg - step)
        : Math.min(0, helm.rudderCmdDeg + step);
    }
    if (helm.keys.up) board.sheetDeg = Math.min(SAIL_MAX_DEG, board.sheetDeg + SAIL_RATE_DEG * dt);
    if (helm.keys.down) board.sheetDeg = Math.max(0, board.sheetDeg - SAIL_RATE_DEG * dt);

    const before = helm.servoDeg;
    helm.servoDeg = rudderServoStep(helm.servoDeg, helm.rudderCmdDeg, dt, state.cfgs.rudder);
    helm.servoMoving = Math.abs(helm.servoDeg - before) > 1e-3;
    board.rudderDeg = helm.servoDeg;

    // Fixed-step physics so the browser's frame rate never changes the result.
    const psi = board.headingDeg * RAD;
    let arr = [helm.x, helm.y, Math.cos(psi), Math.sin(psi), board.u, board.v, board.r];
    const simDt = Number(state.cfgs.simulation.dt ?? 0.02);
    let remaining = dt;
    let guard = 0;
    while (remaining > 1e-6 && guard < 8) {
      const step = Math.min(simDt, remaining);
      const out = stepSim(arr, step, state.cfgs, {
        sheetRad: board.sheetDeg * RAD,
        rudderRad: helm.servoDeg * RAD,
      }, helm.frames);
      arr = out.arr;
      helm.frames = out.frames;
      remaining -= step;
      guard += 1;
    }

    // The models can run away if the state is driven somewhere the real sim
    // never goes (huge leeway makes the keel produce kilonewtons on a 27 kg
    // boat). Catch it here rather than rendering NaN geometry.
    if (!arr.every(Number.isFinite) || Math.hypot(arr[4], arr[5]) > 40 || Math.abs(arr[6]) > 40) {
      helm.diverged = true;
      setHelm(false);
      resetHelmState();
      return;
    }
    helm.diverged = false;

    helm.distance += Math.hypot(arr[0] - helm.x, arr[1] - helm.y);
    helm.x = arr[0];
    helm.y = arr[1];
    board.headingDeg = Math.atan2(arr[3], arr[2]) * DEG;
    board.u = arr[4];
    board.v = arr[5];
    board.r = arr[6];

    helm.wake.push([helm.x, helm.y]);
    if (helm.wake.length > 260) helm.wake.shift();

    sHeading.set(Math.round(board.headingDeg));
    sU.set(board.u.toFixed(2));
    sV.set(board.v.toFixed(2));
    sR.set(board.r.toFixed(2));
    sSheet.set(Math.round(board.sheetDeg));
    sRudder.set(board.rudderDeg.toFixed(1));

    helm.frameCount += 1;
    render(helm.frameCount % PUBLISH_EVERY === 0);
    requestAnimationFrame(frame);
  }

  const KEYMAP = {
    ArrowLeft: "left", KeyA: "left",
    ArrowRight: "right", KeyD: "right",
    ArrowUp: "up", KeyW: "up",
    ArrowDown: "down", KeyS: "down",
  };

  window.addEventListener("keydown", (ev) => {
    if (!helm.on) return;
    const key = KEYMAP[ev.code];
    if (!key) return;
    helm.keys[key] = true;
    ev.preventDefault();
  });
  window.addEventListener("keyup", (ev) => {
    const key = KEYMAP[ev.code];
    if (key) helm.keys[key] = false;
  });

  /** Integrate to steady state with a heading-hold autopilot, then adopt it. */
  function settle() {
    const target = board.headingDeg * RAD;
    let arr = [0, 0, Math.cos(target), Math.sin(target), Math.max(board.u, 0.3), 0, 0];
    let servo = board.rudderDeg;
    let frames = null;
    const dt = 0.02;
    for (let i = 0; i < 1000; i += 1) {
      const psi = Math.atan2(arr[3], arr[2]);
      const errSin = Math.cos(psi) * Math.sin(target) - Math.sin(psi) * Math.cos(target);
      const errCos = Math.cos(psi) * Math.cos(target) + Math.sin(psi) * Math.sin(target);
      const err = Math.atan2(errSin, errCos);
      const cmd = clamp(-40 * err + 3 * arr[6], -RUDDER_MAX_DEG, RUDDER_MAX_DEG);
      servo = rudderServoStep(servo, cmd, dt, state.cfgs.rudder);
      const out = stepSim(arr, dt, state.cfgs, {
        sheetRad: board.sheetDeg * RAD,
        rudderRad: servo * RAD,
      }, frames);
      arr = out.arr;
      frames = out.frames;
    }
    board.u = arr[4];
    board.v = arr[5];
    board.r = arr[6];
    board.rudderDeg = servo;
    board.headingDeg = Math.atan2(arr[3], arr[2]) * DEG;
    sU.set(board.u.toFixed(2));
    sV.set(board.v.toFixed(2));
    sR.set(board.r.toFixed(2));
    sRudder.set(board.rudderDeg.toFixed(1));
    sHeading.set(Math.round(board.headingDeg));
    render();
  }

  /** Sweep the sheet limit for the most driving force at the current state. */
  function bestTrim() {
    let best = { fx: -Infinity, deg: 0 };
    for (let deg = 0; deg <= SAIL_MAX_DEG; deg += 1) {
      const res = solveAt(board, board.headingDeg * RAD, state.cfgs, {
        sheetRad: deg * RAD, rudderRad: board.rudderDeg * RAD,
      });
      if (res.sail.fx > best.fx) best = { fx: res.sail.fx, deg };
    }
    board.sheetDeg = best.deg;
    sSheet.set(best.deg);
    render();
  }

  // --- drag to steer (inspect mode) -------------------------------------

  let dragging = false;
  const pointerAngle = (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * SIZE - CX;
    const py = ((ev.clientY - rect.top) / rect.height) * SIZE - CY;
    return Math.atan2(-py, px) * DEG;
  };
  let grabOffset = 0;
  svg.addEventListener("pointerdown", (ev) => {
    svg.focus({ preventScroll: true });
    if (helm.on) return;
    dragging = true;
    grabOffset = board.headingDeg - pointerAngle(ev);
    svg.setPointerCapture(ev.pointerId);
    svg.style.cursor = "grabbing";
  });
  svg.addEventListener("pointermove", (ev) => {
    if (!dragging) return;
    const next = ((((pointerAngle(ev) + grabOffset) + 180) % 360) + 360) % 360 - 180;
    board.headingDeg = next;
    sHeading.set(Math.round(next));
    render();
  });
  const endDrag = () => { dragging = false; svg.style.cursor = helm.on ? "default" : "grab"; };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);

  // --- drawing ----------------------------------------------------------

  function drawStage(ppm) {
    gStage.replaceChildren();

    if (helm.on && Number.isFinite(helm.x) && Number.isFinite(helm.y)) {
      // Water grid in world coordinates, so motion is visible. Indexed by step
      // count rather than by accumulating a float, so a huge (or non-finite)
      // position can never stall the loop.
      const spacing = 1.0;
      const reach = (SIZE / 2) / ppm + spacing;
      const steps = Math.min(Math.ceil((2 * reach) / spacing), 64);
      const psi = board.headingDeg * RAD;
      const c = Math.cos(-psi);
      const s = Math.sin(-psi);
      const x0 = Math.floor((helm.x - reach) / spacing) * spacing;
      const y0 = Math.floor((helm.y - reach) / spacing) * spacing;
      for (let i = 0; i <= steps; i += 1) {
        for (let j = 0; j <= steps; j += 1) {
          const dx = x0 + i * spacing - helm.x;
          const dy = y0 + j * spacing - helm.y;
          const [px, py] = toScreen(c * dx - s * dy, s * dx + c * dy, ppm);
          if (px < 0 || px > SIZE || py < 0 || py > SIZE) continue;
          gStage.append(el("circle", {
            cx: px.toFixed(1), cy: py.toFixed(1), r: 1.1,
            fill: "rgba(140,180,230,0.22)",
          }));
        }
      }
    } else if (!helm.on) {
      for (let r = 1; r <= 3; r += 1) {
        gStage.append(el("circle", {
          cx: CX, cy: CY, r: r * ppm, fill: "none",
          stroke: "rgba(140,180,230,0.09)", "stroke-width": 1,
        }));
        gStage.append(el("text", {
          x: CX + 4, y: CY - r * ppm - 3, fill: "#4a5769",
          "font-size": 9, "font-family": "var(--mono)",
        }, `${r} m`));
      }
    }

    gStage.append(el("line", {
      x1: CX, y1: 22, x2: CX, y2: SIZE - 22,
      stroke: "rgba(140,180,230,0.12)", "stroke-dasharray": "4 6", "stroke-width": 1,
    }));
    gStage.append(el("text", {
      x: CX, y: 16, "text-anchor": "middle", fill: "#4a5769",
      "font-size": 9.5, "font-family": "var(--mono)",
    }, "BOW"));
    gStage.append(el("text", {
      x: SIZE - 18, y: CY - 6, "text-anchor": "end", fill: "#4a5769",
      "font-size": 9.5, "font-family": "var(--mono)",
    }, "STARBOARD +y"));
  }

  function drawWake(ppm) {
    gWake.replaceChildren();
    if (!helm.on || helm.wake.length < 2) return;
    const psi = board.headingDeg * RAD;
    const c = Math.cos(-psi);
    const s = Math.sin(-psi);
    let d = "";
    helm.wake.forEach(([wx, wy], i) => {
      const dx = wx - helm.x;
      const dy = wy - helm.y;
      const [px, py] = toScreen(c * dx - s * dy, s * dx + c * dy, ppm);
      d += `${i ? "L" : "M"}${px.toFixed(1)} ${py.toFixed(1)}`;
    });
    gWake.append(el("path", {
      d, fill: "none", stroke: "var(--accent)", "stroke-width": 1.4,
      opacity: 0.4, "stroke-linecap": "round",
    }));
  }

  function drawBoat(ppm, res) {
    gBoat.replaceChildren();
    const hull = state.cfgs.hull;
    const L = Number(hull.L);
    const B = Number(hull.B);
    const bow = 0.6 * L;
    const stern = -0.4 * L;
    const half = B / 2;
    const pts = [
      [bow, 0], [0.32 * L, half * 0.78], [stern + 0.12 * L, half], [stern, half * 0.82],
      [stern, -half * 0.82], [stern + 0.12 * L, -half], [0.32 * L, -half * 0.78],
    ];
    gBoat.append(el("path", {
      d: `${pts.map(([bx, by], i) => {
        const [px, py] = toScreen(bx, by, ppm);
        return `${i ? "L" : "M"}${px.toFixed(1)} ${py.toFixed(1)}`;
      }).join("")}Z`,
      fill: "rgba(220,231,245,0.07)",
      stroke: "rgba(220,231,245,0.5)",
      "stroke-width": 1.4,
      "stroke-linejoin": "round",
    }));

    const keelChord = Math.min(Math.sqrt(Number(state.cfgs.keel.area)) * 0.75, 0.5 * L);
    const keelX = Number(state.cfgs.keel.x_pos ?? 0);
    const [kx0, ky0] = toScreen(keelX + keelChord / 2, 0, ppm);
    const [kx1, ky1] = toScreen(keelX - keelChord / 2, 0, ppm);
    gBoat.append(el("line", {
      x1: kx0, y1: ky0, x2: kx1, y2: ky1,
      stroke: "var(--c-keel)", "stroke-width": 4, "stroke-linecap": "round", opacity: 0.55,
    }));

    const rudChord = Math.min(Math.sqrt(Number(state.cfgs.rudder.area)) * 0.9, 0.3 * L);
    const rx = Number(state.cfgs.rudder.x_pos ?? 0);
    const dr = board.rudderDeg * RAD;
    const [rx0, ry0] = toScreen(rx, 0, ppm);
    const [rx1, ry1] = toScreen(rx - rudChord * Math.cos(dr), -rudChord * Math.sin(dr), ppm);
    gBoat.append(el("line", {
      x1: rx0, y1: ry0, x2: rx1, y2: ry1,
      stroke: "var(--c-rudder)", "stroke-width": 4, "stroke-linecap": "round",
    }));

    // Boom, kept inside the boat's own envelope so it reads as a boom.
    const sx = Number(state.cfgs.sail.x_pos ?? 0);
    const boomLen = Math.min(Math.sqrt(Number(state.cfgs.sail.area)) * 0.6, sx + 0.45 * L);
    const ds = res.sailRad;
    const [mx, my] = toScreen(sx, 0, ppm);
    const clewX = sx - boomLen * Math.cos(ds);
    const clewY = -boomLen * Math.sin(ds);
    const [bx1, by1] = toScreen(clewX, clewY, ppm);

    if (res.sheet.sheetLimit > 0.01) {
      const side = Math.sign(res.sailRad) || -1;
      const lim = res.sheet.sheetLimit * side;
      const [lx, ly] = toScreen(sx - boomLen * 1.18 * Math.cos(lim), -boomLen * 1.18 * Math.sin(lim), ppm);
      gBoat.append(el("line", {
        x1: mx, y1: my, x2: lx, y2: ly,
        stroke: "var(--warn)", "stroke-width": 1.2, "stroke-dasharray": "3 3", opacity: 0.75,
      }));
    }

    // Sail: a cambered panel off the boom, bellied away from the wind so the
    // working side is obvious at a glance. Dashed and flat when luffing.
    const luffing = res.sail.luffScale < 0.999;
    const belly = luffing ? 0 : 0.22 * boomLen * (Math.sign(res.sailRad) || -1);
    const [cx1, cy1] = toScreen(
      (sx + clewX) / 2 - belly * Math.sin(ds),
      (0 + clewY) / 2 + belly * Math.cos(ds),
      ppm,
    );
    gBoat.append(el("path", {
      d: `M${mx} ${my}Q${cx1.toFixed(1)} ${cy1.toFixed(1)} ${bx1.toFixed(1)} ${by1.toFixed(1)}`,
      fill: "none", stroke: "var(--c-sail)", "stroke-width": 3.4, "stroke-linecap": "round",
      opacity: luffing ? 0.45 : 1,
      "stroke-dasharray": luffing ? "5 4" : null,
    }));
    if (!luffing) {
      gBoat.append(el("path", {
        d: `M${mx} ${my}Q${cx1.toFixed(1)} ${cy1.toFixed(1)} ${bx1.toFixed(1)} ${by1.toFixed(1)}L${mx} ${my}Z`,
        fill: "var(--c-sail)", "fill-opacity": 0.16, stroke: "none",
      }));
    }
    gBoat.append(el("circle", { cx: mx, cy: my, r: 3.4, fill: "var(--c-sail)" }));
    gBoat.append(el("circle", {
      cx: CX, cy: CY, r: 3.2, fill: "none", stroke: "var(--text)", "stroke-width": 1.2,
    }));
  }

  function drawFlow(ppm, res) {
    gFlow.replaceChildren();
    const psi = board.headingDeg * RAD;
    const vecLen = 1.15 * ppm;

    const draw = (bx, by, color, label, dash) => {
      const m = Math.hypot(bx, by);
      if (m < 1e-6) return;
      const sxv = (by / m) * vecLen;
      const syv = -(bx / m) * vecLen;
      gFlow.append(el("path", {
        d: arrowPath(CX - sxv, CY - syv, CX - sxv * 0.18, CY - syv * 0.18, 8),
        stroke: color, "stroke-width": 1.6, fill: "none",
        "stroke-dasharray": dash, opacity: 0.95,
      }));
      gFlow.append(el("text", {
        x: CX - sxv + (sxv > 0 ? 8 : -8), y: CY - syv - 8,
        "text-anchor": sxv > 0 ? "start" : "end",
        fill: color, "font-size": 10, "font-family": "var(--mono)",
      }, label));
    };

    const windSpeed = Number(state.cfgs.sail.wind_speed);
    const windDir = Number(state.cfgs.sail.wind_dir_deg) * RAD;
    draw(
      -Math.cos(windDir - psi) * windSpeed,
      -Math.sin(windDir - psi) * windSpeed,
      "var(--c-wind)", `TWS ${fmt(windSpeed, 1)}`, "5 3",
    );

    const aw = res.sail.awBoat;
    draw(-aw[0], -aw[1], "var(--accent)", `AWS ${fmt(res.sail.awSpeed, 1)}`);

    const speed = Math.hypot(board.u, board.v);
    if (speed > 1e-3) {
      const sxv = (board.v / speed) * ppm * 0.9;
      const syv = -(board.u / speed) * ppm * 0.9;
      gFlow.append(el("path", {
        d: arrowPath(CX, CY, CX + sxv, CY + syv, 8),
        stroke: "#ffffff", "stroke-width": 1.5, fill: "none", opacity: 0.55,
      }));
      gFlow.append(el("text", {
        x: CX + sxv + 8, y: CY + syv + 4, fill: "rgba(255,255,255,0.65)",
        "font-size": 10, "font-family": "var(--mono)",
      }, `V ${fmt(speed, 2)} m/s`));
    }
  }

  function drawForces(ppm, res) {
    gForce.replaceChildren();
    gLabel.replaceChildren();
    if (board.showForces === "none") return;

    const show = {
      all: ["hull", "keel", "sail", "rudder", "total"],
      sail: ["sail"],
      hydro: ["hull", "keel", "rudder"],
    }[board.showForces];

    const peak = Math.max(...show.map((k) => {
      const f = k === "total" ? res.total : res[k];
      return Math.hypot(f.fx, f.fy);
    }), 1);
    const pxPerN = 150 / peak;

    for (const key of show) {
      const f = key === "total" ? res.total : res[key];
      const mag = Math.hypot(f.fx, f.fy);
      if (mag * pxPerN < 3) continue;
      const cfg = key === "total" ? { x_pos: 0, y_pos: 0 } : state.cfgs[key];
      const [ox, oy] = toScreen(Number(cfg.x_pos ?? 0), Number(cfg.y_pos ?? 0), ppm);
      const ex = ox + f.fy * pxPerN;
      const ey = oy - f.fx * pxPerN;
      gForce.append(el("path", {
        d: arrowPath(ox, oy, ex, ey, 9),
        stroke: COLORS[key], "stroke-width": key === "total" ? 2.6 : 1.9,
        fill: "none", "stroke-linecap": "round",
        opacity: key === "total" ? 1 : 0.92,
      }));
      gLabel.append(el("text", {
        x: ex + (f.fy >= 0 ? 7 : -7), y: ey + 4,
        "text-anchor": f.fy >= 0 ? "start" : "end",
        fill: COLORS[key], "font-size": 10, "font-family": "var(--mono)",
      }, `${key} ${fmt(mag, 0)} N`));
    }

    gLabel.append(el("text", {
      x: 20, y: SIZE - 18, fill: "var(--text-faint)",
      "font-size": 9.5, "font-family": "var(--mono)",
    }, `arrow scale ≈ ${fmt(peak, 0)} N full length`));

    const mz = res.total.mz;
    if (Math.abs(mz) > 0.5) {
      const rad = 0.55 * ppm;
      const sweep = clamp(Math.abs(mz) / 40, 0.12, 1) * Math.PI;
      const dir = Math.sign(mz);
      const a0 = -Math.PI / 2;
      const a1 = a0 - dir * sweep;
      const p0 = [CX + rad * Math.cos(a0), CY + rad * Math.sin(a0)];
      const p1 = [CX + rad * Math.cos(a1), CY + rad * Math.sin(a1)];
      gForce.append(el("path", {
        d: `M${p0[0].toFixed(1)} ${p0[1].toFixed(1)}A${rad} ${rad} 0 0 ${dir > 0 ? 0 : 1} ${p1[0].toFixed(1)} ${p1[1].toFixed(1)}`,
        fill: "none", stroke: "var(--c-total)", "stroke-width": 1.6,
        "stroke-dasharray": "5 3", opacity: 0.7,
      }));
      gLabel.append(el("text", {
        x: p1[0] + 6, y: p1[1], fill: "var(--c-total)",
        "font-size": 9.5, "font-family": "var(--mono)",
      }, `Mz ${fmt(mz, 1)} N·m`));
    }
  }

  /** How close the boat is to the polar's steady-state speed for this angle. */
  function updateGauge(twa, speed) {
    const polar = polarAt(state.config, Number(state.cfgs.sail.wind_speed));
    if (!polar) {
      gaugeVal.innerHTML = '<span style="color:var(--text-faint)">no polar for this boat</span>';
      gaugeFill.style.width = "0%";
      return;
    }
    const twaList = polar.twa_deg;
    let i = 0;
    while (i < twaList.length - 2 && twaList[i + 1] < twa) i += 1;
    const span = twaList[i + 1] - twaList[i] || 1;
    const f = clamp((twa - twaList[i]) / span, 0, 1);
    const target = polar.speed[i] + f * (polar.speed[i + 1] - polar.speed[i]);
    const ratio = target > 1e-6 ? speed / target : 0;
    gaugeVal.innerHTML = `${fmt(speed, 2)} / ${fmt(target, 2)} m/s <span class="u">${(ratio * 100).toFixed(0)}%</span>`;
    gaugeFill.style.width = `${clamp(ratio * 100, 0, 100).toFixed(1)}%`;
    gaugeFill.style.background = ratio > 0.92
      ? "var(--new)"
      : ratio > 0.7 ? "var(--accent)" : "var(--warn)";
  }

  function setTell(key, on) {
    if (tellNodes[key]) tellNodes[key].classList.toggle("on", Boolean(on));
  }

  // --- solve + render ---------------------------------------------------

  function render(publish = true) {
    const psi = board.headingDeg * RAD;
    const res = solveAt(board, psi, state.cfgs, {
      sheetRad: board.sheetDeg * RAD,
      rudderRad: board.rudderDeg * RAD,
    });

    const L = Number(state.cfgs.hull.L);
    const ppm = (SIZE * 0.5) / (L * 1.9);

    drawStage(ppm);
    drawWake(ppm);
    drawBoat(ppm, res);
    drawFlow(ppm, res);
    drawForces(ppm, res);

    const mass = Number(state.cfgs.boat.mass ?? 27);
    const iz = Number(state.cfgs.boat.inertia_z ?? 10);
    const speed = Math.hypot(board.u, board.v);
    const windDir = Number(state.cfgs.sail.wind_dir_deg);
    const twa = Math.abs(((((windDir + 180 - board.headingDeg) % 360) + 540) % 360) - 180);
    const vmg = speed * Math.cos(twa * RAD);

    setTell("sheeted", res.sheet.sheeted);
    setTell("luffing", res.sail.luffScale < 0.999);
    setTell("rudderstall", res.rudder.clamped);
    setTell("keelstall", Math.abs(res.keel.leewayDeg) > 11);
    setTell("servo", helm.on && helm.servoMoving);
    setTell("diverged", Boolean(helm.diverged));
    updateGauge(twa, speed);

    ro.set("heading", uval(board.headingDeg, 1, "°"));
    ro.set("twa", uval(twa, 1, "°"));
    ro.set("speed", uval(speed, 3, " m/s"));
    ro.set("vmg", uval(vmg, 3, " m/s"));
    ro.set("leeway", uval(Math.atan2(board.v, board.u) * DEG, 2, "°"));
    ro.set("aws", uval(res.sail.awSpeed, 3, " m/s"));
    ro.set("awa", uval(res.sail.awaDeg, 1, "°"));
    ro.set("aoa", uval(res.sail.aoaDeg ?? 0, 1, "°"));
    ro.set("sheetstate", res.sheet.sheeted
      ? '<span style="color:var(--warn)">sheeted — on the stop</span>'
      : '<span style="color:var(--text-faint)">free — weather-vaning</span>');
    ro.set("boom", uval(res.sailRad * DEG, 1, "°"));
    ro.set("luff", `${fmt(res.sail.luffScale, 2)}${res.sail.luffScale < 0.999 ? ' <span class="u">luffing</span>' : ""}`);
    ro.set("clcd", `${fmt(res.sail.cl, 3)} / ${fmt(res.sail.cd, 3)}`);
    ro.set("liftdrag", `${fmt(res.sail.lift, 1)} / ${fmt(res.sail.drag, 1)}<span class="u">N</span>`);

    for (const key of ["hull", "keel", "sail", "rudder"]) {
      ro.set(`f${key}`, `${fmt(res[key].fx, 1)}, ${fmt(res[key].fy, 1)}`);
      ro.set(`m${key}`, fmt(res.moments[key], 2));
    }
    ro.set("ftotal", `${fmt(res.total.fx, 1)}, ${fmt(res.total.fy, 1)}`);
    ro.set("mtotal", fmt(res.total.mz, 2));

    const du = res.total.fx / mass + board.r * board.v;
    const dv = res.total.fy / mass - board.r * board.u;
    ro.set("acc", `${fmt(du, 2)}, ${fmt(dv, 2)}<span class="u">m/s²</span>`);
    ro.set("yawacc", uval(res.total.mz / iz, 2, " rad/s²"));

    if (publish) {
      update({
        board: {
          ...board,
          sailing: helm.on,
          twa,
          sailDeg: res.sailRad * DEG,
          leewayDeg: Math.atan2(board.v, board.u) * DEG,
          forces: {
            hull: res.hull, keel: res.keel, sail: res.sail, rudder: res.rudder, total: res.total,
          },
        },
      });
    }
  }

  // --- mount ------------------------------------------------------------

  mount("board-fig", svg);
  mount("board-readout", ro.node);
  mount("board-tells", tellRow, gauge);
  mount(
    "board-controls",
    h("div", { class: "grid g-2" }, [
      stateSliders,
      h("div", {}, [sSheet.node, sRudder.node, sR.node]),
    ]),
    h("div", { class: "btn-row", style: "margin-top:4px" }, [btnSail, filter.node, btnSettle, btnTrim]),
    h("div", {
      class: "fig-cap",
      html: '<span class="kbd">←</span> <span class="kbd">→</span> rudder · '
        + '<span class="kbd">↑</span> <span class="kbd">↓</span> sheet · '
        + "same bindings as the 3D simulator. Physics runs in this page — no backend needed.",
    }),
  );

  const legend = document.getElementById("board-legend");
  if (legend) {
    legend.replaceChildren(...Object.entries(COLORS).map(([k, c]) => h(
      "span", {}, [h("i", { style: `background:${c}` }), document.createTextNode(k)],
    )));
  }

  render();
  subscribe((_s, changed) => {
    if (touched(changed, "cfgs", "config", "windSpeed", "windDirDeg") && !helm.on) render();
  });
}
