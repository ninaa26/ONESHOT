/**
 * Canvas renderer for the SailBench game.
 *
 * North-up world view: the map stays put and the boat sails across it. The
 * camera follows with a dead zone, so short manoeuvres read as the boat moving
 * rather than the world sliding underneath it.
 */

import { rot, RAD, DEG } from "../shared/physics/models.js";
import { hullShape, rigDimensions } from "./boats.js";
import { TASK } from "./task.js";

export const WORLD_HALF = 200; // metres from the origin to the course boundary

/**
 * Minimum on-screen hull length, in pixels.
 *
 * A 1 m boat on a 400 m course is a speck at any honest scale, so the sprite is
 * drawn at a floor size. This is purely visual — positions, the wake, marks and
 * the physics are all still in true metres.
 */
const MIN_BOAT_PX = 46;

/** How much the hull sprite is exaggerated at the current zoom. */
const boatScaleFor = (L, ppm) => Math.max(1, MIN_BOAT_PX / Math.max(L * ppm, 1e-6));

const COMPONENT_COLORS = {
  hull: "#94a3b8",
  keel: "#5b8cff",
  sail: "#5ee0ff",
  rudder: "#ff9459",
  total: "#ffd84a",
};

/** Body-frame metres -> world metres, using the sim's own rotation. */
const toWorld = (bx, by, psi, ox, oy) => {
  const [wx, wy] = rot([bx, by], psi);
  return [ox + wx, oy + wy];
};

export function createRenderer(canvas) {
  const ctx = canvas.getContext("2d");
  let W = 0;
  let H = 0;
  let dpr = 1;

  /** Wind streaks, in world coordinates. */
  const streaks = [];
  const STREAK_COUNT = 260;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    W = Math.max(rect.width, 1);
    H = Math.max(rect.height, 1);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // --- projection -------------------------------------------------------

  let cam = { x: 0, y: 0, ppm: 18 };
  const sx = (wx) => (wx - cam.x) * cam.ppm + W / 2;
  const sy = (wy) => H / 2 - (wy - cam.y) * cam.ppm;

  /** Visible world rectangle, with a margin for spawning. */
  function viewBounds(margin = 0) {
    const halfW = W / 2 / cam.ppm + margin;
    const halfH = H / 2 / cam.ppm + margin;
    return { x0: cam.x - halfW, x1: cam.x + halfW, y0: cam.y - halfH, y1: cam.y + halfH };
  }

  // --- wind streaks -----------------------------------------------------

  function seedStreaks() {
    const b = viewBounds(10);
    streaks.length = 0;
    for (let i = 0; i < STREAK_COUNT; i += 1) {
      streaks.push({
        x: b.x0 + Math.random() * (b.x1 - b.x0),
        y: b.y0 + Math.random() * (b.y1 - b.y0),
        life: Math.random(),
      });
    }
  }

  function stepStreaks(dt, windSpeed, windDirDeg) {
    if (!streaks.length) seedStreaks();
    const b = viewBounds(12);
    const vx = windSpeed * Math.cos(windDirDeg * RAD);
    const vy = windSpeed * Math.sin(windDirDeg * RAD);
    for (const s of streaks) {
      s.x += vx * dt;
      s.y += vy * dt;
      s.life -= dt * 0.28;
      if (s.life <= 0 || s.x < b.x0 || s.x > b.x1 || s.y < b.y0 || s.y > b.y1) {
        s.x = b.x0 + Math.random() * (b.x1 - b.x0);
        s.y = b.y0 + Math.random() * (b.y1 - b.y0);
        s.life = 0.7 + Math.random() * 0.6;
      }
    }
  }

  function drawStreaks(windSpeed, windDirDeg) {
    const len = Math.max(0.9, windSpeed * 0.32);
    const dx = Math.cos(windDirDeg * RAD) * len;
    const dy = Math.sin(windDirDeg * RAD) * len;
    ctx.lineCap = "round";
    ctx.lineWidth = 1.4;
    for (const s of streaks) {
      const a = Math.max(0, Math.min(1, s.life)) * 0.32;
      if (a <= 0.01) continue;
      ctx.strokeStyle = `rgba(126, 232, 192, ${a.toFixed(3)})`;
      ctx.beginPath();
      ctx.moveTo(sx(s.x), sy(s.y));
      ctx.lineTo(sx(s.x + dx), sy(s.y + dy));
      ctx.stroke();
    }
  }

  // --- water and grid ---------------------------------------------------

  function drawWater() {
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "#071427");
    g.addColorStop(0.55, "#0a1c33");
    g.addColorStop(1, "#061020");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  }

  function drawGrid() {
    const b = viewBounds(0);
    // Pick a spacing that stays legible at any zoom.
    const target = 90 / cam.ppm;
    const step = [1, 2, 5, 10, 20, 50, 100].find((s) => s >= target) || 100;

    ctx.lineWidth = 1;
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.textBaseline = "top";

    for (let x = Math.ceil(b.x0 / step) * step; x <= b.x1; x += step) {
      const major = Math.abs(x % (step * 5)) < 1e-6;
      ctx.strokeStyle = major ? "rgba(140,180,230,0.16)" : "rgba(140,180,230,0.07)";
      ctx.beginPath();
      ctx.moveTo(sx(x), 0);
      ctx.lineTo(sx(x), H);
      ctx.stroke();
      if (major) {
        ctx.fillStyle = "rgba(140,180,230,0.34)";
        ctx.fillText(`${x}`, sx(x) + 4, 4);
      }
    }
    for (let y = Math.ceil(b.y0 / step) * step; y <= b.y1; y += step) {
      const major = Math.abs(y % (step * 5)) < 1e-6;
      ctx.strokeStyle = major ? "rgba(140,180,230,0.16)" : "rgba(140,180,230,0.07)";
      ctx.beginPath();
      ctx.moveTo(0, sy(y));
      ctx.lineTo(W, sy(y));
      ctx.stroke();
      if (major) {
        ctx.fillStyle = "rgba(140,180,230,0.34)";
        ctx.fillText(`${y}`, 4, sy(y) + 4);
      }
    }
  }

  function drawBoundary() {
    ctx.strokeStyle = "rgba(255,107,107,0.35)";
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 8]);
    ctx.strokeRect(
      sx(-WORLD_HALF),
      sy(WORLD_HALF),
      2 * WORLD_HALF * cam.ppm,
      2 * WORLD_HALF * cam.ppm,
    );
    ctx.setLineDash([]);
  }

  // --- task ---------------------------------------------------------------

  /**
   * The waypoint task: goal, its success circle, the failure radius the episode
   * ends at, and the straight-line reference the efficiency score is against.
   */
  function drawTask(episode, boat) {
    if (!episode) return;
    const [wx, wy] = episode.waypoint;

    // Failure radius — leave this circle and the episode is over.
    ctx.strokeStyle = "rgba(255,107,107,0.22)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 9]);
    ctx.beginPath();
    ctx.arc(sx(wx), sy(wy), TASK.failRadiusM * cam.ppm, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Straight-line reference from where the episode started.
    ctx.strokeStyle = "rgba(140,180,230,0.22)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.moveTo(sx(episode.start[0]), sy(episode.start[1]));
    ctx.lineTo(sx(wx), sy(wy));
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "rgba(140,180,230,0.5)";
    ctx.beginPath();
    ctx.arc(sx(episode.start[0]), sy(episode.start[1]), 3, 0, Math.PI * 2);
    ctx.fill();

    // Live bearing to the goal.
    ctx.strokeStyle = "rgba(255,216,74,0.32)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 7]);
    ctx.beginPath();
    ctx.moveTo(sx(boat.x), sy(boat.y));
    ctx.lineTo(sx(wx), sy(wy));
    ctx.stroke();
    ctx.setLineDash([]);

    // Success circle, drawn at true scale so the tolerance is honest.
    const done = episode.result === "success";
    const rPix = Math.max(TASK.successRadiusM * cam.ppm, 4);
    ctx.strokeStyle = done ? "rgba(110,242,168,0.9)" : "rgba(255,216,74,0.75)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(sx(wx), sy(wy), rPix, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = done ? "rgba(110,242,168,0.18)" : "rgba(255,216,74,0.14)";
    ctx.fill();

    ctx.fillStyle = done ? "#6ef2a8" : "#ffd84a";
    ctx.beginPath();
    ctx.arc(sx(wx), sy(wy), 3.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.textBaseline = "middle";
    ctx.fillText(
      `waypoint · ${TASK.successRadiusM} m`,
      sx(wx) + rPix + 8,
      sy(wy),
    );
  }

  // --- boat -------------------------------------------------------------

  function drawWake(wake) {
    if (wake.length < 2) return;
    ctx.lineCap = "round";
    for (let i = 1; i < wake.length; i += 1) {
      const a = (i / wake.length) * 0.5;
      ctx.strokeStyle = `rgba(160,220,255,${a.toFixed(3)})`;
      ctx.lineWidth = 1 + (i / wake.length) * 2.2;
      ctx.beginPath();
      ctx.moveTo(sx(wake[i - 1][0]), sy(wake[i - 1][1]));
      ctx.lineTo(sx(wake[i][0]), sy(wake[i][1]));
      ctx.stroke();
    }
  }

  function drawBoat(boat, cfgs, solved, livery) {
    const { x, y, psi } = boat;
    const k = boatScaleFor(Number(cfgs.hull.L), cam.ppm);
    const P = (bx, by) => {
      const [wx, wy] = toWorld(bx * k, by * k, psi, x, y);
      return [sx(wx), sy(wy)];
    };
    const trace = (points) => {
      ctx.beginPath();
      points.forEach(([bx, by], i) => {
        const [px, py] = P(bx, by);
        if (i) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
      });
      ctx.closePath();
    };

    const shape = hullShape(cfgs, livery);
    const rig = rigDimensions(cfgs);
    const scale = cam.ppm * k; // effective px per metre for line weights

    // Underwater shadow: deeper draft reads as a darker, wider smudge.
    ctx.save();
    ctx.globalAlpha = 0.35;
    ctx.fillStyle = "#020814";
    ctx.filter = `blur(${Math.max(1, rig.draft * scale * 0.5).toFixed(1)}px)`;
    trace(shape.outline.map(([bx, by]) => [bx - 0.04 * shape.L, by * 1.08]));
    ctx.fill();
    ctx.restore();

    // Hull, shaded across the beam so it reads as a solid object.
    const [hx0, hy0] = P(0, shape.half);
    const [hx1, hy1] = P(0, -shape.half);
    const grad = ctx.createLinearGradient(hx0, hy0, hx1, hy1);
    grad.addColorStop(0, livery.hullShade);
    grad.addColorStop(0.45, livery.hull);
    grad.addColorStop(1, livery.hullShade);
    trace(shape.outline);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = "rgba(8,16,30,0.85)";
    ctx.lineWidth = Math.max(1, 0.012 * shape.L * scale);
    ctx.stroke();

    // Boot stripe along the sheer, and the cockpit.
    ctx.save();
    ctx.clip();
    ctx.strokeStyle = livery.stripe;
    ctx.lineWidth = Math.max(1.5, 0.035 * shape.L * scale);
    trace(shape.outline);
    ctx.stroke();
    ctx.restore();

    trace(shape.deck);
    ctx.fillStyle = livery.deck;
    ctx.fill();

    // Keel: a fin on the centreline, wider with more planform area.
    ctx.strokeStyle = livery.keel;
    ctx.globalAlpha = 0.65;
    ctx.lineCap = "round";
    ctx.lineWidth = Math.max(2, 0.05 * shape.L * scale);
    ctx.beginPath();
    ctx.moveTo(...P(rig.keelX + rig.keelChord / 2, 0));
    ctx.lineTo(...P(rig.keelX - rig.keelChord / 2, 0));
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Rudder, swung to the servo angle.
    const dr = boat.rudderDeg * RAD;
    ctx.strokeStyle = livery.rudder;
    ctx.lineWidth = Math.max(2, 0.045 * shape.L * scale);
    ctx.beginPath();
    ctx.moveTo(...P(rig.rudderX, 0));
    ctx.lineTo(...P(
      rig.rudderX - rig.rudderChord * Math.cos(dr),
      -rig.rudderChord * Math.sin(dr),
    ));
    ctx.stroke();

    // Sail: a cambered panel off the boom; flat and dashed when it luffs.
    const ds = solved.sailRad;
    const luffing = solved.sail.luffScale < 0.999;
    const clewX = rig.sailX - rig.boom * Math.cos(ds);
    const clewY = -rig.boom * Math.sin(ds);
    const belly = luffing ? 0 : 0.26 * rig.boom * (Math.sign(ds) || -1);
    const [mx, my] = P(rig.sailX, 0);
    const [cxp, cyp] = P(
      (rig.sailX + clewX) / 2 - belly * Math.sin(ds),
      clewY / 2 + belly * Math.cos(ds),
    );
    const [ex, ey] = P(clewX, clewY);

    if (!luffing) {
      ctx.beginPath();
      ctx.moveTo(mx, my);
      ctx.quadraticCurveTo(cxp, cyp, ex, ey);
      ctx.closePath();
      ctx.fillStyle = livery.sail;
      ctx.globalAlpha = 0.72;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    ctx.setLineDash(luffing ? [5, 4] : []);
    ctx.strokeStyle = livery.sailEdge;
    ctx.globalAlpha = luffing ? 0.6 : 1;
    ctx.lineWidth = Math.max(1.8, 0.035 * shape.L * scale);
    ctx.beginPath();
    ctx.moveTo(mx, my);
    ctx.quadraticCurveTo(cxp, cyp, ex, ey);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // Sail mark, once the boat is big enough on screen to carry it.
    if (livery.mark && rig.boom * scale > 34 && !luffing) {
      const [tx, ty] = P(
        (rig.sailX + clewX) / 2 - belly * 0.55 * Math.sin(ds),
        clewY / 2 + belly * 0.55 * Math.cos(ds),
      );
      ctx.save();
      ctx.translate(tx, ty);
      ctx.rotate(-psi - ds + Math.PI / 2);
      ctx.fillStyle = "rgba(20,30,45,0.78)";
      ctx.font = `600 ${Math.min(16, rig.boom * scale * 0.3).toFixed(0)}px ui-monospace, Menlo, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(livery.mark, 0, 0);
      ctx.restore();
      ctx.textAlign = "left";
    }

    // Mast
    ctx.fillStyle = "#f4f9ff";
    ctx.beginPath();
    ctx.arc(mx, my, Math.max(1.8, 0.03 * shape.L * scale), 0, Math.PI * 2);
    ctx.fill();
  }

  /**
   * Smoothed force vectors and arrow scale.
   *
   * The raw per-component forces update every 20 ms and the keel in particular
   * swings hard near stall, so drawing them straight makes the arrows flicker.
   * Both the vectors and the scale they are drawn at are low-pass filtered:
   * the scale rises quickly so a gust is not clipped, and falls slowly so the
   * picture does not breathe.
   */
  const smoothed = {
    hull: [0, 0], keel: [0, 0], sail: [0, 0], rudder: [0, 0], total: [0, 0],
    peak: 1,
  };

  function drawForces(boat, cfgs, solved, dt) {
    const { x, y, psi } = boat;
    const k = boatScaleFor(Number(cfgs.hull.L), cam.ppm);
    const entries = [
      ["hull", solved.hull, cfgs.hull],
      ["keel", solved.keel, cfgs.keel],
      ["sail", solved.sail, cfgs.sail],
      ["rudder", solved.rudder, cfgs.rudder],
      ["total", solved.total, { x_pos: 0, y_pos: 0 }],
    ];

    // Exponential smoothing, frame-rate independent.
    const aVec = 1 - Math.exp(-Math.max(dt, 1e-3) / 0.12);
    let rawPeak = 1;
    for (const [name, f] of entries) {
      const s = smoothed[name];
      s[0] += (f.fx - s[0]) * aVec;
      s[1] += (f.fy - s[1]) * aVec;
      rawPeak = Math.max(rawPeak, Math.hypot(s[0], s[1]));
    }

    // Decay the scale slowly so the picture does not breathe, but never let it
    // lag a rising force: taking the max with rawPeak keeps the longest arrow
    // inside its cap during a spike instead of shooting off the screen.
    const aPeak = 1 - Math.exp(-Math.max(dt, 1e-3) / 1.2);
    smoothed.peak += (rawPeak - smoothed.peak) * aPeak;
    const peak = Math.max(smoothed.peak, rawPeak, 1);
    const pxPerN = (Math.min(W, H) * 0.22) / peak;

    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.textBaseline = "middle";
    ctx.lineCap = "round";

    for (const [name, , cfg] of entries) {
      const [fx, fy] = smoothed[name];
      const mag = Math.hypot(fx, fy);
      if (mag * pxPerN < 6) continue;

      const [ox, oy] = toWorld(
        Number(cfg.x_pos ?? 0) * k,
        Number(cfg.y_pos ?? 0) * k,
        psi, x, y,
      );
      const [dxw, dyw] = rot([fx, fy], psi);
      const x0 = sx(ox);
      const y0 = sy(oy);
      const x1 = x0 + dxw * pxPerN;
      const y1 = y0 - dyw * pxPerN;

      ctx.strokeStyle = COMPONENT_COLORS[name];
      ctx.fillStyle = COMPONENT_COLORS[name];
      ctx.lineWidth = name === "total" ? 2.6 : 1.8;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();

      const ang = Math.atan2(y1 - y0, x1 - x0);
      const head = name === "total" ? 10 : 8;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - head * Math.cos(ang - 0.4), y1 - head * Math.sin(ang - 0.4));
      ctx.lineTo(x1 - head * Math.cos(ang + 0.4), y1 - head * Math.sin(ang + 0.4));
      ctx.closePath();
      ctx.fill();

      // Two significant figures is as much as the eye can use, and it stops
      // the labels from twitching digit by digit.
      const shown = mag >= 100 ? mag.toFixed(0)
        : mag >= 10 ? mag.toFixed(0)
          : mag.toFixed(1);
      ctx.fillText(`${name} ${shown} N`, x1 + 8, y1);
    }

    // Clear of the HUD panels, which sit in the corners.
    ctx.fillStyle = "rgba(140,180,230,0.55)";
    ctx.fillText(
      `force arrows · longest = ${peak < 10 ? peak.toFixed(1) : peak.toFixed(0)} N`,
      262,
      H - 26,
    );
  }

  /** True-wind arrow pinned to the boat, so the breeze is always readable. */
  function drawWindAtBoat(boat, windSpeed, windDirDeg) {
    const reach = Math.max(2.2, Math.min(W, H) * 0.14 / cam.ppm);
    const fromX = boat.x - Math.cos(windDirDeg * RAD) * reach;
    const fromY = boat.y - Math.sin(windDirDeg * RAD) * reach;
    const x0 = sx(fromX);
    const y0 = sy(fromY);
    const x1 = sx(boat.x - Math.cos(windDirDeg * RAD) * reach * 0.35);
    const y1 = sy(boat.y - Math.sin(windDirDeg * RAD) * reach * 0.35);

    ctx.strokeStyle = "rgba(126,232,192,0.85)";
    ctx.fillStyle = "rgba(126,232,192,0.85)";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    ctx.setLineDash([]);

    const ang = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - 11 * Math.cos(ang - 0.4), y1 - 11 * Math.sin(ang - 0.4));
    ctx.lineTo(x1 - 11 * Math.cos(ang + 0.4), y1 - 11 * Math.sin(ang + 0.4));
    ctx.closePath();
    ctx.fill();

    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.fillText(`${windSpeed.toFixed(1)} m/s`, x0 + 8, y0 - 6);
  }

  // --- camera -----------------------------------------------------------

  /**
   * Follow the boat with a dead zone: it can wander across the middle of the
   * screen before the camera moves, which keeps the sense of sailing across a
   * map rather than dragging the map around.
   */
  function follow(boat, dt) {
    const dzX = (W * 0.2) / cam.ppm;
    const dzY = (H * 0.2) / cam.ppm;
    let tx = cam.x;
    let ty = cam.y;
    if (boat.x > cam.x + dzX) tx = boat.x - dzX;
    if (boat.x < cam.x - dzX) tx = boat.x + dzX;
    if (boat.y > cam.y + dzY) ty = boat.y - dzY;
    if (boat.y < cam.y - dzY) ty = boat.y + dzY;

    const k = 1 - Math.exp(-dt * 6);
    cam.x += (tx - cam.x) * k;
    cam.y += (ty - cam.y) * k;

    // Hard stop: however fast the boat moves or the zoom changes, it never
    // leaves the view. The eased follow above is what gives the sense of
    // motion; this only catches the extremes.
    const maxX = (W * 0.36) / cam.ppm;
    const maxY = (H * 0.36) / cam.ppm;
    cam.x = Math.min(Math.max(cam.x, boat.x - maxX), boat.x + maxX);
    cam.y = Math.min(Math.max(cam.y, boat.y - maxY), boat.y + maxY);
  }

  function centerOn(x, y) {
    cam.x = x;
    cam.y = y;
  }

  function setZoom(ppm) {
    cam.ppm = Math.max(4, Math.min(70, ppm));
  }

  /**
   * Canvas CSS pixels -> world metres; the inverse of sx / sy.
   *
   * Named apart from the module-level `toWorld(bx, by, psi, ox, oy)`, which
   * maps body metres to world metres — a local `toWorld` here would shadow it
   * inside every drawing function.
   */
  function screenToWorld(px, py) {
    return [(px - W / 2) / cam.ppm + cam.x, cam.y + (H / 2 - py) / cam.ppm];
  }

  // --- frame ------------------------------------------------------------

  function draw(scene, dt) {
    const { boat, cfgs, solved, episode, wind, wake, options, livery } = scene;

    stepStreaks(dt, wind.speed, wind.dirDeg);

    drawWater();
    drawGrid();
    drawBoundary();
    if (options.showWind) drawStreaks(wind.speed, wind.dirDeg);
    drawTask(episode, boat);
    if (options.showTrail) drawWake(wake);
    if (options.showWind) drawWindAtBoat(boat, wind.speed, wind.dirDeg);
    drawBoat(boat, cfgs, solved, livery);
    if (options.showForces) drawForces(boat, cfgs, solved, dt);
  }

  resize();
  seedStreaks();

  return {
    resize,
    draw,
    follow,
    centerOn,
    setZoom,
    screenToWorld,
    get zoom() { return cam.ppm; },
    get camera() { return cam; },
    seedStreaks,
  };
}

/** Minimap: the whole course at once, with the boat's track. */
export function drawMinimap(canvas, scene) {
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const size = canvas.getBoundingClientRect().width || 160;
  const px = Math.round(size * dpr);
  // A canvas defaults to 300x150, so check both axes — matching only the width
  // leaves the height at 150 and silently stretches everything drawn into it.
  if (canvas.width !== px || canvas.height !== px) {
    canvas.width = px;
    canvas.height = px;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);

  const scale = size / (2 * WORLD_HALF);
  const mx = (wx) => size / 2 + wx * scale;
  const my = (wy) => size / 2 - wy * scale;

  ctx.fillStyle = "rgba(6,16,32,0.85)";
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = "rgba(140,180,230,0.25)";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, size - 1, size - 1);

  // Track
  const { wake, boat, episode, wind } = scene;
  if (wake.length > 1) {
    ctx.strokeStyle = "rgba(94,224,255,0.4)";
    ctx.beginPath();
    wake.forEach(([wx, wy], i) => {
      if (i) ctx.lineTo(mx(wx), my(wy));
      else ctx.moveTo(mx(wx), my(wy));
    });
    ctx.stroke();
  }

  if (episode) {
    ctx.strokeStyle = "rgba(140,180,230,0.3)";
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(mx(episode.start[0]), my(episode.start[1]));
    ctx.lineTo(mx(episode.waypoint[0]), my(episode.waypoint[1]));
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = episode.result === "success" ? "#6ef2a8" : "#ffd84a";
    ctx.beginPath();
    ctx.arc(mx(episode.waypoint[0]), my(episode.waypoint[1]), 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // Boat, as a little arrow pointing where it is heading
  const a = -boat.psi;
  ctx.save();
  ctx.translate(mx(boat.x), my(boat.y));
  ctx.rotate(a);
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.moveTo(6, 0);
  ctx.lineTo(-4, 3.4);
  ctx.lineTo(-4, -3.4);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  // Wind direction, top-left corner
  ctx.strokeStyle = "rgba(126,232,192,0.8)";
  ctx.lineWidth = 1.5;
  const wx0 = 16;
  const wy0 = 16;
  const wdx = Math.cos(wind.dirDeg * RAD) * 9;
  const wdy = -Math.sin(wind.dirDeg * RAD) * 9;
  ctx.beginPath();
  ctx.moveTo(wx0 - wdx, wy0 - wdy);
  ctx.lineTo(wx0 + wdx, wy0 + wdy);
  ctx.stroke();
  ctx.fillStyle = "rgba(126,232,192,0.8)";
  ctx.beginPath();
  ctx.arc(wx0 + wdx, wy0 + wdy, 2.4, 0, Math.PI * 2);
  ctx.fill();
}

/** Wind rose: where the breeze is relative to the bow, plus the no-go zone. */
export function drawWindRose(canvas, scene, noGoDeg = 35) {
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const size = canvas.getBoundingClientRect().width || 150;
  const px = Math.round(size * dpr);
  if (canvas.width !== px || canvas.height !== px) {
    canvas.width = px;
    canvas.height = px;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);

  const c = size / 2;
  const r = size / 2 - 12;
  const { boat, wind } = scene;

  // Bow-up dial. World bearings increase counter-clockwise (psi = atan2(y, x)),
  // so the dial has to as well or it would mirror the main view.
  const dial = (deg) => (-deg - 90) * RAD;

  // No-go wedge
  ctx.fillStyle = "rgba(255,107,107,0.16)";
  ctx.beginPath();
  ctx.moveTo(c, c);
  ctx.arc(c, c, r, dial(noGoDeg), dial(-noGoDeg));
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = "rgba(140,180,230,0.22)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(c, c, r, 0, Math.PI * 2);
  ctx.stroke();

  ctx.font = "9px ui-monospace, Menlo, monospace";
  ctx.fillStyle = "rgba(140,180,230,0.55)";
  ctx.textAlign = "center";
  for (const [deg, label] of [[0, "0"], [90, "90"], [180, "180"], [-90, "90"]]) {
    const a = dial(deg);
    ctx.fillText(label, c + Math.cos(a) * (r + 7), c + Math.sin(a) * (r + 7) + 3);
  }

  // Boat outline at the centre, always bow-up
  ctx.fillStyle = "rgba(226,238,252,0.9)";
  ctx.beginPath();
  ctx.moveTo(c, c - 11);
  ctx.lineTo(c + 5, c + 8);
  ctx.lineTo(c - 5, c + 8);
  ctx.closePath();
  ctx.fill();

  // True wind: the bearing the wind blows FROM, relative to the bow
  const twa = ((((wind.dirDeg + 180 - boat.psi * DEG) % 360) + 540) % 360) - 180;
  const a = dial(twa);
  ctx.strokeStyle = "#7ee8c0";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  ctx.moveTo(c + Math.cos(a) * r, c + Math.sin(a) * r);
  ctx.lineTo(c + Math.cos(a) * 16, c + Math.sin(a) * 16);
  ctx.stroke();
  ctx.fillStyle = "#7ee8c0";
  ctx.beginPath();
  ctx.arc(c + Math.cos(a) * 16, c + Math.sin(a) * 16, 3.2, 0, Math.PI * 2);
  ctx.fill();

  // Apparent wind
  if (scene.awaDeg !== undefined) {
    const aa = dial(scene.awaDeg);
    ctx.strokeStyle = "rgba(94,224,255,0.85)";
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(c + Math.cos(aa) * (r - 6), c + Math.sin(aa) * (r - 6));
    ctx.lineTo(c + Math.cos(aa) * 20, c + Math.sin(aa) * 20);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.textAlign = "left";
}
