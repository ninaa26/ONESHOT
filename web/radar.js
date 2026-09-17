// A circular plan view, drawn in 2-D beside the scene.
//
// The 3-D camera looks straight down at a world where sim +y maps to world +z,
// and viewing that plane from above mirrors one axis: it can be north-up or
// east-right, not both. Here the axes are drawn directly from the simulator's
// own frame, so this is the one view that is north-up AND east-right, and the
// heading on the HUD matches the picture.
//
// The boat sits at the centre and the world moves under it, which is what makes
// a course readable: the trail shows where it has been and the waypoint sits at
// its true bearing, clamped to the rim with an arrowhead when it is out of range.

const TRAIL_MAX_POINTS = 4000;
// A point every quarter metre rather than every frame, so the trail is long in
// metres rather than in seconds and does not grow without bound while moored.
const TRAIL_MIN_STEP_M = 0.25;

const RANGE_MIN_M = 15;
const RANGE_MAX_M = 400;
const RANGE_MARGIN = 1.25;
const RANGE_EASE = 0.08; // per frame, so the ring breathes instead of jumping

const COLORS = {
  face: "rgba(5, 10, 30, 0.72)",
  ring: "rgba(120, 190, 255, 0.30)",
  ringFaint: "rgba(120, 190, 255, 0.16)",
  tick: "rgba(160, 210, 255, 0.55)",
  trail: "rgba(136, 229, 255, 0.85)",
  boat: "#f1f5ff",
  waypoint: "#ffd166",
  text: "rgba(190, 220, 255, 0.8)",
};

export function createRadar(canvas) {
  const ctx = canvas.getContext("2d");
  const trail = [];
  let range = 40;
  let lastWaypoint = null;

  function pushTrail(x, y) {
    const last = trail[trail.length - 1];
    if (last) {
      const dx = x - last.x;
      const dy = y - last.y;
      if (dx * dx + dy * dy < TRAIL_MIN_STEP_M * TRAIL_MIN_STEP_M) return;
    }
    trail.push({ x, y });
    if (trail.length > TRAIL_MAX_POINTS) trail.shift();
  }

  // Range enough to hold the waypoint and the recent trail, eased so it does
  // not snap on every sample.
  function retarget(x, y) {
    let needed = RANGE_MIN_M;
    if (lastWaypoint) {
      needed = Math.max(needed, Math.hypot(lastWaypoint.x - x, lastWaypoint.y - y));
    }
    for (let i = Math.max(0, trail.length - 400); i < trail.length; i++) {
      needed = Math.max(needed, Math.hypot(trail[i].x - x, trail[i].y - y));
    }
    const target = Math.min(RANGE_MAX_M, Math.max(RANGE_MIN_M, needed * RANGE_MARGIN));
    range += (target - range) * RANGE_EASE;
  }

  function draw(x, y, headingRad) {
    const dpr = window.devicePixelRatio || 1;
    const size = canvas.clientWidth;
    if (canvas.width !== size * dpr) {
      canvas.width = size * dpr;
      canvas.height = size * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    let waypointDistance = null;
    const c = size / 2;
    const radius = c - 6;
    // Sim metres to pixels. Screen y is inverted so that +y (north) is up, and
    // screen x is not, so +x (east) is right.
    const scale = radius / range;
    const px = (wx) => c + (wx - x) * scale;
    const py = (wy) => c - (wy - y) * scale;

    ctx.save();
    ctx.beginPath();
    ctx.arc(c, c, radius, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.face;
    ctx.fill();
    ctx.clip();

    // Range rings, at a third and two thirds of the sweep.
    ctx.strokeStyle = COLORS.ringFaint;
    ctx.lineWidth = 1;
    for (const frac of [1 / 3, 2 / 3]) {
      ctx.beginPath();
      ctx.arc(c, c, radius * frac, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Cardinal ticks.
    ctx.strokeStyle = COLORS.tick;
    for (let i = 0; i < 4; i++) {
      const a = (i * Math.PI) / 2;
      ctx.beginPath();
      ctx.moveTo(c + Math.cos(a) * (radius - 7), c + Math.sin(a) * (radius - 7));
      ctx.lineTo(c + Math.cos(a) * radius, c + Math.sin(a) * radius);
      ctx.stroke();
    }

    // Where it has been.
    if (trail.length > 1) {
      ctx.strokeStyle = COLORS.trail;
      ctx.lineWidth = 1.5;
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(px(trail[0].x), py(trail[0].y));
      for (let i = 1; i < trail.length; i++) {
        ctx.lineTo(px(trail[i].x), py(trail[i].y));
      }
      ctx.stroke();
    }

    // The waypoint, at its true bearing, pinned to the rim when out of range.
    if (lastWaypoint) {
      const dx = lastWaypoint.x - x;
      const dy = lastWaypoint.y - y;
      const dist = Math.hypot(dx, dy);
      const bearing = Math.atan2(dy, dx);
      const inRange = dist * scale < radius - 8;
      const r = inRange ? dist * scale : radius - 8;
      const mx = c + Math.cos(bearing) * r;
      const my = c - Math.sin(bearing) * r;

      ctx.fillStyle = COLORS.waypoint;
      ctx.strokeStyle = COLORS.waypoint;
      ctx.lineWidth = 1.5;
      if (inRange) {
        ctx.beginPath();
        ctx.arc(mx, my, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(mx, my, 7.5, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        // An arrowhead on the rim: the direction still matters when the
        // distance no longer fits.
        ctx.save();
        ctx.translate(mx, my);
        ctx.rotate(-bearing);
        ctx.beginPath();
        ctx.moveTo(6, 0);
        ctx.lineTo(-4, 4.5);
        ctx.lineTo(-4, -4.5);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }
      waypointDistance = dist;
    }

    // The boat, bow along its heading. Heading is measured from +x, and screen
    // y is inverted, so the drawn angle is its negation.
    ctx.save();
    ctx.translate(c, c);
    ctx.rotate(-headingRad);
    ctx.fillStyle = COLORS.boat;
    ctx.beginPath();
    ctx.moveTo(9, 0);
    ctx.lineTo(-6, 5.5);
    ctx.lineTo(-3.5, 0);
    ctx.lineTo(-6, -5.5);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    ctx.restore();

    // Rim, north mark and the range this sweep covers.
    ctx.beginPath();
    ctx.arc(c, c, radius, 0, Math.PI * 2);
    ctx.strokeStyle = COLORS.ring;
    ctx.lineWidth = 1;
    ctx.stroke();

    // Both labels sit on the vertical centre line: a corner would be outside
    // the disc, and the element is masked to a circle.
    ctx.fillStyle = COLORS.text;
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.fillText("N", c, 13);
    // Inset from the rim: a disc is only a few pixels wide at its very bottom,
    // so text on the last row is clipped by the mask. Backed, because the trail
    // or a marker may be underneath it.
    const readout =
      waypointDistance === null
        ? `${range.toFixed(0)} m`
        : `${range.toFixed(0)} m · wpt ${waypointDistance.toFixed(0)} m`;
    const readoutY = size - 22;
    const textWidth = ctx.measureText(readout).width;
    ctx.fillStyle = "rgba(5, 10, 30, 0.78)";
    ctx.fillRect(c - textWidth / 2 - 5, readoutY - 9, textWidth + 10, 13);
    ctx.fillStyle = COLORS.text;
    ctx.fillText(readout, c, readoutY);
  }

  return {
    /** Plot the boat, and whatever it is sailing towards. */
    update({ x, y, headingRad, waypoint }) {
      if (!isFinite(x) || !isFinite(y)) return;
      lastWaypoint = waypoint || lastWaypoint;
      pushTrail(x, y);
      retarget(x, y);
      draw(x, y, headingRad || 0);
    },
    /** A new boat starts a new track. */
    reset() {
      trail.length = 0;
      lastWaypoint = null;
      range = 40;
    },
  };
}
