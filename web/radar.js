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

// Wind streaks. The count is fixed and the length carries the strength, so a
// glance gives direction from the angle and strength from how far they reach.
const WIND_STREAKS = 5;
const WIND_LEN_MIN = 0.16; // of the radius, at a dead calm
const WIND_LEN_MAX = 0.62; // of the radius, at WIND_FULL_SCALE_MS and above
const WIND_FULL_SCALE_MS = 14.0;

// Points of sail, as angles off the wind. The boundaries are the usual ones: a
// boat cannot make ground inside roughly 40 degrees of the wind, and everything
// wider than that is named by how far off it is bearing away.
const POINTS_OF_SAIL = [
  { to: 40, label: "IN IRONS", short: "NO-GO" },
  { to: 60, label: "CLOSE HAULED", short: "CLOSE" },
  { to: 100, label: "BEAM REACH", short: "BEAM" },
  { to: 150, label: "BROAD REACH", short: "BROAD" },
  { to: 180, label: "RUNNING", short: "RUN" },
];

/** Angle between two bearings in degrees, 0..180. */
function angleBetween(a, b) {
  return Math.abs((((a - b) % 360) + 540) % 360 - 180);
}

/** Which point of sail a heading is on, given where the wind blows to. */
export function pointOfSail(headingDeg, windToDeg) {
  // The wind blows *to* windToDeg, so it comes from the opposite bearing, and
  // that is what a heading is measured against.
  const off = angleBetween(headingDeg, windToDeg + 180);
  const band = POINTS_OF_SAIL.find((b) => off <= b.to) || POINTS_OF_SAIL[POINTS_OF_SAIL.length - 1];
  return { label: band.label, offWindDeg: off };
}

const COLORS = {
  face: "rgba(5, 10, 30, 0.72)",
  ring: "rgba(120, 190, 255, 0.30)",
  ringFaint: "rgba(120, 190, 255, 0.16)",
  tick: "rgba(160, 210, 255, 0.55)",
  trail: "rgba(136, 229, 255, 0.85)",
  wind: "rgba(150, 195, 240, 0.34)",
  noGo: "rgba(255, 120, 110, 0.13)",
  noGoEdge: "rgba(255, 140, 130, 0.38)",
  sector: "rgba(120, 190, 255, 0.14)",
  sectorText: "rgba(175, 205, 240, 0.62)",
  boat: "#f1f5ff",
  waypoint: "#ffd166",
  text: "rgba(190, 220, 255, 0.8)",
};

export function createRadar(canvas) {
  // A missing canvas must not take the simulator down with it. This is a
  // read-out, not a control: without it the boat still sails, and an element
  // that has gone missing -- a stale page against fresh scripts, say -- should
  // cost the radar, not the whole app.
  const ctx = canvas ? canvas.getContext("2d") : null;
  if (!ctx) {
    return { update() {}, reset() {} };
  }
  const trail = [];
  let range = 40;
  let lastWaypoint = null;
  let lastWind = null;
  let lastHeadingRad = 0;

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
    lastHeadingRad = headingRad;
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

    // Points of sail, drawn about the bearing the wind comes from, so the
    // wedges turn with the weather and the boat turns within them.
    if (lastWind && lastWind.speed > 0.05) {
      const fromRad = ((lastWind.dirDeg + 180) * Math.PI) / 180;
      // Screen angles run clockwise from +x, bearings counter-clockwise.
      const screenOf = (bearingRad) => -bearingRad;
      let previous = 0;
      for (const band of POINTS_OF_SAIL) {
        const inner = (previous * Math.PI) / 180;
        const outer = (band.to * Math.PI) / 180;
        if (band.to === 40) {
          // The no-go zone is the one worth colouring: it is the only band you
          // cannot sail in.
          ctx.beginPath();
          ctx.moveTo(c, c);
          ctx.arc(c, c, radius, screenOf(fromRad + outer), screenOf(fromRad - outer));
          ctx.closePath();
          ctx.fillStyle = COLORS.noGo;
          ctx.fill();
        }
        // Boundaries either side, so the bands read as a fan around the wind.
        ctx.strokeStyle = band.to === 40 ? COLORS.noGoEdge : COLORS.sector;
        ctx.lineWidth = 1;
        for (const sign of [1, -1]) {
          const a = screenOf(fromRad + sign * outer);
          ctx.beginPath();
          ctx.moveTo(c, c);
          ctx.lineTo(c + Math.cos(a) * radius, c + Math.sin(a) * radius);
          ctx.stroke();
        }
        // One label per band, on the starboard side, at the band's middle.
        const mid = ((previous + band.to) / 2) * (Math.PI / 180);
        const la = screenOf(fromRad + mid);
        ctx.fillStyle = COLORS.sectorText;
        ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.textAlign = "center";
        // Positioned by its own width, not just its centre: a label centred
        // near the rim still has half of itself outside the disc.
        const halfText = ctx.measureText(band.short).width / 2 + 4;
        const labelR = Math.max(0, radius * 0.86 - halfText);
        ctx.fillText(
          band.short,
          c + Math.cos(la) * labelR,
          c + Math.sin(la) * labelR + 3,
        );
        previous = band.to;
      }
    }

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

    // Which way the wind is going, and how hard. Drawn beneath the track so it
    // reads as weather rather than as something the boat did.
    if (lastWind && lastWind.speed > 0.05) {
      const dir = (lastWind.dirDeg * Math.PI) / 180;
      // Screen y grows downwards, so the bearing's sine is negated.
      const ux = Math.cos(dir);
      const uy = -Math.sin(dir);
      const strength = Math.min(1, lastWind.speed / WIND_FULL_SCALE_MS);
      const half = (radius * (WIND_LEN_MIN + (WIND_LEN_MAX - WIND_LEN_MIN) * strength)) / 2;
      const spacing = (radius * 1.5) / (WIND_STREAKS - 1);

      ctx.strokeStyle = COLORS.wind;
      ctx.fillStyle = COLORS.wind;
      ctx.lineWidth = 1.25;
      for (let i = 0; i < WIND_STREAKS; i++) {
        // Offsets step across the disc, perpendicular to the wind.
        const off = (i - (WIND_STREAKS - 1) / 2) * spacing;
        const ox = c + -uy * off;
        const oy = c + ux * off;
        const x1 = ox - ux * half;
        const y1 = oy - uy * half;
        const x2 = ox + ux * half;
        const y2 = oy + uy * half;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        // A head at the downwind end: the line alone is an axis, not a direction.
        ctx.save();
        ctx.translate(x2, y2);
        ctx.rotate(Math.atan2(uy, ux));
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(-4.5, 3);
        ctx.lineTo(-4.5, -3);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }
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
    // Stacked on the vertical centre line at fractions of the radius rather
    // than at fixed insets from the bottom: the disc narrows towards its edge,
    // so a line placed by pixels from the rim gets clipped once it grows.
    const lines = [];
    if (lastWind) {
      const pos = pointOfSail((lastHeadingRad * 180) / Math.PI, lastWind.dirDeg);
      lines.push(`${pos.label} ${pos.offWindDeg.toFixed(0)}°`);
      lines.push(`wind ${lastWind.speed.toFixed(1)} m/s`);
    }
    lines.push(readout);

    lines.forEach((line, i) => {
      const y = c + radius * (0.52 + i * 0.13);
      const textWidth = ctx.measureText(line).width;
      ctx.fillStyle = "rgba(5, 10, 30, 0.78)";
      ctx.fillRect(c - textWidth / 2 - 5, y - 9, textWidth + 10, 13);
      ctx.fillStyle = COLORS.text;
      ctx.fillText(line, c, y);
    });
  }

  return {
    /** Plot the boat, and whatever it is sailing towards. */
    update({ x, y, headingRad, waypoint, wind }) {
      if (!isFinite(x) || !isFinite(y)) return;
      lastWaypoint = waypoint || lastWaypoint;
      lastWind = wind || lastWind;
      pushTrail(x, y);
      retarget(x, y);
      draw(x, y, headingRad || 0);
    },
    /** A new boat starts a new track. */
    reset() {
      trail.length = 0;
      lastWaypoint = null;
      lastWind = null;
      range = 40;
    },
  };
}
