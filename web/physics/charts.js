/**
 * SVG chart primitives for the physics lab (plain-DOM widgets live in ui.js).
 *
 * Deliberately dependency-free: every figure on the page is a handful of paths
 * whose `d` attribute is rewritten on each interaction, which keeps slider
 * dragging smooth without pulling in a plotting library.
 */

const NS = "http://www.w3.org/2000/svg";

/** Create an SVG element with attributes (and optional text). */
export function el(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    node.setAttribute(k, String(v));
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Nice-ish tick values across [lo, hi]. */
export function ticks(lo, hi, count = 5) {
  const span = hi - lo;
  if (span <= 0) return [lo];
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-6; t += step) {
    out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  }
  return out;
}

/**
 * Cartesian line chart.
 *
 * @returns {{svg: SVGElement, setSeries: Function, setYDomain: Function,
 *            setBands: Function, setMarkers: Function, x: Function, y: Function}}
 */
export function lineChart(opts) {
  const {
    width = 520,
    height = 260,
    pad = { l: 46, r: 14, t: 12, b: 30 },
    xDomain = [0, 1],
    yDomain = [0, 1],
    xLabel = "",
    yLabel = "",
    xTicks = 6,
    yTicks = 5,
    xFmt = (v) => String(Math.round(v * 100) / 100),
    yFmt = (v) => String(Math.round(v * 1000) / 1000),
    zeroLine = true,
  } = opts;

  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  const iw = width - pad.l - pad.r;
  const ih = height - pad.t - pad.b;

  let yd = [...yDomain];
  const xd = [...xDomain];

  const x = (v) => pad.l + ((v - xd[0]) / (xd[1] - xd[0])) * iw;
  const y = (v) => pad.t + ih - ((v - yd[0]) / (yd[1] - yd[0])) * ih;

  const gBands = el("g");
  const gGrid = el("g");
  const gSeries = el("g");
  const gMarkers = el("g");
  svg.append(gBands, gGrid, gSeries, gMarkers);

  function drawGrid() {
    gGrid.replaceChildren();
    for (const t of ticks(xd[0], xd[1], xTicks)) {
      gGrid.append(el("line", {
        x1: x(t), x2: x(t), y1: pad.t, y2: pad.t + ih,
        stroke: "rgba(140,180,230,0.10)", "stroke-width": 1,
      }));
      gGrid.append(el("text", {
        x: x(t), y: height - pad.b + 15, "text-anchor": "middle",
        fill: "#5c6b81", "font-size": 9.5, "font-family": "var(--mono)",
      }, xFmt(t)));
    }
    for (const t of ticks(yd[0], yd[1], yTicks)) {
      const isZero = Math.abs(t) < 1e-9;
      gGrid.append(el("line", {
        x1: pad.l, x2: pad.l + iw, y1: y(t), y2: y(t),
        stroke: isZero && zeroLine ? "rgba(140,180,230,0.28)" : "rgba(140,180,230,0.10)",
        "stroke-width": 1,
      }));
      gGrid.append(el("text", {
        x: pad.l - 7, y: y(t) + 3, "text-anchor": "end",
        fill: "#5c6b81", "font-size": 9.5, "font-family": "var(--mono)",
      }, yFmt(t)));
    }
    if (xLabel) {
      gGrid.append(el("text", {
        x: pad.l + iw / 2, y: height - 2, "text-anchor": "middle",
        fill: "#5c6b81", "font-size": 10, "font-family": "var(--mono)",
      }, xLabel));
    }
    if (yLabel) {
      gGrid.append(el("text", {
        x: 10, y: pad.t + ih / 2, "text-anchor": "middle",
        transform: `rotate(-90 10 ${pad.t + ih / 2})`,
        fill: "#5c6b81", "font-size": 10, "font-family": "var(--mono)",
      }, yLabel));
    }
  }

  function path(points) {
    let d = "";
    let pen = false;
    for (const p of points) {
      if (!p || !Number.isFinite(p[1])) { pen = false; continue; }
      const px = x(p[0]);
      const py = y(Math.max(Math.min(p[1], yd[1] + (yd[1] - yd[0])), yd[0] - (yd[1] - yd[0])));
      d += `${pen ? "L" : "M"}${px.toFixed(2)} ${py.toFixed(2)}`;
      pen = true;
    }
    return d;
  }

  function setSeries(series) {
    gSeries.replaceChildren();
    for (const s of series) {
      if (!s) continue;
      if (s.fillTo !== undefined && s.points.length) {
        const base = y(s.fillTo);
        const d = `${path(s.points)}L${x(s.points[s.points.length - 1][0]).toFixed(2)} ${base.toFixed(2)}L${x(s.points[0][0]).toFixed(2)} ${base.toFixed(2)}Z`;
        gSeries.append(el("path", { d, fill: s.color, "fill-opacity": s.fillOpacity ?? 0.1, stroke: "none" }));
      }
      gSeries.append(el("path", {
        d: path(s.points),
        fill: "none",
        stroke: s.color,
        "stroke-width": s.width ?? 1.8,
        "stroke-dasharray": s.dash,
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
        opacity: s.opacity ?? 1,
      }));
      for (const p of s.dots || []) {
        gSeries.append(el("circle", { cx: x(p[0]), cy: y(p[1]), r: s.dotR ?? 2.4, fill: s.color }));
      }
    }
  }

  function setBands(bands) {
    gBands.replaceChildren();
    for (const b of bands || []) {
      if (b.x !== undefined) {
        const x0 = x(Math.max(b.x[0], xd[0]));
        const x1 = x(Math.min(b.x[1], xd[1]));
        gBands.append(el("rect", {
          x: Math.min(x0, x1), y: pad.t, width: Math.abs(x1 - x0), height: ih,
          fill: b.color, opacity: b.opacity ?? 0.1,
        }));
        if (b.label) {
          gBands.append(el("text", {
            x: (x0 + x1) / 2, y: pad.t + 12, "text-anchor": "middle",
            fill: b.color, "font-size": 9.5, "font-family": "var(--mono)", opacity: 0.85,
          }, b.label));
        }
      }
    }
  }

  function setMarkers(markers) {
    gMarkers.replaceChildren();
    for (const m of markers || []) {
      if (m.type === "vline") {
        gMarkers.append(el("line", {
          x1: x(m.x), x2: x(m.x), y1: pad.t, y2: pad.t + ih,
          stroke: m.color, "stroke-width": 1.2, "stroke-dasharray": m.dash ?? "3 3", opacity: 0.85,
        }));
        if (m.label) {
          gMarkers.append(el("text", {
            x: x(m.x) + 4, y: pad.t + 11, fill: m.color,
            "font-size": 9.5, "font-family": "var(--mono)",
          }, m.label));
        }
      } else if (m.type === "dot") {
        gMarkers.append(el("circle", {
          cx: x(m.x), cy: y(m.y), r: m.r ?? 4,
          fill: m.color, stroke: "#05070c", "stroke-width": 1.5,
        }));
        if (m.label) {
          gMarkers.append(el("text", {
            x: x(m.x) + 8, y: y(m.y) - 6, fill: m.color,
            "font-size": 10, "font-family": "var(--mono)",
          }, m.label));
        }
      } else if (m.type === "text") {
        gMarkers.append(el("text", {
          x: x(m.x), y: y(m.y), fill: m.color, "text-anchor": m.anchor ?? "start",
          "font-size": m.size ?? 10, "font-family": "var(--mono)",
        }, m.label));
      }
    }
  }

  function setYDomain(next) {
    yd = [...next];
    drawGrid();
  }

  drawGrid();
  return { svg, setSeries, setBands, setMarkers, setYDomain, x, y, pad, iw, ih, width, height };
}

/**
 * Nautical polar plot: 0 deg TWA at the top, clockwise, mirrored to both tacks.
 */
export function polarChart(opts) {
  const { size = 420, rMax = 2, rings = 4, label = "m/s" } = opts;
  const svg = el("svg", { viewBox: `0 0 ${size} ${size}` });
  const cx = size / 2;
  const cy = size / 2 + 6;
  const rPix = size / 2 - 34;

  let max = rMax;
  const gGrid = el("g");
  const gData = el("g");
  svg.append(gGrid, gData);

  /** TWA (deg, 0 = dead upwind) + radius -> screen point. */
  function pt(twaDeg, r) {
    const a = (twaDeg - 90) * (Math.PI / 180); // 0 deg at top, clockwise
    const rr = (r / max) * rPix;
    return [cx + rr * Math.cos(a), cy + rr * Math.sin(a)];
  }

  function drawGrid() {
    gGrid.replaceChildren();
    for (let i = 1; i <= rings; i += 1) {
      const r = (max * i) / rings;
      gGrid.append(el("circle", {
        cx, cy, r: (r / max) * rPix, fill: "none",
        stroke: "rgba(140,180,230,0.12)", "stroke-width": 1,
      }));
      gGrid.append(el("text", {
        x: cx + 4, y: cy - (r / max) * rPix - 3,
        fill: "#5c6b81", "font-size": 9, "font-family": "var(--mono)",
      }, i === rings ? `${r.toFixed(1)} ${label}` : r.toFixed(1)));
    }
    for (let a = 0; a < 360; a += 30) {
      const [px, py] = pt(a, max);
      gGrid.append(el("line", {
        x1: cx, y1: cy, x2: px, y2: py,
        stroke: "rgba(140,180,230,0.10)", "stroke-width": 1,
      }));
      const [lx, ly] = pt(a, max * 1.11);
      gGrid.append(el("text", {
        x: lx, y: ly + 3, "text-anchor": "middle",
        fill: "#5c6b81", "font-size": 9, "font-family": "var(--mono)",
      }, `${a > 180 ? 360 - a : a}°`));
    }
  }

  function setMax(next) {
    max = next;
    drawGrid();
  }

  function clear() {
    gData.replaceChildren();
  }

  /** Draw one polar curve, mirrored across the wind axis. */
  function addCurve(twa, radii, { color, width = 2, dash, fill = false, opacity = 1, mirror = true }) {
    const pts = twa.map((t, i) => pt(t, radii[i]));
    let d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join("");
    if (mirror) {
      const mir = twa.map((t, i) => pt(-t, radii[i])).reverse();
      d += mir.map((p) => `L${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join("");
    }
    if (fill) {
      gData.append(el("path", { d: `${d}Z`, fill: color, "fill-opacity": 0.08, stroke: "none" }));
    }
    gData.append(el("path", {
      d, fill: "none", stroke: color, "stroke-width": width,
      "stroke-dasharray": dash, "stroke-linejoin": "round", opacity,
    }));
  }

  function addRay(twaDeg, r, { color, dash = "4 4", width = 1.3, mirror = true, label: text }) {
    for (const sign of mirror ? [1, -1] : [1]) {
      const [px, py] = pt(twaDeg * sign, r);
      gData.append(el("line", {
        x1: cx, y1: cy, x2: px, y2: py, stroke: color,
        "stroke-width": width, "stroke-dasharray": dash, opacity: 0.9,
      }));
    }
    if (text) {
      const [px, py] = pt(twaDeg, r * 1.04);
      gData.append(el("text", {
        x: px + 6, y: py, fill: color, "font-size": 9.5, "font-family": "var(--mono)",
      }, text));
    }
  }

  function addWedge(halfAngle, { color, label: text }) {
    const steps = 24;
    let d = `M${cx} ${cy}`;
    for (let i = 0; i <= steps; i += 1) {
      const a = -halfAngle + (2 * halfAngle * i) / steps;
      const [px, py] = pt(a, max);
      d += `L${px.toFixed(1)} ${py.toFixed(1)}`;
    }
    gData.append(el("path", { d: `${d}Z`, fill: color, "fill-opacity": 0.12, stroke: "none" }));
    if (text) {
      gData.append(el("text", {
        x: cx, y: cy - rPix * 0.55, "text-anchor": "middle",
        fill: color, "font-size": 10, "font-family": "var(--mono)", opacity: 0.9,
      }, text));
    }
  }

  function addDot(twaDeg, r, { color, label: text }) {
    const [px, py] = pt(twaDeg, r);
    gData.append(el("circle", { cx: px, cy: py, r: 4.5, fill: color, stroke: "#05070c", "stroke-width": 1.5 }));
    if (text) {
      gData.append(el("text", {
        x: px + 9, y: py + 3, fill: color, "font-size": 10, "font-family": "var(--mono)",
      }, text));
    }
  }

  /** Wind arrow at the top, blowing down the page (toward the boat). */
  function addWindGlyph() {
    const top = cy - rPix - 20;
    gData.append(el("path", {
      d: `M${cx} ${top}L${cx} ${top + 16}M${cx - 4} ${top + 10}L${cx} ${top + 16}L${cx + 4} ${top + 10}`,
      stroke: "var(--c-wind)", "stroke-width": 1.6, fill: "none", "stroke-linecap": "round",
    }));
    gData.append(el("text", {
      x: cx + 10, y: top + 11, fill: "var(--c-wind)",
      "font-size": 9.5, "font-family": "var(--mono)",
    }, "TRUE WIND"));
  }

  drawGrid();
  return { svg, pt, setMax, clear, addCurve, addRay, addWedge, addDot, addWindGlyph, cx, cy, rPix };
}

/** Arrow path helper used by the plan-view figures (screen coords). */
export function arrowPath(x0, y0, x1, y1, head = 7) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) return "";
  const ux = dx / len;
  const uy = dy / len;
  const hx = x1 - ux * head;
  const hy = y1 - uy * head;
  const px = -uy * head * 0.45;
  const py = ux * head * 0.45;
  return `M${x0.toFixed(1)} ${y0.toFixed(1)}L${hx.toFixed(1)} ${hy.toFixed(1)}` +
    `M${(hx + px).toFixed(1)} ${(hy + py).toFixed(1)}L${x1.toFixed(1)} ${y1.toFixed(1)}` +
    `L${(hx - px).toFixed(1)} ${(hy - py).toFixed(1)}`;
}
