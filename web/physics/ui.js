/**
 * DOM helpers shared by the physics-lab sections.
 *
 * Same split as the rest of `web/`: `hud.js` owns readouts, `scene.js` owns
 * scaffolding. Here `ui.js` owns plain-DOM widgets and `charts.js` owns SVG.
 */

/**
 * Create an HTML element.
 * @param {string} tag
 * @param {object} attrs `class`, `text`, `html`, `onclick`-style listeners, or attributes
 * @param {(Node|string|null)[]|Node|string} children
 */
export function h(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2).toLowerCase(), v);
    else node.setAttribute(k, String(v));
  }
  for (const child of [].concat(children)) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
}

/** Fixed-decimal format that degrades to an em dash. */
export const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "—");

/** `value <small>unit</small>` markup for readout rows. */
export const uval = (v, d, unit) => `${fmt(v, d)}<span class="u">${unit}</span>`;

/** Replace a mount point's contents, by id. */
export function mount(id, ...nodes) {
  const host = document.getElementById(id);
  if (!host) return null;
  host.replaceChildren(...nodes);
  return host;
}

/**
 * Labelled range slider.
 * @returns {{node: HTMLElement, set: Function, get: Function, input: HTMLInputElement}}
 */
export function slider({ label, hint, min, max, step, value, unit = "", fmt: f = (v) => v.toFixed(0), onInput }) {
  const valEl = h("span", { class: "ctl-val", text: `${f(value)}${unit}` });
  const input = h("input", {
    type: "range",
    min, max, step, value,
    oninput: (ev) => {
      const v = Number(ev.target.value);
      valEl.textContent = `${f(v)}${unit}`;
      onInput(v);
    },
  });
  const node = h("div", { class: "ctl" }, [
    h("div", { class: "ctl-head" }, [
      h("span", { class: "ctl-label", html: `${label}${hint ? ` <span class="hint">${hint}</span>` : ""}` }),
      valEl,
    ]),
    input,
  ]);
  return {
    node,
    input,
    set(v, silent = true) {
      input.value = String(v);
      valEl.textContent = `${f(Number(v))}${unit}`;
      if (!silent) onInput(Number(v));
    },
    get: () => Number(input.value),
  };
}

/**
 * Key/value readout block; `{sep}` rows render as sub-headings.
 * @returns {{node: HTMLElement, set: Function, flag: Function}}
 */
export function readout(rows) {
  const node = h("div", { class: "readout" });
  const refs = {};
  const rowRefs = {};
  for (const row of rows) {
    if (row.sep) {
      node.append(h("div", { class: "ro-sep", text: row.sep }));
      continue;
    }
    const value = h("span", { class: "ro-v", html: "—" });
    refs[row.key] = value;
    const line = h("div", { class: `ro-row${row.hl ? " hl" : ""}` }, [
      h("span", { class: "ro-k", text: row.label }),
      value,
    ]);
    rowRefs[row.key] = line;
    node.append(line);
  }
  return {
    node,
    set(key, html) {
      if (refs[key]) refs[key].innerHTML = html;
    },
    /** Toggle the accent highlight on a row. */
    flag(key, on) {
      if (rowRefs[key]) rowRefs[key].classList.toggle("hl", Boolean(on));
    },
  };
}

/** Toggle-button group. `values` are `[id, label]` pairs. */
export function chips(values, active, onPick) {
  const buttons = new Map();
  const row = h("div", { class: "btn-row" });
  for (const [id, label] of values) {
    const btn = h("button", {
      class: `chip${id === active ? " on" : ""}`,
      text: label,
      onclick: () => {
        for (const [key, node] of buttons) node.classList.toggle("on", key === id);
        onPick(id);
      },
    });
    buttons.set(id, btn);
    row.append(btn);
  }
  return {
    node: row,
    set(id) {
      for (const [key, node] of buttons) node.classList.toggle("on", key === id);
    },
  };
}

/** Spec tile used by the boat spec sheets. */
export function spec(k, v, d, tone = "") {
  return h("div", { class: `spec ${tone}` }, [
    h("div", { class: "k", text: k }),
    h("div", { class: "v", html: v }),
    d ? h("div", { class: "d", html: d }) : null,
  ]);
}

/** Build a `<table class="data">` from a header row and body rows. */
export function table(el, head, rows) {
  const thead = h("thead", {}, h("tr", {}, head.map((c) => h("th", { html: c }))));
  const tbody = h("tbody", {}, rows.map((r) => h(
    "tr",
    {},
    r.map((c) => (typeof c === "object" && c !== null && !(c instanceof Node)
      ? h("td", { class: c.cls || "", html: c.html })
      : h("td", { html: String(c) }))),
  )));
  el.replaceChildren(thead, tbody);
  return el;
}

/**
 * Minimal syntax highlighter for the Python excerpts.
 *
 * Comments and strings are lifted out into placeholders before the keyword and
 * number passes run, so the markup those passes insert is never re-scanned.
 * A `>>` line prefix marks the line as highlighted.
 */
export function pycode(node, source) {
  if (!node) return node;

  node.innerHTML = source
    .replace(/\t/g, "  ")
    .split("\n")
    .map((raw) => {
      const isHighlighted = raw.startsWith(">>");
      const line = isHighlighted ? ` ${raw.slice(2)}` : raw;

      const held = [];
      const hold = (html) => `\u0000${held.push(html) - 1}\u0000`;

      const escaped = line
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

      const coloured = escaped
        .replace(/#.*$/, (m) => hold(`<span class="cm">${m}</span>`))
        .replace(/"[^"]*"/g, (m) => hold(`<span class="str">${m}</span>`))
        .replace(
          /\b(def|return|if|else|elif|for|in|not|and|or|float|abs|min|max|self)\b/g,
          '<span class="kw">$1</span>',
        )
        .replace(/\b\d+\.?\d*(?:e-?\d+)?\b/g, '<span class="num">$&</span>')
        .replace(/\u0000(\d+)\u0000/g, (_, i) => held[Number(i)]);

      return isHighlighted ? `<span class="hi">${coloured}</span>` : coloured;
    })
    .join("\n");

  return node;
}
