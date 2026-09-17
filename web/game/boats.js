/**
 * Per-boat appearance for the game.
 *
 * Everything that affects *behaviour* comes from `configs/*.yaml` through
 * `boatConfig()` — nothing here changes the physics. This file only decides how
 * each boat is drawn, and the proportions it draws are read back out of the
 * config so a hull always matches its real L / B / T and appendage areas.
 *
 * The colours are a first pass. Swap them for the real boats' paint and sail
 * numbers whenever someone has a photo to hand.
 */

/** Fallback used for any config without an entry. */
const DEFAULT_LIVERY = {
  name: "Sailbot",
  mark: "",
  hull: "#e2eefc",
  hullShade: "#a9c0dc",
  deck: "#1e3452",
  stripe: "rgba(94,224,255,0.9)",
  sail: "#dff4ff",
  sailEdge: "#5ee0ff",
  keel: "#5b8cff",
  rudder: "#ff9459",
};

export const LIVERY = {
  "flingo_floty.yaml": {
    name: "Flingo Floty",
    mark: "FF",
    hull: "#fff3e6",
    hullShade: "#e0a978",
    deck: "#7a3f18",
    stripe: "#ff8a3d",
    sail: "#ffe7cf",
    sailEdge: "#ff9459",
    keel: "#ff8a3d",
    rudder: "#ffb877",
    /** Short and beamy, so give her a blunter bow and a fuller stern. */
    bowFineness: 0.46,
    sternFullness: 0.94,
  },
  "basic_sailbot.yaml": {
    ...DEFAULT_LIVERY,
    name: "Basic Sailbot",
    mark: "SB",
  },
  "real_boat.yaml": {
    ...DEFAULT_LIVERY,
    name: "Real Boat",
    mark: "RB",
    hull: "#dfe9f7",
    deck: "#22324b",
    stripe: "rgba(110,242,168,0.9)",
    sailEdge: "#6ef2a8",
  },
  "fun_boat.yaml": {
    ...DEFAULT_LIVERY,
    name: "Fun Boat",
    mark: "FB",
    hull: "#f7e9ff",
    deck: "#3a2450",
    stripe: "rgba(199,146,234,0.9)",
    sailEdge: "#c792ea",
  },
};

/** Livery for a config name, with defaults filled in. */
export function liveryFor(config) {
  return { ...DEFAULT_LIVERY, ...(LIVERY[config] || {}), name: (LIVERY[config] || {}).name || config.replace(".yaml", "") };
}

/**
 * Plan-view hull outline in body metres, built from the config's real
 * dimensions. A beamier, shorter boat comes out visibly stubbier because the
 * numbers say so, not because of a hand-drawn sprite.
 *
 * @returns {{outline: number[][], deck: number[][], bow: number, stern: number, half: number}}
 */
export function hullShape(cfgs, livery) {
  const L = Number(cfgs.hull.L);
  const B = Number(cfgs.hull.B);
  const half = B / 2;
  const bow = 0.6 * L;
  const stern = -0.4 * L;

  // Slender boats get a finer entry; beamy ones a blunter one.
  const slenderness = L / Math.max(B, 1e-6);
  const fineness = livery.bowFineness ?? Math.min(0.42, 0.18 + slenderness * 0.07);
  const fullness = livery.sternFullness ?? 0.82;

  const outline = [
    [bow, 0],
    [bow - fineness * L, half * 0.72],
    [0.18 * L, half],
    [stern + 0.1 * L, half * fullness],
    [stern, half * fullness * 0.88],
    [stern, -half * fullness * 0.88],
    [stern + 0.1 * L, -half * fullness],
    [0.18 * L, -half],
    [bow - fineness * L, -half * 0.72],
  ];

  const deck = [
    [0.22 * L, half * 0.44],
    [stern + 0.06 * L, half * 0.52],
    [stern + 0.06 * L, -half * 0.52],
    [0.22 * L, -half * 0.44],
  ];

  return { outline, deck, bow, stern, half, L, B };
}

/** Appendage sizes in body metres, from the config's planform areas. */
export function rigDimensions(cfgs) {
  const L = Number(cfgs.hull.L);
  const sailX = Number(cfgs.sail.x_pos ?? 0);
  return {
    keelChord: Math.min(Math.sqrt(Number(cfgs.keel.area)) * 0.8, 0.5 * L),
    keelX: Number(cfgs.keel.x_pos ?? 0),
    rudderChord: Math.min(Math.sqrt(Number(cfgs.rudder.area)) * 1.1, 0.32 * L),
    rudderX: Number(cfgs.rudder.x_pos ?? 0),
    boom: Math.min(Math.sqrt(Number(cfgs.sail.area)) * 0.62, sailX + 0.52 * L),
    sailX,
    draft: Number(cfgs.hull.T),
  };
}
