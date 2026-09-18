/**
 * A shipyard that isn't there.
 *
 * `sim-app.js` calls the shipyard on every connection event and keypress so it
 * doesn't have to branch on whether the boat-assembly UI is present. Apps that
 * ship without that UI pass this instead: it reports itself permanently closed,
 * so every guard in the sim (`if (shipyard.isOpen())`, `handleKey`, …) falls
 * through to the plain sailing path.
 *
 * Reporting `hasCatalog() === false` while never opening also means the sim
 * never enters the "waiting for a build" state, so it sails whatever the
 * backend was launched with — which is exactly the old pre-shipyard behaviour.
 */
export function createNullShipyard() {
  return {
    isOpen: () => false,
    hasCatalog: () => false,
    wantsCatalog: () => false,
    show: () => {},
    hide: () => {},
    setStatus: () => {},
    showError: () => {},
    setCatalog: () => {},
    handleKey: () => false,
    /** @param {{ boat?: string }} build */
    describe: (build) => (build && build.boat) || "—",
    /**
     * No catalog means no measured geometry to hand the renderer, so the boat
     * draws at `boat.js`'s fallback size. Present so the sim can ask without
     * caring which shipyard it got.
     */
    geometryFor: () => null,
  };
}
