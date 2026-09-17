/**
 * sailbench — Shipyard.
 *
 * The same sim as `web/sim/`, plus the boat-assembly screen in front of it:
 * the backend offers a catalog, the shipyard collects a build, and the sim
 * only starts sailing once one is chosen. Esc returns here.
 *
 * Requires a backend with `sailbench.sim.shipyard` (it must send `catalog`).
 */
import { createSimApp } from "../shared/sim-app.js";
import { createShipyard } from "./shipyard.js";

createSimApp({ createShipyard });
