/**
 * sailbench — the plain sailing simulation.
 *
 * The original sim: connect, and sail whatever boat the backend was launched
 * with (`--config`). No boat picker, so the null shipyard stands in for one.
 */
import { createSimApp } from "../shared/sim-app.js";
import { createNullShipyard } from "../shared/null-shipyard.js";

createSimApp({ createShipyard: () => createNullShipyard() });
