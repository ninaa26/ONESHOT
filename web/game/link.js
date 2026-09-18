/**
 * Link to the Python backend.
 *
 * Same endpoint the 3D viewer uses (`ws://<host>:8765/sim`, see `web/shared/sim-app.js`).
 * Start the backend with a checkpoint to watch a trained policy sail in this
 * bench:
 *
 *   uv run python -m sailbench.sim.web_runner \
 *     --config basic_sailbot.yaml \
 *     --policy-model runs/<run>/best_model/best_model.zip \
 *     --policy-config configs/rl_waypoint_sb3.yaml
 *
 * In backend mode Python owns the physics and this page is a viewer; the bench
 * still scores the run, so a policy and the in-browser controllers are measured
 * the same way.
 */

const PORT = 8765;

export function createLink({ onState, onStatus }) {
  let socket = null;
  let wantOpen = false;
  let retryMs = 1000;

  const url = () => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const host = window.location.hostname || "127.0.0.1";
    return `${proto}://${host}:${PORT}/sim`;
  };

  function connect() {
    if (socket) return;
    wantOpen = true;
    onStatus("connecting", url());
    try {
      socket = new WebSocket(url());
    } catch {
      socket = null;
      onStatus("error", "could not open a socket");
      return;
    }

    socket.onopen = () => {
      retryMs = 1000;
      onStatus("connected", url());
    };
    socket.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg && msg.type === "state") onState(msg);
      } catch {
        // Ignore malformed frames, as web/shared/sim-app.js does.
      }
    };
    socket.onerror = () => onStatus("error", `no backend on :${PORT}`);
    socket.onclose = () => {
      socket = null;
      onStatus("disconnected", "");
      if (wantOpen) {
        window.setTimeout(() => { if (wantOpen) connect(); }, retryMs);
        retryMs = Math.min(retryMs * 2, 8000);
      }
    };
  }

  function disconnect() {
    wantOpen = false;
    if (socket) socket.close();
  }

  /** Send a helm command; the backend ignores these while a policy is driving. */
  function send(rudderDeg, sailDeg) {
    if (!socket || socket.readyState !== WebSocket.OPEN) return false;
    socket.send(JSON.stringify({
      type: "control",
      rudder_deg: rudderDeg,
      sail_deg: sailDeg,
    }));
    return true;
  }

  return {
    connect,
    disconnect,
    send,
    get open() { return Boolean(socket && socket.readyState === WebSocket.OPEN); },
    get url() { return url(); },
  };
}
