"""
Generate a polar diagram for the basic_sailbot configuration.

For each True Wind Angle (TWA), sweeps sail trim to find maximum
steady-state boat speed. Prints a polar table and saves a plot to disk.

Usage:
    python scripts/polar_diagram.py
    python scripts/polar_diagram.py --twa-step 2 --sail-step 5
    python scripts/polar_diagram.py --output my_polar.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sailbench.models.model import State
from sailbench.sim.sailboat_hub import SailboatHub
from sailbench.solvers.rk4 import rk4_step

_CONFIG = "basic_sailbot.yaml"
_DT = 0.02
_WARMUP_STEPS = 600   # 12 s: let sail/rudder servos settle and speed converge
_MEASURE_STEPS = 400  # 8 s: average speed over this window
_KP = 40.0            # heading-hold P-gain (deg/rad); negative because positive rudder = starboard turn
_KD = 3.0             # yaw-rate damping term
_MAX_RUDDER_DEG = 35.0
_SPEED_THRESHOLD = 0.10  # m/s: below this, treat as "cannot sail this angle"


def _heading_error_rad(psi: tuple[float, float], target_rad: float) -> float:
    """Signed angle from current heading to target, in [-π, π]."""
    c, s = psi
    sin_err = c * math.sin(target_rad) - s * math.cos(target_rad)
    cos_err = c * math.cos(target_rad) + s * math.sin(target_rad)
    return math.atan2(sin_err, cos_err)


def _steady_speed(
    heading_rad: float,
    sail_limit_rad: float,
    warmup: int = _WARMUP_STEPS,
    measure: int = _MEASURE_STEPS,
) -> float:
    """
    Simulate at fixed heading and sail trim; return mean boat speed at steady state.

    A fresh SailboatHub is created so servo state doesn't bleed between runs.
    The heading is held by a PD controller (negated proportional gain because
    positive rudder angle produces a starboard/clockwise turn in this model).
    """
    hub = SailboatHub(config_file=_CONFIG)
    state = State(
        x=0.0, y=0.0,
        psi=(math.cos(heading_rad), math.sin(heading_rad)),
        u=0.3, v=0.0, r=0.0,
    )
    speeds: list[float] = []
    for i in range(warmup + measure):
        err = _heading_error_rad(state.psi, heading_rad)
        # Negated KP: positive error (need to turn left) → negative rudder (turn left)
        rudder = float(np.clip(-_KP * err + _KD * state.r, -_MAX_RUDDER_DEG, _MAX_RUDDER_DEG))
        state = hub.step(state=state, dt=_DT, solver=rk4_step,
                         sail_angle=sail_limit_rad, rudder_angle=rudder)
        if i >= warmup:
            speeds.append(math.hypot(state.u, state.v))
    return float(np.mean(speeds))


def compute_polar(
    twa_deg: np.ndarray,
    sail_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (speeds, best_sail_deg) arrays indexed by twa_deg.

    For each TWA, sweeps all sail trim angles and keeps the maximum speed.
    Wind direction is read directly from the simulator config.
    """
    hub_ref = SailboatHub(config_file=_CONFIG)
    wind_dir_rad = math.radians(float(hub_ref.sail_cfg.get("wind_dir_deg", 90.0)))
    # upwind_rad: direction toward the wind source (opposite of wind-blows-to)
    upwind_rad = wind_dir_rad + math.pi

    n_twa = len(twa_deg)
    n_sail = len(sail_deg)
    total = n_twa * n_sail
    print(f"  {n_twa} TWA angles × {n_sail} sail angles = {total} simulations")

    speeds = np.zeros(n_twa)
    best_sail = np.zeros(n_twa)
    done = 0

    for i, twa in enumerate(twa_deg):
        heading = upwind_rad - math.radians(twa)  # port-tack convention (wind on left)
        best_spd = 0.0
        best_s = sail_deg[0]

        for s in sail_deg:
            spd = _steady_speed(heading, math.radians(float(s)))
            done += 1
            if spd > best_spd:
                best_spd = spd
                best_s = float(s)
            pct = 100.0 * done / total
            print(
                f"\r  [{pct:5.1f}%]  TWA={twa:5.1f}°  sail={s:5.1f}°  "
                f"spd={spd:.3f} m/s  best={best_spd:.3f} m/s",
                end="", flush=True,
            )

        speeds[i] = best_spd
        best_sail[i] = best_s

    print()
    return speeds, best_sail


def find_no_go_deg(twa_deg: np.ndarray, speeds: np.ndarray) -> float:
    """Return the smallest TWA where steady-state speed exceeds the threshold."""
    for twa, spd in zip(twa_deg, speeds):
        if spd > _SPEED_THRESHOLD:
            return float(twa)
    return float(twa_deg[-1])


def print_table(twa_deg: np.ndarray, speeds: np.ndarray, best_sail: np.ndarray) -> None:
    vmg = speeds * np.cos(np.radians(twa_deg))
    best_idx = int(np.argmax(vmg))
    print(f"\n  {'TWA':>6}  {'Speed':>7}  {'VMG':>7}  {'BestSail':>9}")
    print("  " + "-" * 38)
    for i, (twa, spd, s) in enumerate(zip(twa_deg, speeds, best_sail)):
        tag = "  <-- best VMG" if i == best_idx else ""
        print(f"  {twa:6.1f}  {spd:7.3f}  {vmg[i]:7.3f}  {s:9.1f}{tag}")


def plot_polar(
    twa_deg: np.ndarray,
    speeds: np.ndarray,
    no_go_deg: float,
    output: Path,
) -> None:
    vmg = speeds * np.cos(np.radians(twa_deg))
    best_vmg_idx = int(np.argmax(vmg))
    best_vmg_twa = float(twa_deg[best_vmg_idx])
    best_vmg_spd = float(speeds[best_vmg_idx])
    wind_speed = SailboatHub(config_file=_CONFIG).sail_cfg.get("wind_speed", "?")

    # Mirror for port tack (symmetric hull → same speeds)
    twa_port = twa_deg[::-1][:-1]
    twa_full = np.concatenate([-twa_port, twa_deg])
    spd_full = np.concatenate([speeds[::-1][:-1], speeds])

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        f"Polar Diagram — basic_sailbot.yaml  |  Wind {wind_speed} m/s  |  "
        f"No-go ≈ ±{no_go_deg:.0f}°  |  Best upwind VMG at TWA ≈ {best_vmg_twa:.0f}°",
        fontsize=11,
    )

    # ── Polar (circular) subplot ──────────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1, polar=True)
    ax1.set_theta_zero_location("N")   # 0° (upwind) at top
    ax1.set_theta_direction(-1)        # clockwise, matching nautical convention

    ax1.plot(np.radians(twa_full), spd_full, "b-", lw=2, label="Boat speed")

    # VMG tangent line from origin to optimal upwind point
    ax1.plot(
        [0, math.radians(best_vmg_twa)],
        [0, best_vmg_spd],
        "g--", lw=1.5, alpha=0.8,
        label=f"VMG tangent (TWA={best_vmg_twa:.0f}°)",
    )
    ax1.plot([0, -math.radians(best_vmg_twa)], [0, best_vmg_spd], "g--", lw=1.5, alpha=0.8)

    # No-go zone shading
    r_ceil = float(np.max(spd_full)) * 1.15
    theta_ng = np.linspace(-math.radians(no_go_deg), math.radians(no_go_deg), 80)
    ax1.fill_between(theta_ng, 0, r_ceil, alpha=0.2, color="red",
                     label=f"No-go (±{no_go_deg:.0f}°)")

    ax1.set_title("Polar  (N = upwind)", pad=15)
    ax1.legend(loc="lower right", fontsize=8)

    # ── Cartesian subplot ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(twa_deg, speeds, "b-o", ms=4, lw=2, label="Boat speed (m/s)")
    ax2.plot(twa_deg, vmg, "g--o", ms=4, lw=1.5, label="Upwind VMG (m/s)")
    ax2.axvline(no_go_deg, color="red", ls="--",
                label=f"No-go boundary ≈ {no_go_deg:.0f}°")
    ax2.axvspan(0, no_go_deg, alpha=0.12, color="red")
    ax2.axvline(best_vmg_twa, color="green", ls=":", lw=1.5,
                label=f"Best VMG angle ≈ {best_vmg_twa:.0f}°")
    ax2.set_xlabel("True Wind Angle (°)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_title("Speed and VMG vs. True Wind Angle")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 180)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate sailing polar diagram for basic_sailbot.yaml")
    p.add_argument("--twa-step", type=float, default=5.0, metavar="DEG",
                   help="TWA resolution in degrees (default 5)")
    p.add_argument("--sail-step", type=float, default=10.0, metavar="DEG",
                   help="Sail trim sweep step in degrees (default 10)")
    p.add_argument("--output", type=str, default="scripts/polar_diagram.png",
                   help="Output PNG path (default: scripts/polar_diagram.png)")
    args = p.parse_args()

    twa = np.arange(0.0, 181.0, args.twa_step)
    # Start sail at first non-zero step: sheet_limit=0 returns no sail force,
    # which produces an anomalous residual speed from hull/keel dynamics.
    sail = np.arange(args.sail_step, 86.0, args.sail_step)

    hub_ref = SailboatHub(config_file=_CONFIG)
    wind_speed = hub_ref.sail_cfg.get("wind_speed", "?")
    wind_dir = hub_ref.sail_cfg.get("wind_dir_deg", "?")
    print(f"Config  : {_CONFIG}")
    print(f"Wind    : {wind_speed} m/s toward {wind_dir}°")
    print(f"TWA     : 0° – 180°, step {args.twa_step:.0f}°")
    print(f"Sail    : 0° – 85°, step {args.sail_step:.0f}°")
    print()

    speeds, best_sail = compute_polar(twa, sail)
    no_go = find_no_go_deg(twa, speeds)

    vmg = speeds * np.cos(np.radians(twa))
    best_vmg_twa = float(twa[int(np.argmax(vmg))])

    print_table(twa, speeds, best_sail)
    print(f"\nNo-go zone half-angle : ~{no_go:.0f}°")
    print(f"Best upwind VMG angle : ~{best_vmg_twa:.0f}° TWA")

    plot_polar(twa, speeds, no_go, Path(args.output))


if __name__ == "__main__":
    main()
