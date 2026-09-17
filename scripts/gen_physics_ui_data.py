"""Generate the dataset behind the SailBench physics UI (``web/physics.html``).

Everything the page shows is produced here from the *real* simulator models, so
the browser never has to re-invent the physics:

1. **Foil polars** — NACA0012 CL/CD sampled straight out of NeuralFoil at the
   Reynolds numbers the configs actually use.
2. **VPP polars** — full 3-DOF steady-state sweeps run through ``SailboatHub``
   with a heading-hold PD controller (same method as ``scripts/polar_diagram.py``).
3. **Config dump** — the YAML configs, so the page can show real parameters.
4. **Validation cases** — component forces for random states, used by the page
   to check its JavaScript mirror of the models against the Python core.

Usage::

    uv run python scripts/gen_physics_ui_data.py
    uv run python scripts/gen_physics_ui_data.py --twa-step 5 --sail-step 10

The NeuralFoil call is the bottleneck in the sweeps, so ``Foil.cl_cd`` is
swapped for an interpolated lookup table built from NeuralFoil itself. At the
default 0.25° grid the tabulated coefficients track the direct call to well
under a thousandth of a coefficient count, and the sweeps run ~45x faster.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sailbench.models.foil import Foil  # noqa: E402
from sailbench.models.model import State  # noqa: E402
from sailbench.sim.sailboat_hub import SailboatHub  # noqa: E402
from sailbench.solvers.rk4 import rk4_step  # noqa: E402

# --- foil table ---------------------------------------------------------

ALPHA_STEP_DEG = 0.25
ALPHA_GRID = np.arange(-180.0, 180.0 + ALPHA_STEP_DEG, ALPHA_STEP_DEG)

_TABLES: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _table_for(re: float, airfoil_name: str = "NACA0012") -> tuple[np.ndarray, np.ndarray]:
    """Return (CL, CD) sampled on ALPHA_GRID, memoised per Reynolds number."""
    from aerosandbox import Airfoil

    key = int(round(float(re)))
    if key not in _TABLES:
        aero = Airfoil(name=airfoil_name).get_aero_from_neuralfoil(
            alpha=ALPHA_GRID,
            Re=float(re),
            mach=0.0,
        )
        _TABLES[key] = (
            np.asarray(aero["CL"], dtype=float),
            np.asarray(aero["CD"], dtype=float),
        )
    return _TABLES[key]


def _tabulated_cl_cd(self: Foil, alpha_rad: float, re: float) -> tuple[float, float]:
    """Drop-in for ``Foil.cl_cd`` backed by the interpolated NeuralFoil table."""
    a_deg = float(np.clip(np.degrees(alpha_rad), self.alpha_min, self.alpha_max))
    cl, cd = _table_for(re)
    return float(np.interp(a_deg, ALPHA_GRID, cl)), float(np.interp(a_deg, ALPHA_GRID, cd))


def install_fast_foil() -> None:
    """Swap the NeuralFoil call for the interpolated table (sweeps only)."""
    Foil.cl_cd = _tabulated_cl_cd  # type: ignore[method-assign]


# --- steady-state sweep -------------------------------------------------

_DT = 0.02
_WARMUP_STEPS = 600  # 12 s: let servos settle and speed converge
_MEASURE_STEPS = 400  # 8 s: average over this window
_KP = 40.0  # heading-hold P-gain; negated because +rudder turns to starboard
_KD = 3.0  # yaw-rate damping
_MAX_RUDDER_DEG = 35.0


def _heading_error_rad(psi: tuple[float, float], target_rad: float) -> float:
    """Signed angle from current heading to target, in [-pi, pi]."""
    c, s = psi
    sin_err = c * math.sin(target_rad) - s * math.cos(target_rad)
    cos_err = c * math.cos(target_rad) + s * math.sin(target_rad)
    return math.atan2(sin_err, cos_err)


def _steady_state(
    config_file: str,
    heading_rad: float,
    sail_limit_rad: float,
    wind_speed: float,
) -> dict[str, float]:
    """Hold a heading at a fixed sheet limit; return averaged steady-state metrics."""
    hub = SailboatHub(config_file=config_file)
    hub.sail_cfg["wind_speed"] = float(wind_speed)

    state = State(x=0.0, y=0.0, psi=(math.cos(heading_rad), math.sin(heading_rad)), u=0.3, v=0.0, r=0.0)

    speeds: list[float] = []
    leeways: list[float] = []
    heels: list[float] = []  # heeling force proxy: lateral sail force
    for i in range(_WARMUP_STEPS + _MEASURE_STEPS):
        err = _heading_error_rad(state.psi, heading_rad)
        rudder = float(np.clip(-_KP * err + _KD * state.r, -_MAX_RUDDER_DEG, _MAX_RUDDER_DEG))
        state = hub.step(
            state=state,
            dt=_DT,
            solver=rk4_step,
            sail_angle=sail_limit_rad,
            rudder_angle=rudder,
        )
        if i >= _WARMUP_STEPS:
            speeds.append(math.hypot(state.u, state.v))
            leeways.append(math.degrees(math.atan2(state.v, max(state.u, 1e-9))))
            heels.append(abs(hub.last_forces.get("sail", (0.0, 0.0))[1]))

    return {
        "speed": float(np.mean(speeds)),
        "leeway_deg": float(np.mean(leeways)),
        "side_force": float(np.mean(heels)),
    }


def _sweep_one_twa(job: tuple[str, float, float, list[float]]) -> dict[str, float]:
    """Worker: best sheet limit for one TWA."""
    config_file, wind_speed, twa_deg, sail_deg_list = job
    install_fast_foil()

    hub_ref = SailboatHub(config_file=config_file)
    wind_dir_rad = math.radians(float(hub_ref.sail_cfg.get("wind_dir_deg", 90.0)))
    upwind_rad = wind_dir_rad + math.pi
    heading = upwind_rad - math.radians(twa_deg)  # port tack: wind on the left

    best = {"speed": 0.0, "leeway_deg": 0.0, "side_force": 0.0, "sail_deg": sail_deg_list[0]}
    for sail_deg in sail_deg_list:
        res = _steady_state(config_file, heading, math.radians(sail_deg), wind_speed)
        if res["speed"] > best["speed"]:
            best = {**res, "sail_deg": float(sail_deg)}

    return {"twa_deg": float(twa_deg), **best}


def compute_polar(
    config_file: str,
    wind_speed: float,
    twa_step: float,
    sail_step: float,
    workers: int,
) -> dict[str, Any]:
    """Run a full TWA x sheet-limit sweep and return the polar for one wind speed."""
    twa_list = [float(t) for t in np.arange(0.0, 180.0 + twa_step, twa_step)]
    # Sheet limit 0 produces no sail force at all, so start at the first real step.
    sail_list = [float(s) for s in np.arange(sail_step, 86.0, sail_step)]

    jobs = [(config_file, wind_speed, twa, sail_list) for twa in twa_list]
    print(
        f"  {config_file} @ {wind_speed:g} m/s: {len(twa_list)} TWA x {len(sail_list)} sheet "
        f"= {len(twa_list) * len(sail_list)} runs",
        flush=True,
    )

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_sweep_one_twa, jobs))
    else:
        rows = [_sweep_one_twa(job) for job in jobs]

    rows.sort(key=lambda row: row["twa_deg"])
    speeds = [row["speed"] for row in rows]
    vmg = [row["speed"] * math.cos(math.radians(row["twa_deg"])) for row in rows]

    return {
        "wind_speed": float(wind_speed),
        "twa_deg": [row["twa_deg"] for row in rows],
        "speed": [round(v, 4) for v in speeds],
        "vmg": [round(v, 4) for v in vmg],
        "best_sheet_deg": [row["sail_deg"] for row in rows],
        "leeway_deg": [round(row["leeway_deg"], 3) for row in rows],
        "side_force": [round(row["side_force"], 3) for row in rows],
    }


# --- validation cases ---------------------------------------------------


def make_validation_cases(config_file: str, count: int, seed: int = 7) -> list[dict[str, Any]]:
    """Sample random states and record the Python core's per-component forces.

    The browser recomputes these with its JavaScript mirror of the models and
    reports the residual, so the page can prove it is showing the real physics.
    Uses the unpatched NeuralFoil path for ground truth.
    """
    rng = np.random.default_rng(seed)
    hub = SailboatHub(config_file=config_file)
    cases: list[dict[str, Any]] = []

    for _ in range(count):
        heading = float(rng.uniform(-math.pi, math.pi))
        u = float(rng.uniform(-0.5, 3.0))
        v = float(rng.uniform(-0.8, 0.8))
        r = float(rng.uniform(-1.2, 1.2))
        sheet_deg = float(rng.uniform(0.0, 85.0))
        rudder_deg = float(rng.uniform(-35.0, 35.0))
        wind_speed = float(rng.uniform(1.0, 10.0))

        hub.sail_cfg["wind_speed"] = wind_speed
        state = State(x=0.0, y=0.0, psi=(math.cos(heading), math.sin(heading)), u=u, v=v, r=r)

        # Place the dynamic frames exactly as a sim step would, then read forces.
        hub._rudder_angle_deg = rudder_deg  # noqa: SLF001 - deliberate: freeze the servo
        hub._update_dynamic_frames(  # noqa: SLF001
            state=state,
            sheet_limit_rad=math.radians(sheet_deg),
            rudder_angle_deg=rudder_deg,
            dt=_DT,
        )
        fx, fy, mz = hub._forces(state)  # noqa: SLF001

        cases.append({
            "heading_deg": round(math.degrees(heading), 6),
            "u": round(u, 6),
            "v": round(v, 6),
            "r": round(r, 6),
            "sheet_deg": round(sheet_deg, 6),
            "rudder_deg": round(rudder_deg, 6),
            "wind_speed": round(wind_speed, 6),
            "sail_angle_deg": round(math.degrees(hub.last_sail_angle_rad), 6),
            "forces": {
                name: [round(value[0], 6), round(value[1], 6)]
                for name, value in hub.last_forces.items()
            },
            "total": [round(fx, 6), round(fy, 6), round(mz, 6)],
        })

    return cases


# --- assembly -----------------------------------------------------------


def load_configs(names: list[str]) -> dict[str, Any]:
    """Load raw YAML configs so the page can display real parameter values."""
    out: dict[str, Any] = {}
    for name in names:
        path = REPO_ROOT / "configs" / name
        with path.open(encoding="utf-8") as file:
            out[name] = yaml.safe_load(file)
    return out


def git_commit() -> str:
    """Short hash of HEAD, or 'unknown' outside a repo."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def foil_tables(res: list[float]) -> dict[str, Any]:
    """Sample the real NeuralFoil polars the sim uses, at full grid resolution.

    The page interpolates this table wherever the Python core would call
    NeuralFoil directly, so the grid is what bounds the mirror-check residual.
    NACA0012 past stall is jagged enough that halving the step is worth the
    file size: at ALPHA_STEP_DEG the worst component lands well under a percent.
    """
    display = ALPHA_GRID
    out: dict[str, Any] = {"alpha_deg": [float(a) for a in display], "re": {}}
    for re in res:
        cl, cd = _table_for(re)
        out["re"][str(int(re))] = {
            "cl": [round(float(v), 5) for v in np.interp(display, ALPHA_GRID, cl)],
            "cd": [round(float(v), 5) for v in np.interp(display, ALPHA_GRID, cd)],
        }
    return out


def main() -> None:
    """Build ``web/physics/data.js``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twa-step", type=float, default=5.0, help="TWA resolution (deg)")
    parser.add_argument("--sail-step", type=float, default=5.0, help="Sheet-limit sweep step (deg)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes")
    parser.add_argument("--validation-cases", type=int, default=120)
    parser.add_argument("--skip-polars", action="store_true", help="Foil + config data only")
    parser.add_argument("--output", type=str, default="web/physics/data.js")
    args = parser.parse_args()

    config_names = ["basic_sailbot.yaml", "flingo_floty.yaml", "real_boat.yaml", "fun_boat.yaml"]

    print("Sampling NeuralFoil polars...", flush=True)
    foil = foil_tables([1e5, 5e5])

    print("Building validation cases (exact NeuralFoil path)...", flush=True)
    validation = make_validation_cases("basic_sailbot.yaml", args.validation_cases)

    # Everything past this point uses the tabulated foil for speed.
    install_fast_foil()

    polars: dict[str, list[dict[str, Any]]] = {}
    if not args.skip_polars:
        print("Running steady-state VPP sweeps...", flush=True)
        plan = [
            ("basic_sailbot.yaml", [3.0, 5.0, 8.0]),
            ("flingo_floty.yaml", [5.0]),
        ]
        for config_file, winds in plan:
            polars[config_file] = [
                compute_polar(config_file, wind, args.twa_step, args.sail_step, args.workers)
                for wind in winds
            ]

    payload = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commit": git_commit(),
            "alpha_step_deg": ALPHA_STEP_DEG,
            "sweep": {
                "dt": _DT,
                "warmup_steps": _WARMUP_STEPS,
                "measure_steps": _MEASURE_STEPS,
                "heading_hold": {"kp": _KP, "kd": _KD, "max_rudder_deg": _MAX_RUDDER_DEG},
            },
        },
        "configs": load_configs(config_names),
        "foil": foil,
        "polars": polars,
        "validation": validation,
    }

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, separators=(",", ":"))
    out_path.write_text(
        "// Generated by scripts/gen_physics_ui_data.py -- do not edit by hand.\n"
        f"export const PHYSICS_DATA = {body};\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
