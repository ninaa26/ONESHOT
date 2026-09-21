"""Score the trained policies under `runs/` so the shipyard can name them.

A directory listing tells you a run exists. It does not tell you what the run
tried or whether the policy it produced can sail, and `waypoint_ppo_<timestamp>`
tells you neither. This writes `scorecard.json` next to each checkpoint with the
numbers that actually separate these policies.

Which numbers: not success rate. Every Flingo run on disk reaches the mark
essentially every time, so success rate ranks nothing -- see `runs/README.md`.
What separates them is how fast they sail and how much of the episode they spend
pinching inside 25 degrees of the true wind, which is the habit the simulator
rewards and the water will not (the note on `no_go_zone_penalty` in
`configs/flingo_rl.yaml` has the argument). `sheet_pinned_share` is the third:
100% means the policy gave up on trimming and steers on the rudder alone.

Every policy is scored on the same episodes: the seeds are fixed here, and the
one task knob that changes what a seed produces -- `upwind_waypoint_bias`, which
some runs trained at 1.0 and some at 0.5 -- is overridden to a common value.
Everything else, the observation scales especially, comes from the run's own
`config_used.yaml`, because those are part of the policy's input encoding and
changing them would score a different policy than the one on disk.

Usage:

    uv run python scripts/score_runs.py                 # score what is not scored
    uv run python scripts/score_runs.py --force         # score everything again
    uv run python scripts/score_runs.py --episodes 40   # a bigger sample
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Ensure the repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sailbench.rl.envs.waypoint_env import WaypointEnv, WaypointEnvConfig

SCORECARD_NAME = "scorecard.json"

#: The checkpoints a run keeps, and the id each is scored under.
CHECKPOINTS: tuple[tuple[str, str], ...] = (
    ("best", "best_model/best_model.zip"),
    ("final", "final_model.zip"),
)

#: Angle off the true wind inside which a step counts as pinching [deg]. Fixed
#: here rather than read from each run's `no_go_zone_half_angle_deg`, which is
#: 35 on some runs and 45 on others: a metric that moves with the config being
#: measured compares nothing.
PINCH_THRESHOLD_DEG = 25.0

#: Sail command this close to an extreme counts as pinned.
SHEET_PINNED_EPS = 0.01

#: Speed past which the episode is the integrator running away, not the boat
#: sailing [m/s]. Nothing in this repo floats a hull that does a tenth of it;
#: `basic_sailbot` under some of the older policies reaches 1e30 a few steps
#: before the wave-drag term overflows outright. Such an episode is counted as
#: diverged and left out of the behaviour averages, which it would otherwise
#: be the only contributor to.
DIVERGED_SPEED_M_S = 20.0

#: Task knobs forced to a common value so every policy meets the same marks.
#: Only knobs that change what a seed produces belong here -- never an
#: observation scale, which is part of the encoding the policy was trained with.
TASK_OVERRIDES: dict[str, Any] = {"upwind_waypoint_bias": 0.5}


@dataclass(slots=True)
class Score:
    """What one checkpoint did over the scored episodes.

    Success and divergence are counted over every episode. The behaviour
    numbers -- speed, pinch, sheet, steps, path -- are averaged over the
    episodes that stayed finite, because an episode that blew up contributes a
    number with no physical meaning and swamps the others when it does.
    """

    model: str
    sha256: str
    size: int
    episodes: int
    success_rate: float
    mean_speed_m_s: float
    pinch_share: float
    sheet_pinned_share: float
    mean_steps: float
    mean_time_to_goal_s: float | None
    mean_path_efficiency: float
    mean_final_distance_m: float
    #: Share of episodes the integrator could not finish. Non-zero means this
    #: policy drives the boat somewhere the solver blows up in, which is a fact
    #: about the pair and worth saying out loud rather than averaging away.
    diverged_share: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _true_wind_angle_deg(env: WaypointEnv) -> float:
    """Angle between the bow and the wind's source [deg], 0 dead into it.

    Same geometry as `WaypointEnv._no_go_zone_penalty`: `wind_dir_deg` is the
    direction the wind blows *toward*, so its component along the bow is
    negative when the boat points at the source.
    """
    wind_dir_rad = math.radians(float(env.hub.sail_cfg.get("wind_dir_deg", 90.0)))
    c, s = env.state.psi
    wind_boat_x = c * math.cos(wind_dir_rad) + s * math.sin(wind_dir_rad)
    return math.degrees(math.acos(max(-1.0, min(1.0, -wind_boat_x))))


def score_checkpoint(model_path: Path, env_cfg: WaypointEnvConfig, episodes: int, seed_base: int) -> Score:
    """Run one checkpoint over `episodes` fixed-seed episodes and measure it."""
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path))
    env = WaypointEnv(config=env_cfg)

    successes: list[float] = []
    diverged: list[float] = []
    speeds: list[float] = []
    pinches: list[float] = []
    pinned: list[float] = []
    steps: list[int] = []
    times_to_goal: list[float] = []
    efficiencies: list[float] = []
    final_distances: list[float] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed_base + episode)
        start = (env.state.x, env.state.y)
        start_distance = math.hypot(env.waypoint[0] - start[0], env.waypoint[1] - start[1])
        sailed = 0.0
        previous = start
        episode_speeds: list[float] = []
        episode_pinch = 0
        episode_pinned = 0
        info: dict[str, Any] = {}
        done = False
        blew_up = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            try:
                obs, _, terminated, truncated, info = env.step(action)
            except (ArithmeticError, ValueError):
                blew_up = True
                break
            if not np.all(np.isfinite(obs)):
                blew_up = True
                break
            done = bool(terminated or truncated)

            speed = math.hypot(env.state.u, env.state.v)
            if not math.isfinite(speed) or speed > DIVERGED_SPEED_M_S:
                blew_up = True
                break
            episode_speeds.append(speed)
            if _true_wind_angle_deg(env) < PINCH_THRESHOLD_DEG:
                episode_pinch += 1
            if abs(float(np.clip(action[1], -1.0, 1.0))) >= 1.0 - SHEET_PINNED_EPS:
                episode_pinned += 1
            sailed += math.hypot(env.state.x - previous[0], env.state.y - previous[1])
            previous = (env.state.x, env.state.y)

        diverged.append(1.0 if blew_up else 0.0)
        success = bool(info.get("success", False)) and not blew_up
        successes.append(1.0 if success else 0.0)
        if blew_up:
            continue

        count = max(len(episode_speeds), 1)
        speeds.append(sum(episode_speeds) / count)
        pinches.append(episode_pinch / count)
        pinned.append(episode_pinned / count)
        steps.append(int(info.get("step_count", count)))
        final_distances.append(float(info.get("distance_to_waypoint", 0.0)))
        efficiencies.append(min(start_distance / sailed, 1.0) if sailed > 0.0 else 0.0)
        if success:
            times_to_goal.append(float(info.get("sim_time_s", 0.0)))

    def mean(values: list[float] | list[int]) -> float:
        """Average the episodes that finished; 0 when none of them did."""
        return float(np.mean(values)) if values else 0.0

    return Score(
        model=str(model_path),
        sha256=_sha256(model_path),
        size=model_path.stat().st_size,
        episodes=episodes,
        success_rate=float(np.mean(successes)),
        mean_speed_m_s=mean(speeds),
        pinch_share=mean(pinches),
        sheet_pinned_share=mean(pinned),
        mean_steps=mean(steps),
        mean_time_to_goal_s=float(np.mean(times_to_goal)) if times_to_goal else None,
        mean_path_efficiency=mean(efficiencies),
        mean_final_distance_m=mean(final_distances),
        diverged_share=float(np.mean(diverged)),
    )


def score_run(run: Path, episodes: int, seed_base: int, force: bool) -> dict[str, Any] | None:
    """Score every checkpoint in one run directory, reusing what is still valid."""
    used = run / "config_used.yaml"
    if not used.is_file():
        return None
    full_cfg = dict(yaml.safe_load(used.read_text(encoding="utf-8")) or {})
    env_section = dict(full_cfg.get("env", {}))
    env_section.update(TASK_OVERRIDES)
    env_cfg = WaypointEnvConfig(**env_section)

    card_path = run / SCORECARD_NAME
    previous: dict[str, Any] = {}
    if card_path.is_file() and not force:
        # A malformed scorecard is a cache miss, not an error: rewriting it is the fix.
        with contextlib.suppress(OSError, ValueError):
            previous = json.loads(card_path.read_text(encoding="utf-8"))

    task = {
        "episodes": episodes,
        "seed_base": seed_base,
        "pinch_threshold_deg": PINCH_THRESHOLD_DEG,
        "overrides": TASK_OVERRIDES,
        "boat": env_section.get("simulator_config"),
    }
    kept = previous.get("checkpoints", {}) if previous.get("task") == task else {}

    scored: dict[str, Any] = {}
    for label, relative in CHECKPOINTS:
        model_path = run / relative
        if not model_path.is_file():
            continue
        earlier = kept.get(label)
        if earlier and earlier.get("size") == model_path.stat().st_size and not force:
            print(f"  {label:5s} already scored")
            scored[label] = earlier
            continue
        print(f"  {label:5s} scoring {episodes} episodes ...", flush=True)
        scored[label] = asdict(score_checkpoint(model_path, env_cfg, episodes, seed_base))

    if not scored:
        return None
    card = {"task": task, "scored_at": datetime.now(UTC).isoformat(timespec="seconds"), "checkpoints": scored}
    card_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    return card


def main(argv: list[str] | None = None) -> None:
    """Score the runs named on the command line, or every run under `runs/`."""
    parser = argparse.ArgumentParser(description="Score trained policies for the shipyard.")
    parser.add_argument("runs", nargs="*", type=Path, help="Run directories (default: every one under runs/).")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per checkpoint (default: 20).")
    parser.add_argument("--seed-base", type=int, default=1000, help="First episode seed (default: 1000).")
    parser.add_argument("--force", action="store_true", help="Rescore checkpoints that already have a scorecard.")
    args = parser.parse_args(argv)

    runs = args.runs or sorted(p for p in Path("runs").iterdir() if p.is_dir())
    for run in runs:
        if not (run / "config_used.yaml").is_file():
            continue
        print(f"{run}:")
        card = score_run(run, args.episodes, args.seed_base, args.force)
        if card is None:
            print("  no checkpoints")
            continue
        for label, score in card["checkpoints"].items():
            blew_up = score.get("diverged_share", 0.0)
            note = f" · {blew_up * 100:.0f}% diverged" if blew_up else ""
            print(
                f"  {label:5s} {score['mean_speed_m_s']:.2f} m/s · "
                f"{score['pinch_share'] * 100:.0f}% pinch · "
                f"{score['sheet_pinned_share'] * 100:.0f}% sheet pinned · "
                f"{score['success_rate'] * 100:.0f}% success{note}"
            )


if __name__ == "__main__":
    main()
