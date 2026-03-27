"""Train a waypoint-navigation policy with a simple genetic algorithm."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import time
import sys
import threading
from dataclasses import dataclass, replace
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
from sailbench.rl.live_vis import LiveTrainingVisServer


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return dict(yaml.safe_load(file) or {})


def _build_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"waypoint_ga_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _parse_hidden_sizes(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return (64, 64)
    if isinstance(raw, list) and raw and all(isinstance(v, int) and v > 0 for v in raw):
        return tuple(raw)
    raise ValueError("train.hidden_sizes must be a non-empty list of positive integers")


@dataclass(slots=True)
class MlpGenomeLayout:
    input_dim: int
    hidden_sizes: tuple[int, ...]
    output_dim: int

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        return (self.input_dim, *self.hidden_sizes, self.output_dim)

    @property
    def n_params(self) -> int:
        total = 0
        sizes = self.layer_sizes
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:], strict=True):
            total += in_dim * out_dim + out_dim
        return total

    def unflatten(self, genome: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        idx = 0
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        sizes = self.layer_sizes
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:], strict=True):
            w_count = in_dim * out_dim
            w = genome[idx : idx + w_count].reshape(in_dim, out_dim)
            idx += w_count
            b = genome[idx : idx + out_dim]
            idx += out_dim
            weights.append(w)
            biases.append(b)
        return weights, biases


def _policy_action(obs: np.ndarray, genome: np.ndarray, layout: MlpGenomeLayout) -> np.ndarray:
    weights, biases = layout.unflatten(genome)
    x = obs.astype(np.float64, copy=False)
    for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
        x = x @ w + b
        x = np.tanh(x) if i < len(weights) - 1 else np.tanh(x)
    return np.asarray(x, dtype=np.float32)


def _evaluate_genome(
    genome: np.ndarray,
    layout: MlpGenomeLayout,
    env_cfg: WaypointEnvConfig,
    episodes: int,
    base_seed: int,
    eval_index: int,
) -> tuple[float, float, float]:
    env = WaypointEnv(config=env_cfg)
    returns: list[float] = []
    successes = 0
    final_distances: list[float] = []
    try:
        for ep in range(episodes):
            seed = base_seed + eval_index * 100_000 + ep
            obs, _ = env.reset(seed=seed)
            done = False
            ep_return = 0.0
            final_info: dict[str, Any] | None = None
            while not done:
                action = _policy_action(obs, genome, layout)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)
                done = terminated or truncated
                final_info = info
            returns.append(ep_return)
            success = bool(final_info and final_info.get("success", False))
            successes += int(success)
            final_distances.append(float(final_info.get("distance_to_waypoint", 0.0)) if final_info else 0.0)
    finally:
        env.close()

    mean_return = float(np.mean(returns)) if returns else float("-inf")
    success_rate = float(successes / max(len(returns), 1))
    mean_final_distance = float(np.mean(final_distances)) if final_distances else float("inf")
    return mean_return, success_rate, mean_final_distance


def _evaluate_genome_task(
    idx: int,
    genome: np.ndarray,
    layout: MlpGenomeLayout,
    env_cfg: WaypointEnvConfig,
    episodes: int,
    base_seed: int,
) -> tuple[int, float, float, float]:
    fit, succ_rate, mean_dist = _evaluate_genome(
        genome=genome,
        layout=layout,
        env_cfg=env_cfg,
        episodes=episodes,
        base_seed=base_seed,
        eval_index=idx,
    )
    return idx, fit, succ_rate, mean_dist


def _watch_rollout_loop(
    stop_event: threading.Event,
    genome_lock: threading.Lock,
    genome_box: dict[str, np.ndarray],
    layout: MlpGenomeLayout,
    env_cfg: WaypointEnvConfig,
    base_seed: int,
) -> None:
    """Continuously stream a demo rollout using the latest best genome."""
    env = WaypointEnv(config=env_cfg)
    episode = 0
    try:
        while not stop_event.is_set():
            with genome_lock:
                genome = genome_box["genome"].copy()
            obs, _ = env.reset(seed=base_seed + episode)
            done = False
            while not done and not stop_event.is_set():
                action = _policy_action(obs, genome, layout)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            episode += 1
    finally:
        env.close()


def _tournament_select(
    rng: np.random.Generator,
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_size: int,
) -> np.ndarray:
    n = population.shape[0]
    sample_idx = rng.choice(n, size=max(tournament_size, 1), replace=False)
    best_idx = sample_idx[int(np.argmax(fitness[sample_idx]))]
    return population[best_idx].copy()


def _crossover(
    rng: np.random.Generator,
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    crossover_rate: float,
) -> np.ndarray:
    if rng.random() >= crossover_rate:
        return parent_a.copy()
    mask = rng.random(parent_a.shape[0]) < 0.5
    child = parent_a.copy()
    child[mask] = parent_b[mask]
    return child


def _mutate(
    rng: np.random.Generator,
    genome: np.ndarray,
    mutation_rate: float,
    mutation_std: float,
) -> np.ndarray:
    mask = rng.random(genome.shape[0]) < mutation_rate
    if not np.any(mask):
        return genome
    mutated = genome.copy()
    mutated[mask] += rng.normal(loc=0.0, scale=mutation_std, size=int(np.sum(mask)))
    return mutated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train waypoint GA policy on SailBench.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl_waypoint_ga.yaml"),
        help="Path to GA training config YAML.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs"),
        help="Base directory for training outputs.",
    )
    parser.add_argument(
        "--watch-web",
        action="store_true",
        help="Broadcast best-genome rollout over websocket for the browser visualizer.",
    )
    parser.add_argument(
        "--watch-host",
        default="127.0.0.1",
        help="Host for training visualization websocket server.",
    )
    parser.add_argument(
        "--watch-port",
        type=int,
        default=8765,
        help="Port for training visualization websocket server.",
    )
    parser.add_argument(
        "--watch-stride",
        type=int,
        default=None,
        help="Publish every N env steps (default from config or 2).",
    )
    args = parser.parse_args(argv)

    full_cfg = _load_yaml(args.config)
    env_cfg = WaypointEnvConfig(**full_cfg.get("env", {}))
    train_cfg = full_cfg.get("train", {})

    run_dir = _build_run_dir(args.run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(full_cfg, file, sort_keys=False)

    seed = int(train_cfg.get("seed", 7))
    rng = np.random.default_rng(seed)
    population_size = int(train_cfg.get("population_size", 128))
    generations = int(train_cfg.get("generations", 200))
    elite_count = int(train_cfg.get("elite_count", 8))
    tournament_size = int(train_cfg.get("tournament_size", 8))
    crossover_rate = float(train_cfg.get("crossover_rate", 0.7))
    mutation_rate = float(train_cfg.get("mutation_rate", 0.05))
    mutation_std = float(train_cfg.get("mutation_std", 0.10))
    init_std = float(train_cfg.get("init_std", 0.5))
    episodes_per_candidate = int(train_cfg.get("episodes_per_candidate", 2))
    checkpoint_freq = int(train_cfg.get("checkpoint_freq_gens", 10))
    log_candidate_every = int(train_cfg.get("log_candidate_every", 8))
    n_workers = max(int(train_cfg.get("n_workers", 1)), 1)
    watch_enabled = bool(args.watch_web or train_cfg.get("watch_web", False))
    watch_stride = int(args.watch_stride if args.watch_stride is not None else train_cfg.get("watch_stride", 2))

    shape_env = WaypointEnv(config=env_cfg)
    obs_dim = int(np.prod(shape_env.observation_space.shape))
    act_dim = int(np.prod(shape_env.action_space.shape))
    shape_env.close()
    hidden_sizes = _parse_hidden_sizes(train_cfg.get("hidden_sizes"))
    layout = MlpGenomeLayout(input_dim=obs_dim, hidden_sizes=hidden_sizes, output_dim=act_dim)

    population = rng.normal(loc=0.0, scale=init_std, size=(population_size, layout.n_params))
    best_genome = population[0].copy()
    best_fitness = float("-inf")
    history: list[dict[str, float | int]] = []

    vis_server: LiveTrainingVisServer | None = None
    watch_thread: threading.Thread | None = None
    watch_stop_event = threading.Event()
    watch_genome_lock = threading.Lock()
    watch_genome_box: dict[str, np.ndarray] = {"genome": best_genome.copy()}
    if watch_enabled:
        vis_server = LiveTrainingVisServer(host=args.watch_host, port=args.watch_port)
        vis_server.start()
        print(f"[vis] streaming to ws://{args.watch_host}:{args.watch_port}/sim", flush=True)
        vis_env_cfg = replace(env_cfg, vis_callback=vis_server.publish, vis_stride_steps=max(watch_stride, 1))
        watch_thread = threading.Thread(
            target=_watch_rollout_loop,
            args=(watch_stop_event, watch_genome_lock, watch_genome_box, layout, vis_env_cfg, seed + 1_000_000),
            name="ga-live-watch",
            daemon=True,
        )
        watch_thread.start()

    print(
        (
            "[ga] "
            f"population={population_size} generations={generations} "
            f"episodes_per_candidate={episodes_per_candidate} params={layout.n_params} "
            f"n_workers={n_workers}"
        ),
        flush=True,
    )
    print(
        "[ga] note: each generation may take a while; candidate progress logs are enabled",
        flush=True,
    )

    executor: concurrent.futures.ProcessPoolExecutor | None = None
    if n_workers > 1:
        # Spawn workers for robust cross-platform behavior.
        mp_ctx = mp.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx)

    try:
        for gen in range(generations):
            gen_start = time.perf_counter()
            fitness = np.zeros(population_size, dtype=np.float64)
            success_rates = np.zeros(population_size, dtype=np.float64)
            final_distances = np.zeros(population_size, dtype=np.float64)

            base_seed = seed + gen * 17

            # Evaluate candidate 0 in-process so we always have deterministic baseline timing.
            idx_offset = 0
            eval_cfg_for_first = env_cfg
            fit0, succ0, dist0 = _evaluate_genome(
                genome=population[0],
                layout=layout,
                env_cfg=eval_cfg_for_first,
                episodes=episodes_per_candidate,
                base_seed=base_seed,
                eval_index=0,
            )
            fitness[0] = fit0
            success_rates[0] = succ0
            final_distances[0] = dist0
            idx_offset = 1
            print(f"[gen {gen:04d}] candidate 1/{population_size} elapsed={time.perf_counter() - gen_start:.1f}s", flush=True)

            if executor is None:
                for idx in range(idx_offset, population_size):
                    fit, succ_rate, mean_dist = _evaluate_genome(
                        genome=population[idx],
                        layout=layout,
                        env_cfg=env_cfg,
                        episodes=episodes_per_candidate,
                        base_seed=base_seed,
                        eval_index=idx,
                    )
                    fitness[idx] = fit
                    success_rates[idx] = succ_rate
                    final_distances[idx] = mean_dist
                    if (idx + 1) % max(log_candidate_every, 1) == 0 or idx + 1 == population_size:
                        elapsed = time.perf_counter() - gen_start
                        per_candidate = elapsed / (idx + 1)
                        eta_s = per_candidate * (population_size - (idx + 1))
                        print(
                            (
                                f"[gen {gen:04d}] "
                                f"candidate {idx + 1}/{population_size} "
                                f"elapsed={elapsed:.1f}s eta={eta_s:.1f}s"
                            ),
                            flush=True,
                        )
            else:
                futures = [
                    executor.submit(
                        _evaluate_genome_task,
                        idx,
                        population[idx],
                        layout,
                        env_cfg,
                        episodes_per_candidate,
                        base_seed,
                    )
                    for idx in range(idx_offset, population_size)
                ]
                completed = 1
                for fut in concurrent.futures.as_completed(futures):
                    idx, fit, succ_rate, mean_dist = fut.result()
                    fitness[idx] = fit
                    success_rates[idx] = succ_rate
                    final_distances[idx] = mean_dist
                    completed += 1
                    if completed % max(log_candidate_every, 1) == 0 or completed == population_size:
                        elapsed = time.perf_counter() - gen_start
                        per_candidate = elapsed / max(completed, 1)
                        eta_s = per_candidate * (population_size - completed)
                        print(
                            (
                                f"[gen {gen:04d}] "
                                f"candidate {completed}/{population_size} "
                                f"elapsed={elapsed:.1f}s eta={eta_s:.1f}s"
                            ),
                            flush=True,
                        )

            ranked_idx = np.argsort(fitness)[::-1]
            elites = population[ranked_idx[:elite_count]].copy()
            gen_best_idx = int(ranked_idx[0])
            gen_best_fitness = float(fitness[gen_best_idx])
            gen_mean_fitness = float(np.mean(fitness))
            gen_best_success = float(success_rates[gen_best_idx])
            gen_best_distance = float(final_distances[gen_best_idx])
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_genome = population[gen_best_idx].copy()
                if watch_enabled:
                    with watch_genome_lock:
                        watch_genome_box["genome"] = best_genome.copy()

            history.append(
                {
                    "generation": gen,
                    "best_fitness": gen_best_fitness,
                    "mean_fitness": gen_mean_fitness,
                    "best_success_rate": gen_best_success,
                    "best_mean_final_distance": gen_best_distance,
                }
            )
            print(
                f"[gen {gen:04d}] best={gen_best_fitness:.3f} mean={gen_mean_fitness:.3f} "
                f"best_success={gen_best_success:.2f} best_dist={gen_best_distance:.3f} "
                f"gen_elapsed={time.perf_counter() - gen_start:.1f}s"
            , flush=True
            )

            if checkpoint_freq > 0 and ((gen + 1) % checkpoint_freq == 0):
                ckpt_path = checkpoints_dir / f"ga_waypoint_gen_{gen + 1:04d}.npz"
                np.savez_compressed(
                    ckpt_path,
                    generation=gen + 1,
                    best_fitness=best_fitness,
                    best_genome=best_genome,
                    hidden_sizes=np.asarray(hidden_sizes, dtype=np.int64),
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                )

            next_population = [*elites]
            while len(next_population) < population_size:
                parent_a = _tournament_select(rng, population, fitness, tournament_size)
                parent_b = _tournament_select(rng, population, fitness, tournament_size)
                child = _crossover(rng, parent_a, parent_b, crossover_rate)
                child = _mutate(rng, child, mutation_rate, mutation_std)
                next_population.append(child)
            population = np.asarray(next_population, dtype=np.float64)

        final_model_path = run_dir / "final_model.npz"
        np.savez_compressed(
            final_model_path,
            best_fitness=best_fitness,
            best_genome=best_genome,
            hidden_sizes=np.asarray(hidden_sizes, dtype=np.int64),
            obs_dim=obs_dim,
            act_dim=act_dim,
        )
        with (run_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        summary = {
            "run_dir": str(run_dir),
            "seed": seed,
            "population_size": population_size,
            "generations": generations,
            "episodes_per_candidate": episodes_per_candidate,
            "best_fitness": best_fitness,
            "final_model_path": str(final_model_path),
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        watch_stop_event.set()
        if watch_thread is not None:
            watch_thread.join(timeout=5.0)
        if vis_server is not None:
            vis_server.close()


if __name__ == "__main__":
    main()

