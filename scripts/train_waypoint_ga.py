"""Minimal Torch genetic algorithm trainer for WaypointEnv."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

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


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but no CUDA device is available.")
    return torch.device(device_name)


def _make_model(obs_dim: int, act_dim: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = obs_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.Tanh())
        in_dim = h
    layers.append(nn.Linear(in_dim, act_dim))
    layers.append(nn.Tanh())  # Actions in [-1, 1]
    return nn.Sequential(*layers)


def _evaluate_genome(
    genome: torch.Tensor,
    model: nn.Module,
    env_cfg: WaypointEnvConfig,
    episodes: int,
    base_seed: int,
    eval_index: int,
    device: torch.device,
) -> tuple[float, float, float]:
    vector_to_parameters(genome, model.parameters())
    env = WaypointEnv(config=env_cfg)
    returns: list[float] = []
    successes = 0
    final_distances: list[float] = []
    try:
        with torch.no_grad():
            for ep in range(episodes):
                obs, _ = env.reset(seed=base_seed + eval_index * 100_000 + ep)
                done = False
                total_reward = 0.0
                final_info: dict[str, Any] | None = None
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    action = model(obs_t).detach().cpu().numpy().astype(np.float32, copy=False)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    done = terminated or truncated
                    final_info = info
                returns.append(total_reward)
                successes += int(bool(final_info and final_info.get("success", False)))
                final_distances.append(float(final_info.get("distance_to_waypoint", 0.0)) if final_info else 0.0)
    finally:
        env.close()

    mean_return = float(np.mean(returns)) if returns else float("-inf")
    success_rate = float(successes / max(len(returns), 1))
    mean_final_distance = float(np.mean(final_distances)) if final_distances else float("inf")
    return mean_return, success_rate, mean_final_distance


def _mutate(child: torch.Tensor, mutation_rate: float, mutation_std: float) -> torch.Tensor:
    if mutation_rate <= 0.0 or mutation_std <= 0.0:
        return child
    mask = torch.rand_like(child) < mutation_rate
    if not bool(mask.any().item()):
        return child
    noise = torch.randn_like(child) * mutation_std
    return torch.where(mask, child + noise, child)


def _watch_rollout_loop(
    stop_event: threading.Event,
    genome_lock: threading.Lock,
    genome_box: dict[str, torch.Tensor],
    watch_model: nn.Module,
    env_cfg: WaypointEnvConfig,
    device: torch.device,
    base_seed: int,
) -> None:
    """Stream continuous rollouts of the latest best genome to the browser."""
    env = WaypointEnv(config=env_cfg)
    episode = 0
    try:
        while not stop_event.is_set():
            with genome_lock:
                genome = genome_box["genome"].clone()
            vector_to_parameters(genome, watch_model.parameters())
            obs, _ = env.reset(seed=base_seed + episode)
            done = False
            while not done and not stop_event.is_set():
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    action = watch_model(obs_t).detach().cpu().numpy().astype(np.float32, copy=False)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            episode += 1
    finally:
        env.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Minimal Torch GA trainer for SailBench waypoint navigation.")
    parser.add_argument("--config", type=Path, default=Path("configs/rl_waypoint_ga.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default=None, help="Torch device override (cpu, cuda, cuda:0).")
    parser.add_argument(
        "--watch-web",
        action="store_true",
        help="Stream the current best policy to the browser visualizer (websocket).",
    )
    parser.add_argument("--watch-host", default="127.0.0.1", help="Websocket bind host.")
    parser.add_argument("--watch-port", type=int, default=8765, help="Websocket port (frontend default: 8765).")
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
    torch.manual_seed(seed)

    population_size = int(train_cfg.get("population_size", 64))
    generations = int(train_cfg.get("generations", 100))
    elite_count = int(train_cfg.get("elite_count", 8))
    episodes_per_candidate = int(train_cfg.get("episodes_per_candidate", 2))
    mutation_rate = float(train_cfg.get("mutation_rate", 0.05))
    mutation_std = float(train_cfg.get("mutation_std", 0.10))
    init_std = float(train_cfg.get("init_std", 0.5))
    checkpoint_freq = int(train_cfg.get("checkpoint_freq_gens", 10))
    log_candidate_every = int(train_cfg.get("log_candidate_every", 8))
    watch_enabled = bool(args.watch_web or train_cfg.get("watch_web", False))
    watch_stride = int(args.watch_stride if args.watch_stride is not None else train_cfg.get("watch_stride", 2))

    elite_count = max(1, min(elite_count, population_size))
    hidden_sizes = _parse_hidden_sizes(train_cfg.get("hidden_sizes"))
    device_name = str(args.device if args.device is not None else train_cfg.get("device", "cuda_if_available"))
    device = _resolve_device(device_name)

    shape_env = WaypointEnv(config=env_cfg)
    obs_dim = int(np.prod(shape_env.observation_space.shape))
    act_dim = int(np.prod(shape_env.action_space.shape))
    shape_env.close()

    model = _make_model(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes).to(device)
    n_params = int(parameters_to_vector(model.parameters()).numel())

    population = torch.randn(population_size, n_params, dtype=torch.float32, device=device) * init_std
    best_fitness = float("-inf")
    best_genome = population[0].clone()
    history: list[dict[str, float | int]] = []

    vis_server: LiveTrainingVisServer | None = None
    watch_thread: threading.Thread | None = None
    watch_stop = threading.Event()
    watch_lock = threading.Lock()
    watch_genome_box: dict[str, torch.Tensor] = {"genome": best_genome.clone()}
    if watch_enabled:
        vis_server = LiveTrainingVisServer(host=args.watch_host, port=args.watch_port)
        vis_server.start()
        print(f"[vis] open frontend + connect to ws://{args.watch_host}:{args.watch_port}/sim", flush=True)
        vis_env_cfg = replace(
            env_cfg,
            vis_callback=vis_server.publish,
            vis_stride_steps=max(watch_stride, 1),
        )
        watch_model = _make_model(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes).to(device)
        watch_thread = threading.Thread(
            target=_watch_rollout_loop,
            args=(
                watch_stop,
                watch_lock,
                watch_genome_box,
                watch_model,
                vis_env_cfg,
                device,
                seed + 1_000_000,
            ),
            name="ga-web-watch",
            daemon=True,
        )
        watch_thread.start()

    print(
        (
            "[ga] "
            f"population={population_size} generations={generations} "
            f"episodes_per_candidate={episodes_per_candidate} params={n_params} device={device}"
            f"{' watch_web' if watch_enabled else ''}"
        ),
        flush=True,
    )

    try:
        for gen in range(generations):
            fitness = np.zeros(population_size, dtype=np.float64)
            success_rates = np.zeros(population_size, dtype=np.float64)
            final_distances = np.zeros(population_size, dtype=np.float64)
            base_seed = seed + gen * 17
            gen_start = datetime.now(tz=UTC)

            for idx in range(population_size):
                fit, succ_rate, mean_dist = _evaluate_genome(
                    genome=population[idx],
                    model=model,
                    env_cfg=env_cfg,
                    episodes=episodes_per_candidate,
                    base_seed=base_seed,
                    eval_index=idx,
                    device=device,
                )
                fitness[idx] = fit
                success_rates[idx] = succ_rate
                final_distances[idx] = mean_dist
                if (idx + 1) % max(log_candidate_every, 1) == 0 or idx + 1 == population_size:
                    print(f"[gen {gen:04d}] candidate {idx + 1}/{population_size}", flush=True)

            ranked = np.argsort(fitness)[::-1].copy()
            elite_idx = ranked[:elite_count].copy()
            elites = population[torch.as_tensor(elite_idx, dtype=torch.long, device=device)].clone()

            gen_best_idx = int(ranked[0])
            gen_best_fitness = float(fitness[gen_best_idx])
            gen_mean_fitness = float(np.mean(fitness))
            gen_best_success = float(success_rates[gen_best_idx])
            gen_best_distance = float(final_distances[gen_best_idx])

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_genome = population[gen_best_idx].clone()

            if watch_enabled:
                with watch_lock:
                    watch_genome_box["genome"] = population[gen_best_idx].clone()

            history.append(
                {
                    "generation": gen,
                    "best_fitness": gen_best_fitness,
                    "mean_fitness": gen_mean_fitness,
                    "best_success_rate": gen_best_success,
                    "best_mean_final_distance": gen_best_distance,
                }
            )
            elapsed_s = (datetime.now(tz=UTC) - gen_start).total_seconds()
            print(
                f"[gen {gen:04d}] best={gen_best_fitness:.3f} mean={gen_mean_fitness:.3f} "
                f"success={gen_best_success:.2f} dist={gen_best_distance:.3f} elapsed={elapsed_s:.1f}s",
                flush=True,
            )

            if checkpoint_freq > 0 and (gen + 1) % checkpoint_freq == 0:
                np.savez_compressed(
                    checkpoints_dir / f"ga_waypoint_gen_{gen + 1:04d}.npz",
                    generation=gen + 1,
                    best_fitness=best_fitness,
                    best_genome=best_genome.detach().cpu().numpy(),
                    hidden_sizes=np.asarray(hidden_sizes, dtype=np.int64),
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    device=str(device),
                )

            next_population: list[torch.Tensor] = [elites[i].clone() for i in range(elites.shape[0])]
            while len(next_population) < population_size:
                parent = elites[rng.integers(0, elites.shape[0])].clone()
                child = _mutate(parent, mutation_rate=mutation_rate, mutation_std=mutation_std)
                next_population.append(child)
            population = torch.stack(next_population, dim=0)

    finally:
        watch_stop.set()
        if watch_thread is not None:
            watch_thread.join(timeout=10.0)
        if vis_server is not None:
            vis_server.close()

    final_model_path = run_dir / "final_model.npz"
    np.savez_compressed(
        final_model_path,
        best_fitness=best_fitness,
        best_genome=best_genome.detach().cpu().numpy(),
        hidden_sizes=np.asarray(hidden_sizes, dtype=np.int64),
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=str(device),
    )
    with (run_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "run_dir": str(run_dir),
                "seed": seed,
                "population_size": population_size,
                "generations": generations,
                "episodes_per_candidate": episodes_per_candidate,
                "best_fitness": best_fitness,
                "device": str(device),
                "final_model_path": str(final_model_path),
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()

