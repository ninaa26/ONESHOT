"""Genetic algorithm for WaypointEnv using MLGAgentPolicy (Torch, CUDA-parallel eval)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sailbench.rl.envs.waypoint_env import WaypointEnv, WaypointEnvConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return dict(yaml.safe_load(file) or {})


def _run_dir(base: Path) -> Path:
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    d = base / f"waypoint_ga_{ts}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _resolve_device(name: str) -> torch.device:
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


class MLGAgentPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_hidden: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)])
        self.output = nn.Linear(hidden_dim, output_dim)

    def randomize_parameters(self, std: float = 0.1) -> None:
        with torch.no_grad():
            for param in self.parameters():
                param.copy_(torch.randn_like(param) * std)

    def combine_agent_genomes(
        self,
        agents: list[MLGAgentPolicy],
        mutation_rate: float,
        mutation_std: float = 0.1,
    ) -> None:
        """Average matching parameters across agents, then Gaussian mutation."""
        with torch.no_grad():
            per_agent = [list(a.parameters()) for a in agents]
            for pi, self_param in enumerate(self.parameters()):
                stacked = torch.stack([per_agent[j][pi] for j in range(len(agents))], dim=0)
                self_param.copy_(stacked.mean(dim=0))
                mutation_mask = torch.rand_like(self_param) < mutation_rate
                self_param.add_(mutation_mask * torch.randn_like(self_param) * mutation_std)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input(obs))
        for layer in self.hidden:
            x = F.relu(layer(x))
        return torch.tanh(self.output(x))


def _rollout(
    policy: MLGAgentPolicy,
    env_cfg: WaypointEnvConfig,
    episodes: int,
    base_seed: int,
    eval_index: int,
    device: torch.device,
) -> tuple[float, float, float]:
    env = WaypointEnv(config=env_cfg)
    returns: list[float] = []
    successes = 0
    dists: list[float] = []
    try:
        policy.eval()
        with torch.no_grad():
            for ep in range(episodes):
                obs, _ = env.reset(seed=base_seed + eval_index * 100_000 + ep)
                done = False
                total = 0.0
                info_last: dict[str, Any] | None = None
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    action = policy(obs_t).detach().cpu().numpy().astype(np.float32, copy=False)
                    obs, rew, term, trunc, info = env.step(action)
                    total += float(rew)
                    done = term or trunc
                    info_last = info
                returns.append(total)
                successes += int(bool(info_last and info_last.get("success")))
                dists.append(float(info_last.get("distance_to_waypoint", 0.0)) if info_last else 0.0)
    finally:
        env.close()
    return (
        float(np.mean(returns)) if returns else float("-inf"),
        float(successes / max(len(returns), 1)),
        float(np.mean(dists)) if dists else float("inf"),
    )


def _eval_chunk(payload: tuple[Any, ...]) -> list[tuple[int, float, float, float]]:
    """Process pool: one GPU, many genomes in sequence (chunk)."""
    (
        indices,
        genomes_np,
        env_cfg_dict,
        obs_dim,
        act_dim,
        hidden_dim,
        num_hidden,
        episodes,
        base_seed,
        device_str,
    ) = payload
    device = torch.device(device_str)
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        torch.cuda.set_device(idx)
    env_cfg = WaypointEnvConfig(**env_cfg_dict)
    policy = MLGAgentPolicy(obs_dim, hidden_dim, act_dim, num_hidden).to(device)
    out: list[tuple[int, float, float, float]] = []
    for row, idx in enumerate(indices):
        vec = torch.from_numpy(genomes_np[row].astype(np.float32)).to(device)
        vector_to_parameters(vec, policy.parameters())
        fit, sr, md = _rollout(policy, env_cfg, episodes, base_seed, idx, device)
        out.append((idx, fit, sr, md))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rl_waypoint_ga.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-workers", type=int, default=None, help="Parallel eval processes (CUDA chunks).")
    args = parser.parse_args(argv)

    full = _load_yaml(args.config)
    env_cfg = WaypointEnvConfig(**full.get("env", {}))
    t = full.get("train", {})

    out_dir = _run_dir(args.run_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(full, f, sort_keys=False)

    seed = int(t.get("seed", 7))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    pop = int(t.get("population_size", 100))
    elite_n = max(1, min(int(t.get("elite_count", 10)), pop))
    parents_k = max(2, min(int(t.get("parents_per_child", 3)), elite_n))
    gens = int(t.get("generations", 100))
    episodes = int(t.get("episodes_per_candidate", 2))
    hidden_dim = int(t.get("hidden_dim", 64))
    num_hidden = int(t.get("num_hidden", 2))
    init_std = float(t.get("init_std", 0.1))
    combine_mut_rate = float(t.get("combine_mutation_rate", 0.1))
    combine_mut_std = float(t.get("combine_mutation_std", 0.1))
    dev = _resolve_device(str(args.device or t.get("device", "cuda_if_available")))
    n_workers = max(1, int(args.n_workers if args.n_workers is not None else t.get("n_workers", 4)))
    checkpoint_freq = int(t.get("checkpoint_freq_gens", 10))

    e0 = WaypointEnv(config=env_cfg)
    obs_dim = int(np.prod(e0.observation_space.shape))
    act_dim = int(np.prod(e0.action_space.shape))
    e0.close()

    template = MLGAgentPolicy(obs_dim, hidden_dim, act_dim, num_hidden).to(dev)
    n_params = int(parameters_to_vector(template.parameters()).numel())
    env_dict = dict(full.get("env", {}))
    # Workers need an explicit CUDA index for torch.cuda.set_device
    if dev.type == "cuda" and dev.index is None:
        device_str = "cuda:0"
    else:
        device_str = str(dev)

    population = torch.randn(pop, n_params, dtype=torch.float32, device=dev) * init_std
    best_vec = population[0].clone()
    best_fit = float("-inf")
    hist: list[dict[str, float | int]] = []

    executor: concurrent.futures.ProcessPoolExecutor | None = None
    ctx = mp.get_context("spawn")
    if n_workers > 1:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)

    print(
        f"[ga] pop={pop} elite={elite_n} parents={parents_k} gens={gens} "
        f"hidden={hidden_dim}x{num_hidden} params={n_params} device={dev} n_workers={n_workers} "
        f"checkpoint_freq_gens={checkpoint_freq}",
        flush=True,
    )

    try:
        for g in range(gens):
            base_seed = seed + g * 17
            t0 = datetime.now(tz=UTC)
            fitness = np.zeros(pop, dtype=np.float64)
            succ = np.zeros(pop, dtype=np.float64)
            distm = np.zeros(pop, dtype=np.float64)

            genomes_cpu = population.detach().cpu().numpy()

            if executor is None:
                for idx in range(pop):
                    vector_to_parameters(population[idx], template.parameters())
                    fit, sr, md = _rollout(template, env_cfg, episodes, base_seed, idx, dev)
                    fitness[idx], succ[idx], distm[idx] = fit, sr, md
            else:
                chunks = np.array_split(np.arange(pop), min(n_workers, pop))
                futs = []
                for chunk in chunks:
                    if chunk.size == 0:
                        continue
                    inds = chunk.tolist()
                    pay = (
                        inds,
                        genomes_cpu[inds],
                        env_dict,
                        obs_dim,
                        act_dim,
                        hidden_dim,
                        num_hidden,
                        episodes,
                        base_seed,
                        device_str,
                    )
                    futs.append(executor.submit(_eval_chunk, pay))
                for fut in concurrent.futures.as_completed(futs):
                    for idx, fit, sr, md in fut.result():
                        fitness[idx], succ[idx], distm[idx] = fit, sr, md

            order = np.argsort(fitness)[::-1].copy()
            elite_idx = order[:elite_n].astype(np.int64, copy=True)
            elites = population[torch.as_tensor(elite_idx, device=dev, dtype=torch.long)].clone()

            bi = int(order[0])
            gf, gm = float(fitness[bi]), float(np.mean(fitness))
            if gf > best_fit:
                best_fit = gf
                best_vec = population[bi].clone()

            hist.append({"generation": g, "best": gf, "mean": gm})
            dt = (datetime.now(tz=UTC) - t0).total_seconds()
            print(f"[gen {g:04d}] best={gf:.3f} mean={gm:.3f} s={dt:.1f}", flush=True)

            if checkpoint_freq > 0 and (g + 1) % checkpoint_freq == 0:
                np.savez_compressed(
                    ckpt_dir / f"ga_waypoint_gen_{g + 1:04d}.npz",
                    generation=g + 1,
                    best_fitness=best_fit,
                    best_genome=best_vec.detach().cpu().numpy(),
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    hidden_dim=hidden_dim,
                    num_hidden=num_hidden,
                    seed=seed,
                    device=str(dev),
                )

            # Next generation: keep elites; fill rest by averaging `parents_k` elites + mutation
            next_rows: list[torch.Tensor] = [elites[i].clone() for i in range(elite_n)]
            parents = [MLGAgentPolicy(obs_dim, hidden_dim, act_dim, num_hidden).to(dev) for _ in range(parents_k)]
            child = MLGAgentPolicy(obs_dim, hidden_dim, act_dim, num_hidden).to(dev)
            while len(next_rows) < pop:
                parent_ix = rng.choice(elite_n, size=parents_k, replace=False)
                for j, ix in enumerate(parent_ix):
                    vector_to_parameters(elites[ix], parents[j].parameters())
                child.combine_agent_genomes(parents, combine_mut_rate, combine_mut_std)
                next_rows.append(parameters_to_vector(child.parameters()).clone())
            population = torch.stack(next_rows, dim=0)

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    np.savez_compressed(
        out_dir / "final_model.npz",
        best_fitness=best_fit,
        best_genome=best_vec.detach().cpu().numpy(),
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
        device=str(dev),
    )
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(out_dir),
                "seed": seed,
                "best_fitness": best_fit,
                "n_workers": n_workers,
                "device": str(dev),
                "checkpoint_freq_gens": checkpoint_freq,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
