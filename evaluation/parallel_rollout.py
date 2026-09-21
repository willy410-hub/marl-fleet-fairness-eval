"""
Scalable episode collection via RLlib's distributed rollout workers
(Phase 2, addition 3).

`evaluation/episode_runner.py`'s `run_episode()` collects one episode
at a time, sequentially, in the calling process -- fine for the
evaluation suite's small `n_episodes` batches, but not how episode
collection should scale for larger training/evaluation runs. RLlib
already ships a distributed rollout-worker pool for exactly this
(`AlgorithmConfig.env_runners(num_env_runners=...)`, already used by
`training/train_ppo.py`); this module reuses that same, already-built
worker pool to collect a batch of episodes in parallel instead of
writing a custom multiprocessing layer from scratch.

The mechanism: `algo.env_runner_group.foreach_env_runner(fn,
local_env_runner=False)` dispatches `fn` to every *remote* rollout
worker (the local worker has no env by default and can't sample) and
runs them concurrently via Ray, returning each worker's collected
episodes. Timing that call with `num_env_runners=1` vs. `N` on the
identical workload is a real, direct throughput comparison -- not a
theoretical estimate.
"""
import time
from dataclasses import dataclass

import ray
from ray.tune.registry import register_env

from env.config import EnvConfig
from training.train_ppo import ENV_NAME, build_ppo_config, env_creator


@dataclass
class ParallelSampleResult:
    """One timed batch-sampling run against a given rollout-worker pool size."""

    n_rollout_workers: int
    wall_clock_seconds: float
    total_episodes: int
    total_env_steps: int
    episodes_per_second: float


def collect_episodes_parallel(
    env_config: EnvConfig, n_rollout_workers: int, rollout_fragment_length: int = 200
) -> ParallelSampleResult:
    """
    Build a PPO algorithm with `n_rollout_workers` remote rollout
    workers and time exactly one distributed `sample()` round across
    all of them -- each worker collects a batch of complete episodes
    (since `rollout_fragment_length` here equals one full episode's
    `steps_per_episode`) concurrently.
    """
    register_env(ENV_NAME, env_creator)

    config = build_ppo_config(env_config, n_rollout_workers=n_rollout_workers)
    config = config.env_runners(rollout_fragment_length=rollout_fragment_length, batch_mode="complete_episodes")
    algo = config.build()

    t0 = time.perf_counter()
    per_worker_episode_lists = algo.env_runner_group.foreach_env_runner(
        lambda r: r.sample(), local_env_runner=False
    )
    elapsed = time.perf_counter() - t0

    total_episodes = sum(len(episodes) for episodes in per_worker_episode_lists)
    total_env_steps = sum(ep.env_steps() for episodes in per_worker_episode_lists for ep in episodes)

    algo.stop()

    return ParallelSampleResult(
        n_rollout_workers=n_rollout_workers,
        wall_clock_seconds=elapsed,
        total_episodes=total_episodes,
        total_env_steps=total_env_steps,
        episodes_per_second=total_episodes / elapsed if elapsed > 0 else 0.0,
    )


def run_scaling_comparison(
    env_config: EnvConfig, worker_counts: list[int], rollout_fragment_length: int = 200
) -> list[ParallelSampleResult]:
    """
    Run `collect_episodes_parallel` once per entry in `worker_counts`
    (e.g. [1, 2, 4]) against the identical workload, so the results
    are directly comparable -- this is what produces the
    before/after throughput numbers documented in the README /
    benchmark card.
    """
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    try:
        results = [
            collect_episodes_parallel(env_config, n_rollout_workers=n, rollout_fragment_length=rollout_fragment_length)
            for n in worker_counts
        ]
    finally:
        ray.shutdown()
    return results
