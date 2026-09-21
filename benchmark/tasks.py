"""
Fixed benchmark task configurations (Phase 2, addition 4).

Turns this project from a personal experiment into a reusable
benchmark suite: instead of every evaluation run picking its own
ad-hoc `EnvConfig`, this module defines a small, fixed, versioned set
of named tasks that any policy -- this project's own, or someone
else's -- can be run against and get a directly comparable score.
This mirrors real benchmark-release practice (e.g. a fixed task suite
with a pinned config + seeds, so "task X score" means the same thing
across different runs and different people).

Each task is deliberately just an `EnvConfig` plus a fixed evaluation
seed range -- nothing benchmark-specific needs to exist in `env/` or
`evaluation/` for this to work, so the benchmark package is a thin,
additive layer on top of the existing environment and evaluation
code, not a fork of it.
"""
from dataclasses import dataclass

from env.config import EnvConfig
from training.scenarios import build_surge_scenario_config


@dataclass(frozen=True)
class BenchmarkTask:
    """One fixed, versioned benchmark task: an environment config plus a fixed seed range."""

    task_id: str
    description: str
    env_config: EnvConfig
    n_episodes: int
    base_seed: int


_STANDARD_CONFIG = EnvConfig()
_SURGE_CONFIG = build_surge_scenario_config(_STANDARD_CONFIG)
_SHORT_HORIZON_CONFIG = EnvConfig(steps_per_episode=60)

# The fixed task suite. Adding a task is additive (append here); the
# task_id is the benchmark's stable public identifier and must never
# be reused for a different config once published -- see
# BENCHMARK_CARD.md's "Versioning" section.
TASKS: dict[str, BenchmarkTask] = {
    "standard-v1": BenchmarkTask(
        task_id="standard-v1",
        description=(
            "Default 10x10 grid, 8 agents, 200-step episodes, TLC-calibrated demand at the "
            "moderate baseline volume (order_spawn_rate=0.35) that training/train_ppo.py trains "
            "against by default. The primary, headline benchmark task."
        ),
        env_config=_STANDARD_CONFIG,
        n_episodes=10,
        base_seed=5000,
    ),
    "surge-v1": BenchmarkTask(
        task_id="surge-v1",
        description=(
            "Same grid/agents/episode length as standard-v1, but under the sustained-surge demand "
            "scenario from training/scenarios.py (order_spawn_rate=0.60, ~70% higher order volume). "
            "Tests whether a policy generalizes to materially higher demand without retraining, or "
            "whether it needs the fine-tuning workflow in training/finetune_ppo.py."
        ),
        env_config=_SURGE_CONFIG,
        n_episodes=10,
        base_seed=6000,
    ),
    "short-horizon-v1": BenchmarkTask(
        task_id="short-horizon-v1",
        description=(
            "Same standard demand, but a 60-step (rather than 200-step) episode horizon -- a "
            "cheaper, faster task for quick iteration/smoke-testing a policy before running the "
            "full standard-v1 task."
        ),
        env_config=_SHORT_HORIZON_CONFIG,
        n_episodes=10,
        base_seed=7000,
    ),
}


def get_task(task_id: str) -> BenchmarkTask:
    if task_id not in TASKS:
        raise KeyError(f"Unknown benchmark task '{task_id}'. Known tasks: {sorted(TASKS)}")
    return TASKS[task_id]
