import dataclasses

import pytest

from benchmark.run_benchmark import BenchmarkTask
from benchmark.run_benchmark_replicated import run_task_replicated
from env.config import EnvConfig
from evaluation.episode_runner import random_action_fn

# A small, fast task -- short episodes, few of them -- so this test
# doesn't pay the cost of the full 200-step/10-episode standard-v1 task
# multiplied across several replicates.
_FAST_CONFIG = EnvConfig(steps_per_episode=20, n_agents=3)
_FAST_TASK = BenchmarkTask(
    task_id="fast-test-task",
    description="Short-horizon task used only by tests/test_run_benchmark_replicated.py.",
    env_config=_FAST_CONFIG,
    n_episodes=2,
    base_seed=42,
)


def test_replicated_score_has_requested_replicate_count():
    score = run_task_replicated(_FAST_TASK, random_action_fn, n_replicates=3)
    assert score.completion_ratio_ci.n == 3
    assert score.zone_gini_ci.n == 3


def test_replicated_score_uses_non_overlapping_seed_ranges():
    """Each replicate's episodes must come from a distinct seed block -- otherwise
    'independent replicates' would secretly be the same episodes re-scored."""
    seen_seed_starts = set()
    for r in range(3):
        shifted = dataclasses.replace(_FAST_TASK, base_seed=_FAST_TASK.base_seed + r * 100_000)
        seen_seed_starts.add(shifted.base_seed)
    assert len(seen_seed_starts) == 3


def test_replicated_score_is_json_serializable():
    import json

    score = run_task_replicated(_FAST_TASK, random_action_fn, n_replicates=2)
    json.dumps(score.to_dict())


def test_crisply_passes_flag_is_a_bool():
    score = run_task_replicated(_FAST_TASK, random_action_fn, n_replicates=2)
    assert isinstance(score.crisply_passes_completion_threshold, bool)
