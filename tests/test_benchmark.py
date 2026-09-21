import pytest

from benchmark.tasks import TASKS, get_task
from benchmark.run_benchmark import run_full_benchmark, run_task
from evaluation.episode_runner import random_action_fn


def test_all_fixed_tasks_are_present():
    assert set(TASKS) == {"standard-v1", "surge-v1", "short-horizon-v1"}


def test_get_task_returns_the_matching_task():
    task = get_task("standard-v1")
    assert task.task_id == "standard-v1"


def test_get_task_raises_on_unknown_id():
    with pytest.raises(KeyError):
        get_task("does-not-exist")


def test_surge_task_has_a_higher_order_spawn_rate_than_standard():
    standard = get_task("standard-v1")
    surge = get_task("surge-v1")
    assert surge.env_config.order_spawn_rate > standard.env_config.order_spawn_rate


def test_short_horizon_task_has_a_shorter_episode_than_standard():
    standard = get_task("standard-v1")
    short = get_task("short-horizon-v1")
    assert short.env_config.steps_per_episode < standard.env_config.steps_per_episode


def test_each_task_has_a_unique_base_seed():
    seeds = [task.base_seed for task in TASKS.values()]
    assert len(seeds) == len(set(seeds))


def test_run_task_produces_a_scored_result_with_quality_gate():
    task = get_task("short-horizon-v1")
    # Keep the smoke test cheap: override to 2 episodes rather than the full fixed task size.
    import dataclasses

    small_task = dataclasses.replace(task, n_episodes=2)
    score = run_task(small_task, random_action_fn)

    assert score.task_id == "short-horizon-v1"
    assert score.evaluation.n_episodes == 2
    assert score.quality_gate.run_id == "short-horizon-v1"
    assert isinstance(score.quality_gate.passed, bool)


def test_run_full_benchmark_runs_every_requested_task():
    import dataclasses

    from benchmark.tasks import TASKS as ALL_TASKS

    subset_ids = ["short-horizon-v1"]
    original = ALL_TASKS[subset_ids[0]]
    ALL_TASKS[subset_ids[0]] = dataclasses.replace(original, n_episodes=2)
    try:
        scores = run_full_benchmark(random_action_fn, task_ids=subset_ids)
    finally:
        ALL_TASKS[subset_ids[0]] = original

    assert set(scores) == set(subset_ids)
