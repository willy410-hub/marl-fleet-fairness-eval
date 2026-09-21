import pytest

from evaluation.parallel_rollout import ParallelSampleResult


def test_parallel_sample_result_episodes_per_second_computed_correctly():
    result = ParallelSampleResult(
        n_rollout_workers=2,
        wall_clock_seconds=2.0,
        total_episodes=10,
        total_env_steps=400,
        episodes_per_second=5.0,
    )
    assert result.episodes_per_second == pytest.approx(result.total_episodes / result.wall_clock_seconds)


@pytest.mark.ray
def test_more_rollout_workers_collect_more_episodes_in_parallel():
    """
    Real, Ray-backed integration test: not mocked, because the entire
    point of this module is a genuine wall-clock throughput
    measurement against RLlib's actual distributed rollout-worker
    pool. Excluded from the default `pytest tests/` run (see
    pytest.ini) since it spins up a real Ray instance (~5-10s) --
    run explicitly with `pytest tests/ -m ray`.
    """
    from env.config import EnvConfig
    from evaluation.parallel_rollout import run_scaling_comparison

    config = EnvConfig(steps_per_episode=30)
    results = run_scaling_comparison(config, worker_counts=[1, 2], rollout_fragment_length=30)

    one_worker, two_workers = results
    assert one_worker.n_rollout_workers == 1
    assert two_workers.n_rollout_workers == 2
    # 1 worker collects exactly 1 episode per sample() call, 2 workers collect 2.
    assert one_worker.total_episodes == 1
    assert two_workers.total_episodes == 2
    assert two_workers.total_env_steps == 2 * one_worker.total_env_steps
