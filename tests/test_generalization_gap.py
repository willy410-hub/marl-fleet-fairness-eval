import pytest

from env.config import EnvConfig
from evaluation.episode_runner import random_action_fn
from evaluation.generalization_gap import classify_generalization_gap, run_generalization_check


def test_classify_flags_a_large_gap():
    mean_synth, std_synth, gap, flagged = classify_generalization_gap(
        base_ratio=0.90, synthetic_ratios=[0.60, 0.62, 0.58, 0.61], threshold=0.15
    )
    assert mean_synth == pytest.approx(0.6025, abs=1e-4)
    assert gap == pytest.approx(0.90 - 0.6025, abs=1e-4)
    assert flagged is True


def test_classify_does_not_flag_a_small_gap():
    mean_synth, std_synth, gap, flagged = classify_generalization_gap(
        base_ratio=0.70, synthetic_ratios=[0.68, 0.71, 0.69, 0.72], threshold=0.15
    )
    assert flagged is False


def test_classify_handles_synthetic_scoring_higher_than_base():
    """A negative gap (synthetic scenarios scored *better*) must never be flagged --
    the check is one-sided by design (it targets overfitting, not 'got lucky')."""
    mean_synth, std_synth, gap, flagged = classify_generalization_gap(
        base_ratio=0.60, synthetic_ratios=[0.80, 0.82], threshold=0.15
    )
    assert gap < 0
    assert flagged is False


def test_run_generalization_check_end_to_end_with_random_policy():
    """
    Full integration smoke test: the random policy has no notion of
    'the training distribution' at all, so it should show a small,
    unflagged generalization gap (its performance is governed by luck
    and task difficulty, not overfitting) -- this exercises the real
    evaluate_policy + synthetic-scenario-batch pipeline end to end,
    not just the pure classification function above.
    """
    config = EnvConfig(steps_per_episode=25, n_agents=3)
    report = run_generalization_check(
        random_action_fn,
        base_config=config,
        n_synthetic_scenarios=3,
        synthetic_seed=7,
        n_episodes_per_scenario=2,
        base_seed=1234,
    )
    assert len(report.synthetic_completion_ratios) == 3
    assert len(report.scenario_ids) == 3
    assert 0.0 <= report.base_completion_ratio
    assert isinstance(report.flagged, bool)


def test_report_to_dict_is_json_serializable():
    import json

    config = EnvConfig(steps_per_episode=20, n_agents=3)
    report = run_generalization_check(
        random_action_fn,
        base_config=config,
        n_synthetic_scenarios=2,
        synthetic_seed=1,
        n_episodes_per_scenario=1,
        base_seed=555,
    )
    json.dumps(report.to_dict())
