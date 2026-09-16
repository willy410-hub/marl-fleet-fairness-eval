import numpy as np
import pytest

from env.config import EnvConfig
from evaluation.centralized_baseline import run_centralized_baseline_episode
from evaluation.episode_runner import random_action_fn, run_episode
from evaluation.metrics import DecisionQualityMetrics, EfficiencyMetrics, FairnessMetrics, RobustnessMetrics
from evaluation.robustness import build_spike_config, run_robustness_test
from env.fleet_env import FleetDispatchEnv


@pytest.fixture
def config():
    return EnvConfig(seed=1, steps_per_episode=60)


def test_centralized_baseline_beats_random_policy(config):
    baseline_totals = []
    random_totals = []
    for seed in range(5):
        baseline_result = run_centralized_baseline_episode(config, seed=seed)
        baseline_totals.append(baseline_result.total_completed)

        env = FleetDispatchEnv(config)
        trace = run_episode(env, random_action_fn, seed=seed)
        random_totals.append(trace.total_completed)

    assert np.mean(baseline_totals) > np.mean(random_totals)


def test_efficiency_metric_bounds():
    m = EfficiencyMetrics.compute(policy_completions=10, baseline_completions=20)
    assert m.completion_ratio == pytest.approx(0.5)


def test_efficiency_metric_handles_zero_baseline_without_crashing():
    m = EfficiencyMetrics.compute(policy_completions=0, baseline_completions=0)
    assert m.completion_ratio == 0.0


def test_fairness_metric_identifies_correct_low_demand_zone():
    zone_completions = np.array([10, 5, 3, 8])
    zone_rejections = np.array([2, 1, 5, 1])
    zone_offers = np.array([12, 6, 10, 9])
    m = FairnessMetrics.compute(zone_completions, zone_rejections, zone_offers)
    assert m.rejection_rate_low_demand_zone == pytest.approx(0.5)


def test_decision_quality_positive_correlation_when_value_sensitive():
    accepted = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    payouts = np.array([20, 18, 5, 22, 6, 4, 25, 3])
    m = DecisionQualityMetrics.compute(accepted, payouts)
    assert m.value_sensitivity_correlation > 0.5


def test_decision_quality_handles_no_variance_without_crashing():
    m = DecisionQualityMetrics.compute(np.array([1, 1, 1]), np.array([5, 5, 5]))
    assert m.value_sensitivity_correlation == 0.0


def test_spike_config_scales_demand_profile_correctly(config):
    spike_config = build_spike_config(config, spike_multiplier=2.0)
    for base, spiked in zip(config.demand_profile, spike_config.demand_profile):
        assert spiked == pytest.approx(base * 2.0, abs=1e-2)


def test_robustness_runs_without_crashing(config):
    result = run_robustness_test(config, random_action_fn, seed=1, spike_multiplier=2.0)
    assert isinstance(result, RobustnessMetrics)
    assert result.normal_completions >= 0
    assert result.spike_completions >= 0
