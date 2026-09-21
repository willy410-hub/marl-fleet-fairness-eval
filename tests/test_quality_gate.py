import numpy as np
import pytest

from env.config import EnvConfig
from env.fleet_env import FleetDispatchEnv
from evaluation.centralized_baseline import run_centralized_baseline_episode
from evaluation.episode_runner import EpisodeTrace, random_action_fn, run_episode
from evaluation.quality_gate import (
    REJECTION_RATE_THRESHOLD,
    ZONE_GINI_THRESHOLD,
    compute_global_rejection_rate,
    run_quality_gate,
)


def _make_trace(accepted_flags, zone_completions, zone_rejections, zone_offers, total_completed=None):
    zone_completions = np.array(zone_completions, dtype=np.int64)
    zone_rejections = np.array(zone_rejections, dtype=np.int64)
    zone_offers = np.array(zone_offers, dtype=np.int64)
    return EpisodeTrace(
        total_completed=total_completed if total_completed is not None else int(zone_completions.sum()),
        zone_completion_counts=zone_completions,
        zone_offer_counts=zone_offers,
        zone_rejection_counts=zone_rejections,
        accepted_flags=list(accepted_flags),
        offer_payouts=[10.0] * len(accepted_flags),
        total_reward=0.0,
    )


def test_compute_global_rejection_rate_basic():
    trace = _make_trace(
        accepted_flags=[1, 1, 0, 0, 1],
        zone_completions=[3, 0, 0, 0],
        zone_rejections=[0, 0, 0, 0],
        zone_offers=[0, 0, 0, 0],
    )
    assert compute_global_rejection_rate([trace]) == pytest.approx(2 / 5)


def test_compute_global_rejection_rate_aggregates_across_traces():
    trace_a = _make_trace([1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
    trace_b = _make_trace([0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
    # 3 rejections out of 4 total decisions
    assert compute_global_rejection_rate([trace_a, trace_b]) == pytest.approx(3 / 4)


def test_compute_global_rejection_rate_empty_is_zero():
    assert compute_global_rejection_rate([]) == 0.0


def test_quality_gate_passes_for_healthy_run():
    # Low rejection rate, evenly-covered zones, completions well above baseline.
    trace = _make_trace(
        accepted_flags=[1] * 18 + [0] * 2,  # 10% rejection rate
        zone_completions=[10, 9, 11, 10],  # near-perfect balance -> low gini
        zone_rejections=[1, 0, 1, 0],
        zone_offers=[10, 10, 10, 10],
        total_completed=40,
    )
    report = run_quality_gate(run_id="healthy", traces=[trace], baseline_completions=40)
    assert report.passed is True
    assert report.failed_checks == []
    assert "PASSED" in report.summary()


def test_quality_gate_flags_high_rejection_rate():
    trace = _make_trace(
        accepted_flags=[0] * 8 + [1] * 2,  # 80% rejection rate, well above threshold
        zone_completions=[5, 5, 5, 5],
        zone_rejections=[2, 2, 2, 2],
        zone_offers=[10, 10, 10, 10],
        total_completed=20,
    )
    report = run_quality_gate(run_id="degenerate", traces=[trace], baseline_completions=40)
    assert report.passed is False
    failed_names = {c.name for c in report.failed_checks}
    assert "rejection_rate" in failed_names
    rejection_check = next(c for c in report.checks if c.name == "rejection_rate")
    assert rejection_check.value > REJECTION_RATE_THRESHOLD
    assert "FLAGGED" in report.summary()


def test_quality_gate_flags_unfair_zone_collapse():
    trace = _make_trace(
        accepted_flags=[1] * 20,
        zone_completions=[40, 0, 0, 0],  # all completions in one zone -> high gini
        zone_rejections=[0, 0, 0, 0],
        zone_offers=[10, 10, 10, 10],
        total_completed=40,
    )
    report = run_quality_gate(run_id="collapsed", traces=[trace], baseline_completions=40)
    assert report.passed is False
    gini_check = next(c for c in report.checks if c.name == "zone_gini")
    assert gini_check.passed is False
    assert gini_check.value > ZONE_GINI_THRESHOLD


def test_quality_gate_flags_low_completion_ratio_vs_baseline():
    trace = _make_trace(
        accepted_flags=[1] * 5,
        zone_completions=[2, 1, 1, 1],
        zone_rejections=[0, 0, 0, 0],
        zone_offers=[5, 5, 5, 5],
        total_completed=5,
    )
    report = run_quality_gate(run_id="underperforming", traces=[trace], baseline_completions=100)
    completion_check = next(c for c in report.checks if c.name == "completion_ratio_vs_baseline")
    assert completion_check.passed is False
    assert report.passed is False


def test_quality_gate_report_to_dict_round_trips_key_fields():
    trace = _make_trace([1, 1, 0], [3, 2, 2, 2], [1, 0, 0, 0], [4, 4, 4, 4], total_completed=9)
    report = run_quality_gate(run_id="serialization-check", traces=[trace], baseline_completions=9)
    payload = report.to_dict()
    assert payload["run_id"] == "serialization-check"
    assert isinstance(payload["passed"], bool)
    assert len(payload["checks"]) == 3
    assert {c["name"] for c in payload["checks"]} == {
        "rejection_rate",
        "zone_gini",
        "completion_ratio_vs_baseline",
    }


def test_quality_gate_integration_with_real_episode_traces():
    """
    End-to-end: the random policy is expected to be noisy but not
    degenerate over a short episode, and the quality gate should run
    against real EpisodeTrace / baseline data without any adaptation.
    """
    config = EnvConfig(seed=7, steps_per_episode=40)
    traces = []
    baseline_completions = 0
    for seed in range(3):
        env = FleetDispatchEnv(config)
        traces.append(run_episode(env, random_action_fn, seed=seed))
        baseline_result = run_centralized_baseline_episode(config, seed=seed)
        baseline_completions += baseline_result.total_completed

    report = run_quality_gate(run_id="integration-check", traces=traces, baseline_completions=baseline_completions)
    assert report.run_id == "integration-check"
    assert len(report.checks) == 3
    assert isinstance(report.passed, bool)
