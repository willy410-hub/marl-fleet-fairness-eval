import numpy as np
import pytest

from benchmark.statistics import ConfidenceInterval, confidence_interval, crisp_pass


def test_confidence_interval_mean_is_correct():
    ci = confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ci.mean == pytest.approx(3.0)


def test_confidence_interval_contains_the_mean():
    ci = confidence_interval([0.7, 0.75, 0.8, 0.72, 0.78, 0.81])
    assert ci.ci_low <= ci.mean <= ci.ci_high


def test_confidence_interval_widens_with_more_variance():
    tight = confidence_interval([0.80, 0.81, 0.79, 0.80, 0.80])
    wide = confidence_interval([0.5, 0.9, 0.6, 0.95, 0.55])
    assert (wide.ci_high - wide.ci_low) > (tight.ci_high - tight.ci_low)


def test_confidence_interval_narrows_with_more_replicates_at_same_variance():
    rng = np.random.default_rng(0)
    small_n = confidence_interval(rng.normal(0.8, 0.05, size=3))
    large_n = confidence_interval(rng.normal(0.8, 0.05, size=30))
    assert (large_n.ci_high - large_n.ci_low) < (small_n.ci_high - small_n.ci_low)


def test_confidence_interval_with_zero_variance_is_a_point():
    ci = confidence_interval([0.5, 0.5, 0.5, 0.5])
    assert ci.ci_low == pytest.approx(ci.mean)
    assert ci.ci_high == pytest.approx(ci.mean)


def test_confidence_interval_below_two_samples_is_degenerate_not_an_error():
    ci = confidence_interval([0.5])
    assert ci.n == 1
    assert ci.ci_low == ci.ci_high == pytest.approx(0.5)


def test_crisp_pass_requires_lower_bound_above_threshold():
    high_ci = ConfidenceInterval(mean=0.9, std=0.02, n=5, confidence=0.95, ci_low=0.85, ci_high=0.95)
    assert crisp_pass(high_ci, threshold=0.5)


def test_crisp_pass_fails_when_ci_straddles_threshold_even_if_mean_clears_it():
    straddling_ci = ConfidenceInterval(mean=0.55, std=0.2, n=5, confidence=0.95, ci_low=0.30, ci_high=0.80)
    assert not crisp_pass(straddling_ci, threshold=0.5)


def test_crisp_pass_requires_minimum_replicate_count():
    ci = ConfidenceInterval(mean=0.9, std=0.0, n=2, confidence=0.95, ci_low=0.9, ci_high=0.9)
    assert not crisp_pass(ci, threshold=0.5, min_replicates=3)
    assert crisp_pass(ci, threshold=0.5, min_replicates=2)


def test_confidence_interval_to_dict_is_json_serializable():
    import json

    ci = confidence_interval([0.1, 0.2, 0.3])
    json.dumps(ci.to_dict())
