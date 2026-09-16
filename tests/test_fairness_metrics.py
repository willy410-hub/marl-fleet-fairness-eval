import numpy as np
import pytest

from env.fairness_metrics import gini_coefficient, zone_coverage_variance


def test_gini_perfect_equality_is_zero():
    assert gini_coefficient(np.array([10, 10, 10, 10])) == pytest.approx(0.0, abs=1e-9)


def test_gini_matches_independent_mean_absolute_difference_formula():
    def gini_alt(x):
        x = np.sort(np.asarray(x, dtype=np.float64))
        n = len(x)
        total_abs_diff = np.sum(np.abs(x[:, None] - x[None, :]))
        return total_abs_diff / (2 * n * np.sum(x))

    values = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 10])
    assert gini_coefficient(values) == pytest.approx(gini_alt(values), abs=1e-9)


def test_gini_handles_empty_and_all_zero_without_crashing():
    assert gini_coefficient(np.array([])) == 0.0
    assert gini_coefficient(np.array([0, 0, 0])) == 0.0


def test_gini_is_bounded_zero_to_one():
    rng = np.random.default_rng(0)
    for _ in range(20):
        values = rng.integers(0, 100, size=10)
        g = gini_coefficient(values)
        assert 0.0 <= g <= 1.0


def test_zone_coverage_variance_perfect_coverage_is_zero():
    assert zone_coverage_variance(np.array([25, 25, 25, 25])) == pytest.approx(0.0, abs=1e-9)


def test_zone_coverage_variance_max_inequality_is_one():
    # By construction, this is the normalization anchor: one zone has
    # everything, the rest have nothing -- this MUST equal exactly 1.0.
    assert zone_coverage_variance(np.array([0, 0, 0, 100])) == pytest.approx(1.0, abs=1e-9)


def test_zone_coverage_variance_handles_all_zero():
    assert zone_coverage_variance(np.array([0, 0, 0, 0])) == 0.0


def test_zone_coverage_variance_is_bounded_zero_to_one():
    rng = np.random.default_rng(1)
    for _ in range(20):
        counts = rng.integers(0, 50, size=4)
        v = zone_coverage_variance(counts)
        assert 0.0 <= v <= 1.0
