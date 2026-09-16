import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from evaluation.agreement_stats import (
    KRIPPENDORFF_ALPHA_THRESHOLD,
    build_agreement_report,
    compute_krippendorff_alpha,
    compute_pairwise_cohen_kappa,
)

# Canonical reference dataset: Krippendorff (2013) / Krippendorff (2004,
# section 11.3.3), 4 coders x 12 units, nominal values 1-5 with missing
# data. Reproduced independently in Krippendorff's own KALPHA macro
# documentation (afhayes.com/public/kalpha.pdf) and in the R package
# `krippendorffsalpha`'s vignette (arxiv.org/pdf/2103.12170), where it
# yields nominal alpha=0.743, ordinal alpha=0.815, interval alpha=0.849.
# This is the standard textbook validation case for any Krippendorff's
# Alpha implementation.
KRIPPENDORFF_REFERENCE_DATA = np.array(
    [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
        [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan],
    ],
    dtype=float,
)


def test_krippendorff_alpha_matches_canonical_reference_nominal():
    alpha = compute_krippendorff_alpha(KRIPPENDORFF_REFERENCE_DATA, level="nominal")
    assert alpha == pytest.approx(0.743, abs=0.001)


def test_krippendorff_alpha_matches_canonical_reference_ordinal():
    alpha = compute_krippendorff_alpha(KRIPPENDORFF_REFERENCE_DATA, level="ordinal")
    assert alpha == pytest.approx(0.815, abs=0.001)


def test_krippendorff_alpha_matches_canonical_reference_interval():
    alpha = compute_krippendorff_alpha(KRIPPENDORFF_REFERENCE_DATA, level="interval")
    assert alpha == pytest.approx(0.849, abs=0.001)


def test_krippendorff_alpha_perfect_agreement_is_one():
    perfect = np.array([[1, 2, 3, 4, 5]] * 3, dtype=float)
    assert compute_krippendorff_alpha(perfect) == pytest.approx(1.0, abs=1e-6)


def test_krippendorff_alpha_random_ratings_near_zero():
    rng = np.random.default_rng(0)
    random_ratings = rng.integers(1, 6, size=(3, 200)).astype(float)
    alpha = compute_krippendorff_alpha(random_ratings)
    assert abs(alpha) < 0.15


def test_cohen_kappa_matches_direct_sklearn_call():
    r1 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], dtype=float)
    r2 = np.array([1, 2, 3, 1, 2, 2, 1, 3, 3, 1], dtype=float)
    expected = cohen_kappa_score(r1, r2)

    result = compute_pairwise_cohen_kappa(np.vstack([r1, r2]), ["a", "b"])
    assert result["a_vs_b"] == pytest.approx(expected, abs=1e-9)


def test_cohen_kappa_handles_insufficient_overlap():
    r1 = np.array([1, np.nan, np.nan], dtype=float)
    r2 = np.array([1, 2, np.nan], dtype=float)
    result = compute_pairwise_cohen_kappa(np.vstack([r1, r2]), ["a", "b"])
    assert result["a_vs_b"] is None


def test_agreement_report_meets_threshold_flag():
    high_agreement = np.array(
        [
            [3, 4, 2, 5, 1, 3, 4, 2],
            [3, 4, 3, 5, 1, 3, 4, 2],
            [3, 4, 2, 5, 2, 3, 4, 2],
        ],
        dtype=float,
    )
    report = build_agreement_report(high_agreement, ["r1", "r2", "r3"])
    assert report.krippendorff_alpha >= KRIPPENDORFF_ALPHA_THRESHOLD
    assert report.meets_threshold is True
    assert report.n_items == 8
    assert report.n_raters == 3
    assert len(report.pairwise_cohen_kappa) == 3


def test_agreement_report_flags_low_agreement():
    rng = np.random.default_rng(1)
    low_agreement = rng.integers(1, 6, size=(3, 50)).astype(float)
    report = build_agreement_report(low_agreement, ["r1", "r2", "r3"])
    assert report.meets_threshold is False
