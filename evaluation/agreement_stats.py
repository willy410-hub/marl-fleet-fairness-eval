"""
Inter-rater agreement statistics for the human annotation pipeline,
per the design document's requirement: "3+ human raters score a
sample of agent decisions on fairness/soundness; require
Krippendorff's Alpha >= 0.7 before trusting the rubric's labels
(pairwise spot-checks use Cohen's Kappa)."
"""
from dataclasses import dataclass

import krippendorff
import numpy as np
from sklearn.metrics import cohen_kappa_score

KRIPPENDORFF_ALPHA_THRESHOLD = 0.7


@dataclass
class AgreementReport:
    """Full inter-rater agreement summary for one annotation round."""

    krippendorff_alpha: float
    meets_threshold: bool
    pairwise_cohen_kappa: dict
    n_items: int
    n_raters: int


def compute_krippendorff_alpha(ratings_matrix: np.ndarray, level: str = "ordinal") -> float:
    """
    ratings_matrix: shape (n_raters, n_items), with np.nan for any
    item a given rater did not score. `level` follows the
    krippendorff package's own convention ("nominal", "ordinal",
    "interval", "ratio") -- "ordinal" is used here because the
    annotation scale (see annotation_app/scoring_ui.py) is a 1-5
    Likert-style fairness/soundness rating, which has a meaningful
    order but not necessarily equal intervals between points.
    """
    return float(krippendorff.alpha(reliability_data=ratings_matrix, level_of_measurement=level))


def compute_pairwise_cohen_kappa(ratings_matrix: np.ndarray, rater_names: list[str]) -> dict:
    """
    Cohen's Kappa for every pair of raters, over only the items both
    raters in that pair actually scored (NaNs excluded pairwise).
    """
    n_raters = ratings_matrix.shape[0]
    results = {}
    for i in range(n_raters):
        for j in range(i + 1, n_raters):
            row_i, row_j = ratings_matrix[i], ratings_matrix[j]
            both_scored = ~np.isnan(row_i) & ~np.isnan(row_j)
            if both_scored.sum() < 2:
                results[f"{rater_names[i]}_vs_{rater_names[j]}"] = None
                continue
            kappa = cohen_kappa_score(row_i[both_scored], row_j[both_scored])
            results[f"{rater_names[i]}_vs_{rater_names[j]}"] = float(kappa)
    return results


def build_agreement_report(ratings_matrix: np.ndarray, rater_names: list[str]) -> AgreementReport:
    """
    ratings_matrix: shape (n_raters, n_items). Full agreement report
    combining Krippendorff's Alpha (overall reliability) and pairwise
    Cohen's Kappa (spot-checks between individual rater pairs), per
    the design document's dual-metric requirement.
    """
    alpha = compute_krippendorff_alpha(ratings_matrix)
    pairwise = compute_pairwise_cohen_kappa(ratings_matrix, rater_names)

    return AgreementReport(
        krippendorff_alpha=alpha,
        meets_threshold=alpha >= KRIPPENDORFF_ALPHA_THRESHOLD,
        pairwise_cohen_kappa=pairwise,
        n_items=ratings_matrix.shape[1],
        n_raters=ratings_matrix.shape[0],
    )
