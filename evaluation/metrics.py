"""
Evaluation metrics per Section 4 of the design document
("Evaluation Criteria & Metrics"): efficiency, fairness, decision-quality
flagging, and robustness under demand spikes.
"""
from dataclasses import dataclass

import numpy as np

from env.fairness_metrics import gini_coefficient


@dataclass
class EfficiencyMetrics:
    """Design doc: 'average delivery time vs. centralized-optimal dispatch baseline.'"""

    policy_completions: int
    baseline_completions: int
    completion_ratio: float

    @classmethod
    def compute(cls, policy_completions: int, baseline_completions: int) -> "EfficiencyMetrics":
        ratio = policy_completions / baseline_completions if baseline_completions > 0 else 0.0
        return cls(
            policy_completions=policy_completions,
            baseline_completions=baseline_completions,
            completion_ratio=ratio,
        )


@dataclass
class FairnessMetrics:
    """Design doc: 'coverage variance across zones (Gini-style), rejection rate in low-demand zones.'"""

    zone_gini: float
    rejection_rate_low_demand_zone: float
    zone_completion_counts: np.ndarray

    @classmethod
    def compute(
        cls, zone_completion_counts: np.ndarray, zone_rejection_counts: np.ndarray, zone_offer_counts: np.ndarray
    ) -> "FairnessMetrics":
        gini = gini_coefficient(zone_completion_counts)

        low_demand_zone = int(np.argmin(zone_completion_counts))
        offers_in_low_zone = zone_offer_counts[low_demand_zone]
        rejections_in_low_zone = zone_rejection_counts[low_demand_zone]
        rejection_rate = rejections_in_low_zone / offers_in_low_zone if offers_in_low_zone > 0 else 0.0

        return cls(
            zone_gini=gini,
            rejection_rate_low_demand_zone=float(rejection_rate),
            zone_completion_counts=zone_completion_counts,
        )


@dataclass
class DecisionQualityMetrics:
    """
    Design doc: "correlation between accept/reject choices and true
    order value vs. bias signals -- a proxy scoring rubric similar to
    grading real agent decisions."

    Implemented as the correlation between (a) whether an agent
    accepted an offer and (b) that offer's true payout value, computed
    over every accept/reject decision made in the episode. A
    well-calibrated policy should show a *positive* correlation
    (prefers higher-value orders, all else equal) without it being so
    strong that it indicates pure cherry-picking (which the
    cherry_picking_penalty in env/rewards.py already discourages
    during training) -- this metric is reported, not judged
    pass/fail, precisely because the "right" amount of value-sensitivity
    is a judgment call, which is exactly what the human annotation
    pipeline (annotation_app/) exists to calibrate.
    """

    value_sensitivity_correlation: float
    n_decisions: int

    @classmethod
    def compute(cls, accepted_flags: np.ndarray, offer_payouts: np.ndarray) -> "DecisionQualityMetrics":
        if len(accepted_flags) < 2 or np.std(offer_payouts) == 0 or np.std(accepted_flags) == 0:
            return cls(value_sensitivity_correlation=0.0, n_decisions=len(accepted_flags))
        corr = float(np.corrcoef(accepted_flags, offer_payouts)[0, 1])
        return cls(value_sensitivity_correlation=corr, n_decisions=len(accepted_flags))


@dataclass
class RobustnessMetrics:
    """Design doc: 'performance under demand spikes (rush hour, weather) without retraining.'"""

    normal_completions: int
    spike_completions: int
    degradation_ratio: float

    @classmethod
    def compute(cls, normal_completions: int, spike_completions: int) -> "RobustnessMetrics":
        ratio = spike_completions / normal_completions if normal_completions > 0 else 0.0
        return cls(
            normal_completions=normal_completions,
            spike_completions=spike_completions,
            degradation_ratio=ratio,
        )
