"""
Automated evaluation / QA layer (Phase 2, addition 1).

The original pipeline only had *human* evaluation: a Streamlit rating
app feeding Krippendorff's Alpha / Cohen's Kappa
(`evaluation/agreement_stats.py`). That pipeline is valuable but slow
and expensive to run on every training iteration -- exactly the kind
of thing a real "AI-driven evaluation and QA system" is meant to sit
in front of: automated grading, validation, and a feedback loop that
raises a flag *before* a run is ever queued for human review, not a
human reading raw logs to notice a regression.

This module is a literal, working quality gate: it watches a small
set of automatically-computable run-health signals (the design
document's own rejection-rate metric is the primary one) against
documented thresholds, and returns a `QualityGateReport` that is
either "cleared for human review" or "flagged" with the specific
violated checks spelled out -- the same `meets_threshold` pattern
already used for the human-agreement gate in
`evaluation/agreement_stats.py`, applied one stage earlier in the
pipeline.

It is deliberately NOT a replacement for the human annotation
pipeline -- it is the automated pre-filter described in the design
document's "Decision-quality flag" evaluation criterion, made
concrete and runnable.
"""
from dataclasses import dataclass, field

import numpy as np

from evaluation.episode_runner import EpisodeTrace
from evaluation.metrics import FairnessMetrics

# --- Documented thresholds -------------------------------------------------
# Each threshold is a run-health signal that is cheap to compute from
# data the evaluation pipeline already collects (evaluation/metrics.py,
# evaluation/episode_runner.py) -- no new instrumentation, no human
# input required.

# A policy that rejects more than this fraction of the offers it saw
# is very likely degenerate (idling instead of dispatching, or a
# training run that regressed) and should not consume a human rater's
# time until it is investigated.
REJECTION_RATE_THRESHOLD = 0.40

# Gini coefficient over zone completions above this level indicates
# the policy has collapsed toward serving only a subset of the city --
# the exact failure mode this whole project exists to catch.
ZONE_GINI_THRESHOLD = 0.50

# A policy that fails to complete a bare minimum share of the orders a
# random policy would complete is not worth annotating yet.
MIN_COMPLETION_RATIO_VS_RANDOM = 0.5


@dataclass
class QualityCheck:
    """One individual automated check: name, pass/fail, the measured value, and its threshold."""

    name: str
    passed: bool
    value: float
    threshold: float
    description: str


@dataclass
class QualityGateReport:
    """
    Automated QA verdict for one training run / evaluation batch,
    computed BEFORE the run is eligible for human annotation.
    """

    run_id: str
    passed: bool
    checks: list[QualityCheck] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        status = "PASSED -- cleared for human review" if self.passed else "FLAGGED -- held back from human review"
        lines = [f"Quality gate [{self.run_id}]: {status}"]
        for check in self.checks:
            mark = "OK " if check.passed else "FAIL"
            lines.append(f"  [{mark}] {check.name}: {check.value:.4f} (threshold: {check.threshold:.4f})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "passed": self.passed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "value": c.value,
                    "threshold": c.threshold,
                    "description": c.description,
                }
                for c in self.checks
            ],
        }


def compute_global_rejection_rate(traces: list[EpisodeTrace]) -> float:
    """
    Fraction of all accept/reject decisions across `traces` that were
    rejections. Distinct from `FairnessMetrics.rejection_rate_low_demand_zone`
    (which is scoped to the single worst-served zone) -- this is the
    fleet-wide rate the QA gate actually watches, since a fleet-wide
    regression is the more general failure signal to auto-flag on.
    """
    all_flags: list[int] = []
    for trace in traces:
        all_flags.extend(trace.accepted_flags)
    if not all_flags:
        return 0.0
    accepted = np.array(all_flags)
    return float(np.mean(accepted == 0))


def run_quality_gate(
    run_id: str,
    traces: list[EpisodeTrace],
    baseline_completions: int,
    rejection_rate_threshold: float = REJECTION_RATE_THRESHOLD,
    zone_gini_threshold: float = ZONE_GINI_THRESHOLD,
    min_completion_ratio: float = MIN_COMPLETION_RATIO_VS_RANDOM,
) -> QualityGateReport:
    """
    Run every automated check against a batch of episode traces and
    return a QualityGateReport. This is the function a training loop
    or CI job calls after every evaluation batch, before anything is
    queued into `annotation_app/`.
    """
    rejection_rate = compute_global_rejection_rate(traces)
    zone_completions = np.sum([t.zone_completion_counts for t in traces], axis=0)
    zone_rejections = np.sum([t.zone_rejection_counts for t in traces], axis=0)
    zone_offers = np.sum([t.zone_offer_counts for t in traces], axis=0)
    fairness = FairnessMetrics.compute(zone_completions, zone_rejections, zone_offers)
    total_completions = int(np.sum([t.total_completed for t in traces]))
    completion_ratio = total_completions / baseline_completions if baseline_completions > 0 else 0.0

    checks = [
        QualityCheck(
            name="rejection_rate",
            passed=rejection_rate <= rejection_rate_threshold,
            value=rejection_rate,
            threshold=rejection_rate_threshold,
            description="Fleet-wide fraction of offers rejected -- a spike usually means the policy "
            "is idling instead of dispatching (training regression or degenerate policy).",
        ),
        QualityCheck(
            name="zone_gini",
            passed=fairness.zone_gini <= zone_gini_threshold,
            value=fairness.zone_gini,
            threshold=zone_gini_threshold,
            description="Gini coefficient over zone completion counts -- a spike means coverage has "
            "collapsed onto a subset of the city, the primary failure mode this project targets.",
        ),
        QualityCheck(
            name="completion_ratio_vs_baseline",
            passed=completion_ratio >= min_completion_ratio,
            value=completion_ratio,
            threshold=min_completion_ratio,
            description="Policy completions as a fraction of the centralized-optimal baseline -- "
            "too low means the run is not yet worth a human rater's time.",
        ),
    ]

    return QualityGateReport(run_id=run_id, passed=all(c.passed for c in checks), checks=checks)
