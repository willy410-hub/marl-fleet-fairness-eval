"""
Statistical aggregation for the benchmark suite (Phase 2, addition 7):
turns a single-seed point estimate into a mean +/- confidence interval
across independent replicates, and defines a *crisp*, documented
pass/fail rule on top of it -- addressing interview feedback that
evaluation thresholds and pass/fail aggregation should be more
statistically rigorous than one run's raw number.

`benchmark/run_benchmark.py` (Phase 2, addition 4) already runs each
fixed task for `n_episodes` episodes under one `base_seed` -- that
controls for within-task episode-to-episode noise, but not for the
"what if this seed range happened to be favorable/unfavorable"
question, since every episode in a task shares the same base_seed
range. `run_benchmark_replicated.py` (next to this module) re-runs a
task several times under independent, non-overlapping seed ranges and
this module aggregates the resulting per-replicate scores.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ConfidenceInterval:
    """Mean and two-sided confidence interval over a set of replicate values."""

    mean: float
    std: float
    n: int
    confidence: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "n": self.n,
            "confidence": self.confidence,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }

    def __str__(self) -> str:
        pct = int(round(self.confidence * 100))
        return f"{self.mean:.4f} (n={self.n}, {pct}% CI [{self.ci_low:.4f}, {self.ci_high:.4f}])"


def confidence_interval(values, confidence: float = 0.95) -> ConfidenceInterval:
    """
    Two-sided (1 - confidence) Student's t confidence interval for the
    mean of `values`. Uses the t distribution rather than a normal
    approximation because benchmark replicate counts are typically
    small (n < 30), where the t distribution's heavier tails give an
    honestly wider -- not artificially tight -- interval.

    With fewer than 2 values, std/width are undefined; returns a
    zero-width interval at the sample mean rather than raising, so a
    caller can still record *some* result for an accidental
    single-replicate run (and should treat n=1 as "not yet
    statistically meaningful," which `crisp_pass` below accounts for
    by refusing to pass with n < 2).
    """
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    mean = float(np.mean(values)) if n > 0 else 0.0

    if n < 2:
        return ConfidenceInterval(mean=mean, std=0.0, n=n, confidence=confidence, ci_low=mean, ci_high=mean)

    std = float(np.std(values, ddof=1))
    sem = std / np.sqrt(n)
    t_critical = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    margin = t_critical * sem
    return ConfidenceInterval(mean=mean, std=std, n=n, confidence=confidence, ci_low=mean - margin, ci_high=mean + margin)


def crisp_pass(ci: ConfidenceInterval, threshold: float, min_replicates: int = 3) -> bool:
    """
    The documented pass/fail aggregation rule: a metric "crisply"
    clears `threshold` only if its confidence interval's *lower* bound
    is at or above the threshold (not just its point estimate), and
    only once at least `min_replicates` independent replicates back
    it up. This is deliberately conservative -- a policy that clears
    the threshold on average but with a wide, threshold-straddling CI
    is reported as NOT crisply passing, which is the whole point of
    aggregating replicates instead of trusting one seed's number.
    """
    return bool(ci.n >= min_replicates and ci.ci_low >= threshold)
