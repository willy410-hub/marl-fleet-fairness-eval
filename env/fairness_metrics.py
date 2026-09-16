"""
Fairness/coverage metrics shared by the reward function (env/rewards.py)
and the offline evaluation suite (evaluation/fairness.py).

Defined once, here, so the *training-time* fairness signal and the
*evaluation-time* fairness metric are mathematically the same
function -- a common bug in RL projects is for the reward shaping to
diverge from the metric actually reported in results, which silently
invalidates the evaluation.
"""
import numpy as np


def gini_coefficient(values: np.ndarray) -> float:
    """
    Standard Gini coefficient of an array of non-negative values, in [0, 1].

    0 = perfectly equal distribution across values; 1 = maximally
    unequal (all mass concentrated in one value). Used here over
    zone-level order-completion counts as the headline fairness
    metric requested in the design doc ("Gini-style" coverage variance).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if np.all(values == 0):
        return 0.0  # no orders served anywhere -- define as "perfectly equal" (all zero), not undefined

    sorted_vals = np.sort(values)
    n = sorted_vals.size
    cumulative = np.cumsum(sorted_vals)
    # Standard Gini formula via the Lorenz curve trapezoid method.
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_vals) - (n + 1) * cumulative[-1]) / (
        n * cumulative[-1]
    )
    return float(np.clip(gini, 0.0, 1.0))


def zone_coverage_variance(zone_counts: np.ndarray) -> float:
    """
    Normalized variance of per-zone order-completion counts, in [0, 1].

    Used as the per-step intrinsic fairness signal (cheaper to compute
    every step than a full Gini sort): 0 means every zone has received
    an identical share of completions so far; 1 is the theoretical
    maximum for the given zone count when all completions are
    concentrated in a single zone.
    """
    zone_counts = np.asarray(zone_counts, dtype=np.float64)
    n_zones = zone_counts.size
    if n_zones <= 1 or zone_counts.sum() == 0:
        return 0.0

    shares = zone_counts / zone_counts.sum()
    uniform_share = 1.0 / n_zones
    variance = np.sum((shares - uniform_share) ** 2) / n_zones
    # Max possible variance (one zone gets everything, rest get zero):
    max_variance = ((1 - uniform_share) ** 2 + (n_zones - 1) * uniform_share ** 2) / n_zones
    return float(np.clip(variance / max_variance, 0.0, 1.0)) if max_variance > 0 else 0.0
