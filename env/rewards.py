"""
Reward architecture: r = r_ext + omega * r_int, per the design document's
"Intrinsic Motivation + Extrinsic Rewards" specification.

This module is the direct implementation of Section 3 of the design
doc (marl_delivery_lite.md). Every term below is traceable to a named
line in that document; see the docstring on each function for the
exact mapping.
"""
from dataclasses import dataclass

import numpy as np

from env.config import EnvConfig
from env.fairness_metrics import zone_coverage_variance


@dataclass
class RewardBreakdown:
    """Every term contributing to one agent's reward at one step, kept separate for auditability."""

    extrinsic: float
    intrinsic_fairness: float
    cherry_picking_penalty: float
    abandonment_penalty: float
    unsafe_shortcut_penalty: float

    @property
    def total(self) -> float:
        return (
            self.extrinsic
            + self.intrinsic_fairness
            - self.cherry_picking_penalty
            - self.abandonment_penalty
            - self.unsafe_shortcut_penalty
        )


def extrinsic_reward_on_delivery(
    payout: float, delivery_time_steps: int, distance_traveled: int, config: EnvConfig
) -> float:
    """
    r_ext = payout - (time_penalty * delivery_time) - (fuel_cost * distance)

    Design doc quote: "Extrinsic (profit-driven), per delivery:
    r_ext = payout - (time_penalty * delivery_time) - (fuel_cost * distance)".
    Applied once, at the step a delivery completes.
    """
    return (
        payout
        - config.time_penalty_per_step * delivery_time_steps
        - config.fuel_cost_per_cell * distance_traveled
    )


def intrinsic_fairness_reward(zone_completion_counts: np.ndarray, config: EnvConfig) -> float:
    """
    r_int = coverage_bonus * (1 - zone_coverage_variance)

    Design doc quote: "Intrinsic (fairness-driven), per step:
    r_int = coverage_bonus * (1 - zone_coverage_variance)". Applied
    every step (not just on delivery), using the fleet-wide zone
    completion distribution so far this episode.
    """
    variance = zone_coverage_variance(zone_completion_counts)
    return config.coverage_bonus_scale * (1.0 - variance)


def combine_reward(extrinsic: float, intrinsic: float, config: EnvConfig) -> float:
    """
    r = r_ext + omega * r_int

    Design doc quote: "Total: r = r_ext + omega * r_int (omega tunes
    how much fairness matters vs. raw profit)". config.fairness_weight
    is omega.
    """
    return extrinsic + config.fairness_weight * intrinsic


def cherry_picking_penalty(
    rejected_payout: float, mean_visible_payout: float, was_idle: bool, config: EnvConfig
) -> float:
    """
    Penalizes rejecting a below-average-value offer while otherwise idle
    nearby -- the design doc's "cherry-picking only high-value orders"
    penalty. Only triggers when the agent (a) was idle (had capacity to
    take the order) and (b) rejected an offer priced below the mean of
    what was visible to it at that step -- i.e. passed on a
    below-average opportunity while having nothing better queued.
    """
    if not was_idle:
        return 0.0
    if rejected_payout >= mean_visible_payout:
        return 0.0
    return config.cherry_picking_penalty


def abandonment_penalty(zone_completion_counts: np.ndarray, config: EnvConfig) -> float:
    """
    Penalizes leaving a zone essentially unserved while others are
    well-served -- the design doc's "abandoning low-demand zones"
    penalty. Fires when at least one zone has zero completions so far
    this episode while the fleet-wide total is already meaningfully
    non-zero (i.e. the fleet is actively serving *some* zones, just not
    all of them).
    """
    zone_completion_counts = np.asarray(zone_completion_counts, dtype=np.float64)
    total = zone_completion_counts.sum()
    if total < zone_completion_counts.size:  # not enough completions yet to judge abandonment
        return 0.0
    if np.any(zone_completion_counts == 0):
        return config.abandonment_penalty
    return 0.0


def unsafe_shortcut_penalty(used_shortcut: bool, config: EnvConfig) -> float:
    """
    Penalizes the MOVE_TO_ZONE_SHORTCUT action variant -- the design
    doc's "unsafe shortcuts (risk penalty)". This is a flat penalty
    applied whenever an agent chooses the risky movement action
    (env/actions.py's ActionKind.MOVE_TO_ZONE_SHORTCUT), modeling a
    real dispatch platform's incentive to discourage agents from
    choosing routes/behaviors that trade safety for speed.
    """
    return config.unsafe_shortcut_penalty if used_shortcut else 0.0
