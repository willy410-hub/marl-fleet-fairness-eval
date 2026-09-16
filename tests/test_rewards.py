import numpy as np
import pytest

from env.config import EnvConfig
from env.rewards import (
    RewardBreakdown,
    abandonment_penalty,
    cherry_picking_penalty,
    combine_reward,
    extrinsic_reward_on_delivery,
    intrinsic_fairness_reward,
    unsafe_shortcut_penalty,
)


@pytest.fixture
def config():
    return EnvConfig()


def test_extrinsic_reward_matches_design_doc_formula(config):
    payout, delivery_time, distance = 20.0, 5, 8
    expected = (
        payout
        - config.time_penalty_per_step * delivery_time
        - config.fuel_cost_per_cell * distance
    )
    assert extrinsic_reward_on_delivery(payout, delivery_time, distance, config) == pytest.approx(expected)


def test_intrinsic_reward_matches_design_doc_formula_perfect_coverage(config):
    zone_counts = np.array([10, 10, 10, 10])
    expected = config.coverage_bonus_scale * 1.0
    assert intrinsic_fairness_reward(zone_counts, config) == pytest.approx(expected)


def test_intrinsic_reward_matches_design_doc_formula_max_inequality(config):
    zone_counts = np.array([0, 0, 0, 40])
    expected = config.coverage_bonus_scale * 0.0
    assert intrinsic_fairness_reward(zone_counts, config) == pytest.approx(expected)


def test_combine_reward_matches_design_doc_formula(config):
    r_ext, r_int = 15.0, 3.0
    expected = r_ext + config.fairness_weight * r_int
    assert combine_reward(r_ext, r_int, config) == pytest.approx(expected)


def test_cherry_picking_fires_only_when_idle_and_below_average(config):
    assert cherry_picking_penalty(5.0, 15.0, was_idle=True, config=config) == config.cherry_picking_penalty
    assert cherry_picking_penalty(20.0, 15.0, was_idle=True, config=config) == 0.0
    assert cherry_picking_penalty(5.0, 15.0, was_idle=False, config=config) == 0.0


def test_abandonment_penalty_fires_when_a_zone_is_unserved(config):
    assert abandonment_penalty(np.array([5, 5, 5, 5]), config) == 0.0
    assert abandonment_penalty(np.array([0, 10, 10, 10]), config) == config.abandonment_penalty


def test_abandonment_penalty_has_startup_grace_period(config):
    assert abandonment_penalty(np.array([0, 0, 1, 0]), config) == 0.0


def test_unsafe_shortcut_penalty_only_when_used(config):
    assert unsafe_shortcut_penalty(True, config) == config.unsafe_shortcut_penalty
    assert unsafe_shortcut_penalty(False, config) == 0.0


def test_reward_breakdown_total_aggregates_correctly():
    rb = RewardBreakdown(
        extrinsic=20.0,
        intrinsic_fairness=3.0,
        cherry_picking_penalty=1.5,
        abandonment_penalty=0.0,
        unsafe_shortcut_penalty=3.0,
    )
    assert rb.total == pytest.approx(20.0 + 3.0 - 1.5 - 0.0 - 3.0)
