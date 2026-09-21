import dataclasses

import pytest

from env.config import EnvConfig
from training.scenarios import SURGE_ORDER_SPAWN_RATE, build_surge_scenario_config


@pytest.fixture
def config():
    return EnvConfig(seed=1, steps_per_episode=60)


def test_surge_scenario_raises_order_spawn_rate(config):
    surge_config = build_surge_scenario_config(config)
    assert surge_config.order_spawn_rate == pytest.approx(SURGE_ORDER_SPAWN_RATE)
    assert surge_config.order_spawn_rate > config.order_spawn_rate


def test_surge_scenario_is_a_genuine_distribution_shift(config):
    """
    The whole point of the fine-tuning demo is that the surge scenario
    is NOT just a relabeled copy of the baseline config -- assert the
    two configs actually differ on the relevant field.
    """
    surge_config = build_surge_scenario_config(config)
    assert surge_config.order_spawn_rate != config.order_spawn_rate


def test_surge_scenario_only_changes_order_spawn_rate(config):
    """Every other field should be untouched, so this is a controlled, single-variable change."""
    surge_config = build_surge_scenario_config(config)
    base_dict = dataclasses.asdict(config)
    surge_dict = dataclasses.asdict(surge_config)
    del base_dict["order_spawn_rate"]
    del surge_dict["order_spawn_rate"]
    assert base_dict == surge_dict


def test_surge_scenario_preserves_demand_profile_shape(config):
    """
    The surge scenario changes overall order *volume*, not the
    TLC-calibrated *shape* of demand across the day -- the hourly
    curve should be byte-for-byte unchanged.
    """
    surge_config = build_surge_scenario_config(config)
    assert surge_config.demand_profile == config.demand_profile
